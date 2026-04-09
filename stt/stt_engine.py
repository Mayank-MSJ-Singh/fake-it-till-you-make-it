"""
STT Engine — Speech-to-Text with Semantic Pause Detection
==========================================================

This is the neural network that listens to audio and produces:
  1. Transcribed text (word by word)
  2. A "pause probability" signal (is the user done speaking?)

It runs Kyutai's stt-1b-en_fr model — the same model that powers
Unmute's production system. The model has 1 billion parameters and
runs on the GPU in a dedicated worker thread.

Architecture of the model:

    Raw Audio (24kHz)
        │
        ▼
    ┌─────────────────┐
    │   Mimi Codec     │  Neural audio compressor
    │   (encoder)      │  Converts 1920 raw samples → 20 audio tokens
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Transformer    │  16-layer decoder-only transformer
    │   (1B params)    │  Processes audio tokens, predicts text
    │                  │  Maintains KV-cache for full conversation
    │   d_model=2048   │  context (everything said so far)
    │   16 heads       │
    │   16 layers      │
    └────────┬────────┘
             │
         Hidden State
         (2048 dims)
             │
    ┌────────┼────────────────────┐
    │        │                    │
    ▼        ▼                    ▼
  text_linear    extra_heads[0-3]
  (2048→8000)    (2048→6 each)
    │                    │
    ▼                    ▼
  Text Token          Classification
  Prediction:         "What state is the audio in?"
  0=boundary          head[2] class 0 = pause probability
  3=PAD (silence)     ~0.001 during speech (even between words!)
  4+=real text        ~0.9 on genuine end-of-utterance pause

Why extra_heads instead of text_linear's P(PAD)?
  text_linear predicts the NEXT TOKEN. Between words, the correct
  prediction IS a PAD token, so P(PAD) spikes to 0.997 on every
  single word gap. That's useless for pause detection.

  extra_heads are CLASSIFICATION heads. They classify the SEMANTIC
  STATE of the audio. "User is mid-sentence" stays classified as
  "speaking" even during brief word gaps. Only when the user
  genuinely finishes their thought does head[2][0] rise.

Threading:
  The model MUST run in a dedicated thread because:
  1. mimi.streaming() and lm_gen.streaming() create contexts that
     hold the KV-cache. These MUST stay open for the whole session.
  2. We can't create/destroy these per-frame (too slow, state lost).
  3. The main thread is async (event loop) — can't hold a streaming
     context open permanently.

  So the worker thread opens the streaming context ONCE, then loops
  forever processing frames from a queue.

This mirrors Unmute's Rust STT server (which runs the same model),
but we do it all in Python with PyTorch.
"""
import threading
import queue
from dataclasses import dataclass

import numpy as np
import torch
import moshi.models


@dataclass
class STTResult:
    """One result from processing a single audio frame (80ms).

    The worker produces one of these for EVERY frame, whether or not
    a word was generated. The orchestrator drains these from result_queue.

    Fields:
        text_token: Raw token ID from the model.
                    0 = word boundary ("new word starts here")
                    3 = PAD ("no text at this timestep")
                    4+ = real SentencePiece text token
        pr_vad:     Pause probability from extra_heads[2][0].
                    0.0 = user is speaking
                    1.0 = user is paused/silent
                    This is the SAME signal as Unmute's prs[2].
        word:       Decoded word string, or None if no word this frame.
                    Words are only emitted on token boundaries (see _worker).
    """
    text_token: int
    pr_vad: float
    word: str | None


class STTEngine:
    """Kyutai STT running in a dedicated worker thread.

    Lifecycle:
        1. load_model()  — downloads and loads weights onto GPU
        2. warmup()      — runs 15 silence frames to compile CUDA kernels
        3. start()       — launches the worker thread
        4. feed_frame()  — called by orchestrator each loop iteration
        5. get_result_nowait() — orchestrator pops results
        6. stop()        — signals worker to exit
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.running = False

        # === QUEUES ===
        # These are the communication channels between the main thread
        # (orchestrator) and the worker thread (GPU inference).
        #
        # mic_queue: main → worker. Orchestrator puts audio frames here.
        #   The worker blocks on .get(), processes the frame, produces a result.
        #
        # result_queue: worker → main. Worker puts STTResult here.
        #   The orchestrator calls get_result_nowait() to drain these.
        self.mic_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.result_queue: queue.Queue[STTResult] = queue.Queue()

        # Model components — set during load_model()
        self.mimi = None         # Mimi codec (audio encoder/decoder)
        self.lm_gen = None       # LMGen wrapper around the transformer
        self.tokenizer = None    # SentencePiece tokenizer (token IDs → text)
        self.thread: threading.Thread | None = None

        # === INTERNAL STATE ===
        self._frame_count = 0          # How many frames the worker has processed
        self._warmup_frames = 12       # First 12 frames are noisy, skip them
        self._text_token_buffer = []   # Accumulates tokens until word boundary

        # The model's built-in lookahead delay (from Unmute's STT_DELAY_SEC).
        # The model sees 0.5s of "future" audio when making predictions.
        # After detecting a pause, we must flush this buffer with silence.
        self.delay_sec = 0.5

    @property
    def current_time(self) -> float:
        """How much audio time the worker has processed (in seconds).

        Used by the orchestrator to know when a flush is complete:
          stt_end_of_flush_time = current_time + delay_sec
          ...later...
          if current_time > stt_end_of_flush_time: flush done!

        This mirrors Unmute's stt.current_time (speech_to_text.py line 81).
        """
        return self._frame_count * 0.08  # 0.08 = FRAME_TIME_SEC

    # ================================================================
    # MODEL LOADING
    # ================================================================

    def load_model(self):
        """Load the STT model WITH extra heads onto the GPU.

        Uses kyutai/stt-1b-en_fr-candle — a single repo that includes:
          - The main transformer (1B params)
          - Mimi audio codec
          - 4 extra classification heads for semantic VAD
          - SentencePiece tokenizer

        The candle repo's config.json has extra_heads_num_heads=4,
        so the model is created with 4 heads and their weights are
        loaded automatically from model.safetensors. No manual
        weight injection needed — everything in one download.

        The "candle" name just means it was originally packaged for
        the Rust (Candle) server, but the safetensors format works
        perfectly with PyTorch too.
        """
        print("Loading STT model (kyutai/stt-1b-en_fr-candle)...")

        # Download config + weights from HuggingFace (cached after first run)
        info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
            "kyutai/stt-1b-en_fr-candle"
        )

        # Load Mimi audio codec (neural audio compressor)
        self.mimi = info.get_mimi(device=self.device)

        # Load the main transformer model
        # Because the candle config has extra_heads_num_heads=4,
        # LMModel.__init__ creates 4 Linear(2048→6) heads automatically,
        # and load_state_dict fills them with the trained weights.
        lm = info.get_moshi(device=self.device, dtype=torch.bfloat16)
        print(f"  Extra heads: {len(lm.extra_heads)} (should be 4)")

        # Wrap the transformer in LMGen — this handles sampling/decoding
        # temp=0 means greedy decoding (always pick the most likely token)
        self.lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

        # SentencePiece tokenizer — converts token IDs back to text
        self.tokenizer = info.get_text_tokenizer()

        print(f"STT loaded! GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ================================================================
    # WARMUP
    # ================================================================

    def warmup(self):
        """Run 15 silence frames to compile CUDA kernels.

        The first time PyTorch runs a CUDA operation, it compiles the
        kernel. This takes 100-500ms per operation. By running dummy
        frames during startup, we pay this cost BEFORE the user speaks,
        so real-time audio doesn't stutter.

        This uses a SEPARATE streaming context (with ... block) that
        gets closed after warmup. The real streaming context is created
        in _worker().
        """
        print("Warming up STT (compiling CUDA kernels)...")
        silence = torch.zeros((1, 1, 1920), device=self.device)

        # Open a temporary streaming context just for warmup
        with self.mimi.streaming(1), self.lm_gen.streaming(1):
            for i in range(15):
                tokens = self.mimi.encode(silence)
                # Use step_with_extra_heads so those kernels compile too
                self.lm_gen.step_with_extra_heads(tokens)

        # Reset frame count — warmup frames don't count
        self._frame_count = 0
        print("STT warmup done!")

    # ================================================================
    # THREAD LIFECYCLE
    # ================================================================

    def start(self):
        """Launch the worker thread.

        The thread runs _worker() which opens a streaming context
        and processes frames forever until stop() is called.
        """
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("STT worker thread started")

    def stop(self):
        """Signal the worker to stop and wait for it to finish.

        Puts None into mic_queue to unblock the worker if it's
        waiting on .get(), then joins the thread.
        """
        self.running = False
        self.mic_queue.put(None)  # Unblock worker's mic_queue.get()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            print("STT worker thread stopped")

    # ================================================================
    # PUBLIC API (called by orchestrator on main thread)
    # ================================================================

    def feed_frame(self, audio: np.ndarray):
        """Feed an audio frame to the STT. Non-blocking.

        The orchestrator calls this once per loop iteration.
        The frame goes into mic_queue, and the worker picks it up.

        Args:
            audio: numpy array, shape (1920,), dtype float32
        """
        self.mic_queue.put(audio)

    def get_result_nowait(self) -> STTResult | None:
        """Pop a result if one is ready. Returns None if queue is empty.

        The orchestrator calls this in a loop to drain all available
        results. Multiple results might be ready if the worker is
        faster than the main loop (it usually is — 15ms vs 80ms).
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    # ================================================================
    # THE WORKER (runs in its own thread for the entire conversation)
    # ================================================================

    @torch.no_grad()  # Disable gradient tracking — we're doing inference only
    def _worker(self):
        """The heart of the STT — processes audio frames on the GPU.

        This runs in its own thread for the ENTIRE conversation.
        The streaming contexts (mimi.streaming + lm_gen.streaming)
        stay open the whole time — they hold the model's KV-cache
        (memory of everything said so far).

        For each frame:
          1. Encode raw audio → Mimi tokens (20 tokens per frame)
          2. Run transformer → get text token + extra head outputs
          3. Extract pause probability from vad_heads[2][0]
          4. Buffer text tokens, emit word on boundaries
          5. Put result into result_queue
        """
        print("STT worker: entering streaming context...")

        # Open streaming context — this creates the KV-cache and codec buffers.
        # These MUST stay open for the whole conversation!
        with self.mimi.streaming(1), self.lm_gen.streaming(1):
            while self.running:

                # ── STEP 0: Wait for a frame from the orchestrator ──
                # This blocks until feed_frame() puts something in the queue.
                # Could be real mic audio, or silence (during flush).
                raw_audio = self.mic_queue.get()

                # None is the signal to stop (sent by stop())
                if raw_audio is None:
                    break

                # ── STEP 1: Convert numpy → torch tensor ──
                # sounddevice gives us (1920,) float32 numpy array
                # Mimi wants (batch=1, channels=1, samples=1920) torch tensor
                audio_tensor = torch.from_numpy(raw_audio).to(self.device)
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

                # ── STEP 2: Encode audio → Mimi tokens ──
                # Mimi is a neural audio codec (like MP3 but neural).
                # It compresses 1920 raw samples → 20 discrete tokens.
                # These tokens are much smaller and easier for the
                # transformer to process.
                audio_tokens = self.mimi.encode(audio_tensor)

                # ── STEP 3: Run transformer + extra heads ──
                # step_with_extra_heads() does two things:
                #   a) Runs the transformer to predict the next text token
                #   b) Runs the 4 extra classification heads
                # Returns: (text_tokens, list_of_head_outputs)
                text_tokens, vad_heads = self.lm_gen.step_with_extra_heads(
                    audio_tokens
                )

                # ── STEP 4: Extract pause probability ──
                # vad_heads[2] = output of the THIRD extra head
                #   Shape: (batch=1, seq=1, classes=6)
                # We take class 0 — the "user is paused" probability.
                #
                # This is what Unmute sends as prs[2] from the Rust server.
                # Key property: stays at ~0.001 between words during speech,
                # only rises to ~0.9 on genuine end-of-utterance pauses.
                pr_vad = vad_heads[2][0, 0, 0].float().item()

                # ── STEP 5: Extract the text token ──
                # text_tokens has shape (batch=1, codebooks=1, seq=1)
                # We take the single token: text_tokens[0, 0, 0]
                text_token = text_tokens[0, 0, 0].cpu().item()

                # ── STEP 6: Decode token → word ──
                # The model outputs one token per frame (every 80ms).
                # But a word might span multiple tokens:
                #   "thinking" → [token("▁th"), token("ink"), token("ing")]
                #
                # We BUFFER tokens and EMIT the word at boundaries:
                #   - Token 0 (word boundary) → next word starting, emit buffer
                #   - Token 3 (PAD) → silence, emit any buffered tokens
                #   - Any other token → add to buffer, don't emit yet
                #
                # Why emit on PAD too?
                #   Without it, the LAST word of an utterance stays stuck
                #   in the buffer forever. "Hello world" → "Hello" emitted
                #   on boundary, but "world" stays buffered because no more
                #   boundaries come. Emitting on the first PAD after text
                #   tokens solves this.
                word = None
                self._frame_count += 1

                # Skip warmup frames — model output is garbage for first ~960ms
                if self._frame_count > self._warmup_frames:
                    if text_token not in (0, 3):
                        # Real text token → add to buffer
                        self._text_token_buffer.append(text_token)
                    elif self._text_token_buffer:
                        # Boundary or PAD → emit everything in the buffer
                        # tokenizer.decode() handles multi-token words correctly:
                        #   [tok("▁th"), tok("ink")] → "think"
                        word = self.tokenizer.decode(self._text_token_buffer)
                        self._text_token_buffer = []

                # ── STEP 7: Send result to orchestrator ──
                # One result per frame, always. The orchestrator reads
                # pr_vad on every result to update the EMA.
                # word is None most of the time (only set when we emit).
                self.result_queue.put(STTResult(
                    text_token=text_token,
                    # During warmup, report pr_vad=1.0 ("paused") to prevent
                    # any pause detection logic from firing on garbage data
                    pr_vad=pr_vad if self._frame_count > self._warmup_frames else 1.0,
                    word=word,
                ))

        print("STT worker: exited streaming context")

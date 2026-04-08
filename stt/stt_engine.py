import threading
import queue
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import moshi.models


@dataclass
class STTResult:
    """One result from processing a single audio frame."""
    text_token: int       # The text token ID (0 or 3 = no text, otherwise a real token)
    pr_vad: float         # Pause probability from semantic VAD (0.0 = speaking, 1.0 = paused)
    word: str | None      # Decoded word, if any (None if no new text)


class STTEngine:
    """Kyutai STT running in a dedicated worker thread.
    
    The streaming context (mimi.streaming + lm_gen.streaming) stays alive
    for the entire conversation — that's why we need a dedicated thread
    instead of per-frame to_thread() calls.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.running = False

        # Queues for communication between main thread and worker thread
        self.mic_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.result_queue: queue.Queue[STTResult] = queue.Queue()

        # Will be set during loading
        self.mimi = None
        self.lm_gen = None
        self.tokenizer = None
        self.thread: threading.Thread | None = None

        # Internal state
        self._frame_count = 0
        self._warmup_frames = 12  # Skip first 12 frames (~960ms) — noisy during warmup
        self._captured_logits = {}  # Hook captures text logits here
        self._text_token_buffer = []  # Accumulate tokens between word boundaries
        self.sent_frames = 0  # How many frames we've fed (for flush timing)

    @property
    def current_time(self) -> float:
        """How much time the STT has processed. Mirrors Unmute's stt.current_time."""
        return self._frame_count * 0.08  # FRAME_TIME_SEC

    def load_model(self):
        """Load the STT model. Call this before start()."""
        print("Loading STT model...")
        info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")

        self.mimi = info.get_mimi(device=self.device)
        lm = info.get_moshi(device=self.device, dtype=torch.bfloat16)

        # Hook into text_linear to capture raw logits before sampling.
        # We use P(PAD token=3) as the semantic pause signal.
        # This matches what the Rust server in Unmute sends as prs[2]:
        #   - During speech: P(PAD) ≈ 0.0 (model predicts real text tokens)
        #   - Mid-sentence pause: P(PAD) ≈ 0.7 (model unsure, won't trigger)
        #   - End-of-sentence: P(PAD) ≈ 0.999 (model confident speech is over)
        def _hook_fn(module, input, output):
            self._captured_logits['logits'] = output.detach()
        lm.text_linear.register_forward_hook(_hook_fn)

        self.lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
        self.tokenizer = info.get_text_tokenizer()

        print(f"STT loaded! GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    def warmup(self):
        """Feed silence frames to warm up GPU kernels before real conversation."""
        print("Warming up STT (compiling CUDA kernels)...")
        silence = torch.zeros((1, 1, 1920), device=self.device)
        with self.mimi.streaming(1), self.lm_gen.streaming(1):
            for i in range(15):  # 15 frames of silence
                tokens = self.mimi.encode(silence)
                self.lm_gen.step(tokens)
        # Reset frame count since warmup used a separate streaming context
        self._frame_count = 0
        print("STT warmup done!")

    def start(self):
        """Launch the worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("STT worker thread started")

    def stop(self):
        """Signal the worker to stop and wait for it."""
        self.running = False
        # Put None to unblock the worker if it's waiting on mic_queue.get()
        self.mic_queue.put(None)
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            print("STT worker thread stopped")

    def feed_frame(self, audio: np.ndarray):
        """Feed an audio frame to the STT. Called from main thread. Non-blocking."""
        self.mic_queue.put(audio)
        self.sent_frames += 1

    def get_result_nowait(self) -> STTResult | None:
        """Try to get a result without blocking. Returns None if no result ready."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    @torch.no_grad()
    def _worker(self):
        """The worker loop — runs in its own thread for the entire conversation.
        
        The key insight: mimi.streaming() and lm_gen.streaming() create contexts
        that hold the model's internal state (KV-cache, codec buffers).
        These MUST stay alive for the whole session.
        """
        print("STT worker: entering streaming context...")

        with self.mimi.streaming(1), self.lm_gen.streaming(1):
            while self.running:
                # Block until a frame arrives from the main thread
                raw_audio = self.mic_queue.get()

                # None = signal to stop
                if raw_audio is None:
                    break

                # Convert numpy array to torch tensor
                # Shape needed: (batch=1, channels=1, samples=1920)
                audio_tensor = torch.from_numpy(raw_audio).to(self.device)
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1920,) → (1, 1, 1920)

                # Step 1: Encode audio to Mimi tokens
                audio_tokens = self.mimi.encode(audio_tensor)

                # Step 2: Run through STT transformer
                text_tokens = self.lm_gen.step(audio_tokens)

                # Compute P(PAD) from captured text logits as the pause signal.
                # The text_linear hook captured the raw logits before sampling.
                pr_vad = 1.0  # Default: assume paused
                if 'logits' in self._captured_logits:
                    logits = self._captured_logits['logits'].float()  # [1, 1, 8000]
                    probs = F.softmax(logits[0, 0, :], dim=0)
                    pr_vad = probs[3].item()  # P(token 3 = PAD)

                # Extract text token
                text_token = text_tokens[0, 0, 0].cpu().item()

                # Decode token to text.
                # Unlike Unmute (which gets pre-decoded text from the Rust server),
                # we need to decode tokens ourselves. SentencePiece may split a word
                # into byte tokens (<0xC4>, <0x90>) that must be decoded together.
                #
                # We accumulate text tokens into a buffer and emit the word when:
                #   - Token 0 (word boundary) arrives = next word starting
                #   - Token 3 (PAD) arrives = silence after word ended
                # This matches the Rust STT server behavior. Without emitting on
                # PAD, the LAST word of an utterance stays stuck in the buffer
                # until the user speaks again.
                word = None
                self._frame_count += 1

                if self._frame_count > self._warmup_frames:
                    if text_token not in (0, 3):  # Real text token
                        self._text_token_buffer.append(text_token)
                    elif self._text_token_buffer:  # Boundary (0) or PAD (3) → emit word
                        word = self.tokenizer.decode(self._text_token_buffer)
                        self._text_token_buffer = []

                # Send result back to main thread
                self.result_queue.put(STTResult(
                    text_token=text_token,
                    pr_vad=pr_vad if self._frame_count > self._warmup_frames else 1.0,
                    word=word,
                ))

        print("STT worker: exited streaming context")

"""
Orchestrator — The Brain of the System
========================================

This is the main loop that ties everything together. It runs as an
async function that never returns (until Ctrl+C).

Each iteration of the loop takes ~80ms (paced by the microphone).

The loop has two modes:

  NORMAL MODE (stt_end_of_flush_time is None):
    1. Get mic frame → feed to STT
    2. Drain results from STT worker → update EMA, process words
    3. Check for long silence (7s timeout)
    4. Check if EMA crossed pause threshold → start flush

  FLUSH MODE (stt_end_of_flush_time is set):
    1. Get mic frame → feed to STT (still feeding audio!)
    2. Drain results (late words from pipeline still arrive!)
    3. Skip pause/silence detection
    4. When STT has processed all silence frames → respond to user

Response generation (LLM → TTS → Speaker) runs in a BACKGROUND THREAD
so the main loop keeps ticking. This is critical: the mic must keep
recording during the bot's response for interruption detection (Phase 7).

This mirrors Unmute's receive() method: unmute/unmute_handler.py lines 280-370
"""
import asyncio
import math
import threading
import time

import numpy as np
import torch
from stream2sentence import generate_sentences

from core.audio_io import MicrophoneInput, SpeakerOutput
from core.config import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, FRAME_TIME_SEC,
    EMA_ATTACK_TIME, EMA_RELEASE_TIME, EMA_INITIAL_VALUE,
    PAUSE_THRESHOLD, SILENCE_TIMEOUT, DEBUG_MODE
)
from core.conversation import Conversation
from core.ui import UIManager
from llm.llm_engine import LLMEngine
from stt.stt_engine import STTEngine
from stt.ema import ExponentialMovingAverage
from tts.tts_engine import TTSEngine


class Orchestrator:
    """The main loop that coordinates mic, STT, pause detection, and response.

    Components it owns:
      - MicrophoneInput  (core/audio_io.py)   — captures audio
      - SpeakerOutput    (core/audio_io.py)   — plays audio through speakers
      - STTEngine        (stt/stt_engine.py)   — GPU inference thread
      - LLMEngine        (llm/llm_engine.py)   — generates text responses
      - TTSEngine        (tts/tts_engine.py)   — converts text → audio (CPU)
      - Conversation     (core/conversation.py) — chat history + state machine
      - EMA              (stt/ema.py)          — smoothed pause signal
    """

    def __init__(self, device: str = "cuda"):
        # === COMPONENTS ===
        self.mic = MicrophoneInput()       # Records audio from system mic
        self.speaker = SpeakerOutput()     # Plays TTS audio through speakers
        self.stt = STTEngine(device=device)  # GPU STT model + worker thread
        self.llm = LLMEngine()             # Language model for responses
        self.tts = TTSEngine()             # Pocket TTS (CPU) for speech synthesis
        self.conversation = Conversation()   # Chat history + state machine

        # === PAUSE DETECTION ===
        # The EMA smooths the raw pr_vad signal from the STT's extra heads.
        # With attack=0.01 and release=0.01, it barely smooths at all
        # (because the extra heads signal is already smooth).
        # See stt/ema.py for how the math works.
        self.ema = ExponentialMovingAverage(
            attack_time=EMA_ATTACK_TIME,     # 0.01 — fast rise
            release_time=EMA_RELEASE_TIME,   # 0.01 — fast drop
            initial_value=EMA_INITIAL_VALUE,  # 1.0 — start as "paused"
        )

        # === STATE ===
        self.running = False
        self.last_activity_time: float = 0  # Wall clock time of last word/event
        self.user_turn_start_time: float = 0  # When the user started this turn

        # === RESPONSE THREAD ===
        # LLM→TTS generation runs in a background thread so the main loop
        # keeps running (mic stays active for interruption detection).
        # When None → no response being generated.
        # When set → a thread is actively generating LLM text + TTS audio.
        self._response_thread: threading.Thread | None = None

        # === INTERRUPTION ===
        # Set by the main loop when the user speaks during bot response.
        # The response worker checks this flag and exits early.
        self._interrupted = threading.Event()

        # === FLUSH STATE ===
        # When None → normal mode (check pauses)
        # When set → flush mode (wait for STT to catch up, then respond)
        # Set to stt.current_time + stt.delay_sec when pause is detected.
        # Mirrors Unmute's self.stt_end_of_flush_time
        self.stt_end_of_flush_time: float | None = None

        # === TERMINAL UI ===
        self.ui = UIManager()
        from rich.live import Live
        self.live_context = Live(self.ui.get_renderable(), auto_refresh=False, screen=True)

        # === POLISH ===
        self._greeted = False

    # ================================================================
    # LIFECYCLE
    # ================================================================

    async def start(self):
        """Load models, start all components, and prepare for conversation.

        Order matters:
          1. Load STT model (downloads if not cached, ~30s first time)
          2. Load LLM model (downloads if not cached, ~30s first time)
          3. Load TTS model (downloads if not cached, ~150MB)
          4. Warmup STT (compiles CUDA kernels with dummy frames, ~5s)
          5. Start mic (begins recording immediately)
          6. Start speaker (begins playback loop, plays silence until fed audio)
          7. Start STT worker thread (opens streaming context, waits for frames)
        """
        self.stt.load_model()   # Downloads STT model, loads onto GPU
        self.llm.load_model()   # Downloads LLM, loads onto configured device
        self.tts.load_model()   # Downloads Pocket TTS, loads voice on CPU
        self.stt.warmup()       # Compiles CUDA kernels with dummy frames

        self.mic.start()        # Opens mic stream, callback starts firing
        self.speaker.start()    # Opens speaker stream
        self.stt.start()        # Launches worker thread

        self.running = True
        self.last_activity_time = time.time()

        print("\n" + "=" * 50)
        print("🎤 Listening... (speak into your mic)")
        print("=" * 50 + "\n")

    async def stop(self):
        """Shut everything down gracefully."""
        self.running = False
        self.stt.stop()
        self.mic.stop()
        self.speaker.stop()
        
        self.ui.debug_line = "👋 Goodbye!"
        if hasattr(self, 'live_context'):
            # Update one last time to show goodbye message before screen=True restores term
            try:
                self.live_context.update(self.ui.get_renderable(), refresh=True)
            except Exception:
                pass
                
        # Minor sleep to allow underlying audio threads to clean up
        await asyncio.sleep(0.5)

    async def _greet(self):
        """Say the initial greeting."""
        greeting = "Hello! I'm listening. How can I help you today?"
        self.ui.debug_line = "🗣️ Greeting..."
        
        # Add to history
        self.conversation.add_message_delta(greeting, "assistant")
        
        # We'll run this in the response worker thread to avoid blocking the main loop
        # We pass static text instead of prompting the LLM
        self._response_thread = threading.Thread(
            target=self._greeting_worker,
            args=(greeting,),
            daemon=True,
        )
        self._response_thread.start()
        self._greeted = True

    def _greeting_worker(self, text: str):
        """Worker thread version for greeting to keep logic consistent."""
        t_start = time.time()
        for audio_chunk in self.tts.synthesize_stream(text):
            if self._interrupted.is_set():
                break
            self._push_audio_to_speaker(audio_chunk)
        
        if not self._interrupted.is_set():
            self._flush_rechunk_buffer()
        
        # Transition to waiting_for_user
        self.conversation.add_message_delta("", "user")
        self.last_activity_time = time.time()
        self._response_thread = None
        self.ui.debug_line = f"⏱  Greeting finished in {time.time() - t_start:.2f}s"

    # ================================================================
    # THE MAIN LOOP
    # ================================================================

    async def run(self):
        """The infinite main loop. Each iteration = ~80ms (one audio frame).

        This is the equivalent of Unmute's receive() method
        (unmute_handler.py lines 280-370).

        Flow per iteration:
          1. await mic.get_frame()  — blocks ~80ms until next frame
          2. stt.feed_frame(frame)  — puts frame in worker's queue (instant)
          3. drain results          — pop all STTResults, update EMA
          4. decide                 — check pause/silence, or monitor flush
        """
        await self.start()
        self._start_time = time.time()

        try:
            with self.live_context as live:
                # Trigger greeting once everything is ready
                if not self._greeted:
                    await self._greet()

                while self.running:
                    # Update UI state before processing this frame
                    self.ui.update(
                        history=self.conversation.chat_history,
                        state=self.conversation.conversation_state(),
                        ema=self.ema.value,
                        time_sec=time.time() - self._start_time if getattr(self, '_start_time', None) else 0.0,
                        threshold=PAUSE_THRESHOLD,
                        debug_line=self.ui.debug_line # Keep existing debug lines
                    )
                    live.update(self.ui.get_renderable(), refresh=True)

                    # ── 1. Get the next audio frame ──
                    # This awaits until the mic callback puts a frame in the queue.
                    # Normally takes ~80ms (one frame period).
                    # If frames are buffered (e.g., after flush), returns instantly.
                    frame = await self.mic.get_frame()

                    # ── 2. Feed to STT worker ──
                    # Just puts the numpy array into mic_queue. The worker thread
                    # will pick it up and process it on the GPU.
                    # This is instant (non-blocking).
                    self.stt.feed_frame(frame)

                    # ── 3. Drain all available results ──
                    # The worker might have produced multiple results since our
                    # last drain (it runs at ~15ms per frame, we drain every ~80ms).
                    # Each result updates the EMA and may emit a word.
                    self._drain_and_process_results()

                    # ── 4. State-dependent decisions ──
                    if self.stt_end_of_flush_time is None:
                        # === NORMAL MODE ===
                        # Check if user has been silent too long (7s → "...")
                        self._check_silence()

                        # Check if EMA crossed pause threshold
                        if self._determine_pause():
                            self._start_flush()
                    else:
                        # === FLUSH MODE ===
                        # We detected a pause and fed silence to flush the pipeline.
                        # Now we're waiting for the STT worker to process all those
                        # silence frames. During this time:
                        #   - We still feed real mic audio (so we don't lose frames)
                        #   - We still drain results (late words arrive during flush!)
                        #   - We skip pause/silence detection
                        #
                        # Once the worker has caught up → generate response.
                        if self.stt.current_time > self.stt_end_of_flush_time:
                            self.stt_end_of_flush_time = None
                            self._generate_response()

        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    # ================================================================
    # RESULT PROCESSING
    # ================================================================

    def _drain_and_process_results(self):
        """Pop all available STTResults from the worker and process them.

        Called once per main loop iteration. For each result:
          1. Update EMA with pr_vad (pause probability)
          2. If the result contains a word → call _on_word()

        The worker produces one result per frame. Since the worker is
        faster than the main loop (~15ms vs ~80ms), there might be
        multiple results waiting.
        """
        while True:
            result = self.stt.get_result_nowait()
            if result is None:
                break  # No more results available

            # Update EMA with the pause probability from this frame.
            # dt=FRAME_TIME_SEC because each result represents one 80ms frame.
            self.ema.update(dt=FRAME_TIME_SEC, new_value=result.pr_vad)

            # Debug: print EMA every ~1 second (every 12 frames) if DEBUG_MODE is True
            if DEBUG_MODE and self.stt._frame_count % 12 == 0 and result.pr_vad > 0.01:
                self.ui.debug_line = f"raw={result.pr_vad:.3f} ema={self.ema.value:.3f}"

            # If a word was emitted this frame, process it
            if result.word:
                self._on_word(result.word)
                
                # Exit detection
                clean_word = result.word.lower().strip()
                if clean_word in ["bye.", "goodbye.", "exit.", "quit.", "bye", "goodbye"]:
                    self.ui.debug_line = "👋 Exit command detected"
                    self.running = False

    def _on_word(self, word: str):
        """Called when the STT produces a new word.

        This mirrors Unmute's _stt_loop() (handler lines 436-466).

        Two critical things happen here:
          1. Add the word to the conversation
          2. If it's the FIRST word of a new turn → reset EMA to 0.0

        The EMA reset is essential. Without it, the EMA would still be
        high from the previous silence (~1.0), and the first word would
        immediately trigger a pause (1.0 > 0.6).

        Args:
            word: The decoded word string (e.g., "Hello", "world")
        """
        # Add word to conversation. Returns True if this is a NEW message
        # (first word after empty "" message = new turn).
        # See core/conversation.py add_message_delta() for details.
        is_new = self.conversation.add_message_delta(word, "user")
        self.last_activity_time = time.time()

        # === INTERRUPTION: user spoke while bot is generating ===
        # If the response thread is running, the user is talking over the bot.
        # Stop everything immediately.
        if self._response_thread is not None:
            self._interrupt_bot()

        # === FALSE ALARM PAUSE: user kept talking during the flush window ===
        # If we had triggered a pause flush, but new words arrive, they aren't
        # done! Cancel the flush so we don't prematurely generate a response.
        if self.stt_end_of_flush_time is not None:
            self.ui.debug_line = "❌ Pause canceled (user continued speaking)"
            self.stt_end_of_flush_time = None

        if is_new:
            # FIRST word of a new turn!
            # Reset EMA to 0.0 so we don't immediately trigger pause.
            # This is exactly what Unmute does: handler line 465.
            self.ema.value = 0.0
            self.user_turn_start_time = time.time()  # Track when turn began

    # ================================================================
    # PAUSE DETECTION
    # ================================================================

    def _determine_pause(self) -> bool:
        """Should we trigger a response right now?

        Two conditions (from Unmute's determine_pause(), handler line 387):
          1. State must be "user_speaking" (user has said at least one word)
          2. EMA must exceed the threshold (0.6)

        If state is "waiting_for_user" or "bot_speaking", we never trigger.

        Returns:
            True if a pause should be triggered
        """
        # Only detect pauses when the user IS speaking
        if self.conversation.conversation_state() != "user_speaking":
            return False

        # Don't trigger a new response while one is already being generated
        if self._response_thread is not None:
            return False

        # (Removed the hardcoded 1.5s delay because the flush window + cancel mechanism
        # gracefully handles mid-sentence pauses now).

        # Check if the smoothed pause probability crossed the threshold
        return self.ema.value > PAUSE_THRESHOLD

    # ================================================================
    # THE FLUSH (non-blocking pipeline drain)
    # ================================================================

    def _start_flush(self):
        """Begin the non-blocking flush after detecting a pause.

        The STT model has a 0.5s lookahead buffer. When we detect a pause,
        there might be words still "in the pipeline" that haven't come out
        yet. We need to push them out by feeding silence.

        This mirrors Unmute handler lines 340-351:
          1. Set stt_end_of_flush_time — marks when flush will be done
          2. Feed silence frames into the STT — pushes out buffered words
          3. Return immediately (non-blocking!)

        After this, the main loop continues normally. It still feeds mic
        audio and drains results. The flush is "done" when
        stt.current_time > stt_end_of_flush_time.
        """
        self.ui.debug_line = f"⏸️ Pause detected (EMA={self.ema.value:.2f})"

        # Set the "finish line" — when the worker has processed this much
        # time, all the silence frames have been consumed.
        # Mirrors: self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
        self.stt_end_of_flush_time = self.stt.current_time + self.stt.delay_sec

        # Feed silence frames to push any remaining words through the pipeline.
        # Number of frames: ceil(0.5 / 0.08) + 1 = 8 frames of silence
        num_frames = int(math.ceil(self.stt.delay_sec / FRAME_TIME_SEC)) + 1
        silence = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        for _ in range(num_frames):
            self.stt.feed_frame(silence)

        # We return here — the main loop continues immediately.
        # No sleep, no blocking. This is the key difference from a
        # naive "wait for flush" approach.

    # ================================================================
    # INTERRUPTION (stop bot mid-speech when user talks over it)
    # ================================================================

    def _interrupt_bot(self):
        """Stop the bot immediately when the user starts talking.

        Called from _on_word() when a word arrives while _response_thread
        is active. This does three things:
          1. Set the _interrupted event → worker thread sees it and exits
          2. Clear the speaker queue → audio stops immediately
          3. Reset the rechunk buffer → no stale audio leaks

        The worker thread checks _interrupted.is_set() at every:
          - LLM token yield
          - TTS audio chunk
          - Sentence fragment boundary
        So it exits within ~80ms of being signaled.
        """
        self.ui.debug_line = "⚡ Interrupting bot..."
        self._interrupted.set()       # Signal the worker to stop
        self.speaker.clear()          # Dump all queued audio
        self._rechunk_buf = np.array([], dtype=np.float32)  # Reset buffer

    # ================================================================
    # RESPONSE GENERATION (LLM → stream2sentence → TTS → Speaker)
    # ================================================================

    def _generate_response(self):
        """Kick off response generation in a background thread.

        Called when stt.current_time > stt_end_of_flush_time, meaning
        all silence frames have been processed and any late words have
        been collected.

        The actual work (LLM streaming → sentence chunking → TTS → speaker)
        happens in _response_worker() on a background thread. This lets
        the main loop keep running so:
          - The mic stays active (STT keeps processing)
          - Interruption can be detected later (Phase 7)

        The conversation state transitions to "bot_speaking" immediately
        (add_message_delta with role="assistant"). When the worker finishes,
        it transitions to "waiting_for_user".
        """
        # Drain any final words that arrived during the flush
        self._drain_and_process_results()

        # Get what the user said
        user_text = self.conversation.get_last_user_text()
        self.ui.debug_line = f"📝 User said: \"{user_text.strip()}\""

        # Get cleaned conversation history
        messages = self.conversation.preprocessed_messages()

        # Reset interruption flag before launching new response
        self._interrupted.clear()

        # Launch the response worker thread
        self._response_thread = threading.Thread(
            target=self._response_worker,
            args=(messages,),
            daemon=True,
        )
        self._response_thread.start()

    def _response_worker(self, messages: list[dict[str, str]]):
        """Background thread: LLM → stream2sentence → TTS → Speaker.

        Pipeline:
          generate_stream(messages) → yields LLM tokens
                ↓
          generate_sentences()      → yields smart sentence fragments
                ↓                     (after 10 chars or 7 words)
          tts.synthesize_stream()   → yields audio chunks per sentence
                ↓
          speaker.put_audio()       → gapless playback through speakers

        The main loop keeps running in parallel (mic + STT stay active).
        When this method returns, conversation transitions to waiting_for_user.
        """
        response_parts = []  # accumulate full response text
        t_start = time.time()

        self.ui.debug_line = "🤖 Generating..."

        def _llm_stream():
            """Yield LLM tokens, printing them as they arrive.
            Stops early if interrupted."""
            for chunk in self.llm.generate_stream(messages):
                if self._interrupted.is_set():
                    return
                # Print removed since UI loop catches chat_history changes!
                response_parts.append(chunk)
                yield chunk

        # stream2sentence wraps the LLM stream and yields sentence fragments.
        # It triggers TTS much sooner than waiting for punctuation:
        #   - minimum_first_fragment_length=10  → first ~10 chars = first audio
        #   - force_first_fragment_after_words=7 → force after 7 words max
        #   - quick_yield=True → yield partial fragments eagerly
        n_fragments = 0
        for sentence in generate_sentences(
            _llm_stream(),
            minimum_sentence_length=12,
            minimum_first_fragment_length=10,
            force_first_fragment_after_words=7,
            quick_yield_single_sentence_fragment=True,
            cleanup_text_links=True,
            cleanup_text_emojis=True,
        ):
            frag = sentence.strip()
            if not frag:
                continue

            # Synthesize this fragment and push audio chunks to the speaker.
            # synthesize_stream() yields chunks as they're generated (low TTFB).
            # put_audio() pushes into the speaker's queue — the sounddevice
            # callback pulls from it every 80ms for gapless playback.
            for audio_chunk in self.tts.synthesize_stream(frag):
                if self._interrupted.is_set():
                    break
                # Chunk from pocket_tts may be arbitrary length.
                # SpeakerOutput._callback expects exactly SAMPLES_PER_FRAME.
                # We need to rechunk into fixed-size blocks.
                self._push_audio_to_speaker(audio_chunk)

            if self._interrupted.is_set():
                break
            n_fragments += 1

        # Flush any remaining audio from the rechunk buffer
        if not self._interrupted.is_set():
            self._flush_rechunk_buffer()

        # All fragments synthesized and queued. The speaker will keep
        # playing from its queue until it drains.
        total_s = time.time() - t_start
        full_response = "".join(response_parts).strip()

        if self._interrupted.is_set():
            # Bot was interrupted — add partial response + marker
            self.ui.debug_line = f"⚡ Interrupted after {total_s:.2f}s"
            if full_response:
                self.conversation.add_message_delta(full_response, "assistant")
                self.conversation.mark_interruption()  # Appends "—"
        else:
            self.ui.debug_line = f"⏱  {total_s:.2f}s | Fragments: {n_fragments}"
            # Add the full response as an assistant message
            self.conversation.add_message_delta(full_response, "assistant")

            # Transition to waiting_for_user.
            # Add empty user message so pause detection is disabled until
            # the user speaks again. Mirrors Unmute's _tts_loop() line 577.
            self.conversation.add_message_delta("", "user")

        self.last_activity_time = time.time()

        # Clear thread reference
        self._response_thread = None

    # ================================================================
    # AUDIO RECHUNKING (TTS chunks → fixed-size speaker frames)
    # ================================================================

    def __init_rechunk(self):
        """Initialize the rechunk buffer. Called lazily."""
        if not hasattr(self, '_rechunk_buf'):
            self._rechunk_buf = np.array([], dtype=np.float32)

    def _push_audio_to_speaker(self, audio_chunk: np.ndarray):
        """Rechunk arbitrary-length TTS audio into SAMPLES_PER_FRAME blocks.

        The SpeakerOutput._callback() expects each queue item to be exactly
        SAMPLES_PER_FRAME (1920) samples. But TTS yields variable-length chunks.

        We accumulate into a buffer and flush complete frames to the speaker.
        Any remainder stays in the buffer for the next call.
        """
        self.__init_rechunk()

        # Flatten and append to buffer
        self._rechunk_buf = np.concatenate([
            self._rechunk_buf,
            audio_chunk.flatten().astype(np.float32),
        ])

        # Flush complete frames
        while len(self._rechunk_buf) >= SAMPLES_PER_FRAME:
            frame = self._rechunk_buf[:SAMPLES_PER_FRAME]
            self._rechunk_buf = self._rechunk_buf[SAMPLES_PER_FRAME:]
            self.speaker.put_audio(frame)

    def _flush_rechunk_buffer(self):
        """Flush any remaining audio in the rechunk buffer (zero-padded)."""
        self.__init_rechunk()
        if len(self._rechunk_buf) > 0:
            padded = np.pad(
                self._rechunk_buf,
                (0, SAMPLES_PER_FRAME - len(self._rechunk_buf)),
            )
            self.speaker.put_audio(padded)
            self._rechunk_buf = np.array([], dtype=np.float32)

    # ================================================================
    # SILENCE DETECTION
    # ================================================================

    def _check_silence(self):
        """Respond to the user if they've been silent too long (7s).

        If the user doesn't speak for SILENCE_TIMEOUT seconds after
        the bot finishes, we inject "..." into the conversation.
        This makes the state transition to "user_speaking" (because "..."
        is non-empty content). Then _determine_pause() fires on the
        next frame (EMA is already at ~1.0 since nobody is speaking),
        and the bot responds to the silence.

        This mirrors Unmute's detect_long_silence() (handler line 626).
        """
        # Only trigger during waiting_for_user state
        if self.conversation.conversation_state() != "waiting_for_user":
            return

        # Don't trigger silence while bot is still speaking
        if self._response_thread is not None:
            return

        # Don't trigger while the speaker is still playing queued audio.
        # The response thread finishes when audio is QUEUED, not when it's
        # done PLAYING. Without this check, the silence timer fires while
        # the bot is still audibly speaking → infinite self-talk loop.
        if not self.speaker.audio_queue.empty():
            self.last_activity_time = time.time()  # reset timer while playing
            return

        elapsed = time.time() - self.last_activity_time
        if elapsed > SILENCE_TIMEOUT:
            # Inject "..." as user message → state becomes "user_speaking"
            # → _determine_pause() will trigger → bot responds
            self.conversation.add_message_delta("...", "user")
            self.last_activity_time = time.time()
            self.ui.debug_line = "🤫 Long silence detected..."

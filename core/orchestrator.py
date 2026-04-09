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

This mirrors Unmute's receive() method: unmute/unmute_handler.py lines 280-370
"""
import asyncio
import math
import time

import numpy as np
import torch

from core.audio_io import MicrophoneInput, SpeakerOutput
from core.config import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, FRAME_TIME_SEC,
    EMA_ATTACK_TIME, EMA_RELEASE_TIME, EMA_INITIAL_VALUE,
    PAUSE_THRESHOLD, SILENCE_TIMEOUT,
)
from core.conversation import Conversation
from stt.stt_engine import STTEngine
from stt.ema import ExponentialMovingAverage


class Orchestrator:
    """The main loop that coordinates mic, STT, pause detection, and response.

    Components it owns:
      - MicrophoneInput  (core/audio_io.py)   — captures audio
      - SpeakerOutput    (core/audio_io.py)   — plays audio (for TTS later)
      - STTEngine        (stt/stt_engine.py)   — GPU inference thread
      - Conversation     (core/conversation.py) — chat history + state machine
      - EMA              (stt/ema.py)          — smoothed pause signal
    """

    def __init__(self, device: str = "cuda"):
        # === COMPONENTS ===
        self.mic = MicrophoneInput()       # Records audio from system mic
        self.speaker = SpeakerOutput()     # Plays TTS audio (Phase 6)
        self.stt = STTEngine(device=device)  # GPU STT model + worker thread
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

        # === FLUSH STATE ===
        # When None → normal mode (check pauses)
        # When set → flush mode (wait for STT to catch up, then respond)
        # Set to stt.current_time + stt.delay_sec when pause is detected.
        # Mirrors Unmute's self.stt_end_of_flush_time
        self.stt_end_of_flush_time: float | None = None

    # ================================================================
    # LIFECYCLE
    # ================================================================

    async def start(self):
        """Load models, start all components, and prepare for conversation.

        Order matters:
          1. Load STT model (downloads if not cached, ~30s first time)
          2. Warmup STT (compiles CUDA kernels, ~5s)
          3. Start mic (begins recording immediately)
          4. Start speaker (begins playback loop, plays silence until fed audio)
          5. Start STT worker thread (opens streaming context, waits for frames)
        """
        self.stt.load_model()   # Downloads model, loads onto GPU
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
        """Shut everything down gracefully.

        Order: STT first (joins the worker thread), then mic and speaker.
        """
        self.running = False
        self.stt.stop()         # Sends None to worker, joins thread
        self.mic.stop()         # Stops mic stream
        self.speaker.stop()     # Stops speaker stream
        print("\n👋 Goodbye!")

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

        try:
            while self.running:

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

            # If a word was emitted this frame, process it
            if result.word:
                self._on_word(result.word)

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

        if is_new:
            # FIRST word of a new turn!
            # Reset EMA to 0.0 so we don't immediately trigger pause.
            # This is exactly what Unmute does: handler line 465.
            self.ema.value = 0.0
            print(f"\n[YOU] {word}", end="", flush=True)
        else:
            # Subsequent word in the same turn — just append
            print(f" {word}", end="", flush=True)

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
        print(f"\n\n  ⏸️  Pause detected (EMA={self.ema.value:.2f})")

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
    # RESPONSE GENERATION
    # ================================================================

    def _generate_response(self):
        """Generate a response after the flush is complete.

        Called when stt.current_time > stt_end_of_flush_time, meaning
        all silence frames have been processed and any late words have
        been collected.

        This mirrors Unmute's _generate_response() → _tts_loop() flow.

        Currently a placeholder — just echoes what the user said.
        Phase 5 will connect this to an LLM (Ollama/Qwen).
        Phase 6 will add TTS (PocketTTS) for spoken responses.
        """
        # Drain any final words that arrived during the flush
        self._drain_and_process_results()

        # Get what the user said
        user_text = self.conversation.get_last_user_text()
        print(f"  📝 User said: \"{user_text.strip()}\"")

        # --- TODO: Replace with real LLM call (Phase 5) ---
        fake_response = f"I heard you say: {user_text.strip()}"
        self.conversation.add_message_delta(fake_response, "assistant")
        print(f"\n[BOT] {fake_response}")
        print()

        # === CRITICAL: Transition to waiting_for_user ===
        # Add an empty user message: {"role": "user", "content": ""}
        # This makes conversation_state() return "waiting_for_user",
        # which disables pause detection until the user speaks again.
        #
        # When the user says their first word, it appends to this empty
        # message, add_message_delta() returns True (is_new_message),
        # and the EMA gets reset to 0.0.
        #
        # Mirrors Unmute's _tts_loop() line 577.
        self.conversation.add_message_delta("", "user")
        self.last_activity_time = time.time()

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

        elapsed = time.time() - self.last_activity_time
        if elapsed > SILENCE_TIMEOUT:
            # Inject "..." as user message → state becomes "user_speaking"
            # → _determine_pause() will trigger → bot responds
            self.conversation.add_message_delta("...", "user")
            self.last_activity_time = time.time()
            print("\n  🤫 Long silence detected...")

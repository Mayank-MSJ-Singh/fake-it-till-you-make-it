import asyncio
import math
import time

import numpy as np
import torch

from core.audio_io import MicrophoneInput, SpeakerOutput
from core.config import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, FRAME_TIME_SEC,
    EMA_ATTACK_TIME, EMA_RELEASE_TIME, EMA_INITIAL_VALUE,
    PAUSE_THRESHOLD, STT_DELAY_SEC, SILENCE_TIMEOUT, FLUSH_FRAMES,
)
from core.conversation import Conversation
from stt.stt_engine import STTEngine
from stt.ema import ExponentialMovingAverage


class Orchestrator:
    """The main loop that coordinates everything.
    
    Mirrors Unmute's handler flow:

    NORMAL mode (stt_end_of_flush_time is None):
      1. Feed audio to STT, process words
      2. detect_long_silence() → adds "..." if user silent too long
      3. determine_pause() → if EMA > threshold → start flush

    FLUSHING mode (stt_end_of_flush_time is set):
      1. Feed audio, process words (catch late words from pipeline)
      2. Skip pause/silence detection
      3. When stt.current_time > stt_end_of_flush_time → flush done → respond
    """

    def __init__(self, device: str = "cuda"):
        # Components
        self.mic = MicrophoneInput()
        self.speaker = SpeakerOutput()
        self.stt = STTEngine(device=device)
        self.conversation = Conversation()

        # Pause detection — EMA smooths P(PAD) from text logits
        self.ema = ExponentialMovingAverage(
            attack_time=EMA_ATTACK_TIME,
            release_time=EMA_RELEASE_TIME,
            initial_value=EMA_INITIAL_VALUE,
        )

        # State tracking
        self.running = False
        self.last_activity_time: float = 0

        # Flush state — mirrors Unmute's stt_end_of_flush_time
        self.stt_end_of_flush_time: float | None = None

    async def start(self):
        """Load models and start all components."""
        self.stt.load_model()
        self.stt.warmup()

        self.mic.start()
        self.speaker.start()
        self.stt.start()

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
        print("\n👋 Goodbye!")

    async def run(self):
        """The main loop — runs until interrupted.
        
        Mirrors Unmute's receive() method.
        Each iteration = one audio frame (~80ms).
        """
        await self.start()

        try:
            while self.running:
                # 1. Get audio frame from mic (blocks ~80ms)
                frame = await self.mic.get_frame()

                # 2. Feed to STT worker (non-blocking)
                self.stt.feed_frame(frame)

                # 3. Drain ALL available STT results.
                # The worker runs in a separate thread and may have produced
                # multiple results since our last drain (especially after flush).
                self._drain_and_process_results()

                # 4. Normal mode vs Flushing mode
                if self.stt_end_of_flush_time is None:
                    # NORMAL: check for silence and pauses
                    self._check_silence()

                    if self._determine_pause():
                        self._start_flush()
                else:
                    # FLUSHING: wait for STT to finish processing silence frames.
                    # Words still arrive from _drain_and_process_results() above.
                    if self.stt.current_time > self.stt_end_of_flush_time:
                        self.stt_end_of_flush_time = None
                        self._generate_response()

        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    def _drain_and_process_results(self):
        """Drain all available results from the STT worker.
        
        This is called once per main loop iteration. The worker may have
        produced multiple results since our last call (it runs continuously
        in its own thread).
        """
        while True:
            result = self.stt.get_result_nowait()
            if result is None:
                break

            # Update EMA with pause probability on every result
            self.ema.update(dt=FRAME_TIME_SEC, new_value=result.pr_vad)

            # Process transcribed word (if any)
            if result.word:
                self._on_word(result.word)

    def _on_word(self, word: str):
        """Called when STT produces a new word.
        
        Mirrors Unmute's _stt_loop() (handler lines 436-466).
        """
        is_new = self.conversation.add_message_delta(word, "user")
        self.last_activity_time = time.time()

        if is_new:
            # CRITICAL: Reset EMA to 0.0 so we don't immediately trigger pause.
            # Exactly what Unmute does (line 465).
            self.ema.value = 0.0
            print(f"\n[YOU] {word}", end="", flush=True)
        else:
            print(f" {word}", end="", flush=True)

    def _determine_pause(self) -> bool:
        """Check if EMA crossed the pause threshold.
        
        Mirrors Unmute's determine_pause() (handler line 372).
        """
        if self.conversation.conversation_state() != "user_speaking":
            return False
        return self.ema.value > PAUSE_THRESHOLD

    def _start_flush(self):
        """Begin the non-blocking flush.
        
        Mirrors Unmute handler lines 340-351:
          1. Set end-of-flush time
          2. Feed silence to push out any buffered words
          3. Return to main loop (non-blocking!)
        """
        print(f"\n\n  ⏸️  Pause detected (EMA={self.ema.value:.2f})")

        self.stt_end_of_flush_time = self.stt.current_time + STT_DELAY_SEC

        # Feed silence to flush the STT pipeline's 0.5s buffer
        num_frames = int(math.ceil(STT_DELAY_SEC / FRAME_TIME_SEC)) + 1
        silence = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        for _ in range(num_frames):
            self.stt.feed_frame(silence)

    def _generate_response(self):
        """Generate a response after flush is complete.
        
        Mirrors Unmute's flow: _generate_response() → _tts_loop()
        After TTS finishes → add("", "user") → waiting_for_user
        """
        # Drain any final words that came during flush
        self._drain_and_process_results()

        user_text = self.conversation.get_last_user_text()
        print(f"  📝 User said: \"{user_text.strip()}\"")

        # TODO: Phase 5 — send to LLM and get response
        fake_response = f"I heard you say: {user_text.strip()}"
        self.conversation.add_message_delta(fake_response, "assistant")
        print(f"\n[BOT] {fake_response}")
        print()

        # CRITICAL: Add empty user message to transition to waiting_for_user.
        # Mirrors Unmute's _tts_loop() line 577.
        self.conversation.add_message_delta("", "user")
        self.last_activity_time = time.time()

    def _check_silence(self):
        """If no one has spoken for a while, the bot responds to silence.
        
        Mirrors Unmute's detect_long_silence() (handler line 626).
        """
        if self.conversation.conversation_state() != "waiting_for_user":
            return

        elapsed = time.time() - self.last_activity_time
        if elapsed > SILENCE_TIMEOUT:
            self.conversation.add_message_delta("...", "user")
            self.last_activity_time = time.time()
            print("\n  🤫 Long silence detected...")

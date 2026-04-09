"""
Audio I/O — Microphone Input & Speaker Output
===============================================

Handles the low-level audio: recording from the mic and playing through speakers.
Both use sounddevice, which runs callbacks in C threads (NOT Python threads).

The tricky part: sounddevice's C callbacks can't use asyncio directly.
So we bridge between worlds using thread-safe queues:

  Mic C thread → call_soon_threadsafe → asyncio.Queue → orchestrator
  Orchestrator → queue.Queue → Speaker C thread

See the orchestrator (core/orchestrator.py) for how these are used.
"""
import asyncio
import queue
import numpy as np
import sounddevice as sd
from core.config import SAMPLE_RATE, SAMPLES_PER_FRAME


class MicrophoneInput:
    """Captures audio from the system microphone in real-time.

    How it works:
      1. sounddevice opens a mic stream with a callback
      2. Every 80ms, the callback fires (in a C thread!) with 1920 samples
      3. The callback puts the audio into an asyncio.Queue
      4. The orchestrator awaits get_frame() to pop from that queue

    The C thread → asyncio bridge:
      sounddevice callbacks run in a C thread. asyncio.Queue.put_nowait()
      is NOT thread-safe from a C thread. So we use the event loop's
      call_soon_threadsafe() to schedule the put on the event loop thread.
    """

    def __init__(self):
        # asyncio.Queue — the bridge between C thread and async world
        # Each item is a numpy array of shape (1920,), dtype float32
        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.stream: sd.InputStream | None = None

        # We save the event loop reference so the C-thread callback
        # can schedule work on it via call_soon_threadsafe()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, time, status):
        """Called by sounddevice every 80ms with new audio.

        ⚠️  THIS RUNS IN A C THREAD — not the main thread, not a Python thread.
        We cannot use asyncio directly here. We MUST go through
        call_soon_threadsafe() to put items into the asyncio queue.

        Args:
            indata: numpy array, shape (1920, 1) — 1920 samples, 1 channel
            frames: number of frames (always 1920 for us)
            time: timing info (we don't use this)
            status: error flags (e.g., buffer overflow)
        """
        if status:
            # This means audio was lost (buffer overflow, device error, etc.)
            print(f"Mic warning: {status}")

        # indata shape = (1920, 1) — 1920 samples, 1 channel
        # We extract column 0 to get shape (1920,)
        # .copy() is CRITICAL — sounddevice reuses the indata buffer!
        # Without copy, the data gets overwritten by the next callback.
        audio_data = indata[:, 0].copy()

        # Schedule putting the audio into our asyncio queue.
        # call_soon_threadsafe is the ONLY safe way to touch asyncio
        # objects from a non-asyncio thread.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio_data)

    def start(self):
        """Open the mic stream and start recording.

        Must be called AFTER the asyncio event loop is running
        (because we need to capture the loop reference).
        """
        # Save the current event loop — the callback will need it
        self._loop = asyncio.get_event_loop()

        # Create and start the mic stream
        # blocksize=1920 means the callback fires every 1920 samples = 80ms
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,       # 24000 Hz — required by Mimi codec
            blocksize=SAMPLES_PER_FRAME,  # 1920 samples = 80ms per callback
            channels=1,                   # Mono audio (one channel)
            dtype="float32",              # Samples as floats between -1.0 and 1.0
            callback=self._callback,      # Our function, called every 80ms
        )
        self.stream.start()
        print("🎤 Microphone started")

    def stop(self):
        """Stop and close the mic stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("🎤 Microphone stopped")

    async def get_frame(self) -> np.ndarray:
        """Wait for and return the next audio frame.

        Returns:
            numpy array, shape (1920,), dtype float32
            This is 80ms of audio at 24kHz.

        This blocks (yields) until a frame is available.
        The orchestrator calls this once per loop iteration.
        """
        return await self.audio_queue.get()


class SpeakerOutput:
    """Plays audio through the system speakers in real-time.

    How it works:
      1. sounddevice opens an output stream with a callback
      2. Every 80ms, the callback fires asking for audio to play
      3. If we have audio in our queue → play it
      4. If queue is empty → play silence (zeros)

    Unlike MicrophoneInput, this uses a regular queue.Queue (not asyncio)
    because the speaker callback is simpler — it just needs to pop audio
    from the main thread. No async bridging needed.
    """

    def __init__(self):
        # Regular thread-safe queue (not asyncio) — fine for C thread access
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.OutputStream | None = None

    def _callback(self, outdata, frames, time, status):
        """Called by sounddevice every 80ms asking for audio to play.

        ⚠️  THIS RUNS IN A C THREAD.

        Args:
            outdata: numpy array to FILL with audio, shape (1920, 1)
            frames: number of frames requested (always 1920)
            time: timing info
            status: error flags
        """
        if status:
            print(f"Speaker warning: {status}")

        try:
            # Try to get audio from our queue (don't block!)
            audio_data = self.audio_queue.get_nowait()
            # Fill the output buffer with our audio
            outdata[:, 0] = audio_data
        except queue.Empty:
            # No audio ready — play silence (zeros)
            # This prevents crackling/popping that happens if we
            # leave the buffer with stale data.
            outdata.fill(0)

    def start(self):
        """Open the speaker stream and start playback."""
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        print("🔊 Speaker started")

    def stop(self):
        """Stop and close the speaker stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("🔊 Speaker stopped")

    def put_audio(self, data: np.ndarray):
        """Add audio to the playback queue. Called from any thread.

        Args:
            data: numpy array, shape (1920,), dtype float32
        """
        self.audio_queue.put_nowait(data)

    def clear(self):
        """Clear all pending audio — used when the user interrupts the bot.

        Discards everything in the queue so the bot stops speaking immediately.
        """
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

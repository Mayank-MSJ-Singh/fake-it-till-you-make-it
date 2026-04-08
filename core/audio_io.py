import asyncio
import queue
import numpy as np
import sounddevice as sd
from core.config import SAMPLE_RATE, SAMPLES_PER_FRAME


class MicrophoneInput:
    """Captures audio from the microphone in real-time."""

    def __init__(self):
        # This queue is the "mailbox" between the mic thread and our main code
        # asyncio.Queue is safe to use from both threads
        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.stream: sd.InputStream | None = None
        # We need a reference to the running event loop so the C-thread callback
        # can put items into the asyncio queue safely
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, time, status):
        """Called by sounddevice every 80ms with new audio from the mic.
        
        This runs in a C thread — NOT in Python's main thread!
        So we can't directly use asyncio here, we use call_soon_threadsafe.
        """
        if status:
            print(f"Mic warning: {status}")

        # indata shape = (1920, 1) — 1920 samples, 1 channel
        # .copy() is CRITICAL — indata is a temporary buffer that gets reused!
        audio_data = indata[:, 0].copy()

        # Put the audio into our asyncio queue from the C thread
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio_data)

    def start(self):
        """Open the mic stream and start recording."""
        # Save the current event loop so the callback can use it
        self._loop = asyncio.get_event_loop()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,    # 24000 Hz — required by Mimi codec
            blocksize=SAMPLES_PER_FRAME,  # 1920 samples = 80ms per callback
            channels=1,                # Mono audio
            dtype="float32",           # Float samples between -1.0 and 1.0
            callback=self._callback,   # Our function gets called every 80ms
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
        """Wait for and return the next audio frame (1920 samples)."""
        return await self.audio_queue.get()


class SpeakerOutput:
    """Plays audio through the speakers in real-time."""

    def __init__(self):
        # Regular queue.Queue (not asyncio) because the speaker callback
        # runs in a C thread which can't use asyncio
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.OutputStream | None = None

    def _callback(self, outdata, frames, time, status):
        """Called by sounddevice every 80ms asking for audio to play.
        
        This runs in a C thread — NOT in Python's main thread!
        """
        if status:
            print(f"Speaker warning: {status}")

        try:
            # Try to get audio from our queue (non-blocking)
            audio_data = self.audio_queue.get_nowait()
            # Fill the output buffer with our audio
            outdata[:, 0] = audio_data
        except queue.Empty:
            # No audio ready — fill with silence (zeros)
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
        """Add audio to play. Called from any thread."""
        self.audio_queue.put_nowait(data)

    def clear(self):
        """Clear all pending audio — used for interruption."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

"""
TTS Engine — Text-to-Speech using Pocket TTS (with true streaming)
===================================================================

Converts text to spoken audio using Kyutai's Pocket TTS.
Now supports streaming: feed text incrementally, get audio chunks in real time.

All configuration lives in core/config.py:
  - TTS_VOICE   → built-in voice name or path to .wav for cloning
  - TTS_DEVICE  → "cpu" recommended (keeps GPU for STT + LLM)
  - SAMPLES_PER_FRAME → audio chunk size for streaming (e.g., 1920)

Key methods:
  synthesize(text)          → numpy array of full audio (blocking)
  synthesize_stream(generator) → yields audio chunks as they're generated
"""
import numpy as np
import torch
from pocket_tts import TTSModel

from core.config import TTS_VOICE, TTS_DEVICE, SAMPLES_PER_FRAME


class TTSEngine:
    """Pocket TTS wrapper with both blocking and streaming synthesis."""

    def __init__(self):
        self.model = None
        self.voice_state = None

    def load_model(self):
        """Download and load Pocket TTS + voice state.

        The model downloads from HuggingFace on first run (~150MB).
        Voice state is loaded from the built-in voice or a .wav file.
        """
        print(f"Loading TTS (Pocket TTS) on {TTS_DEVICE}...")

        # Load the TTS model on the specified device
        self.model = TTSModel.load_model()

        # Load the voice — either a built-in name or a path to a .wav
        print(f"  Loading voice: {TTS_VOICE}")
        self.voice_state = self.model.get_state_for_audio_prompt(TTS_VOICE)
        
        # Move voice state to the same device if it's a tensor
        if hasattr(self.voice_state, 'to'):
            self.voice_state = self.voice_state.to(TTS_DEVICE)

        print(f"  Sample rate: {self.model.sample_rate} Hz")
        print("TTS loaded!")

    def synthesize(self, text: str) -> np.ndarray:
        """Blocking synthesis: generate entire audio for the given text.

        Args:
            text: The text to speak.

        Returns:
            numpy float32 array of audio samples at the model's sample rate.
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        # generate_audio returns a 1D torch tensor
        audio_tensor = self.model.generate_audio(self.voice_state, text)

        # Convert to numpy float32
        audio = audio_tensor.cpu().float().numpy()
        return audio

    def synthesize_stream(self, text_generator):
        """
        True streaming synthesis: feed text incrementally, yield audio chunks.

        This method uses Pocket TTS's built-in `generate_audio_stream` API.
        It consumes a generator of text chunks and yields audio frames as they
        are synthesized — without waiting for the full text.

        Args:
            text_generator: A generator that yields strings (text chunks)
                            as they become available (e.g., from an LLM).

        Yields:
            numpy float32 arrays, each a chunk of audio at the model's sample rate.
            Chunk sizes are determined by Pocket TTS (typically ~0.1s each).
        """
        # Pocket TTS's native streaming method
        audio_stream = self.model.generate_audio_stream(
            self.voice_state,
            text_generator,
            stream_chunk_size=SAMPLES_PER_FRAME,   # optional, matches speaker frame size
            frames_after_eos=None,                 # optional: extra frames after sentence end
            copy_state=True                        # optional: copy state for each call
        )
        for audio_chunk in audio_stream:
            # Convert from torch tensor to numpy float32 if needed
            if isinstance(audio_chunk, torch.Tensor):
                audio_chunk = audio_chunk.cpu().float().numpy()
            yield audio_chunk

    def synthesize_to_chunks(self, text: str) -> list[np.ndarray]:
        """Convert text to a list of speaker-ready audio chunks (blocking).

        Each chunk is exactly SAMPLES_PER_FRAME samples, ready for SpeakerOutput.
        The last chunk is zero-padded if needed.

        Args:
            text: The text to speak.

        Returns:
            List of numpy arrays, each shape (SAMPLES_PER_FRAME,), dtype float32.
        """
        audio = self.synthesize(text)
        if len(audio) == 0:
            return []

        chunks = []
        for i in range(0, len(audio), SAMPLES_PER_FRAME):
            chunk = audio[i:i + SAMPLES_PER_FRAME]
            if len(chunk) < SAMPLES_PER_FRAME:
                chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))
            chunks.append(chunk)
        return chunks
"""Test: Load Pocket TTS and speak out loud through the speaker."""
import sounddevice as sd
from tts.tts_engine import TTSEngine
from core.config import TTS_VOICE

print(f"Voice: {TTS_VOICE}")
print()

engine = TTSEngine()
engine.load_model()

print(f"\nSample rate: {engine.model.sample_rate} Hz")
print()

# Test 1: Generate and play audio
test_texts = [
    "Hello! I am your voice assistant.",
    "The weather today is sunny with a high of 25 degrees.",
    "That's a great question. Let me think about it.",
]

for text in test_texts:
    print(f"Speaking: \"{text}\"")

    # Synthesize to chunks (1920 samples each, ready for speaker)
    chunks = engine.synthesize_to_chunks(text)
    print(f"  Generated {len(chunks)} chunks ({len(chunks) * 0.08:.1f}s of audio)")

    # Play through sounddevice directly (simpler than SpeakerOutput for testing)
    audio = engine.synthesize(text)
    sd.play(audio, samplerate=engine.model.sample_rate)
    sd.wait()  # Wait until playback finishes

    print(f"  ✅ Done!\n")

print("All tests passed! 🎉")

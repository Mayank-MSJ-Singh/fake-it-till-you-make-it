"""
Test: Gemma LLM streaming → Pocket TTS ONNX streaming → Speaker

Pipeline:
  1. LLM streams text tokens
  2. Buffer into sentence fragments (split on . ! ? : ;)
  3. Each fragment → tts.stream() → yields audio chunks
  4. Audio chunks play through speaker immediately

You hear the bot start speaking ~200ms after the first sentence completes,
while the LLM is still generating the rest.
"""
import sys
import time
import numpy as np
import sounddevice as sd

# Add pocket-tts-onnx to path so we can import it
sys.path.insert(0, "pocket-tts-onnx")
from pocket_tts_onnx import PocketTTSOnnx

from llm.llm_engine import LLMEngine
from core.config import LLM_MODEL_ID

# ── Config ──
VOICE_REF = "pocket-tts-onnx/reference_sample.wav"
SAMPLE_RATE = 24000
SENTENCE_ENDERS = {'.', '!', '?', ':', ';'}

print(f"LLM: {LLM_MODEL_ID}")
print(f"Voice: {VOICE_REF}")
print()

# ── Load models ──
print("Loading LLM...")
llm = LLMEngine()
llm.load_model()

print("\nLoading TTS (ONNX)...")
tts = PocketTTSOnnx(precision="int8", device="auto")
print(f"TTS ready on {tts.device}")

# Pre-cache voice embeddings (do this once, not per sentence)
print(f"Encoding voice reference...")
voice_emb = tts.encode_voice(VOICE_REF)
print(f"Voice encoded: shape {voice_emb.shape}")

print("\n" + "=" * 60)
print("LLM → Streaming TTS! Type 'quit' to exit.")
print("=" * 60 + "\n")

history = []

while True:
    user_input = input("[YOU] ")
    if user_input.strip().lower() in ("quit", "exit", "q"):
        break

    history.append({"role": "user", "content": user_input})

    t_start = time.time()
    t_first_audio = None

    full_response = ""
    sentence_buffer = ""
    sentence_count = 0

    print("[BOT] ", end="", flush=True)

    # Stream from LLM
    for chunk in llm.generate_stream(history):
        print(chunk, end="", flush=True)
        full_response += chunk
        sentence_buffer += chunk

        # Check if we've hit a sentence boundary
        should_speak = any(ender in sentence_buffer for ender in SENTENCE_ENDERS)

        # Also speak if buffer gets too long (run-on sentences)
        if len(sentence_buffer) > 120:
            should_speak = True

        if should_speak and sentence_buffer.strip():
            # Stream this sentence through TTS → speaker
            for audio_chunk in tts.stream(
                sentence_buffer.strip(),
                voice=voice_emb,         # Use pre-cached embeddings
                first_chunk_frames=2,    # Low TTFB
            ):
                if t_first_audio is None:
                    t_first_audio = time.time()
                # Play each audio chunk immediately
                sd.play(audio_chunk, samplerate=SAMPLE_RATE)
                sd.wait()

            sentence_count += 1
            sentence_buffer = ""

    # Speak any remaining text
    if sentence_buffer.strip():
        for audio_chunk in tts.stream(
            sentence_buffer.strip(),
            voice=voice_emb,
            first_chunk_frames=2,
        ):
            if t_first_audio is None:
                t_first_audio = time.time()
            sd.play(audio_chunk, samplerate=SAMPLE_RATE)
            sd.wait()
        sentence_count += 1

    t_end = time.time()

    first_audio_ms = (t_first_audio - t_start) * 1000 if t_first_audio else 0
    print(f"\n  ⏱️  Total: {t_end - t_start:.2f}s | "
          f"First audio: {first_audio_ms:.0f}ms | "
          f"Sentences spoken: {sentence_count}\n")

    history.append({"role": "assistant", "content": full_response.strip()})

print("\n👋 Bye!")

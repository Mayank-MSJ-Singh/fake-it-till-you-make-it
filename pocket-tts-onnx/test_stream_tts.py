#!/usr/bin/env python3
"""
Streaming TTS test for PocketTTS ONNX.

Tests both batch and streaming modes, plays audio in real-time,
and reports timing/performance metrics.
"""

import time
import sys
import numpy as np

# ── Check sounddevice availability ──────────────────────────────────
try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False
    print("⚠️  sounddevice not installed — audio will be saved to file instead of played.")
    print("   Install with:  pip install sounddevice")

from pocket_tts_onnx import PocketTTSOnnx


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
TEXT = (
    "Hello! This is a streaming text to speech test. "
    "If you can hear this, the pocket TTS ONNX model is working correctly."
)
VOICE_REF = "pocket-tts-onnx/reference_sample.wav"
SAMPLE_RATE = 24000


def play_audio_blocking(audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Play an audio array through speakers and block until done."""
    if not HAS_SD:
        return
    sd.play(audio, samplerate=sr)
    sd.wait()


# ────────────────────────────────────────────────────────────────────
# Test 1 — Batch (offline) generation
# ────────────────────────────────────────────────────────────────────
def test_batch(tts: PocketTTSOnnx):
    print("\n" + "=" * 60)
    print("TEST 1 — Batch (offline) generation")
    print("=" * 60)
    print(f"  Text   : {TEXT}")
    print(f"  Voice  : {VOICE_REF}")

    t0 = time.time()
    audio = tts.generate(TEXT, voice=VOICE_REF)
    elapsed = time.time() - t0

    duration = len(audio) / SAMPLE_RATE
    rtfx = duration / elapsed if elapsed > 0 else float("inf")

    print(f"  Audio  : {duration:.2f}s  ({len(audio)} samples)")
    print(f"  Time   : {elapsed:.2f}s")
    print(f"  RTFx   : {rtfx:.1f}x realtime")

    # Basic sanity checks
    assert len(audio) > 0, "❌ No audio generated!"
    assert not np.all(audio == 0), "❌ Audio is all zeros!"
    assert np.isfinite(audio).all(), "❌ Audio contains NaN/Inf!"
    peak = np.abs(audio).max()
    print(f"  Peak   : {peak:.4f}")
    assert peak > 0.001, f"❌ Audio too quiet (peak={peak})"

    print("  ✅ Batch generation PASSED")

    # Play it
    if HAS_SD:
        print("  🔊 Playing batch audio...")
        play_audio_blocking(audio)
    else:
        tts.save_audio(audio, "test_batch_output.wav")
        print("  💾 Saved to test_batch_output.wav")

    return audio


# ────────────────────────────────────────────────────────────────────
# Test 2 — Streaming generation (with real-time playback)
# ────────────────────────────────────────────────────────────────────
def test_streaming(tts: PocketTTSOnnx):
    print("\n" + "=" * 60)
    print("TEST 2 — Streaming generation")
    print("=" * 60)
    print(f"  Text   : {TEXT}")
    print(f"  Voice  : {VOICE_REF}")

    chunks = []
    chunk_times = []

    t0 = time.time()
    ttfb = None

    print("  🔊 Streaming chunks...")

    for i, chunk in enumerate(tts.stream(TEXT, voice=VOICE_REF)):
        t_now = time.time() - t0

        if ttfb is None:
            ttfb = t_now
            print(f"  ⚡ Time to first byte (TTFB): {ttfb * 1000:.0f} ms")

        chunk_dur = len(chunk) / SAMPLE_RATE
        chunk_times.append(t_now)
        chunks.append(chunk)
        print(f"    chunk {i:3d}  |  {len(chunk):6d} samples  |  {chunk_dur:.3f}s audio  |  @ {t_now:.2f}s")

    total_elapsed = time.time() - t0

    # Combine all chunks
    full_audio = np.concatenate(chunks)
    total_duration = len(full_audio) / SAMPLE_RATE
    rtfx = total_duration / total_elapsed if total_elapsed > 0 else float("inf")

    print(f"\n  Summary:")
    print(f"    Chunks : {len(chunks)}")
    print(f"    Audio  : {total_duration:.2f}s  ({len(full_audio)} samples)")
    print(f"    Time   : {total_elapsed:.2f}s")
    print(f"    RTFx   : {rtfx:.1f}x realtime")
    print(f"    TTFB   : {ttfb * 1000:.0f} ms")

    # Sanity checks
    assert len(chunks) > 1, f"❌ Expected multiple chunks, got {len(chunks)}"
    assert len(full_audio) > 0, "❌ No audio generated!"
    assert not np.all(full_audio == 0), "❌ Audio is all zeros!"
    assert np.isfinite(full_audio).all(), "❌ Audio contains NaN/Inf!"
    peak = np.abs(full_audio).max()
    print(f"    Peak   : {peak:.4f}")
    assert peak > 0.001, f"❌ Audio too quiet (peak={peak})"
    assert ttfb < 10.0, f"❌ TTFB too high: {ttfb:.2f}s"

    print("  ✅ Streaming generation PASSED")

    # Play the concatenated result
    if HAS_SD:
        print("  🔊 Playing streamed audio...")
        play_audio_blocking(full_audio)
    else:
        tts.save_audio(full_audio, "test_stream_output.wav")
        print("  💾 Saved to test_stream_output.wav")

    return full_audio


# ────────────────────────────────────────────────────────────────────
# Test 3 — Compare batch vs streaming output similarity
# ────────────────────────────────────────────────────────────────────
def test_similarity(batch_audio: np.ndarray, stream_audio: np.ndarray):
    print("\n" + "=" * 60)
    print("TEST 3 — Batch vs Streaming comparison")
    print("=" * 60)

    batch_dur = len(batch_audio) / SAMPLE_RATE
    stream_dur = len(stream_audio) / SAMPLE_RATE
    dur_diff = abs(batch_dur - stream_dur)

    print(f"  Batch duration  : {batch_dur:.2f}s")
    print(f"  Stream duration : {stream_dur:.2f}s")
    print(f"  Duration diff   : {dur_diff:.2f}s")

    # Durations should be reasonably close (same text/voice, but stochastic)
    if dur_diff < 1.0:
        print("  ✅ Duration similarity OK (within 1s)")
    else:
        print(f"  ⚠️  Duration difference is large ({dur_diff:.2f}s) — this is expected due to stochastic generation")

    # Both should have similar energy levels
    batch_rms = np.sqrt(np.mean(batch_audio ** 2))
    stream_rms = np.sqrt(np.mean(stream_audio ** 2))
    print(f"  Batch RMS  : {batch_rms:.4f}")
    print(f"  Stream RMS : {stream_rms:.4f}")

    # RMS should be in the same ballpark (within 10x of each other)
    ratio = max(batch_rms, stream_rms) / (min(batch_rms, stream_rms) + 1e-10)
    assert ratio < 10.0, f"❌ RMS ratio too large: {ratio:.1f}x"
    print(f"  RMS ratio  : {ratio:.2f}x")
    print("  ✅ Output similarity PASSED")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main():
    print("Loading PocketTTS ONNX model...")
    t0 = time.time()
    tts = PocketTTSOnnx(precision="int8", device="auto")
    load_time = time.time() - t0
    print(f"Model ready on {tts.device}  (loaded in {load_time:.2f}s)")
    print(f"Config: {tts}")

    batch_audio = test_batch(tts)
    stream_audio = test_streaming(tts)
    test_similarity(batch_audio, stream_audio)

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

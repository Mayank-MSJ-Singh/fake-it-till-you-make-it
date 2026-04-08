"""Diagnostic: check if STT worker keeps up with real-time audio.
Run this to verify the pipeline isn't falling behind.
"""
import time
import numpy as np
import sounddevice as sd
from stt.stt_engine import STTEngine

SAMPLE_RATE = 24000
FRAMES = 1920

def main():
    stt = STTEngine(device="cuda")
    stt.load_model()
    stt.warmup()
    stt.start()

    print("\n🎤 Speak! Monitoring queue depth. Ctrl+C to stop.\n")
    print(f" {'sec':>5} | {'fed':>5} | {'got':>5} | {'Qin':>5} | {'Qout':>5} | {'word':>15} | {'token':>5} | {'P(PAD)':>7} | {'EMA':>5}")
    print("-" * 95)

    # Open mic
    audio_queue = []
    def callback(indata, frames, t, status):
        audio_queue.append(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=FRAMES,
                            channels=1, dtype="float32", callback=callback)
    stream.start()

    fed = 0
    got = 0
    ema = 1.0
    alpha_up = 0.309  # attack for 0.15s
    alpha_dn = 0.672  # release for 0.05s
    start = time.time()

    try:
        while True:
            # Wait for mic frame
            while not audio_queue:
                time.sleep(0.001)
            frame = audio_queue.pop(0)

            stt.feed_frame(frame)
            fed += 1

            # Drain results
            while True:
                r = stt.get_result_nowait()
                if r is None:
                    break
                got += 1

                if r.pr_vad > ema:
                    ema = (1 - alpha_up) * ema + alpha_up * r.pr_vad
                else:
                    ema = (1 - alpha_dn) * ema + alpha_dn * r.pr_vad

                elapsed = time.time() - start
                word_str = r.word if r.word else ""
                lag = fed - got
                print(f" {elapsed:5.1f} | {fed:5} | {got:5} | {stt.mic_queue.qsize():5} | {stt.result_queue.qsize():5} | {word_str:>15} | {r.text_token:5} | {r.pr_vad:7.4f} | {ema:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        stt.stop()
        print(f"\nFed {fed} frames, got {got} results. Lag: {fed - got}")

if __name__ == "__main__":
    main()

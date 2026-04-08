"""
VAD Tuning Tool — Speak and watch the pause detection values in real-time.

Displays a live bar showing:
  - Raw pr_vad from the model (instant, noisy)
  - Smoothed EMA value (what we actually use for pause detection)
  - Current threshold line
  - Words as they appear

Adjust these values at the top and restart to experiment:
"""

# ============================================================
# 🎛️  TUNE THESE VALUES
# ============================================================
ATTACK_TIME = 0.15       # How slow EMA rises (speaking → paused). Higher = more patience
RELEASE_TIME = 0.05      # How fast EMA drops (paused → speaking). Lower = faster response
PAUSE_THRESHOLD = 0.75   # EMA must exceed this to trigger "pause detected"
MIN_SPEAKING_TIME = 0.5  # Seconds user must speak before pause can trigger
# ============================================================

import asyncio
import time
import sys

from core.audio_io import MicrophoneInput
from core.config import FRAME_TIME_SEC
from stt.stt_engine import STTEngine
from stt.ema import ExponentialMovingAverage


def draw_bar(label: str, value: float, threshold: float = None, width: int = 40) -> str:
    """Draw a horizontal bar with optional threshold marker."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)

    # Insert threshold marker
    if threshold is not None:
        pos = int(threshold * width)
        if 0 <= pos < width:
            bar = bar[:pos] + "│" + bar[pos + 1:]

    return f"  {label}: [{bar}] {value:.3f}"


async def main():
    print("Loading STT model...")
    stt = STTEngine(device="cuda")
    stt.load_model()
    stt.warmup()

    mic = MicrophoneInput()
    mic.start()
    stt.start()

    ema = ExponentialMovingAverage(
        attack_time=ATTACK_TIME,
        release_time=RELEASE_TIME,
        initial_value=1.0,
    )

    print(f"\n{'=' * 60}")
    print(f"  🎛️  VAD Tuning Tool")
    print(f"  attack={ATTACK_TIME}  release={RELEASE_TIME}  threshold={PAUSE_THRESHOLD}")
    print(f"  min_speaking={MIN_SPEAKING_TIME}s")
    print(f"{'=' * 60}")
    print(f"  Speak naturally. Watch the bars. Press Ctrl+C to stop.\n")

    transcript = ""
    speaking_since = None
    last_raw = 0.0
    pause_triggered = False

    try:
        while True:
            frame = await mic.get_frame()
            stt.feed_frame(frame)

            while True:
                result = stt.get_result_nowait()
                if result is None:
                    break

                last_raw = result.pr_vad
                ema.update(dt=FRAME_TIME_SEC, new_value=result.pr_vad)

                if result.word:
                    transcript += result.word
                    if speaking_since is None:
                        speaking_since = time.time()
                    pause_triggered = False

                # Check pause
                speaking_time = (time.time() - speaking_since) if speaking_since else 0
                is_paused = (
                    ema.value > PAUSE_THRESHOLD
                    and speaking_since is not None
                    and speaking_time > MIN_SPEAKING_TIME
                )

                if is_paused and not pause_triggered:
                    pause_triggered = True
                    speaking_since = None  # Reset for next utterance

                # Draw the display
                raw_bar = draw_bar("raw_vad", last_raw, PAUSE_THRESHOLD)
                ema_bar = draw_bar("ema    ", ema.value, PAUSE_THRESHOLD)

                # Status indicator
                if is_paused or pause_triggered:
                    status = "  ⏸️  PAUSE DETECTED"
                elif speaking_since:
                    status = f"  🎤 Speaking ({speaking_time:.1f}s)"
                else:
                    status = "  ⏳ Waiting..."

                # Last 60 chars of transcript
                shown_text = transcript[-60:] if len(transcript) > 60 else transcript

                # Clear and redraw (simple terminal refresh)
                sys.stdout.write(f"\033[2K\033[A" * 5)  # Clear 5 lines
                print(raw_bar)
                print(ema_bar)
                print(f"  status: {status}")
                print(f"  text:   {shown_text}")
                print(f"  threshold: {PAUSE_THRESHOLD} | attack: {ATTACK_TIME} | release: {RELEASE_TIME}")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 60}")
        print("  Final transcript:")
        print(f"  {transcript}")
        print(f"{'=' * 60}")
    finally:
        stt.stop()
        mic.stop()

# Print initial blank lines so the display has room
print("\n" * 5)
asyncio.run(main())

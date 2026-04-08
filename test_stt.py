"""Test STT — speak into mic and see words appear."""
import asyncio
from core.audio_io import MicrophoneInput
from stt.stt_engine import STTEngine
import time


async def main():
    # Load model
    stt = STTEngine(device="cuda")
    stt.load_model()

    # Start mic and STT
    mic = MicrophoneInput()
    mic.start()
    stt.start()

    print("\n🎤 Speak and watch your words appear! Press Ctrl+C to stop.\n")

    try:
        while True:
            frame = await mic.get_frame()
            stt.feed_frame(frame)
            
            # Drain ALL available results (not just one!)
            while True:
                result = stt.get_result_nowait()
                if result is None:
                    break
                if result.word:
                    print(result.word, end="", flush=True)
            
            # Print timing every 100 frames
            if hasattr(stt, '_frame_count') and stt._frame_count % 100 == 0 and stt._frame_count > 0:
                print(f"\n[Frame {stt._frame_count}, queue size: {stt.result_queue.qsize()}]")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stt.stop()
        mic.stop()


asyncio.run(main())

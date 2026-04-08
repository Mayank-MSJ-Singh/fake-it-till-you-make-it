"""Echo test — records from mic and immediately plays back through speakers.
Wear headphones to avoid feedback loop!"""

import asyncio
from core.audio_io import MicrophoneInput, SpeakerOutput


async def main():
    mic = MicrophoneInput()
    speaker = SpeakerOutput()

    mic.start()
    speaker.start()

    print("Speaking into mic should echo back through headphones...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Get audio frame from mic
            frame = await mic.get_frame()
            # Immediately send it to speakers
            speaker.put_audio(frame)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        speaker.stop()


asyncio.run(main())

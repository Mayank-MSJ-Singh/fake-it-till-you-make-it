"""
Fake It Till You Make It — Terminal Voice AI
=============================================

This is the entry point. It does exactly one thing:
create an Orchestrator and run its main loop.

The Orchestrator handles everything — loading models, capturing audio,
running STT, detecting pauses, and generating responses. We just need
to kick it off inside an asyncio event loop.

Why asyncio?
  The microphone (sounddevice) runs in a C thread. The STT runs in a
  Python thread. The orchestrator needs to coordinate them without
  blocking. asyncio gives us `await mic.get_frame()` which waits for
  audio without freezing the whole program.

Run with:
  python main.py
"""
import asyncio
from core.orchestrator import Orchestrator


async def main():
    # Create the orchestrator on the GPU.
    # This is the brain of the whole system — see core/orchestrator.py
    orch = Orchestrator(device="cuda")

    # This call never returns (until Ctrl+C).
    # Inside, it runs an infinite loop: mic → STT → pause detect → respond
    await orch.run()


if __name__ == "__main__":
    # asyncio.run() creates an event loop and runs our async main().
    # The event loop is what makes `await` work — it's the scheduler
    # that juggles between waiting for mic frames, draining results, etc.
    asyncio.run(main())

"""Fake It Till You Make It — Terminal Voice AI"""
import asyncio
from core.orchestrator import Orchestrator


async def main():
    orch = Orchestrator(device="cuda")
    await orch.run()


if __name__ == "__main__":
    asyncio.run(main())

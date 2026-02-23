# main.py
"""AlphaTemp -- async entrypoint."""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from core.db import init_db
from services.ingestor import SynopticIngestor
from services.forecast import HRRRFetcher
from services.bias import BiasEngine


async def main():
    load_dotenv()
    token = os.getenv("SYNOPTIC_TOKEN")
    if not token:
        logger.error("SYNOPTIC_TOKEN not found in environment")
        return

    init_db()

    ingestor = SynopticIngestor(token=token)
    fetcher = HRRRFetcher()
    engine = BiasEngine()

    await asyncio.gather(
        ingestor.run(),
        fetcher.run(),
        engine.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())

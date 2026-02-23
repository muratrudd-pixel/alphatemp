# main.py
"""AlphaTemp -- async entrypoint."""

import asyncio
import os
import sys

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

    tasks = [
        ingestor.run(),
        fetcher.run(),
        engine.run(),
    ]

    if "--dashboard" in sys.argv:
        import uvicorn
        config = uvicorn.Config("ui.web_dashboard:app", host="0.0.0.0", port=8050)
        server = uvicorn.Server(config)
        tasks.append(server.serve())
        logger.info("Dashboard will be available at http://localhost:8050")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

# main.py
"""AlphaTemp -- async entrypoint."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from core.db import init_db, DEFAULT_DB_PATH
from services.ingestor import SynopticIngestor
from services.forecast import HRRRFetcher
from services.bias import BiasEngine
from services.market_fetcher import MarketFetcher
from services.nws_fetcher import NWSFetcher
from services.iem_ingestor import IEMIngestor
from services.paper_trader import PaperTrader
from services.strategy_engine import StrategyEngine
from services.settlement import SettlementService
from services.multi_model_fetcher import MultiModelFetcher


async def main():
    load_dotenv()
    init_db()

    iem = IEMIngestor()
    fetcher = HRRRFetcher()
    engine = BiasEngine()
    market = MarketFetcher()
    nws = NWSFetcher()
    paper_trader = PaperTrader()
    strategy = StrategyEngine(db_path=DEFAULT_DB_PATH, paper_trader=paper_trader)
    settlement = SettlementService(db_path=DEFAULT_DB_PATH, paper_trader=paper_trader)
    multi_model = MultiModelFetcher(db_path=DEFAULT_DB_PATH)

    tasks = [
        iem.run(),         # IEM — 5 settlement stations (low-latency SPECI)
        fetcher.run(),
        engine.run(),
        market.run(),
        nws.run(),
        multi_model.run(),   # GFS + ECMWF daily forecasts (Open-Meteo)
        paper_trader.run(),  # Paper trading (placeholder strategy)
        strategy.run(),      # QR model evaluation + trade signals
        settlement.run(),    # Settlement resolution + P&L
    ]

    # Synoptic is optional — trial may be expired
    token = os.getenv("SYNOPTIC_TOKEN")
    if token:
        ingestor = SynopticIngestor(token=token)
        tasks.append(ingestor.run())
        logger.info("Synoptic ingestor enabled (11 stations)")
    else:
        logger.warning("SYNOPTIC_TOKEN not set — Synoptic ingestor disabled (AWC/IEM only)")

    if "--dashboard" in sys.argv:
        import uvicorn
        config = uvicorn.Config("ui.web_dashboard:app", host="0.0.0.0", port=8050)
        server = uvicorn.Server(config)
        tasks.append(server.serve())
        logger.info("Dashboard will be available at http://localhost:8050")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

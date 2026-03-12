"""Strategy Engine — orchestrates the QR model, feature builder, circuit breakers,
and paper trader into an autonomous trading loop.

Runs every 5 minutes. Each cycle:
1. Determine target date (tomorrow)
2. Check data freshness (skip if nothing new)
3. Get training data and fit model
4. Build today's features
5. Predict bracket probabilities
6. Get market prices
7. Generate trade signals
8. Check edge reversals for open positions
9. Send signals through circuit breakers to paper trader

Python 3.9 compatible (no subscripted builtins).
"""

import asyncio
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import duckdb
from loguru import logger

from services.circuit_breakers import CircuitBreakers
from services.feature_builder import FeatureBuilder
from services.model import QRModel

# Timezone for ET
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    _ET = timezone(timedelta(hours=-5))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CYCLE_INTERVAL = 300  # seconds (5 minutes)
BRACKET_WIDTH = 2     # Kalshi brackets are 2°F wide


class StrategyEngine:
    """Autonomous strategy engine that evaluates edge and sends trade signals."""

    def __init__(self, db_path, paper_trader):
        # type: (str, Any) -> None
        self.db_path = db_path
        self.paper_trader = paper_trader
        self.model = QRModel()
        self.feature_builder = FeatureBuilder(db_path)
        self.circuit_breakers = CircuitBreakers(db_path)
        self.min_edge_pct = 5.0  # will be read from paper_config
        self._last_data_hash = None  # type: Optional[str]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        # type: () -> None
        """Main async loop — runs every 5 minutes."""
        logger.info("StrategyEngine started (cycle every {}s)", CYCLE_INTERVAL)
        while True:
            try:
                await self._cycle()
            except Exception as e:
                logger.error("StrategyEngine cycle error: {}", e)
            await asyncio.sleep(CYCLE_INTERVAL)

    async def _cycle(self):
        # type: () -> None
        """Single evaluation cycle."""
        now_et = datetime.now(_ET)
        target_date = (now_et + timedelta(days=1)).date()
        update_hour = now_et.hour
        date_key = target_date.isoformat()

        logger.info(
            "Strategy cycle: target={}, update_hour={}, now_et={}",
            target_date, update_hour, now_et.strftime("%H:%M"),
        )

        # 1. Load config (min_edge_pct may have been adjusted via dashboard)
        self._load_config()

        # 2. Check data freshness — skip if nothing new
        data_hash = self._get_data_hash()
        if data_hash == self._last_data_hash:
            logger.debug("No new data since last cycle, skipping")
            return
        self._last_data_hash = data_hash

        # 3. Get training data and fit model
        train_result = self.feature_builder.get_training_data(
            target_date, update_hour
        )
        if train_result is None:
            logger.warning("Insufficient training data for {}", target_date)
            return
        X_train, y_train, train_dates = train_result

        coefficients = self.model.fit(
            X_train, y_train, run_hour=0, date_key=date_key
        )
        if coefficients is None:
            logger.warning("Model fit failed for {}", target_date)
            return
        logger.info(
            "Model fit OK: {} samples, {} features",
            len(y_train), X_train.shape[1],
        )

        # 4. Build today's features
        feat_result = self.feature_builder.build_features(
            target_date, update_hour
        )
        if feat_result is None:
            logger.warning("Feature build failed for {}", target_date)
            return
        features, fcst_high = feat_result

        # 5. Predict bracket probabilities
        bracket_probs = self.model.predict_bracket_probs(
            features, fcst_high, run_hour=0, date_key=date_key
        )
        if bracket_probs is None:
            logger.warning("Prediction failed for {}", target_date)
            return
        logger.info(
            "Predicted {} brackets, center ~{}°F",
            len(bracket_probs),
            round(fcst_high),
        )

        # 5b. Persist model state for dashboard consumption
        self._persist_model_state(
            city="NYC",
            target_date=target_date,
            update_hour=update_hour,
            bracket_probs=bracket_probs,
            fcst_high=fcst_high,
        )

        # 6. Get market prices
        market_prices = self._get_market_prices(target_date)
        if not market_prices:
            logger.warning("No market prices for {}", target_date)
            return

        # 7. Generate trade signals
        signals = self._generate_signals(bracket_probs, market_prices)
        logger.info("Generated {} trade signals", len(signals))

        # 8. Check edge reversals for open positions
        open_positions = self._get_open_positions(target_date)
        if open_positions:
            self._check_edge_reversals(
                bracket_probs, market_prices, open_positions
            )

        # 9. Send signals through circuit breakers to paper trader
        market_date_str = target_date.isoformat()
        for sig in signals:
            bracket_floor = sig["bracket_floor"]
            bracket_cap = bracket_floor + BRACKET_WIDTH

            allowed, reason = self.circuit_breakers.check(
                bracket_floor=bracket_floor,
                bracket_cap=bracket_cap,
                edge_pct=sig["edge_pct"],
                market_date=market_date_str,
            )

            if allowed:
                logger.info(
                    "TRADE: {} [{}–{}) edge={:.1f}% model={:.1f}% market={}c",
                    sig["direction"], bracket_floor, bracket_cap,
                    sig["edge_pct"], sig["model_prob"] * 100,
                    sig["market_price"],
                )
                self.paper_trader.enter_position(
                    city="nyc",
                    event_date=market_date_str,
                    bracket_floor=bracket_floor,
                    bracket_cap=bracket_cap,
                    direction=sig["direction"],
                    model_prob=sig["model_prob"],
                    market_price=sig["market_price"],
                    entry_price=sig["market_price"],
                )
            else:
                logger.debug(
                    "Blocked: [{}–{}) {} — {}",
                    bracket_floor, bracket_cap, sig["direction"], reason,
                )

    # ------------------------------------------------------------------
    # Pure logic methods (tested directly)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_edge(model_prob, market_ask_cents):
        # type: (float, int) -> float
        """Compute edge in percentage points.

        Parameters
        ----------
        model_prob : float
            Model's predicted probability (0.0 to 1.0).
        market_ask_cents : int
            Market ask price in cents (0-100).

        Returns
        -------
        float
            Edge in percentage points: (model_prob - market/100) * 100.
        """
        return (model_prob - market_ask_cents / 100.0) * 100.0

    def _generate_signals(self, bracket_probs, market_prices):
        # type: (Dict[int, float], Dict[int, Dict[str, int]]) -> List[Dict[str, Any]]
        """Generate trade signals by comparing model probs to market prices.

        For each bracket:
        - Check YES edge: model_prob vs yes_ask
        - Check NO edge: (1 - model_prob) vs no_ask
        - Only include signals where edge >= min_edge_pct
        - Don't signal both YES and NO on same bracket (prefer YES)

        Parameters
        ----------
        bracket_probs : dict
            {bracket_floor: probability} from model (1°F brackets, normalized).
        market_prices : dict
            {bracket_floor: {yes_bid, yes_ask, no_bid, no_ask}} in cents.

        Returns
        -------
        list of dicts
            Each: {bracket_floor, direction, model_prob, market_price, edge_pct}.
        """
        signals = []  # type: List[Dict[str, Any]]

        for bracket_floor, model_prob in bracket_probs.items():
            prices = market_prices.get(bracket_floor)
            if prices is None:
                continue

            yes_ask = prices["yes_ask"]
            no_ask = prices["no_ask"]

            # YES edge: model says bracket more likely than market
            yes_edge = self._compute_edge(model_prob, yes_ask)

            # NO edge: model says bracket less likely than market
            no_prob = 1.0 - model_prob
            no_edge = self._compute_edge(no_prob, no_ask)

            if yes_edge >= self.min_edge_pct:
                signals.append({
                    "bracket_floor": bracket_floor,
                    "direction": "YES",
                    "model_prob": model_prob,
                    "market_price": yes_ask,
                    "edge_pct": yes_edge,
                })
            elif no_edge >= self.min_edge_pct:
                signals.append({
                    "bracket_floor": bracket_floor,
                    "direction": "NO",
                    "model_prob": model_prob,
                    "market_price": no_ask,
                    "edge_pct": no_edge,
                })

        return signals

    def _check_edge_reversals(self, bracket_probs, market_prices, open_positions):
        # type: (Dict[int, float], Dict[int, Dict[str, int]], List[Dict[str, Any]]) -> None
        """Check open positions for edge reversals and exit if flipped.

        For each open position, recompute edge with latest model/market.
        If edge < 0 (flipped), call paper_trader.exit_position.

        Parameters
        ----------
        bracket_probs : dict
            Latest model bracket probabilities.
        market_prices : dict
            Latest market prices in cents.
        open_positions : list of dicts
            Each: {id, bracket_floor, bracket_cap, direction, entry_price}.
        """
        for pos in open_positions:
            bracket_floor = pos["bracket_floor"]
            model_prob = bracket_probs.get(bracket_floor)
            prices = market_prices.get(bracket_floor)

            if model_prob is None or prices is None:
                # Can't evaluate — skip rather than force-exit
                continue

            if pos["direction"] == "YES":
                # Recompute YES edge against current yes_ask
                edge = self._compute_edge(model_prob, prices["yes_ask"])
                exit_price = prices["yes_bid"]
            else:
                # Recompute NO edge against current no_ask
                no_prob = 1.0 - model_prob
                edge = self._compute_edge(no_prob, prices["no_ask"])
                exit_price = prices["no_bid"]

            if edge < 0:
                logger.info(
                    "Edge reversal: pos {} [{}–{}) {} edge={:.1f}%",
                    pos["id"], bracket_floor, pos["bracket_cap"],
                    pos["direction"], edge,
                )
                self.paper_trader.exit_position(
                    pos["id"], exit_price, "edge_reversal"
                )

    # ------------------------------------------------------------------
    # DB query methods
    # ------------------------------------------------------------------

    def _persist_model_state(self, city, target_date, update_hour, bracket_probs, fcst_high):
        # type: (str, date, int, Dict[int, float], float) -> None
        """Write latest bracket probabilities to model_state for dashboard."""
        probs_json = json.dumps(
            {str(k): round(v, 6) for k, v in bracket_probs.items()}
        )
        con = duckdb.connect(self.db_path)
        try:
            con.execute("""
                INSERT INTO model_state (city, target_date, update_hour, bracket_probs, fcst_high, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (city, target_date) DO UPDATE SET
                    update_hour = EXCLUDED.update_hour,
                    bracket_probs = EXCLUDED.bracket_probs,
                    fcst_high = EXCLUDED.fcst_high,
                    updated_at = CURRENT_TIMESTAMP
            """, [city, target_date.isoformat(), update_hour, probs_json, fcst_high])
            logger.debug(
                "Persisted model_state: city={}, date={}, hour={}, brackets={}",
                city, target_date, update_hour, len(bracket_probs),
            )
        except Exception as e:
            logger.warning("Failed to persist model_state: {}", e)
        finally:
            con.close()

    def _load_config(self):
        # type: () -> None
        """Load min_edge_pct from paper_config table."""
        try:
            con = duckdb.connect(self.db_path)
            row = con.execute(
                "SELECT value FROM paper_config WHERE key = 'min_edge_pct'"
            ).fetchone()
            con.close()
            if row:
                self.min_edge_pct = float(row[0])
        except Exception:
            pass  # keep current value

    def _get_market_prices(self, target_date):
        # type: (date) -> Dict[int, Dict[str, int]]
        """Query latest market_ticks for NYC, return prices in cents.

        Market prices are stored as decimals (0-1) in DB, multiply by 100
        for cents.

        Returns
        -------
        dict
            {bracket_floor_int: {yes_bid, yes_ask, no_bid, no_ask}} in cents.
        """
        con = duckdb.connect(self.db_path)
        try:
            # Get the latest tick per bracket for the target date's markets
            # market_ticks uses floor_strike/cap_strike as DOUBLEs
            rows = con.execute("""
                WITH latest AS (
                    SELECT floor_strike, cap_strike,
                           yes_bid, yes_ask, no_bid, no_ask,
                           ROW_NUMBER() OVER (
                               PARTITION BY floor_strike, cap_strike
                               ORDER BY captured_at DESC
                           ) AS rn
                    FROM market_ticks
                    WHERE city = 'nyc'
                )
                SELECT floor_strike, cap_strike,
                       yes_bid, yes_ask, no_bid, no_ask
                FROM latest
                WHERE rn = 1
            """).fetchall()
        finally:
            con.close()

        prices = {}  # type: Dict[int, Dict[str, int]]
        for row in rows:
            floor_strike, cap_strike, yes_bid, yes_ask, no_bid, no_ask = row
            if any(v is None for v in [yes_bid, yes_ask, no_bid, no_ask]):
                continue
            bracket_floor = int(floor_strike)
            prices[bracket_floor] = {
                "yes_bid": int(round(yes_bid * 100)),
                "yes_ask": int(round(yes_ask * 100)),
                "no_bid": int(round(no_bid * 100)),
                "no_ask": int(round(no_ask * 100)),
            }
        return prices

    def _get_open_positions(self, target_date):
        # type: (date) -> List[Dict[str, Any]]
        """Get open positions for the target date."""
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT id, bracket_floor, bracket_cap, direction, entry_price
                FROM paper_positions
                WHERE status = 'open'
                  AND city = 'nyc'
                  AND event_date = ?
            """, [target_date.isoformat()]).fetchall()
        finally:
            con.close()

        return [
            {
                "id": r[0],
                "bracket_floor": r[1],
                "bracket_cap": r[2],
                "direction": r[3],
                "entry_price": r[4],
            }
            for r in rows
        ]

    def _get_data_hash(self):
        # type: () -> str
        """Check latest timestamps for forecasts, drift_signals, market_ticks.

        Returns a hash string. If same as last cycle, we skip processing.
        """
        con = duckdb.connect(self.db_path)
        try:
            parts = []  # type: List[str]

            # Latest HRRR forecast
            r = con.execute(
                "SELECT MAX(model_run) FROM forecasts WHERE model_name = 'hrrr'"
            ).fetchone()
            parts.append(str(r[0]) if r and r[0] else "")

            # Latest GFS forecast
            r = con.execute(
                "SELECT MAX(model_run) FROM forecasts WHERE model_name = 'gfs'"
            ).fetchone()
            parts.append(str(r[0]) if r and r[0] else "")

            # Latest ECMWF forecast
            r = con.execute(
                "SELECT MAX(model_run) FROM forecasts WHERE model_name = 'ecmwf'"
            ).fetchone()
            parts.append(str(r[0]) if r and r[0] else "")

            # Latest drift signal
            r = con.execute(
                "SELECT MAX(calculated_at) FROM drift_signals"
            ).fetchone()
            parts.append(str(r[0]) if r and r[0] else "")

            # Latest market tick
            r = con.execute(
                "SELECT MAX(captured_at) FROM market_ticks"
            ).fetchone()
            parts.append(str(r[0]) if r and r[0] else "")

        finally:
            con.close()

        combined = "|".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()

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
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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
        self.min_edge_pct = 5.0       # will be read from paper_config
        self.min_ev_cents = 2         # will be read from paper_config
        self.min_model_prob = 0.15    # will be read from paper_config
        self.starting_capital = 100.0  # will be read from paper_config
        self.max_per_bracket = 10     # will be read from paper_config
        self._last_data_hash = None   # type: Optional[str]

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
        target_date = now_et.date()  # Today's date — markets for today are live
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
        X_train, y_train, train_dates, run_hour = train_result

        coefficients = self.model.fit(
            X_train, y_train, run_hour=run_hour, date_key=date_key
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
        features, fcst_high, _feat_run_hour = feat_result

        # 4b. Get observed running max as CDF floor
        running_max = self._get_running_max(target_date)
        if running_max is not None:
            logger.info("Running max: {:.1f}°F", running_max)

        # 5. Predict bracket probabilities
        bracket_probs = self.model.predict_bracket_probs(
            features, fcst_high, run_hour=run_hour, date_key=date_key,
            running_max=running_max,
        )
        if bracket_probs is None:
            logger.warning("Prediction failed for {}", target_date)
            return

        # 5a. Post-model late-day clamp: collapse distribution when high is set
        if running_max is not None:
            current_temp = self._get_current_temp(target_date)
            temp_drop = (running_max - current_temp) if current_temp is not None else 0.0
            forecast_upside = self._get_forecast_upside(target_date, update_hour, running_max)
            from services.model import clamp_late_day
            bracket_probs = clamp_late_day(bracket_probs, running_max, temp_drop, forecast_upside)

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
        logger.info("Generated {} raw trade signals", len(signals))

        # 8. Check edge reversals for open positions
        open_positions = self._get_open_positions(target_date)
        if open_positions:
            await self._check_edge_reversals(
                bracket_probs, market_prices, open_positions
            )

        # 8b. Dedup guard — filter signals matching ANY existing position (open or settled).
        # Without this, settled positions get re-opened every cycle.
        all_positions = self._get_positions_for_dedup(target_date)
        if all_positions:
            held = {(p["bracket_floor"], p["bracket_cap"], p["direction"]) for p in all_positions}
            before = len(signals)
            signals = [s for s in signals if (s["bracket_floor"], s["bracket_cap"], s["direction"]) not in held]
            if before != len(signals):
                logger.info("Dedup: filtered {} -> {} signals", before, len(signals))

        # 9. Send signals through circuit breakers to paper trader
        market_date_str = target_date.isoformat()
        for sig in signals:
            bracket_floor = sig["bracket_floor"]
            bracket_cap = sig["bracket_cap"]

            allowed, reason = self.circuit_breakers.check(
                bracket_floor=bracket_floor,
                bracket_cap=bracket_cap,
                edge_pct=sig["edge_pct"],
                market_date=market_date_str,
            )

            if allowed:
                # Format bracket label for logging
                if bracket_floor is None:
                    label = "<={}".format(bracket_cap)
                elif bracket_cap is None:
                    label = ">={}".format(bracket_floor)
                else:
                    label = "[{}-{})".format(bracket_floor, bracket_cap)

                logger.info(
                    "TRADE: {} {} edge={:.1f}% model={:.1f}% market={}c x{}",
                    sig["direction"], label,
                    sig["edge_pct"], sig["model_prob"] * 100,
                    sig["market_price"], sig["contracts"],
                )
                await self.paper_trader.enter_position(
                    city="NYC",
                    event_date=market_date_str,
                    bracket_floor=bracket_floor,
                    bracket_cap=bracket_cap,
                    direction=sig["direction"],
                    model_prob=sig["model_prob"],
                    market_price=sig["market_price"],
                    edge=sig["edge_pct"],
                    contracts=sig["contracts"],
                )
            else:
                logger.debug(
                    "Blocked: ({},{}) {} — {}",
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

    @staticmethod
    def _compute_contracts(model_prob, price_cents, bankroll, max_per_bracket):
        # type: (float, int, float, int) -> int
        """Half-Kelly position sizing for binary outcome markets.

        Formula:
            kelly_f = model_prob - price/100  (= edge as decimal)
            half_kelly = kelly_f / 2
            contracts = floor(half_kelly * bankroll / price_per_contract)
            contracts = clamp(contracts, 1, max_per_bracket)

        Parameters
        ----------
        model_prob : float
            The traded probability (model_prob for YES, 1-model_prob for NO).
        price_cents : int
            Ask price in cents (cost per contract).
        bankroll : float
            Current bankroll in dollars.
        max_per_bracket : int
            Maximum contracts per bracket (risk cap).

        Returns
        -------
        int
            Number of contracts to trade (at least 1, at most max_per_bracket).
        """
        if price_cents <= 0 or bankroll <= 0:
            return 1
        edge_decimal = model_prob - price_cents / 100.0
        if edge_decimal <= 0:
            return 1
        half_kelly = edge_decimal / 2.0
        price_dollars = price_cents / 100.0
        raw = half_kelly * bankroll / price_dollars
        return max(1, min(int(raw), max_per_bracket))

    def _get_bankroll(self):
        # type: () -> float
        """Compute current bankroll: starting_capital + sum(net_pnl)."""
        con = duckdb.connect(self.db_path)
        try:
            row = con.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) FROM paper_positions"
            ).fetchone()
            total_pnl = float(row[0]) if row else 0.0
            return self.starting_capital + total_pnl
        finally:
            con.close()

    @staticmethod
    def _aggregate_to_kalshi_brackets(bracket_probs, market_prices):
        # type: (Dict[int, float], Dict) -> Dict
        """Aggregate 1-degree model probabilities into Kalshi brackets.

        Handles three bracket types:
        - Lower tail (None, cap): sum probs for k < cap
        - Upper tail (floor, None): sum probs for k > floor
        - Interior (floor, cap): sum probs for floor <= k <= cap
        """
        aggregated = {}
        for key in market_prices:
            floor, cap = key
            if floor is None and cap is not None:
                total = sum(p for k, p in bracket_probs.items() if k < cap)
            elif cap is None and floor is not None:
                total = sum(p for k, p in bracket_probs.items() if k > floor)
            elif floor is not None and cap is not None:
                total = sum(p for k, p in bracket_probs.items() if floor <= k <= cap)
            else:
                continue
            aggregated[key] = total
        return aggregated

    def _generate_signals(self, bracket_probs, market_prices):
        # type: (Dict[int, float], Dict) -> List[Dict[str, Any]]
        """Generate trade signals by comparing model probs to market prices.

        For each Kalshi bracket:
        - Aggregate 1°F model probs into Kalshi brackets (including tails)
        - Apply min_model_prob filter (reject tail noise)
        - Check YES/NO edge
        - Apply minimum EV filter (blocks penny bets with negative fee-adjusted EV)
        - Compute half-Kelly position size
        """
        kalshi_probs = self._aggregate_to_kalshi_brackets(bracket_probs, market_prices)
        bankroll = self._get_bankroll()

        signals = []  # type: List[Dict[str, Any]]

        for key, model_prob in kalshi_probs.items():
            floor, cap = key
            prices = market_prices[key]

            yes_ask = prices["yes_ask"]
            no_ask = prices["no_ask"]

            yes_edge = self._compute_edge(model_prob, yes_ask)
            no_prob = 1.0 - model_prob
            no_edge = self._compute_edge(no_prob, no_ask)

            if yes_edge >= self.min_edge_pct:
                # Min model probability filter — reject tail noise
                if model_prob < self.min_model_prob:
                    continue
                # EV filter: edge_pct - fee_cents >= min_ev_cents
                price_decimal = yes_ask / 100.0
                fee_cents = max(math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)), 1)
                ev_cents = yes_edge - fee_cents
                if ev_cents < self.min_ev_cents:
                    continue
                contracts = self._compute_contracts(
                    model_prob, yes_ask, bankroll, self.max_per_bracket,
                )
                signals.append({
                    "bracket_floor": floor,
                    "bracket_cap": cap,
                    "direction": "YES",
                    "model_prob": model_prob,
                    "market_price": yes_ask,
                    "edge_pct": yes_edge,
                    "contracts": contracts,
                })
            elif no_edge >= self.min_edge_pct:
                # Min model probability filter — reject tail noise
                if no_prob < self.min_model_prob:
                    continue
                price_decimal = no_ask / 100.0
                fee_cents = max(math.ceil(round(0.07 * price_decimal * (1 - price_decimal) * 100, 10)), 1)
                ev_cents = no_edge - fee_cents
                if ev_cents < self.min_ev_cents:
                    continue
                contracts = self._compute_contracts(
                    no_prob, no_ask, bankroll, self.max_per_bracket,
                )
                signals.append({
                    "bracket_floor": floor,
                    "bracket_cap": cap,
                    "direction": "NO",
                    "model_prob": model_prob,
                    "market_price": no_ask,
                    "edge_pct": no_edge,
                    "contracts": contracts,
                })

        return signals

    async def _check_edge_reversals(self, bracket_probs, market_prices, open_positions):
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
        kalshi_probs = self._aggregate_to_kalshi_brackets(bracket_probs, market_prices)

        for pos in open_positions:
            key = (pos["bracket_floor"], pos["bracket_cap"])
            model_prob = kalshi_probs.get(key)
            prices = market_prices.get(key)

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

            if edge <= 0:
                logger.info(
                    "Edge reversal: pos {} ({},{}) {} edge={:.1f}%",
                    pos["id"], pos["bracket_floor"], pos["bracket_cap"],
                    pos["direction"], edge,
                )
                await self.paper_trader.exit_position(
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
        now = datetime.now(_ET).replace(tzinfo=None)
        con = duckdb.connect(self.db_path)
        try:
            con.execute("""
                INSERT INTO model_state (city, target_date, update_hour, bracket_probs, fcst_high, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (city, target_date) DO UPDATE SET
                    update_hour = EXCLUDED.update_hour,
                    bracket_probs = EXCLUDED.bracket_probs,
                    fcst_high = EXCLUDED.fcst_high,
                    updated_at = EXCLUDED.updated_at
            """, [city, target_date.isoformat(), update_hour, probs_json, fcst_high, now])
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
        """Load trading parameters from paper_config table."""
        try:
            con = duckdb.connect(self.db_path)
            try:
                rows = con.execute(
                    "SELECT key, value FROM paper_config"
                ).fetchall()
                cfg = {k: v for k, v in rows}
                if "min_edge_pct" in cfg:
                    self.min_edge_pct = float(cfg["min_edge_pct"])
                if "min_ev_cents" in cfg:
                    self.min_ev_cents = float(cfg["min_ev_cents"])
                if "min_model_prob" in cfg:
                    self.min_model_prob = float(cfg["min_model_prob"])
                if "starting_capital" in cfg:
                    self.starting_capital = float(cfg["starting_capital"])
                if "max_per_bracket" in cfg:
                    self.max_per_bracket = int(cfg["max_per_bracket"])
            finally:
                con.close()
        except Exception:
            pass  # keep current values

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
            # Get the latest tick per bracket, filtered to the target event date.
            # This prevents mixing tomorrow's uncertain prices with today's model.
            rows = con.execute("""
                WITH latest AS (
                    SELECT floor_strike, cap_strike,
                           yes_bid, yes_ask, no_bid, no_ask,
                           ROW_NUMBER() OVER (
                               PARTITION BY floor_strike, cap_strike
                               ORDER BY captured_at DESC
                           ) AS rn
                    FROM market_ticks
                    WHERE city = 'NYC'
                      AND event_date = ?
                )
                SELECT floor_strike, cap_strike,
                       yes_bid, yes_ask, no_bid, no_ask
                FROM latest
                WHERE rn = 1
            """, [target_date.isoformat()]).fetchall()
        finally:
            con.close()

        prices = {}
        for row in rows:
            floor_strike, cap_strike, yes_bid, yes_ask, no_bid, no_ask = row
            # Skip degenerate (both None) or any missing price
            if (floor_strike is None and cap_strike is None) or any(v is None for v in [yes_bid, yes_ask, no_bid, no_ask]):
                continue
            floor_key = int(floor_strike) if floor_strike is not None else None
            cap_key = int(cap_strike) if cap_strike is not None else None
            prices[(floor_key, cap_key)] = {
                "yes_bid": int(round(yes_bid * 100)),
                "yes_ask": int(round(yes_ask * 100)),
                "no_bid": int(round(no_bid * 100)),
                "no_ask": int(round(no_ask * 100)),
            }
        return prices

    def _get_running_max(self, target_date):
        # type: (date) -> Optional[float]
        """Get the observed running max temperature for today."""
        con = duckdb.connect(self.db_path)
        try:
            row = con.execute("""
                SELECT MAX(temp_f) FROM observations
                WHERE station_id = 'KNYC'
                  AND observed_at::DATE = ?
                  AND temp_f IS NOT NULL
            """, [target_date.isoformat()]).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return None
        finally:
            con.close()

    def _get_current_temp(self, target_date):
        # type: (date) -> Optional[float]
        """Get the most recent observed temperature for today."""
        con = duckdb.connect(self.db_path)
        try:
            row = con.execute("""
                SELECT temp_f FROM observations
                WHERE station_id = 'KNYC'
                  AND observed_at::DATE = ?
                  AND temp_f IS NOT NULL
                ORDER BY observed_at DESC LIMIT 1
            """, [target_date.isoformat()]).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return None
        finally:
            con.close()

    def _get_forecast_upside(self, target_date, update_hour, running_max):
        # type: (date, int, float) -> float
        """Max remaining HRRR forecast temp minus running max.

        Returns 0 if HRRR shows no warming beyond current running max.
        """
        con = duckdb.connect(self.db_path)
        try:
            # Get HRRR forecast temps for hours AFTER current update_hour
            # Convert ET update_hour to UTC for valid_at comparison
            utc_cutoff_hour = update_hour + 5  # EST approximation
            row = con.execute("""
                SELECT MAX(temp_f) FROM forecasts
                WHERE station_id = 'KNYC'
                  AND model_run::DATE = ?
                  AND model_name = 'hrrr'
                  AND temp_f IS NOT NULL
                  AND EXTRACT(HOUR FROM valid_at) > ?
                  AND valid_at::DATE = ?
            """, [target_date.isoformat(), utc_cutoff_hour,
                  target_date.isoformat()]).fetchone()
            if row and row[0] is not None:
                return max(0.0, float(row[0]) - running_max)
            return 0.0
        finally:
            con.close()

    def _get_open_positions(self, target_date):
        # type: (date) -> List[Dict[str, Any]]
        """Get open positions for the target date."""
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT id, bracket_floor, bracket_cap, direction, entry_price
                FROM paper_positions
                WHERE status = 'open'
                  AND city = 'NYC'
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

    def _get_positions_for_dedup(self, target_date):
        # type: (date) -> List[Dict[str, Any]]
        """Get all positions (open and settled) for dedup check.

        Unlike _get_open_positions which only returns open positions (for edge
        reversals), this returns every position for the date so we don't
        re-enter a bracket that already settled.
        """
        con = duckdb.connect(self.db_path)
        try:
            rows = con.execute("""
                SELECT bracket_floor, bracket_cap, direction
                FROM paper_positions
                WHERE city = 'NYC'
                  AND event_date = ?
            """, [target_date.isoformat()]).fetchall()
        finally:
            con.close()

        return [
            {
                "bracket_floor": r[0],
                "bracket_cap": r[1],
                "direction": r[2],
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

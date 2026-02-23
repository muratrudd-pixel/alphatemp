"""Kalshi exchange client with RSA-PSS authentication."""

import base64
import time
from typing import Any, Dict, List, Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from loguru import logger

# Kalshi API base URLs
KALSHI_PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Temperature market series tickers
SERIES_MAP = {
    "NYC": "KXHIGHNY",
    "PHL": "KXHIGHPHL",
    "CHI": "KXHIGHCHI",
    "MIA": "KXHIGHMI",
    "LA": "KXHIGHLA",
}


class KalshiClient:
    """Authenticated Kalshi API client using RSA-PSS signing."""

    def __init__(
        self,
        api_key: str,
        private_key_pem: str,
        demo: bool = False,
    ):
        self.api_key = api_key
        self.base_url = KALSHI_DEMO_URL if demo else KALSHI_PROD_URL
        self._private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )

    def _sign_request(self, method: str, path: str) -> Dict[str, str]:
        """Generate Kalshi auth headers via RSA-PSS signing.

        Signs: timestamp_ms + method + path (path excludes query params).
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = timestamp_ms + method.upper() + path

        signature = self._private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: Optional[dict] = None) -> Dict[str, Any]:
        """Authenticated GET request."""
        headers = self._sign_request("GET", path)
        url = self.base_url + path
        resp = httpx.get(url, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: Optional[dict] = None) -> Dict[str, Any]:
        """Authenticated POST request."""
        headers = self._sign_request("POST", path)
        url = self.base_url + path
        resp = httpx.post(url, headers=headers, json=body or {}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    # --- Market Data ---

    def get_exchange_status(self) -> Dict[str, Any]:
        """Check exchange status and validate authentication."""
        return self._get("/exchange/status")

    def search_markets(
        self, series_ticker: str, status: str = "open", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for markets by series ticker."""
        params = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": limit,
        }
        data = self._get("/markets", params=params)
        return data.get("markets", [])

    def get_orderbook(self, ticker: str, depth: int = 5) -> Dict[str, Any]:
        """Get order book for a specific market ticker."""
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get details for a specific market."""
        return self._get(f"/markets/{ticker}")

    # --- Trading ---

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        return self._get("/portfolio/balance")

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Place an order on Kalshi.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            count: Number of contracts
            type: 'limit' or 'market'
            yes_price: Price in cents (1-99) for yes side
            no_price: Price in cents (1-99) for no side
        """
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price

        logger.info(f"Placing order: {action} {count}x {side} @ {yes_price or no_price}c on {ticker}")
        return self._post("/portfolio/orders", body=body)

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return self._get("/portfolio/positions")

    def get_orders(self, status: str = "resting") -> Dict[str, Any]:
        """Get current orders."""
        return self._get("/portfolio/orders", params={"status": status})


def get_temperature_markets(client: KalshiClient, city: str) -> List[Dict[str, Any]]:
    """Get all open temperature high markets for a city."""
    series = SERIES_MAP.get(city)
    if not series:
        logger.warning(f"No series ticker mapped for city: {city}")
        return []
    return client.search_markets(series_ticker=series, status="open")

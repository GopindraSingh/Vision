"""
engine.py

Core quantitative/trading engine.

Classes
-------
DhanRESTClient
    DhanHQ REST API interface.

DataProcessor
    Fully vectorized indicator and strategy calculations.

OrderExecutionEngine
    Position sizing, circuit protection and bracket-order construction.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_env_file(path: str = "config.env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            value = value.strip()

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            os.environ.setdefault(key, value)


_load_env_file()


def env_str(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name)

    if value is None:
        if default is None:
            raise ValueError(f"Missing configuration: {name}")
        return default

    return value.strip()


def env_float(name: str, default: float = 0.0) -> float:
    return float(env_str(name, str(default)))


def env_int(name: str, default: int = 0) -> int:
    return int(env_str(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name, str(default)).lower()
    return value in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DhanAPIError(RuntimeError):
    pass


class DhanRateLimitError(DhanAPIError):
    pass


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """
    Thread-safe rate limiter.

    Dhan documents 5 requests/sec for Data APIs.
    The limiter therefore defaults to 5 requests/sec.

    It deliberately throttles rather than attempting to overwhelm the
    endpoint with 20 simultaneous requests.
    """

    def __init__(self, rate_per_second: float = 5.0):
        self.rate = float(rate_per_second)
        self.minimum_interval = 1.0 / self.rate
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()

            wait = self._next_allowed - now

            if wait > 0:
                time.sleep(wait)

            current = time.monotonic()
            self._next_allowed = max(
                current,
                self._next_allowed,
            ) + self.minimum_interval


# ---------------------------------------------------------------------------
# Dhan REST client
# ---------------------------------------------------------------------------

class DhanRESTClient:
    """
    Production-oriented DhanHQ REST client.

    Environment:
        SANDBOX and PRODUCTION are accepted for configuration compatibility.

    Dhan's current public REST endpoint is api.dhan.co/v2.
    """

    BASE_URL = "https://api.dhan.co/v2"
    SECURITY_MASTER_URL = (
        "https://images.dhan.co/api-data/api-scrip-master.csv"
    )

    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

    def __init__(
        self,
        access_token: Optional[str] = None,
        client_id: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.client_id = client_id or env_str("DHAN_CLIENT_ID")
        self.access_token = access_token
        self.timeout = timeout

        self.session = requests.Session()

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=self.RETRY_STATUS_CODES,
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=25,
            pool_maxsize=25,
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "DhanMomentumEngine/1.0",
            }
        )

        self.data_limiter = TokenBucketRateLimiter(5.0)
        self.quote_limiter = TokenBucketRateLimiter(1.0)
        self.order_limiter = TokenBucketRateLimiter(10.0)

    def set_access_token(
        self,
        access_token: str,
        client_id: Optional[str] = None,
    ) -> None:
        self.access_token = access_token

        if client_id:
            self.client_id = client_id

    def _headers(self) -> Dict[str, str]:
        if not self.access_token:
            raise DhanAPIError("Dhan access token has not been initialized.")

        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        limiter: Optional[TokenBucketRateLimiter] = None,
    ) -> Any:

        if limiter:
            limiter.acquire()

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise DhanAPIError(
                f"Network error calling {endpoint}: {exc}"
            ) from exc

        if response.status_code == 429:
            raise DhanRateLimitError(
                f"Dhan rate limit reached: {response.text[:500]}"
            )

        if response.status_code >= 400:
            raise DhanAPIError(
                f"Dhan HTTP {response.status_code} "
                f"for {endpoint}: {response.text[:1000]}"
            )

        if not response.content:
            return {}

        try:
            return response.json()
        except ValueError as exc:
            raise DhanAPIError(
                f"Invalid JSON from {endpoint}: {response.text[:500]}"
            ) from exc

    # ------------------------------------------------------------------
    # Security master
    # ------------------------------------------------------------------

    def download_security_master(
        self,
        destination: str = "security_master.csv",
    ) -> pd.DataFrame:

        LOGGER.info("Downloading Dhan security master.")

        try:
            response = self.session.get(
                self.SECURITY_MASTER_URL,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DhanAPIError(
                f"Unable to download security master: {exc}"
            ) from exc

        with open(destination, "wb") as file:
            file.write(response.content)

        df = pd.read_csv(destination, low_memory=False)

        LOGGER.info(
            "Security master downloaded: %d rows.",
            len(df),
        )

        return df

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def historical_intraday(
        self,
        security_id: int | str,
        from_date: str,
        to_date: str,
        interval: str = "5",
        exchange_segment: str = "NSE_EQ",
        instrument: str = "EQUITY",
    ) -> pd.DataFrame:

        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": str(interval),
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date,
        }

        response = self._request(
            "POST",
            "/charts/intraday",
            payload,
            limiter=self.data_limiter,
        )

        return self._historical_response_to_dataframe(response)

    @staticmethod
    def _historical_response_to_dataframe(
        response: Dict[str, Any],
    ) -> pd.DataFrame:

        if not response:
            return pd.DataFrame()

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
        ]

        if not all(key in response for key in required):
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "open": response["open"],
                "high": response["high"],
                "low": response["low"],
                "close": response["close"],
                "volume": response["volume"],
                "timestamp": response["timestamp"],
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="s",
            utc=True,
        ).dt.tz_convert("Asia/Kolkata")

        df = df.sort_values("timestamp").reset_index(drop=True)

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df.dropna(
            subset=["open", "high", "low", "close", "volume"]
        )

    # ------------------------------------------------------------------
    # Market quote
    # ------------------------------------------------------------------

    def market_quote(
        self,
        security_ids: Sequence[int | str],
        exchange_segment: str = "NSE_EQ",
    ) -> Dict[str, Any]:

        ids = [int(x) for x in security_ids]

        if not ids:
            return {}

        if len(ids) > 1000:
            raise ValueError(
                "Dhan market quote request supports at most 1000 instruments."
            )

        payload = {
            exchange_segment: ids,
        }

        return self._request(
            "POST",
            "/marketfeed/quote",
            payload,
            limiter=self.quote_limiter,
        )

    def market_ltp(
        self,
        security_ids: Sequence[int | str],
        exchange_segment: str = "NSE_EQ",
    ) -> Dict[str, Any]:

        ids = [int(x) for x in security_ids]

        if not ids:
            return {}

        if len(ids) > 1000:
            raise ValueError(
                "Dhan LTP request supports at most 1000 instruments."
            )

        return self._request(
            "POST",
            "/marketfeed/ltp",
            {exchange_segment: ids},
            limiter=self.quote_limiter,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:

        return self._request(
            "POST",
            "/orders",
            payload,
            limiter=self.order_limiter,
        )


# ---------------------------------------------------------------------------
# Data processor
# ---------------------------------------------------------------------------

class DataProcessor:
    """
    Vectorized technical-analysis processor.
    """

    EMA_FAST = 9
    EMA_SLOW = 21

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA(9), EMA(21), session VWAP, Cross_Up and Cross_Down.

        The crossover is defined strictly from the immediately preceding
        closed bar to the current closed bar.
        """

        if df.empty:
            return df.copy()

        required = {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {sorted(missing)}"
            )

        result = df.copy()

        result = result.sort_values(
            "timestamp" if "timestamp" in result.columns else result.index
        ).reset_index(drop=True)

        result["EMA_9"] = (
            result["close"]
            .ewm(
                span=DataProcessor.EMA_FAST,
                adjust=False,
                min_periods=DataProcessor.EMA_FAST,
            )
            .mean()
        )

        result["EMA_21"] = (
            result["close"]
            .ewm(
                span=DataProcessor.EMA_SLOW,
                adjust=False,
                min_periods=DataProcessor.EMA_SLOW,
            )
            .mean()
        )

        typical_price = (
            result["high"]
            + result["low"]
            + result["close"]
        ) / 3.0

        cumulative_pv = (
            typical_price * result["volume"]
        ).cumsum()

        cumulative_volume = result["volume"].cumsum()

        result["VWAP"] = np.divide(
            cumulative_pv,
            cumulative_volume,
            out=np.full(len(result), np.nan, dtype=float),
            where=cumulative_volume.to_numpy() != 0,
        )

        previous_fast = result["EMA_9"].shift(1)
        previous_slow = result["EMA_21"].shift(1)

        result["Cross_Up"] = (
            (result["EMA_9"] > result["EMA_21"])
            & (previous_fast <= previous_slow)
        ).fillna(False)

        result["Cross_Down"] = (
            (result["EMA_9"] < result["EMA_21"])
            & (previous_fast >= previous_slow)
        ).fillna(False)

        return result

    @staticmethod
    def _time_bucket(
        timestamp_series: pd.Series,
    ) -> pd.Series:

        ts = pd.to_datetime(
            timestamp_series,
            errors="coerce",
        )

        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("Asia/Kolkata")
        else:
            ts = ts.dt.tz_convert("Asia/Kolkata")

        return ts.dt.strftime("%H:%M")

    @staticmethod
    def calculate_rvol(
        history: pd.DataFrame,
        target_timestamp: str | pd.Timestamp,
        lookback_sessions: int = 10,
    ) -> float:

        if history.empty:
            return float("nan")

        df = history.copy()

        if "timestamp" not in df.columns:
            raise ValueError("history must contain timestamp.")

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if getattr(df["timestamp"].dt, "tz", None) is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(
                "Asia/Kolkata"
            )
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(
                "Asia/Kolkata"
            )

        target = pd.Timestamp(target_timestamp)

        if target.tzinfo is None:
            target = target.tz_localize("Asia/Kolkata")
        else:
            target = target.tz_convert("Asia/Kolkata")

        target_time = target.strftime("%H:%M")

        df["session_date"] = df["timestamp"].dt.date
        df["clock_time"] = df["timestamp"].dt.strftime("%H:%M")

        target_date = target.date()

        target_rows = df[
            (df["session_date"] == target_date)
            & (df["clock_time"] == target_time)
        ]

        if target_rows.empty:
            return float("nan")

        current_volume = float(target_rows.iloc[-1]["volume"])

        sessions = sorted(
            date
            for date in df["session_date"].dropna().unique()
            if date < target_date
        )

        sessions = sessions[-lookback_sessions:]

        if len(sessions) < lookback_sessions:
            return float("nan")

        prior = df[
            df["session_date"].isin(sessions)
            & (df["clock_time"] == target_time)
        ]

        if prior.empty:
            return float("nan")

        mean_volume = float(prior["volume"].mean())

        if mean_volume <= 0:
            return float("nan")

        return current_volume / mean_volume

    @staticmethod
    def extract_target_bar(
        df: pd.DataFrame,
        target_time: str = "09:40",
    ) -> Optional[pd.Series]:

        if df.empty:
            return None

        working = df.copy()

        if "timestamp" not in working.columns:
            return None

        timestamps = pd.to_datetime(
            working["timestamp"],
            errors="coerce",
        )

        if getattr(timestamps.dt, "tz", None) is None:
            timestamps = timestamps.dt.tz_localize(
                "Asia/Kolkata"
            )
        else:
            timestamps = timestamps.dt.tz_convert(
                "Asia/Kolkata"
            )

        mask = timestamps.dt.strftime("%H:%M") == target_time

        candidates = working.loc[mask]

        if candidates.empty:
            return None

        return candidates.iloc[-1]

    @staticmethod
    def calculate_velocity(
        close: float,
        previous_close: float,
    ) -> float:

        if previous_close <= 0:
            return float("nan")

        return ((close / previous_close) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeCandidate:
    security_id: int
    symbol: str
    side: str
    entry_price: float
    previous_day_close: float
    vwap: float
    rvol: float
    velocity: float
    upper_circuit: float
    lower_circuit: float
    tick_size: float


class OrderExecutionEngine:
    """
    Position sizing, circuit protection and Dhan bracket order generation.
    """

    PROFIT_TARGET_PCT = 0.0070
    STOPLOSS_PCT = 0.0035

    def __init__(
        self,
        dhan_client: DhanRESTClient,
    ):
        self.client = dhan_client

        self.total_capital = env_float(
            "TOTAL_CAPITAL",
            2000.0,
        )

        self.max_active_trades = env_int(
            "MAX_ACTIVE_TRADES",
            2,
        )

        self.leverage = env_float(
            "LEVERAGE",
            5.0,
        )

        self.dry_run = env_bool(
            "DRY_RUN",
            False,
        )

    @property
    def allocated_capital(self) -> float:
        """
        Capital allocated per trade.

        MAX_ACTIVE_TRADES=2 therefore divides TOTAL_CAPITAL equally.
        """
        return self.total_capital / max(
            1,
            self.max_active_trades,
        )

    def calculate_quantity(
        self,
        entry_price: float,
    ) -> int:

        if entry_price <= 0:
            LOGGER.warning(
                "Invalid entry price %.4f. Skipping stock.",
                entry_price,
            )
            return 0

        quantity = int(
            (
                self.allocated_capital
                * self.leverage
            )
            // entry_price
        )

        if quantity <= 0:
            LOGGER.info(
                "Quantity resolved to zero at entry price %.4f. "
                "Skipping stock.",
                entry_price,
            )
            return 0

        return quantity

    @staticmethod
    def circuit_guard(
        entry_price: float,
        upper_circuit: float,
        lower_circuit: float,
    ) -> bool:
        """
        Return True when execution must be blocked.

        A trade is blocked when entry is within 1% of either circuit.
        """

        if entry_price <= 0:
            return True

        if upper_circuit > 0:
            distance_upper = (
                upper_circuit - entry_price
            ) / entry_price

            if distance_upper <= 0.01:
                return True

        if lower_circuit > 0:
            distance_lower = (
                entry_price - lower_circuit
            ) / entry_price

            if distance_lower <= 0.01:
                return True

        return False

    @staticmethod
    def _ceil_to_tick(
        value: float,
        tick_size: float,
    ) -> float:

        if tick_size <= 0:
            raise ValueError("tick_size must be positive.")

        ticks = math.ceil(
            (value / tick_size) - 1e-12
        )

        normalized = ticks * tick_size

        decimals = max(
            0,
            int(
                math.ceil(
                    -math.log10(tick_size)
                )
            )
            if tick_size < 1
            else 0,
        )

        return round(normalized, decimals + 2)

    def build_bracket_payload(
        self,
        *,
        security_id: int,
        transaction_type: str,
        quantity: int,
        entry_price: float,
        tick_size: float,
        trading_symbol: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        if transaction_type not in {"BUY", "SELL"}:
            raise ValueError(
                "transaction_type must be BUY or SELL."
            )

        if quantity <= 0:
            raise ValueError("quantity must be positive.")

        if entry_price <= 0:
            raise ValueError("entry_price must be positive.")

        raw_target_offset = (
            entry_price * self.PROFIT_TARGET_PCT
        )

        raw_stop_offset = (
            entry_price * self.STOPLOSS_PCT
        )

        target_offset = self._ceil_to_tick(
            raw_target_offset,
            tick_size,
        )

        stop_offset = self._ceil_to_tick(
            raw_stop_offset,
            tick_size,
        )

        payload: Dict[str, Any] = {
            "dhanClientId": self.client.client_id,
            "correlationId": correlation_id or "",
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_EQ",
            "productType": "BO",
            "orderType": "LIMIT",
            "validity": "DAY",
            "securityId": str(security_id),
            "quantity": int(quantity),
            "disclosedQuantity": 0,
            "price": float(entry_price),
            "triggerPrice": 0,
            "afterMarketOrder": False,
            "amoTime": "",
            "boProfitValue": float(target_offset),
            "boStopLossValue": float(stop_offset),
        }

        if trading_symbol:
            payload["tradingSymbol"] = trading_symbol

        return payload

    def execute_candidate(
        self,
        candidate: TradeCandidate,
    ) -> Optional[Dict[str, Any]]:

        if self.circuit_guard(
            candidate.entry_price,
            candidate.upper_circuit,
            candidate.lower_circuit,
        ):
            LOGGER.warning(
                "%s blocked by circuit guard. "
                "entry=%.4f upper=%.4f lower=%.4f",
                candidate.symbol,
                candidate.entry_price,
                candidate.upper_circuit,
                candidate.lower_circuit,
            )
            return None

        quantity = self.calculate_quantity(
            candidate.entry_price
        )

        if quantity == 0:
            return None

        transaction_type = (
            "BUY"
            if candidate.side == "LONG"
            else "SELL"
        )

        payload = self.build_bracket_payload(
            security_id=candidate.security_id,
            transaction_type=transaction_type,
            quantity=quantity,
            entry_price=candidate.entry_price,
            tick_size=candidate.tick_size,
            trading_symbol=candidate.symbol,
            correlation_id=(
                f"MOM-{candidate.side}-"
                f"{candidate.security_id}-"
                f"{int(time.time() * 1000)}"
            ),
        )

        if self.dry_run:
            LOGGER.warning(
                "DRY_RUN=true | Order NOT submitted | "
                "%s | %s | qty=%d | entry=%.4f | "
                "target_offset=%.4f | stop_offset=%.4f",
                candidate.symbol,
                candidate.side,
                quantity,
                candidate.entry_price,
                payload["boProfitValue"],
                payload["boStopLossValue"],
            )
            return {
                "dryRun": True,
                "payload": payload,
            }

        try:
            response = self.client.submit_order(
                payload
            )

            LOGGER.info(
                "ORDER SUBMITTED | symbol=%s side=%s qty=%d "
                "entry=%.4f response=%s",
                candidate.symbol,
                candidate.side,
                quantity,
                candidate.entry_price,
                response,
            )

            return response

        except Exception:
            LOGGER.exception(
                "Order submission failed for %s.",
                candidate.symbol,
            )
            return None

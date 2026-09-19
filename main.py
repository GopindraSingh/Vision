"""
main.py

Autonomous intraday momentum orchestration.

Workflow
--------
06:00+
    Authentication
    Security-master download
    Universe preparation
    Silent wait

09:46 IST
    Historical 5-minute scan
    Indicator calculation
    RVOL calculation
    Candidate filtering
    Ranking
    Circuit protection
    Bracket-order submission

Important:
    Dhan's documented historical-data rate limit is 5 requests/second.
    ThreadPoolExecutor(max_workers=20) is therefore used for concurrent
    processing while the REST client rate limiter protects the API.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from dhan_auth import (
    DhanAuthenticationError,
    get_daily_access_token,
)
from engine import (
    DataProcessor,
    DhanAPIError,
    DhanRESTClient,
    OrderExecutionEngine,
    TradeCandidate,
    env_bool,
    env_float,
    env_int,
    env_str,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d IST | "
    "%(levelname)s | %(threadName)s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger("DhanMomentum")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IST = "Asia/Kolkata"

MARKET_OPEN = dt_time(9, 15)
TARGET_BAR_START = dt_time(9, 40)
TARGET_BAR_END = dt_time(9, 45)
TRIGGER_TIME = dt_time(9, 46)

MAX_WORKERS = 20

SECURITY_MASTER_FILE = "security_master.csv"


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

STOP_EVENT = threading.Event()


def _signal_handler(signum, frame):
    LOGGER.warning(
        "Shutdown signal received (%s).",
        signum,
    )
    STOP_EVENT.set()


signal.signal(
    signal.SIGINT,
    _signal_handler,
)

signal.signal(
    signal.SIGTERM,
    _signal_handler,
)


# ---------------------------------------------------------------------------
# Universe model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Security:
    security_id: int
    symbol: str
    exchange_segment: str
    tick_size: float
    upper_circuit: float
    lower_circuit: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_ist() -> datetime:
    return pd.Timestamp.now(
        tz=IST
    ).to_pydatetime()


def today_ist() -> date:
    return now_ist().date()


def market_day(day: date) -> bool:
    return day.weekday() < 5


def previous_weekdays(
    target: date,
    count: int,
) -> List[date]:

    days: List[date] = []

    cursor = target - timedelta(days=1)

    while len(days) < count:
        if market_day(cursor):
            days.append(cursor)

        cursor -= timedelta(days=1)

    return list(reversed(days))


def sleep_until(
    target_date: date,
    target_time: dt_time,
) -> None:

    target = pd.Timestamp(
        datetime.combine(
            target_date,
            target_time,
        ),
        tz=IST,
    )

    while not STOP_EVENT.is_set():

        current = pd.Timestamp.now(tz=IST)

        remaining = (
            target - current
        ).total_seconds()

        if remaining <= 0:
            return

        # Deep sleep for the majority of the wait.
        # Wake periodically to permit clean shutdown.
        sleep_seconds = min(
            max(remaining, 0.1),
            60.0,
        )

        STOP_EVENT.wait(
            timeout=sleep_seconds
        )


def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Dhan/NSE CSV column names.
    """

    result = df.copy()

    result.columns = [
        str(column).strip()
        for column in result.columns
    ]

    return result


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:

    normalized = {
        str(column).strip().upper(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.upper()

        if key in normalized:
            return normalized[key]

    return None


def parse_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# Index constituent download
# ---------------------------------------------------------------------------

class IndexUniverseProvider:
    """
    Retrieves current Nifty 500 and Nifty 50 constituent lists.

    Nifty Indices publishes constituent CSV files. The URLs can change;
    therefore this class supports several documented/current URL patterns.
    """

    NIFTY_500_URLS = (
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_nifty500list.csv",
    )

    NIFTY_50_URLS = (
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_nifty50list.csv",
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_nifty50list.csv",
    )

    def __init__(
        self,
        timeout: float = 20.0,
    ):
        import requests

        self.requests = requests
        self.timeout = timeout

    def _download_csv(
        self,
        urls: Sequence[str],
    ) -> pd.DataFrame:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(X11; Linux aarch64) "
                "AppleWebKit/537.36 "
                "Chrome/130 Safari/537.36"
            ),
            "Accept": (
                "text/csv,text/plain,"
                "application/octet-stream,*/*"
            ),
            "Referer": "https://www.niftyindices.com/",
        }

        last_error: Optional[Exception] = None

        for url in urls:
            try:
                response = self.requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                return pd.read_csv(
                    pd.io.common.BytesIO(
                        response.content
                    )
                )

            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Unable to download index constituents: {last_error}"
        )

    @staticmethod
    def _extract_symbols(
        df: pd.DataFrame,
    ) -> Set[str]:

        symbol_column = find_column(
            df,
            [
                "Symbol",
                "SYMBOL",
                "Ticker",
                "Ticker Symbol",
            ],
        )

        if not symbol_column:
            raise RuntimeError(
                "Could not identify Symbol column in index CSV."
            )

        symbols = (
            df[symbol_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        return {
            symbol
            for symbol in symbols
            if symbol
            and symbol != "NAN"
        }

    def get_nifty500_minus_nifty50(
        self,
    ) -> Set[str]:

        nifty500 = self._download_csv(
            self.NIFTY_500_URLS
        )

        nifty50 = self._download_csv(
            self.NIFTY_50_URLS
        )

        n500 = self._extract_symbols(
            nifty500
        )

        n50 = self._extract_symbols(
            nifty50
        )

        universe = n500 - n50

        LOGGER.info(
            "Nifty 500 constituents=%d | "
            "Nifty 50 excluded=%d | "
            "eligible symbols before price filter=%d",
            len(n500),
            len(n50),
            len(universe),
        )

        return universe


# ---------------------------------------------------------------------------
# Security-master processing
# ---------------------------------------------------------------------------

class UniverseBuilder:

    def __init__(
        self,
        rest_client: DhanRESTClient,
    ):
        self.client = rest_client

    def download_master(
        self,
    ) -> pd.DataFrame:

        return self.client.download_security_master(
            SECURITY_MASTER_FILE
        )

    def build(
        self,
        master: pd.DataFrame,
        allowed_symbols: Set[str],
    ) -> List[Security]:

        df = normalize_columns(master)

        exchange_col = find_column(
            df,
            [
                "SEM_EXM_EXCH_ID",
                "EXCH_ID",
            ],
        )

        segment_col = find_column(
            df,
            [
                "SEM_SEGMENT",
                "SEGMENT",
            ],
        )

        security_id_col = find_column(
            df,
            [
                "SEM_SMST_SECURITY_ID",
                "SECURITY_ID",
                "SECURITYID",
            ],
        )

        symbol_col = find_column(
            df,
            [
                "SEM_CUSTOM_SYMBOL",
                "CUSTOM_SYMBOL",
                "SYMBOL_NAME",
                "SYMBOL",
            ],
        )

        tick_col = find_column(
            df,
            [
                "SEM_TICK_SIZE",
                "TICK_SIZE",
            ],
        )

        upper_col = find_column(
            df,
            [
                "SEM_UPPER_CKT_LIMIT",
                "UPPER_CKT_LIMIT",
                "UPPER_CIRCUIT",
            ],
        )

        lower_col = find_column(
            df,
            [
                "SEM_LOWER_CKT_LIMIT",
                "LOWER_CKT_LIMIT",
                "LOWER_CIRCUIT",
            ],
        )

        instrument_col = find_column(
            df,
            [
                "SEM_INSTRUMENT_NAME",
                "INSTRUMENT",
                "INSTRUMENT_NAME",
            ],
        )

        required = {
            "exchange": exchange_col,
            "segment": segment_col,
            "security_id": security_id_col,
            "symbol": symbol_col,
            "tick": tick_col,
        }

        missing = [
            key
            for key, value in required.items()
            if value is None
        ]

        if missing:
            raise RuntimeError(
                "Security master missing columns: "
                + ", ".join(missing)
            )

        working = df.copy()

        working[security_id_col] = parse_numeric(
            working[security_id_col]
        )

        working[tick_col] = parse_numeric(
            working[tick_col]
        )

        if upper_col:
            working[upper_col] = parse_numeric(
                working[upper_col]
            )
        else:
            working["_upper"] = np.nan
            upper_col = "_upper"

        if lower_col:
            working[lower_col] = parse_numeric(
                working[lower_col]
            )
        else:
            working["_lower"] = np.nan
            lower_col = "_lower"

        working["_symbol_normalized"] = (
            working[symbol_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # NSE equity only.
        mask = (
            working[exchange_col]
            .astype(str)
            .str.upper()
            .eq("NSE")
        )

        # Avoid derivative instruments.
        if instrument_col:
            instrument_text = (
                working[instrument_col]
                .astype(str)
                .str.upper()
            )

            mask &= (
                instrument_text.str.contains(
                    "EQUITY",
                    na=False,
                )
                | instrument_text.eq("EQUITY")
            )

        mask &= working["_symbol_normalized"].isin(
            allowed_symbols
        )

        filtered = working.loc[
            mask
        ].copy()

        securities: List[Security] = []

        for _, row in filtered.iterrows():

            try:
                security_id = int(
                    row[security_id_col]
                )

                symbol = str(
                    row[symbol_col]
                ).strip().upper()

                tick_size = float(
                    row[tick_col]
                )

                upper = float(
                    row[upper_col]
                ) if pd.notna(
                    row[upper_col]
                ) else 0.0

                lower = float(
                    row[lower_col]
                ) if pd.notna(
                    row[lower_col]
                ) else 0.0

                if not symbol:
                    continue

                if tick_size <= 0:
                    continue

                securities.append(
                    Security(
                        security_id=security_id,
                        symbol=symbol,
                        exchange_segment="NSE_EQ",
                        tick_size=tick_size,
                        upper_circuit=upper,
                        lower_circuit=lower,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        # Deduplicate by security ID.
        deduped: Dict[int, Security] = {
            item.security_id: item
            for item in securities
        }

        result = list(
            deduped.values()
        )

        LOGGER.info(
            "Universe mapped to %d NSE equity securities.",
            len(result),
        )

        return result


# ---------------------------------------------------------------------------
# Quote processing
# ---------------------------------------------------------------------------

def flatten_ltp_response(
    response: Dict[str, Any],
) -> Dict[int, float]:

    result: Dict[int, float] = {}

    if not response:
        return result

    data = response.get("data", response)

    if not isinstance(data, dict):
        return result

    for segment_payload in data.values():

        if not isinstance(
            segment_payload,
            dict,
        ):
            continue

        for security_id, payload in segment_payload.items():

            try:
                sid = int(security_id)

                if isinstance(payload, dict):
                    ltp = (
                        payload.get("last_price")
                        or payload.get("ltp")
                        or payload.get("LTP")
                    )
                else:
                    ltp = payload

                if ltp is not None:
                    result[sid] = float(ltp)

            except (
                ValueError,
                TypeError,
            ):
                continue

    return result


def flatten_quote_response(
    response: Dict[str, Any],
) -> Dict[int, Dict[str, float]]:

    result: Dict[int, Dict[str, float]] = {}

    if not response:
        return result

    data = response.get("data", response)

    if not isinstance(data, dict):
        return result

    for segment_payload in data.values():

        if not isinstance(
            segment_payload,
            dict,
        ):
            continue

        for security_id, payload in segment_payload.items():

            try:
                sid = int(security_id)
            except (
                ValueError,
                TypeError,
            ):
                continue

            if not isinstance(
                payload,
                dict,
            ):
                continue

            row: Dict[str, float] = {}

            for source, target in (
                ("last_price", "last_price"),
                ("ltp", "last_price"),
                ("prev_close", "prev_close"),
                ("previous_close", "prev_close"),
                ("upper_circuit", "upper_circuit"),
                ("lower_circuit", "lower_circuit"),
            ):

                value = payload.get(source)

                if value is not None:
                    try:
                        row[target] = float(value)
                    except (
                        ValueError,
                        TypeError,
                    ):
                        pass

            result[sid] = row

    return result


# ---------------------------------------------------------------------------
# Historical scan
# ---------------------------------------------------------------------------

class HistoricalScanner:

    def __init__(
        self,
        rest_client: DhanRESTClient,
        processor: DataProcessor,
    ):
        self.client = rest_client
        self.processor = processor

    def _history_window(
        self,
        trigger_date: date,
    ) -> tuple[str, str]:

        sessions = previous_weekdays(
            trigger_date,
            11,
        )

        start_date = sessions[0]

        # Dhan toDate is non-inclusive according to current documentation.
        end_date = trigger_date + timedelta(days=1)

        return (
            f"{start_date.isoformat()} 09:15:00",
            f"{end_date.isoformat()} 09:46:00",
        )

    def scan_one(
        self,
        security: Security,
        trigger_date: date,
    ) -> Optional[Dict[str, Any]]:

        from_date, to_date = self._history_window(
            trigger_date
        )

        try:
            history = self.client.historical_intraday(
                security_id=security.security_id,
                from_date=from_date,
                to_date=to_date,
                interval="5",
                exchange_segment=security.exchange_segment,
                instrument="EQUITY",
            )

            if history.empty:
                return None

            history = self.processor.add_indicators(
                history
            )

            target = self.processor.extract_target_bar(
                history,
                target_time="09:40",
            )

            if target is None:
                return None

            timestamp = target["timestamp"]

            rvol = self.processor.calculate_rvol(
                history,
                target_timestamp=timestamp,
                lookback_sessions=10,
            )

            if not np.isfinite(rvol):
                return None

            close = float(target["close"])
            vwap = float(target["VWAP"])
            cross_up = bool(target["Cross_Up"])
            cross_down = bool(target["Cross_Down"])

            # Previous day close from the final regular-market candle
            # belonging to the immediately preceding session.
            target_ts = pd.Timestamp(timestamp)

            previous_date = (
                target_ts.date()
                - timedelta(days=1)
            )

            prior_dates = sorted(
                {
                    pd.Timestamp(ts).date()
                    for ts in history["timestamp"]
                    if pd.Timestamp(ts).date()
                    < target_ts.date()
                }
            )

            if not prior_dates:
                return None

            previous_session = prior_dates[-1]

            previous_day = history[
                history["timestamp"].apply(
                    lambda x:
                    pd.Timestamp(x).date()
                    == previous_session
                )
            ]

            if previous_day.empty:
                return None

            previous_close = float(
                previous_day.iloc[-1]["close"]
            )

            velocity = (
                DataProcessor.calculate_velocity(
                    close,
                    previous_close,
                )
            )

            long_candidate = (
                close > vwap
                and close > previous_close
                and rvol > 2.0
                and cross_up
            )

            short_candidate = (
                close < vwap
                and close < previous_close
                and rvol > 2.0
                and cross_down
            )

            if not (
                long_candidate
                or short_candidate
            ):
                return None

            side = (
                "LONG"
                if long_candidate
                else "SHORT"
            )

            return {
                "security": security,
                "side": side,
                "entry_price": close,
                "previous_day_close": previous_close,
                "vwap": vwap,
                "rvol": rvol,
                "velocity": velocity,
                "timestamp": timestamp,
            }

        except Exception:
            LOGGER.exception(
                "Scan failed for %s (%s).",
                security.symbol,
                security.security_id,
            )
            return None


# ---------------------------------------------------------------------------
# Main strategy engine
# ---------------------------------------------------------------------------

class MomentumTradingSystem:

    def __init__(
        self,
        dhan_pin: str,
    ):
        self.dhan_pin = dhan_pin

        self.client = DhanRESTClient()

        self.processor = DataProcessor()

        self.execution = OrderExecutionEngine(
            self.client
        )

        self.universe_builder = UniverseBuilder(
            self.client
        )

        self.index_provider = IndexUniverseProvider()

        self.scanner = HistoricalScanner(
            self.client,
            self.processor,
        )

        self.securities: List[Security] = []

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> None:

        LOGGER.info(
            "Starting Dhan authentication."
        )

        token = get_daily_access_token(
            dhan_pin=self.dhan_pin
        )

        self.client.set_access_token(
            access_token=token.access_token,
            client_id=token.client_id,
        )

        LOGGER.info(
            "Dhan REST layer initialized."
        )

    # ------------------------------------------------------------------
    # Pre-market universe
    # ------------------------------------------------------------------

    def prepare_universe(self) -> None:

        LOGGER.info(
            "Preparing pre-market universe."
        )

        nifty_exclusions = (
            self.index_provider
            .get_nifty500_minus_nifty50()
        )

        master = (
            self.universe_builder
            .download_master()
        )

        self.securities = (
            self.universe_builder
            .build(
                master,
                nifty_exclusions,
            )
        )

        if not self.securities:
            raise RuntimeError(
                "Universe construction returned zero securities."
            )

        # Price filter is deliberately performed through a single/batched
        # quote request rather than one request per security.
        LOGGER.info(
            "Fetching batched pre-market LTP data."
        )

        ids = [
            security.security_id
            for security in self.securities
        ]

        eligible: List[Security] = []

        # Dhan permits up to 1000 instruments in one quote request.
        for start in range(
            0,
            len(ids),
            1000,
        ):

            chunk = ids[
                start:start + 1000
            ]

            try:
                quote_response = (
                    self.client.market_ltp(
                        chunk
                    )
                )

                prices = flatten_ltp_response(
                    quote_response
                )

                for security in self.securities:

                    if (
                        security.security_id
                        not in prices
                    ):
                        continue

                    price = prices[
                        security.security_id
                    ]

                    if (
                        price > 0
                        and price <= 1000
                    ):
                        eligible.append(
                            security
                        )

            except Exception:
                LOGGER.exception(
                    "Unable to obtain batched prices."
                )

        # Deduplicate.
        self.securities = list(
            {
                security.security_id: security
                for security in eligible
            }.values()
        )

        LOGGER.info(
            "Final sub-₹1000 universe: %d securities.",
            len(self.securities),
        )

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    @staticmethod
    def rank_candidates(
        raw_candidates: List[Dict[str, Any]],
    ) -> tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:

        longs = [
            candidate
            for candidate in raw_candidates
            if candidate["side"] == "LONG"
        ]

        shorts = [
            candidate
            for candidate in raw_candidates
            if candidate["side"] == "SHORT"
        ]

        # Highest gain first.
        longs.sort(
            key=lambda x: x["velocity"],
            reverse=True,
        )

        # Lowest gain first.
        shorts.sort(
            key=lambda x: x["velocity"],
        )

        return longs, shorts

    # ------------------------------------------------------------------
    # Trigger scan
    # ------------------------------------------------------------------

    def scan_universe(
        self,
        trigger_date: date,
    ) -> List[Dict[str, Any]]:

        LOGGER.info(
            "09:46 trigger fired. Scanning %d securities.",
            len(self.securities),
        )

        candidates: List[Dict[str, Any]] = []

        start = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix="DhanScan",
        ) as executor:

            futures = {
                executor.submit(
                    self.scanner.scan_one,
                    security,
                    trigger_date,
                ): security
                for security in self.securities
            }

            for future in as_completed(futures):

                security = futures[future]

                try:
                    result = future.result()

                    if result is not None:
                        candidates.append(
                            result
                        )

                except Exception:
                    LOGGER.exception(
                        "Worker failure for %s.",
                        security.symbol,
                    )

        elapsed = (
            time.perf_counter()
            - start
        )

        LOGGER.info(
            "Universe scan completed in %.3f seconds. "
            "Qualifying candidates=%d.",
            elapsed,
            len(candidates),
        )

        return candidates

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        candidates: List[Dict[str, Any]],
    ) -> None:

        if not candidates:
            LOGGER.info(
                "No qualifying LONG or SHORT candidates."
            )
            return

        longs, shorts = self.rank_candidates(
            candidates
        )

        selected: List[Dict[str, Any]] = []

        if longs:
            selected.append(
                longs[0]
            )

        if shorts:
            selected.append(
                shorts[0]
            )

        maximum = self.execution.max_active_trades

        selected = selected[:maximum]

        LOGGER.info(
            "Selected %d candidate(s) for routing.",
            len(selected),
        )

        for item in selected:

            security: Security = item["security"]

            candidate = TradeCandidate(
                security_id=security.security_id,
                symbol=security.symbol,
                side=item["side"],
                entry_price=float(
                    item["entry_price"]
                ),
                previous_day_close=float(
                    item["previous_day_close"]
                ),
                vwap=float(
                    item["vwap"]
                ),
                rvol=float(
                    item["rvol"]
                ),
                velocity=float(
                    item["velocity"]
                ),
                upper_circuit=float(
                    security.upper_circuit
                ),
                lower_circuit=float(
                    security.lower_circuit
                ),
                tick_size=float(
                    security.tick_size
                ),
            )

            LOGGER.info(
                "ROUTING | %s | %s | "
                "entry=%.4f | velocity=%.4f%% | "
                "RVOL=%.2f | VWAP=%.4f",
                candidate.symbol,
                candidate.side,
                candidate.entry_price,
                candidate.velocity,
                candidate.rvol,
                candidate.vwap,
            )

            self.execution.execute_candidate(
                candidate
            )

    # ------------------------------------------------------------------
    # Main autonomous loop
    # ------------------------------------------------------------------

    def run(self) -> None:

        self.authenticate()

        current_date = today_ist()

        # If started on weekend, move to next weekday.
        while not market_day(current_date):
            current_date += timedelta(days=1)

        LOGGER.info(
            "System initialization complete. "
            "Trading date=%s.",
            current_date,
        )

        self.prepare_universe()

        # Silent/deep wait until exactly 09:46 IST.
        LOGGER.info(
            "Entering silent scheduler wait until 09:46:00 IST."
        )

        sleep_until(
            current_date,
            TRIGGER_TIME,
        )

        if STOP_EVENT.is_set():
            LOGGER.warning(
                "Shutdown requested before trigger."
            )
            return

        trigger_now = now_ist()

        LOGGER.info(
            "TRIGGER | %s | 09:46 execution phase started.",
            trigger_now.strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3],
        )

        candidates = self.scan_universe(
            current_date
        )

        self.route(
            candidates
        )

        LOGGER.info(
            "Daily strategy cycle completed."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:

    LOGGER.info(
        "Dhan Intraday Momentum Engine starting."
    )

    dhan_pin = os.getenv(
        "DHAN_PIN"
    )

    if not dhan_pin:
        LOGGER.error(
            "DHAN_PIN is not available in the runtime environment."
        )
        LOGGER.error(
            "Dhan's current TOTP authentication API requires "
            "the six-digit Dhan PIN in addition to client ID and TOTP."
        )
        return 2

    try:

        system = MomentumTradingSystem(
            dhan_pin=dhan_pin
        )

        system.run()

        return 0

    except DhanAuthenticationError:
        LOGGER.exception(
            "Dhan authentication failed."
        )
        return 3

    except DhanAPIError:
        LOGGER.exception(
            "Dhan API failure."
        )
        return 4

    except Exception:
        LOGGER.exception(
            "Fatal trading-system failure."
        )
        return 5


if __name__ == "__main__":
    sys.exit(
        main()
    )

#!/usr/bin/env python3

"""
Download Coinone OHLCV candles using Coinone official API with KST (UTC+9) time inputs and save to CSV.

Examples:
  # 1m candles for KRW/BTC within full KST timestamps (yy-mm-dd_HH:MM)
  python scripts/download_coinone_candle.py \
    --ticker KRW-BTC \
    --interval 1m \
    --from 23-11-01_23:14 \
    --to 25-08-22_23:00 \
    --out data/coinone/btc_1m_23-11-01_2314_25-08-22_2300.csv.gz

  # 5m candles for XRP/KRW using shorthand ticker (defaults quote to KRW)
  python scripts/download_coinone_candle.py \
    --ticker XRP \
    --interval 5m \
    --from 24-01-01_00:00 \
    --to 24-01-02_00:00

  # Split output by KST month for incremental storage
  python scripts/download_coinone_candle.py \
    --ticker USDT \
    --interval 1m \
    --from 23-11-01_00:00 \
    --to 24-02-01_00:00 \
    --split-monthly \
    --out-dir data/coinone/monthly

Notes:
- All input times are interpreted as KST (UTC+9). Output includes both timestamp_kst and timestamp_utc.
- API is UTC; this tool converts KST inputs to UTC for requests.
- If --out ends with .gz, the CSV is gzip-compressed.
- If a single asset is provided for --ticker (e.g., XRP or USDT), the quote
  currency defaults to KRW (e.g., KRW-XRP, KRW-USDT). You can also use
  KRW-BTC or BTC/KRW formats.
 - With --split-monthly, files are saved as {symbol}_{interval}_YYYY-MM.csv.gz
   under --out-dir (default: data/coinone/monthly/{symbol}). Existing files are
   appended to and deduplicated by timestamp_utc for incremental updates.
"""

import argparse
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Set

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_BASE = "https://api.coinone.co.kr/public/v2/chart"

# Per docs: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 6h, 1d, 1w, 1mon
SUPPORTED_INTERVALS = {
    "1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "6h", "1d", "1w", "1mon"
}


def parse_kst_ts(s: str) -> datetime:
    for fmt in ("%y-%m-%d_%H:%M", "%Y-%m-%d_%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc).astimezone(timezone.utc).astimezone(datetime.now().astimezone().tzinfo).astimezone(timezone.utc)
        except ValueError:
            continue
    raise ValueError("Time must be 'yy-mm-dd_HH:MM' or 'yyyy-mm-dd_HH:MM' in KST (UTC+9)")


def parse_kst_strict(s: str) -> pd.Timestamp:
    """Parse user input strictly as KST using explicit formats to avoid ambiguity.

    Accepts either yy-mm-dd_HH:MM or yyyy-mm-dd_HH:MM and returns a tz-aware
    pandas Timestamp localized to Asia/Seoul.
    """
    for fmt in ("%y-%m-%d_%H:%M", "%Y-%m-%d_%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return pd.Timestamp(dt).tz_localize("Asia/Seoul")
        except ValueError:
            continue
    raise ValueError("Time must be 'yy-mm-dd_HH:MM' or 'yyyy-mm-dd_HH:MM' in KST (UTC+9)")


def parse_kst_inputs(from_s: str, to_s: str) -> Tuple[datetime, datetime]:
    # Parse strictly as KST and convert to UTC
    kst = parse_kst_strict(from_s)
    kst_to = parse_kst_strict(to_s)
    start_utc = kst.tz_convert("UTC").to_pydatetime()
    end_utc = kst_to.tz_convert("UTC").to_pydatetime()
    if not (end_utc > start_utc):
        raise ValueError("--to must be after --from (KST)")
    return start_utc, end_utc


def split_ticker_to_quote_target(ticker: str) -> Tuple[str, str]:
    t = ticker.strip().upper()
    if "/" in t:
        a, b = t.split("/", 1)
    elif "-" in t:
        a, b = t.split("-", 1)
    else:
        # default to KRW-COIN style (quote-target requires both)
        a, b = "KRW", t
    # Ensure a is quote (market currency like KRW)
    if a in {"KRW", "USD", "USDT", "BTC"}:
        quote, target = a, b
    else:
        quote, target = b, a
    return quote, target


def create_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "coinone-candle-downloader/1.0",
    })
    return s


def is_error_payload(data: object) -> Tuple[bool, str]:
    """Best-effort detection of error payloads returned by the API."""
    if not isinstance(data, dict):
        return False, ""
    # Common patterns
    result_val = str(data.get("result", "")).lower()
    status_val = str(data.get("status", "")).lower()
    code_val = str(data.get("errorCode", data.get("code", ""))).lower()
    message = str(data.get("error", data.get("message", data.get("msg", ""))))
    if result_val in {"error", "fail", "failed"}:
        return True, message or f"result={result_val}"
    if status_val and status_val not in {"ok", "success"}:
        return True, message or f"status={status_val}"
    if code_val and code_val not in {"0", "ok", "success"}:
        return True, message or f"errorCode={code_val}"
    if "error" in data and data["error"]:
        return True, message or "error present"
    return False, ""


def fetch_chart_page(
    session: requests.Session,
    quote: str,
    target: str,
    interval: str,
    size: int,
    last_ts_ms: Optional[int] = None,
) -> List[dict]:
    url = f"{API_BASE}/{quote}/{target}"
    params = {"interval": interval, "size": int(size)}
    if last_ts_ms is not None:
        params["timestamp"] = int(last_ts_ms)

    max_attempts = 5
    base_backoff = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=20)
            status = resp.status_code
            if status >= 400:
                # Retry for rate limit and server errors
                if status in (429, 500, 502, 503, 504):
                    backoff = base_backoff * (2 ** (attempt - 1))
                    print(f"HTTP {status}; retrying in {backoff:.2f}s (attempt {attempt}/{max_attempts})")
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()

            try:
                data = resp.json()
            except ValueError:
                # Non-JSON; retry
                backoff = base_backoff * (2 ** (attempt - 1))
                print(f"Invalid JSON payload; retrying in {backoff:.2f}s (attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                continue

            # Check for in-band error responses
            is_err, msg = is_error_payload(data)
            if is_err:
                backoff = base_backoff * (2 ** (attempt - 1))
                printable = f"API error: {msg}" if msg else "API error response"
                print(f"{printable}; retrying in {backoff:.2f}s (attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                continue

            # Try to find array payload
            if isinstance(data, list):
                arr = data
            elif isinstance(data, dict):
                for key in ("candles", "chart", "data", "body", "result"):
                    if key in data and isinstance(data[key], list):
                        arr = data[key]
                        break
                else:
                    arr = data.get("items", []) if isinstance(data.get("items"), list) else []
            else:
                arr = []
            return arr

        except (requests.ConnectionError, requests.Timeout) as exc:
            backoff = base_backoff * (2 ** (attempt - 1))
            print(f"Request failed: {exc}; retrying in {backoff:.2f}s (attempt {attempt}/{max_attempts})")
            time.sleep(backoff)

    # If we reach here, all attempts failed
    raise requests.HTTPError("Failed to fetch chart page after retries")


def extract_candles(rows: List[dict]) -> List[Tuple[int, float, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float, float]] = []
    for r in rows:
        # Candidate keys according to common chart schemas
        ts = r.get("timestamp") or r.get("ts") or r.get("time") or r.get("t")
        # Timestamps can be in seconds or ms; try to detect
        if ts is None:
            continue
        ts_int = int(ts)
        if ts_int < 1e12:
            ts_ms = ts_int * 1000
        else:
            ts_ms = ts_int
        o = float(r.get("open") or r.get("o") or r.get("opening_price") or 0.0)
        h = float(r.get("high") or r.get("h") or r.get("high_price") or 0.0)
        l = float(r.get("low") or r.get("l") or r.get("low_price") or 0.0)
        c = float(r.get("close") or r.get("c") or r.get("trade_price") or 0.0)
        v = float(r.get("volume") or r.get("v") or r.get("candle_acc_trade_volume") or r.get("target_volume") or 0.0)
        out.append((ts_ms, o, h, l, c, v))
    return out


def download_range_official(
    ticker: str,
    interval: str,
    start_utc: datetime,
    end_utc: datetime,
    size: int = 500,
    max_pages: Optional[int] = None,
    sleep_sec: float = 0.08,
    progress_logging: bool = True,
) -> List[Tuple[int, float, float, float, float, float]]:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(SUPPORTED_INTERVALS)}")
    quote, target = split_ticker_to_quote_target(ticker)
    session = create_session()

    end_ms = int(end_utc.timestamp() * 1000)
    start_ms = int(start_utc.timestamp() * 1000)

    all_rows: List[Tuple[int, float, float, float, float, float]] = []
    seen_months_kst: Set[str] = set()
    cursor = end_ms
    pages = 0
    last_min_ts = None

    while cursor > start_ms:
        pages += 1
        batch_raw = fetch_chart_page(session, quote, target, interval, size=size, last_ts_ms=cursor)
        if not batch_raw:
            break
        batch = extract_candles(batch_raw)
        if not batch:
            break
        # Many APIs return newest->oldest. Normalize ordering ascending for ease later.
        batch_sorted = sorted(batch, key=lambda x: x[0])

        # Progress: log KST month(s) encountered in this batch once
        if progress_logging:
            try:
                ts_utc_series = pd.to_datetime([row[0] for row in batch_sorted], unit="ms", utc=True)
                month_series = ts_utc_series.tz_convert("Asia/Seoul").strftime("%Y-%m")
                for month_label in pd.Index(month_series).drop_duplicates().tolist():
                    if month_label not in seen_months_kst:
                        print(f"Downloading data {month_label}...", flush=True)
                        seen_months_kst.add(month_label)
            except (ValueError, TypeError, OverflowError):
                # Best-effort logging; do not fail downloads on log error
                pass
        # Keep only within range
        for ts_ms, o, h, l, c, v in batch_sorted:
            if start_ms <= ts_ms < end_ms:
                all_rows.append((ts_ms, o, h, l, c, v))
        # Advance cursor to just before the oldest ts we saw
        min_ts = batch_sorted[0][0]
        if last_min_ts is not None and min_ts >= last_min_ts:
            # Prevent infinite loop if API doesn't move window as expected
            cursor = min_ts - 1
        else:
            cursor = min_ts - 1
        last_min_ts = min_ts

        if max_pages is not None and pages >= max_pages:
            break
        time.sleep(max(0.02, float(sleep_sec)))

    # Deduplicate and sort
    all_rows = sorted({row[0]: row for row in all_rows}.values(), key=lambda x: x[0])
    return all_rows


def normalize_records(rows: List[Tuple[int, float, float, float, float, float]], pair_label: str, timeframe: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "timestamp_kst",
            "market",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp_utc",
        ])
    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    kst_series = df["timestamp_utc"].dt.tz_convert("Asia/Seoul")
    df["timestamp_kst"] = kst_series.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["market"] = pair_label
    df["timeframe"] = timeframe
    preferred_cols = [
        "timestamp_kst",
        "market",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp_utc",
    ]
    remaining = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + remaining].sort_values("timestamp_utc").reset_index(drop=True)
    return df


def save_monthly_incremental(
    df: pd.DataFrame,
    out_dir: str,
    symbol_short: str,
    interval: str,
) -> List[str]:
    """Save DataFrame split by KST month to gzip CSVs with incremental dedup.

    Returns list of written file paths.
    """
    if df.empty:
        return []

    # Compute KST month key for grouping
    kst_series = df["timestamp_utc"].dt.tz_convert("Asia/Seoul")
    month_keys = kst_series.dt.strftime("%Y-%m")
    df = df.copy()
    df["_month_kst"] = month_keys

    written_paths: List[str] = []
    base_dir = os.path.join(out_dir, symbol_short)
    os.makedirs(base_dir, exist_ok=True)

    for month_key, df_month in df.groupby("_month_kst"):
        filename = f"{symbol_short}_{interval}_{month_key}.csv.gz"
        path = os.path.join(base_dir, filename)

        # Deduplicate against existing file if present
        if os.path.exists(path):
            try:
                existing = pd.read_csv(path, parse_dates=["timestamp_utc"], dtype=str)
            except (pd.errors.ParserError, ValueError, OSError, UnicodeError):
                existing = pd.read_csv(path)
                if "timestamp_utc" in existing.columns:
                    existing["timestamp_utc"] = pd.to_datetime(existing["timestamp_utc"], utc=True, errors="coerce")
            all_df = pd.concat([existing, df_month], ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
        else:
            all_df = df_month.sort_values("timestamp_utc").reset_index(drop=True)

        compression = "gzip"
        tmp_path = path + ".tmp"
        try:
            all_df.to_csv(tmp_path, index=False, compression=compression)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

        written_paths.append(path)

    return written_paths


def iter_month_ranges_kst(kst_from: pd.Timestamp, kst_to: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Yield KST month ranges [start, end) and label YYYY-MM in ascending order.

    The start bound for the first month is max(month_start, kst_from). The end bound is
    exclusive and min(next_month_start, kst_to).
    """
    kst_from = kst_from.tz_convert("Asia/Seoul")
    kst_to = kst_to.tz_convert("Asia/Seoul")

    # First of the start month at 00:00 KST
    curr = pd.Timestamp(year=kst_from.year, month=kst_from.month, day=1, tz="Asia/Seoul")
    out: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    while curr < kst_to:
        next_start = curr + pd.offsets.MonthBegin(1)
        start_bound = max(curr, kst_from)
        end_bound = min(next_start, kst_to)
        label = curr.strftime('%Y-%m')
        out.append((start_bound, end_bound, label))
        curr = next_start
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download Coinone candles using official API with KST inputs (UTC+9) and save CSV. "
            "API works in UTC internally; this tool converts to/from KST."
        )
    )
    p.add_argument(
        "--ticker",
        required=True,
        help=(
            "Symbol or asset, e.g., KRW-BTC, BTC/KRW, or XRP. "
            "Single-asset form defaults quote to KRW (e.g., XRP => KRW-XRP)."
        ),
    )
    p.add_argument("--interval", required=True, help="Interval: 1m,3m,5m,10m,15m,30m,1h,2h,4h,6h,1d,1w,1mon")
    p.add_argument("--from", dest="from_s", required=True, help="KST start: yy-mm-dd_HH:MM or yyyy-mm-dd_HH:MM")
    p.add_argument("--to", dest="to_s", required=True, help="KST end: yy-mm-dd_HH:MM or yyyy-mm-dd_HH:MM")
    p.add_argument("--size", type=int, default=500, help="Candles per request (1..500)")
    p.add_argument("--max-pages", type=int, default=None, help="Limit number of pagination requests")
    p.add_argument("--sleep-sec", type=float, default=0.08, help="Sleep between requests")
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV(.gz) path. Defaults under data/coinone/ with inferred name",
    )
    p.add_argument(
        "--split-monthly",
        action="store_true",
        help="Split and save output by KST month with incremental dedup",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Base directory for monthly outputs (used with --split-monthly)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.interval not in SUPPORTED_INTERVALS:
        raise SystemExit(f"Unsupported interval '{args.interval}'. Allowed: {sorted(SUPPORTED_INTERVALS)}")

    start_utc, end_utc = parse_kst_inputs(args.from_s, args.to_s)
    kst_from = parse_kst_strict(args.from_s)
    kst_to = parse_kst_strict(args.to_s)
    quote, target = split_ticker_to_quote_target(args.ticker)
    pair_label = f"{target}/{quote}"

    print(
        f"Fetching {pair_label} {args.interval} candles\n"
        f"KST: {kst_from.strftime('%Y-%m-%d %H:%M:%S')} .. "
        f"{kst_to.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"UTC: {start_utc.strftime('%Y-%m-%d %H:%M:%S%z')} .. {end_utc.strftime('%Y-%m-%d %H:%M:%S%z')}"
    )

    if args.split_monthly:
        base_dir = args.out_dir or os.path.join("data", "coinone", "monthly")
        month_ranges = iter_month_ranges_kst(kst_from, kst_to)
        total_rows = 0
        files_written: List[str] = []
        for month_start_kst, month_end_kst, label in month_ranges:
            print(f"Downloading data {label}...", flush=True)
            rows_m = download_range_official(
                ticker=args.ticker,
                interval=args.interval,
                start_utc=month_start_kst.tz_convert("UTC").to_pydatetime(),
                end_utc=month_end_kst.tz_convert("UTC").to_pydatetime(),
                size=int(args.size),
                max_pages=args.max_pages,
                sleep_sec=args.sleep_sec,
                progress_logging=False,
            )
            df_m = normalize_records(rows_m, pair_label=pair_label, timeframe=args.interval)
            if df_m.empty:
                print(f"No data for {label}")
                continue
            short = target.lower()
            written = save_monthly_incremental(df_m, out_dir=base_dir, symbol_short=short, interval=args.interval)
            total_rows += len(df_m)
            files_written.extend(written)
            # For monthly save, we expect a single file path per month
            if written:
                print(f"Saved {len(df_m):,} rows to {os.path.abspath(written[0])}")

        if total_rows == 0:
            print("No data to write for any month in range")
        else:
            print(
                f"Saved {total_rows:,} total rows across {len(set(files_written))} file(s) under {os.path.abspath(base_dir)}"
            )
    else:
        rows = download_range_official(
            ticker=args.ticker,
            interval=args.interval,
            start_utc=start_utc,
            end_utc=end_utc,
            size=int(args.size),
            max_pages=args.max_pages,
            sleep_sec=args.sleep_sec,
            progress_logging=True,
        )

        df = normalize_records(rows, pair_label=pair_label, timeframe=args.interval)

        short = target.lower()
        out_path = args.out
        if out_path is None:
            os.makedirs(os.path.join("data", "coinone"), exist_ok=True)
            from_token = kst_from.strftime('%y-%m-%d_%H%M')
            to_token = kst_to.strftime('%y-%m-%d_%H%M')
            out_path = os.path.join("data", "coinone", f"{short}_{args.interval}_{from_token}_{to_token}.csv.gz")

        compression = "gzip" if str(out_path).endswith(".gz") else None
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df.to_csv(out_path, index=False, compression=compression)

        if df.empty:
            print(f"Saved 0 rows to {out_path} (no data in range)")
        else:
            kst_min = df["timestamp_utc"].dt.tz_convert("Asia/Seoul").min().strftime("%Y-%m-%d %H:%M:%S")
            kst_max = df["timestamp_utc"].dt.tz_convert("Asia/Seoul").max().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"Saved {len(df):,} rows to {out_path}.\n"
                f"KST range: {kst_min} .. {kst_max}\n"
                f"UTC range: {df['timestamp_utc'].min()} .. {df['timestamp_utc'].max()}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



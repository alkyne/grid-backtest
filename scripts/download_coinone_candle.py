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

Notes:
- All input times are interpreted as KST (UTC+9). Output includes both timestamp_kst and timestamp_utc.
- API is UTC; this tool converts KST inputs to UTC for requests.
- If --out ends with .gz, the CSV is gzip-compressed.
"""

import argparse
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import requests


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


def parse_kst_inputs(from_s: str, to_s: str) -> Tuple[datetime, datetime]:
    # Parse naive local times as KST and convert to UTC
    # We can't import zoneinfo on all envs? It's available; but keep requests-only; use pandas for tz
    kst = pd.Timestamp(from_s.replace("_", " ")).tz_localize("Asia/Seoul")
    kst_to = pd.Timestamp(to_s.replace("_", " ")).tz_localize("Asia/Seoul")
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
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "coinone-candle-downloader/1.0",
    })
    return s


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
    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # Try to find array payload
    if isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        for key in ("candles", "chart", "data", "body", "result"):
            if key in data and isinstance(data[key], list):
                arr = data[key]
                break
        else:
            # Some APIs nest under 'response' etc.; fallback: try 'items'
            arr = data.get("items", []) if isinstance(data.get("items"), list) else []
    else:
        arr = []
    return arr


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
) -> List[Tuple[int, float, float, float, float, float]]:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(SUPPORTED_INTERVALS)}")
    quote, target = split_ticker_to_quote_target(ticker)
    session = create_session()

    end_ms = int(end_utc.timestamp() * 1000)
    start_ms = int(start_utc.timestamp() * 1000)

    all_rows: List[Tuple[int, float, float, float, float, float]] = []
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download Coinone candles using official API with KST inputs (UTC+9) and save CSV. "
            "API works in UTC internally; this tool converts to/from KST."
        )
    )
    p.add_argument("--ticker", required=True, help="Symbol, e.g., KRW-BTC or BTC/KRW")
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
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.interval not in SUPPORTED_INTERVALS:
        raise SystemExit(f"Unsupported interval '{args.interval}'. Allowed: {sorted(SUPPORTED_INTERVALS)}")

    start_utc, end_utc = parse_kst_inputs(args.from_s, args.to_s)
    quote, target = split_ticker_to_quote_target(args.ticker)
    pair_label = f"{target}/{quote}"

    print(
        f"Fetching {pair_label} {args.interval} candles\n"
        f"KST: {pd.Timestamp(args.from_s.replace('_',' ')).tz_localize('Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S')} .. "
        f"{pd.Timestamp(args.to_s.replace('_',' ')).tz_localize('Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"UTC: {start_utc.strftime('%Y-%m-%d %H:%M:%S%z')} .. {end_utc.strftime('%Y-%m-%d %H:%M:%S%z')}"
    )

    rows = download_range_official(
        ticker=args.ticker,
        interval=args.interval,
        start_utc=start_utc,
        end_utc=end_utc,
        size=int(args.size),
        max_pages=args.max_pages,
        sleep_sec=args.sleep_sec,
    )

    df = normalize_records(rows, pair_label=pair_label, timeframe=args.interval)

    out_path = args.out
    if out_path is None:
        os.makedirs(os.path.join("data", "coinone"), exist_ok=True)
        from_token = pd.Timestamp(args.from_s.replace('_',' ')).strftime('%y-%m-%d_%H%M')
        to_token = pd.Timestamp(args.to_s.replace('_',' ')).strftime('%y-%m-%d_%H%M')
        short = target.lower()
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



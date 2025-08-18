# python scripts/download_upbit_seconds.py --market KRW-XRP --unit 1 --lookback-days 89 --out data/upbit/krw-xrp_1s_last3m.csv.gz
#!/usr/bin/env python3

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE_SECONDS = "https://api.upbit.com/v1/candles/seconds"


def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "grid-backtest/1.0 (+https://docs.upbit.com/kr/reference/list-candles-seconds)",
    })
    return session


def format_to_param(dt: datetime) -> str:
    # Use space-separated format per docs examples (UTC time)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fetch_seconds_candles(
    session: requests.Session,
    market: str,
    unit_seconds: int,
    to_dt: Optional[datetime],
    count: int,
    timeout: float,
) -> List[dict]:
    # Try query-param style first: /v1/candles/seconds?unit=1
    params = {"market": market, "count": int(count), "unit": int(unit_seconds)}
    if to_dt is not None:
        params["to"] = format_to_param(to_dt)

    resp = session.get(API_BASE_SECONDS, params=params, timeout=timeout)

    # Fallback to path-param style on 404 (or 405/400 just in case)
    if resp.status_code == 404:
        url_fallback = f"{API_BASE_SECONDS}/{unit_seconds}"
        params_fb = {"market": market, "count": int(count)}
        if to_dt is not None:
            params_fb["to"] = format_to_param(to_dt)
        resp = session.get(url_fallback, params=params_fb, timeout=timeout)

    resp.raise_for_status()
    return resp.json()


def normalize_records(records: List[dict], market: str, unit_seconds: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)

    expected_cols = [
        "candle_date_time_utc",
        "opening_price",
        "high_price",
        "low_price",
        "trade_price",
        "candle_acc_trade_volume",
        "candle_acc_trade_price",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["timestamp_utc"] = pd.to_datetime(
        df["candle_date_time_utc"], format="%Y-%m-%dT%H:%M:%S", utc=True, errors="coerce"
    )

    df = df.rename(
        columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "quote_volume",
        }
    )

    df["market"] = market
    df["unit_seconds"] = int(unit_seconds)

    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    preferred_cols = [
        "timestamp_utc",
        "market",
        "unit_seconds",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]
    remaining = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + remaining]
    return df


def download_seconds_range(
    market: str,
    unit_seconds: int,
    lookback_days: int,
    out_path: str,
    count_per_req: int = 200,
    sleep_sec: float = 0.12,
    timeout: float = 20.0,
    max_requests: Optional[int] = None,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    session = create_session()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(lookback_days))

    all_records: List[dict] = []
    requests_made = 0

    while end_dt > start_dt:
        requests_made += 1
        try:
            batch = fetch_seconds_candles(
                session=session,
                market=market,
                unit_seconds=unit_seconds,
                to_dt=end_dt,
                count=count_per_req,
                timeout=timeout,
            )
        except requests.HTTPError as http_err:
            print(f"HTTP error: {http_err}")
            break
        except Exception as exc:
            print(f"Request failed: {exc}")
            break

        if not batch:
            print("No more data returned by API; stopping.")
            break

        all_records.extend(batch)

        oldest_utc_str = batch[-1].get("candle_date_time_utc")
        if oldest_utc_str is None:
            print("Missing candle_date_time_utc; stopping to avoid loop.")
            break

        try:
            oldest_dt = datetime.strptime(oldest_utc_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            oldest_dt = pd.to_datetime(oldest_utc_str, utc=True).to_pydatetime()

        end_dt = oldest_dt - timedelta(seconds=1)

        time.sleep(max(0.1, float(sleep_sec)))

        if max_requests is not None and requests_made >= max_requests:
            print(f"Reached max_requests={max_requests}; stopping early (for smoke test).")
            break

        if end_dt <= start_dt:
            break

        if requests_made % 20 == 0:
            print(
                f"Made {requests_made} requests; oldest so far: {oldest_dt.isoformat()} (target start: {start_dt.isoformat()})"
            )

    if not all_records:
        print("No records fetched. Nothing to save.")
        return out_path

    df = normalize_records(all_records, market=market, unit_seconds=unit_seconds)

    df = df[df["timestamp_utc"] >= pd.Timestamp(start_dt)].reset_index(drop=True)

    compression = "gzip" if out_path.endswith(".gz") else None
    df.to_csv(out_path, index=False, compression=compression)

    print(
        f"Saved {len(df):,} rows to {out_path}. Range: {df['timestamp_utc'].min()} .. {df['timestamp_utc'].max()}"
    )
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Upbit seconds candles for backtesting. "
            "Docs: https://docs.upbit.com/kr/reference/list-candles-seconds"
        )
    )
    parser.add_argument("--market", default="KRW-XRP", help="Market code, e.g., KRW-XRP")
    parser.add_argument("--unit", dest="unit_seconds", type=int, default=1, help="Seconds candle unit, e.g., 1")
    parser.add_argument(
        "--lookback-days", type=int, default=89, help="How many days back to fetch (Upbit supports ~3 months for seconds)"
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "upbit", "krw-xrp_1s_last3m.csv.gz"),
        help="Output CSV(.gz) path",
    )
    parser.add_argument("--count", type=int, default=200, help="Candles per request (max 200)")
    parser.add_argument("--sleep-sec", type=float, default=0.12, help="Sleep between requests to respect rate limits")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout seconds")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Limit requests for smoke testing (None for full run)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_path_abs = os.path.abspath(args.out)

    print(
        f"Fetching {args.market} {args.unit_seconds}s candles for ~{args.lookback_days} days\n"
        f"Output: {out_path_abs}\n"
        f"Endpoint: {API_BASE_SECONDS} (unit={args.unit_seconds})"
    )

    try:
        download_seconds_range(
            market=args.market,
            unit_seconds=args.unit_seconds,
            lookback_days=args.lookback_days,
            out_path=out_path_abs,
            count_per_req=args.count,
            sleep_sec=args.sleep_sec,
            timeout=args.timeout,
            max_requests=args.max_requests,
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130
    except Exception as exc:
        print(f"Failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

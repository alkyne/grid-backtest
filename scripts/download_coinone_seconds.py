'''
python scripts/download_coinone_seconds.py \
  --symbol XRP/KRW \
  --lookback-days 1 \
  --out data/coinone/xrp-krw_1s_last1d.csv.gz
'''
#!/usr/bin/env python3

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import ccxt  # type: ignore
import pandas as pd


def floor_second(dt: datetime) -> datetime:
    return dt.replace(microsecond=0)


def aggregate_trades_to_1s(trades: List[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            'timestamp_utc','market','unit_seconds','open','high','low','close','volume','quote_volume'
        ])
    rows: Dict[int, Dict[str, float]] = {}
    for t in trades:
        ts_ms = t['timestamp']
        price = float(t['price'])
        amount = float(t['amount'])
        cost = float(t.get('cost', price * amount))
        ts_sec = int(ts_ms // 1000)
        bucket = rows.get(ts_sec)
        if bucket is None:
            rows[ts_sec] = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': amount,
                'quote_volume': cost,
            }
        else:
            b = bucket
            if price > b['high']:
                b['high'] = price
            if price < b['low']:
                b['low'] = price
            # For close, use latest trade in that second; ccxt returns most-recent-first by default, so overwrite
            b['close'] = price
            b['volume'] += amount
            b['quote_volume'] += cost
    # Build DataFrame ordered by time
    records = []
    for ts_sec in sorted(rows.keys()):
        dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        rec = rows[ts_sec]
        records.append({
            'timestamp_utc': dt,
            'open': rec['open'],
            'high': rec['high'],
            'low': rec['low'],
            'close': rec['close'],
            'volume': rec['volume'],
            'quote_volume': rec['quote_volume'],
        })
    df = pd.DataFrame.from_records(records)
    return df


def fetch_coinone_trades(market_symbol: str, since_ms: Optional[int], limit: int = 1000) -> List[dict]:
    ex = ccxt.coinone()
    # coinone symbols are like 'XRP/KRW'
    trades = ex.fetch_trades(symbol=market_symbol, since=since_ms, limit=limit)
    return trades


def download_coinone_seconds(
    market_symbol: str,
    lookback_days: int,
    out_path: str,
    batch_limit: int = 1000,
    sleep_sec: float = 0.1,
    max_requests: Optional[int] = None,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    # Start from the most recent trades; paginate backwards by updating since_ms to the oldest seen - 1
    since_ms: Optional[int] = None
    all_frames: List[pd.DataFrame] = []
    total_trades = 0

    requests_made = 0
    prev_oldest_ms: Optional[int] = None
    no_progress_iters = 0
    while True:
        requests_made += 1
        trades = fetch_coinone_trades(market_symbol, since_ms=since_ms, limit=batch_limit)
        if not trades:
            break
        if sleep_sec and float(sleep_sec) > 0:
            time.sleep(float(sleep_sec))
        total_trades += len(trades)
        # ccxt returns most-recent-first; find oldest timestamp to paginate backwards
        oldest_ms = min(t['timestamp'] for t in trades)
        # Detect pagination stalls (e.g., exchange ignoring 'since')
        if prev_oldest_ms is not None and oldest_ms >= prev_oldest_ms:
            no_progress_iters += 1
        else:
            no_progress_iters = 0
        prev_oldest_ms = oldest_ms
        # Aggregate to seconds and store
        df = aggregate_trades_to_1s(trades)
        all_frames.append(df)
        # Move the window back by 1 ms before oldest to avoid duplicates
        since_ms = oldest_ms - 1
        # Stop when we've passed the start time
        if oldest_ms < int(start_dt.timestamp() * 1000):
            break
        # Stop if too many requests or no progress
        if max_requests is not None and requests_made >= max_requests:
            print(f"Reached max_requests={max_requests}; stopping early.")
            break
        if no_progress_iters >= 3:
            print("Pagination made no progress for 3 iterations; the exchange may ignore 'since'. Stopping.")
            break
        if requests_made % 10 == 0:
            oldest_dt = datetime.fromtimestamp(oldest_ms / 1000.0, tz=timezone.utc)
            print(f"Requests={requests_made}, oldest so far (UTC) {oldest_dt.isoformat()}")
    if not all_frames:
        # write empty file with correct header
        empty = pd.DataFrame(columns=['timestamp_utc','market','unit_seconds','open','high','low','close','volume','quote_volume'])
        empty.to_csv(out_path, index=False, compression='gzip' if out_path.endswith('.gz') else None)
        return out_path

    merged = pd.concat(all_frames, ignore_index=True)
    # Deduplicate by timestamp_utc (last occurrence wins), sort ascending
    merged = merged.drop_duplicates(subset=['timestamp_utc'], keep='last').sort_values('timestamp_utc').reset_index(drop=True)

    # Add required columns to match Upbit schema
    merged['market'] = market_symbol.replace('/', '-')
    merged['unit_seconds'] = 1
    # Add KST timestamp column as formatted string (UTC+9)
    kst_series = merged['timestamp_utc'].dt.tz_convert('Asia/Seoul')
    merged['timestamp_kst'] = kst_series.dt.strftime('%Y-%m-%d %H:%M:%S')

    # Trim to requested window
    merged = merged[(merged['timestamp_utc'] >= pd.Timestamp(start_dt)) & (merged['timestamp_utc'] <= pd.Timestamp(end_dt))].reset_index(drop=True)

    # Prefer KST-first column ordering when saving
    preferred_cols = [
        'timestamp_kst',
        'market',
        'unit_seconds',
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'timestamp_utc',
    ]
    remaining_cols = [c for c in merged.columns if c not in preferred_cols]
    merged = merged[preferred_cols + remaining_cols]

    compression = 'gzip' if out_path.endswith('.gz') else None
    tmp_path = out_path + '.tmp'
    try:
        merged.to_csv(tmp_path, index=False, compression=compression)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

    utc_min = merged['timestamp_utc'].min()
    utc_max = merged['timestamp_utc'].max()
    kst_min_str = kst_series.min().strftime('%Y-%m-%d %H:%M:%S')
    kst_max_str = kst_series.max().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f"Saved {len(merged):,} rows from {total_trades:,} trades to {out_path}\n"
        f"KST range: {kst_min_str} .. {kst_max_str}\n"
        f"UTC range: {utc_min} .. {utc_max}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Download Coinone 1-second OHLCV by aggregating public trades')
    p.add_argument('--symbol', default='XRP/KRW', help='Coinone symbol, e.g., XRP/KRW')
    p.add_argument('--lookback-days', type=int, default=1, help='Days to look back')
    p.add_argument('--out', default=os.path.join('data','coinone','xrp-krw_1s_last1d.csv.gz'))
    p.add_argument('--batch-limit', type=int, default=1000)
    p.add_argument('--sleep-sec', type=float, default=0.1, help='Sleep between requests to respect rate limits')
    p.add_argument('--max-requests', type=int, default=None, help='Limit total requests (safety against stalls)')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_abs = os.path.abspath(args.out)
    download_coinone_seconds(
        market_symbol=args.symbol,
        lookback_days=args.lookback_days,
        out_path=out_abs,
        batch_limit=args.batch_limit,
        sleep_sec=args.sleep_sec,
        max_requests=args.max_requests,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

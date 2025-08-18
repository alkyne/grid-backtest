#!/usr/bin/env python3

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root is on sys.path for 'scripts' package import
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the trade history engine
from scripts.trade_history import run_trade_history


@dataclass
class BacktestSweepConfig:
    data_path: str
    start_local: str  # yy-mm-dd:hh-mm in KST
    end_local: str
    first_ladder_price: float
    tick_min: int = 5
    tick_max: int = 50
    maker_fee_rate: float = 0.0002
    ladder_rungs: int = 4
    order_krw_per_buy: float = 10000.0
    min_order_krw: float = 5000.0
    qty_unit: float = 0.0001


def sweep_ticks(cfg: BacktestSweepConfig) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for tick in range(cfg.tick_min, cfg.tick_max + 1):
        _, metrics = run_trade_history(
            data_path=cfg.data_path,
            start_local=cfg.start_local,
            end_local=cfg.end_local,
            tick_size=tick,
            first_ladder_price=cfg.first_ladder_price,
            maker_fee=cfg.maker_fee_rate,
            ladder_rungs=cfg.ladder_rungs,
            order_krw=cfg.order_krw_per_buy,
            min_order_krw=cfg.min_order_krw,
            qty_unit=cfg.qty_unit,
            tz='Asia/Seoul',
        )
        records.append({
            'tick': tick,
            'realized_pnl_krw': metrics['realized_pnl_krw'],
            'unrealized_pnl_krw': metrics['unrealized_pnl_krw'],
            'pnl_rw': metrics['total_pnl_krw'],
            'end_inventory_qty': metrics['end_inventory_qty'],
        })
    df = pd.DataFrame.from_records(records).sort_values('pnl_rw', ascending=False).reset_index(drop=True)
    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep TICK sizes using trade_history engine (times in KST)")
    p.add_argument("--data", default=os.path.join("data", "upbit", "krw-xrp_1s_last3m.csv.gz"))
    p.add_argument("--from", dest="from_s", required=True, help="Start (yy-mm-dd:hh-mm) in KST")
    p.add_argument("--to", dest="to_s", required=True, help="End (yy-mm-dd:hh-mm) in KST")
    p.add_argument("--ladder", dest="ladder_price", type=float, required=True, help="First ladder buy price (KRW)")
    p.add_argument("--tick-min", type=int, default=5)
    p.add_argument("--tick-max", type=int, default=50)
    p.add_argument("--maker-fee", type=float, default=0.0002)
    p.add_argument("--ladder-rungs", type=int, default=4)
    p.add_argument("--order-krw", type=float, default=10000.0)
    p.add_argument("--min-order-krw", type=float, default=5000.0)
    p.add_argument("--qty-unit", type=float, default=0.0001)
    p.add_argument("--out", default=None, help="Optional CSV to save sweep results")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    cfg = BacktestSweepConfig(
        data_path=args.data,
        start_local=args.from_s,
        end_local=args.to_s,
        first_ladder_price=args.ladder_price,
        tick_min=args.tick_min,
        tick_max=args.tick_max,
        maker_fee_rate=args.maker_fee,
        ladder_rungs=args.ladder_rungs,
        order_krw_per_buy=args.order_krw,
        min_order_krw=args.min_order_krw,
        qty_unit=args.qty_unit,
    )

    results = sweep_ticks(cfg)
    best = results.iloc[0]
    print(f"Best TICK: {int(best['tick'])} with PnL_RW {best['pnl_rw']:.2f} KRW (realized {best['realized_pnl_krw']:.2f}, unrealized {best['unrealized_pnl_krw']:.2f})")
    print(results.head(10).to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"Saved results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

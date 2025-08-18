#!/usr/bin/env python3

import argparse
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo


@dataclass
class Config:
    data_path: str
    start_dt_utc: datetime
    end_dt_utc: datetime
    tz: ZoneInfo
    tick_size: int
    first_ladder_price: float
    maker_fee_rate: float = 0.0002  # 0.02%
    min_order_krw: float = 5000.0
    qty_unit: float = 0.0001
    ladder_rungs: int = 4
    order_krw_per_buy: float = 10000.0


@dataclass
class Lot:
    qty: float
    price: float
    buy_fee_krw_remaining: float


@dataclass
class State:
    buy_orders: Dict[int, float] = field(default_factory=dict)  # price -> qty
    sell_orders: Dict[int, float] = field(default_factory=dict)
    buy_buckets: Dict[int, float] = field(default_factory=dict)
    sell_buckets: Dict[int, float] = field(default_factory=dict)
    inventory: List[Lot] = field(default_factory=list)
    realized_pnl_krw: float = 0.0
    seq_counter: int = 0


# ---------- Helpers ----------

def parse_user_dt_local(s: str, tz: ZoneInfo) -> datetime:
    # Format: yy-mm-dd:hh-mm (interpreted in provided timezone)
    dt_local = datetime.strptime(s, "%y-%m-%d:%H:%M")
    return dt_local.replace(tzinfo=tz)


def floor_to_qty_unit(qty: float, unit: float) -> float:
    return math.floor(qty / unit) * unit


def q_price(p: float) -> int:
    return int(math.floor(p))


def format_ts(ts_utc: pd.Timestamp, tz: ZoneInfo) -> str:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_local = ts_utc.tz_convert(tz)
    return ts_local.strftime("%Y-%m-%d %H:%M:%S")


def log_event(st: State, events: List[dict], candle_ts_utc: pd.Timestamp, tz: ZoneInfo, **kwargs) -> None:
    ev = {
        "candle_ts": format_ts(candle_ts_utc, tz),
        "seq": st.seq_counter,
        **kwargs,
    }
    st.seq_counter += 1
    events.append(ev)


# ---------- Core Mechanics ----------
# (Placement logs for orders are currently disabled by user edits)


def place_initial_ladder(cfg: Config, st: State, events: List[dict], candle_ts_utc: pd.Timestamp) -> None:
    for i in range(cfg.ladder_rungs):
        rung_price = q_price(cfg.first_ladder_price - i * cfg.tick_size)
        if rung_price <= 0:
            continue
        qty = floor_to_qty_unit(cfg.order_krw_per_buy / rung_price, cfg.qty_unit)
        if qty * rung_price < cfg.min_order_krw or qty <= 0:
            continue
        st.buy_orders[rung_price] = st.buy_orders.get(rung_price, 0.0) + qty
        # ORDER_PLACED_BUY logging disabled


def on_buy_fill(cfg: Config, st: State, events: List[dict], candle_ts_utc: pd.Timestamp, price: int, qty: float) -> None:
    fee_krw = price * qty * cfg.maker_fee_rate
    st.inventory.append(Lot(qty=qty, price=float(price), buy_fee_krw_remaining=fee_krw))

    log_event(
        st, events, candle_ts_utc, cfg.tz,
        event_type="BUY_FILL",
        price=price,
        qty=qty,
        fee_krw=fee_krw,
        realized_pnl_krw=0.0,
        cumulative_realized_pnl_krw=st.realized_pnl_krw,
        inventory_qty_after=sum(l.qty for l in st.inventory),
        filled_at=format_ts(candle_ts_utc, cfg.tz),
    )

    target = q_price(price + cfg.tick_size)
    st.sell_buckets[target] = st.sell_buckets.get(target, 0.0) + qty
    if st.sell_buckets[target] * target >= cfg.min_order_krw:
        place_qty = floor_to_qty_unit(st.sell_buckets[target], cfg.qty_unit)
        if place_qty * target >= cfg.min_order_krw and place_qty > 0:
            st.sell_orders[target] = st.sell_orders.get(target, 0.0) + place_qty
            st.sell_buckets[target] = 0.0
            # ORDER_PLACED_SELL logging disabled


def on_sell_fill(cfg: Config, st: State, events: List[dict], candle_ts_utc: pd.Timestamp, price: int, qty: float) -> None:
    remaining = qty
    realized_for_this_fill = 0.0
    while remaining > 0 and st.inventory:
        lot = st.inventory[0]
        take = min(lot.qty, remaining)
        if lot.qty > 0:
            buy_fee_part = lot.buy_fee_krw_remaining * (take / lot.qty)
        else:
            buy_fee_part = 0.0
        lot.buy_fee_krw_remaining = max(0.0, lot.buy_fee_krw_remaining - buy_fee_part)
        sell_fee_krw = price * take * cfg.maker_fee_rate
        pnl = (price - lot.price) * take - (buy_fee_part + sell_fee_krw)
        realized_for_this_fill += pnl
        st.realized_pnl_krw += pnl
        lot.qty -= take
        remaining -= take
        if lot.qty <= 0:
            st.inventory.pop(0)

    log_event(
        st, events, candle_ts_utc, cfg.tz,
        event_type="SELL_FILL",
        price=price,
        qty=qty,
        fee_krw=price * qty * cfg.maker_fee_rate,
        realized_pnl_krw=realized_for_this_fill,
        cumulative_realized_pnl_krw=st.realized_pnl_krw,
        inventory_qty_after=sum(l.qty for l in st.inventory),
        filled_at=format_ts(candle_ts_utc, cfg.tz),
    )

    target = q_price(price - cfg.tick_size)
    if target > 0:
        st.buy_buckets[target] = st.buy_buckets.get(target, 0.0) + qty
        if st.buy_buckets[target] * target >= cfg.min_order_krw:
            place_qty = floor_to_qty_unit(st.buy_buckets[target], cfg.qty_unit)
            if place_qty * target >= cfg.min_order_krw and place_qty > 0:
                st.buy_orders[target] = st.buy_orders.get(target, 0.0) + place_qty
                st.buy_buckets[target] = 0.0
                # ORDER_PLACED_BUY logging disabled


def fill_segment(cfg: Config, st: State, events: List[dict], candle_ts_utc: pd.Timestamp, p0: float, p1: float) -> None:
    if p1 == p0:
        return
    seg_min = q_price(min(p0, p1))
    seg_max = q_price(max(p0, p1))

    if p1 > p0:
        if st.sell_orders:
            prices = [p for p in list(st.sell_orders.keys()) if seg_min <= p <= seg_max]
            for price in sorted(prices):
                qty = st.sell_orders.pop(price)
                if qty > 0:
                    on_sell_fill(cfg, st, events, candle_ts_utc, price, qty)
    else:
        if st.buy_orders:
            prices = [p for p in list(st.buy_orders.keys()) if seg_min <= p <= seg_max]
            for price in sorted(prices, reverse=True):
                qty = st.buy_orders.pop(price)
                if qty > 0:
                    on_buy_fill(cfg, st, events, candle_ts_utc, price, qty)


def simulate(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, State]:
    events: List[dict] = []
    st = State()

    first_ts_utc = pd.to_datetime(df.iloc[0]["timestamp_utc"], utc=True)  # type: ignore[index]
    place_initial_ladder(cfg, st, events, first_ts_utc)

    for _, row in df.iterrows():
        o = float(row["open"])  # type: ignore[index]
        h = float(row["high"])  # type: ignore[index]
        l = float(row["low"])   # type: ignore[index]
        c = float(row["close"]) # type: ignore[index]
        ts_utc = pd.to_datetime(row["timestamp_utc"], utc=True)  # type: ignore[index]

        if c >= o:
            fill_segment(cfg, st, events, ts_utc, o, l)
            fill_segment(cfg, st, events, ts_utc, l, h)
            fill_segment(cfg, st, events, ts_utc, h, c)
        else:
            fill_segment(cfg, st, events, ts_utc, o, h)
            fill_segment(cfg, st, events, ts_utc, h, l)
            fill_segment(cfg, st, events, ts_utc, l, c)

    if not events:
        empty = pd.DataFrame(columns=[
            "candle_ts","seq","event_type","price","qty","notional_krw","fee_krw","realized_pnl_krw","cumulative_realized_pnl_krw","inventory_qty_after","filled_at"
        ])
        return empty, st

    df_ev = pd.DataFrame.from_records(events)

    numeric_cols = [
        "price","qty","notional_krw","fee_krw","realized_pnl_krw","cumulative_realized_pnl_krw","inventory_qty_after"
    ]
    for col in numeric_cols:
        if col not in df_ev.columns:
            df_ev[col] = 0.0
        else:
            df_ev[col] = df_ev[col].fillna(0.0)

    df_ev = df_ev.sort_values(["candle_ts","seq"]).reset_index(drop=True)
    return df_ev, st


def load_and_slice_data(data_path: str, start_dt_utc: datetime, end_dt_utc: datetime) -> pd.DataFrame:
    if not str(data_path).endswith(".csv.gz"):
        raise ValueError("Input dataset must be gzip-compressed with .csv.gz")
    df = pd.read_csv(data_path)
    if "timestamp_utc" not in df.columns:
        raise ValueError("Dataset missing 'timestamp_utc' column")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    mask = (df["timestamp_utc"] >= start_dt_utc) & (df["timestamp_utc"] < end_dt_utc)
    sliced = df.loc[mask].copy()
    if sliced.empty:
        raise ValueError("No data in the requested time range")
    for col in ["open", "high", "low", "close"]:
        if col not in sliced.columns:
            raise ValueError(f"Dataset missing column: {col}")
    return sliced


def load_and_slice_df(df: pd.DataFrame, start_dt_utc: datetime, end_dt_utc: datetime) -> pd.DataFrame:
    if "timestamp_utc" not in df.columns:
        raise ValueError("DataFrame missing 'timestamp_utc' column")
    df2 = df.copy()
    df2["timestamp_utc"] = pd.to_datetime(df2["timestamp_utc"], utc=True)
    mask = (df2["timestamp_utc"] >= start_dt_utc) & (df2["timestamp_utc"] < end_dt_utc)
    sliced = df2.loc[mask].copy()
    if sliced.empty:
        raise ValueError("No data in the requested time range")
    for col in ["open", "high", "low", "close"]:
        if col not in sliced.columns:
            raise ValueError(f"DataFrame missing column: {col}")
    return sliced


def run_trade_history_df(
    df: pd.DataFrame,
    start_local: str,
    end_local: str,
    tick_size: int,
    first_ladder_price: float,
    maker_fee: float = 0.0002,
    ladder_rungs: int = 4,
    order_krw: float = 10000.0,
    min_order_krw: float = 5000.0,
    qty_unit: float = 0.0001,
    tz: str = 'Asia/Seoul',
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tzinfo = ZoneInfo(tz)
    start_dt_local = parse_user_dt_local(start_local, tzinfo)
    end_dt_local = parse_user_dt_local(end_local, tzinfo)
    if not (end_dt_local > start_dt_local):
        raise ValueError("End must be after start")

    cfg = Config(
        data_path="<in-memory>",
        start_dt_utc=start_dt_local.astimezone(timezone.utc),
        end_dt_utc=end_dt_local.astimezone(timezone.utc),
        tz=tzinfo,
        tick_size=tick_size,
        first_ladder_price=first_ladder_price,
        maker_fee_rate=maker_fee,
        min_order_krw=min_order_krw,
        qty_unit=qty_unit,
        ladder_rungs=ladder_rungs,
        order_krw_per_buy=order_krw,
    )

    sliced = load_and_slice_df(df, cfg.start_dt_utc, cfg.end_dt_utc)
    hist, st = simulate(cfg, sliced)

    last_close = float(sliced.iloc[-1]["close"])  # type: ignore[index]
    unreal = 0.0
    inv_qty = 0.0
    for lot in st.inventory:
        hypothetical_sell_fee = last_close * lot.qty * cfg.maker_fee_rate
        unreal += (last_close - lot.price) * lot.qty - lot.buy_fee_krw_remaining - hypothetical_sell_fee
        inv_qty += lot.qty

    realized = float(hist["cumulative_realized_pnl_krw"].iloc[-1]) if not hist.empty else 0.0
    total = realized + unreal

    metrics = {
        "realized_pnl_krw": realized,
        "unrealized_pnl_krw": unreal,
        "total_pnl_krw": total,
        "end_inventory_qty": inv_qty,
    }
    return hist, metrics


def run_trade_history(
    data_path: str,
    start_local: str,
    end_local: str,
    tick_size: int,
    first_ladder_price: float,
    maker_fee: float = 0.0002,
    ladder_rungs: int = 4,
    order_krw: float = 10000.0,
    min_order_krw: float = 5000.0,
    qty_unit: float = 0.0001,
    tz: str = 'Asia/Seoul',
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tzinfo = ZoneInfo(tz)
    start_dt_local = parse_user_dt_local(start_local, tzinfo)
    end_dt_local = parse_user_dt_local(end_local, tzinfo)
    if not (end_dt_local > start_dt_local):
        raise ValueError("End must be after start")

    cfg = Config(
        data_path=data_path,
        start_dt_utc=start_dt_local.astimezone(timezone.utc),
        end_dt_utc=end_dt_local.astimezone(timezone.utc),
        tz=tzinfo,
        tick_size=tick_size,
        first_ladder_price=first_ladder_price,
        maker_fee_rate=maker_fee,
        min_order_krw=min_order_krw,
        qty_unit=qty_unit,
        ladder_rungs=ladder_rungs,
        order_krw_per_buy=order_krw,
    )

    df = load_and_slice_data(cfg.data_path, cfg.start_dt_utc, cfg.end_dt_utc)
    hist, st = simulate(cfg, df)

    last_close = float(df.iloc[-1]["close"])  # type: ignore[index]
    unreal = 0.0
    inv_qty = 0.0
    for lot in st.inventory:
        hypothetical_sell_fee = last_close * lot.qty * cfg.maker_fee_rate
        unreal += (last_close - lot.price) * lot.qty - lot.buy_fee_krw_remaining - hypothetical_sell_fee
        inv_qty += lot.qty

    realized = float(hist["cumulative_realized_pnl_krw"].iloc[-1]) if not hist.empty else 0.0
    total = realized + unreal

    metrics = {
        "realized_pnl_krw": realized,
        "unrealized_pnl_krw": unreal,
        "total_pnl_krw": total,
        "end_inventory_qty": inv_qty,
    }
    return hist, metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce detailed trade history for a fixed TICK size (times in KST)")
    p.add_argument("--data", default=os.path.join("data", "upbit", "krw-xrp_1s_last3m.csv.gz"))
    p.add_argument("--from", dest="from_s", required=True, help="Start (yy-mm-dd:hh-mm) in KST")
    p.add_argument("--to", dest="to_s", required=True, help="End (yy-mm-dd:hh-mm) in KST")
    p.add_argument("--tick", type=int, required=True, help="TICK size in KRW")
    p.add_argument("--first-ladder-price", type=float, required=True, help="First ladder buy price (KRW)")
    p.add_argument("--maker-fee", type=float, default=0.0002)
    p.add_argument("--ladder-rungs", type=int, default=4)
    p.add_argument("--order-krw", type=float, default=10000.0)
    p.add_argument("--min-order-krw", type=float, default=5000.0)
    p.add_argument("--qty-unit", type=float, default=0.0001)
    p.add_argument("--out", default=None, help="Optional CSV to save trade history (auto if omitted)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    hist, metrics = run_trade_history(
        data_path=args.data,
        start_local=args.from_s,
        end_local=args.to_s,
        tick_size=args.tick,
        first_ladder_price=args.first_ladder_price,
        maker_fee=args.maker_fee,
        ladder_rungs=args.ladder_rungs,
        order_krw=args.order_krw,
        min_order_krw=args.min_order_krw,
        qty_unit=args.qty_unit,
        tz='Asia/Seoul',
    )

    print(hist.head(30).to_string(index=False)) if not hist.empty else print("No events generated.")
    print(
        f"\nSummary: realized={metrics['realized_pnl_krw']:.2f} KRW, unrealized={metrics['unrealized_pnl_krw']:.2f} KRW, total={metrics['total_pnl_krw']:.2f} KRW, end_inventory_qty={metrics['end_inventory_qty']:.4f}"
    )

    out_path = args.out
    if out_path is None:
        from_token = args.from_s.replace(":", "_")
        to_token = args.to_s.replace(":", "_")
        ladder_token = str(int(round(args.first_ladder_price)))
        base_dir = os.path.dirname(os.path.abspath(args.data))
        out_path = os.path.join(
            base_dir,
            f"trade_history_tick{args.tick}_ladder{ladder_token}_{from_token}_{to_token}.csv",
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hist.to_csv(out_path, index=False)
    print(f"Saved trade history to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

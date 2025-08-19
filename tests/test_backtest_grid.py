import gzip
import io
import os

import pandas as pd
import pytest

import scripts.backtest_grid as bg


def test_sweep_picks_best_tick_with_monkeypatch(monkeypatch):
    calls = []

    def fake_run_trade_history(**kwargs):
        tick = kwargs['tick_size']
        # Construct PnL that prefers higher tick
        realized = float(tick)
        unreal = float(tick) / 10.0
        total = realized + unreal
        return pd.DataFrame(), {
            'realized_pnl_krw': realized,
            'unrealized_pnl_krw': unreal,
            'total_pnl_krw': total,
            'end_inventory_qty': 0.0,
        }

    # Monkeypatch the imported symbol inside backtest_grid
    monkeypatch.setattr(bg, 'run_trade_history', fake_run_trade_history)

    cfg = bg.BacktestSweepConfig(
        data_path='ignored.csv.gz',
        start_local='25-01-01:00:00',
        end_local='25-01-01:00:05',
        first_ladder_price=1000.0,
        tick_min=5,
        tick_max=8,
    )

    results = bg.sweep_ticks(cfg)
    assert not results.empty
    # Best should be highest tick (8)
    assert int(results.iloc[0]['tick']) == 8
    # Ensure pnl_rw matches stubbed total
    assert results.iloc[0]['pnl_rw'] == pytest.approx(8 + 0.8)


def test_integration_single_tick_with_temp_data(tmp_path):
    # Build a simple one-candle DataFrame that touches ladder (BUY) and target (SELL)
    df = pd.DataFrame([
        {
            'timestamp_utc': '2025-01-01T00:00:00',
            'open': 110.0,
            'low': 100.0,
            'high': 120.0,
            'close': 115.0,
            'volume': 0.0,
            'quote_volume': 0.0,
        }
    ])

    out_file = tmp_path / 'mini.csv.gz'
    df.to_csv(out_file, index=False, compression='gzip')

    cfg = bg.BacktestSweepConfig(
        data_path=str(out_file),
        start_local='25-01-01:09:00',  # KST
        end_local='25-01-01:09:01',
        first_ladder_price=100.0,
        tick_min=20,
        tick_max=20,
        maker_fee_rate=0.0002,
        ladder_rungs=1,
        order_krw_per_buy=10000.0,
        min_order_krw=1.0,
        qty_unit=0.0001,
    )

    results = bg.sweep_ticks(cfg)
    assert len(results) == 1
    row = results.iloc[0]
    # Expect positive realized profit (120 - 100) minus small fees
    assert row['pnl_rw'] > 0

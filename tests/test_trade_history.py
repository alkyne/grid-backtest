import pandas as pd
from scripts.trade_history import run_trade_history_df


def make_df(rows):
    return pd.DataFrame(rows)


def test_buy_then_sell_fifo_and_fees():
    # Two candles where price touches 100 then 120 in same second path O->L->H->C
    df = make_df([
        {
            'timestamp_utc': '2025-01-01T00:00:00',
            'open': 110, 'low': 100, 'high': 120, 'close': 115,
        },
    ])
    # Ladder at 100, tick=20 -> BUY at 100 then SELL target 120
    hist, metrics = run_trade_history_df(
        df=df,
        start_local='25-01-01:09:00',  # KST +9
        end_local='25-01-01:09:01',
        tick_size=20,
        first_ladder_price=100,
        maker_fee=0.0002,
        ladder_rungs=1,
        order_krw=10000,
        min_order_krw=1,
        qty_unit=0.0001,
        tz='Asia/Seoul',
    )
    # Expect one BUY_FILL and one SELL_FILL in same candle second
    fills = hist[hist['event_type'].isin(['BUY_FILL', 'SELL_FILL'])]
    assert len(fills) == 2
    # PnL > 0 and realized close to (120-100)*qty - fees
    assert metrics['realized_pnl_krw'] > 0
    assert metrics['unrealized_pnl_krw'] == 0


def test_timezone_kst_alignment():
    # Candle at exact UTC second; ensure KST adds 9 hours
    df = make_df([
        {'timestamp_utc': '2025-08-11T11:22:39', 'open': 4431, 'low': 4430, 'high': 4432, 'close': 4431},
    ])
    hist, _ = run_trade_history_df(
        df=df,
        start_local='25-08-11:20:22',
        end_local='25-08-11:20:23',
        tick_size=1,
        first_ladder_price=4430,
        maker_fee=0.0002,
        ladder_rungs=1,
        order_krw=10000,
        min_order_krw=1,
        qty_unit=0.0001,
        tz='Asia/Seoul',
    )
    # The filled_at should be 2025-08-11 20:22:39 KST
    fills = hist[hist['event_type'] == 'BUY_FILL']
    assert not fills.empty
    assert fills['filled_at'].iloc[0].startswith('2025-08-11 20:22:39')

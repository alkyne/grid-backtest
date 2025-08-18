# Coinone Maker-Only Grid Strategy (Reconstructed)

> This document reconstructs the exact trading logic implemented in the `alkyne/coinone-grid` repository, suitable for automated backtesting. All rules below are taken directly from code; any open items are listed as explicit assumptions.

---

## 1) Instruments & Market Assumptions

- **Exchange surface**  
  The strategy uses an abstract exchange with methods: market info, tick bands (`range_units`), trade fees, place/cancel orders, active orders, order detail, and completed trades. :contentReference[oaicite:0]{index=0}

- **Asset class / Pair**  
  - Quote currency: **KRW**.  
  - Target/base currency: configurable (examples include **USDT**, **SOL**, **BTC**).  
  - Example default in `config.py`: `QUOTE_CCY="KRW"`, `TARGET_CCY="USDT"`. :contentReference[oaicite:1]{index=1}

- **Order type**  
  - **Limit, post-only** orders only (maker). No market orders. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

- **Fees prerequisite**  
  - Strategy **requires maker (and in some versions taker) fees to be zero**; otherwise it raises and exits. (The latest file enforces maker=0 and logs taker; the initial commit checked both.) :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## 2) Timeframe, Data & Polling

- **Time granularity**: event-driven around fills; no candle indicators used.
- **Completed trades polling**:  
  - Poll rolling window **60s** (`window_ms=60_000`), initial `from_ts=now-60s`.  
  - Each loop: fetch last fills `[last_from, now]`, then set `last_from = now - 5s` to overlap and avoid gaps.  
  - Sleep between loops = `config.INTERVAL_SEC` (e.g., 2.5s). :contentReference[oaicite:6]{index=6}

- **Market metadata fetched at start**:
  - `qty_unit`, `min_order_amount` (uses `max(exchange_min, GLOBAL_MIN_ORDER_AMOUNT)`), and **tick bands** (`range_units`).  
  - Guards: maintenance off, tradable `trade_status`, **LIMIT** supported. :contentReference[oaicite:7]{index=7}

---

## 3) Price Grid Mechanics (Tick Bands)

- **Tick definition**  
  - One **TICK** = `TICK_STEPS` × `price_unit`, where `price_unit` depends on the **price band** returned by the exchange.  
  - Example in config comments: if `price_unit=1` KRW and `TICK_STEPS=3` → ±3 KRW; if `price_unit=100` and `TICK_STEPS=3` → ±300 KRW. :contentReference[oaicite:8]{index=8}

- **Band-aware stepping & quantization**  
  - Given a `start_price` and integer `steps`, price is moved one `price_unit` at a time, **respecting band boundaries**; final price is **floored** to band unit via `quantize_down`.  
  - Quantity is floored to `qty_unit`. :contentReference[oaicite:9]{index=9}

---

## 4) Parameters (from `config.py`)

- Example defaults (adjustable):  
  - `LADDER_START_PRICE` (KRW): **1363**  
  - `LADDER_RUNG_COUNT`: **4** (number of initial BUY rungs)  
  - `ORDER_KRW_PER_BUY`: **10000** KRW per rung  
  - `TICK_STEPS`: **1**  
  - `INTERVAL_SEC`: **2.5** s  
  - `POST_ONLY_RETRY_SLEEP_SEC`: **3.0** s  
  - `POST_ONLY_MAX_RETRIES`: **3**  
  - `GLOBAL_MIN_ORDER_AMOUNT`: **5000** KRW  
  (Also includes logging, HTTP timeouts, and soft cap `MAX_CONCURRENT_ORDERS`.) :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

---

## 5) End-to-End Strategy Flow

### 5.1 Startup Sequence
1. **Fetch market meta** → set `qty_unit`, `min_order_amount = max(exchange_min, GLOBAL_MIN_ORDER_AMOUNT)`, and tick bands. Validate tradability and LIMIT support. :contentReference[oaicite:13]{index=13}  
2. **Fee constraint** → assert **maker fee = 0** (and optionally taker=0 in older version). Otherwise, raise. :contentReference[oaicite:14]{index=14}  
3. **Place initial BUY ladder** (Section 5.2).  
4. **Start background poller** for completed orders (Section 5.4). :contentReference[oaicite:15]{index=15}

### 5.2 Initial Ladder Placement (BUY only)
- Compute per-rung **amount in KRW**: `rung_amount = max(ORDER_KRW_PER_BUY, min_order_amount)`.  
- Starting at `price = quantize_price(LADDER_START_PRICE)`, place **`LADDER_RUNG_COUNT` BUY orders**, each **lower** by `i * TICK_STEPS` price-unit steps from the start.  
- For each rung:
  - `rung_price = price_after_steps(start_price, -i * tick_steps)`  
  - `qty = quantize_qty(rung_amount / rung_price)`  
  - If `qty * rung_price < min_order_amount`, **skip rung**.  
  - Place **limit post-only** via retry helper (Section 5.3). :contentReference[oaicite:16]{index=16}

### 5.3 Post-Only Placement & Retry (Maker Enforcement)
- Place **post-only LIMIT** at the computed `price, qty`.  
- If order rejected (e.g., would cross), **retry up to `POST_ONLY_MAX_RETRIES`**, sleeping `POST_ONLY_RETRY_SLEEP_SEC` between attempts.  
- After exhausting retries, **nudge** the price by **±1 step** to remain maker:
  - For **SELL**, adjust **up** by +1 step; for **BUY**, adjust **down** by −1 step.  
- Use a **new `user_order_id`** after nudging.  
- If still failing, log final error and **give up** (no further fallback). :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}

### 5.4 Event Loop: Handling Fills & Emitting Counter-Orders
- The background poller fetches **completed trades** and dispatches handlers:
  - If `t.is_ask == False` → it’s a **BUY fill** → `_handle_buy_fill`  
  - If `t.is_ask == True`  → it’s a **SELL fill** → `_handle_sell_fill` :contentReference[oaicite:19]{index=19}

#### 5.4.1 On BUY Fill
- Append a FIFO **inventory lot**: `(qty=t.qty, price=t.price, fee_krw=t.fee if fee_currency=="KRW" else 0)`.  
- **Emit a SELL** one TICK above the fill **with the filled quantity**, subject to min-notional logic below. :contentReference[oaicite:20]{index=20}

#### 5.4.2 On SELL Fill
- **Realize PnL** by matching against FIFO inventory lots: for each consumed portion, compute realized profit in **KRW** (includes proportional buy/sell fees in KRW), sum across portions, and update `cumulative_profit_krw` (thread-safe). :contentReference[oaicite:21]{index=21} :contentReference[oaicite:22]{index=22}  
- **Emit a BUY** one TICK below the sell price **with the filled quantity**, subject to min-notional logic below. :contentReference[oaicite:23]{index=23}

#### 5.4.3 Counter-Order Emission (Min-Notional Accumulation)
- To satisfy `min_order_amount`, fills are **bucketed by target price**. For a needed counter-order side `S`:
  - Compute `target_price = price_after_steps(ref_price, +tick_steps)` if placing **SELL**; else `ref_price - tick_steps` for **BUY**.  
  - Accumulate `filled_qty` in a map keyed by `target_price` (separate maps for BUY/SELL).  
  - When `bucket_qty * target_price ≥ min_order_amount`, quantize qty, re-check notional, and **place the order** via maker-only helper (Section 5.3).  
  - After placement, reset that bucket to 0. :contentReference[oaicite:24]{index=24}

---

## 6) Entry & Exit Rules (Operational Definition)

- **Entries**  
  - **Initial entries** are **BUY limit, post-only** orders forming a **downward ladder** from `LADDER_START_PRICE` with spacing `TICK_STEPS × price_unit`. :contentReference[oaicite:25]{index=25}  
  - **Reactive entries** after a SELL fill: place a **BUY** at **`ref_price - 1*TICK`**, bucketed until min notional is met. :contentReference[oaicite:26]{index=26} :contentReference[oaicite:27]{index=27}

- **Exits**  
  - After each **BUY fill**, place a **SELL** at **`ref_price + 1*TICK`**, bucketed until min notional is met. Realized PnL is booked only on SELL fills via FIFO. :contentReference[oaicite:28]{index=28} :contentReference[oaicite:29]{index=29}

- **Spacing**  
  - Exactly **1 grid step** between a fill and its counter-order; the magnitude of one step equals `TICK_STEPS × price_unit` at the relevant band. :contentReference[oaicite:30]{index=30} :contentReference[oaicite:31]{index=31}

---

## 7) Position Sizing

- **Initial ladder**: each rung’s notional KRW = `max(ORDER_KRW_PER_BUY, min_order_amount)`; quantity = floored `(rung_amount / rung_price)`. :contentReference[oaicite:32]{index=32}

- **Reactive orders**: **size equals the sum of filled quantities** accumulated for that exact `target_price` until `notional ≥ min_order_amount`, then floored to `qty_unit` before placement. :contentReference[oaicite:33]{index=33}

---

## 8) Risk Controls & Operational Constraints

- **Maker-only enforcement** with retry & **±1 step** nudge after `POST_ONLY_MAX_RETRIES`. :contentReference[oaicite:34]{index=34}
- **Minimum notional guard**: skip initial rung or withhold reactive order until `min_order_amount` satisfied. :contentReference[oaicite:35]{index=35}
- **Global minimum notional**: `min_order_amount = max(exchange_min, GLOBAL_MIN_ORDER_AMOUNT)`. :contentReference[oaicite:36]{index=36}
- **Tradeability guards**: maintenance off; `trade_status` supports both sides; exchange supports **LIMIT**. :contentReference[oaicite:37]{index=37}
- **Concurrency cap (soft)**: `MAX_CONCURRENT_ORDERS` present in config (used as a soft limit in design; not enforced in the shown flow). :contentReference[oaicite:38]{index=38}
- **Stop loss / take profit**: **None implemented** beyond the intrinsic ±1 TICK grid reversion and FIFO realization on SELL. (No cancels/TTLs/trailing logic present in shown code.)

---

## 9) Exact Entry/Exit Criteria (LLM-Ready Rules)

- **Grid step**: `tick_unit(price) = price_unit(price)` from band; `TICK = TICK_STEPS * tick_unit`.
- **On startup**:
  1. Ensure maker fee = 0; ensure market tradable & LIMIT supported. :contentReference[oaicite:39]{index=39}  
  2. For `i = 0..LADDER_RUNG_COUNT-1`:
     - `rung_price = price_after_steps(LADDER_START_PRICE, -i * TICK_STEPS)`  
     - `rung_qty = floor_to(qty_unit, max(ORDER_KRW_PER_BUY, min_order_amount) / rung_price)`  
     - If `rung_qty * rung_price ≥ min_order_amount`: **place BUY limit post-only** at `rung_price` with retries+nudge. :contentReference[oaicite:40]{index=40} :contentReference[oaicite:41]{index=41}

- **On each completed trade** (polled loop):
  - If **BUY fill** at price `p` with qty `q`:
    - Append FIFO lot `(q, p, fee_krw)`;  
    - Target **SELL** price `p' = price_after_steps(p, +TICK_STEPS)`;  
    - Accumulate `q` into `sell_bucket[p']`;  
    - When `sell_bucket[p'] * p' ≥ min_order_amount`: place **SELL limit post-only** at `p'` with size `floor_to(qty_unit, sell_bucket[p'])`, then reset that bucket. :contentReference[oaicite:42]{index=42} :contentReference[oaicite:43]{index=43}
  - If **SELL fill** at price `p` with qty `q`:
    - **Realize PnL** vs FIFO inventory (pro-rata fees); update cumulative PnL;  
    - Target **BUY** price `p' = price_after_steps(p, -TICK_STEPS)`;  
    - Accumulate `q` into `buy_bucket[p']`;  
    - When `buy_bucket[p'] * p' ≥ min_order_amount`: place **BUY limit post-only** at `p'` with size `floor_to(qty_unit, buy_bucket[p'])`, then reset that bucket. :contentReference[oaicite:44]{index=44} :contentReference[oaicite:45]{index=45}

- **Order placement helper (used everywhere)**:
  - Try place **post-only**; on failure retry `POST_ONLY_MAX_RETRIES` with sleeps; then **nudge** by ±1 step (SELL:+1, BUY:−1); new `user_order_id`; try once more; if still failure, give up. :contentReference[oaicite:46]{index=46}

---

## 10) PnL Accounting

- **When realized**: only on **SELL fills**.  
- **Method**: **FIFO** matching against stored BUY lots; pro-rata allocation of **KRW fees** both on buy and sell; profit per matched slice = `(sell_price - buy_price) * qty - (buy_fee_krw_part + sell_fee_krw_part)`.  
- **Aggregation**: sum slices → `pnl`; add to `cumulative_profit_krw`. :contentReference[oaicite:47]{index=47} :contentReference[oaicite:48]{index=48}

---

## 11) Assumptions & Open Questions

- **Fees**: Code path enforces **maker fee = 0**; some versions also assert taker=0. If exchange introduces maker fees, strategy as-is will **exit**. :contentReference[oaicite:49]{index=49} :contentReference[oaicite:50]{index=50}
- **Inventory / exposure cap**: `MAX_CONCURRENT_ORDERS` exists but not enforced in the core loop; no explicit cap on inventory or KRW usage is shown. :contentReference[oaicite:51]{index=51}
- **Order lifecycle**: No periodic cancel/replace of **stale** or **unfilled** ladder orders beyond maker-price nudge on placement. (No TTL in code.)
- **Downtrend handling**: Strategy continues accumulating via BUYs; no **stop-loss** or **grid rebalance** beyond ±1*TICK counter-orders.

---

## 12) Backtest Prerequisites

- **Trades tape**: Simulate exchange **completed trades** with fields used in code: `is_ask`, `price`, `qty`, `fee`, `fee_currency`, `timestamp`. :contentReference[oaicite:52]{index=52}
- **Band model**: Provide `range_units` bands mapping `price → price_unit` to reproduce `price_after_steps` and quantization. :contentReference[oaicite:53]{index=53} :contentReference[oaicite:54]{index=54}
- **Min order**: Provide `exchange_min_order_amount` and `qty_unit`. :contentReference[oaicite:55]{index=55}
- **Config knobs**: `LADDER_START_PRICE`, `LADDER_RUNG_COUNT`, `ORDER_KRW_PER_BUY`, `TICK_STEPS`, `INTERVAL_SEC`, `POST_ONLY_RETRY_SLEEP_SEC`, `POST_ONLY_MAX_RETRIES`, `GLOBAL_MIN_ORDER_AMOUNT`. :contentReference[oaicite:56]{index=56} :contentReference[oaicite:57]{index=57}

---

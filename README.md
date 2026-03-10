# Kalman Pair Trading — Dynamic Hedge Ratio 

This repository implements an end-to-end **pairs trading** pipeline using a **Kalman filter** to estimate a **time-varying hedge ratio** \(\beta_t\). The overall flow is:

1) clean tick-level trades  
2) aggregate into bar series (VWAP)  
3) align pair series under configurable merge/fill policies  
4) estimate \(\beta_t\) via a scalar Kalman filter  
5) compute a tradable series (posterior spread or innovation) and standardize into a z-score  
6) backtest a threshold mean-reversion strategy across all pair permutations  

This codebase was refactored from a notebook-style submission into a small library + script entrypoints.

---

## Repository layout

```
data/
  A.csv
  B.csv
  C.csv

experiments/
  configs.py     # CONFIGS: merge/fill policy variants (DataCleaningConfig)
  params.py      # PARAMS: model + signal parameters (Params)
  __init__.py

scripts/
  run.py         # run_one: run 1 pair under 1 cfg + 1 prm end-to-end
  run_all.py     # run_all: run permutations (currently hardcoded to 1 cfg + 1 prm for testing)

src/
  kalman/
    clean.py         # ingest/clean ticks; build aligned pair series (merge/fill/drop policies)
    bars.py          # bar aggregation (VWAP) + fill helpers
    xmin.py          # choose_x: bar frequency selection based on overlap/coverage
    features.py      # build_features: beta0, Q, R estimation helpers
    kalman_filter.py # scalar Kalman filter estimating beta_t
    strategy.py      # signal selection, z-score, state-machine trading, backtest
    plots.py         # plotting helpers
    stat_tests.py    # ADF/Johansen/ACF diagnostics (if used)
    __init__.py

results/ 
  Different Pickle files for different configuration
  use ./notebooks/eda.ipynb to load and read the pickle file for a single config and params
```

---

## Data expectations

Place the provided tick files under:

```
data/A.csv
data/B.csv
data/C.csv
```

Minimum expected columns:
- `exg_time` (timestamp)
- `trade_price`
- `trade_qty`

---

## Running

### Current behavior (testing mode)
At the moment, `scripts/run_all.py` is used in **testing mode**: it is typically hardcoded to run **one** `cfg` and **one** `prm` (but loops across all pair permutations).

Run from the project root:

```bash
python -m scripts.run_all
```

This evaluates all 6 directed permutations:
- B vs A, A vs B  
- C vs A, A vs C  
- C vs B, B vs C  

---

## Configuration system

### Data cleaning / merge policy (`cfg`)
Configs are defined in `experiments/configs.py` as `DataCleaningConfig` and stored in `CONFIGS`. :contentReference[oaicite:2]{index=2}

Each config controls:
- `resample_method`: currently `"vwap"`
- `pre_merge_fill`: fill policy applied to each leg before merge (e.g. `none`, `ffill`)
- `post_merge_fill`: fill policy applied after merge (e.g. `none`, `ffill`)
- `dropna_after_merge`: whether to drop rows with missing values after merge/fill
- optional fill limits: `pre_merge_limit`, `post_merge_limit`

Available configs: :contentReference[oaicite:3]{index=3}  
- `S0_strict_no_fill`  
  Strict baseline: no fill; drop rows missing either leg.
- `S1_post_ffill_unlimited`  
  Common baseline: fill after merge (unlimited), drop remaining NaNs.
- `S2_post_ffill_L1`  
  Conservative: fill only 1-bar gaps after merge.
- `S3_pre_ffill_unlimited_only`  
  Fill each leg pre-merge (unlimited), no post-fill.
- `S4_pre_ffill_L1_only`  
  Fill each leg pre-merge (1-bar), no post-fill.
- `S5_pre_and_post_ffill_unlimited`  
  “Most filled”: pre-fill + post-fill (both unlimited).

### Model + strategy parameters (`prm`)
Params are defined in `experiments/params.py` as `Params` and stored in `PARAMS`. :contentReference[oaicite:4]{index=4}

Each parameter set controls:
- Kalman/feature calibration:
  - `window_pct_R` (rolling beta window fraction used to estimate \(R\))
  - `P0` (initial beta variance)
  - `n0_beta_0` (fraction of data used to initialize \(\beta_0\))
- Signal definition:
  - `trade_by`: `"posterior_spread"` or `"innovation"` (implemented)
- Trading thresholds:
  - `z_sco_win` (rolling window length for z-score)
  - `entry_z`, `exit_z`

Available params: :contentReference[oaicite:5]{index=5}  
- `P0_baseline`  
  Default: trade by `posterior_spread`, `z_sco_win=60`, `entry_z=2.0`, `exit_z=0.5`.
- `P1_tighter_exit`  
  Same idea but tighter exits (`exit_z=0.2`). *(Note: `n0_beta_0=1.0` here uses 100% data for beta0 init.)*

---

## Core pipeline (high level)

### 1) Tick cleaning
Tick data is cleaned per asset (timestamp parsing, dropping invalid rows, etc.). During this stage the code prints “dropped counts” (WIP: export these diagnostics to structured logs).

### 2) Bar aggregation (VWAP)
Ticks are converted into VWAP bars:
\[
\mathrm{VWAP}_t = \frac{\sum_i p_i q_i}{\sum_i q_i}.
\]

### 3) Choosing bar interval \(x\)
`choose_x` selects a bar size (minutes) from a candidate set based on overlap/coverage between the two legs.

### 4) Pair series construction
Two legs are merged and filled according to the chosen `cfg` (pre-merge fill, post-merge fill, dropna policy).

### 5) Kalman filter
Model:
- Measurement: \(y_t = \beta_t x_t + \varepsilon_t,\ \varepsilon_t \sim N(0,Q)\)
- State: \(\beta_t = \beta_{t-1} + \eta_t,\ \eta_t \sim N(0,R)\)

Outputs include:
- \(\hat\beta_{t|t}\) (posterior beta)
- posterior spread: \(s_t = y_t - \hat\beta_{t|t}x_t\)
- innovation: \(e_t = y_t - \hat\beta_{t|t-1}x_t\)

### 6) Signal and trading
The traded series depends on `prm.trade_by`:
- `"posterior_spread"`: trade z-score on posterior spread
- `"innovation"`: trade z-score on innovation residual

The z-score is computed with **past-only** rolling moments (shifted), and the strategy executes using a **state machine** (`current_state`):
- `current_state == 0`: flat, can enter when \(|z| \ge entry_z\)
- `current_state != 0`: in position, evaluate exit conditions when \(|z| \le exit_z\)

This design avoids “same-bar” leakage.

---

## Plans / WIP

### Per-trade logs (future)
For each `(cfg, prm, pair)`:
- export a `trades.csv` containing entry/exit timestamps, side, z-score at entry/exit, quantities/notional, and PnL per trade.

### Scaling beyond testing mode (future)
Move from “hardcode 1 cfg + 1 prm” to looping across all `CONFIGS × PARAMS`, with comparable result artifacts per run.

### Specific data quality monitoring 
Monitor how much data is lost in resample, cleaning, and combining series. 
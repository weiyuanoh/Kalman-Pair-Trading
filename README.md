# Kalman Pair Trading

> **Data note:** The input data used in this project is treated as externally provided and unverified. The repository focuses on modelling, filtering, signal construction, and backtesting rather than validating the provenance or completeness of the raw data.

This repository implements a pair-trading research pipeline built around a random-walk Kalman filter for a dynamic hedge ratio. It also includes an ECM-KF extension that connects the filtering framework to an error-correction, cointegration-motivated view of the spread.

The core workflow is:

1. Clean tick-level trade data.
2. Aggregate trades into VWAP bars.
3. Align two assets under configurable merge and fill policies.
4. Estimate a time-varying hedge ratio with a scalar Kalman filter.
5. Build a tradable posterior spread or innovation series.
6. Standardize the signal with past-only rolling z-scores.
7. Backtest a threshold-based mean-reversion state machine across directed pairs.

## Repository Layout

```text
125981811.pdf       # Retained reference/source document for the project context
data.zip            # External raw data bundle containing A.csv, B.csv, and C.csv
extension.ipynb     # Supplementary ETF-based extension experiment
engine.py           # Backward-compatible wrapper for kalman.engine
pyproject.toml      # Editable package configuration
requirements.txt    # Python dependency pins

data/
  A.csv             # Extracted tick data for asset A
  B.csv             # Extracted tick data for asset B
  C.csv             # Extracted tick data for asset C

experiments/
  configs.py     # DataCleaningConfig variants for merge and fill policy
  params.py      # Params variants for model, signal, and strategy settings

scripts/
  run.py         # Run one directed pair under one config and parameter set
  run_all.py     # Run all directed pair permutations and save result pickles

src/
  kalman/
    bars.py          # VWAP aggregation and fill helpers
    clean.py         # Tick cleaning and aligned pair construction
    engine.py        # Minimal portfolio engine and analytics helpers
    features.py      # Initial beta and Q/R calibration helpers
    kalman_filter.py # Scalar Kalman filter for beta_t
    plots.py         # Plotting helpers
    stat_tests.py    # ADF, Johansen, and ACF diagnostics
    strategy.py      # Signal construction, backtest, and portfolio analytics
    xmin.py          # Bar interval selection from overlap and coverage

notebooks/
  01_original_tick_kf_workflow.ipynb
  02_random_walk_kf_experiment.ipynb
  03_random_walk_kf_refined_experiment.ipynb
  04_yfinance_random_walk_kf_variant.ipynb
  05_ecm_guided_kf_extension.ipynb
  eda.ipynb          # Exploratory inspection of result artifacts

results/
  *.pkl              # Saved experiment outputs
```

The numbered notebooks preserve the development path from the original tick-data Kalman filter workflow through refined random-walk KF experiments and the ECM-guided KF extension. Ignored local notebook folders are not part of the public project.

## Data Expectations

Place tick files under:

```text
data/A.csv
data/B.csv
data/C.csv
```

The repository also includes `data.zip`, which contains the same three raw CSV files as an external data bundle. The code reads the extracted files in `data/`.

Minimum expected columns:

- `exg_time`: trade timestamp
- `trade_price`: executed price
- `trade_qty`: executed quantity

## Running Experiments

Install the package in editable mode from your preferred Python environment, then run from the project root:

```bash
pip install -r requirements.txt
pip install -e .
python -m scripts.run_all
```

By default, this runs config `S1_post_ffill_unlimited` with parameter set `P0_baseline` across the six directed pair permutations:

- `B_vs_A`
- `A_vs_B`
- `C_vs_A`
- `A_vs_C`
- `C_vs_B`
- `B_vs_C`

You can choose a different config or parameter set:

```bash
python -m scripts.run_all --cfg S0_strict_no_fill --prm P2_trade_innovation
```

The run writes a timestamped pickle to `results/` containing equity curves, portfolio statistics, config metadata, parameter metadata, and diagnostics captured during cleaning and resampling.

## Model

The baseline model treats the hedge ratio as a random walk:

```text
y_t = beta_t x_t + epsilon_t
beta_t = beta_{t-1} + eta_t
```

The filter estimates `beta_t` recursively and produces:

- `beta_hat`: posterior hedge ratio estimate
- `posterior_spread`: `y_t - beta_hat_t * x_t`
- `innovation`: one-step prediction residual

The strategy can trade either the posterior spread or the innovation residual, depending on the selected parameter set.

## Signal And Backtest

Signals are converted into z-scores using shifted rolling moments so each decision only uses information available before the current bar. The backtest then applies a simple state machine:

- Flat state: enter when `abs(z) >= entry_z`.
- Long or short state: exit when `abs(z) <= exit_z`.

This keeps entry and exit behavior explicit and avoids same-bar lookahead in the z-score calculation.

## Configuration

Cleaning and merge policies live in `experiments/configs.py`.

Available config families include:

- Strict no-fill alignment.
- Post-merge forward fill, unlimited or one-bar limited.
- Pre-merge forward fill, unlimited or one-bar limited.
- Combined pre- and post-merge forward fill.

Model and strategy parameters live in `experiments/params.py`.

Parameter variants cover:

- Posterior spread versus innovation signal source.
- Z-score window length.
- Entry and exit thresholds.
- Initial state uncertainty.
- Burn-in fraction used to initialize beta and calibration inputs.

## ECM-KF Extension

The ECM-KF extension explores an error-correction interpretation of the pair relationship. The motivation is to connect the dynamic hedge ratio framework to cointegration-style mean reversion, where deviations from the long-run relation inform short-run adjustment. This extension is kept alongside the original random-walk Kalman filter work as a related modelling path.

## Project Scope

This repository is focused on research code for filtering, signal construction, and backtest mechanics. It does not claim production execution readiness, transaction cost completeness, or independent validation of the raw market data.

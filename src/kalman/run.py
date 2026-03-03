from kalman.clean import prepare_data  # <- you implement: cfg branching lives here
from kalman.features import build_features     # <- your existing
from kalman.kalman import kalman_beta_filter   # <- your existing
from kalman.strategy import generate_signals   # <- your existing
from kalman.strategy import backtest             # <- if engine.py stays in root, see note below


def run_one(pair, cfg):
    """
    Run 1 pair under 1 config end-to-end.
    Return metrics dict.
    """

    # 1) Load raw data for A and B
    df_a, df_b = load_pair(pair, cfg)

    # 2) Make bars / resample / aggregate (if needed)
    bars_a = make_bars(df_a, cfg, side="A")
    bars_b = make_bars(df_b, cfg, side="B")

    # 3) Align + apply config-dependent cleaning rules (THIS is where variants differ)
    df = prepare_pair(bars_a, bars_b, cfg)

    # 4) Build features needed by Kalman + strategy
    df = build_features(df, cfg)

    # 5) Run Kalman filter (adds hedge ratio/state/innovations/etc)
    df = run_kalman(df, cfg)

    # 6) Convert to trading signals/positions
    df = make_signals(df, cfg)

    # 7) Backtest to get pnl + metrics
    metrics = backtest(df, cfg)

    # Ensure config name/pair name are included
    metrics["pair"] = str(pair)
    metrics["config"] = cfg.get("name", "unnamed")

    return metrics
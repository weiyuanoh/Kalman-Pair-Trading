import clean 
from kalman.features import build_features     # <- your existing
from kalman.kalman import kalman_beta_filter   # <- your existing
from kalman.xmin import choose_x
from kalman.strategy import generate_signals   # <- your existing
from kalman.strategy import TradingEngine           # <- if engine.py stays in root, see note below


def run_one(cfg, indep_var = "A", dep_var = "B"):
    """
    Run 1 pair under 1 config end-to-end.
    Return metrics dict.
    """
    # 1. load trade level data 
    X, Y = clean.get_data(indep_var, dep_var)
    
    # 2. clean trade level data 
    X_clean = clean.data_cleaning_single_asset(X, cfg)
    Y_clean = clean.data_cleaning_single_asset(Y, cfg)

    # 3. choose x min to resample to 
    x_min = choose_x()

    # 4. build pair series 
    X_algined, Y_aligned, XY = clean.build_pair_series(X_clean, 
                                                      Y_clean, 
                                                       x )

    # 3) Align + apply config-dependent cleaning rules (THIS is where variants differ)
    df = build_pair_series(bars_a, bars_b, cfg)

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
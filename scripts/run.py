from __future__ import annotations

import math
from typing import Any, Dict

from kalman.clean import build_pair_series, data_cleaning_single_asset, get_data
from kalman.features import build_features
from kalman.kalman_filter import kalman_beta_filter
from kalman.strategy import backtest, make_signal
from kalman.xmin import choose_x


def run_one(
    cfg,
    prm,
    indep_var="A",
    dep_var="B",
    *,
    coms_bps: float = 0.0,
    return_trades: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one directed pair under one cleaning config and parameter set."""
    X_raw, Y_raw = get_data(indep_var, dep_var)

    X_clean, X_clean_diag = data_cleaning_single_asset(X_raw, return_diag=True, verbose=verbose)
    Y_clean, Y_clean_diag = data_cleaning_single_asset(Y_raw, return_diag=True, verbose=verbose)

    x_min, x_diag = choose_x(X_clean, Y_clean, (5, 15, 45, 90), 0.8, cfg.resample_method)
    X_aligned, Y_aligned, XY = build_pair_series(X_clean, Y_clean, x_min, cfg, indep_var, dep_var)

    beta_0, Q_est, R_est = build_features(X_aligned, Y_aligned, prm)

    if verbose:
        print("beta0:", beta_0)
        print("Q:", Q_est)
        print("R:", R_est)

    n0_abs = math.ceil(prm.n0_beta_0 * len(XY))
    XY_trade = XY.iloc[n0_abs:].copy()
    X_trade = X_aligned.iloc[n0_abs:]
    Y_trade = Y_aligned.iloc[n0_abs:]

    beta_hat, P, post_spread, diag = kalman_beta_filter(
        x=X_trade,
        y=Y_trade,
        Q=Q_est,
        R=R_est,
        prm=prm,
        beta0=beta_0,
    )
    XY_trade["beta_hat"] = beta_hat
    XY_trade["posterior_spread"] = post_spread
    XY_trade["innov"] = diag["resid"]

    if verbose:
        realized_burn_in = (len(XY) - len(XY_trade)) / len(XY)
        print("burn-in pct error:", realized_burn_in - prm.n0_beta_0)

    XY_trade = make_signal(prm, XY_trade)
    equity = backtest(
        XY_trade=XY_trade,
        coms_bps=coms_bps,
        prm=prm,
        indep_var=indep_var,
        dep_var=dep_var,
        verbose=verbose,
    )

    meta: Dict[str, Any] = {
        "indep_var": indep_var,
        "dep_var": dep_var,
        "x_min": x_min,
        "n_bars_total": int(len(XY)),
        "n_bars_trade": int(len(XY_trade)),
        "n0_pct": prm.n0_beta_0,
        "n0_abs": int(n0_abs),
        "beta0": float(beta_0),
        "Q": float(Q_est),
        "R": float(R_est),
        "diagnostics": {
            "x_clean_diag": X_clean_diag,
            "y_clean_diag": Y_clean_diag,
            "x_resample_diag": x_diag,
            "pair_build_diag": {},
        },
    }

    if verbose:
        print("[run_one] meta:", meta)

    return {
        "equity": equity,
        "trades": None if not return_trades else [],
        "meta": meta,
        "xy_trade": XY_trade if verbose else None,
    }


if __name__ == "__main__":
    from experiments.configs import CONFIGS
    from experiments.params import PARAMS

    cfg = CONFIGS["S1_post_ffill_unlimited"]
    prm = PARAMS["P0_baseline"]

    result = run_one(cfg, prm, indep_var="A", dep_var="B", verbose=True)
    print(result["equity"].tail())

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import math
from kalman.clean import get_data, data_cleaning_single_asset, build_pair_series
from kalman.xmin import choose_x
from kalman.features import build_features
from kalman.strategy import make_signal, backtest
from kalman.kalman_filter import kalman_beta_filter



def run_one(cfg, prm, indep_var = "A", dep_var = "B",*,
    coms_bps: float = 0.0,
    return_trades: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run 1 pair under 1 config end-to-end.
    Return metrics dict.
    """
    # 1. load trade level data 
    X_raw, Y_raw= get_data(indep_var, dep_var)
    
    # 2. clean trade level data 
    X_clean = data_cleaning_single_asset(X_raw, cfg)
    Y_clean = data_cleaning_single_asset(Y_raw, cfg)

    # 3. choose x min to resample to 
    x_min, x_diag = choose_x(X_clean, Y_clean, (5,15,45,90), 0.8, cfg.resample_method)

    # 4. build pair series 
    X_aligned, Y_aligned, XY = build_pair_series(X_clean, 
                                                       Y_clean, 
                                                         x_min, 
                                                           cfg,
                                                        indep_var, 
                                                        dep_var)

    # 5. build kalman feature Q and R 
    beta_0, Q_est , R_est = build_features(X_aligned, Y_aligned, prm)
    print("beta 0:", beta_0)
    print("Q:", Q_est)
    print("R:", R_est)

    # 6. split data to be traded on
    n0_abs  = math.ceil(prm.n0_beta_0 * len(XY))
    XY_trade = XY.iloc[n0_abs:].copy()
    X_trade  = X_aligned.iloc[n0_abs:]
    Y_trade  = Y_aligned.iloc[n0_abs:]

    # 6. run kalman - using split data that is not seen after computing Q and R 
    beta_hat, P, post_spread, diag = kalman_beta_filter(
        x=X_trade,
        y=Y_trade,
        Q=Q_est,
        R=R_est,
        prm=prm,
        beta0=beta_0
        )
    XY_trade["beta_hat"] = beta_hat
    XY_trade["posterior_spread"] = post_spread
    XY_trade["innov"] = diag["resid"] 

    # check 
    diff = len(XY) - len(XY_trade) 
    print("IS/OS data check:", diff/len(XY) - prm.n0_beta_0) 

    # 7. make signals
    XY_trade = make_signal(prm, XY_trade)

    # 9) Backtest (future-compatible with trades)
    trades = None

    equity = backtest(
        XY_trade=XY_trade,
        coms_bps=coms_bps,
        prm=prm,
        indep_var=indep_var,
        dep_var=dep_var)
        #return_trades=return_trades

    meta: Dict[str, Any] = {
        "indep_var": indep_var,
        "dep_var": dep_var,
        "x_min": x_min,
        "n_bars_total": int(len(XY)),
        "n_bars_trade": int(len(XY_trade)),
        "n0_pct": prm.n0_beta_0,
        "n0_abs": int(n0_abs),
        "beta0": float(beta_0) if beta_0 is not None else None,
        "Q": float(Q_est) if Q_est is not None else None,
        "R": float(R_est) if R_est is not None else None,
        "x_diag": x_diag,
    }

    if verbose:
        print("[run_one] meta:", meta)

    return {
        "equity": equity,
        "trades": trades,
        "meta": meta,
        "xy_trade": XY_trade if verbose else None,
    }



if __name__ == "__main__": 
    # Local smoke test
    from experiments.configs import CONFIGS
    from experiments.params import PARAMS

    cfg_name = "S1_post_ffill_unlimited"
    prm_name = "P0_baseline"
    cfg = CONFIGS[cfg_name]
    prm = PARAMS[prm_name]

    res = run_one(cfg, prm, indep_var="A", dep_var="B", verbose=True, return_trades=False)
    print(res["equity"].tail())

   
from experiments.configs import CONFIGS
from experiments.params import PARAMS

import math
from kalman.clean import get_data, data_cleaning_single_asset, build_pair_series
from kalman.xmin import choose_x
from kalman.features import build_features
from kalman.strategy import make_signal, backtest
from kalman.kalman_filter import kalman_beta_filter



def run_one(cfg, prm, indep_var = "A", dep_var = "B"):
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

    # 7) Backtest to get pnl + metrics
    equity = backtest(
        XY_trade=XY_trade,
        coms_bps=0,
        prm = prm, 
        indep_var=indep_var,
        dep_var=dep_var
    )

    return equity




if __name__ == "__main__": 
    print("RUN.PY MAIN START")
    cfg = CONFIGS["S1_post_ffill_unlimited"]
    prm = PARAMS["P0_baseline"]
    print(run_one(cfg, prm, indep_var="A", dep_var="B"))
   
"""Scalar Kalman filter for a dynamic hedge ratio."""

import numpy as np
import pandas as pd

from kalman.features import init_beta0_ols_no_intercept


def kalman_beta_filter(x, y, Q, R, prm, beta0=None):
    """
    Estimate a time-varying hedge ratio beta_t with a scalar Kalman filter.

    Model:
        y_t = beta_t * x_t + eps_t
        beta_t = beta_{t-1} + eta_t
    """
    P0 = prm.P0
    n0_beta_0 = prm.n0_beta_0
    x = x.astype(float)
    y = y.astype(float)

    if not x.index.equals(y.index):
        raise ValueError("x and y must be aligned on the same index.")
    if not np.isfinite(Q) or not np.isfinite(R) or not np.isfinite(P0):
        raise ValueError("Require finite Q, R, and P0.")
    if Q <= 0 or R < 0 or P0 <= 0:
        raise ValueError("Require Q > 0, R >= 0, and P0 > 0.")
    if not 0 < n0_beta_0 <= 1:
        raise ValueError("n0_beta_0 must be in the interval (0, 1].")

    beta = init_beta0_ols_no_intercept(x, y, n0_beta_0) if beta0 is None else float(beta0)
    P = float(P0)

    n = len(x)
    beta_prior = np.empty(n)
    P_prior = np.empty(n)
    beta_post = np.empty(n)
    P_post = np.empty(n)
    K_list = np.empty(n)
    resid = np.empty(n)
    S_list = np.empty(n)
    spread_post = np.empty(n)

    for i in range(n):
        xt = x.iat[i]
        yt = y.iat[i]

        b_prior = beta
        Pp = P + R

        e = yt - xt * b_prior
        S = (xt * xt) * Pp + Q
        K = (Pp * xt) / S

        b_post = b_prior + K * e
        Pn = (1.0 - K * xt) * Pp

        beta_prior[i] = b_prior
        P_prior[i] = Pp
        resid[i] = e
        S_list[i] = S
        K_list[i] = K
        beta_post[i] = b_post
        P_post[i] = Pn
        spread_post[i] = yt - xt * b_post

        beta = b_post
        P = Pn

    idx = x.index
    return (
        pd.Series(beta_post, index=idx, name="beta_hat"),
        pd.Series(P_post, index=idx, name="P"),
        pd.Series(spread_post, index=idx, name="spread"),
        {
            "beta_prior": pd.Series(beta_prior, index=idx, name="beta_prior"),
            "P_prior": pd.Series(P_prior, index=idx, name="P_prior"),
            "K": pd.Series(K_list, index=idx, name="K"),
            "resid": pd.Series(resid, index=idx, name="resid"),
            "S": pd.Series(S_list, index=idx, name="S"),
        },
    )

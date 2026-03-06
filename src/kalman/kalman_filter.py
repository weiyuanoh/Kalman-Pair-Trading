"""
Library for all Kalman related functions 
1. Initialising OLS with a certain percentage of total observed bars. Denoted by 'n0_beta_0' 
2. Kalman beta filter: Algorithm to update beta based on estimated values of Q and R 
3. Functions to estimate Q (variance of residuals), and R (variance of betas)

"""

import pandas as pd
import numpy as np 
import math 
import statsmodels.api as sm 
import matplotlib.pyplot as plt

def init_beta0_ols_no_intercept(x, y, n0_beta_0 ):
    """
    Initialize beta0 via OLS slope with no intercept on the first n0 points.
    x, y: aligned pd.Series

    Inputs: 
        x: independent asset 
        y: dependent variable asset 
        n0_beta_0: percentage of length of dataframe use to intialise beta_0 - correction from a fixed absolute value., 
    """
    if n0_beta_0 > 1 :
        raise Exception("Error: n0_beta_0 is a percentage length of data")
    n0_abs = math.floor(n0_beta_0 * len(x))
    x0 = x.iloc[:n0_abs].astype(float).values
    y0 = y.iloc[:n0_abs].astype(float).values

    denom = np.dot(x0, x0)
    if denom == 0:
        raise ValueError("init_beta0: x has zero energy in the init window.")
    beta0 = np.dot(x0, y0) / denom
    return float(beta0)


def init_beta0_ols_with_intercept(x, y, n0_beta_0 ):
  
    if n0_beta_0 > 1 :
        raise Exception("Error: n0_beta_0 is a percentage length of data")
    n0_abs = math.floor(n0_beta_0 * len(x))
    X = sm.add_constant(x.iloc[:n0_abs].astype(float).values)
    Y = y.iloc[:n0_abs].astype(float).values
    res = sm.OLS(Y, X).fit()
    alpha0, beta0 = res.params
    return float(alpha0), float(beta0)

def kalman_beta_filter(x, y, Q, R, prm, beta0=None):
    """
    Scalar Kalman filter for time-varying hedge ratio beta_t in:
        y_t = beta_t * x_t + eps_t,  eps_t ~ N(0, Q)
        beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, R)

    Inputs:
      x, y: aligned pd.Series (same index), float-like
      Q: measurement noise variance (spread noise)
      R: process noise variance (beta drift)
      beta0: optional override initial beta
      P0: initial variance of beta estimate
      n0: init percentage window for beta0 if beta0 is None
      clip_beta: e.g. (0, 3) to keep hedge ratio sane (optional)

    Returns:
      beta_post: pd.Series of posterior beta estimates
      P_post: pd.Series of posterior variances
      spread_post: pd.Series of posterior spread (y - beta*x)
      diagnostics: dict of Series (beta_prior, P_prior, K, resid, S)
    """
    P0 = prm.P0
    n0_beta_0 = prm.n0_beta_0
    x = x.astype(float)
    y = y.astype(float)
    if not x.index.equals(y.index):
        raise ValueError("x and y must be aligned on the same index.")

    if Q <= 0 or R < 0 or P0 <= 0:
        raise ValueError("Require Q>0, R>=0, P0>0.")

    if n0_beta_0 > 1 :
        raise Exception("Error: n0_beta_0 is a percentage length of data")
    
    # init beta
    beta = init_beta0_ols_no_intercept(x, y, n0_beta_0) if beta0 is None else float(beta0)
    P = float(P0)

    # init storing 
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

        # prior
        b_prior = beta
        Pp = P + R

        # estimation errors 
        e = yt - xt * b_prior
        S = (xt * xt) * Pp + Q

        # Gain 
        K = (Pp * xt) / S

        # Posterior (update) 
        b_post = b_prior + K * e
        Pn = (1.0 - K * xt) * Pp

        # store
        beta_prior[i] = b_prior
        P_prior[i] = Pp
        resid[i] = e
        S_list[i] = S
        K_list[i] = K
        beta_post[i] = b_post
        P_post[i] = Pn
        spread_post[i] = yt - xt * b_post  # posterior spread

        # reassignment
        beta = b_post
        P = Pn

    idx = x.index
    out = (
        pd.Series(beta_post, index=idx, name="beta_hat"),
        pd.Series(P_post, index=idx, name="P"),
        pd.Series(spread_post, index=idx, name="spread"),
        {
            "beta_prior": pd.Series(beta_prior, index=idx, name="beta_prior"),
            "P_prior": pd.Series(P_prior, index=idx, name="P_prior"),
            "K": pd.Series(K_list, index=idx, name="K"),
            "resid": pd.Series(resid, index=idx, name="resid"),
            "S": pd.Series(S_list, index=idx, name="S"),
        }
    )
    return out
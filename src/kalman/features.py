"""
Library for Kalman features 
"""
import pandas as pd 
import numpy as np 
import math 
import matplotlib.pyplot as plt 


def init_beta0_ols_no_intercept(x, y, n0_beta_0):
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


def init_beta0_ols_with_intercept(x, y, n0_beta_0):
    import statsmodels.api as sm 
    if n0_beta_0 > 1 :
        raise Exception("Error: n0_beta_0 is a percentage length of data")
    n0_abs = math.floor(n0_beta_0 * len(x))
    X = sm.add_constant(x.iloc[:n0_abs].astype(float).values)
    Y = y.iloc[:n0_abs].astype(float).values
    res = sm.OLS(Y, X).fit()
    alpha0, beta0 = res.params
    return float(alpha0), float(beta0)

def rolling_beta_no_intercept(x, y, window_pct_R):
    '''
    Estimate variance of beta parameter R using the initial OLS parameters
    Inputs: 
     x: independent variable asset 
     y: dependent variable asset y 
     window_pct_R: pecentage of total data to be used to estimate the variance of betas. 
    
    '''

    R_abs = math.floor(window_pct_R * len(x))
    num = (x*y).rolling(R_abs).sum()
    den = (x*x).rolling(R_abs).sum()
    return (num/den).dropna()

def estimate_R_from_rolling_beta(x, y, window_pct_R):
    beta_roll = rolling_beta_no_intercept(x, y, window_pct_R)
    d_beta = beta_roll.diff()
    R_hat = float(d_beta.var(ddof=1))
    return R_hat, beta_roll


def QR(x, y, beta0, n0_beta_0, window_pct_R):
    """
    Q: variance of initial residuals using beta0
    R: small process variance (beta drift), proportional to Q
    Inputs: 
        - n0_beta_0 : init window for beta0 if beta0 is None - longer the better
        - window_pct_R: init window for R - longer the better
    """
    n0_abs = math.floor(n0_beta_0 * len(x))
    x0 = x.iloc[:n0_abs].astype(float)
    y0 = y.iloc[:n0_abs].astype(float)
    e0 = (y0 - beta0 * x0)
    Q = float(e0.var(ddof=1))
    R, beta_roll = estimate_R_from_rolling_beta(x, y, window_pct_R)
    return Q, R


def sanity_check(x, y, beta_hat, P_hat, spread, diag, Q, plot = True):
    # S identity check
    S_recon = diag["P_prior"] * (x**2) + Q
    err = (diag["S"] - S_recon).abs()
    print("max |S - (x^2 P_prior + Q)| =", float(err.max()))

    # Normalized residuals
    z = diag["resid"] / np.sqrt(diag["S"])
    print("normalized residuals z: mean =", float(z.mean()), "std =", float(z.std()))

    if plot == True:
        beta_hat.plot(title="beta_hat (posterior)")
        plt.show()
        P_hat.plot(title="P (posterior variance)")
        plt.show()
        diag["K"].plot(title="Kalman gain K_t")
        plt.show()
        
    return z   

def build_features(x, y, prm): 
    """
    Wrapper to build the features needed to compute Q, R.
    Takes in full data, cutting is done in inidividual functions. 
    """
    
    beta_0 = init_beta0_ols_no_intercept(x, y, prm.window_pct_R)
    Q, R = QR(x, y, beta_0, prm.n0_beta_0, prm.window_pct_R)

    return beta_0, Q, R 

"""Feature calibration helpers for the Kalman hedge-ratio model."""

import math

import matplotlib.pyplot as plt
import numpy as np


def _window_length(frac: float, n_obs: int, label: str) -> int:
    if not 0 < frac <= 1:
        raise ValueError(f"{label} must be in the interval (0, 1].")
    n_window = math.floor(frac * n_obs)
    if n_window < 2:
        raise ValueError(f"{label} selects fewer than two observations.")
    return n_window


def init_beta0_ols_no_intercept(x, y, n0_beta_0):
    """Initialize beta0 via an OLS slope with no intercept."""
    n0_abs = _window_length(n0_beta_0, len(x), "n0_beta_0")
    x0 = x.iloc[:n0_abs].astype(float).values
    y0 = y.iloc[:n0_abs].astype(float).values

    denom = np.dot(x0, x0)
    if denom == 0:
        raise ValueError("Cannot initialize beta: x has zero energy in the init window.")
    return float(np.dot(x0, y0) / denom)


def init_beta0_ols_with_intercept(x, y, n0_beta_0):
    import statsmodels.api as sm

    n0_abs = _window_length(n0_beta_0, len(x), "n0_beta_0")
    X = sm.add_constant(x.iloc[:n0_abs].astype(float).values)
    Y = y.iloc[:n0_abs].astype(float).values
    res = sm.OLS(Y, X).fit()
    alpha0, beta0 = res.params
    return float(alpha0), float(beta0)


def rolling_beta_no_intercept(x, y, window_pct_R):
    """Estimate rolling no-intercept hedge ratios for process variance calibration."""
    window = _window_length(window_pct_R, len(x), "window_pct_R")
    num = (x * y).rolling(window).sum()
    den = (x * x).rolling(window).sum()
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def estimate_R_from_rolling_beta(x, y, window_pct_R):
    beta_roll = rolling_beta_no_intercept(x, y, window_pct_R)
    d_beta = beta_roll.diff()
    R_hat = float(d_beta.var(ddof=1))
    if not np.isfinite(R_hat):
        raise ValueError("Unable to estimate finite process variance R from rolling beta.")
    return R_hat, beta_roll


def QR(x, y, beta0, n0_beta_0, window_pct_R):
    """Estimate measurement variance Q and beta process variance R."""
    n0_abs = _window_length(n0_beta_0, len(x), "n0_beta_0")
    x0 = x.iloc[:n0_abs].astype(float)
    y0 = y.iloc[:n0_abs].astype(float)

    residual = y0 - beta0 * x0
    Q = float(residual.var(ddof=1))
    if not np.isfinite(Q) or Q <= 0:
        raise ValueError("Unable to estimate positive finite measurement variance Q.")

    R, beta_roll = estimate_R_from_rolling_beta(x, y, window_pct_R)
    return Q, R


def sanity_check(x, y, beta_hat, P_hat, spread, diag, Q, plot=True):
    S_recon = diag["P_prior"] * (x**2) + Q
    err = (diag["S"] - S_recon).abs()
    print("max |S - (x^2 P_prior + Q)| =", float(err.max()))

    z = diag["resid"] / np.sqrt(diag["S"])
    print("normalized residuals z: mean =", float(z.mean()), "std =", float(z.std()))

    if plot:
        beta_hat.plot(title="beta_hat (posterior)")
        plt.show()
        P_hat.plot(title="P (posterior variance)")
        plt.show()
        diag["K"].plot(title="Kalman gain K_t")
        plt.show()

    return z


def build_features(x, y, prm):
    """Build beta0, Q, and R for the Kalman filter."""
    beta_0 = init_beta0_ols_no_intercept(x, y, prm.n0_beta_0)
    Q, R = QR(x, y, beta_0, prm.n0_beta_0, prm.window_pct_R)
    return beta_0, Q, R

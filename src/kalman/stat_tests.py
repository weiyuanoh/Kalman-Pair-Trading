"""Statistical diagnostics for pair-trading research."""

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def adf_summary(series: pd.Series, *, regression: str = "c", autolag: str = "AIC") -> dict:
    """Return a compact Augmented Dickey-Fuller test summary."""
    clean = series.dropna()
    if clean.empty:
        raise ValueError("ADF test requires a non-empty series.")

    stat, pvalue, used_lag, n_obs, critical_values, ic_best = adfuller(
        clean,
        regression=regression,
        autolag=autolag,
    )
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {key: float(value) for key, value in critical_values.items()},
        "ic_best": float(ic_best),
    }


def johansen_summary(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Return Johansen trace and max-eigenvalue statistics for aligned price series."""
    clean = df.dropna()
    if clean.shape[0] < 3 or clean.shape[1] < 2:
        raise ValueError("Johansen test requires at least two series and three observations.")

    result = coint_johansen(clean, det_order=det_order, k_ar_diff=k_ar_diff)
    return {
        "trace_stat": result.lr1.tolist(),
        "trace_crit": result.cvt.tolist(),
        "max_eig_stat": result.lr2.tolist(),
        "max_eig_crit": result.cvm.tolist(),
        "eigenvectors": result.evec.tolist(),
    }


def plot_acf_diagnostic(series: pd.Series, *, lags: int = 40):
    """Plot autocorrelation for a cleaned series."""
    clean = series.dropna()
    if clean.empty:
        raise ValueError("ACF plot requires a non-empty series.")
    return plot_acf(clean, lags=lags)

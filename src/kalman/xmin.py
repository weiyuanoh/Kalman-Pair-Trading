"""
Explores different methods to resample data. Tick data -> x_bars. 

Criterion for choosing "x"
1. Choose x simply based on percentage overlap of X in Y. 
"""

import pandas as pd 
import numpy as np
import kalman.bars as bars 

import pandas as pd
import kalman.bars as bars

def choose_x(
    df1_clean,
    df2_clean,
    candidates=(5, 15, 45, 90),
    min_overlap_ratio=0.8,
    bar_method="vwap",
):
    """
    Choose x (minutes) for a pair using ONLY coverage/overlap.
    No fill. No pair-policy decisions.

    Returns:
      best_x (int), diagnostics (dict)
    """

    # time overlap window based on cleaned trades timestamps
    t1 = pd.to_datetime(df1_clean["exg_time"], utc=True, errors="coerce").dropna()
    t2 = pd.to_datetime(df2_clean["exg_time"], utc=True, errors="coerce").dropna()

    if len(t1) == 0 or len(t2) == 0:
        return candidates[-1], {"error": "empty timestamps after cleaning"}

    start = max(t1.min(), t2.min())
    end   = min(t1.max(), t2.max())

    diag = {"start": str(start), "end": str(end), "candidates": list(candidates)}
    best = None

    for m in candidates:
        freq = f"{m}min"

        if bar_method == "vwap":
            b1 = bars.make_vwap_bars(df1_clean, freq=freq)["vwap"]
            b2 = bars.make_vwap_bars(df2_clean, freq=freq)["vwap"]
        elif bar_method == "vol":
            # Only if you implemented it in bars.py
            b1 = bars.make_vol_bars(df1_clean, freq=freq)["price"]
            b2 = bars.make_vol_bars(df2_clean, freq=freq)["price"]
        else:
            raise ValueError(f"Unknown bar_method: {bar_method}")

        # restrict to common time window
        b1 = b1.loc[start:end]
        b2 = b2.loc[start:end]

        # overlap WITHOUT filling
        overlap_idx = b1.index.intersection(b2.index)

        overlap = int(len(overlap_idx))
        n1 = int(len(b1))
        n2 = int(len(b2))

        # ratio relative to the smaller series length (your original intention)
        denom = max(1, min(n1, n2))
        ratio = overlap / denom

        diag[m] = {
            "n1": n1,
            "n2": n2,
            "overlap": overlap,
            "ratio": ratio,
        }

        # pick first m that satisfies threshold (smallest acceptable)
        if best is None and ratio >= min_overlap_ratio:
            best = m

    if best is None:
        best = candidates[-1]  # fallback to coarsest

    diag["best"] = best
    return best, diag
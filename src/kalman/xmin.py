"""Bar-size selection based on pair overlap and coverage."""

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
    Choose the smallest candidate bar size that satisfies the overlap threshold.

    The overlap calculation is based on observed bars only; fill policy is applied
    later when constructing the pair series.
    """
    if bar_method != "vwap":
        raise ValueError(f"Unsupported bar_method: {bar_method}")

    t1 = pd.to_datetime(df1_clean["exg_time"], utc=True, errors="coerce").dropna()
    t2 = pd.to_datetime(df2_clean["exg_time"], utc=True, errors="coerce").dropna()

    if len(t1) == 0 or len(t2) == 0:
        return candidates[-1], {"error": "empty timestamps after cleaning"}

    start = max(t1.min(), t2.min())
    end = min(t1.max(), t2.max())

    diag = {"start": str(start), "end": str(end), "candidates": list(candidates)}
    best = None

    for minutes in candidates:
        freq = f"{minutes}min"
        b1 = bars.make_vwap_bars(df1_clean, freq=freq)["vwap"].loc[start:end]
        b2 = bars.make_vwap_bars(df2_clean, freq=freq)["vwap"].loc[start:end]

        overlap = int(len(b1.index.intersection(b2.index)))
        n1 = int(len(b1))
        n2 = int(len(b2))
        ratio = overlap / max(1, min(n1, n2))

        diag[minutes] = {
            "n1": n1,
            "n2": n2,
            "overlap": overlap,
            "ratio": ratio,
        }

        if best is None and ratio >= min_overlap_ratio:
            best = minutes

    diag["best"] = best or candidates[-1]
    return diag["best"], diag

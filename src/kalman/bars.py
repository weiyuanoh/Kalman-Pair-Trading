"""Resampling helpers for trade-level data."""

import pandas as pd


def _fill_series(s, method, limit):
    if method == "ffill":
        return s.ffill(limit=limit)
    if method == "bfill":
        return s.bfill(limit=limit)
    if method == "none":
        return s
    raise ValueError(f"Unknown fill method: {method}")


def _fill_df(df, method, limit):
    if method == "ffill":
        return df.ffill(limit=limit)
    if method == "bfill":
        return df.bfill(limit=limit)
    if method == "none":
        return df
    raise ValueError(f"Unknown fill method: {method}")


def make_vol_bars(*args, **kwargs):
    raise NotImplementedError("Volume bars are not implemented. Use resample_method='vwap'.")


def make_vwap_bars(
    df,
    freq="15min",
    time_col="exg_time",
    price_col="trade_price",
    qty_col="trade_qty",
):
    required = [time_col, price_col, qty_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=required).sort_values(time_col)
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]

    d["pxq"] = d[price_col] * d[qty_col]
    grouped = d.set_index(time_col).groupby(pd.Grouper(freq=freq))

    vol = grouped[qty_col].sum()
    vwap = grouped["pxq"].sum() / vol

    out = pd.DataFrame({"vwap": vwap, "vol": vol})
    return out[(out["vol"] > 0) & out["vwap"].notna()]

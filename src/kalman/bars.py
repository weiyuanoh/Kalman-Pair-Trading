"""
Library for all possible ways to resample trade level data -> bars
After choosing x, resample by 
1. Simple time bars (no Volume information incorporated) 
2. VWAP Bars: suitable for intraday trading 
3. Volatility Bars: suitable for longer horizons 
"""
import pandas as pd 
import numpy as np 
import kalman.xmin as res


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

def make_vol_bars():
    pass

def make_vwap_bars(df, freq="15min",
                   time_col="exg_time",
                   price_col="trade_price",
                   qty_col="trade_qty"):
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col)
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]

    d["pxq"] = d[price_col] * d[qty_col]
    g = d.set_index(time_col).groupby(pd.Grouper(freq=freq))

    vol = g[qty_col].sum()
    vwap = g["pxq"].sum() / vol

    out = pd.DataFrame({"vwap": vwap, "vol": vol})
    out = out[(out["vol"] > 0) & out["vwap"].notna()]
    return out

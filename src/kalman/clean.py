"""
Library for all possible ways to clean data. 
1. Checks for 
- nan values in "exg_time" 
- <0 values of "trade_qty" and "trade_price"

2. 

"""



import pandas as pd 
import numpy as np 
import kalman.resample as res
from experiments.variants import CONFIGS
from experiments.variants import DataCleaningConfig
import kalman.bars as bars 


cfg = CONFIGS["B_ffill_after_merge_then_drop"]

def get_data(): 
    A = pd.read_csv(r"data\A.csv")
    B = pd.read_csv(r"data\B.csv")
    C = pd.read_csv(r"data\C.csv")
    return A, B,C 

def data_analysis(df, name = "X"):
    # understanding data 
    print("Data Description:", "\n")
    print(df.describe())

    require = ["exg_time","trade_price","trade_qty"]
    missing = [c for c in require if c not in df.columns]
    if missing:
        return {"name": name, "error": f"missing columns: {missing}"}

    d = df.copy()
    # time stamp analysis 
    d["exg_time"] = pd.to_datetime(d["exg_time"], utc=True, errors="coerce")
    d = d.sort_values("exg_time")

    out = {}
    out["name"] = name
    out["rows"] = len(d)

    # nan values, missing values 
    out["null_exg_time"] = int(d["exg_time"].isna().sum())
    out["null_price"]    = int(d["trade_price"].isna().sum())
    out["null_qty"]      = int(d["trade_qty"].isna().sum())
    out["nonpos_price"]  = int((d["trade_price"] <= 0).sum())
    out["nonpos_qty"]    = int((d["trade_qty"] <= 0).sum())

    # time range
    out["start"] = str(d["exg_time"].min())
    out["end"]   = str(d["exg_time"].max())


    return out

def data_cleaning_single_asset(df, cfg: DataCleaningConfig):
    """
    Data cleaning function that cleans data of a single asset according to configuration rules. 
    Trade level cleaning. 
    
    Inputs: 
    - df: pd.DataFrame of one Asset 
    - cfg: configuration rules for data cleaning
    """

    d = df.copy()
    d["exg_time"] = pd.to_datetime(d["exg_time"], utc=True, errors="coerce")

    d = d.dropna(subset=["exg_time", "trade_price", "trade_qty"])
    
    # always drop non-positive trades (this is "hard cleaning")
    d = d[(d["trade_price"] > 0) & (d["trade_qty"] > 0)]

    d = d.sort_values("exg_time").reset_index(drop=True)
    return d



def build_pair_series(df1_clean, df2_clean, x_minutes, cfg, name1="X1", name2="X2"):
    """
    
    """
    freq = f"{x_minutes}min"

    # resample -> Series
    if cfg.resample_method == "vwap":
        b1 = bars.make_vwap_bars(df1_clean, freq=freq)["vwap"].rename(name1)
        b2 = bars.make_vwap_bars(df2_clean, freq=freq)["vwap"].rename(name2)
    else:
        b1 = bars.make_vol_bars(df1_clean, freq=freq)["price"].rename(name1)
        b2 = bars.make_vol_bars(df2_clean, freq=freq)["price"].rename(name2)

    # after resample, before merge 
    b1 = bars._fill_series(b1, cfg.pre_merge_fill, cfg.pre_merge_limit)
    b2 = bars._fill_series(b2, cfg.pre_merge_fill, cfg.pre_merge_limit)

    # merge (align)
    pair = pd.concat([b1, b2], axis=1)

    # Stage 3: after merge (fill pair)
    pair = bars._fill_df(pair, cfg.post_merge_fill, cfg.post_merge_limit)

    # dropna after merge 
    if cfg.dropna_after_merge:
        pair = pair.dropna(subset=[name1, name2])

    return pair[name1], pair[name2], pair

def prepare_data(cfg):
    A, B, C = get_data()
    
    # unpack configurations: 


    A_clean = data_cleaning_single_asset(A, cfg)
    B_clean = data_cleaning_single_asset(B, cfg)
    C_clean = data_cleaning_single_asset(C, cfg)

    x_AB, diag_AB = res.choose_x(A_clean, B_clean)
    x_AC, diag_AC = res.choose_x(A_clean, C_clean)
    x_BC, diag_BC = res.choose_x(B_clean, C_clean)

    xA_AB, yB_AB, AB = build_pair_series(A_clean, B_clean, x_AB, cfg, "A", "B")
    xA_AC, yC_AC, AC = build_pair_series(A_clean, C_clean, x_AC, cfg, "A", "C")
    xB_BC, yC_BC, BC = build_pair_series(B_clean, C_clean, x_BC, cfg, "B", "C")

    return {
        "A_clean": A_clean, "B_clean": B_clean, "C_clean": C_clean,
        "x_AB": x_AB, "x_AC": x_AC, "x_BC": x_BC,
        "diag_AB": diag_AB, "diag_AC": diag_AC, "diag_BC": diag_BC,
        "AB": AB, "AC": AC, "BC": BC
    }

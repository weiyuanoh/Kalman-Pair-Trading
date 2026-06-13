"""Data loading, cleaning, and pair-series construction."""

from pathlib import Path

import pandas as pd

import kalman.bars as bars
import kalman.xmin as res

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_asset_data(asset: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def get_data(indep_var: str, dep_var: str):
    return load_asset_data(indep_var), load_asset_data(dep_var)


def data_analysis(df: pd.DataFrame, name: str = "X") -> dict:
    required = ["exg_time", "trade_price", "trade_qty"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return {"name": name, "error": f"missing columns: {missing}"}

    d = df.copy()
    d["exg_time"] = pd.to_datetime(d["exg_time"], utc=True, errors="coerce")
    d = d.sort_values("exg_time")

    return {
        "name": name,
        "rows": len(d),
        "null_exg_time": int(d["exg_time"].isna().sum()),
        "null_price": int(d["trade_price"].isna().sum()),
        "null_qty": int(d["trade_qty"].isna().sum()),
        "nonpos_price": int((d["trade_price"] <= 0).sum()),
        "nonpos_qty": int((d["trade_qty"] <= 0).sum()),
        "start": str(d["exg_time"].min()),
        "end": str(d["exg_time"].max()),
    }


def data_cleaning_single_asset(df: pd.DataFrame, return_diag: bool = True, verbose: bool = False):
    """
    Clean one asset at trade level.

    Rows with missing timestamps, prices, or quantities are dropped. Non-positive
    price and quantity observations are also removed.
    """
    required = ["exg_time", "trade_price", "trade_qty"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_len = len(df)
    d = df.copy()
    d["exg_time"] = pd.to_datetime(d["exg_time"], utc=True, errors="coerce")

    d = d.dropna(subset=required)
    dropped_na = raw_len - len(d)

    before_nonpos = len(d)
    d = d[(d["trade_price"] > 0) & (d["trade_qty"] > 0)]
    dropped_nonpos = before_nonpos - len(d)

    d = d.sort_values("exg_time").reset_index(drop=True)

    diag = {
        "trade_level": {
            "raw_len": int(raw_len),
            "after_dropna_len": int(raw_len - dropped_na),
            "dropped_na": int(dropped_na),
            "after_nonpos_len": int(len(d)),
            "dropped_nonpos": int(dropped_nonpos),
            "dropped_total": int(raw_len - len(d)),
        }
    }

    if verbose:
        print("Trade-level rows dropped:", diag["trade_level"]["dropped_total"])

    return (d, diag) if return_diag else d


def build_pair_series(df1_clean, df2_clean, x_minutes, cfg, name1="X1", name2="X2"):
    freq = f"{x_minutes}min"

    if cfg.resample_method != "vwap":
        raise ValueError(f"Unsupported resample_method: {cfg.resample_method}")

    b1 = bars.make_vwap_bars(df1_clean, freq=freq)["vwap"].rename(name1)
    b2 = bars.make_vwap_bars(df2_clean, freq=freq)["vwap"].rename(name2)

    b1 = bars._fill_series(b1, cfg.pre_merge_fill, cfg.pre_merge_limit)
    b2 = bars._fill_series(b2, cfg.pre_merge_fill, cfg.pre_merge_limit)

    pair = pd.concat([b1, b2], axis=1)
    pair = bars._fill_df(pair, cfg.post_merge_fill, cfg.post_merge_limit)

    if cfg.dropna_after_merge:
        pair = pair.dropna(subset=[name1, name2])

    return pair[name1], pair[name2], pair


def prepare_data(cfg):
    A_clean = data_cleaning_single_asset(load_asset_data("A"), return_diag=False)
    B_clean = data_cleaning_single_asset(load_asset_data("B"), return_diag=False)
    C_clean = data_cleaning_single_asset(load_asset_data("C"), return_diag=False)

    x_AB, diag_AB = res.choose_x(A_clean, B_clean, bar_method=cfg.resample_method)
    x_AC, diag_AC = res.choose_x(A_clean, C_clean, bar_method=cfg.resample_method)
    x_BC, diag_BC = res.choose_x(B_clean, C_clean, bar_method=cfg.resample_method)

    xA_AB, yB_AB, AB = build_pair_series(A_clean, B_clean, x_AB, cfg, "A", "B")
    xA_AC, yC_AC, AC = build_pair_series(A_clean, C_clean, x_AC, cfg, "A", "C")
    xB_BC, yC_BC, BC = build_pair_series(B_clean, C_clean, x_BC, cfg, "B", "C")

    return {
        "A_clean": A_clean,
        "B_clean": B_clean,
        "C_clean": C_clean,
        "x_AB": x_AB,
        "x_AC": x_AC,
        "x_BC": x_BC,
        "diag_AB": diag_AB,
        "diag_AC": diag_AC,
        "diag_BC": diag_BC,
        "AB": AB,
        "AC": AC,
        "BC": BC,
        "xA_AB": xA_AB,
        "yB_AB": yB_AB,
        "xA_AC": xA_AC,
        "yC_AC": yC_AC,
        "xB_BC": xB_BC,
        "yC_BC": yC_BC,
    }

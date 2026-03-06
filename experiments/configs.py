# experiments/variants.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataCleaningConfig:
    resample_method: str # 'vwap, 'vol'
    pre_merge_fill: str 
    post_merge_fill: str
    dropna_after_merge: bool 
    pre_merge_limit: Optional[int] = None
    post_merge_limit: Optional[int] = None 
    
# Now store your variants as instances of the dataclass
CONFIGS = {
    # 1) Strict baseline: no fill anywhere, just drop rows missing either leg
    "S0_strict_no_fill": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="none",
        post_merge_fill="none",
        dropna_after_merge=True,
    ),

    # 2) Common baseline: fill after merge (unlimited), then drop remaining NaNs
    "S1_post_ffill_unlimited": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="none",
        post_merge_fill="ffill",
        post_merge_limit=None,
        dropna_after_merge=True,
    ),

    # 3) Conservative post-fill: only fill 1-bar gaps after merge
    "S2_post_ffill_L1": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="none",
        post_merge_fill="ffill",
        post_merge_limit=1,
        dropna_after_merge=True,
    ),

    # 4) Fill each leg before merge (unlimited), no post-fill
    "S3_pre_ffill_unlimited_only": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="ffill",
        pre_merge_limit=None,
        post_merge_fill="none",
        dropna_after_merge=True,
    ),

    # 5) Fill each leg before merge (limit 1), no post-fill
    "S4_pre_ffill_L1_only": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="ffill",
        pre_merge_limit=1,
        post_merge_fill="none",
        dropna_after_merge=True,
    ),

    # 6) “Most filled”: pre-fill + post-fill (both unlimited)
    "S5_pre_and_post_ffill_unlimited": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="ffill",
        pre_merge_limit=None,
        post_merge_fill="ffill",
        post_merge_limit=None,
        dropna_after_merge=True,
    ),
}
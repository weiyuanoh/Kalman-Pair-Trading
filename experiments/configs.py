from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataCleaningConfig:
    resample_method: str
    pre_merge_fill: str
    post_merge_fill: str
    dropna_after_merge: bool
    pre_merge_limit: Optional[int] = None
    post_merge_limit: Optional[int] = None


CONFIGS = {
    "S0_strict_no_fill": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="none",
        post_merge_fill="none",
        dropna_after_merge=True,
    ),
    "S1_post_ffill_unlimited": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="none",
        post_merge_fill="ffill",
        post_merge_limit=None,
        dropna_after_merge=True,
    ),
    "S2_post_ffill_L1": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="none",
        post_merge_fill="ffill",
        post_merge_limit=1,
        dropna_after_merge=True,
    ),
    "S3_pre_ffill_unlimited_only": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="ffill",
        pre_merge_limit=None,
        post_merge_fill="none",
        dropna_after_merge=True,
    ),
    "S4_pre_ffill_L1_only": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="ffill",
        pre_merge_limit=1,
        post_merge_fill="none",
        dropna_after_merge=True,
    ),
    "S5_pre_and_post_ffill_unlimited": DataCleaningConfig(
        resample_method="vwap",
        pre_merge_fill="ffill",
        pre_merge_limit=None,
        post_merge_fill="ffill",
        post_merge_limit=None,
        dropna_after_merge=True,
    ),
}

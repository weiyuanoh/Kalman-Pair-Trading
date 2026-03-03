# experiments/variants.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataCleaningConfig:
    resample_method: str # 'vwap, 'vol'
    pre_merge_fill: str 
    post_merge_fill: str
    dropna_after_merge: bool 
    post_merge_limit: Optional[int] = None 
    
# Now store your variants as instances of the dataclass
CONFIGS = {
  "A_no_fill_drop": DataCleaningConfig(
      resample_method="vwap",
      pre_merge_fill="none",
      post_merge_fill="none",
      dropna_after_merge=True
  ),
  "B_post_ffill_drop": DataCleaningConfig(
      resample_method="vwap",
      pre_merge_fill="none",
      post_merge_fill="ffill",
      post_merge_limit=None,
      dropna_after_merge=True
  ),
  "C_post_ffill1_drop": DataCleaningConfig(
      resample_method="vwap",
      pre_merge_fill="none",
      post_merge_fill="ffill",
      post_merge_limit=1,
      dropna_after_merge=True
  ),
}
from dataclasses import dataclass
from typing import Optional
from typing import Dict

@dataclass
class Params: 
    name: str
    window_pct_R: float
    P0: float
    n0_beta_0: float
    trade_by: str # "innovation", "posterior spread"
    z_sco_win: int
    entry_z: float
    exit_z: float

PARAMS: Dict[str, Params] = {
    "P0_baseline": Params(
        name="P0_baseline",
        window_pct_R=0.02,
        P0= 1.0,
        n0_beta_0=0.0375,
        trade_by= "posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    "P1_tighter_exit": Params(
        name="P1_tighter_exit",
        window_pct_R=0.02,
        n0_beta_0=1.0,
        trade_by= "posterior_spread",
        P0= 1.0,
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.2,
    ),
}
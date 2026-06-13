from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Params:
    name: str
    window_pct_R: float
    P0: float
    n0_beta_0: float
    trade_by: str
    z_sco_win: int
    entry_z: float
    exit_z: float


PARAMS: Dict[str, Params] = {
    # Baseline posterior-spread strategy.
    "P0_baseline": Params(
        name="P0_baseline",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    # Exit sensitivity.
    "P1_tighter_exit": Params(
        name="P1_tighter_exit",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=1.0,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.2,
    ),
    # Hypothesis 1: signal source, posterior spread versus Kalman innovation.
    "P2_trade_innovation": Params(
        name="P2_trade_innovation",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="innovation",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    # Hypothesis 2: volatility adaptation through z-score window length.
    "P3_fast_z_window": Params(
        name="P3_fast_z_window",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=20,
        entry_z=2.0,
        exit_z=0.5,
    ),
    "P4_slow_z_window": Params(
        name="P4_slow_z_window",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=120,
        entry_z=2.0,
        exit_z=0.5,
    ),
    # Hypothesis 3: trade frequency and expected edge through thresholds.
    "P5_aggressive_entry": Params(
        name="P5_aggressive_entry",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=1.5,
        exit_z=0.0,
    ),
    "P6_conservative_entry": Params(
        name="P6_conservative_entry",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.5,
        exit_z=0.5,
    ),
    # Hypothesis 4: Kalman state initialization uncertainty.
    "P7_high_initial_uncertainty": Params(
        name="P7_high_initial_uncertainty",
        window_pct_R=0.02,
        P0=10.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    # Hypothesis 5: train/test split and burn-in sensitivity.
    "P8_split_micro_1pct": Params(
        name="P8_split_micro_1pct",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.01,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    "P9_split_short_5pct": Params(
        name="P9_split_short_5pct",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.05,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    "P10_split_moderate_10pct": Params(
        name="P10_split_moderate_10pct",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.10,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    "P11_split_long_25pct": Params(
        name="P11_split_long_25pct",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.25,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
}

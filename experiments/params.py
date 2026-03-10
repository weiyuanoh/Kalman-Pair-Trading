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
    # ---------------------------------------------------------
    # HYPOTHESIS 1: The signal source (Spread vs. Innovation)
    # ---------------------------------------------------------
    "P2_trade_innovation": Params(
        name="P2_trade_innovation",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="innovation", # Testing if the prediction error is a cleaner signal
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),

    # ---------------------------------------------------------
    # HYPOTHESIS 2: Volatility Adaptation (Z-Score Window)
    # ---------------------------------------------------------
    "P3_fast_z_window": Params(
        name="P3_fast_z_window",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=20, # Adapts to local volatility clustering quickly
        entry_z=2.0,
        exit_z=0.5,
    ),
    "P4_slow_z_window": Params(
        name="P4_slow_z_window",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=120, # Requires longer-term structural mean reversion
        entry_z=2.0,
        exit_z=0.5,
    ),

    # ---------------------------------------------------------
    # HYPOTHESIS 3: Trade Frequency & Expected Edge
    # ---------------------------------------------------------
    "P5_aggressive_entry": Params(
        name="P5_aggressive_entry",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=1.5, # Enter earlier, trade more frequently
        exit_z=0.0,  # Hold exactly until the mean is crossed
    ),
    "P6_conservative_entry": Params(
        name="P6_conservative_entry",
        window_pct_R=0.02,
        P0=1.0,
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.5, # Wait for tail events, trade rarely
        exit_z=0.5,
    ),

    # ---------------------------------------------------------
    # HYPOTHESIS 4: Kalman State Initialization
    # ---------------------------------------------------------
    "P7_high_initial_uncertainty": Params(
        name="P7_high_initial_uncertainty",
        window_pct_R=0.02,
        P0=10.0, # High P0 tells the filter to ignore beta_0 and adapt rapidly
        n0_beta_0=0.0375,
        trade_by="posterior_spread",
        z_sco_win=60,
        entry_z=2.0,
        exit_z=0.5,
    ),
    # ---------------------------------------------------------
    # HYPOTHESIS 5: Train/Test Split & Burn-in Sensitivity
    # ---------------------------------------------------------
    
    # The "Cold Start" - 1% burn-in
    # Tests if the Kalman filter is mathematically robust enough 
    # to adapt and find the true beta dynamically with almost no prior training.
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

    # The "Standard Burn-in" - 5% burn-in
    # A solid baseline for intraday data, giving the OLS enough rows 
    # to establish a statistically significant initial beta without wasting data.
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

    # The "Regime Anchor" - 10% burn-in
    # Provides a highly stable anchor for Q and R. If this drastically 
    # outperforms the 1% split, it tells you that accurate static initialization 
    # is critical to your model's success.
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

    # The "Traditional ML Split" - 25% burn-in
    # Closer to a traditional machine learning train/test split. 
    # It sacrifices a quarter of your trading PnL to guarantee the priors 
    # are completely dialed in before execution begins.
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
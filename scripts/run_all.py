from .run import run_one
from experiments.configs import CONFIGS
from experiments.params import PARAMS
import kalman.plots as plts
from kalman.strategy import portfolio_analytics
import pandas as pd
import numpy as np
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _safe(s: Any) -> str:
    s = str(s)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def _reverse_lookup(d: Dict[str, Any], obj: Any, fallback: str) -> str:
    for k, v in d.items():
        if v is obj:
            return k
    return fallback


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _align_equities(equity_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build one wide DF of equity columns for all pairs.
    Assumes each equity DF has a column named 'equity'.
    Outer-joins on index.
    """
    out = None
    for name, eq in equity_map.items():
        if "equity" not in eq.columns:
            raise ValueError(f"Equity DF for {name} missing 'equity' column.")
        s = eq["equity"].rename(name)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    return out.sort_index()

def run_all(cfg_name, prm_name, *, out_root: str = "results",
            save_per_pair_equity: bool = True,
            save_trades: bool = False,
            return_debug_xy: bool = False) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Path]:
    """
    Runs the Kalman Strategy on all 6 permutations of the assets.
    Exports:
      - equities_all.csv   (wide: one column per pair)
      - summary.csv        (one row per pair)
      - stats.json         (meta + tests + stats per pair)
      - defs.json          (cfg/prm definitions)
      - equity/<pair>.csv  (optional)
      - trades/<pair>.csv  (future-ready, optional)

    Returns: (equity_map, stats_map, run_dir)
    """
    cfg = CONFIGS[cfg_name]
    prm = PARAMS[prm_name]
    run_dir = Path(out_root) / f"{_ts()}__{_safe(cfg_name)}__{_safe(prm_name)}"
    (run_dir / "equity").mkdir(parents=True, exist_ok=True)
    (run_dir / "trades").mkdir(parents=True, exist_ok=True)

    # Save cfg/prm definitions for reproducibility
    defs = {"cfg_name": cfg_name, "prm_name": prm_name,
            "cfg": _to_jsonable(cfg), "prm": _to_jsonable(prm)}
    (run_dir / "defs.json").write_text(json.dumps(defs, indent=2, default=str))

    results = {}
    stats_dict = {}
    
    # (Data Key in 'res', Independent Var X, Dependent Var Y)
    # Y = beta * X + spread
    permutations = [
        ("AB", "A", "B"), # Model B against A
        ("AB", "B", "A"), # Model A against B
        ("AC", "A", "C"),
        ("AC", "C", "A"),
        ("BC", "B", "C"),
        ("BC", "C", "B")
    ]
    for key, x_col, y_col in permutations:
        #  "Y_vs_X" means we are trading Y, hedging with X
        strat_name = f"{y_col}_vs_{x_col}" 
        print(f"\n{'='*10} RUNNING STRATEGY: {strat_name} {'='*10}")
        try:

                equity = run_one(cfg, prm, indep_var=x_col, dep_var=y_col,
                          return_debug_xy=return_debug_xy,
                          return_trades=save_trades)

                results[strat_name] = equity

                print(results)
                
                final_val = equity['equity'].iloc[-1]
                ret = (final_val / 100000.0) - 1.0
                print(f"-> Final Equity: ${final_val:,.2f} ({ret*100:.2f}%)")
                stats, dd, rets = portfolio_analytics(equity['equity'])
                stats_dict[strat_name] = stats
                print(f"Portfolio Stats for pair {y_col}_vs_{x_col}: \n", pd.Series(stats))
        except Exception as e:
            print(f"!! Error running {strat_name}: {e}")
            
    return results, stats_dict

if __name__ == "__main__":
    cfg = CONFIGS["S1_post_ffill_unlimited"]
    prm = PARAMS["P0_baseline"]
    all_strategies, stat_dict = run_all(cfg, prm)

    print("All Strats:", all_strategies)
    plts.plot_strategy_performance(all_strategies)
    print("Best Strat: Results")
    best_strat = max(all_strategies, key=lambda k: all_strategies[k]['equity'].iloc[-1])
    best_strat_stats = stat_dict[best_strat]
    print(f" Best Performing Pair: {best_strat} with sharpe ratio: {best_strat_stats['Sharpe (rf=0)']}")

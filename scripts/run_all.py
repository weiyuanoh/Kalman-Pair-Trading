from __future__ import annotations

import pickle
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import kalman.plots as plts
from experiments.configs import CONFIGS
from experiments.params import PARAMS
from kalman.strategy import portfolio_analytics
from scripts.run import run_one


PERMUTATIONS = [
    ("A", "B"),
    ("B", "A"),
    ("A", "C"),
    ("C", "A"),
    ("B", "C"),
    ("C", "B"),
]


def run_all(cfg_name, prm_name, out_root: str = "results"):
    """Run all directed pair permutations and persist one experiment artifact."""
    cfg = CONFIGS[cfg_name]
    prm = PARAMS[prm_name]

    Path(out_root).mkdir(parents=True, exist_ok=True)

    results = {}
    stats_dict = {}
    all_meta = {}

    for x_col, y_col in PERMUTATIONS:
        strat_name = f"{y_col}_vs_{x_col}"
        print(f"\n{'=' * 10} Running strategy: {strat_name} {'=' * 10}")
        try:
            res_dict = run_one(cfg, prm, indep_var=x_col, dep_var=y_col)
            equity_df = res_dict["equity"]
            if equity_df.empty:
                raise ValueError("Backtest returned an empty equity curve.")

            results[strat_name] = equity_df
            all_meta[strat_name] = res_dict.get("meta", {})

            final_val = equity_df["equity"].iloc[-1]
            ret = (final_val / 100000.0) - 1.0
            print(f"Final equity: ${final_val:,.2f} ({ret * 100:.2f}%)")

            stats, dd, rets = portfolio_analytics(equity_df["equity"])
            stats_dict[strat_name] = stats
            print(f"Portfolio stats for {strat_name}:\n{pd.Series(stats)}")

        except Exception as exc:
            print(f"Error running {strat_name}: {exc}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = Path(out_root) / f"{timestamp}_{cfg_name}_{prm_name}.pkl"

    experiment_data = {
        "cfg_name": cfg_name,
        "prm_name": prm_name,
        "cfg": asdict(cfg),
        "prm": asdict(prm),
        "results_equity": results,
        "stats": stats_dict,
        "meta": all_meta,
    }

    with open(file_path, "wb") as f:
        pickle.dump(experiment_data, f)

    print(f"\nExperiment saved to: {file_path}")
    return results, stats_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="S1_post_ffill_unlimited")
    parser.add_argument("--prm", type=str, default="P0_baseline")
    parser.add_argument("--out-root", type=str, default="results")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    all_strategies, stat_dict = run_all(args.cfg, args.prm, out_root=args.out_root)

    print("\nAll strategies:", list(all_strategies.keys()))
    if all_strategies and not args.no_plot:
        plts.plot_strategy_performance(all_strategies)

    if all_strategies:
        best_strat = max(all_strategies, key=lambda key: all_strategies[key]["equity"].iloc[-1])
        best_strat_stats = stat_dict[best_strat]
        print(
            "Best performing pair: "
            f"{best_strat} with Sharpe ratio: {best_strat_stats.get('Sharpe (rf=0)', 'N/A')}"
        )

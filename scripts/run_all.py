from .run import run_one
from experiments.configs import CONFIGS
from experiments.params import PARAMS
import kalman.plots as plts
from kalman.strategy import portfolio_analytics
import pandas as pd
import pickle
import time
from pathlib import Path
from dataclasses import asdict

def run_all(cfg_name, prm_name, out_root: str = "results"):
    """
    Runs the Kalman Strategy on all permutations of the assets.
    Exports a single robust pickle file containing all configs, metrics, and equity curves.
    """
    cfg = CONFIGS[cfg_name]
    prm = PARAMS[prm_name]
    
    Path(out_root).mkdir(parents=True, exist_ok=True)
    
    results = {}
    stats_dict = {}
    all_meta = {}
    
    permutations = [
        ("AB", "A", "B"), 
        ("AB", "B", "A"), 
        ("AC", "A", "C"),
        ("AC", "C", "A"),
        ("BC", "B", "C"),
        ("BC", "C", "B")
    ]
    
    for key, x_col, y_col in permutations:
        strat_name = f"{y_col}_vs_{x_col}" 
        print(f"\n{'='*10} RUNNING STRATEGY: {strat_name} {'='*10}")
        try:
            res_dict = run_one(cfg, prm, indep_var=x_col, dep_var=y_col)
            equity_df = res_dict["equity"] 
            
            results[strat_name] = equity_df
            all_meta[strat_name] = res_dict.get("meta", {})
            
            # Quick check to ensure diagnostics are flowing
            diags = all_meta[strat_name].get("diagnostics", {})
            if diags:
                print(f"-> Diagnostics captured for X and Y cleaning phases.")

            final_val = equity_df['equity'].iloc[-1]
            ret = (final_val / 100000.0) - 1.0
            print(f"-> Final Equity: ${final_val:,.2f} ({ret*100:.2f}%)")
            
            stats, dd, rets = portfolio_analytics(equity_df['equity'])
            stats_dict[strat_name] = stats
            print(f"Portfolio Stats for {strat_name}: \n{pd.Series(stats)}")
            
        except Exception as e:
            print(f"!! Error running {strat_name}: {e}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = Path(out_root) / f"{timestamp}_{cfg_name}_{prm_name}.pkl"
    
    experiment_data = {
        "cfg_name": cfg_name,
        "prm_name": prm_name,
        "cfg": asdict(cfg),
        "prm": asdict(prm),
        "results_equity": results,
        "stats": stats_dict,
        "meta": all_meta
    }
    
    with open(file_path, "wb") as f:
        pickle.dump(experiment_data, f)
        
    print(f"\n✅ Experiment saved successfully to: {file_path}")
            
    return results, stats_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="S1_post_ffill_unlimited")
    parser.add_argument("--prm", type=str, default="P0_baseline")
    args = parser.parse_args()
    
    all_strategies, stat_dict = run_all(args.cfg, args.prm)

    print("\nAll Strats:", list(all_strategies.keys()))
    plts.plot_strategy_performance(all_strategies)
    
    if all_strategies:
        best_strat = max(all_strategies, key=lambda k: all_strategies[k]['equity'].iloc[-1])
        best_strat_stats = stat_dict[best_strat]
        print(f"🏆 Best Performing Pair: {best_strat} with Sharpe ratio: {best_strat_stats.get('Sharpe (rf=0)', 'N/A')}")
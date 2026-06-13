"""Plotting helpers for experiment outputs."""

import matplotlib.pyplot as plt
import numpy as np


def plot_strategy_performance(results):
    """Plot equity curves from a mapping of strategy name to equity DataFrame."""
    if not results:
        raise ValueError("No strategy results to plot.")

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # type: ignore[attr-defined]

    for (name, df), color in zip(results.items(), colors):
        if df.empty or "equity" not in df:
            continue

        final_equity = df["equity"].iloc[-1]
        start_equity = df["equity"].iloc[0]
        total_ret = ((final_equity / start_equity) - 1) * 100
        plt.plot(
            df.index,
            df["equity"],
            label=f"{name} ({total_ret:+.1f}%)",
            linewidth=2,
            alpha=0.8,
            color=color,
        )

    plt.title("Kalman Pairs Trading: Strategy Comparison", fontsize=14, pad=15)
    plt.ylabel("Portfolio Equity ($)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

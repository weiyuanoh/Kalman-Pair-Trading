"""Trading engine and portfolio analytics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd


@dataclass
class TradingEngine:
    """Minimal long/short portfolio engine for bar-based backtests."""

    initial_capital: float = 100_000.0
    commission_bps: float = 0.0
    cash: float = field(init=False)
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    history: List[Dict[str, object]] = field(default_factory=list)
    equity_curve: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_capital)

    def get_portfolio_value(self, current_prices: Mapping[str, float]) -> float:
        market_value = 0.0
        for ticker, pos in self.positions.items():
            if ticker in current_prices:
                market_value += pos["qty"] * float(current_prices[ticker])
        return self.cash + market_value

    def execute_order(self, ticker: str, qty: float, price: float, date, action_label: str) -> None:
        if qty == 0:
            return

        notional = abs(qty * price)
        commission = notional * self.commission_bps / 10_000.0

        self.cash -= qty * price
        self.cash -= commission

        current_pos = self.positions.get(ticker, {"qty": 0.0, "avg_price": 0.0})
        current_qty = float(current_pos["qty"])
        new_total_qty = current_qty + qty

        if new_total_qty == 0:
            self.positions.pop(ticker, None)
        elif current_qty == 0 or current_qty * qty > 0:
            old_notional = current_qty * float(current_pos["avg_price"])
            new_trade_notional = qty * price
            avg_price = (old_notional + new_trade_notional) / new_total_qty
            self.positions[ticker] = {"qty": new_total_qty, "avg_price": avg_price}
        else:
            is_flip = current_qty * new_total_qty < 0
            avg_price = price if is_flip else float(current_pos["avg_price"])
            self.positions[ticker] = {"qty": new_total_qty, "avg_price": avg_price}

        self.history.append(
            {
                "date": date,
                "ticker": ticker,
                "action": action_label,
                "qty": qty,
                "price": price,
                "commission": commission,
            }
        )


def portfolio_analytics(equity: pd.Series):
    if equity.empty:
        raise ValueError("Cannot compute portfolio analytics for an empty equity series.")

    equity_daily = equity.resample("B").last().dropna()
    if equity_daily.empty:
        raise ValueError("Cannot compute portfolio analytics without daily equity observations.")

    rets_daily = equity_daily.pct_change().dropna()

    sharpe = 0.0
    ann_vol = 0.0
    if not rets_daily.empty and rets_daily.std() != 0:
        sharpe = (rets_daily.mean() / rets_daily.std()) * np.sqrt(252)
        ann_vol = rets_daily.std() * np.sqrt(252)

    start_date = equity_daily.index[0]
    end_date = equity_daily.index[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (equity_daily.iloc[-1] / equity_daily.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    peak_daily = equity_daily.cummax()
    underwater_daily = equity_daily < peak_daily
    dd_duration = (
        underwater_daily.groupby((underwater_daily != underwater_daily.shift()).cumsum()).cumcount() + 1
    )
    max_dd_duration = dd_duration[underwater_daily].max() if underwater_daily.any() else 0

    return {
        "Final Equity": float(equity.iloc[-1]),
        "Total Return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "CAGR": float(cagr),
        "Ann Vol": float(ann_vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Max DD Duration (days)": int(max_dd_duration),
    }, dd, rets_daily

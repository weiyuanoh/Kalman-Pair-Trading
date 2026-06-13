"""Backward-compatible import path for the trading engine."""

from kalman.engine import TradingEngine, portfolio_analytics

__all__ = ["TradingEngine", "portfolio_analytics"]

"""
TradingGuru: A comprehensive trading strategy framework.

This package provides tools for developing, backtesting, and analyzing
quantitative trading strategies, with a focus on the Indian equity market.
"""

__version__ = "0.1.0"
__author__ = "Pratyush Kumar"

# Import main components for easy access
from tradingguru.core.base import BaseStrategy
from tradingguru.core.engine import BacktestEngine
from tradingguru.core.risk import RiskManager
from tradingguru.core.utils import calculate_cagr, calculate_max_drawdown

__all__ = [
    "BaseStrategy",
    "BacktestEngine", 
    "RiskManager",
    "calculate_cagr",
    "calculate_max_drawdown",
]
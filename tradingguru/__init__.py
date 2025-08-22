"""
TradingGuru: A comprehensive trading strategy framework.

This package provides tools for developing, backtesting, and analyzing
quantitative trading strategies, with a focus on the Indian equity market.
"""

__version__ = "0.1.0"
__author__ = "Pratyush Kumar"

# Core components available for import
# Import only when explicitly requested to avoid dependency issues during setup
__all__ = [
    "BaseStrategy",
    "BacktestEngine", 
    "RiskManager",
    "calculate_cagr",
    "calculate_max_drawdown",
]

def __getattr__(name):
    """Lazy loading of main components."""
    if name == "BaseStrategy":
        from tradingguru.core.base import BaseStrategy
        return BaseStrategy
    elif name == "BacktestEngine":
        from tradingguru.core.engine import BacktestEngine
        return BacktestEngine
    elif name == "RiskManager":
        from tradingguru.core.risk import RiskManager
        return RiskManager
    elif name == "calculate_cagr":
        from tradingguru.core.utils import calculate_cagr
        return calculate_cagr
    elif name == "calculate_max_drawdown":
        from tradingguru.core.utils import calculate_max_drawdown
        return calculate_max_drawdown
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
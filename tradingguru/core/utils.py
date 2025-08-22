"""
Utility functions for TradingGuru.

This module provides common financial calculations and utility functions
used across the trading framework.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from datetime import datetime, date


def calculate_cagr(returns: pd.Series, 
                  periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR) from returns.
    
    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized compound growth rate
    """
    if len(returns) == 0:
        return 0.0
        
    total_return = (1 + returns).prod() - 1
    num_periods = len(returns)
    
    if num_periods == 0 or total_return <= -1:
        return 0.0
        
    cagr = (1 + total_return) ** (periods_per_year / num_periods) - 1
    return cagr


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, Optional[date], Optional[date]]:
    """
    Calculate maximum drawdown from returns series.
    
    Args:
        returns: Series of periodic returns with date index
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    if len(returns) == 0:
        return 0.0, None, None
        
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    rolling_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative / rolling_max) - 1
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    
    if pd.isna(max_dd):
        return 0.0, None, None
        
    # Find peak and trough dates
    max_dd_idx = drawdown.idxmin()
    peak_idx = rolling_max.loc[:max_dd_idx].idxmax()
    
    peak_date = peak_idx if hasattr(peak_idx, 'date') else peak_idx
    trough_date = max_dd_idx if hasattr(max_dd_idx, 'date') else max_dd_idx
    
    return max_dd, peak_date, trough_date


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        return 0.0
        
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio from returns.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
        
    downside_std = negative_returns.std()
    if downside_std == 0:
        return 0.0
        
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    return sortino


def calculate_calmar_ratio(returns: pd.Series,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(returns, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(returns)
    
    if max_dd >= 0:  # No drawdown
        return np.inf if cagr > 0 else 0.0
        
    calmar = abs(cagr / max_dd)
    return calmar


def calculate_win_rate(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        returns: Series of periodic returns
        threshold: Minimum return to consider a "win"
        
    Returns:
        Win rate as a percentage (0-1)
    """
    if len(returns) == 0:
        return 0.0
        
    wins = (returns > threshold).sum()
    total = len(returns)
    
    return wins / total if total > 0 else 0.0


def calculate_information_ratio(strategy_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio (excess return / tracking error).
    
    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        periods_per_year: Number of periods in a year
        
    Returns:
        Information ratio
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
        
    # Align series
    aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) == 0:
        return 0.0
        
    strategy_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]
    
    # Calculate excess returns
    excess_returns = strategy_aligned - benchmark_aligned
    
    if excess_returns.std() == 0:
        return 0.0
        
    # Annualized information ratio
    ir = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return ir


def calculate_beta(strategy_returns: pd.Series,
                  market_returns: pd.Series) -> float:
    """
    Calculate beta (sensitivity to market movements).
    
    Args:
        strategy_returns: Strategy returns series
        market_returns: Market returns series
        
    Returns:
        Beta coefficient
    """
    if len(strategy_returns) == 0 or len(market_returns) == 0:
        return 0.0
        
    # Align series
    aligned_data = pd.concat([strategy_returns, market_returns], axis=1).dropna()
    if len(aligned_data) < 2:
        return 0.0
        
    strategy_aligned = aligned_data.iloc[:, 0]
    market_aligned = aligned_data.iloc[:, 1]
    
    # Calculate covariance and variance
    covariance = np.cov(strategy_aligned, market_aligned)[0, 1]
    market_variance = np.var(market_aligned)
    
    if market_variance == 0:
        return 0.0
        
    beta = covariance / market_variance
    return beta


def winsorize_series(series: pd.Series, 
                    lower_percentile: float = 0.01,
                    upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize a series by clipping extreme values.
    
    Args:
        series: Input series
        lower_percentile: Lower clipping percentile
        upper_percentile: Upper clipping percentile
        
    Returns:
        Winsorized series
    """
    if len(series) == 0:
        return series
        
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    
    return series.clip(lower=lower_bound, upper=upper_bound)


def standardize_series(series: pd.Series, 
                      method: str = 'zscore') -> pd.Series:
    """
    Standardize a series using different methods.
    
    Args:
        series: Input series
        method: Standardization method ('zscore', 'minmax', 'rank')
        
    Returns:
        Standardized series
    """
    if len(series) == 0:
        return series
        
    if method == 'zscore':
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std
        
    elif method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
        
    elif method == 'rank':
        return series.rank(pct=True) - 0.5
        
    else:
        raise ValueError(f"Unknown standardization method: {method}")


def calculate_rolling_correlation(x: pd.Series, 
                                 y: pd.Series,
                                 window: int = 60) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        x: First series
        y: Second series
        window: Rolling window size
        
    Returns:
        Rolling correlation series
    """
    return x.rolling(window).corr(y)


def format_performance_report(stats: dict, 
                            strategy_name: str = "Strategy") -> str:
    """
    Format performance statistics into a readable report.
    
    Args:
        stats: Dictionary of performance statistics
        strategy_name: Name of the strategy
        
    Returns:
        Formatted performance report string
    """
    report = f"\n{'='*50}\n"
    report += f"{strategy_name.upper()} PERFORMANCE REPORT\n"
    report += f"{'='*50}\n\n"
    
    if not stats:
        report += "No performance data available.\n"
        return report
    
    # Format each statistic
    metrics = [
        ("Total Return", "total_return", "{:.2%}"),
        ("Annualized Return", "annualized_return", "{:.2%}"),
        ("Annualized Volatility", "annualized_volatility", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Maximum Drawdown", "max_drawdown", "{:.2%}"),
        ("Win Rate", "win_rate", "{:.2%}"),
        ("Number of Trades", "num_trades", "{:,.0f}"),
        ("Number of Periods", "num_periods", "{:,.0f}"),
    ]
    
    for label, key, fmt in metrics:
        if key in stats:
            value = stats[key]
            if pd.isna(value):
                formatted_value = "N/A"
            else:
                try:
                    formatted_value = fmt.format(value)
                except (ValueError, TypeError):
                    formatted_value = str(value)
            report += f"{label:<25}: {formatted_value}\n"
    
    report += f"\n{'='*50}\n"
    return report
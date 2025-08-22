"""
Base strategy framework for TradingGuru.

This module provides the foundational BaseStrategy class that all trading
strategies should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides the basic structure and interface that all trading
    strategies must implement. It handles common functionality like parameter
    management, universe definition, and portfolio tracking.
    """
    
    def __init__(self, 
                 name: str,
                 universe: List[str],
                 params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            universe: List of symbols to trade
            params: Strategy parameters dictionary
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.universe = universe
        self.params = params or {}
        self.params.update(kwargs)
        
        # Portfolio tracking
        self.positions = pd.Series(dtype=float, name='positions')
        self.cash = 0.0
        self.portfolio_value = 0.0
        
        # Performance tracking
        self.returns = pd.Series(dtype=float, name='returns')
        self.trades = []
        
        # Strategy state
        self.current_date = None
        self.is_initialized = False
        
    def initialize(self, initial_cash: float = 1000000.0) -> None:
        """
        Initialize the strategy with starting capital.
        
        Args:
            initial_cash: Starting cash amount
        """
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.positions = pd.Series(0.0, index=self.universe, name='positions')
        self.is_initialized = True
        
    @abstractmethod
    def generate_signals(self, 
                        data: pd.DataFrame, 
                        current_date: date) -> Dict[str, float]:
        """
        Generate trading signals for the given date.
        
        Args:
            data: Market data DataFrame with price/volume information
            current_date: Current trading date
            
        Returns:
            Dictionary mapping symbols to target weights (-1 to 1)
        """
        pass
        
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features needed for signal generation.
        
        Args:
            data: Raw market data
            
        Returns:
            DataFrame with calculated features
        """
        pass
        
    def update_positions(self, 
                        signals: Dict[str, float], 
                        prices: Dict[str, float],
                        current_date: date) -> List[Dict[str, Any]]:
        """
        Update positions based on signals and return list of trades.
        
        Args:
            signals: Target weights for each symbol
            prices: Current prices for each symbol
            current_date: Current trading date
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        total_value = self.get_portfolio_value(prices)
        
        for symbol in self.universe:
            if symbol in signals and symbol in prices:
                target_weight = signals[symbol]
                target_value = total_value * target_weight
                target_shares = target_value / prices[symbol] if prices[symbol] > 0 else 0
                
                current_shares = self.positions.get(symbol, 0)
                shares_to_trade = target_shares - current_shares
                
                if abs(shares_to_trade) > 1e-6:  # Minimum trade threshold
                    trade_value = shares_to_trade * prices[symbol]
                    
                    # Record the trade
                    trade = {
                        'date': current_date,
                        'symbol': symbol,
                        'shares': shares_to_trade,
                        'price': prices[symbol],
                        'value': trade_value,
                        'type': 'BUY' if shares_to_trade > 0 else 'SELL'
                    }
                    trades.append(trade)
                    
                    # Update positions and cash
                    self.positions[symbol] = target_shares
                    self.cash -= trade_value
        
        self.trades.extend(trades)
        self.current_date = current_date
        return trades
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            prices: Current prices for all symbols
            
        Returns:
            Total portfolio value
        """
        position_value = sum(
            self.positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in self.universe
        )
        return self.cash + position_value
        
    def get_position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Get current position weights.
        
        Args:
            prices: Current prices for all symbols
            
        Returns:
            Dictionary mapping symbols to current weights
        """
        total_value = self.get_portfolio_value(prices)
        if total_value <= 0:
            return {symbol: 0.0 for symbol in self.universe}
            
        return {
            symbol: (self.positions.get(symbol, 0) * prices.get(symbol, 0)) / total_value
            for symbol in self.universe
        }
        
    def update_performance(self, prices: Dict[str, float], current_date: date) -> float:
        """
        Update performance metrics.
        
        Args:
            prices: Current prices for all symbols
            current_date: Current date
            
        Returns:
            Current portfolio return since last update
        """
        new_portfolio_value = self.get_portfolio_value(prices)
        
        if self.portfolio_value > 0:
            period_return = (new_portfolio_value / self.portfolio_value) - 1
        else:
            period_return = 0.0
            
        self.returns.loc[current_date] = period_return
        self.portfolio_value = new_portfolio_value
        
        return period_return
        
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Calculate summary performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.returns) == 0:
            return {}
            
        returns = self.returns.dropna()
        
        # Basic stats
        total_return = (1 + returns).prod() - 1
        num_periods = len(returns)
        
        # Annualized metrics (assuming daily returns)
        trading_days_per_year = 252
        annualized_return = (1 + total_return) ** (trading_days_per_year / num_periods) - 1
        annualized_vol = returns.std() * np.sqrt(trading_days_per_year)
        
        # Risk metrics
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Win rate
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'num_periods': num_periods
        }
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        return drawdown.min()
        
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        self.positions = pd.Series(0.0, index=self.universe, name='positions')
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.returns = pd.Series(dtype=float, name='returns')
        self.trades = []
        self.current_date = None
        self.is_initialized = False
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', universe={len(self.universe)} symbols)"
"""
Backtesting engine for TradingGuru.

This module provides a comprehensive backtesting framework with transaction
cost modeling, benchmark comparison, and detailed performance analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date
import logging
from dataclasses import dataclass

from .base import BaseStrategy
from .risk import RiskManager, RiskConstraints
from .utils import (
    calculate_cagr, calculate_max_drawdown, calculate_sharpe_ratio,
    calculate_information_ratio, calculate_beta, format_performance_report
)

logger = logging.getLogger(__name__)


@dataclass
class TransactionCosts:
    """Configuration for transaction cost modeling."""
    commission_rate: float = 0.001      # Commission as % of trade value
    bid_ask_spread: float = 0.0005      # Half-spread as % of price
    market_impact: float = 0.0001       # Market impact as % of trade value
    min_commission: float = 0.0         # Minimum commission per trade


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    start_date: Union[str, date]
    end_date: Union[str, date]
    initial_capital: float = 1000000.0
    rebalance_frequency: str = 'D'      # 'D', 'W', 'M', 'Q'
    benchmark_symbol: str = '^NSEI'
    transaction_costs: TransactionCosts = None
    risk_constraints: RiskConstraints = None
    
    def __post_init__(self):
        if self.transaction_costs is None:
            self.transaction_costs = TransactionCosts()
        if self.risk_constraints is None:
            self.risk_constraints = RiskConstraints()


class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    
    This engine handles strategy execution, transaction cost modeling,
    risk management, and performance analysis with benchmark comparison.
    """
    
    def __init__(self, 
                 strategy: BaseStrategy,
                 config: BacktestConfig,
                 sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the backtest engine.
        
        Args:
            strategy: Trading strategy to backtest
            config: Backtest configuration
            sector_mapping: Mapping of symbols to sectors for risk management
        """
        self.strategy = strategy
        self.config = config
        self.sector_mapping = sector_mapping or {}
        
        # Initialize components
        self.risk_manager = RiskManager(
            constraints=config.risk_constraints,
            sector_mapping=sector_mapping
        )
        
        # Results storage
        self.results = {}
        self.daily_portfolio = pd.DataFrame()
        self.trades_log = []
        self.benchmark_returns = pd.Series(dtype=float)
        
        # Performance tracking
        self.portfolio_values = []
        self.cash_history = []
        self.positions_history = []
        
    def run_backtest(self, 
                    price_data: pd.DataFrame,
                    benchmark_data: Optional[pd.Series] = None,
                    features_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute the backtest.
        
        Args:
            price_data: Price data with symbols as columns, dates as index
            benchmark_data: Benchmark price series for comparison
            features_data: Pre-calculated features (optional)
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest for {self.strategy.name}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.0f}")
        
        # Prepare data
        price_data = self._prepare_price_data(price_data)
        returns_data = price_data.pct_change().dropna()
        
        if benchmark_data is not None:
            self.benchmark_returns = self._prepare_benchmark_data(
                benchmark_data, price_data.index
            )
        
        # Initialize strategy
        self.strategy.initialize(self.config.initial_capital)
        
        # Calculate features if not provided
        if features_data is None:
            logger.info("Calculating strategy features...")
            features_data = self.strategy.calculate_features(price_data)
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(price_data.index)
        logger.info(f"Rebalancing on {len(rebalance_dates)} dates")
        
        # Main backtest loop
        current_weights = {symbol: 0.0 for symbol in self.strategy.universe}
        
        for i, current_date in enumerate(price_data.index):
            try:
                # Get current prices
                current_prices = price_data.loc[current_date].to_dict()
                
                # Check if it's a rebalancing date
                if current_date in rebalance_dates:
                    # Generate signals
                    signals = self.strategy.generate_signals(features_data, current_date)
                    
                    # Apply risk management
                    target_weights = self.risk_manager.apply_position_constraints(
                        signals, current_weights, current_prices
                    )
                    
                    # Calculate and apply transaction costs
                    adjusted_weights, trade_costs = self._apply_transaction_costs(
                        target_weights, current_weights, current_prices
                    )
                    
                    # Execute trades
                    trades = self.strategy.update_positions(
                        adjusted_weights, current_prices, current_date
                    )
                    
                    # Log trades with costs
                    for trade in trades:
                        trade['transaction_cost'] = trade_costs.get(trade['symbol'], 0)
                        self.trades_log.append(trade)
                    
                    # Deduct transaction costs from cash
                    total_costs = sum(trade_costs.values())
                    self.strategy.cash -= total_costs
                    
                    current_weights = adjusted_weights
                
                # Update portfolio performance
                self.strategy.update_performance(current_prices, current_date)
                
                # Store daily data
                self._record_daily_data(current_date, current_prices, current_weights)
                
                # Check risk limits
                current_return = self.strategy.returns.iloc[-1] if len(self.strategy.returns) > 0 else 0
                portfolio_metrics = self.risk_manager.calculate_portfolio_risk(
                    current_weights, returns_data.iloc[:i+1]
                )
                
                if not self.risk_manager.check_risk_limits(current_return, portfolio_metrics):
                    logger.warning(f"Risk limits breached on {current_date}, stopping strategy")
                    break
                
                # Progress logging
                if i % 100 == 0:
                    progress = (i + 1) / len(price_data) * 100
                    logger.info(f"Progress: {progress:.1f}% ({current_date})")
                    
            except Exception as e:
                logger.error(f"Error on {current_date}: {str(e)}")
                continue
        
        # Calculate final results
        self.results = self._calculate_results()
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.strategy.portfolio_value:,.0f}")
        return self.results
        
    def _prepare_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and filter price data for backtest period."""
        # Convert date columns to datetime if needed
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        
        # Filter by date range
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        price_data = price_data.loc[start_date:end_date]
        
        # Filter to strategy universe
        available_symbols = [s for s in self.strategy.universe if s in price_data.columns]
        if len(available_symbols) < len(self.strategy.universe):
            missing = set(self.strategy.universe) - set(available_symbols)
            logger.warning(f"Missing price data for symbols: {missing}")
        
        price_data = price_data[available_symbols]
        
        # Forward fill missing values
        price_data = price_data.fillna(method='ffill')
        
        return price_data
        
    def _prepare_benchmark_data(self, 
                               benchmark_data: pd.Series, 
                               date_index: pd.DatetimeIndex) -> pd.Series:
        """Prepare benchmark data and calculate returns."""
        if not isinstance(benchmark_data.index, pd.DatetimeIndex):
            benchmark_data.index = pd.to_datetime(benchmark_data.index)
        
        # Align with strategy dates
        benchmark_data = benchmark_data.reindex(date_index, method='ffill')
        
        # Calculate returns
        benchmark_returns = benchmark_data.pct_change().dropna()
        return benchmark_returns
        
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[date]:
        """Get rebalancing dates based on frequency."""
        if self.config.rebalance_frequency == 'D':
            return date_index.tolist()
        elif self.config.rebalance_frequency == 'W':
            return date_index[date_index.dayofweek == 4].tolist()  # Fridays
        elif self.config.rebalance_frequency == 'M':
            return date_index[date_index.is_month_end].tolist()
        elif self.config.rebalance_frequency == 'Q':
            return date_index[date_index.is_quarter_end].tolist()
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.config.rebalance_frequency}")
            
    def _apply_transaction_costs(self, 
                               target_weights: Dict[str, float],
                               current_weights: Dict[str, float],
                               prices: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Apply transaction cost model and return adjusted weights and costs.
        
        Returns:
            Tuple of (adjusted_weights, transaction_costs_by_symbol)
        """
        costs = self.config.transaction_costs
        portfolio_value = self.strategy.get_portfolio_value(prices)
        
        transaction_costs = {}
        adjusted_weights = target_weights.copy()
        
        for symbol in set(target_weights.keys()) | set(current_weights.keys()):
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            price = prices.get(symbol, 0)
            
            if price <= 0:
                continue
                
            # Calculate trade value
            weight_change = abs(target_w - current_w)
            trade_value = weight_change * portfolio_value
            
            if trade_value < 1e-6:  # Skip tiny trades
                continue
            
            # Calculate cost components
            commission = max(trade_value * costs.commission_rate, costs.min_commission)
            spread_cost = trade_value * costs.bid_ask_spread
            impact_cost = trade_value * costs.market_impact
            
            total_cost = commission + spread_cost + impact_cost
            transaction_costs[symbol] = total_cost
            
            # Adjust target weight for costs (reduce trade size slightly)
            cost_adjustment = total_cost / portfolio_value
            if target_w > current_w:  # Buying
                adjusted_weights[symbol] = target_w - (cost_adjustment * 0.5)
            elif target_w < current_w:  # Selling
                adjusted_weights[symbol] = target_w + (cost_adjustment * 0.5)
        
        return adjusted_weights, transaction_costs
        
    def _record_daily_data(self, 
                          current_date: date,
                          prices: Dict[str, float],
                          weights: Dict[str, float]) -> None:
        """Record daily portfolio data."""
        portfolio_value = self.strategy.get_portfolio_value(prices)
        
        daily_data = {
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.strategy.cash,
            'returns': self.strategy.returns.iloc[-1] if len(self.strategy.returns) > 0 else 0
        }
        
        # Add position values
        for symbol in self.strategy.universe:
            position_value = self.strategy.positions.get(symbol, 0) * prices.get(symbol, 0)
            daily_data[f'{symbol}_value'] = position_value
            daily_data[f'{symbol}_weight'] = weights.get(symbol, 0)
        
        self.portfolio_values.append(daily_data)
        
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        # Convert daily data to DataFrame
        daily_df = pd.DataFrame(self.portfolio_values)
        if not daily_df.empty:
            daily_df.set_index('date', inplace=True)
            self.daily_portfolio = daily_df
        
        # Strategy performance
        strategy_stats = self.strategy.get_summary_stats()
        
        # Benchmark comparison if available
        benchmark_stats = {}
        if len(self.benchmark_returns) > 0:
            # Align benchmark with strategy returns
            aligned_returns = pd.concat([
                self.strategy.returns, 
                self.benchmark_returns
            ], axis=1).dropna()
            
            if len(aligned_returns) > 0:
                benchmark_returns_aligned = aligned_returns.iloc[:, 1]
                strategy_returns_aligned = aligned_returns.iloc[:, 0]
                
                # Calculate benchmark stats
                benchmark_stats = {
                    'total_return': (1 + benchmark_returns_aligned).prod() - 1,
                    'annualized_return': calculate_cagr(benchmark_returns_aligned),
                    'annualized_volatility': benchmark_returns_aligned.std() * np.sqrt(252),
                    'sharpe_ratio': calculate_sharpe_ratio(benchmark_returns_aligned),
                    'max_drawdown': calculate_max_drawdown(benchmark_returns_aligned)[0]
                }
                
                # Relative performance
                strategy_stats['information_ratio'] = calculate_information_ratio(
                    strategy_returns_aligned, benchmark_returns_aligned
                )
                strategy_stats['beta'] = calculate_beta(
                    strategy_returns_aligned, benchmark_returns_aligned
                )
                strategy_stats['alpha'] = (
                    strategy_stats['annualized_return'] - 
                    benchmark_stats['annualized_return'] * strategy_stats.get('beta', 1)
                )
        
        # Transaction cost analysis
        trades_df = pd.DataFrame(self.trades_log)
        transaction_analysis = {}
        if not trades_df.empty:
            total_costs = trades_df['transaction_cost'].sum()
            total_volume = trades_df['value'].abs().sum()
            
            transaction_analysis = {
                'total_transaction_costs': total_costs,
                'total_volume': total_volume,
                'cost_ratio': total_costs / self.config.initial_capital,
                'avg_cost_per_trade': total_costs / len(trades_df),
                'num_trades': len(trades_df)
            }
        
        # Risk analysis
        risk_metrics = {}
        if len(self.strategy.returns) > 0:
            returns = self.strategy.returns.dropna()
            
            # Calculate VaR and CVaR
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            cvar_95 = returns[returns <= var_95].mean() if var_95 < 0 else 0
            cvar_99 = returns[returns <= var_99].mean() if var_99 < 0 else 0
            
            risk_metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'downside_deviation': returns[returns < 0].std() * np.sqrt(252)
            }
        
        return {
            'strategy_stats': strategy_stats,
            'benchmark_stats': benchmark_stats,
            'transaction_analysis': transaction_analysis,
            'risk_metrics': risk_metrics,
            'daily_portfolio': self.daily_portfolio,
            'trades_log': trades_df,
            'config': self.config,
            'risk_breaches': self.risk_manager.risk_breaches
        }
        
    def generate_report(self) -> str:
        """Generate a comprehensive backtest report."""
        if not self.results:
            return "No backtest results available. Run backtest first."
        
        report = "\n" + "="*80 + "\n"
        report += f"BACKTEST REPORT: {self.strategy.name.upper()}\n"
        report += "="*80 + "\n\n"
        
        # Configuration
        report += "CONFIGURATION:\n"
        report += f"  Period: {self.config.start_date} to {self.config.end_date}\n"
        report += f"  Initial Capital: ${self.config.initial_capital:,.0f}\n"
        report += f"  Rebalance Frequency: {self.config.rebalance_frequency}\n"
        report += f"  Benchmark: {self.config.benchmark_symbol}\n\n"
        
        # Strategy performance
        strategy_stats = self.results['strategy_stats']
        report += format_performance_report(strategy_stats, "STRATEGY")
        
        # Benchmark comparison
        if self.results['benchmark_stats']:
            benchmark_stats = self.results['benchmark_stats']
            report += format_performance_report(benchmark_stats, "BENCHMARK")
            
            # Relative performance
            report += "RELATIVE PERFORMANCE:\n"
            report += f"  Alpha: {strategy_stats.get('alpha', 0):.2%}\n"
            report += f"  Beta: {strategy_stats.get('beta', 0):.3f}\n"
            report += f"  Information Ratio: {strategy_stats.get('information_ratio', 0):.3f}\n\n"
        
        # Transaction costs
        if self.results['transaction_analysis']:
            tc = self.results['transaction_analysis']
            report += "TRANSACTION COST ANALYSIS:\n"
            report += f"  Total Costs: ${tc.get('total_transaction_costs', 0):,.0f}\n"
            report += f"  Cost Ratio: {tc.get('cost_ratio', 0):.2%}\n"
            report += f"  Total Volume: ${tc.get('total_volume', 0):,.0f}\n"
            report += f"  Number of Trades: {tc.get('num_trades', 0):,.0f}\n\n"
        
        # Risk metrics
        if self.results['risk_metrics']:
            risk = self.results['risk_metrics']
            report += "RISK METRICS:\n"
            report += f"  VaR (95%): {risk.get('var_95', 0):.2%}\n"
            report += f"  VaR (99%): {risk.get('var_99', 0):.2%}\n"
            report += f"  CVaR (95%): {risk.get('cvar_95', 0):.2%}\n"
            report += f"  CVaR (99%): {risk.get('cvar_99', 0):.2%}\n"
            report += f"  Skewness: {risk.get('skewness', 0):.3f}\n"
            report += f"  Kurtosis: {risk.get('kurtosis', 0):.3f}\n\n"
        
        # Risk management
        if self.results['risk_breaches']:
            report += f"Risk breaches: {len(self.results['risk_breaches'])}\n"
            for breach in self.results['risk_breaches'][-3:]:  # Last 3
                report += f"  {breach['type']}: {breach['value']:.2%} on {breach['date']}\n"
            report += "\n"
        
        report += "="*80 + "\n"
        return report
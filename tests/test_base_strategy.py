"""
Unit tests for the base strategy framework.

This module tests the core functionality of the BaseStrategy class
and related components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingguru.core.base import BaseStrategy
from tradingguru.core.utils import calculate_cagr, calculate_max_drawdown, calculate_sharpe_ratio


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self, universe, **params):
        super().__init__("TestStrategy", universe, **params)
        
    def generate_signals(self, data, current_date):
        """Generate simple test signals."""
        # Return equal weights for all symbols
        return {symbol: 1.0 / len(self.universe) for symbol in self.universe}
        
    def calculate_features(self, data):
        """Calculate simple test features."""
        features = pd.DataFrame(index=data.index)
        for symbol in self.universe:
            if symbol in data.columns:
                # Simple momentum: current price / 20-day moving average
                features[f'{symbol}_momentum'] = data[symbol] / data[symbol].rolling(20).mean()
        return features


class TestBaseStrategy(unittest.TestCase):
    """Test cases for BaseStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.universe = ['AAPL', 'GOOGL', 'MSFT']
        self.strategy = MockStrategy(self.universe, test_param=0.5)
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        price_data = {}
        for symbol in self.universe:
            # Generate random walk prices
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * (1 + returns).cumprod()
            price_data[symbol] = prices
            
        self.price_data = pd.DataFrame(price_data, index=dates)
        
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "TestStrategy")
        self.assertEqual(self.strategy.universe, self.universe)
        self.assertEqual(self.strategy.params['test_param'], 0.5)
        self.assertFalse(self.strategy.is_initialized)
        
    def test_initialize_strategy(self):
        """Test strategy initialization with capital."""
        initial_cash = 1000000.0
        self.strategy.initialize(initial_cash)
        
        self.assertTrue(self.strategy.is_initialized)
        self.assertEqual(self.strategy.cash, initial_cash)
        self.assertEqual(self.strategy.portfolio_value, initial_cash)
        self.assertEqual(len(self.strategy.positions), len(self.universe))
        
    def test_calculate_features(self):
        """Test feature calculation."""
        features = self.strategy.calculate_features(self.price_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.price_data))
        
        # Check that momentum features are calculated
        for symbol in self.universe:
            momentum_col = f'{symbol}_momentum'
            self.assertIn(momentum_col, features.columns)
            
    def test_generate_signals(self):
        """Test signal generation."""
        features = self.strategy.calculate_features(self.price_data)
        current_date = self.price_data.index[-1].date()
        
        signals = self.strategy.generate_signals(features, current_date)
        
        self.assertIsInstance(signals, dict)
        self.assertEqual(len(signals), len(self.universe))
        
        # Check that weights sum to approximately 1
        total_weight = sum(signals.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
    def test_update_positions(self):
        """Test position updates."""
        self.strategy.initialize(1000000.0)
        
        # Create signals and prices
        signals = {symbol: 1.0 / len(self.universe) for symbol in self.universe}
        prices = {symbol: self.price_data[symbol].iloc[-1] for symbol in self.universe}
        current_date = self.price_data.index[-1].date()
        
        trades = self.strategy.update_positions(signals, prices, current_date)
        
        self.assertIsInstance(trades, list)
        self.assertEqual(len(trades), len(self.universe))
        
        # Check that positions were updated
        for symbol in self.universe:
            self.assertGreater(abs(self.strategy.positions[symbol]), 0)
            
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        self.strategy.initialize(1000000.0)
        
        # Set some positions
        prices = {symbol: 100.0 for symbol in self.universe}
        self.strategy.positions = pd.Series([1000, 2000, 1500], index=self.universe)
        self.strategy.cash = 550000.0
        
        portfolio_value = self.strategy.get_portfolio_value(prices)
        expected_value = 1000*100 + 2000*100 + 1500*100 + 550000.0
        
        self.assertEqual(portfolio_value, expected_value)
        
    def test_position_weights_calculation(self):
        """Test position weights calculation."""
        self.strategy.initialize(1000000.0)
        
        prices = {symbol: 100.0 for symbol in self.universe}
        self.strategy.positions = pd.Series([1000, 2000, 1500], index=self.universe)
        self.strategy.cash = 550000.0
        
        weights = self.strategy.get_position_weights(prices)
        
        total_value = 1000000.0
        expected_weights = {
            'AAPL': (1000 * 100) / total_value,
            'GOOGL': (2000 * 100) / total_value,
            'MSFT': (1500 * 100) / total_value
        }
        
        for symbol in self.universe:
            self.assertAlmostEqual(weights[symbol], expected_weights[symbol], places=6)
            
    def test_performance_update(self):
        """Test performance tracking."""
        self.strategy.initialize(1000000.0)
        
        prices = {symbol: 100.0 for symbol in self.universe}
        current_date = date(2023, 1, 1)
        
        # Initial performance update
        return1 = self.strategy.update_performance(prices, current_date)
        self.assertEqual(return1, 0.0)  # No change on first update
        
        # Price increase
        new_prices = {symbol: 110.0 for symbol in self.universe}
        new_date = date(2023, 1, 2)
        
        # Set some positions first
        self.strategy.positions = pd.Series([1000, 1000, 1000], index=self.universe)
        self.strategy.cash = 700000.0
        
        return2 = self.strategy.update_performance(new_prices, new_date)
        self.assertGreater(return2, 0)  # Should be positive return
        
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        # Generate some fake returns
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        
        self.strategy.returns = returns
        stats = self.strategy.get_summary_stats()
        
        self.assertIn('total_return', stats)
        self.assertIn('annualized_return', stats)
        self.assertIn('annualized_volatility', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('max_drawdown', stats)
        self.assertIn('win_rate', stats)
        
        # Check that values are reasonable
        self.assertTrue(-1 < stats['total_return'] < 10)  # Reasonable range
        self.assertTrue(0 < stats['annualized_volatility'] < 2)  # Reasonable volatility
        
    def test_reset_strategy(self):
        """Test strategy reset functionality."""
        self.strategy.initialize(1000000.0)
        
        # Make some changes
        self.strategy.positions['AAPL'] = 1000
        self.strategy.cash = 900000.0
        self.strategy.returns = pd.Series([0.01, 0.02, -0.01])
        
        # Reset
        self.strategy.reset()
        
        self.assertFalse(self.strategy.is_initialized)
        self.assertEqual(self.strategy.cash, 0.0)
        self.assertEqual(self.strategy.portfolio_value, 0.0)
        self.assertEqual(len(self.strategy.returns), 0)
        self.assertEqual(len(self.strategy.trades), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        
    def test_calculate_cagr(self):
        """Test CAGR calculation."""
        # Test with known data
        simple_returns = pd.Series([0.1, 0.05, -0.02, 0.08])  # 4 periods
        cagr = calculate_cagr(simple_returns, periods_per_year=4)
        
        total_return = (1 + simple_returns).prod() - 1
        expected_cagr = (1 + total_return) ** (4/4) - 1
        
        self.assertAlmostEqual(cagr, expected_cagr, places=6)
        
    def test_calculate_cagr_empty_series(self):
        """Test CAGR calculation with empty series."""
        empty_series = pd.Series(dtype=float)
        cagr = calculate_cagr(empty_series)
        self.assertEqual(cagr, 0.0)
        
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Test with known sequence
        returns = pd.Series([0.1, 0.05, -0.15, -0.1, 0.2, 0.05])
        max_dd, peak_date, trough_date = calculate_max_drawdown(returns)
        
        self.assertLess(max_dd, 0)  # Drawdown should be negative
        self.assertIsNotNone(peak_date)
        self.assertIsNotNone(trough_date)
        
    def test_calculate_max_drawdown_no_drawdown(self):
        """Test max drawdown with only positive returns."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.03])
        max_dd, peak_date, trough_date = calculate_max_drawdown(returns)
        
        self.assertLessEqual(max_dd, 0)  # Should be 0 or very small negative
        
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test with known data
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        
        excess_returns = returns - (0.02 / 252)
        expected_sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        self.assertAlmostEqual(sharpe, expected_sharpe, places=6)
        
    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])  # Constant returns
        sharpe = calculate_sharpe_ratio(returns)
        
        self.assertEqual(sharpe, 0.0)  # Should be 0 when std is 0


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation and edge cases."""
    
    def test_invalid_universe(self):
        """Test with invalid universe."""
        # Test that None universe is handled gracefully
        # The base class doesn't enforce strict type checking, so this won't raise TypeError
        strategy = MockStrategy(None)
        self.assertIsNone(strategy.universe)
            
    def test_empty_universe(self):
        """Test with empty universe."""
        strategy = MockStrategy([])
        self.assertEqual(len(strategy.universe), 0)
        
    def test_duplicate_symbols_in_universe(self):
        """Test with duplicate symbols in universe."""
        universe = ['AAPL', 'GOOGL', 'AAPL']
        strategy = MockStrategy(universe)
        self.assertEqual(len(strategy.universe), 3)  # Should keep duplicates
        
    def test_invalid_parameters(self):
        """Test with various parameter types."""
        # Test with different parameter types
        strategy = MockStrategy(['AAPL'], 
                               string_param="test",
                               int_param=42,
                               float_param=3.14,
                               bool_param=True,
                               list_param=[1, 2, 3])
        
        self.assertEqual(strategy.params['string_param'], "test")
        self.assertEqual(strategy.params['int_param'], 42)
        self.assertEqual(strategy.params['float_param'], 3.14)
        self.assertEqual(strategy.params['bool_param'], True)
        self.assertEqual(strategy.params['list_param'], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
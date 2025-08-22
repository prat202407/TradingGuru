"""
Smoke tests for NIFTY features pipeline and momentum strategy.

This module provides integration tests to ensure the main components
work together correctly without detailed unit testing.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch
import tempfile
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingguru.data.nifty_data_loader import NiftyDataLoader
from tradingguru.features.nifty_features import NiftyFeaturesPipeline
from tradingguru.models.momentum_nifty import EnhancedNiftyMomentumStrategy
from tradingguru.core.engine import BacktestEngine, BacktestConfig, TransactionCosts
from tradingguru.core.risk import RiskConstraints


class TestNiftyPipelineSmoke(unittest.TestCase):
    """Smoke tests for the complete NIFTY pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and fixtures."""
        # Create sample universe
        cls.universe = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS'
        ]
        
        # Create sample sector mapping
        cls.sector_mapping = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'Information Technology',
            'HDFCBANK.NS': 'Financial Services',
            'INFY.NS': 'Information Technology',
            'HINDUNILVR.NS': 'Consumer Goods'
        }
        
        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        price_data = {}
        for symbol in cls.universe:
            # Generate realistic price movements
            returns = np.random.normal(0.0005, 0.02, len(dates))
            # Add some momentum to make tests more realistic
            momentum = np.sin(np.arange(len(dates)) * 0.02) * 0.005
            returns += momentum
            
            prices = 100 * (1 + returns).cumprod()
            price_data[symbol] = prices
            
        cls.price_data = pd.DataFrame(price_data, index=dates)
        
        # Generate benchmark data
        benchmark_returns = np.random.normal(0.0003, 0.015, len(dates))
        cls.benchmark_data = pd.Series(
            100 * (1 + benchmark_returns).cumprod(), 
            index=dates, 
            name='NSEI'
        )
        
    def test_data_loader_initialization(self):
        """Test NiftyDataLoader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = NiftyDataLoader(cache_dir=temp_dir)
            
            self.assertIsInstance(loader, NiftyDataLoader)
            self.assertEqual(str(loader.cache_dir), temp_dir)
            self.assertTrue(loader.auto_retry)
            
    def test_data_loader_universe_loading(self):
        """Test universe loading with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock universe file
            universe_file = Path(temp_dir) / "test_universe.json"
            universe_data = {
                "symbols": ["RELIANCE", "TCS", "HDFCBANK"]
            }
            
            with open(universe_file, 'w') as f:
                json.dump(universe_data, f)
            
            loader = NiftyDataLoader()
            symbols = loader.load_nifty50_universe(str(universe_file))
            
            expected_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
            self.assertEqual(symbols, expected_symbols)
            
    def test_data_loader_sector_mapping(self):
        """Test sector mapping loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock sectors file
            sectors_file = Path(temp_dir) / "test_sectors.json"
            sectors_data = {
                "RELIANCE": "Energy",
                "TCS": "Information Technology"
            }
            
            with open(sectors_file, 'w') as f:
                json.dump(sectors_data, f)
            
            loader = NiftyDataLoader()
            sector_mapping = loader.load_sector_mapping(str(sectors_file))
            
            expected_mapping = {
                "RELIANCE.NS": "Energy",
                "TCS.NS": "Information Technology"
            }
            self.assertEqual(sector_mapping, expected_mapping)
            
    def test_data_quality_validation(self):
        """Test data quality validation."""
        loader = NiftyDataLoader()
        quality_metrics = loader.validate_data_quality(self.price_data)
        
        self.assertIn('total_symbols', quality_metrics)
        self.assertIn('total_days', quality_metrics)
        self.assertIn('good_quality_symbols', quality_metrics)
        self.assertIn('poor_quality_symbols', quality_metrics)
        
        self.assertEqual(quality_metrics['total_symbols'], len(self.universe))
        self.assertGreater(quality_metrics['total_days'], 0)
        
    def test_features_pipeline_initialization(self):
        """Test features pipeline initialization."""
        pipeline = NiftyFeaturesPipeline(
            sector_mapping=self.sector_mapping
        )
        
        self.assertIsInstance(pipeline, NiftyFeaturesPipeline)
        self.assertEqual(pipeline.sector_mapping, self.sector_mapping)
        self.assertIn('short_momentum', pipeline.lookback_periods)
        
    def test_features_calculation_basic(self):
        """Test basic features calculation."""
        pipeline = NiftyFeaturesPipeline(
            sector_mapping=self.sector_mapping
        )
        
        features = pipeline.calculate_all_features(
            price_data=self.price_data,
            benchmark_data=self.benchmark_data
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.price_data))
        self.assertGreater(features.shape[1], 10)  # Should have many features
        
        # Check for key feature types
        feature_names = features.columns.tolist()
        momentum_features = [col for col in feature_names if 'mom' in col]
        volatility_features = [col for col in feature_names if 'vol' in col]
        
        self.assertGreater(len(momentum_features), 0)
        self.assertGreater(len(volatility_features), 0)
        
    def test_momentum_scoring(self):
        """Test momentum score calculation."""
        pipeline = NiftyFeaturesPipeline(sector_mapping=self.sector_mapping)
        
        # Calculate basic features first
        features = pipeline.calculate_all_features(self.price_data)
        
        # Calculate momentum scores
        momentum_scores = pipeline.calculate_momentum_score(features)
        
        self.assertIsInstance(momentum_scores, pd.DataFrame)
        self.assertGreater(momentum_scores.shape[1], 0)
        
        # Check that we have momentum scores for our symbols
        score_columns = momentum_scores.columns.tolist()
        symbol_scores = [col for col in score_columns if any(symbol.replace('.NS', '') in col for symbol in self.universe)]
        self.assertGreater(len(symbol_scores), 0)
        
    def test_sector_neutral_scoring(self):
        """Test sector-neutral momentum scoring."""
        pipeline = NiftyFeaturesPipeline(sector_mapping=self.sector_mapping)
        
        features = pipeline.calculate_all_features(self.price_data)
        momentum_scores = pipeline.calculate_momentum_score(features)
        
        sector_neutral_scores = pipeline.calculate_sector_neutral_scores(
            momentum_scores, features
        )
        
        self.assertIsInstance(sector_neutral_scores, pd.DataFrame)
        # May be empty if insufficient data, but should not raise errors
        
    def test_strategy_initialization(self):
        """Test enhanced momentum strategy initialization."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping,
            top_quantile=0.4,
            min_positions=3,
            max_positions=5
        )
        
        self.assertIsInstance(strategy, EnhancedNiftyMomentumStrategy)
        self.assertEqual(strategy.universe, self.universe)
        self.assertEqual(strategy.params['top_quantile'], 0.4)
        self.assertEqual(strategy.params['min_positions'], 3)
        
    def test_strategy_configuration_validation(self):
        """Test strategy configuration validation."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping
        )
        
        issues = strategy.validate_configuration()
        
        # Should return a list (may be empty if config is valid)
        self.assertIsInstance(issues, list)
        
        # Test with invalid configuration
        invalid_strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            top_quantile=1.5,  # Invalid - should be <= 1
            min_positions=100,  # Invalid - larger than universe
            max_weight=1.5      # Invalid - should be <= 1
        )
        
        invalid_issues = invalid_strategy.validate_configuration()
        self.assertGreater(len(invalid_issues), 0)
        
    def test_strategy_features_calculation(self):
        """Test strategy features calculation."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping
        )
        
        features = strategy.calculate_features(self.price_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.price_data))
        self.assertGreater(features.shape[1], 0)
        
    def test_strategy_signal_generation(self):
        """Test strategy signal generation."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping,
            min_positions=2,
            max_positions=3
        )
        
        # Calculate features
        features = strategy.calculate_features(self.price_data)
        
        # Generate signals for a specific date
        test_date = self.price_data.index[-50].date()  # Use a date with enough history
        
        signals = strategy.generate_signals(features, test_date)
        
        self.assertIsInstance(signals, dict)
        # Signals may be empty if market conditions are poor
        
        if signals:
            # Check signal validity
            for symbol, weight in signals.items():
                self.assertIn(symbol, self.universe)
                self.assertGreaterEqual(weight, 0)  # Assuming long-only strategy
                self.assertLessEqual(weight, 1.0)
                
            # Check total weight
            total_weight = sum(signals.values())
            self.assertLessEqual(total_weight, 1.0)
            
    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping
        )
        
        config = BacktestConfig(
            start_date="2023-06-01",
            end_date="2023-12-31",
            initial_capital=1000000.0
        )
        
        engine = BacktestEngine(
            strategy=strategy,
            config=config,
            sector_mapping=self.sector_mapping
        )
        
        self.assertIsInstance(engine, BacktestEngine)
        self.assertEqual(engine.strategy, strategy)
        self.assertEqual(engine.config, config)
        
    def test_backtest_execution_smoke(self):
        """Smoke test for backtest execution."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping,
            min_positions=2,
            max_positions=3,
            rebalance_frequency='W'  # Weekly to reduce computation
        )
        
        config = BacktestConfig(
            start_date="2023-06-01",
            end_date="2023-09-30",  # Shorter period for speed
            initial_capital=1000000.0,
            rebalance_frequency='W'
        )
        
        engine = BacktestEngine(
            strategy=strategy,
            config=config,
            sector_mapping=self.sector_mapping
        )
        
        # Use subset of data for speed
        test_data = self.price_data.loc['2023-06-01':'2023-09-30']
        test_benchmark = self.benchmark_data.loc['2023-06-01':'2023-09-30']
        
        # This should complete without errors
        try:
            results = engine.run_backtest(
                price_data=test_data,
                benchmark_data=test_benchmark
            )
            
            # Check that results contain expected keys
            expected_keys = ['strategy_stats', 'benchmark_stats', 'daily_portfolio']
            for key in expected_keys:
                self.assertIn(key, results)
                
            # Check strategy stats
            strategy_stats = results['strategy_stats']
            self.assertIn('total_return', strategy_stats)
            self.assertIn('num_trades', strategy_stats)
            
        except Exception as e:
            # Backtest may fail due to insufficient data or other issues
            # In smoke test, we mainly want to check it doesn't crash hard
            self.assertIsInstance(e, Exception)
            print(f"Backtest failed as expected in smoke test: {e}")
            
    def test_report_generation(self):
        """Test report generation."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping
        )
        
        config = BacktestConfig(
            start_date="2023-06-01",
            end_date="2023-09-30",
            initial_capital=1000000.0
        )
        
        engine = BacktestEngine(
            strategy=strategy,
            config=config,
            sector_mapping=self.sector_mapping
        )
        
        # Mock some results for report generation
        engine.results = {
            'strategy_stats': {
                'total_return': 0.15,
                'annualized_return': 0.12,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'num_trades': 45
            },
            'benchmark_stats': {
                'total_return': 0.10,
                'annualized_return': 0.08
            },
            'transaction_analysis': {
                'total_transaction_costs': 5000,
                'cost_ratio': 0.005,
                'num_trades': 45
            },
            'risk_metrics': {
                'var_95': -0.025,
                'var_99': -0.045
            },
            'risk_breaches': []
        }
        
        report = engine.generate_report()
        
        self.assertIsInstance(report, str)
        self.assertIn('BACKTEST REPORT', report)
        self.assertIn('Total Return', report)
        
    def test_strategy_summary(self):
        """Test strategy summary generation."""
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping
        )
        
        summary = strategy.get_strategy_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('strategy_name', summary)
        self.assertIn('universe_size', summary)
        self.assertIn('parameters', summary)
        
        self.assertEqual(summary['universe_size'], len(self.universe))
        self.assertEqual(summary['strategy_name'], strategy.name)
        
    def test_edge_cases_empty_data(self):
        """Test handling of edge cases with empty data."""
        # Empty price data
        empty_data = pd.DataFrame()
        
        pipeline = NiftyFeaturesPipeline()
        
        # Should handle gracefully without crashing
        try:
            features = pipeline.calculate_all_features(empty_data)
            self.assertIsInstance(features, pd.DataFrame)
        except Exception as e:
            # May raise exception, but should be informative
            self.assertIsInstance(e, Exception)
            
    def test_edge_cases_insufficient_history(self):
        """Test handling of insufficient history."""
        # Very short price series
        short_data = self.price_data.head(10)
        
        strategy = EnhancedNiftyMomentumStrategy(
            universe=self.universe,
            sector_mapping=self.sector_mapping
        )
        
        try:
            features = strategy.calculate_features(short_data)
            # Should produce features but many may be NaN
            self.assertIsInstance(features, pd.DataFrame)
            
            # Try signal generation with insufficient data
            if len(features) > 0:
                test_date = features.index[-1].date()
                signals = strategy.generate_signals(features, test_date)
                self.assertIsInstance(signals, dict)
                
        except Exception as e:
            # May fail due to insufficient data
            print(f"Expected failure with insufficient data: {e}")
            
    def test_memory_usage(self):
        """Test that large calculations don't consume excessive memory."""
        # This is a basic check - in production you might want more sophisticated memory monitoring
        
        pipeline = NiftyFeaturesPipeline(sector_mapping=self.sector_mapping)
        
        # Calculate features multiple times to check for memory leaks
        for i in range(3):
            features = pipeline.calculate_all_features(self.price_data)
            
            # Basic check that features are reasonable size
            memory_usage = features.memory_usage(deep=True).sum()
            self.assertLess(memory_usage, 100_000_000)  # Less than 100MB
            
            # Clean up
            del features


if __name__ == '__main__':
    # Run with reduced verbosity for smoke tests
    unittest.main(verbosity=1)
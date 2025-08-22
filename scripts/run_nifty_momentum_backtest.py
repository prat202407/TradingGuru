#!/usr/bin/env python3
"""
Single backtest runner for NIFTY 50 momentum strategy.

This script runs a backtest for the enhanced NIFTY momentum strategy
using configuration from a YAML file.
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime, date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingguru.data.nifty_data_loader import NiftyDataLoader
from tradingguru.models.momentum_nifty import EnhancedNiftyMomentumStrategy
from tradingguru.core.engine import BacktestEngine, BacktestConfig, TransactionCosts
from tradingguru.core.risk import RiskConstraints


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_backtest_config(config: dict) -> BacktestConfig:
    """Create BacktestConfig from configuration dictionary."""
    # Transaction costs
    tc_config = config.get('transaction_costs', {})
    transaction_costs = TransactionCosts(
        commission_rate=tc_config.get('commission_rate', 0.001),
        bid_ask_spread=tc_config.get('bid_ask_spread', 0.0005),
        market_impact=tc_config.get('market_impact', 0.0001),
        min_commission=tc_config.get('min_commission', 0.0)
    )
    
    # Risk constraints
    risk_config = config.get('risk_constraints', {})
    risk_constraints = RiskConstraints(
        max_position_weight=risk_config.get('max_position_weight', 0.05),
        max_sector_weight=risk_config.get('max_sector_weight', 0.30),
        max_leverage=risk_config.get('max_leverage', 1.0),
        min_diversification=risk_config.get('min_diversification', 10),
        max_turnover=risk_config.get('max_turnover', 2.0),
        max_drawdown_limit=risk_config.get('max_drawdown_limit', -0.20),
        volatility_target=risk_config.get('volatility_target'),
        var_limit=risk_config.get('var_limit')
    )
    
    # Backtest configuration
    bt_config = config.get('backtest', {})
    backtest_config = BacktestConfig(
        start_date=bt_config['start_date'],
        end_date=bt_config['end_date'],
        initial_capital=bt_config.get('initial_capital', 1000000.0),
        rebalance_frequency=bt_config.get('rebalance_frequency', 'D'),
        benchmark_symbol=bt_config.get('benchmark_symbol', '^NSEI'),
        transaction_costs=transaction_costs,
        risk_constraints=risk_constraints
    )
    
    return backtest_config


def create_strategy(config: dict, universe: list, sector_mapping: dict) -> EnhancedNiftyMomentumStrategy:
    """Create strategy from configuration."""
    strategy_config = config.get('strategy', {})
    
    strategy = EnhancedNiftyMomentumStrategy(
        universe=universe,
        sector_mapping=sector_mapping,
        **strategy_config
    )
    
    return strategy


def run_backtest(config_path: str, output_dir: str = "results"):
    """Run the backtest with given configuration."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting backtest with config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize data loader
    data_loader = NiftyDataLoader(
        cache_dir=config.get('data', {}).get('cache_dir', 'data/cache'),
        auto_retry=True
    )
    
    # Load universe and sector mapping
    universe_file = config.get('data', {}).get('universe_file', 'data/universe/nifty50.json')
    sectors_file = config.get('data', {}).get('sectors_file', 'data/universe/nifty50_sectors.json')
    
    logger.info("Loading universe and sector mapping...")
    universe = data_loader.load_nifty50_universe(universe_file)
    sector_mapping = data_loader.load_sector_mapping(sectors_file)
    
    logger.info(f"Universe: {len(universe)} symbols")
    logger.info(f"Sector mapping: {len(sector_mapping)} symbols")
    
    # Create strategy
    strategy = create_strategy(config, universe, sector_mapping)
    logger.info(f"Strategy created: {strategy}")
    
    # Validate strategy configuration
    issues = strategy.validate_configuration()
    if issues:
        logger.warning("Strategy configuration issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Create backtest configuration
    backtest_config = create_backtest_config(config)
    logger.info(f"Backtest period: {backtest_config.start_date} to {backtest_config.end_date}")
    
    # Fetch data
    logger.info("Fetching market data...")
    price_data, benchmark_data = data_loader.fetch_complete_dataset(
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
        universe_file=universe_file,
        use_cache=config.get('data', {}).get('use_cache', True)
    )
    
    logger.info(f"Price data shape: {price_data.shape}")
    logger.info(f"Benchmark data length: {len(benchmark_data)}")
    
    # Validate data quality
    quality_metrics = data_loader.validate_data_quality(price_data)
    logger.info(f"Data quality: {len(quality_metrics['good_quality_symbols'])} good quality symbols")
    
    if len(quality_metrics['poor_quality_symbols']) > 0:
        logger.warning(f"Poor quality symbols: {quality_metrics['poor_quality_symbols']}")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        config=backtest_config,
        sector_mapping=sector_mapping
    )
    
    # Run backtest
    logger.info("Starting backtest execution...")
    start_time = datetime.now()
    
    results = engine.run_backtest(
        price_data=price_data,
        benchmark_data=benchmark_data
    )
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.info(f"Backtest completed in {execution_time}")
    
    # Generate and save results
    logger.info("Generating results...")
    
    # Performance report
    report = engine.generate_report()
    report_file = output_path / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to: {report_file}")
    
    # Save detailed results
    results_file = output_path / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Save daily portfolio data
    if not results['daily_portfolio'].empty:
        portfolio_file = output_path / f"daily_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['daily_portfolio'].to_csv(portfolio_file)
        logger.info(f"Daily portfolio data saved to: {portfolio_file}")
    
    # Save trades log
    if not results['trades_log'].empty:
        trades_file = output_path / f"trades_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['trades_log'].to_csv(trades_file, index=False)
        logger.info(f"Trades log saved to: {trades_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    
    strategy_stats = results['strategy_stats']
    print(f"Total Return: {strategy_stats.get('total_return', 0):.2%}")
    print(f"Annualized Return: {strategy_stats.get('annualized_return', 0):.2%}")
    print(f"Sharpe Ratio: {strategy_stats.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {strategy_stats.get('max_drawdown', 0):.2%}")
    print(f"Number of Trades: {strategy_stats.get('num_trades', 0):,.0f}")
    
    if results['benchmark_stats']:
        benchmark_stats = results['benchmark_stats']
        print(f"\nBenchmark Return: {benchmark_stats.get('total_return', 0):.2%}")
        print(f"Alpha: {strategy_stats.get('alpha', 0):.2%}")
        print(f"Information Ratio: {strategy_stats.get('information_ratio', 0):.3f}")
    
    print(f"\nExecution Time: {execution_time}")
    print(f"Results saved to: {output_path}")
    print("="*80)
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run NIFTY 50 momentum strategy backtest")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/nifty_momentum_backtest.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Run backtest
        results = run_backtest(args.config, args.output)
        
        print("\nBacktest completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Parameter sweep utility for NIFTY 50 momentum strategy.

This script runs multiple backtests with different parameter combinations
to find optimal strategy configurations.
"""

import sys
import os
import argparse
import yaml
import logging
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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
            logging.FileHandler(f'param_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_parameter_combinations(param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations


def create_strategy_config(base_config: dict, param_overrides: dict) -> dict:
    """Create strategy configuration with parameter overrides."""
    strategy_config = base_config.get('strategy', {}).copy()
    strategy_config.update(param_overrides)
    
    config = base_config.copy()
    config['strategy'] = strategy_config
    
    return config


def run_single_backtest(args: Tuple[dict, int, dict, str]) -> Dict[str, Any]:
    """Run a single backtest with given parameters."""
    config, run_id, param_overrides, temp_dir = args
    
    # Setup logging for this process
    logger = logging.getLogger(f"backtest_{run_id}")
    
    try:
        # Create strategy configuration with overrides
        test_config = create_strategy_config(config, param_overrides)
        
        # Initialize data loader
        data_loader = NiftyDataLoader(
            cache_dir=config.get('data', {}).get('cache_dir', 'data/cache'),
            auto_retry=False  # Disable retry in parallel runs
        )
        
        # Load universe and sector mapping (use cached data)
        universe_file = config.get('data', {}).get('universe_file', 'data/universe/nifty50.json')
        sectors_file = config.get('data', {}).get('sectors_file', 'data/universe/nifty50_sectors.json')
        
        universe = data_loader.load_nifty50_universe(universe_file)
        sector_mapping = data_loader.load_sector_mapping(sectors_file)
        
        # Create strategy
        strategy = EnhancedNiftyMomentumStrategy(
            universe=universe,
            sector_mapping=sector_mapping,
            **test_config['strategy']
        )
        
        # Create backtest configuration
        tc_config = config.get('transaction_costs', {})
        transaction_costs = TransactionCosts(
            commission_rate=tc_config.get('commission_rate', 0.001),
            bid_ask_spread=tc_config.get('bid_ask_spread', 0.0005),
            market_impact=tc_config.get('market_impact', 0.0001),
            min_commission=tc_config.get('min_commission', 0.0)
        )
        
        risk_config = config.get('risk_constraints', {})
        risk_constraints = RiskConstraints(
            max_position_weight=risk_config.get('max_position_weight', 0.05),
            max_sector_weight=risk_config.get('max_sector_weight', 0.30),
            max_leverage=risk_config.get('max_leverage', 1.0),
            min_diversification=risk_config.get('min_diversification', 10),
            max_turnover=risk_config.get('max_turnover', 2.0),
            max_drawdown_limit=risk_config.get('max_drawdown_limit', -0.20)
        )
        
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
        
        # Fetch data (this should hit cache)
        price_data, benchmark_data = data_loader.fetch_complete_dataset(
            start_date=backtest_config.start_date,
            end_date=backtest_config.end_date,
            universe_file=universe_file,
            use_cache=True
        )
        
        # Initialize backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            config=backtest_config,
            sector_mapping=sector_mapping
        )
        
        # Run backtest
        start_time = datetime.now()
        results = engine.run_backtest(
            price_data=price_data,
            benchmark_data=benchmark_data
        )
        end_time = datetime.now()
        
        # Extract key metrics
        strategy_stats = results['strategy_stats']
        benchmark_stats = results['benchmark_stats']
        
        result = {
            'run_id': run_id,
            'parameters': param_overrides,
            'execution_time': (end_time - start_time).total_seconds(),
            'total_return': strategy_stats.get('total_return', 0),
            'annualized_return': strategy_stats.get('annualized_return', 0),
            'annualized_volatility': strategy_stats.get('annualized_volatility', 0),
            'sharpe_ratio': strategy_stats.get('sharpe_ratio', 0),
            'max_drawdown': strategy_stats.get('max_drawdown', 0),
            'win_rate': strategy_stats.get('win_rate', 0),
            'num_trades': strategy_stats.get('num_trades', 0),
            'information_ratio': strategy_stats.get('information_ratio', 0),
            'alpha': strategy_stats.get('alpha', 0),
            'beta': strategy_stats.get('beta', 0),
            'benchmark_return': benchmark_stats.get('total_return', 0) if benchmark_stats else 0,
            'success': True,
            'error': None
        }
        
        # Add transaction cost metrics if available
        if results['transaction_analysis']:
            tc = results['transaction_analysis']
            result.update({
                'total_transaction_costs': tc.get('total_transaction_costs', 0),
                'cost_ratio': tc.get('cost_ratio', 0)
            })
        
        logger.info(f"Run {run_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Run {run_id} failed: {str(e)}")
        return {
            'run_id': run_id,
            'parameters': param_overrides,
            'success': False,
            'error': str(e),
            'total_return': np.nan,
            'annualized_return': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan
        }


def run_parameter_sweep(config_path: str, 
                       param_ranges: Dict[str, List],
                       output_dir: str = "param_sweep_results",
                       n_jobs: int = None) -> pd.DataFrame:
    """Run parameter sweep with multiple parameter combinations."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting parameter sweep with config: {config_path}")
    
    # Load base configuration
    config = load_config(config_path)
    logger.info("Base configuration loaded successfully")
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(param_ranges)
    logger.info(f"Generated {len(param_combinations)} parameter combinations")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Pre-cache data by running a dummy data fetch
    logger.info("Pre-caching market data...")
    data_loader = NiftyDataLoader(
        cache_dir=config.get('data', {}).get('cache_dir', 'data/cache'),
        auto_retry=True
    )
    
    bt_config = config.get('backtest', {})
    universe_file = config.get('data', {}).get('universe_file', 'data/universe/nifty50.json')
    
    try:
        price_data, benchmark_data = data_loader.fetch_complete_dataset(
            start_date=bt_config['start_date'],
            end_date=bt_config['end_date'],
            universe_file=universe_file,
            use_cache=True
        )
        logger.info(f"Data cached successfully: {price_data.shape}")
    except Exception as e:
        logger.error(f"Failed to cache data: {e}")
        raise
    
    # Setup parallel execution
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, len(param_combinations))
    
    logger.info(f"Running {len(param_combinations)} backtests using {n_jobs} processes")
    
    # Prepare arguments for parallel execution
    args_list = [
        (config, i, params, str(output_path))
        for i, params in enumerate(param_combinations)
    ]
    
    # Run backtests in parallel
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        future_to_args = {
            executor.submit(run_single_backtest, args): args[1] 
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_args):
            run_id = future_to_args[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:
                    progress = completed / len(param_combinations) * 100
                    logger.info(f"Progress: {completed}/{len(param_combinations)} ({progress:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Failed to get result for run {run_id}: {e}")
                results.append({
                    'run_id': run_id,
                    'success': False,
                    'error': str(e),
                    'total_return': np.nan,
                    'sharpe_ratio': np.nan
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f"parameter_sweep_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file}")
    
    # Generate summary report
    generate_sweep_report(results_df, param_ranges, output_path, timestamp)
    
    return results_df


def generate_sweep_report(results_df: pd.DataFrame, 
                         param_ranges: Dict[str, List],
                         output_path: Path,
                         timestamp: str):
    """Generate parameter sweep summary report."""
    logger = logging.getLogger(__name__)
    
    # Filter successful runs
    successful_runs = results_df[results_df['success'] == True].copy()
    
    if len(successful_runs) == 0:
        logger.warning("No successful runs to analyze")
        return
    
    report = []
    report.append("="*80)
    report.append("PARAMETER SWEEP REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total runs: {len(results_df)}")
    report.append(f"Successful runs: {len(successful_runs)}")
    report.append(f"Failed runs: {len(results_df) - len(successful_runs)}")
    report.append("")
    
    # Parameter ranges tested
    report.append("PARAMETER RANGES TESTED:")
    for param, values in param_ranges.items():
        report.append(f"  {param}: {values}")
    report.append("")
    
    # Best performing configurations
    metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'information_ratio']
    
    for metric in metrics:
        if metric in successful_runs.columns:
            report.append(f"BEST {metric.upper().replace('_', ' ')}:")
            best_run = successful_runs.loc[successful_runs[metric].idxmax()]
            report.append(f"  Value: {best_run[metric]:.4f}")
            report.append(f"  Parameters: {best_run['parameters']}")
            report.append("")
    
    # Statistics summary
    report.append("PERFORMANCE STATISTICS:")
    for metric in metrics:
        if metric in successful_runs.columns and not successful_runs[metric].isna().all():
            values = successful_runs[metric].dropna()
            report.append(f"  {metric}:")
            report.append(f"    Mean: {values.mean():.4f}")
            report.append(f"    Std:  {values.std():.4f}")
            report.append(f"    Min:  {values.min():.4f}")
            report.append(f"    Max:  {values.max():.4f}")
            report.append("")
    
    # Parameter sensitivity analysis
    report.append("PARAMETER SENSITIVITY:")
    for param in param_ranges.keys():
        param_col = f"parameters.{param}"
        # Extract parameter values from the parameters dict
        param_values = successful_runs['parameters'].apply(lambda x: x.get(param))
        
        if len(param_values.unique()) > 1:
            correlations = {}
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
                if metric in successful_runs.columns:
                    corr = param_values.corr(successful_runs[metric])
                    if not np.isnan(corr):
                        correlations[metric] = corr
            
            if correlations:
                report.append(f"  {param}:")
                for metric, corr in correlations.items():
                    report.append(f"    vs {metric}: {corr:+.3f}")
                report.append("")
    
    # Top 10 configurations
    if len(successful_runs) >= 10:
        report.append("TOP 10 CONFIGURATIONS (by Sharpe Ratio):")
        top_configs = successful_runs.nlargest(10, 'sharpe_ratio')
        
        for i, (idx, row) in enumerate(top_configs.iterrows(), 1):
            report.append(f"  {i}. Sharpe: {row['sharpe_ratio']:.3f}, "
                         f"Return: {row['total_return']:.2%}, "
                         f"MaxDD: {row['max_drawdown']:.2%}")
            report.append(f"     Params: {row['parameters']}")
        report.append("")
    
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    report_file = output_path / f"parameter_sweep_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Summary report saved to: {report_file}")
    
    # Print key results
    print("\n" + "="*60)
    print("PARAMETER SWEEP SUMMARY")
    print("="*60)
    print(f"Successful runs: {len(successful_runs)}/{len(results_df)}")
    
    if len(successful_runs) > 0:
        best_sharpe = successful_runs.loc[successful_runs['sharpe_ratio'].idxmax()]
        print(f"Best Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f}")
        print(f"Best Configuration: {best_sharpe['parameters']}")
        
        best_return = successful_runs.loc[successful_runs['total_return'].idxmax()]
        print(f"Best Total Return: {best_return['total_return']:.2%}")
        print(f"Best Return Config: {best_return['parameters']}")
    
    print(f"Detailed results: {report_file}")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run NIFTY 50 momentum strategy parameter sweep")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/nifty_momentum_backtest.yaml",
        help="Path to base configuration YAML file"
    )
    parser.add_argument(
        "--param-config", 
        type=str, 
        required=True,
        help="Path to parameter ranges YAML file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="param_sweep_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--jobs", 
        type=int, 
        default=None,
        help="Number of parallel jobs (default: CPU count - 1)"
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
        # Load parameter ranges
        with open(args.param_config, 'r') as f:
            param_config = yaml.safe_load(f)
        
        param_ranges = param_config.get('parameter_ranges', {})
        if not param_ranges:
            raise ValueError("No parameter ranges found in configuration")
        
        # Run parameter sweep
        results_df = run_parameter_sweep(
            config_path=args.config,
            param_ranges=param_ranges,
            output_dir=args.output,
            n_jobs=args.jobs
        )
        
        print(f"\nParameter sweep completed successfully!")
        print(f"Results saved to: {args.output}")
        return 0
        
    except Exception as e:
        logging.error(f"Parameter sweep failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
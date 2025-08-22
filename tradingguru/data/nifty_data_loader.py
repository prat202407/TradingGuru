"""
NIFTY 50 data loader for TradingGuru.

This module provides utilities to fetch daily OHLCV data for NIFTY 50
constituents and the benchmark index from Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import logging
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class NiftyDataLoader:
    """
    Data loader for NIFTY 50 constituents and benchmark index.
    
    This class handles fetching, caching, and preprocessing of daily OHLCV data
    for Indian equity markets using Yahoo Finance.
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 auto_retry: bool = True,
                 retry_delay: float = 1.0):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
            auto_retry: Whether to automatically retry failed downloads
            retry_delay: Delay between retry attempts in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_retry = auto_retry
        self.retry_delay = retry_delay
        
        # Yahoo Finance suffixes for Indian stocks
        self.nse_suffix = ".NS"
        self.benchmark_symbol = "^NSEI"
        
    def load_nifty50_universe(self, universe_file: Optional[str] = None) -> List[str]:
        """
        Load NIFTY 50 universe from JSON file.
        
        Args:
            universe_file: Path to universe JSON file
            
        Returns:
            List of stock symbols
        """
        if universe_file is None:
            universe_file = "data/universe/nifty50.json"
            
        try:
            with open(universe_file, 'r') as f:
                universe_data = json.load(f)
                
            if isinstance(universe_data, dict):
                symbols = universe_data.get('symbols', [])
            elif isinstance(universe_data, list):
                symbols = universe_data
            else:
                raise ValueError("Invalid universe file format")
                
            # Add .NS suffix for Yahoo Finance
            symbols_with_suffix = [
                symbol if symbol.endswith('.NS') else f"{symbol}.NS"
                for symbol in symbols
            ]
            
            logger.info(f"Loaded {len(symbols_with_suffix)} symbols from {universe_file}")
            return symbols_with_suffix
            
        except FileNotFoundError:
            logger.warning(f"Universe file {universe_file} not found, using default NIFTY 50")
            return self._get_default_nifty50()
        except Exception as e:
            logger.error(f"Error loading universe file: {e}")
            return self._get_default_nifty50()
            
    def _get_default_nifty50(self) -> List[str]:
        """Get default NIFTY 50 symbols."""
        # Default NIFTY 50 constituents (as of 2024)
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "ASIANPAINT",
            "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
            "BAJFINANCE", "WIPRO", "ONGC", "HCLTECH", "NTPC", "BAJAJFINSV",
            "POWERGRID", "TATAMOTORS", "TECHM", "COALINDIA", "INDUSINDBK",
            "DRREDDY", "JSWSTEEL", "GRASIM", "HINDALCO", "ADANIENT", "TATASTEEL",
            "CIPLA", "BRITANNIA", "EICHERMOT", "APOLLOHOSP", "BPCL", "DIVISLAB",
            "HEROMOTOCO", "UPL", "ADANIPORTS", "BAJAJ-AUTO", "SHRIRAMFIN",
            "TATACONSUM", "SBILIFE", "HDFCLIFE", "LTIM"
        ]
        
        return [f"{symbol}.NS" for symbol in symbols]
        
    def fetch_price_data(self, 
                        symbols: List[str],
                        start_date: Union[str, date],
                        end_date: Union[str, date],
                        use_cache: bool = True,
                        price_column: str = 'Adj Close') -> pd.DataFrame:
        """
        Fetch price data for given symbols.
        
        Args:
            symbols: List of Yahoo Finance symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            use_cache: Whether to use cached data
            price_column: Which price column to use ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns:
            DataFrame with symbols as columns and dates as index
        """
        logger.info(f"Fetching price data for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Check cache first
        cache_file = self.cache_dir / f"prices_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        
        if use_cache and cache_file.exists():
            try:
                logger.info("Loading data from cache...")
                cached_data = pd.read_pickle(cache_file)
                
                # Check if all symbols are in cache
                missing_symbols = set(symbols) - set(cached_data.columns)
                if not missing_symbols:
                    return cached_data[symbols]
                else:
                    logger.info(f"Cache missing {len(missing_symbols)} symbols, fetching fresh data")
                    
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        # Fetch fresh data
        price_data = {}
        failed_symbols = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.debug(f"Fetching {symbol} ({i+1}/{len(symbols)})")
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),  # Include end date
                    auto_adjust=False,
                    prepost=False
                )
                
                if hist.empty:
                    logger.warning(f"No data found for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Use specified price column
                if price_column in hist.columns:
                    price_data[symbol] = hist[price_column]
                else:
                    logger.warning(f"Column {price_column} not found for {symbol}, using Close")
                    price_data[symbol] = hist['Close']
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                failed_symbols.append(symbol)
                
                if self.auto_retry:
                    logger.info(f"Retrying {symbol} after {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                        
                        if not hist.empty:
                            price_data[symbol] = hist[price_column] if price_column in hist.columns else hist['Close']
                            failed_symbols.remove(symbol)
                            logger.info(f"Successfully retried {symbol}")
                            
                    except Exception as retry_error:
                        logger.error(f"Retry failed for {symbol}: {retry_error}")
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        if not price_data:
            raise ValueError("No price data was successfully fetched")
        
        # Combine into DataFrame
        df = pd.DataFrame(price_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Remove columns with too much missing data
        missing_pct = df.isnull().sum() / len(df)
        good_symbols = missing_pct[missing_pct < 0.5].index.tolist()
        
        if len(good_symbols) < len(df.columns):
            removed = set(df.columns) - set(good_symbols)
            logger.warning(f"Removing symbols with >50% missing data: {removed}")
            df = df[good_symbols]
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Cache the data
        if use_cache:
            try:
                df.to_pickle(cache_file)
                logger.info(f"Cached data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        logger.info(f"Successfully fetched data for {len(df.columns)} symbols")
        return df
        
    def fetch_benchmark_data(self, 
                           start_date: Union[str, date],
                           end_date: Union[str, date],
                           use_cache: bool = True) -> pd.Series:
        """
        Fetch benchmark (NSEI) price data.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            use_cache: Whether to use cached data
            
        Returns:
            Series with benchmark prices
        """
        logger.info(f"Fetching benchmark data ({self.benchmark_symbol})")
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Check cache
        cache_file = self.cache_dir / f"benchmark_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        
        if use_cache and cache_file.exists():
            try:
                logger.info("Loading benchmark data from cache...")
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Error loading benchmark cache: {e}")
        
        # Fetch fresh data
        try:
            ticker = yf.Ticker(self.benchmark_symbol)
            hist = ticker.history(
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=False,
                prepost=False
            )
            
            if hist.empty:
                raise ValueError(f"No benchmark data found for {self.benchmark_symbol}")
            
            benchmark_series = hist['Adj Close']
            benchmark_series.index = pd.to_datetime(benchmark_series.index)
            benchmark_series = benchmark_series.sort_index()
            
            # Cache the data
            if use_cache:
                try:
                    benchmark_series.to_pickle(cache_file)
                    logger.info(f"Cached benchmark data to {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to cache benchmark data: {e}")
            
            logger.info(f"Successfully fetched benchmark data: {len(benchmark_series)} points")
            return benchmark_series
            
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {e}")
            raise
            
    def fetch_complete_dataset(self, 
                             start_date: Union[str, date],
                             end_date: Union[str, date],
                             universe_file: Optional[str] = None,
                             use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fetch complete dataset with both universe and benchmark data.
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            universe_file: Path to universe JSON file
            use_cache: Whether to use cached data
            
        Returns:
            Tuple of (price_data_df, benchmark_series)
        """
        logger.info("Fetching complete dataset...")
        
        # Load universe
        symbols = self.load_nifty50_universe(universe_file)
        
        # Fetch price data
        price_data = self.fetch_price_data(
            symbols, start_date, end_date, use_cache=use_cache
        )
        
        # Fetch benchmark data
        benchmark_data = self.fetch_benchmark_data(
            start_date, end_date, use_cache=use_cache
        )
        
        # Align dates
        common_dates = price_data.index.intersection(benchmark_data.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between price data and benchmark")
        
        price_data = price_data.loc[common_dates]
        benchmark_data = benchmark_data.loc[common_dates]
        
        logger.info(f"Dataset ready: {len(price_data.columns)} symbols, {len(price_data)} days")
        return price_data, benchmark_data
        
    def get_ohlcv_data(self,
                      symbols: List[str],
                      start_date: Union[str, date],
                      end_date: Union[str, date],
                      use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete OHLCV data for given symbols.
        
        Args:
            symbols: List of Yahoo Finance symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to OHLCV DataFrames
        """
        logger.info(f"Fetching OHLCV data for {len(symbols)} symbols")
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        ohlcv_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    auto_adjust=False,
                    prepost=False
                )
                
                if hist.empty:
                    logger.warning(f"No OHLCV data found for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Ensure we have all OHLCV columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in hist.columns for col in required_columns):
                    logger.warning(f"Missing OHLCV columns for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                ohlcv_data[symbol] = hist[required_columns + ['Adj Close']]
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to fetch OHLCV for {len(failed_symbols)} symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched OHLCV data for {len(ohlcv_data)} symbols")
        return ohlcv_data
        
    def load_sector_mapping(self, sectors_file: Optional[str] = None) -> Dict[str, str]:
        """
        Load sector mapping from JSON file.
        
        Args:
            sectors_file: Path to sectors JSON file
            
        Returns:
            Dictionary mapping symbols to sectors
        """
        if sectors_file is None:
            sectors_file = "data/universe/nifty50_sectors.json"
            
        try:
            with open(sectors_file, 'r') as f:
                sector_mapping = json.load(f)
                
            # Add .NS suffix to symbols if not present
            standardized_mapping = {}
            for symbol, sector in sector_mapping.items():
                symbol_with_suffix = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
                standardized_mapping[symbol_with_suffix] = sector
                
            logger.info(f"Loaded sector mapping for {len(standardized_mapping)} symbols")
            return standardized_mapping
            
        except FileNotFoundError:
            logger.warning(f"Sectors file {sectors_file} not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading sectors file: {e}")
            return {}
            
    def validate_data_quality(self, 
                             price_data: pd.DataFrame,
                             min_history_days: int = 252) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            price_data: Price data DataFrame
            min_history_days: Minimum required history days
            
        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Validating data quality...")
        
        quality_metrics = {
            'total_symbols': len(price_data.columns),
            'total_days': len(price_data),
            'date_range': (price_data.index.min(), price_data.index.max()),
            'missing_data_pct': {},
            'zero_price_days': {},
            'outlier_returns': {},
            'good_quality_symbols': [],
            'poor_quality_symbols': []
        }
        
        for symbol in price_data.columns:
            series = price_data[symbol]
            
            # Missing data percentage
            missing_pct = series.isnull().sum() / len(series)
            quality_metrics['missing_data_pct'][symbol] = missing_pct
            
            # Zero price days
            zero_days = (series == 0).sum()
            quality_metrics['zero_price_days'][symbol] = zero_days
            
            # Outlier returns (>50% single day move)
            returns = series.pct_change().dropna()
            outliers = (abs(returns) > 0.5).sum()
            quality_metrics['outlier_returns'][symbol] = outliers
            
            # Overall quality assessment
            has_sufficient_history = len(series.dropna()) >= min_history_days
            has_low_missing = missing_pct < 0.1
            has_few_zeros = zero_days < 10
            has_few_outliers = outliers < 5
            
            if has_sufficient_history and has_low_missing and has_few_zeros and has_few_outliers:
                quality_metrics['good_quality_symbols'].append(symbol)
            else:
                quality_metrics['poor_quality_symbols'].append(symbol)
        
        logger.info(f"Data quality: {len(quality_metrics['good_quality_symbols'])} good, "
                   f"{len(quality_metrics['poor_quality_symbols'])} poor quality symbols")
        
        return quality_metrics
        
    def clean_cache(self, older_than_days: int = 7) -> None:
        """
        Clean old cache files.
        
        Args:
            older_than_days: Remove cache files older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_date.timestamp():
                cache_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old cache files")
        else:
            logger.info("No old cache files to remove")
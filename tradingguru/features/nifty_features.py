"""
NIFTY 50 features pipeline for TradingGuru.

This module provides feature engineering utilities for calculating returns,
volatility, momentum scores, sector-neutral momentum, and regime filters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)


class NiftyFeaturesPipeline:
    """
    Feature engineering pipeline for NIFTY 50 momentum strategy.
    
    This class calculates various technical and fundamental features used
    in momentum strategy signal generation.
    """
    
    def __init__(self, 
                 sector_mapping: Optional[Dict[str, str]] = None,
                 lookback_periods: Optional[Dict[str, int]] = None):
        """
        Initialize the features pipeline.
        
        Args:
            sector_mapping: Mapping of symbols to sectors
            lookback_periods: Dictionary of lookback periods for different features
        """
        self.sector_mapping = sector_mapping or {}
        
        # Default lookback periods
        default_periods = {
            'short_momentum': 21,      # 1 month
            'medium_momentum': 63,     # 3 months  
            'long_momentum': 252,      # 1 year
            'volatility': 21,          # 1 month volatility
            'volume': 21,              # Average volume
            'regime': 63,              # Regime detection
            'correlation': 21,         # Rolling correlation
            'rsi': 14,                 # RSI period
            'bollinger': 20            # Bollinger bands
        }
        
        self.lookback_periods = {**default_periods, **(lookback_periods or {})}
        
        # Scaler for standardization
        self.scaler = StandardScaler()
        
    def calculate_all_features(self, 
                              price_data: pd.DataFrame,
                              volume_data: Optional[pd.DataFrame] = None,
                              benchmark_data: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate all features for the momentum strategy.
        
        Args:
            price_data: Price data with symbols as columns
            volume_data: Volume data (optional)
            benchmark_data: Benchmark price series (optional)
            
        Returns:
            DataFrame with calculated features
        """
        logger.info("Calculating all features...")
        
        # Calculate returns first
        returns = price_data.pct_change().dropna()
        
        features_list = []
        
        # Basic return features
        logger.debug("Calculating return features...")
        return_features = self._calculate_return_features(returns)
        features_list.append(return_features)
        
        # Momentum features
        logger.debug("Calculating momentum features...")
        momentum_features = self._calculate_momentum_features(price_data, returns)
        features_list.append(momentum_features)
        
        # Volatility features
        logger.debug("Calculating volatility features...")
        volatility_features = self._calculate_volatility_features(returns)
        features_list.append(volatility_features)
        
        # Technical indicators
        logger.debug("Calculating technical indicators...")
        technical_features = self._calculate_technical_features(price_data, returns)
        features_list.append(technical_features)
        
        # Volume features (if available)
        if volume_data is not None:
            logger.debug("Calculating volume features...")
            volume_features = self._calculate_volume_features(volume_data, returns)
            features_list.append(volume_features)
        
        # Sector features (if mapping available)
        if self.sector_mapping:
            logger.debug("Calculating sector features...")
            sector_features = self._calculate_sector_features(returns, price_data)
            features_list.append(sector_features)
        
        # Market regime features (if benchmark available)
        if benchmark_data is not None:
            logger.debug("Calculating market regime features...")
            regime_features = self._calculate_regime_features(benchmark_data, returns)
            features_list.append(regime_features)
        
        # Cross-sectional features
        logger.debug("Calculating cross-sectional features...")
        cross_sectional_features = self._calculate_cross_sectional_features(returns)
        features_list.append(cross_sectional_features)
        
        # Combine all features
        features_df = pd.concat(features_list, axis=1)
        
        # Add price and return data for reference
        for symbol in price_data.columns:
            features_df[f'{symbol}_price'] = price_data[symbol]
            if len(returns) > 0 and symbol in returns.columns:
                features_df[f'{symbol}_return'] = returns[symbol]
        
        logger.info(f"Calculated {features_df.shape[1]} features for {features_df.shape[0]} periods")
        return features_df
        
    def _calculate_return_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic return-based features."""
        features = pd.DataFrame(index=returns.index)
        
        for symbol in returns.columns:
            ret_series = returns[symbol]
            
            # Multi-period returns
            features[f'{symbol}_ret_1d'] = ret_series
            features[f'{symbol}_ret_5d'] = ret_series.rolling(5).sum()
            features[f'{symbol}_ret_21d'] = ret_series.rolling(21).sum()
            features[f'{symbol}_ret_63d'] = ret_series.rolling(63).sum()
            
            # Return statistics
            features[f'{symbol}_ret_mean_21d'] = ret_series.rolling(21).mean()
            features[f'{symbol}_ret_std_21d'] = ret_series.rolling(21).std()
            features[f'{symbol}_ret_skew_21d'] = ret_series.rolling(21).skew()
            
        return features
        
    def _calculate_momentum_features(self, 
                                   price_data: pd.DataFrame, 
                                   returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features."""
        features = pd.DataFrame(index=price_data.index)
        
        for symbol in price_data.columns:
            prices = price_data[symbol]
            
            # Price momentum (price relative to moving averages)
            features[f'{symbol}_mom_short'] = prices / prices.rolling(self.lookback_periods['short_momentum']).mean() - 1
            features[f'{symbol}_mom_medium'] = prices / prices.rolling(self.lookback_periods['medium_momentum']).mean() - 1
            features[f'{symbol}_mom_long'] = prices / prices.rolling(self.lookback_periods['long_momentum']).mean() - 1
            
            # Trend strength
            ma_short = prices.rolling(21).mean()
            ma_long = prices.rolling(63).mean()
            features[f'{symbol}_trend'] = (ma_short / ma_long - 1).fillna(0)
            
            # Price acceleration (momentum of momentum)
            mom_21 = prices / prices.shift(21) - 1
            features[f'{symbol}_acceleration'] = mom_21 - mom_21.shift(21)
            
        return features
        
    def _calculate_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features."""
        features = pd.DataFrame(index=returns.index)
        
        for symbol in returns.columns:
            ret_series = returns[symbol]
            
            # Rolling volatility
            vol_window = self.lookback_periods['volatility']
            features[f'{symbol}_vol'] = ret_series.rolling(vol_window).std() * np.sqrt(252)
            
            # Volatility percentile (current vol vs historical)
            vol_252 = ret_series.rolling(252).std() * np.sqrt(252)
            features[f'{symbol}_vol_rank'] = vol_252.rolling(252).rank(pct=True)
            
            # Volatility regime
            vol_ma = vol_252.rolling(63).mean()
            features[f'{symbol}_vol_regime'] = (vol_252 / vol_ma - 1).fillna(0)
            
            # Downside volatility
            downside_returns = ret_series[ret_series < 0]
            features[f'{symbol}_downside_vol'] = downside_returns.rolling(vol_window).std() * np.sqrt(252)
            
        return features
        
    def _calculate_technical_features(self, 
                                    price_data: pd.DataFrame, 
                                    returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicator features."""
        features = pd.DataFrame(index=price_data.index)
        
        for symbol in price_data.columns:
            prices = price_data[symbol]
            ret_series = returns[symbol] if symbol in returns.columns else pd.Series(index=price_data.index)
            
            # RSI
            rsi_period = self.lookback_periods['rsi']
            gains = ret_series.where(ret_series > 0, 0)
            losses = -ret_series.where(ret_series < 0, 0)
            
            avg_gain = gains.rolling(rsi_period).mean()
            avg_loss = losses.rolling(rsi_period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            features[f'{symbol}_rsi'] = rsi
            
            # Bollinger Bands
            bb_period = self.lookback_periods['bollinger']
            bb_ma = prices.rolling(bb_period).mean()
            bb_std = prices.rolling(bb_period).std()
            features[f'{symbol}_bb_upper'] = bb_ma + (2 * bb_std)
            features[f'{symbol}_bb_lower'] = bb_ma - (2 * bb_std)
            features[f'{symbol}_bb_position'] = (prices - bb_ma) / (2 * bb_std)
            
            # Price channels
            high_252 = prices.rolling(252).max()
            low_252 = prices.rolling(252).min()
            features[f'{symbol}_channel_position'] = (prices - low_252) / (high_252 - low_252)
            
            # Moving average crossovers
            ma_10 = prices.rolling(10).mean()
            ma_30 = prices.rolling(30).mean()
            features[f'{symbol}_ma_cross'] = (ma_10 / ma_30 - 1).fillna(0)
            
        return features
        
    def _calculate_volume_features(self, 
                                 volume_data: pd.DataFrame, 
                                 returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        features = pd.DataFrame(index=volume_data.index)
        
        for symbol in volume_data.columns:
            if symbol not in returns.columns:
                continue
                
            volume = volume_data[symbol]
            ret_series = returns[symbol]
            
            # Volume moving averages
            vol_window = self.lookback_periods['volume']
            features[f'{symbol}_vol_ma'] = volume.rolling(vol_window).mean()
            features[f'{symbol}_vol_ratio'] = volume / features[f'{symbol}_vol_ma']
            
            # Volume-price trends
            features[f'{symbol}_vpt'] = (ret_series * volume).cumsum()
            
            # On-balance volume
            obv = (np.sign(ret_series) * volume).cumsum()
            features[f'{symbol}_obv'] = obv
            features[f'{symbol}_obv_ma'] = obv.rolling(21).mean()
            
            # Volume volatility
            features[f'{symbol}_vol_volatility'] = volume.rolling(21).std()
            
        return features
        
    def _calculate_sector_features(self, 
                                 returns: pd.DataFrame,
                                 price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sector-based features."""
        features = pd.DataFrame(index=returns.index)
        
        # Group symbols by sector
        sector_groups = {}
        for symbol, sector in self.sector_mapping.items():
            if symbol in returns.columns:
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(symbol)
        
        # Calculate sector indices
        sector_returns = {}
        for sector, symbols in sector_groups.items():
            if len(symbols) > 0:
                sector_ret = returns[symbols].mean(axis=1)
                sector_returns[sector] = sector_ret
                features[f'sector_{sector}_return'] = sector_ret
                
                # Sector momentum
                sector_prices = price_data[symbols].mean(axis=1)
                features[f'sector_{sector}_momentum'] = sector_prices / sector_prices.rolling(63).mean() - 1
        
        # Calculate sector-relative features for each stock
        for symbol in returns.columns:
            if symbol in self.sector_mapping:
                sector = self.sector_mapping[symbol]
                if sector in sector_returns:
                    # Sector-relative return
                    features[f'{symbol}_sector_relative'] = returns[symbol] - sector_returns[sector]
                    
                    # Sector beta
                    rolling_corr = returns[symbol].rolling(63).corr(sector_returns[sector])
                    rolling_std_ratio = returns[symbol].rolling(63).std() / sector_returns[sector].rolling(63).std()
                    features[f'{symbol}_sector_beta'] = rolling_corr * rolling_std_ratio
        
        return features
        
    def _calculate_regime_features(self, 
                                 benchmark_data: pd.Series, 
                                 returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features."""
        features = pd.DataFrame(index=returns.index)
        
        # Benchmark returns
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        # Market volatility regime
        market_vol = benchmark_returns.rolling(21).std() * np.sqrt(252)
        market_vol_ma = market_vol.rolling(63).mean()
        vol_regime = market_vol / market_vol_ma - 1
        features['market_vol_regime'] = vol_regime
        
        # Market trend
        benchmark_ma_short = benchmark_data.rolling(21).mean()
        benchmark_ma_long = benchmark_data.rolling(63).mean()
        market_trend = benchmark_ma_short / benchmark_ma_long - 1
        features['market_trend'] = market_trend
        
        # Market momentum
        market_momentum = benchmark_data / benchmark_data.rolling(63).mean() - 1
        features['market_momentum'] = market_momentum
        
        # VIX-like indicator (rolling volatility percentile)
        market_vol_rank = market_vol.rolling(252).rank(pct=True)
        features['market_fear'] = market_vol_rank
        
        # Calculate stock betas to market
        regime_window = self.lookback_periods['regime']
        for symbol in returns.columns:
            if len(benchmark_returns) > 0:
                # Align series
                aligned_data = pd.concat([returns[symbol], benchmark_returns], axis=1).dropna()
                if len(aligned_data) > regime_window:
                    stock_returns = aligned_data.iloc[:, 0]
                    market_returns = aligned_data.iloc[:, 1]
                    
                    # Rolling beta
                    correlation = stock_returns.rolling(regime_window).corr(market_returns)
                    vol_ratio = stock_returns.rolling(regime_window).std() / market_returns.rolling(regime_window).std()
                    beta = correlation * vol_ratio
                    
                    # Reindex to original index
                    beta = beta.reindex(returns.index, method='ffill')
                    features[f'{symbol}_beta'] = beta
        
        return features
        
    def _calculate_cross_sectional_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-sectional (relative) features."""
        features = pd.DataFrame(index=returns.index)
        
        # Cross-sectional momentum rankings
        mom_windows = [21, 63, 252]
        
        for window in mom_windows:
            # Calculate momentum for all stocks
            momentum_scores = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            for symbol in returns.columns:
                mom_scores = returns[symbol].rolling(window).sum()
                momentum_scores[symbol] = mom_scores
            
            # Rank across stocks each day
            momentum_ranks = momentum_scores.rank(axis=1, pct=True)
            
            for symbol in returns.columns:
                features[f'{symbol}_mom_rank_{window}d'] = momentum_ranks[symbol]
        
        # Cross-sectional volatility rankings
        vol_window = 21
        volatility_scores = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for symbol in returns.columns:
            vol_scores = returns[symbol].rolling(vol_window).std()
            volatility_scores[symbol] = vol_scores
            
        volatility_ranks = volatility_scores.rank(axis=1, pct=True)
        
        for symbol in returns.columns:
            features[f'{symbol}_vol_rank_{vol_window}d'] = volatility_ranks[symbol]
        
        # Cross-sectional size proxy (using price levels)
        # Higher prices typically indicate larger market cap for Indian stocks
        price_ranks = returns.index.to_series().apply(
            lambda date: pd.Series({
                symbol: 0.5 for symbol in returns.columns
            })
        )
        
        return features
        
    def calculate_momentum_score(self, 
                               features_df: pd.DataFrame,
                               scoring_method: str = 'composite') -> pd.DataFrame:
        """
        Calculate composite momentum scores for ranking.
        
        Args:
            features_df: DataFrame with calculated features
            scoring_method: Method for combining momentum signals
            
        Returns:
            DataFrame with momentum scores for each symbol
        """
        logger.info(f"Calculating momentum scores using {scoring_method} method")
        
        momentum_scores = pd.DataFrame(index=features_df.index)
        
        # Extract symbols from feature columns
        symbols = set()
        for col in features_df.columns:
            if '_mom_' in col or '_momentum' in col:
                symbol = col.split('_')[0]
                if symbol.endswith('.NS'):
                    symbols.add(symbol)
        
        for symbol in symbols:
            if scoring_method == 'composite':
                # Combine multiple momentum signals
                score_components = []
                
                # Short-term momentum
                if f'{symbol}_mom_short' in features_df.columns:
                    score_components.append(features_df[f'{symbol}_mom_short'] * 0.3)
                
                # Medium-term momentum
                if f'{symbol}_mom_medium' in features_df.columns:
                    score_components.append(features_df[f'{symbol}_mom_medium'] * 0.4)
                
                # Long-term momentum
                if f'{symbol}_mom_long' in features_df.columns:
                    score_components.append(features_df[f'{symbol}_mom_long'] * 0.3)
                
                # Cross-sectional rankings
                if f'{symbol}_mom_rank_63d' in features_df.columns:
                    rank_score = (features_df[f'{symbol}_mom_rank_63d'] - 0.5) * 2  # Scale to -1,1
                    score_components.append(rank_score * 0.2)
                
                if score_components:
                    momentum_scores[f'{symbol}_momentum_score'] = sum(score_components)
                    
            elif scoring_method == 'simple':
                # Simple momentum score
                if f'{symbol}_mom_medium' in features_df.columns:
                    momentum_scores[f'{symbol}_momentum_score'] = features_df[f'{symbol}_mom_medium']
                    
        return momentum_scores
        
    def calculate_sector_neutral_scores(self, 
                                      momentum_scores: pd.DataFrame,
                                      features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector-neutral momentum scores.
        
        Args:
            momentum_scores: Raw momentum scores
            features_df: Features DataFrame for reference
            
        Returns:
            DataFrame with sector-neutral momentum scores
        """
        if not self.sector_mapping:
            logger.warning("No sector mapping available, returning original scores")
            return momentum_scores
            
        logger.info("Calculating sector-neutral momentum scores")
        
        sector_neutral_scores = pd.DataFrame(index=momentum_scores.index)
        
        # Group symbols by sector
        sector_groups = {}
        for symbol, sector in self.sector_mapping.items():
            score_col = f'{symbol}_momentum_score'
            if score_col in momentum_scores.columns:
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(symbol)
        
        # Calculate sector-neutral scores
        for date in momentum_scores.index:
            for sector, symbols in sector_groups.items():
                if len(symbols) < 2:
                    continue
                    
                # Get scores for this sector on this date
                sector_scores = {}
                for symbol in symbols:
                    score_col = f'{symbol}_momentum_score'
                    if score_col in momentum_scores.columns:
                        score = momentum_scores.loc[date, score_col]
                        if pd.notna(score):
                            sector_scores[symbol] = score
                
                if len(sector_scores) < 2:
                    continue
                    
                # Calculate sector mean
                sector_mean = np.mean(list(sector_scores.values()))
                
                # Subtract sector mean from each stock
                for symbol, score in sector_scores.items():
                    neutral_col = f'{symbol}_momentum_score_neutral'
                    sector_neutral_scores.loc[date, neutral_col] = score - sector_mean
        
        return sector_neutral_scores
        
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for calculated features.
        
        Args:
            features_df: DataFrame with calculated features
            
        Returns:
            Dictionary with feature summary statistics
        """
        summary = {
            'total_features': features_df.shape[1],
            'total_periods': features_df.shape[0],
            'date_range': (features_df.index.min(), features_df.index.max()),
            'feature_types': {},
            'missing_data': {},
            'feature_correlations': {}
        }
        
        # Categorize features by type
        feature_types = {
            'momentum': [],
            'volatility': [],
            'technical': [],
            'volume': [],
            'sector': [],
            'regime': [],
            'cross_sectional': [],
            'other': []
        }
        
        for col in features_df.columns:
            if 'mom' in col or 'momentum' in col:
                feature_types['momentum'].append(col)
            elif 'vol' in col and 'volatility' not in col:
                feature_types['volatility'].append(col)
            elif any(x in col for x in ['rsi', 'bb_', 'ma_', 'channel']):
                feature_types['technical'].append(col)
            elif 'vol_' in col or 'obv' in col or 'vpt' in col:
                feature_types['volume'].append(col)
            elif 'sector' in col:
                feature_types['sector'].append(col)
            elif 'market' in col or 'regime' in col:
                feature_types['regime'].append(col)
            elif 'rank' in col:
                feature_types['cross_sectional'].append(col)
            else:
                feature_types['other'].append(col)
        
        summary['feature_types'] = {k: len(v) for k, v in feature_types.items()}
        
        # Missing data analysis
        missing_pct = features_df.isnull().sum() / len(features_df)
        summary['missing_data'] = {
            'features_with_missing': (missing_pct > 0).sum(),
            'avg_missing_pct': missing_pct.mean(),
            'max_missing_pct': missing_pct.max()
        }
        
        logger.info(f"Feature summary: {summary['total_features']} features across {summary['total_periods']} periods")
        return summary
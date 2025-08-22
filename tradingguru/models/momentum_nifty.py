"""
Enhanced NIFTY 50 momentum strategy implementation.

This module implements a sophisticated momentum strategy with sector-neutrality,
inverse-volatility weighting, and risk-off scaling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import logging

from ..core.base import BaseStrategy
from ..features.nifty_features import NiftyFeaturesPipeline
from ..core.utils import winsorize_series, standardize_series

logger = logging.getLogger(__name__)


class EnhancedNiftyMomentumStrategy(BaseStrategy):
    """
    Enhanced momentum strategy for NIFTY 50 with advanced features:
    - Sector-neutral momentum signals
    - Inverse volatility weighting
    - Risk-off regime scaling
    - Multi-timeframe momentum
    - Dynamic position sizing
    """
    
    def __init__(self, 
                 universe: List[str],
                 sector_mapping: Optional[Dict[str, str]] = None,
                 **params):
        """
        Initialize the enhanced momentum strategy.
        
        Args:
            universe: List of stock symbols to trade
            sector_mapping: Mapping of symbols to sectors
            **params: Strategy parameters
        """
        # Default parameters
        default_params = {
            # Signal generation
            'lookback_short': 21,           # Short-term momentum (days)
            'lookback_medium': 63,          # Medium-term momentum (days) 
            'lookback_long': 252,           # Long-term momentum (days)
            'momentum_weights': [0.2, 0.5, 0.3],  # Weights for short/medium/long momentum
            
            # Position sizing
            'top_quantile': 0.3,            # Top quantile of stocks to hold
            'min_positions': 10,            # Minimum number of positions
            'max_positions': 20,            # Maximum number of positions
            'equal_weight': False,          # Whether to use equal weighting
            
            # Risk management
            'volatility_lookback': 21,      # Volatility calculation window
            'vol_target': 0.15,             # Target portfolio volatility
            'max_weight': 0.08,             # Maximum individual position weight
            'sector_neutral': True,         # Whether to apply sector neutrality
            'max_sector_weight': 0.25,      # Maximum sector exposure
            
            # Regime filters
            'market_filter': True,          # Whether to apply market regime filter
            'vol_regime_threshold': 1.5,    # High volatility regime threshold
            'trend_filter': True,           # Whether to apply trend filter
            'min_trend_strength': 0.02,     # Minimum trend strength
            
            # Risk-off scaling
            'risk_off_enabled': True,       # Enable risk-off scaling
            'risk_off_vol_threshold': 0.25, # VIX-like threshold for risk-off
            'risk_off_scale': 0.5,          # Scale factor during risk-off periods
            
            # Signal processing
            'signal_smoothing': True,       # Whether to smooth signals
            'smoothing_window': 3,          # Signal smoothing window
            'winsorize_signals': True,      # Whether to winsorize signals
            'winsorize_percentiles': (0.05, 0.95),  # Winsorization percentiles
        }
        
        # Merge with provided parameters
        strategy_params = {**default_params, **params}
        
        super().__init__(
            name="Enhanced_NIFTY_Momentum",
            universe=universe,
            params=strategy_params
        )
        
        self.sector_mapping = sector_mapping or {}
        
        # Initialize features pipeline
        self.features_pipeline = NiftyFeaturesPipeline(
            sector_mapping=sector_mapping,
            lookback_periods={
                'short_momentum': self.params['lookback_short'],
                'medium_momentum': self.params['lookback_medium'],
                'long_momentum': self.params['lookback_long'],
                'volatility': self.params['volatility_lookback']
            }
        )
        
        # Cache for features
        self._features_cache = None
        self._last_calculation_date = None
        
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for the momentum strategy.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            DataFrame with calculated features
        """
        logger.info("Calculating momentum strategy features...")
        
        # Use cached features if available and up-to-date
        if (self._features_cache is not None and 
            self._last_calculation_date is not None and
            data.index[-1] <= self._last_calculation_date):
            
            return self._features_cache.loc[:data.index[-1]]
        
        # Calculate all features using the pipeline
        features_df = self.features_pipeline.calculate_all_features(
            price_data=data,
            volume_data=None,  # Volume data would be added here if available
            benchmark_data=None  # Benchmark data would be added here if available
        )
        
        # Calculate momentum scores
        momentum_scores = self.features_pipeline.calculate_momentum_score(
            features_df, scoring_method='composite'
        )
        
        # Add momentum scores to features
        for col in momentum_scores.columns:
            features_df[col] = momentum_scores[col]
        
        # Calculate sector-neutral scores if enabled
        if self.params['sector_neutral'] and self.sector_mapping:
            sector_neutral_scores = self.features_pipeline.calculate_sector_neutral_scores(
                momentum_scores, features_df
            )
            
            for col in sector_neutral_scores.columns:
                features_df[col] = sector_neutral_scores[col]
        
        # Cache the features
        self._features_cache = features_df
        self._last_calculation_date = data.index[-1]
        
        logger.info(f"Features calculated: {features_df.shape[1]} features, {features_df.shape[0]} periods")
        return features_df
        
    def generate_signals(self, 
                        data: pd.DataFrame, 
                        current_date: date) -> Dict[str, float]:
        """
        Generate trading signals for the given date.
        
        Args:
            data: Features DataFrame with calculated indicators
            current_date: Current trading date
            
        Returns:
            Dictionary mapping symbols to target weights
        """
        if current_date not in data.index:
            logger.warning(f"No data available for {current_date}")
            return {}
        
        # Get current date data
        current_data = data.loc[current_date]
        
        # Step 1: Calculate raw momentum signals
        momentum_signals = self._calculate_momentum_signals(current_data)
        
        if not momentum_signals:
            logger.warning(f"No momentum signals generated for {current_date}")
            return {}
        
        # Step 2: Apply market regime filters
        if self.params['market_filter']:
            momentum_signals = self._apply_market_filter(momentum_signals, current_data)
        
        # Step 3: Apply trend filters
        if self.params['trend_filter']:
            momentum_signals = self._apply_trend_filter(momentum_signals, current_data)
        
        # Step 4: Select top momentum stocks
        selected_stocks = self._select_top_momentum_stocks(momentum_signals)
        
        if not selected_stocks:
            logger.warning(f"No stocks selected after filtering on {current_date}")
            return {}
        
        # Step 5: Calculate position weights
        position_weights = self._calculate_position_weights(selected_stocks, current_data)
        
        # Step 6: Apply risk management constraints
        position_weights = self._apply_risk_constraints(position_weights, current_data)
        
        # Step 7: Apply risk-off scaling if needed
        if self.params['risk_off_enabled']:
            position_weights = self._apply_risk_off_scaling(position_weights, current_data)
        
        # Step 8: Final signal processing
        position_weights = self._process_final_signals(position_weights)
        
        logger.debug(f"Generated signals for {len(position_weights)} stocks on {current_date}")
        return position_weights
        
    def _calculate_momentum_signals(self, current_data: pd.Series) -> Dict[str, float]:
        """Calculate raw momentum signals for all stocks."""
        momentum_signals = {}
        
        for symbol in self.universe:
            try:
                # Check if we have momentum score for this symbol
                score_col = f'{symbol}_momentum_score'
                neutral_col = f'{symbol}_momentum_score_neutral'
                
                # Use sector-neutral score if available and enabled
                if self.params['sector_neutral'] and neutral_col in current_data:
                    momentum_score = current_data[neutral_col]
                elif score_col in current_data:
                    momentum_score = current_data[score_col]
                else:
                    # Fallback: calculate simple momentum
                    momentum_score = self._calculate_simple_momentum(symbol, current_data)
                
                if pd.notna(momentum_score):
                    momentum_signals[symbol] = float(momentum_score)
                    
            except Exception as e:
                logger.debug(f"Error calculating momentum for {symbol}: {e}")
                continue
        
        return momentum_signals
        
    def _calculate_simple_momentum(self, symbol: str, current_data: pd.Series) -> float:
        """Calculate simple momentum as fallback."""
        try:
            # Try to get price-based momentum
            mom_col = f'{symbol}_mom_medium'
            if mom_col in current_data:
                return current_data[mom_col]
            
            # Try rank-based momentum
            rank_col = f'{symbol}_mom_rank_63d'
            if rank_col in current_data:
                rank = current_data[rank_col]
                return (rank - 0.5) * 2  # Convert to -1,1 scale
            
            return 0.0
            
        except:
            return 0.0
        
    def _apply_market_filter(self, 
                           momentum_signals: Dict[str, float], 
                           current_data: pd.Series) -> Dict[str, float]:
        """Apply market regime filters to momentum signals."""
        # Check market volatility regime
        if 'market_vol_regime' in current_data:
            vol_regime = current_data['market_vol_regime']
            
            if pd.notna(vol_regime) and vol_regime > self.params['vol_regime_threshold']:
                # High volatility regime - reduce signal strength
                scale_factor = 0.5
                logger.debug(f"High volatility regime detected (vol_regime={vol_regime:.3f}), scaling signals by {scale_factor}")
                
                return {symbol: signal * scale_factor for symbol, signal in momentum_signals.items()}
        
        # Check market trend
        if 'market_trend' in current_data:
            market_trend = current_data['market_trend']
            
            if pd.notna(market_trend) and market_trend < -self.params['min_trend_strength']:
                # Negative market trend - reduce long signals
                logger.debug(f"Negative market trend detected (trend={market_trend:.3f}), reducing long signals")
                
                filtered_signals = {}
                for symbol, signal in momentum_signals.items():
                    if signal > 0:
                        filtered_signals[symbol] = signal * 0.3  # Reduce long signals
                    else:
                        filtered_signals[symbol] = signal
                        
                return filtered_signals
        
        return momentum_signals
        
    def _apply_trend_filter(self, 
                          momentum_signals: Dict[str, float], 
                          current_data: pd.Series) -> Dict[str, float]:
        """Apply individual stock trend filters."""
        filtered_signals = {}
        
        for symbol, signal in momentum_signals.items():
            try:
                # Check individual stock trend
                trend_col = f'{symbol}_trend'
                if trend_col in current_data:
                    trend = current_data[trend_col]
                    
                    if pd.notna(trend):
                        # Only trade in direction of trend
                        if signal > 0 and trend < -self.params['min_trend_strength']:
                            # Positive momentum but negative trend - reduce signal
                            filtered_signals[symbol] = signal * 0.2
                        elif signal < 0 and trend > self.params['min_trend_strength']:
                            # Negative momentum but positive trend - reduce signal
                            filtered_signals[symbol] = signal * 0.2
                        else:
                            filtered_signals[symbol] = signal
                    else:
                        filtered_signals[symbol] = signal
                else:
                    filtered_signals[symbol] = signal
                    
            except Exception as e:
                logger.debug(f"Error applying trend filter for {symbol}: {e}")
                filtered_signals[symbol] = signal
        
        return filtered_signals
        
    def _select_top_momentum_stocks(self, momentum_signals: Dict[str, float]) -> List[str]:
        """Select top momentum stocks based on quantile."""
        if not momentum_signals:
            return []
        
        # Sort by momentum score
        sorted_signals = sorted(momentum_signals.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate number of stocks to select
        total_stocks = len(sorted_signals)
        target_count = max(
            self.params['min_positions'],
            min(
                self.params['max_positions'],
                int(total_stocks * self.params['top_quantile'])
            )
        )
        
        # Select top stocks with positive momentum
        selected = []
        for symbol, signal in sorted_signals:
            if len(selected) >= target_count:
                break
            if signal > 0:  # Only select stocks with positive momentum
                selected.append(symbol)
        
        # If we don't have enough positive momentum stocks, relax the constraint
        if len(selected) < self.params['min_positions']:
            for symbol, signal in sorted_signals:
                if len(selected) >= self.params['min_positions']:
                    break
                if symbol not in selected:
                    selected.append(symbol)
        
        logger.debug(f"Selected {len(selected)} stocks from {total_stocks} available")
        return selected
        
    def _calculate_position_weights(self, 
                                  selected_stocks: List[str], 
                                  current_data: pd.Series) -> Dict[str, float]:
        """Calculate position weights for selected stocks."""
        if not selected_stocks:
            return {}
        
        if self.params['equal_weight']:
            # Equal weighting
            weight = 1.0 / len(selected_stocks)
            return {symbol: weight for symbol in selected_stocks}
        
        # Inverse volatility weighting
        volatilities = {}
        for symbol in selected_stocks:
            vol_col = f'{symbol}_vol'
            if vol_col in current_data:
                vol = current_data[vol_col]
                if pd.notna(vol) and vol > 0:
                    volatilities[symbol] = vol
                else:
                    volatilities[symbol] = 0.15  # Default volatility
            else:
                volatilities[symbol] = 0.15  # Default volatility
        
        # Calculate inverse volatility weights
        inv_vol_weights = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol_weights.values())
        
        # Normalize to sum to 1
        weights = {symbol: weight / total_inv_vol for symbol, weight in inv_vol_weights.items()}
        
        return weights
        
    def _apply_risk_constraints(self, 
                              position_weights: Dict[str, float], 
                              current_data: pd.Series) -> Dict[str, float]:
        """Apply risk management constraints to position weights."""
        if not position_weights:
            return {}
        
        adjusted_weights = position_weights.copy()
        
        # Apply maximum individual weight constraint
        max_weight = self.params['max_weight']
        for symbol in adjusted_weights:
            if adjusted_weights[symbol] > max_weight:
                adjusted_weights[symbol] = max_weight
        
        # Apply sector weight constraints if sector mapping is available
        if self.sector_mapping and self.params['max_sector_weight']:
            adjusted_weights = self._apply_sector_constraints(adjusted_weights)
        
        # Renormalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            scale_factor = 1.0 / total_weight
            adjusted_weights = {symbol: weight * scale_factor for symbol, weight in adjusted_weights.items()}
        
        return adjusted_weights
        
    def _apply_sector_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply sector exposure constraints."""
        # Calculate current sector exposures
        sector_exposures = {}
        for symbol, weight in weights.items():
            sector = self.sector_mapping.get(symbol, 'OTHER')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + weight
        
        # Check for sector limit breaches
        max_sector_weight = self.params['max_sector_weight']
        breached_sectors = {
            sector: exposure for sector, exposure in sector_exposures.items()
            if exposure > max_sector_weight
        }
        
        if not breached_sectors:
            return weights
        
        # Scale down weights in breached sectors
        adjusted_weights = weights.copy()
        for sector, current_exposure in breached_sectors.items():
            scale_factor = max_sector_weight / current_exposure
            
            # Find symbols in this sector and scale their weights
            for symbol in weights:
                if self.sector_mapping.get(symbol, 'OTHER') == sector:
                    adjusted_weights[symbol] *= scale_factor
        
        return adjusted_weights
        
    def _apply_risk_off_scaling(self, 
                              position_weights: Dict[str, float], 
                              current_data: pd.Series) -> Dict[str, float]:
        """Apply risk-off scaling based on market conditions."""
        # Check market fear indicator
        if 'market_fear' in current_data:
            market_fear = current_data['market_fear']
            
            if pd.notna(market_fear) and market_fear > self.params['risk_off_vol_threshold']:
                # Risk-off regime detected
                scale_factor = self.params['risk_off_scale']
                logger.debug(f"Risk-off regime detected (fear={market_fear:.3f}), scaling positions by {scale_factor}")
                
                return {symbol: weight * scale_factor for symbol, weight in position_weights.items()}
        
        # Check market volatility regime
        if 'market_vol_regime' in current_data:
            vol_regime = current_data['market_vol_regime']
            
            if pd.notna(vol_regime) and vol_regime > 2.0:  # Very high volatility
                scale_factor = 0.3
                logger.debug(f"Extreme volatility detected (vol_regime={vol_regime:.3f}), scaling positions by {scale_factor}")
                
                return {symbol: weight * scale_factor for symbol, weight in position_weights.items()}
        
        return position_weights
        
    def _process_final_signals(self, position_weights: Dict[str, float]) -> Dict[str, float]:
        """Final signal processing and smoothing."""
        if not position_weights:
            return {}
        
        # Convert to pandas Series for processing
        weights_series = pd.Series(position_weights)
        
        # Winsorize signals if enabled
        if self.params['winsorize_signals']:
            lower_pct, upper_pct = self.params['winsorize_percentiles']
            weights_series = winsorize_series(weights_series, lower_pct, upper_pct)
        
        # Signal smoothing would be applied here if we had historical signals
        # For now, we'll skip this as it requires maintaining signal history
        
        # Ensure weights sum to 1 or less
        total_weight = weights_series.sum()
        if total_weight > 1.0:
            weights_series = weights_series / total_weight
        
        # Convert back to dictionary
        final_weights = weights_series.to_dict()
        
        # Remove very small weights
        final_weights = {
            symbol: weight for symbol, weight in final_weights.items()
            if abs(weight) > 1e-6
        }
        
        return final_weights
        
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy configuration and current state."""
        summary = {
            'strategy_name': self.name,
            'universe_size': len(self.universe),
            'sector_mapping_available': len(self.sector_mapping) > 0,
            'parameters': self.params.copy(),
            'features_calculated': self._features_cache is not None,
            'last_calculation_date': self._last_calculation_date
        }
        
        if self._features_cache is not None:
            summary['features_shape'] = self._features_cache.shape
            summary['feature_summary'] = self.features_pipeline.get_feature_summary(self._features_cache)
        
        return summary
        
    def validate_configuration(self) -> List[str]:
        """Validate strategy configuration and return any issues."""
        issues = []
        
        # Check universe
        if len(self.universe) < self.params['min_positions']:
            issues.append(f"Universe size ({len(self.universe)}) is less than min_positions ({self.params['min_positions']})")
        
        # Check parameter ranges
        if not 0 < self.params['top_quantile'] <= 1:
            issues.append(f"top_quantile ({self.params['top_quantile']}) must be between 0 and 1")
        
        if self.params['min_positions'] > self.params['max_positions']:
            issues.append(f"min_positions ({self.params['min_positions']}) cannot be greater than max_positions ({self.params['max_positions']})")
        
        if not 0 < self.params['max_weight'] <= 1:
            issues.append(f"max_weight ({self.params['max_weight']}) must be between 0 and 1")
        
        # Check sector mapping if sector neutrality is enabled
        if self.params['sector_neutral'] and not self.sector_mapping:
            issues.append("sector_neutral is enabled but no sector_mapping provided")
        
        return issues
        
    def __repr__(self) -> str:
        config_summary = f"top_quantile={self.params['top_quantile']}, sector_neutral={self.params['sector_neutral']}"
        return f"EnhancedNiftyMomentumStrategy({len(self.universe)} stocks, {config_summary})"
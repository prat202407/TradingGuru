"""
Risk management utilities for TradingGuru.

This module provides risk management tools including position sizing,
portfolio constraints, and risk monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskConstraints:
    """Container for risk management constraints."""
    max_position_weight: float = 0.05  # Maximum weight per position
    max_sector_weight: float = 0.30    # Maximum weight per sector
    max_leverage: float = 1.0          # Maximum leverage
    min_diversification: int = 10      # Minimum number of positions
    max_turnover: float = 2.0          # Maximum annual turnover
    max_drawdown_limit: float = -0.20  # Maximum drawdown before stopping
    volatility_target: Optional[float] = None  # Target portfolio volatility
    var_limit: Optional[float] = None  # Value at Risk limit


class RiskManager:
    """
    Risk manager for portfolio position sizing and constraint enforcement.
    
    This class manages position sizing, applies risk constraints, and monitors
    portfolio risk metrics during strategy execution.
    """
    
    def __init__(self, 
                 constraints: Optional[RiskConstraints] = None,
                 sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the risk manager.
        
        Args:
            constraints: Risk constraints configuration
            sector_mapping: Mapping of symbols to sectors
        """
        self.constraints = constraints or RiskConstraints()
        self.sector_mapping = sector_mapping or {}
        
        # Risk monitoring
        self.portfolio_history = []
        self.risk_breaches = []
        self.is_risk_off = False
        
    def apply_position_constraints(self, 
                                  target_weights: Dict[str, float],
                                  current_weights: Optional[Dict[str, float]] = None,
                                  prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Apply risk constraints to target position weights.
        
        Args:
            target_weights: Desired position weights
            current_weights: Current position weights
            prices: Current prices for turnover calculation
            
        Returns:
            Adjusted position weights
        """
        if not target_weights:
            return {}
            
        adjusted_weights = target_weights.copy()
        
        # Apply individual position weight limits
        adjusted_weights = self._apply_position_limits(adjusted_weights)
        
        # Apply sector weight limits
        if self.sector_mapping:
            adjusted_weights = self._apply_sector_limits(adjusted_weights)
            
        # Apply leverage limits
        adjusted_weights = self._apply_leverage_limits(adjusted_weights)
        
        # Apply diversification constraints
        adjusted_weights = self._apply_diversification_limits(adjusted_weights)
        
        # Apply turnover limits if current weights provided
        if current_weights is not None:
            adjusted_weights = self._apply_turnover_limits(
                adjusted_weights, current_weights, prices
            )
            
        # Log any significant adjustments
        self._log_adjustments(target_weights, adjusted_weights)
        
        return adjusted_weights
        
    def _apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply individual position weight limits."""
        max_weight = self.constraints.max_position_weight
        
        for symbol in weights:
            if abs(weights[symbol]) > max_weight:
                sign = np.sign(weights[symbol])
                weights[symbol] = sign * max_weight
                
        return weights
        
    def _apply_sector_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply sector concentration limits."""
        if not self.sector_mapping:
            return weights
            
        # Calculate sector exposures
        sector_weights = {}
        for symbol, weight in weights.items():
            sector = self.sector_mapping.get(symbol, 'OTHER')
            sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)
            
        # Check for sector limit breaches
        max_sector_weight = self.constraints.max_sector_weight
        breached_sectors = {
            sector: weight for sector, weight in sector_weights.items() 
            if weight > max_sector_weight
        }
        
        if not breached_sectors:
            return weights
            
        # Scale down weights in breached sectors
        adjusted_weights = weights.copy()
        for sector, current_weight in breached_sectors.items():
            scale_factor = max_sector_weight / current_weight
            
            # Find all symbols in this sector
            sector_symbols = [
                symbol for symbol, sec in self.sector_mapping.items() 
                if sec == sector and symbol in weights
            ]
            
            # Scale their weights
            for symbol in sector_symbols:
                adjusted_weights[symbol] *= scale_factor
                
        return adjusted_weights
        
    def _apply_leverage_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply leverage limits."""
        total_abs_weight = sum(abs(w) for w in weights.values())
        max_leverage = self.constraints.max_leverage
        
        if total_abs_weight <= max_leverage:
            return weights
            
        # Scale down all weights proportionally
        scale_factor = max_leverage / total_abs_weight
        return {symbol: weight * scale_factor for symbol, weight in weights.items()}
        
    def _apply_diversification_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum diversification constraints."""
        min_positions = self.constraints.min_diversification
        non_zero_positions = sum(1 for w in weights.values() if abs(w) > 1e-6)
        
        if non_zero_positions >= min_positions:
            return weights
            
        # If insufficient diversification, we might need to add positions
        # For now, just log a warning
        logger.warning(
            f"Portfolio has only {non_zero_positions} positions, "
            f"minimum required: {min_positions}"
        )
        
        return weights
        
    def _apply_turnover_limits(self, 
                              target_weights: Dict[str, float],
                              current_weights: Dict[str, float],
                              prices: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Apply turnover limits."""
        if not prices:
            return target_weights
            
        # Calculate turnover
        turnover = self._calculate_turnover(target_weights, current_weights, prices)
        max_turnover = self.constraints.max_turnover / 252  # Daily limit
        
        if turnover <= max_turnover:
            return target_weights
            
        # Scale down changes to meet turnover limit
        scale_factor = max_turnover / turnover if turnover > 0 else 1.0
        
        adjusted_weights = {}
        for symbol in set(target_weights.keys()) | set(current_weights.keys()):
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            change = target_w - current_w
            
            adjusted_weights[symbol] = current_w + (change * scale_factor)
            
        return adjusted_weights
        
    def _calculate_turnover(self, 
                           target_weights: Dict[str, float],
                           current_weights: Dict[str, float],
                           prices: Dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        total_change = 0.0
        
        for symbol in set(target_weights.keys()) | set(current_weights.keys()):
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            total_change += abs(target_w - current_w)
            
        return total_change
        
    def _log_adjustments(self, 
                        original: Dict[str, float], 
                        adjusted: Dict[str, float]) -> None:
        """Log significant weight adjustments."""
        significant_changes = []
        
        for symbol in set(original.keys()) | set(adjusted.keys()):
            orig_w = original.get(symbol, 0)
            adj_w = adjusted.get(symbol, 0)
            
            if abs(orig_w - adj_w) > 0.01:  # 1% threshold
                significant_changes.append(
                    f"{symbol}: {orig_w:.3f} -> {adj_w:.3f}"
                )
                
        if significant_changes:
            logger.info(f"Risk adjustments applied: {'; '.join(significant_changes)}")
            
    def calculate_portfolio_risk(self, 
                               weights: Dict[str, float],
                               returns_data: pd.DataFrame,
                               lookback_days: int = 60) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            weights: Current portfolio weights
            returns_data: Historical returns data
            lookback_days: Days of history to use for risk calculation
            
        Returns:
            Dictionary of risk metrics
        """
        if not weights or returns_data.empty:
            return {}
            
        # Get recent returns
        recent_returns = returns_data.tail(lookback_days)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(weights, recent_returns)
        
        if len(portfolio_returns) == 0:
            return {}
            
        # Calculate risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        var_95 = portfolio_returns.quantile(0.05)
        var_99 = portfolio_returns.quantile(0.01)
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # Beta to equal-weight portfolio (if available)
        equal_weight_returns = recent_returns.mean(axis=1)
        beta = self._calculate_beta(portfolio_returns, equal_weight_returns)
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'tracking_error': (portfolio_returns - equal_weight_returns).std() * np.sqrt(252)
        }
        
    def _calculate_portfolio_returns(self, 
                                   weights: Dict[str, float],
                                   returns_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from weights and returns data."""
        # Align weights with available data
        available_symbols = [s for s in weights.keys() if s in returns_data.columns]
        
        if not available_symbols:
            return pd.Series(dtype=float)
            
        aligned_weights = pd.Series(
            [weights[s] for s in available_symbols], 
            index=available_symbols
        )
        aligned_returns = returns_data[available_symbols]
        
        # Calculate weighted returns
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        return portfolio_returns.dropna()
        
    def _calculate_beta(self, 
                       portfolio_returns: pd.Series,
                       market_returns: pd.Series) -> float:
        """Calculate portfolio beta."""
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 0.0
            
        # Align series
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 2:
            return 0.0
            
        port_aligned = aligned.iloc[:, 0]
        market_aligned = aligned.iloc[:, 1]
        
        covariance = np.cov(port_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        
        return covariance / market_variance if market_variance > 0 else 0.0
        
    def check_risk_limits(self, 
                         current_drawdown: float,
                         portfolio_metrics: Dict[str, float]) -> bool:
        """
        Check if risk limits are breached.
        
        Args:
            current_drawdown: Current portfolio drawdown
            portfolio_metrics: Current portfolio risk metrics
            
        Returns:
            True if trading should continue, False if risk limits breached
        """
        # Check drawdown limit
        if current_drawdown < self.constraints.max_drawdown_limit:
            self.risk_breaches.append({
                'date': pd.Timestamp.now(),
                'type': 'max_drawdown',
                'value': current_drawdown,
                'limit': self.constraints.max_drawdown_limit
            })
            self.is_risk_off = True
            logger.warning(
                f"Maximum drawdown limit breached: {current_drawdown:.2%} "
                f"(limit: {self.constraints.max_drawdown_limit:.2%})"
            )
            return False
            
        # Check VaR limit if specified
        if (self.constraints.var_limit is not None and 
            'var_95' in portfolio_metrics):
            
            if portfolio_metrics['var_95'] < self.constraints.var_limit:
                self.risk_breaches.append({
                    'date': pd.Timestamp.now(),
                    'type': 'var_limit',
                    'value': portfolio_metrics['var_95'],
                    'limit': self.constraints.var_limit
                })
                logger.warning(
                    f"VaR limit breached: {portfolio_metrics['var_95']:.2%} "
                    f"(limit: {self.constraints.var_limit:.2%})"
                )
                return False
                
        return True
        
    def scale_for_volatility_target(self, 
                                   weights: Dict[str, float],
                                   current_volatility: float) -> Dict[str, float]:
        """
        Scale portfolio weights to achieve volatility target.
        
        Args:
            weights: Current portfolio weights
            current_volatility: Current portfolio volatility
            
        Returns:
            Volatility-adjusted weights
        """
        if (self.constraints.volatility_target is None or 
            current_volatility <= 0):
            return weights
            
        target_vol = self.constraints.volatility_target
        scale_factor = target_vol / current_volatility
        
        # Cap scaling to reasonable bounds
        scale_factor = np.clip(scale_factor, 0.1, 3.0)
        
        scaled_weights = {
            symbol: weight * scale_factor 
            for symbol, weight in weights.items()
        }
        
        # Re-apply leverage constraints after scaling
        return self._apply_leverage_limits(scaled_weights)
        
    def get_risk_report(self) -> str:
        """Generate a risk management report."""
        report = "\n" + "="*50 + "\n"
        report += "RISK MANAGEMENT REPORT\n"
        report += "="*50 + "\n\n"
        
        report += f"Risk Status: {'RISK OFF' if self.is_risk_off else 'NORMAL'}\n"
        report += f"Number of Risk Breaches: {len(self.risk_breaches)}\n\n"
        
        report += "Current Constraints:\n"
        report += f"  Max Position Weight: {self.constraints.max_position_weight:.1%}\n"
        report += f"  Max Sector Weight: {self.constraints.max_sector_weight:.1%}\n"
        report += f"  Max Leverage: {self.constraints.max_leverage:.2f}\n"
        report += f"  Min Diversification: {self.constraints.min_diversification}\n"
        report += f"  Max Turnover: {self.constraints.max_turnover:.2f}\n"
        report += f"  Max Drawdown Limit: {self.constraints.max_drawdown_limit:.1%}\n"
        
        if self.constraints.volatility_target:
            report += f"  Volatility Target: {self.constraints.volatility_target:.1%}\n"
            
        if self.constraints.var_limit:
            report += f"  VaR Limit: {self.constraints.var_limit:.1%}\n"
            
        if self.risk_breaches:
            report += "\nRecent Risk Breaches:\n"
            for breach in self.risk_breaches[-5:]:  # Last 5 breaches
                report += f"  {breach['date'].strftime('%Y-%m-%d')}: "
                report += f"{breach['type']} = {breach['value']:.2%} "
                report += f"(limit: {breach['limit']:.2%})\n"
                
        report += "\n" + "="*50 + "\n"
        return report
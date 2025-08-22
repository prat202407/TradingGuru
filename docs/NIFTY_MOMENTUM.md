# NIFTY 50 Momentum Strategy

## Overview

The Enhanced NIFTY 50 Momentum Strategy is a sophisticated quantitative trading strategy designed for the Indian equity market. It implements a multi-factor momentum approach with advanced risk management, sector neutrality, and regime-aware position sizing.

## Strategy Description

### Core Philosophy

The strategy is built on the premise that stocks with strong recent performance tend to continue outperforming in the short to medium term. However, unlike simple momentum strategies, this implementation includes several enhancements:

1. **Multi-timeframe momentum** - Combines short (1 month), medium (3 months), and long-term (1 year) momentum signals
2. **Sector neutrality** - Reduces sector concentration risk by neutralizing sector biases
3. **Inverse volatility weighting** - Allocates more capital to lower volatility stocks
4. **Regime awareness** - Adjusts position sizes based on market volatility and trend conditions
5. **Advanced risk management** - Implements position limits, sector limits, and drawdown controls

### Key Features

#### 1. Signal Generation
- **Momentum Scoring**: Composite momentum scores using price-based and rank-based momentum
- **Sector Neutrality**: Optional sector-neutral momentum calculation to reduce sector biases
- **Multi-factor Approach**: Combines momentum with volatility, technical indicators, and market regime filters

#### 2. Position Sizing
- **Top Quantile Selection**: Selects top N% of stocks by momentum score (default: 30%)
- **Inverse Volatility Weighting**: Weights positions inversely to their volatility
- **Risk Budgeting**: Limits individual position sizes and sector exposures

#### 3. Risk Management
- **Position Limits**: Maximum 8% allocation to any single stock
- **Sector Limits**: Maximum 25% allocation to any sector
- **Drawdown Protection**: Stops trading if portfolio drawdown exceeds 20%
- **Volatility Targeting**: Optional volatility targeting for consistent risk exposure

#### 4. Market Regime Adaptation
- **Volatility Regime Detection**: Reduces exposure during high volatility periods
- **Trend Filtering**: Considers market trend direction in position sizing
- **Risk-off Scaling**: Scales down positions during market stress periods

## Technical Implementation

### Architecture

The strategy is implemented using a modular architecture:

```
tradingguru/
├── core/              # Core framework
│   ├── base.py       # Base strategy class
│   ├── engine.py     # Backtesting engine
│   ├── risk.py       # Risk management
│   └── utils.py      # Utility functions
├── data/             # Data handling
│   └── nifty_data_loader.py
├── features/         # Feature engineering
│   └── nifty_features.py
└── models/           # Strategy implementations
    └── momentum_nifty.py
```

### Data Pipeline

1. **Data Fetching**: Yahoo Finance API for NIFTY 50 constituents
2. **Feature Engineering**: 50+ technical and fundamental features
3. **Signal Generation**: Composite momentum scoring with regime filters
4. **Portfolio Construction**: Risk-aware position sizing and allocation

### Features Calculated

#### Momentum Features
- Short-term momentum (1 month)
- Medium-term momentum (3 months) 
- Long-term momentum (1 year)
- Cross-sectional momentum rankings
- Momentum acceleration (momentum of momentum)

#### Risk Features
- Rolling volatility (multiple timeframes)
- Downside volatility
- Volatility percentile rankings
- Beta to market index

#### Technical Features
- RSI (Relative Strength Index)
- Bollinger Bands position
- Moving average crossovers
- Price channel positions

#### Market Regime Features
- Market volatility regime
- Market trend direction
- VIX-like fear indicator
- Correlation to market

#### Sector Features (if enabled)
- Sector-relative returns
- Sector momentum
- Sector beta coefficients

## Configuration

### Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_quantile` | 0.3 | Top 30% of stocks by momentum |
| `min_positions` | 10 | Minimum number of positions |
| `max_positions` | 20 | Maximum number of positions |
| `max_weight` | 0.08 | Maximum individual position weight |
| `sector_neutral` | True | Enable sector neutrality |
| `max_sector_weight` | 0.25 | Maximum sector exposure |
| `vol_target` | 0.15 | Target portfolio volatility |
| `risk_off_enabled` | True | Enable risk-off scaling |

### Lookback Periods

| Feature | Default | Description |
|---------|---------|-------------|
| `lookback_short` | 21 | Short-term momentum (days) |
| `lookback_medium` | 63 | Medium-term momentum (days) |
| `lookback_long` | 252 | Long-term momentum (days) |
| `volatility_lookback` | 21 | Volatility calculation window |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_drawdown_limit` | -20% | Maximum portfolio drawdown |
| `max_turnover` | 200% | Maximum annual turnover |
| `vol_regime_threshold` | 1.5 | High volatility threshold |
| `risk_off_vol_threshold` | 0.25 | Risk-off volatility threshold |

## Usage

### Running a Backtest

```bash
python scripts/run_nifty_momentum_backtest.py \
    --config config/nifty_momentum_backtest.yaml \
    --output results/
```

### Parameter Sweep

```bash
python scripts/param_sweep_nifty_momentum.py \
    --config config/nifty_momentum_backtest.yaml \
    --param-config config/param_ranges.yaml \
    --output param_sweep_results/ \
    --jobs 4
```

### Python API

```python
from tradingguru.data.nifty_data_loader import NiftyDataLoader
from tradingguru.models.momentum_nifty import EnhancedNiftyMomentumStrategy
from tradingguru.core.engine import BacktestEngine, BacktestConfig

# Load data
data_loader = NiftyDataLoader()
universe = data_loader.load_nifty50_universe()
sector_mapping = data_loader.load_sector_mapping()

# Fetch price data
price_data, benchmark_data = data_loader.fetch_complete_dataset(
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# Create strategy
strategy = EnhancedNiftyMomentumStrategy(
    universe=universe,
    sector_mapping=sector_mapping,
    top_quantile=0.3,
    sector_neutral=True
)

# Configure backtest
config = BacktestConfig(
    start_date="2020-01-01",
    end_date="2024-12-31",
    initial_capital=1000000
)

# Run backtest
engine = BacktestEngine(strategy, config, sector_mapping)
results = engine.run_backtest(price_data, benchmark_data)

# Generate report
print(engine.generate_report())
```

## Performance Expectations

### Historical Performance (Hypothetical)

Based on academic literature and similar strategies:

- **Expected Annual Return**: 12-18% (before costs)
- **Volatility**: 15-25% annually
- **Sharpe Ratio**: 0.6 - 1.2
- **Maximum Drawdown**: 15-30%
- **Win Rate**: 55-65%

### Risk Characteristics

- **Market Beta**: 0.8 - 1.2 (varies with regime)
- **Sector Concentration**: Limited by 25% sector cap
- **Position Concentration**: Limited by 8% position cap
- **Turnover**: 100-300% annually

## Risk Considerations

### Market Risks
1. **Momentum Crashes**: Periods when momentum reverses sharply
2. **Regime Changes**: Strategy may underperform during trend reversals
3. **Concentration Risk**: Limited to NIFTY 50 universe
4. **Liquidity Risk**: Large positions may face execution challenges

### Implementation Risks
1. **Transaction Costs**: High turnover can erode returns
2. **Data Quality**: Depends on accurate price and volume data
3. **Slippage**: Market impact on large trades
4. **Technology Risk**: System failures during critical periods

### Mitigation Strategies
1. **Risk-off Scaling**: Reduces exposure during volatile periods
2. **Position Limits**: Prevents over-concentration
3. **Drawdown Controls**: Stops trading at maximum loss levels
4. **Diversification**: Spreads risk across sectors and positions

## Customization

### Strategy Variants

#### Conservative Version
- Lower `top_quantile` (0.2)
- Higher `min_positions` (15)
- Lower `max_weight` (0.05)
- Enabled `volatility_target` (0.12)

#### Aggressive Version
- Higher `top_quantile` (0.4)
- Lower `min_positions` (8)
- Higher `max_weight` (0.12)
- Disabled sector neutrality

#### Sector-Focused Version
- Disabled `sector_neutral`
- Higher `max_sector_weight` (0.5)
- Sector-specific momentum calculations

### Custom Features

The framework allows easy addition of new features:

```python
class CustomFeaturesPipeline(NiftyFeaturesPipeline):
    def _calculate_custom_features(self, data):
        # Add your custom features here
        features = pd.DataFrame(index=data.index)
        # ... custom calculations
        return features
```

## Monitoring and Maintenance

### Key Metrics to Monitor
1. **Sharpe Ratio**: Risk-adjusted returns
2. **Maximum Drawdown**: Worst peak-to-trough decline
3. **Turnover**: Portfolio churn rate
4. **Sector Exposures**: Concentration risk
5. **Factor Loadings**: Style exposures

### Rebalancing Schedule
- **Daily**: Default rebalancing frequency
- **Weekly**: Reduced transaction costs
- **Monthly**: Lower turnover, may miss short-term opportunities

### Parameter Tuning
- **Quarterly Review**: Assess parameter effectiveness
- **Annual Optimization**: Full parameter sweep
- **Regime Adaptation**: Adjust for changing market conditions

## Conclusion

The Enhanced NIFTY 50 Momentum Strategy represents a sophisticated approach to momentum investing in the Indian equity market. By combining multiple momentum signals with robust risk management and regime awareness, it aims to capture momentum premiums while managing downside risk.

The modular design allows for easy customization and extension, making it suitable for both research and production environments. However, like all quantitative strategies, it requires careful monitoring, regular maintenance, and appropriate risk management to achieve consistent performance.

## References

1. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.

2. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228-250.

3. Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929-985.

4. Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221-247.

5. Barroso, P., & Santa-Clara, P. (2015). Momentum has its moments. *Journal of Financial Economics*, 116(1), 111-120.
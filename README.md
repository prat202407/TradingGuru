# TradingGuru

A comprehensive quantitative trading strategy framework with a focus on the Indian equity market, featuring an advanced NIFTY 50 momentum strategy.

## 🚀 Features

- **Enhanced Momentum Strategy**: Multi-timeframe momentum with sector neutrality and regime awareness
- **Advanced Risk Management**: Position limits, sector constraints, and drawdown protection
- **Comprehensive Backtesting**: Transaction cost modeling, benchmark comparison, and detailed analytics
- **Feature Engineering**: 50+ technical and fundamental features for signal generation
- **Data Management**: Automated data fetching from Yahoo Finance with caching
- **Flexible Framework**: Modular design for easy strategy development and customization

## 📊 Strategy Overview

The Enhanced NIFTY 50 Momentum Strategy implements a sophisticated approach to momentum investing:

### Key Components
- **Multi-timeframe Momentum**: Combines 1-month, 3-month, and 1-year momentum signals
- **Sector Neutrality**: Reduces sector concentration risk through sector-neutral scoring
- **Inverse Volatility Weighting**: Allocates more capital to lower volatility stocks
- **Market Regime Filters**: Adapts to changing market conditions and volatility regimes
- **Risk-off Scaling**: Reduces exposure during high volatility periods

### Performance Features
- **Dynamic Position Sizing**: Intelligent allocation based on momentum and risk metrics
- **Advanced Risk Controls**: Position limits, sector limits, and portfolio-level constraints
- **Transaction Cost Modeling**: Realistic cost estimates including commissions and market impact
- **Comprehensive Analytics**: Detailed performance attribution and risk analysis

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/prat202407/TradingGuru.git
cd TradingGuru

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

## 🎯 Quick Start

### Running a Backtest

```bash
# Run a single backtest with default configuration
python scripts/run_nifty_momentum_backtest.py

# Run with custom configuration
python scripts/run_nifty_momentum_backtest.py \
    --config config/nifty_momentum_backtest.yaml \
    --output results/
```

### Parameter Optimization

```bash
# Run parameter sweep to find optimal settings
python scripts/param_sweep_nifty_momentum.py \
    --config config/nifty_momentum_backtest.yaml \
    --param-config config/param_ranges.yaml \
    --output param_sweep_results/ \
    --jobs 4
```

### Python API Usage

```python
from tradingguru.data.nifty_data_loader import NiftyDataLoader
from tradingguru.models.momentum_nifty import EnhancedNiftyMomentumStrategy
from tradingguru.core.engine import BacktestEngine, BacktestConfig

# Initialize data loader
data_loader = NiftyDataLoader()
universe = data_loader.load_nifty50_universe()
sector_mapping = data_loader.load_sector_mapping()

# Fetch market data
price_data, benchmark_data = data_loader.fetch_complete_dataset(
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# Create and configure strategy
strategy = EnhancedNiftyMomentumStrategy(
    universe=universe,
    sector_mapping=sector_mapping,
    top_quantile=0.3,           # Top 30% momentum stocks
    sector_neutral=True,        # Enable sector neutrality
    max_weight=0.08,           # Max 8% per position
    risk_off_enabled=True      # Enable risk-off scaling
)

# Configure backtest
config = BacktestConfig(
    start_date="2020-01-01",
    end_date="2024-12-31",
    initial_capital=1000000    # 10 lakhs
)

# Run backtest
engine = BacktestEngine(strategy, config, sector_mapping)
results = engine.run_backtest(price_data, benchmark_data)

# Generate report
print(engine.generate_report())
```

## 📁 Project Structure

```
TradingGuru/
├── tradingguru/                 # Main package
│   ├── core/                   # Core framework
│   │   ├── base.py            # Base strategy class
│   │   ├── engine.py          # Backtesting engine
│   │   ├── risk.py            # Risk management
│   │   └── utils.py           # Utility functions
│   ├── data/                  # Data handling
│   │   └── nifty_data_loader.py
│   ├── features/              # Feature engineering
│   │   └── nifty_features.py
│   └── models/                # Strategy implementations
│       └── momentum_nifty.py
├── scripts/                   # Execution scripts
│   ├── run_nifty_momentum_backtest.py
│   └── param_sweep_nifty_momentum.py
├── config/                    # Configuration files
│   └── nifty_momentum_backtest.yaml
├── data/                      # Data files
│   └── universe/
│       ├── nifty50.json
│       └── nifty50_sectors.json
├── docs/                      # Documentation
│   └── NIFTY_MOMENTUM.md
├── tests/                     # Test suite
│   ├── test_base_strategy.py
│   └── test_nifty_pipeline_smoke.py
├── requirements.txt           # Dependencies
├── pyproject.toml            # Project metadata
└── README.md                 # This file
```

## ⚙️ Configuration

The strategy behavior can be customized through YAML configuration files:

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_quantile` | 0.3 | Percentage of top momentum stocks to select |
| `min_positions` | 10 | Minimum number of positions |
| `max_positions` | 20 | Maximum number of positions |
| `max_weight` | 0.08 | Maximum weight per position |
| `sector_neutral` | true | Enable sector-neutral momentum |
| `max_sector_weight` | 0.25 | Maximum sector concentration |
| `vol_target` | 0.15 | Target portfolio volatility |
| `risk_off_enabled` | true | Enable risk-off regime scaling |

### Risk Management

- **Position Limits**: Individual position capped at 8%
- **Sector Limits**: Sector exposure capped at 25%
- **Drawdown Protection**: Trading stops at 20% drawdown
- **Turnover Control**: Maximum 200% annual turnover

## 📈 Performance Expectations

Based on academic literature and similar strategies:

- **Expected Annual Return**: 12-18% (before costs)
- **Volatility**: 15-25% annually
- **Sharpe Ratio**: 0.6 - 1.2
- **Maximum Drawdown**: 15-30%
- **Information Ratio vs NIFTY 50**: 0.3 - 0.8

*Note: These are hypothetical expectations based on research. Actual performance may vary significantly.*

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_base_strategy.py -v
python -m pytest tests/test_nifty_pipeline_smoke.py -v

# Run with coverage
python -m pytest tests/ --cov=tradingguru --cov-report=html
```

## 📖 Documentation

- **Strategy Guide**: [docs/NIFTY_MOMENTUM.md](docs/NIFTY_MOMENTUM.md)
- **API Documentation**: Auto-generated from docstrings
- **Configuration Reference**: See `config/nifty_momentum_backtest.yaml`

## 🔧 Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black tradingguru/ scripts/ tests/
flake8 tradingguru/ scripts/ tests/

# Run type checking
mypy tradingguru/
```

### Adding New Features

1. **New Strategy**: Inherit from `BaseStrategy` in `tradingguru.models`
2. **New Features**: Add to `NiftyFeaturesPipeline` in `tradingguru.features`
3. **New Data Sources**: Extend `NiftyDataLoader` in `tradingguru.data`
4. **New Risk Controls**: Extend `RiskManager` in `tradingguru.core.risk`

## ⚠️ Risk Disclaimer

This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. 

**Important Considerations:**
- **Market Risk**: Momentum strategies can experience severe drawdowns during market reversals
- **Implementation Risk**: Real-world trading involves slippage, costs, and execution challenges
- **Model Risk**: Quantitative models may fail during regime changes or market stress
- **Regulatory Risk**: Trading regulations and market structure may change

Always consult with qualified financial advisors before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/prat202407/TradingGuru/issues)
- **Discussions**: [GitHub Discussions](https://github.com/prat202407/TradingGuru/discussions)
- **Email**: [Your contact email]

## 🙏 Acknowledgments

- Yahoo Finance for providing market data
- Academic research on momentum strategies
- Open source Python ecosystem (pandas, numpy, scipy, scikit-learn)
- NIFTY 50 constituent companies for creating a diversified universe

---

**Disclaimer**: This is a research and educational project. Use at your own risk.
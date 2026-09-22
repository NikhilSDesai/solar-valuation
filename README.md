# Solar Asset Valuation Model

Stochastic valuation model for solar infrastructure assets with live market data integration and Monte Carlo simulation.

## Overview

This project builds a Monte Carlo simulation framework for valuing solar energy projects, incorporating:

- **Live electricity prices** from UK National Grid ESO
- **Solar irradiance data** from Open-Meteo
- **Carbon prices** from EU ETS markets
- **Stochastic price models** (Ornstein-Uhlenbeck, GBM, Jump-Diffusion)
- **Monte Carlo NPV simulation** with 10,000+ scenarios
- **Risk metrics** (VaR, CVaR, sensitivity analysis)

## Project Structure

```
solar-valuation/
├── src/data/           # Data ingestion pipeline
│   ├── ngeso_client.py     # UK electricity prices
│   ├── openmeteo_client.py # Solar irradiance
│   ├── carbon_client.py    # EU ETS carbon prices
│   ├── storage.py          # DuckDB storage layer
│   └── pipeline.py         # Orchestration
├── src/models/         # Valuation models
│   ├── price_models.py     # OU, GBM, jump-diffusion processes
│   ├── solar_asset.py      # Asset cash flow model
│   ├── valuation.py        # Monte Carlo engine
│   └── risk.py             # VaR, CVaR, sensitivity
├── notebooks/          # Analysis notebooks
├── config/             # Configuration
└── data/               # Local data storage
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,viz]"

# Run the dashboard
streamlit run app.py

# Or run data backfill (works without API keys)
python -m src.data.pipeline --backfill
```

## Data Sources

| Source | Data | Auth Required |
|--------|------|---------------|
| [UK NGESO](https://api.nationalgrideso.com/) | GB electricity prices | No |
| [Open-Meteo](https://open-meteo.com/) | Solar irradiance | No |
| [OilPriceAPI](https://www.oilpriceapi.com/) | EU ETS carbon prices | Free tier |

## Roadmap

- [x] Data ingestion pipeline
- [x] DuckDB storage layer
- [x] Live price feeds
- [x] Stochastic price models (OU, GBM, Jump-Diffusion)
- [x] Monte Carlo simulation engine
- [x] DCF valuation model
- [x] Risk metrics (VaR, CVaR)
- [x] Price model calibration from historical data
- [x] Streamlit dashboard
- [ ] Bayesian parameter updating

## License

MIT

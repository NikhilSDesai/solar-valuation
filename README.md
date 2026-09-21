# Solar Asset Valuation Model

Stochastic valuation model for solar infrastructure assets with live market data integration.

## Overview

This project builds a Monte Carlo simulation framework for valuing solar energy projects, incorporating:

- **Live electricity prices** from UK National Grid ESO
- **Solar irradiance data** from Open-Meteo
- **Carbon prices** from EU ETS markets
- **Stochastic cash flow modeling** with risk quantification

## Project Structure

```
solar-valuation/
├── src/data/           # Data ingestion pipeline
│   ├── ngeso_client.py     # UK electricity prices
│   ├── openmeteo_client.py # Solar irradiance
│   ├── carbon_client.py    # EU ETS carbon prices
│   ├── storage.py          # DuckDB storage layer
│   └── pipeline.py         # Orchestration
├── src/models/         # Valuation models (TODO)
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

# Run data backfill (works without API keys)
python -m src.data.pipeline --backfill

# Check database stats
python -m src.data.pipeline --stats
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
- [ ] Stochastic price models
- [ ] Monte Carlo simulation engine
- [ ] DCF valuation model
- [ ] Risk metrics (VaR, CVaR)
- [ ] Streamlit dashboard

## License

MIT

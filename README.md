# Solar Asset Valuation

Teaching myself more about data engineering, Monte Carlo Simulation and building apps on streamlit. Inspired by work being done at Vallorii.

Monte Carlo simulation for solar PV project valuation.

**[Live Demo](https://solar-valuation.streamlit.app/)**

## What this app does

- Stochastic electricity price modeling (Ornstein-Uhlenbeck)
- 10,000+ scenario NPV simulation
- Risk metrics: VaR, CVaR, sensitivity analysis
- Live data from UK NGESO, Open-Meteo, EU ETS

## Run Locally

```bash
pip install -e ".[dev,viz]"
streamlit run app.py
```

## License

MIT

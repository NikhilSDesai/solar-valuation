"""Streamlit dashboard for solar asset valuation.

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

# Color scheme
COLORS = {
    "bg": "#000000",
    "bg_secondary": "#1a1a1a",
    "primary": "#34c9bd",
    "primary_dark": "#24b9ae",
    "accent": "#34c9bd",
    "text": "#e0e0e0",
    "negative": "#e85d75",
    "positive": "#34c9bd",
}

# Custom Plotly template
PLOT_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor": COLORS["bg"],
        "font": {"color": COLORS["text"]},
        "xaxis": {
            "gridcolor": "#2a2a2a",
            "linecolor": "#2a2a2a",
            "zerolinecolor": "#2a2a2a",
        },
        "yaxis": {
            "gridcolor": "#2a2a2a",
            "linecolor": "#2a2a2a",
            "zerolinecolor": "#2a2a2a",
        },
        "colorway": [COLORS["primary"], COLORS["accent"], "#a78bfa", "#f472b6"],
    }
}

from src.models import (
    SolarAsset,
    OUProcess,
    MonteCarloValuation,
    quick_valuation,
)
from src.models.calibration import (
    generate_uk_like_prices,
    calibrate_ou_robust,
)
from src.models.risk import scenario_analysis

# Page config
st.set_page_config(
    page_title="Solar Valuation Model",
    page_icon=None,
    layout="wide",
)

st.title("Solar Asset Valuation")

# Project description
with st.expander("About this model", expanded=False):
    st.markdown("""
    ### What is this?

    This tool estimates the **financial value** of a solar power project by simulating
    thousands of possible future scenarios. Instead of relying on a single "best guess"
    forecast, it shows you the full range of outcomes and their probabilities.

    ### Why does this matter?

    Solar projects are long-term investments (25+ years) with uncertain revenues.
    Electricity prices fluctuate daily, weather varies year to year, and policies change.
    Traditional valuations use fixed assumptions that hide this uncertainty. This model
    makes it visible, helping investors and developers understand the real risks.

    ### Key methods used

    | Method | What it does |
    |--------|--------------|
    | **Ornstein-Uhlenbeck Process** | Models electricity prices as mean-reverting (prices tend to return to a long-term average, unlike stocks) |
    | **Monte Carlo Simulation** | Runs 10,000+ random scenarios to build a probability distribution of outcomes |
    | **Discounted Cash Flow (DCF)** | Calculates Net Present Value by discounting future cash flows to today's value |
    | **Value at Risk (VaR)** | Measures downside risk: the worst-case loss at a given confidence level |
    | **Robust Calibration** | Estimates model parameters from historical data while handling price spikes |

    ### How to use it

    1. Adjust the **asset parameters** in the sidebar (capacity, costs, project life)
    2. Set **price assumptions** or calibrate from synthetic data
    3. View the **NPV distribution** to understand expected value and uncertainty
    4. Check **scenarios** to see how different market conditions affect value
    """)

st.caption("Monte Carlo simulation for solar PV project valuation")

# Sidebar - Asset Parameters
st.sidebar.header("Asset Parameters")

capacity_mw = st.sidebar.slider(
    "Capacity (MW)",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
)

capacity_factor = st.sidebar.slider(
    "Capacity Factor (%)",
    min_value=8,
    max_value=20,
    value=11,
) / 100

capex_per_mw = st.sidebar.slider(
    "CAPEX (£k/MW)",
    min_value=300,
    max_value=600,
    value=450,
    step=25,
) * 1000

project_life = st.sidebar.slider(
    "Project Life (years)",
    min_value=15,
    max_value=35,
    value=25,
)

st.sidebar.header("Price Parameters")

use_calibration = st.sidebar.checkbox("Calibrate from synthetic data", value=False)

if use_calibration:
    st.sidebar.caption("Using robust calibration on 2 years of synthetic UK prices")
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    prices = generate_uk_like_prices(n_days=730, seed=int(seed))
    calibration = calibrate_ou_robust(prices, dt=1/365)
    theta = calibration.process.theta
    kappa = calibration.process.kappa
    sigma = calibration.process.sigma
    st.sidebar.success(f"Calibrated: θ=£{theta:.0f}, κ={kappa:.2f}, σ=£{sigma:.0f}")
else:
    theta = st.sidebar.slider(
        "Long-term price θ (£/MWh)",
        min_value=30,
        max_value=100,
        value=60,
    )
    kappa = st.sidebar.slider(
        "Mean reversion κ",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )
    sigma = st.sidebar.slider(
        "Volatility σ (£/MWh)",
        min_value=5,
        max_value=30,
        value=15,
    )

st.sidebar.header("Simulation")

n_simulations = st.sidebar.select_slider(
    "Number of simulations",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000,
)

# Create asset and process
asset = SolarAsset(
    capacity_mw=capacity_mw,
    capacity_factor=capacity_factor,
    capex_per_mw=capex_per_mw,
    project_life_years=project_life,
)

price_process = OUProcess(
    theta=theta,
    kappa=kappa,
    sigma=sigma,
    x0=theta,
)

# Run valuation
@st.cache_data
def run_valuation(
    capacity_mw, capacity_factor, capex_per_mw, project_life,
    theta, kappa, sigma, n_sims
):
    asset = SolarAsset(
        capacity_mw=capacity_mw,
        capacity_factor=capacity_factor,
        capex_per_mw=capex_per_mw,
        project_life_years=project_life,
    )
    process = OUProcess(theta=theta, kappa=kappa, sigma=sigma, x0=theta)
    mc = MonteCarloValuation(
        asset=asset,
        electricity_process=process,
        n_simulations=n_sims,
        seed=42,
    )
    return mc.run(store_paths=True), asset

result, asset = run_valuation(
    capacity_mw, capacity_factor, capex_per_mw, project_life,
    theta, kappa, sigma, n_simulations
)

# Main content - tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Valuation Results",
    "Price Simulation",
    "Scenario Analysis",
    "Asset Details",
])

with tab1:
    st.header("Valuation Results")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Expected NPV",
            f"£{result.npv_mean/1e6:.1f}M",
            delta=f"σ = £{result.npv_std/1e6:.1f}M",
        )

    with col2:
        prob_positive = result.probability_positive_npv()
        st.metric(
            "P(Profitable)",
            f"{prob_positive:.0%}",
            delta="Good" if prob_positive > 0.7 else "Risky" if prob_positive > 0.5 else "Poor",
            delta_color="normal" if prob_positive > 0.7 else "off" if prob_positive > 0.5 else "inverse",
        )

    with col3:
        irr_display = f"{result.irr_mean:.1%}" if result.irr_mean else "N/A"
        st.metric("Expected IRR", irr_display)

    with col4:
        st.metric(
            "VaR (95%)",
            f"£{result.var_95/1e6:.1f}M",
        )

    st.divider()

    # NPV Distribution
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=result.npv_samples / 1e6,
            nbinsx=80,
            marker_color=COLORS["primary"],
            opacity=0.8,
            name='NPV Distribution',
        ))

        # Add vertical lines
        fig.add_vline(x=result.npv_mean/1e6, line_dash="solid", line_color="#a78bfa",
                      annotation_text=f"Mean: £{result.npv_mean/1e6:.1f}M",
                      annotation_font_color=COLORS["text"])
        fig.add_vline(x=0, line_dash="dot", line_color=COLORS["text"])
        fig.add_vline(x=result.var_95/1e6, line_dash="dash", line_color=COLORS["negative"],
                      annotation_text=f"VaR 95%",
                      annotation_font_color=COLORS["text"])

        fig.update_layout(
            title=f"NPV Distribution ({n_simulations:,} simulations)",
            xaxis_title="NPV (£ millions)",
            yaxis_title="Frequency",
            showlegend=False,
            height=400,
            **PLOT_TEMPLATE["layout"],
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Percentiles")

        percentile_data = {
            "Percentile": ["5th", "25th", "50th", "75th", "95th"],
            "NPV (£M)": [
                f"{result.npv_percentiles[5]/1e6:.2f}",
                f"{result.npv_percentiles[25]/1e6:.2f}",
                f"{result.npv_percentiles[50]/1e6:.2f}",
                f"{result.npv_percentiles[75]/1e6:.2f}",
                f"{result.npv_percentiles[95]/1e6:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(percentile_data), hide_index=True)

        st.subheader("Risk Metrics")
        st.write(f"**CVaR (95%):** £{result.cvar_95/1e6:.2f}M")
        st.write(f"**NPV Range:** £{result.npv_samples.min()/1e6:.1f}M to £{result.npv_samples.max()/1e6:.1f}M")

with tab2:
    st.header("Simulated Price Paths")

    if result.price_paths is not None:
        n_display = min(100, len(result.price_paths))

        fig = go.Figure()

        # Plot sample paths
        for i in range(n_display):
            fig.add_trace(go.Scatter(
                x=list(range(1, project_life + 1)),
                y=result.price_paths[i],
                mode='lines',
                line=dict(color=COLORS["primary_dark"], width=0.5),
                opacity=0.25,
                showlegend=False,
            ))

        # Mean path
        mean_path = result.price_paths.mean(axis=0)
        fig.add_trace(go.Scatter(
            x=list(range(1, project_life + 1)),
            y=mean_path,
            mode='lines',
            line=dict(color="#a78bfa", width=3),
            name='Mean Path',
        ))

        # Long-term mean
        fig.add_hline(y=theta, line_dash="dash", line_color=COLORS["primary"],
                      annotation_text=f"θ = £{theta}/MWh",
                      annotation_font_color=COLORS["text"])

        fig.update_layout(
            title=f"Simulated Electricity Prices ({n_display} paths shown)",
            xaxis_title="Year",
            yaxis_title="Price (£/MWh)",
            height=450,
            **PLOT_TEMPLATE["layout"],
        )

        st.plotly_chart(fig, use_container_width=True)

        # Price statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Year 1 Mean", f"£{result.price_paths[:, 0].mean():.1f}/MWh")
        with col2:
            st.metric(f"Year {project_life} Mean", f"£{result.price_paths[:, -1].mean():.1f}/MWh")
        with col3:
            st.metric("Long-term Mean θ", f"£{theta}/MWh")
    else:
        st.info("Price paths not stored. Re-run with store_paths=True.")

with tab3:
    st.header("Scenario Analysis")

    @st.cache_data
    def get_scenarios(capacity_mw, capacity_factor, capex_per_mw, project_life, theta):
        asset = SolarAsset(
            capacity_mw=capacity_mw,
            capacity_factor=capacity_factor,
            capex_per_mw=capex_per_mw,
            project_life_years=project_life,
        )
        return scenario_analysis(asset, base_price=theta, n_simulations=5000)

    scenarios = get_scenarios(capacity_mw, capacity_factor, capex_per_mw, project_life, theta)

    # Scenario comparison chart
    names = [s.name for s in scenarios]
    npvs = [s.npv_mean / 1e6 for s in scenarios]
    probs = [s.probability_positive for s in scenarios]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = [COLORS["positive"] if npv > 0 else COLORS["negative"] for npv in npvs]

    fig.add_trace(
        go.Bar(x=names, y=npvs, name="NPV (£M)", marker_color=colors),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=names, y=[p*100 for p in probs], name="P(Profitable) %",
                   mode='markers+lines', marker=dict(size=12, color="#a78bfa")),
        secondary_y=True,
    )

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text"], secondary_y=False)

    fig.update_layout(
        title="Scenario Comparison",
        height=400,
        **PLOT_TEMPLATE["layout"],
    )
    fig.update_yaxes(title_text="NPV (£ millions)", secondary_y=False)
    fig.update_yaxes(title_text="P(Profitable) %", secondary_y=True, range=[0, 105])

    st.plotly_chart(fig, use_container_width=True)

    # Scenario details
    st.subheader("Scenario Details")
    scenario_df = pd.DataFrame([
        {
            "Scenario": s.name,
            "Description": s.description,
            "NPV (£M)": f"{s.npv_mean/1e6:+.1f}",
            "P(Profitable)": f"{s.probability_positive:.0%}",
            "IRR": f"{s.irr_mean:.1%}" if s.irr_mean else "N/A",
        }
        for s in scenarios
    ])
    st.dataframe(scenario_df, hide_index=True, use_container_width=True)

with tab4:
    st.header("Asset Details")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Parameters")
        st.write(f"**Capacity:** {capacity_mw} MW")
        st.write(f"**Capacity Factor:** {capacity_factor:.1%}")
        st.write(f"**Project Life:** {project_life} years")
        st.write(f"**Total CAPEX:** £{asset.total_capex/1e6:.1f}M")
        st.write(f"**Annual Fixed OPEX:** £{asset.annual_opex_fixed/1e3:.0f}k")
        st.write(f"**WACC:** {asset.wacc:.2%}")

    with col2:
        st.subheader("Generation Profile")
        years = np.arange(1, project_life + 1)
        generation = asset.generation_profile(project_life)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=generation / 1000,
            fill='tozeroy',
            fillcolor=f"rgba(52, 201, 189, 0.3)",
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Year",
            yaxis_title="Generation (GWh)",
            **PLOT_TEMPLATE["layout"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Model Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Long-term mean (θ):** £{theta}/MWh")
    with col2:
        st.write(f"**Mean reversion (κ):** {kappa:.2f}")
        st.write(f"**Half-life:** {np.log(2)/kappa:.1f} years")
    with col3:
        st.write(f"**Volatility (σ):** £{sigma}/MWh")

# Footer
st.divider()
st.caption("Solar Valuation Model | Monte Carlo simulation with Ornstein-Uhlenbeck price dynamics")

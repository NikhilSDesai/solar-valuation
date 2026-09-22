"""Risk metrics and sensitivity analysis for solar valuations.

Provides:
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Sensitivity analysis (tornado charts)
- Scenario analysis
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .solar_asset import SolarAsset
from .price_models import OUProcess
from .valuation import MonteCarloValuation, ValuationResult


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    parameter: str
    base_value: float
    test_values: NDArray[np.float64]
    npv_results: NDArray[np.float64]

    def impact_range(self) -> tuple[float, float]:
        """Min and max NPV across tested values."""
        return float(np.min(self.npv_results)), float(np.max(self.npv_results))


def calculate_var(
    samples: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Calculate Value at Risk.

    VaR is the loss threshold such that there's a (1-confidence)
    probability of experiencing a worse outcome.

    Args:
        samples: NPV samples
        confidence: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR value (negative = potential loss)
    """
    return float(np.percentile(samples, (1 - confidence) * 100))


def calculate_cvar(
    samples: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that we're in the worst
    (1-confidence) tail of outcomes. More informative than VaR
    for fat-tailed distributions.

    Args:
        samples: NPV samples
        confidence: Confidence level

    Returns:
        CVaR value (expected loss in tail)
    """
    var = calculate_var(samples, confidence)
    tail_samples = samples[samples <= var]
    return float(np.mean(tail_samples)) if len(tail_samples) > 0 else var


def sensitivity_analysis(
    base_asset: SolarAsset,
    base_price_params: dict,
    n_simulations: int = 5_000,
    n_points: int = 11,
) -> list[SensitivityResult]:
    """Run sensitivity analysis on key parameters.

    Tests how NPV changes when each parameter varies
    while others remain at base values.

    Args:
        base_asset: Base case solar asset
        base_price_params: Dict with theta, kappa, sigma, x0
        n_simulations: Simulations per test point
        n_points: Number of test points per parameter

    Returns:
        List of SensitivityResult for each parameter
    """
    results = []

    # Parameters to test and their ranges (as % of base)
    tests = {
        "capacity_factor": (0.7, 1.3),
        "electricity_price": (0.6, 1.4),
        "electricity_volatility": (0.5, 2.0),
        "capex_per_mw": (0.8, 1.2),
        "opex_fixed_per_mw": (0.7, 1.3),
        "discount_rate": (0.8, 1.2),
    }

    for param, (low_mult, high_mult) in tests.items():
        # Get base value
        if param == "electricity_price":
            base = base_price_params["theta"]
        elif param == "electricity_volatility":
            base = base_price_params["sigma"]
        elif param == "discount_rate":
            base = base_asset.wacc
        else:
            base = getattr(base_asset, param)

        test_values = np.linspace(base * low_mult, base * high_mult, n_points)
        npv_results = np.zeros(n_points)

        for i, val in enumerate(test_values):
            # Create modified asset/process
            if param in ("capacity_factor", "capex_per_mw", "opex_fixed_per_mw"):
                test_asset = SolarAsset(
                    capacity_mw=base_asset.capacity_mw,
                    capacity_factor=val if param == "capacity_factor" else base_asset.capacity_factor,
                    capex_per_mw=val if param == "capex_per_mw" else base_asset.capex_per_mw,
                    opex_fixed_per_mw=val if param == "opex_fixed_per_mw" else base_asset.opex_fixed_per_mw,
                )
                test_process = OUProcess(**base_price_params)
                discount = None
            elif param in ("electricity_price", "electricity_volatility"):
                test_asset = base_asset
                price_params = base_price_params.copy()
                if param == "electricity_price":
                    price_params["theta"] = val
                    price_params["x0"] = val
                else:
                    price_params["sigma"] = val
                test_process = OUProcess(**price_params)
                discount = None
            else:  # discount_rate
                test_asset = base_asset
                test_process = OUProcess(**base_price_params)
                discount = val

            # Run valuation
            valuation = MonteCarloValuation(
                asset=test_asset,
                electricity_process=test_process,
                n_simulations=n_simulations,
                seed=42,
            )
            result = valuation.run(discount_rate=discount)
            npv_results[i] = result.npv_mean

        results.append(SensitivityResult(
            parameter=param,
            base_value=base,
            test_values=test_values,
            npv_results=npv_results,
        ))

    return results


@dataclass
class ScenarioResult:
    """Result from scenario analysis."""

    name: str
    description: str
    npv_mean: float
    npv_std: float
    probability_positive: float
    irr_mean: float | None


def scenario_analysis(
    base_asset: SolarAsset,
    base_price: float = 60.0,
    n_simulations: int = 10_000,
) -> list[ScenarioResult]:
    """Run pre-defined scenario analysis.

    Tests common scenarios:
    - Base case
    - High renewable penetration (lower prices, higher volatility)
    - Carbon policy boost (higher effective prices)
    - Technology improvement (lower CAPEX, better degradation)
    - Adverse weather (lower capacity factor)
    """
    scenarios = []

    # Base case
    base_process = OUProcess(theta=base_price, kappa=0.5, sigma=15.0, x0=base_price)
    base_val = MonteCarloValuation(base_asset, base_process, n_simulations, seed=42)
    base_result = base_val.run()
    scenarios.append(ScenarioResult(
        name="Base Case",
        description="Current market conditions",
        npv_mean=base_result.npv_mean,
        npv_std=base_result.npv_std,
        probability_positive=base_result.probability_positive_npv(),
        irr_mean=base_result.irr_mean,
    ))

    # High renewables scenario
    high_re_process = OUProcess(theta=45.0, kappa=0.3, sigma=25.0, x0=50.0)
    high_re_val = MonteCarloValuation(base_asset, high_re_process, n_simulations, seed=42)
    high_re_result = high_re_val.run()
    scenarios.append(ScenarioResult(
        name="High Renewables",
        description="Lower prices, higher volatility from intermittency",
        npv_mean=high_re_result.npv_mean,
        npv_std=high_re_result.npv_std,
        probability_positive=high_re_result.probability_positive_npv(),
        irr_mean=high_re_result.irr_mean,
    ))

    # Carbon boost scenario (add £15/MWh carbon value)
    carbon_process = OUProcess(theta=base_price + 15, kappa=0.5, sigma=15.0, x0=base_price + 15)
    carbon_val = MonteCarloValuation(base_asset, carbon_process, n_simulations, seed=42)
    carbon_result = carbon_val.run()
    scenarios.append(ScenarioResult(
        name="Carbon Policy Boost",
        description="Carbon price adds £15/MWh to effective revenue",
        npv_mean=carbon_result.npv_mean,
        npv_std=carbon_result.npv_std,
        probability_positive=carbon_result.probability_positive_npv(),
        irr_mean=carbon_result.irr_mean,
    ))

    # Technology improvement
    improved_asset = SolarAsset(
        capacity_mw=base_asset.capacity_mw,
        capacity_factor=base_asset.capacity_factor,
        capex_per_mw=base_asset.capex_per_mw * 0.85,  # 15% CAPEX reduction
        annual_degradation=0.003,  # Better degradation
    )
    tech_val = MonteCarloValuation(improved_asset, base_process, n_simulations, seed=42)
    tech_result = tech_val.run()
    scenarios.append(ScenarioResult(
        name="Technology Improvement",
        description="15% lower CAPEX, 0.3% annual degradation",
        npv_mean=tech_result.npv_mean,
        npv_std=tech_result.npv_std,
        probability_positive=tech_result.probability_positive_npv(),
        irr_mean=tech_result.irr_mean,
    ))

    # Adverse weather
    weather_asset = SolarAsset(
        capacity_mw=base_asset.capacity_mw,
        capacity_factor=base_asset.capacity_factor * 0.85,  # 15% lower CF
    )
    weather_val = MonteCarloValuation(weather_asset, base_process, n_simulations, seed=42)
    weather_result = weather_val.run()
    scenarios.append(ScenarioResult(
        name="Adverse Weather",
        description="15% lower capacity factor",
        npv_mean=weather_result.npv_mean,
        npv_std=weather_result.npv_std,
        probability_positive=weather_result.probability_positive_npv(),
        irr_mean=weather_result.irr_mean,
    ))

    return scenarios

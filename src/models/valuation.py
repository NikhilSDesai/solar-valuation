"""Monte Carlo valuation engine for solar assets.

Combines:
- Stochastic price simulations
- Solar asset cash flow models
- Risk metrics (VaR, CVaR, confidence intervals)
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .price_models import OUProcess, GBMProcess, StochasticProcess
from .solar_asset import SolarAsset


@dataclass
class ValuationResult:
    """Results from Monte Carlo valuation."""

    # NPV distribution
    npv_samples: NDArray[np.float64]
    npv_mean: float
    npv_std: float
    npv_median: float
    npv_percentiles: dict[int, float]  # e.g., {5: ..., 25: ..., 75: ..., 95: ...}

    # IRR distribution (if calculable)
    irr_samples: NDArray[np.float64] | None = None
    irr_mean: float | None = None
    irr_median: float | None = None

    # Risk metrics
    var_95: float = 0.0  # Value at Risk (5th percentile)
    cvar_95: float = 0.0  # Conditional VaR (expected shortfall)

    # Simulation inputs
    n_simulations: int = 0
    discount_rate: float = 0.0

    # Price paths (optional, for debugging)
    price_paths: NDArray[np.float64] | None = None

    def probability_positive_npv(self) -> float:
        """Probability that NPV > 0."""
        return float(np.mean(self.npv_samples > 0))

    def probability_above_threshold(self, threshold: float) -> float:
        """Probability that NPV exceeds a threshold."""
        return float(np.mean(self.npv_samples > threshold))

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            "=" * 50,
            "MONTE CARLO VALUATION RESULTS",
            "=" * 50,
            f"Simulations: {self.n_simulations:,}",
            f"Discount rate: {self.discount_rate:.1%}",
            "",
            "NPV Distribution:",
            f"  Mean:   £{self.npv_mean:,.0f}",
            f"  Median: £{self.npv_median:,.0f}",
            f"  Std:    £{self.npv_std:,.0f}",
            "",
            "Percentiles:",
        ]
        for pct, val in sorted(self.npv_percentiles.items()):
            lines.append(f"  {pct}th: £{val:,.0f}")

        lines.extend([
            "",
            "Risk Metrics:",
            f"  VaR (95%):  £{self.var_95:,.0f}",
            f"  CVaR (95%): £{self.cvar_95:,.0f}",
            f"  P(NPV > 0): {self.probability_positive_npv():.1%}",
        ])

        if self.irr_mean is not None:
            lines.extend([
                "",
                "IRR Distribution:",
                f"  Mean:   {self.irr_mean:.1%}",
                f"  Median: {self.irr_median:.1%}",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class MonteCarloValuation:
    """Monte Carlo simulation engine for solar asset valuation.

    Workflow:
    1. Simulate electricity price paths
    2. Optionally simulate carbon price paths
    3. Calculate annual revenues for each path
    4. Compute NPV distribution
    5. Calculate risk metrics
    """

    asset: SolarAsset
    electricity_process: StochasticProcess
    carbon_process: StochasticProcess | None = None

    # Simulation parameters
    n_simulations: int = 10_000
    seed: int | None = 42

    # Carbon value parameters
    grid_carbon_intensity: float = 0.2  # tCO2/MWh (UK grid average)
    include_carbon_value: bool = False

    def _simulate_prices(
        self,
        n_years: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Simulate electricity and carbon price paths.

        Returns:
            Tuple of (electricity_prices, carbon_prices)
            Each array has shape (n_simulations, n_years)
        """
        dt = 1.0  # Annual time steps

        # Simulate electricity prices
        elec_paths = self.electricity_process.simulate(
            n_paths=self.n_simulations,
            n_steps=n_years,
            dt=dt,
            seed=self.seed,
        )

        # Simulate carbon prices if configured
        carbon_paths = None
        if self.carbon_process is not None and self.include_carbon_value:
            carbon_paths = self.carbon_process.simulate(
                n_paths=self.n_simulations,
                n_steps=n_years,
                dt=dt,
                seed=self.seed + 1 if self.seed else None,
            )

        return elec_paths, carbon_paths

    def _calculate_npv(
        self,
        cash_flows: NDArray[np.float64],
        discount_rate: float,
    ) -> float:
        """Calculate NPV of a single cash flow stream."""
        n = len(cash_flows)
        # Year 0 is CAPEX (not discounted), years 1+ are operating CFs
        discount_factors = np.array([1 / (1 + discount_rate) ** t for t in range(n)])
        return float(np.sum(cash_flows * discount_factors))

    def _calculate_irr(
        self,
        cash_flows: NDArray[np.float64],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> float | None:
        """Calculate IRR using Newton-Raphson method."""
        # Initial guess based on simple payback
        if cash_flows[0] >= 0:
            return None  # No CAPEX, IRR undefined

        total_cf = np.sum(cash_flows[1:])
        if total_cf <= 0:
            return None  # Never profitable

        # Newton-Raphson iteration
        r = 0.1  # Initial guess

        for _ in range(max_iter):
            n = len(cash_flows)
            npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
            dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows))

            if abs(dnpv) < 1e-10:
                break

            r_new = r - npv / dnpv

            if abs(r_new - r) < tol:
                return r_new

            r = r_new

            # Bound to reasonable range
            r = max(-0.99, min(r, 1.0))

        return r if -0.5 < r < 1.0 else None

    def run(
        self,
        discount_rate: float | None = None,
        opex_escalation: float = 0.02,
        store_paths: bool = False,
    ) -> ValuationResult:
        """Run Monte Carlo valuation.

        Args:
            discount_rate: Discount rate for NPV (default: asset WACC)
            opex_escalation: Annual OPEX escalation rate
            store_paths: Whether to store price paths in results

        Returns:
            ValuationResult with NPV distribution and risk metrics
        """
        if discount_rate is None:
            discount_rate = self.asset.wacc

        n_years = self.asset.project_life_years

        # Simulate price paths
        elec_prices, carbon_prices = self._simulate_prices(n_years)

        # Calculate NPV for each simulation
        npv_samples = np.zeros(self.n_simulations)
        irr_samples = np.zeros(self.n_simulations)
        irr_valid = np.zeros(self.n_simulations, dtype=bool)

        for i in range(self.n_simulations):
            # Get price path for this simulation
            prices = elec_prices[i]

            # Add carbon value if configured
            if carbon_prices is not None:
                generation = self.asset.generation_profile(n_years)
                carbon_value_per_mwh = (
                    carbon_prices[i] * self.grid_carbon_intensity
                )
                prices = prices + carbon_value_per_mwh

            # Calculate cash flows
            cash_flows = self.asset.cash_flows(
                merchant_prices=prices,
                opex_escalation=opex_escalation,
                include_capex=True,
            )

            # Calculate NPV
            npv_samples[i] = self._calculate_npv(cash_flows, discount_rate)

            # Calculate IRR
            irr = self._calculate_irr(cash_flows)
            if irr is not None:
                irr_samples[i] = irr
                irr_valid[i] = True

        # Calculate statistics
        percentiles = {
            5: float(np.percentile(npv_samples, 5)),
            25: float(np.percentile(npv_samples, 25)),
            50: float(np.percentile(npv_samples, 50)),
            75: float(np.percentile(npv_samples, 75)),
            95: float(np.percentile(npv_samples, 95)),
        }

        # VaR and CVaR (5% tail)
        var_95 = float(np.percentile(npv_samples, 5))
        cvar_95 = float(np.mean(npv_samples[npv_samples <= var_95]))

        # IRR statistics (only for valid samples)
        irr_mean = None
        irr_median = None
        valid_irrs = irr_samples[irr_valid]
        if len(valid_irrs) > 0:
            irr_mean = float(np.mean(valid_irrs))
            irr_median = float(np.median(valid_irrs))

        return ValuationResult(
            npv_samples=npv_samples,
            npv_mean=float(np.mean(npv_samples)),
            npv_std=float(np.std(npv_samples)),
            npv_median=float(np.median(npv_samples)),
            npv_percentiles=percentiles,
            irr_samples=irr_samples if np.any(irr_valid) else None,
            irr_mean=irr_mean,
            irr_median=irr_median,
            var_95=var_95,
            cvar_95=cvar_95,
            n_simulations=self.n_simulations,
            discount_rate=discount_rate,
            price_paths=elec_prices if store_paths else None,
        )


def quick_valuation(
    capacity_mw: float = 50.0,
    capacity_factor: float = 0.11,
    electricity_price: float = 60.0,  # £/MWh
    electricity_volatility: float = 15.0,  # £/MWh annual std
    mean_reversion: float = 0.5,  # Mean reversion speed
    n_simulations: int = 10_000,
) -> ValuationResult:
    """Quick valuation with sensible defaults.

    Convenience function for rapid analysis.
    """
    asset = SolarAsset(
        capacity_mw=capacity_mw,
        capacity_factor=capacity_factor,
    )

    price_process = OUProcess(
        theta=electricity_price,
        kappa=mean_reversion,
        sigma=electricity_volatility,
        x0=electricity_price,
    )

    valuation = MonteCarloValuation(
        asset=asset,
        electricity_process=price_process,
        n_simulations=n_simulations,
    )

    return valuation.run()

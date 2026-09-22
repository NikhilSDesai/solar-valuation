"""Solar asset model for generation and cost estimation.

Models a utility-scale solar PV installation with:
- Capacity and generation profiles
- Degradation over time
- Operating costs (fixed and variable)
- Capital structure
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SolarAsset:
    """Represents a utility-scale solar PV installation.

    All monetary values in GBP unless specified.
    """

    # Capacity
    capacity_mw: float = 50.0  # Nameplate capacity
    capacity_factor: float = 0.11  # UK average ~11% for solar

    # Degradation
    annual_degradation: float = 0.005  # 0.5% per year (typical for mono-Si)

    # Revenue
    ppa_price: float | None = None  # Fixed PPA price (£/MWh), None = merchant
    ppa_fraction: float = 0.0  # Fraction of output under PPA (0-1)

    # Operating costs
    opex_fixed_per_mw: float = 8_000  # £/MW/year fixed O&M
    opex_variable_per_mwh: float = 0.5  # £/MWh variable O&M

    # Capital costs (for NPV calculation)
    capex_per_mw: float = 450_000  # £/MW installed cost
    project_life_years: int = 25

    # Location (for irradiance data)
    latitude: float = 51.75  # Oxfordshire default
    longitude: float = -1.25

    # Financing
    debt_fraction: float = 0.7  # 70% debt
    cost_of_debt: float = 0.05  # 5% interest
    cost_of_equity: float = 0.10  # 10% required return
    tax_rate: float = 0.25  # UK corporation tax

    @property
    def total_capex(self) -> float:
        """Total capital expenditure."""
        return self.capacity_mw * self.capex_per_mw

    @property
    def annual_opex_fixed(self) -> float:
        """Annual fixed operating costs."""
        return self.capacity_mw * self.opex_fixed_per_mw

    @property
    def wacc(self) -> float:
        """Weighted average cost of capital (post-tax)."""
        return (
            self.debt_fraction * self.cost_of_debt * (1 - self.tax_rate)
            + (1 - self.debt_fraction) * self.cost_of_equity
        )

    def annual_generation_mwh(self, year: int = 0) -> float:
        """Expected annual generation accounting for degradation.

        Args:
            year: Years since commissioning (0 = first year)

        Returns:
            Annual generation in MWh
        """
        degradation_factor = (1 - self.annual_degradation) ** year
        hours_per_year = 8760
        return self.capacity_mw * self.capacity_factor * degradation_factor * hours_per_year

    def generation_profile(
        self,
        n_years: int | None = None,
    ) -> NDArray[np.float64]:
        """Annual generation over project lifetime.

        Args:
            n_years: Number of years (default: project_life_years)

        Returns:
            Array of annual generation (MWh) for each year
        """
        n = n_years or self.project_life_years
        years = np.arange(n)
        degradation = (1 - self.annual_degradation) ** years
        base_generation = self.capacity_mw * self.capacity_factor * 8760
        return base_generation * degradation

    def opex_profile(
        self,
        generation: NDArray[np.float64],
        escalation_rate: float = 0.02,
    ) -> NDArray[np.float64]:
        """Operating costs over project lifetime.

        Args:
            generation: Annual generation array (MWh)
            escalation_rate: Annual cost escalation (e.g., 2% inflation)

        Returns:
            Array of annual OPEX (£)
        """
        n_years = len(generation)
        escalation = (1 + escalation_rate) ** np.arange(n_years)

        fixed = self.annual_opex_fixed * escalation
        variable = generation * self.opex_variable_per_mwh * escalation

        return fixed + variable

    def revenue(
        self,
        generation: NDArray[np.float64],
        merchant_prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate annual revenue from generation.

        Args:
            generation: Annual generation array (MWh)
            merchant_prices: Annual average merchant prices (£/MWh)

        Returns:
            Array of annual revenue (£)
        """
        if self.ppa_price is not None and self.ppa_fraction > 0:
            ppa_revenue = generation * self.ppa_fraction * self.ppa_price
            merchant_revenue = generation * (1 - self.ppa_fraction) * merchant_prices
            return ppa_revenue + merchant_revenue
        else:
            return generation * merchant_prices

    def cash_flows(
        self,
        merchant_prices: NDArray[np.float64],
        opex_escalation: float = 0.02,
        include_capex: bool = True,
    ) -> NDArray[np.float64]:
        """Calculate project cash flows.

        Args:
            merchant_prices: Array of annual average prices (£/MWh)
            opex_escalation: Annual OPEX escalation rate
            include_capex: Whether to include initial CAPEX as negative CF

        Returns:
            Array of annual cash flows (£)
        """
        n_years = len(merchant_prices)
        generation = self.generation_profile(n_years)
        revenue = self.revenue(generation, merchant_prices)
        opex = self.opex_profile(generation, opex_escalation)

        # Pre-tax cash flow
        operating_cf = revenue - opex

        # Simple tax (ignoring depreciation for now)
        tax = np.maximum(operating_cf * self.tax_rate, 0)
        after_tax_cf = operating_cf - tax

        if include_capex:
            # Prepend CAPEX as negative cash flow at year 0
            return np.concatenate([[-self.total_capex], after_tax_cf])
        else:
            return after_tax_cf


@dataclass
class SolarAssetWithUncertainty(SolarAsset):
    """Solar asset with stochastic parameters for Monte Carlo.

    Adds distributions for uncertain parameters:
    - Capacity factor (weather uncertainty)
    - OPEX escalation (inflation uncertainty)
    - Degradation rate (technology uncertainty)
    """

    # Capacity factor uncertainty
    cf_mean: float = 0.11
    cf_std: float = 0.015  # ~14% coefficient of variation

    # Degradation uncertainty
    degradation_mean: float = 0.005
    degradation_std: float = 0.001

    def sample_parameters(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> list["SolarAsset"]:
        """Generate Monte Carlo samples of asset parameters.

        Args:
            n_samples: Number of parameter sets to generate
            seed: Random seed

        Returns:
            List of SolarAsset instances with sampled parameters
        """
        rng = np.random.default_rng(seed)

        # Sample capacity factors (truncated normal to stay positive)
        cf_samples = rng.normal(self.cf_mean, self.cf_std, n_samples)
        cf_samples = np.clip(cf_samples, 0.05, 0.25)

        # Sample degradation rates
        deg_samples = rng.normal(self.degradation_mean, self.degradation_std, n_samples)
        deg_samples = np.clip(deg_samples, 0.001, 0.015)

        assets = []
        for i in range(n_samples):
            asset = SolarAsset(
                capacity_mw=self.capacity_mw,
                capacity_factor=cf_samples[i],
                annual_degradation=deg_samples[i],
                ppa_price=self.ppa_price,
                ppa_fraction=self.ppa_fraction,
                opex_fixed_per_mw=self.opex_fixed_per_mw,
                opex_variable_per_mwh=self.opex_variable_per_mwh,
                capex_per_mw=self.capex_per_mw,
                project_life_years=self.project_life_years,
                latitude=self.latitude,
                longitude=self.longitude,
                debt_fraction=self.debt_fraction,
                cost_of_debt=self.cost_of_debt,
                cost_of_equity=self.cost_of_equity,
                tax_rate=self.tax_rate,
            )
            assets.append(asset)

        return assets

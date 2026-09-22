"""Stochastic processes for electricity and carbon price modeling.

Electricity prices exhibit mean reversion (unlike stocks), so we use:
- Ornstein-Uhlenbeck (OU) process for base price dynamics
- Jump-diffusion for price spikes
- Geometric Brownian Motion (GBM) for carbon prices (trending)

References:
- Lucia & Schwartz (2002) - Electricity price modeling
- Cartea & Figueroa (2005) - UK electricity markets
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class StochasticProcess(Protocol):
    """Protocol for stochastic price processes."""

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate price paths.

        Args:
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps per path
            dt: Time step size (in years, e.g., 1/365 for daily)
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
        """
        ...


@dataclass
class OUProcess:
    """Ornstein-Uhlenbeck mean-reverting process.

    dX_t = κ(θ - X_t)dt + σdW_t

    Where:
        κ (kappa): Speed of mean reversion
        θ (theta): Long-term mean price
        σ (sigma): Volatility

    Electricity prices revert to marginal cost of generation,
    making OU a natural choice for power markets.
    """

    theta: float  # Long-term mean (£/MWh)
    kappa: float  # Mean reversion speed (higher = faster reversion)
    sigma: float  # Volatility (£/MWh)
    x0: float  # Initial price

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate OU process paths using exact discretization.

        Uses the analytical solution for OU to avoid discretization bias:
        X_{t+dt} = θ + (X_t - θ)e^{-κdt} + σ√((1-e^{-2κdt})/(2κ)) * Z
        """
        rng = np.random.default_rng(seed)

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = self.x0

        # Precompute constants for efficiency
        exp_kappa = np.exp(-self.kappa * dt)
        std = self.sigma * np.sqrt((1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa))

        for t in range(1, n_steps):
            z = rng.standard_normal(n_paths)
            paths[:, t] = (
                self.theta
                + (paths[:, t - 1] - self.theta) * exp_kappa
                + std * z
            )

        return paths

    @classmethod
    def fit_from_data(
        cls,
        prices: NDArray[np.float64],
        dt: float = 1 / 365,
    ) -> "OUProcess":
        """Estimate OU parameters from historical prices using OLS.

        Uses the discrete approximation:
        X_{t+1} - X_t ≈ κ(θ - X_t)dt + σ√dt * ε

        Which gives the regression:
        ΔX = α + βX_t + ε
        Where: κ = -β/dt, θ = -α/β, σ = std(ε)/√dt
        """
        x = prices[:-1]
        dx = np.diff(prices)

        # OLS regression: dx = alpha + beta * x
        n = len(x)
        x_mean = x.mean()
        dx_mean = dx.mean()

        beta = np.sum((x - x_mean) * (dx - dx_mean)) / np.sum((x - x_mean) ** 2)
        alpha = dx_mean - beta * x_mean

        residuals = dx - (alpha + beta * x)
        sigma_residuals = np.std(residuals)

        # Convert to OU parameters
        kappa = -beta / dt
        theta = -alpha / beta if abs(beta) > 1e-10 else prices.mean()
        sigma = sigma_residuals / np.sqrt(dt)

        return cls(
            theta=theta,
            kappa=max(kappa, 0.01),  # Ensure positive mean reversion
            sigma=max(sigma, 0.01),
            x0=prices[-1],
        )


@dataclass
class GBMProcess:
    """Geometric Brownian Motion for trending prices (e.g., carbon).

    dS_t = μS_t dt + σS_t dW_t

    Where:
        μ (mu): Drift rate (expected return)
        σ (sigma): Volatility

    Carbon prices tend to trend with policy tightening,
    making GBM more appropriate than mean-reverting models.
    """

    mu: float  # Drift (annual, e.g., 0.05 for 5%/year)
    sigma: float  # Volatility (annual)
    s0: float  # Initial price

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate GBM paths using log-normal exact solution."""
        rng = np.random.default_rng(seed)

        # Generate all random increments at once
        z = rng.standard_normal((n_paths, n_steps - 1))

        # Log returns: ln(S_{t+dt}/S_t) = (μ - σ²/2)dt + σ√dt * Z
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z

        # Cumulative sum of log returns
        log_paths = np.zeros((n_paths, n_steps))
        log_paths[:, 0] = np.log(self.s0)
        log_paths[:, 1:] = np.log(self.s0) + np.cumsum(log_returns, axis=1)

        return np.exp(log_paths)

    @classmethod
    def fit_from_data(
        cls,
        prices: NDArray[np.float64],
        dt: float = 1 / 365,
    ) -> "GBMProcess":
        """Estimate GBM parameters from historical prices."""
        log_returns = np.diff(np.log(prices))

        mu = log_returns.mean() / dt + 0.5 * (log_returns.std() / np.sqrt(dt)) ** 2
        sigma = log_returns.std() / np.sqrt(dt)

        return cls(mu=mu, sigma=sigma, s0=prices[-1])


@dataclass
class JumpDiffusionProcess:
    """Merton jump-diffusion for electricity price spikes.

    dS_t = μS_t dt + σS_t dW_t + S_t dJ_t

    Where J_t is a compound Poisson process with:
        λ: Jump intensity (expected jumps per year)
        μ_j: Mean jump size (log)
        σ_j: Jump size volatility (log)

    Captures sudden price spikes from demand surges,
    plant outages, or renewable intermittency.
    """

    mu: float  # Drift
    sigma: float  # Diffusion volatility
    lambda_: float  # Jump intensity (jumps/year)
    mu_j: float  # Mean jump size (log scale)
    sigma_j: float  # Jump size volatility (log scale)
    s0: float  # Initial price

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate jump-diffusion paths."""
        rng = np.random.default_rng(seed)

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = self.s0

        for t in range(1, n_steps):
            # Diffusion component
            z = rng.standard_normal(n_paths)
            diffusion = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z

            # Jump component (Poisson arrivals)
            n_jumps = rng.poisson(self.lambda_ * dt, n_paths)
            jump_sizes = np.zeros(n_paths)

            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jumps = rng.normal(self.mu_j, self.sigma_j, n_jumps[i])
                    jump_sizes[i] = np.sum(jumps)

            # Combined log return
            log_return = diffusion + jump_sizes
            paths[:, t] = paths[:, t - 1] * np.exp(log_return)

        return paths


@dataclass
class SeasonalOUProcess:
    """OU process with deterministic seasonality for electricity.

    X_t = f(t) + Y_t

    Where:
        f(t): Deterministic seasonal component
        Y_t: OU process for deseasonalized price

    Captures daily/weekly/annual patterns in electricity demand.
    """

    ou: OUProcess
    annual_amplitude: float  # Seasonal swing (£/MWh)
    peak_month: int = 1  # Month of peak prices (1=Jan for UK winter)

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int | None = None,
        start_day: int = 0,
    ) -> NDArray[np.float64]:
        """Simulate seasonal OU paths."""
        # Simulate base OU process
        base_paths = self.ou.simulate(n_paths, n_steps, dt, seed)

        # Add seasonal component
        days = start_day + np.arange(n_steps) * dt * 365
        # Convert peak_month to day of year (mid-month)
        peak_day = (self.peak_month - 1) * 30.5 + 15
        seasonal = self.annual_amplitude * np.cos(
            2 * np.pi * (days - peak_day) / 365
        )

        return base_paths + seasonal

"""Model calibration from historical price data.

Provides tools to:
- Estimate OU parameters from historical prices
- Generate synthetic data for testing
- Validate calibration quality
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .price_models import OUProcess, GBMProcess


@dataclass
class CalibrationResult:
    """Results from parameter calibration."""

    process: OUProcess | GBMProcess
    n_observations: int
    r_squared: float  # Goodness of fit
    log_likelihood: float

    # Parameter confidence intervals (if available)
    theta_ci: tuple[float, float] | None = None
    kappa_ci: tuple[float, float] | None = None
    sigma_ci: tuple[float, float] | None = None

    def summary(self) -> str:
        """Human-readable calibration summary."""
        lines = [
            "=" * 50,
            "CALIBRATION RESULTS",
            "=" * 50,
            f"Observations: {self.n_observations}",
            f"R-squared: {self.r_squared:.3f}",
            f"Log-likelihood: {self.log_likelihood:.1f}",
            "",
            "Estimated Parameters:",
        ]

        if isinstance(self.process, OUProcess):
            lines.extend([
                f"  θ (long-term mean): £{self.process.theta:.2f}/MWh",
                f"  κ (mean reversion): {self.process.kappa:.3f}",
                f"    Half-life: {np.log(2)/self.process.kappa:.1f} years",
                f"  σ (volatility): £{self.process.sigma:.2f}/MWh",
            ])

            if self.theta_ci:
                lines.append(f"    θ 95% CI: [{self.theta_ci[0]:.2f}, {self.theta_ci[1]:.2f}]")
            if self.kappa_ci:
                lines.append(f"    κ 95% CI: [{self.kappa_ci[0]:.3f}, {self.kappa_ci[1]:.3f}]")
            if self.sigma_ci:
                lines.append(f"    σ 95% CI: [{self.sigma_ci[0]:.2f}, {self.sigma_ci[1]:.2f}]")

        lines.append("=" * 50)
        return "\n".join(lines)


def calibrate_ou(
    prices: NDArray[np.float64],
    dt: float = 1 / 365,  # Daily data
    compute_ci: bool = True,
) -> CalibrationResult:
    """Calibrate Ornstein-Uhlenbeck parameters from price data.

    Uses maximum likelihood estimation via the discrete-time
    AR(1) representation of the OU process.

    Args:
        prices: Array of historical prices
        dt: Time step between observations (in years)
        compute_ci: Whether to compute confidence intervals

    Returns:
        CalibrationResult with fitted process and diagnostics
    """
    n = len(prices)
    x = prices[:-1]
    y = prices[1:]

    # OLS regression: y = a + b*x + e
    # For OU: y = theta*(1-exp(-kappa*dt)) + exp(-kappa*dt)*x + e
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)

    b = cov_xy / var_x  # exp(-kappa*dt)
    a = y_mean - b * x_mean  # theta*(1-b)

    # Recover OU parameters
    if b > 0 and b < 1:
        kappa = -np.log(b) / dt
        theta = a / (1 - b)
    else:
        # Fall back to simpler estimation
        kappa = 0.5  # Default
        theta = np.mean(prices)

    # Estimate sigma from residuals
    residuals = y - (a + b * x)
    var_residuals = np.var(residuals)

    # For OU: Var(residuals) = sigma^2 * (1 - exp(-2*kappa*dt)) / (2*kappa)
    if kappa > 0:
        sigma = np.sqrt(var_residuals * 2 * kappa / (1 - np.exp(-2 * kappa * dt)))
    else:
        sigma = np.std(prices) * np.sqrt(2 * 0.5)  # Fallback

    # Goodness of fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Log-likelihood (Gaussian)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * var_residuals) + 1)

    # Confidence intervals via asymptotic standard errors
    theta_ci = None
    kappa_ci = None
    sigma_ci = None

    if compute_ci and n > 10:
        # Standard errors from regression
        se_b = np.sqrt(var_residuals / var_x)
        se_a = np.sqrt(var_residuals * (1/n + x_mean**2 / var_x))

        # Delta method for theta and kappa
        if b > 0 and b < 1:
            # theta = a / (1-b), so se_theta ≈ |d(theta)/da| * se_a
            se_theta = se_a / (1 - b)
            theta_ci = (theta - 1.96 * se_theta, theta + 1.96 * se_theta)

            # kappa = -log(b)/dt
            se_kappa = se_b / (b * dt)
            kappa_ci = (max(0.01, kappa - 1.96 * se_kappa), kappa + 1.96 * se_kappa)

        # Sigma CI (chi-squared)
        se_sigma = sigma / np.sqrt(2 * (n - 2))
        sigma_ci = (sigma - 1.96 * se_sigma, sigma + 1.96 * se_sigma)

    process = OUProcess(
        theta=theta,
        kappa=max(kappa, 0.01),  # Ensure positive
        sigma=max(sigma, 0.01),
        x0=prices[-1],
    )

    return CalibrationResult(
        process=process,
        n_observations=n,
        r_squared=r_squared,
        log_likelihood=log_likelihood,
        theta_ci=theta_ci,
        kappa_ci=kappa_ci,
        sigma_ci=sigma_ci,
    )


def generate_synthetic_prices(
    n_days: int = 365,
    theta: float = 60.0,
    kappa: float = 0.5,
    sigma: float = 15.0,
    x0: float = 65.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate synthetic daily electricity prices for testing.

    Simulates an OU process with realistic UK parameters.

    Args:
        n_days: Number of days to simulate
        theta: Long-term mean (£/MWh)
        kappa: Mean reversion speed
        sigma: Volatility (£/MWh)
        x0: Starting price
        seed: Random seed

    Returns:
        Array of daily prices
    """
    process = OUProcess(theta=theta, kappa=kappa, sigma=sigma, x0=x0)

    # Simulate one path at daily frequency
    paths = process.simulate(
        n_paths=1,
        n_steps=n_days,
        dt=1/365,
        seed=seed,
    )

    return paths[0]


def generate_uk_like_prices(
    n_days: int = 730,  # 2 years
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate synthetic prices resembling UK electricity market.

    Includes:
    - Mean reversion to ~£55-65/MWh
    - Seasonal variation (higher in winter)
    - Occasional spikes

    Args:
        n_days: Number of days
        seed: Random seed

    Returns:
        Array of daily prices
    """
    rng = np.random.default_rng(seed)

    # Base OU process
    theta = 60.0
    kappa = 0.5
    sigma = 12.0
    x0 = 58.0

    process = OUProcess(theta=theta, kappa=kappa, sigma=sigma, x0=x0)
    base_prices = process.simulate(1, n_days, dt=1/365, seed=seed)[0]

    # Add seasonality (winter premium)
    days = np.arange(n_days)
    # Peak in January (day 15), trough in July (day 196)
    seasonality = 8 * np.cos(2 * np.pi * (days - 15) / 365)

    # Add occasional spikes (Poisson arrivals)
    n_spikes = rng.poisson(n_days / 30)  # ~1 spike per month
    spike_days = rng.choice(n_days, size=n_spikes, replace=False)
    spikes = np.zeros(n_days)
    spikes[spike_days] = rng.exponential(30, size=n_spikes)  # Mean spike = £30

    prices = base_prices + seasonality + spikes

    # Floor at £10/MWh (rare but possible in high renewable periods)
    prices = np.maximum(prices, 10.0)

    return prices


def calibrate_ou_robust(
    prices: NDArray[np.float64],
    dt: float = 1 / 365,
    outlier_threshold: float = 2.5,
) -> CalibrationResult:
    """Robust OU calibration that handles price spikes.

    Uses iterative reweighting to downweight outliers,
    which is important for electricity markets with spikes.

    Args:
        prices: Array of historical prices
        dt: Time step between observations
        outlier_threshold: Z-score threshold for outlier detection

    Returns:
        CalibrationResult with fitted process
    """
    # First pass: identify outliers based on returns
    returns = np.diff(prices)
    z_scores = np.abs((returns - returns.mean()) / returns.std())

    # Create weights (downweight outliers)
    weights = np.ones(len(returns))
    weights[z_scores > outlier_threshold] = 0.1

    n = len(prices) - 1
    x = prices[:-1]
    y = prices[1:]

    # Weighted OLS
    w_sum = np.sum(weights)
    wx_mean = np.sum(weights * x) / w_sum
    wy_mean = np.sum(weights * y) / w_sum

    cov_wxy = np.sum(weights * (x - wx_mean) * (y - wy_mean))
    var_wx = np.sum(weights * (x - wx_mean) ** 2)

    b = cov_wxy / var_wx
    a = wy_mean - b * wx_mean

    # Recover OU parameters
    if 0 < b < 1:
        kappa = -np.log(b) / dt
        theta = a / (1 - b)
    else:
        # More conservative fallback using long-run mean
        theta = np.median(prices)  # Median is robust to spikes
        kappa = 0.5  # Reasonable default

    # Estimate sigma from weighted residuals
    residuals = y - (a + b * x)
    weighted_var = np.sum(weights * residuals**2) / w_sum

    if kappa > 0 and kappa < 10:  # Sanity check
        sigma = np.sqrt(weighted_var * 2 * kappa / (1 - np.exp(-2 * kappa * dt)))
    else:
        # Use IQR-based volatility estimate (robust)
        iqr = np.percentile(prices, 75) - np.percentile(prices, 25)
        sigma = iqr / 1.35  # IQR to std approximation
        kappa = 0.5
        theta = np.median(prices)

    # Cap parameters to reasonable ranges
    kappa = min(max(kappa, 0.1), 5.0)
    sigma = min(max(sigma, 1.0), 50.0)

    # R-squared on non-outlier points
    mask = z_scores <= outlier_threshold
    if np.sum(mask) > 10:
        y_clean = y[mask]
        pred_clean = a + b * x[mask]
        ss_res = np.sum((y_clean - pred_clean) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = 0.0

    n_outliers = np.sum(z_scores > outlier_threshold)

    process = OUProcess(
        theta=theta,
        kappa=kappa,
        sigma=sigma,
        x0=prices[-1],
    )

    return CalibrationResult(
        process=process,
        n_observations=n,
        r_squared=r_squared,
        log_likelihood=-0.5 * n * np.log(weighted_var),  # Approximate
    )


def validate_calibration(
    true_params: dict,
    calibrated: CalibrationResult,
) -> dict:
    """Compare calibrated parameters to true values.

    Args:
        true_params: Dict with 'theta', 'kappa', 'sigma'
        calibrated: CalibrationResult to validate

    Returns:
        Dict with parameter errors
    """
    if not isinstance(calibrated.process, OUProcess):
        raise ValueError("Only OU process validation supported")

    p = calibrated.process

    return {
        "theta_error": (p.theta - true_params["theta"]) / true_params["theta"],
        "kappa_error": (p.kappa - true_params["kappa"]) / true_params["kappa"],
        "sigma_error": (p.sigma - true_params["sigma"]) / true_params["sigma"],
        "theta_in_ci": (
            calibrated.theta_ci is not None and
            calibrated.theta_ci[0] <= true_params["theta"] <= calibrated.theta_ci[1]
        ),
        "kappa_in_ci": (
            calibrated.kappa_ci is not None and
            calibrated.kappa_ci[0] <= true_params["kappa"] <= calibrated.kappa_ci[1]
        ),
        "sigma_in_ci": (
            calibrated.sigma_ci is not None and
            calibrated.sigma_ci[0] <= true_params["sigma"] <= calibrated.sigma_ci[1]
        ),
    }

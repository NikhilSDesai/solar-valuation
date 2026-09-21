"""Data schemas for market and weather data."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PriceSource(str, Enum):
    """Data source identifiers."""

    NGESO = "ngeso"  # UK National Grid ESO
    EIA = "eia"  # US Energy Information Administration
    ENTSOE = "entsoe"  # European ENTSO-E
    NORDPOOL = "nordpool"  # Nordic power exchange


class CarbonScheme(str, Enum):
    """Carbon trading scheme identifiers."""

    EU_ETS = "EU_ETS"  # EU Emissions Trading System
    UK_ETS = "UK_ETS"  # UK Emissions Trading Scheme
    CA_CaT = "CA_CaT"  # California Cap-and-Trade


class CarbonSource(str, Enum):
    """Carbon price data source identifiers."""

    OILPRICE_API = "oilpriceapi"  # oilpriceapi.com
    EMBER = "ember"  # ember-climate.org
    MANUAL = "manual"  # Manual entry / CSV import


class ElectricityPrice(BaseModel):
    """Spot electricity price record."""

    timestamp: datetime
    price_mwh: float = Field(description="Price in local currency per MWh")
    currency: str = Field(default="GBP", max_length=3)
    source: PriceSource
    region: str = Field(description="Market region/zone identifier")
    settlement_period: int | None = Field(
        default=None, description="Settlement period (UK uses 30-min periods)"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SolarIrradiance(BaseModel):
    """Solar irradiance measurement for capacity factor estimation."""

    timestamp: datetime
    latitude: float
    longitude: float
    ghi_wm2: float = Field(description="Global Horizontal Irradiance (W/m²)")
    dni_wm2: float | None = Field(default=None, description="Direct Normal Irradiance (W/m²)")
    dhi_wm2: float | None = Field(default=None, description="Diffuse Horizontal Irradiance (W/m²)")
    temperature_c: float | None = Field(default=None, description="Air temperature (°C)")
    cloud_cover_pct: float | None = Field(default=None, description="Cloud cover percentage")


class CarbonPrice(BaseModel):
    """Carbon credit/allowance price."""

    timestamp: datetime
    price_tonne: float = Field(description="Price per tonne CO2 equivalent")
    currency: str = Field(default="EUR", max_length=3)
    scheme: CarbonScheme = Field(description="Trading scheme (EU_ETS, UK_ETS, etc.)")
    source: CarbonSource = Field(default=CarbonSource.OILPRICE_API)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

"""Data models for price and weather records."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Market(str, Enum):
    """Supported electricity markets."""

    UK_WHOLESALE = "uk_wholesale"
    UK_IMBALANCE = "uk_imbalance"
    NORDPOOL_DK1 = "nordpool_dk1"
    EIA_CAISO = "eia_caiso"
    EIA_ERCOT = "eia_ercot"


class PriceRecord(BaseModel):
    """Single electricity price observation."""

    timestamp: datetime
    market: Market
    price_mwh: float = Field(description="Price in local currency per MWh")
    currency: str = "GBP"
    settlement_period: int | None = Field(default=None, ge=1, le=50)
    source: str = Field(description="API source identifier")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class WeatherRecord(BaseModel):
    """Weather observation for solar irradiance modeling."""

    timestamp: datetime
    latitude: float
    longitude: float
    ghi_wm2: float = Field(description="Global horizontal irradiance W/m²")
    dni_wm2: float | None = Field(default=None, description="Direct normal irradiance")
    temperature_c: float
    cloud_cover_pct: float = Field(ge=0, le=100)
    source: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class CarbonPriceRecord(BaseModel):
    """Carbon allowance price (e.g., UK ETS, EU ETS)."""

    timestamp: datetime
    scheme: str  # "uk_ets", "eu_ets"
    price_per_tonne: float
    currency: str
    source: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

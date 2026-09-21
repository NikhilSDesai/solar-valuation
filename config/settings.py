"""Configuration for data sources and API endpoints."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # UK National Grid ESO (no auth required for basic endpoints)
    ngeso_base_url: str = "https://api.nationalgrideso.com"

    # EIA (US Energy Information Administration) - free API key
    eia_api_key: str = ""
    eia_base_url: str = "https://api.eia.gov/v2"

    # Open-Meteo (free, no auth) - for solar irradiance
    openmeteo_base_url: str = "https://api.open-meteo.com/v1"

    # ENTSO-E (European electricity) - requires free registration
    entsoe_api_key: str = ""
    entsoe_base_url: str = "https://web-api.tp.entsoe.eu/api"

    # OilPriceAPI (carbon prices) - free tier available
    # Sign up: https://www.oilpriceapi.com/auth/signup
    oilprice_api_key: str = ""
    oilprice_base_url: str = "https://api.oilpriceapi.com/v1"

    # EMBER Climate (carbon intensity) - free API key
    # Sign up: https://ember-climate.org/data/api/
    ember_api_key: str = ""
    ember_base_url: str = "https://api.ember-climate.org/v1"

    # Local storage
    database_url: str = "duckdb:///data/prices.duckdb"
    parquet_dir: str = "data/raw"

    # Polling intervals (seconds)
    price_poll_interval: int = 300  # 5 minutes
    weather_poll_interval: int = 3600  # 1 hour
    carbon_poll_interval: int = 86400  # 24 hours (daily prices)


settings = Settings()

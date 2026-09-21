"""Open-Meteo weather and solar data client.

Open-Meteo provides free access to:
- Historical weather data
- Solar irradiance (GHI, DNI, DHI)
- Weather forecasts

No API key required. Rate limit: 10,000 requests/day.
Docs: https://open-meteo.com/en/docs
"""

from datetime import datetime
from typing import Any

import structlog

from .base_client import BaseAPIClient
from .schemas import SolarIrradiance

logger = structlog.get_logger()

# Default coordinates for a UK solar farm (Oxfordshire)
DEFAULT_LAT = 51.75
DEFAULT_LON = -1.25


class OpenMeteoClient(BaseAPIClient):
    """Client for Open-Meteo weather and solar radiation data."""

    def __init__(self, base_url: str = "https://api.open-meteo.com/v1"):
        super().__init__(base_url)

    async def get_historical_solar(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
    ) -> list[SolarIrradiance]:
        """Fetch historical solar irradiance data.

        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of hourly SolarIrradiance records
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,cloud_cover",
            "timezone": "UTC",
        }

        data = await self._get("/archive", params)
        return self._parse_historical_response(data, latitude, longitude)

    async def get_forecast_solar(
        self,
        latitude: float,
        longitude: float,
        days: int = 7,
    ) -> list[SolarIrradiance]:
        """Fetch solar irradiance forecast.

        Args:
            latitude: Site latitude
            longitude: Site longitude
            days: Number of forecast days (1-16)

        Returns:
            List of hourly SolarIrradiance records
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,cloud_cover",
            "forecast_days": min(days, 16),
            "timezone": "UTC",
        }

        data = await self._get("/forecast", params)
        return self._parse_forecast_response(data, latitude, longitude)

    def _parse_historical_response(
        self, data: dict[str, Any], lat: float, lon: float
    ) -> list[SolarIrradiance]:
        """Parse Open-Meteo historical archive response."""
        return self._parse_hourly_data(data, lat, lon)

    def _parse_forecast_response(
        self, data: dict[str, Any], lat: float, lon: float
    ) -> list[SolarIrradiance]:
        """Parse Open-Meteo forecast response."""
        return self._parse_hourly_data(data, lat, lon)

    def _parse_hourly_data(
        self, data: dict[str, Any], lat: float, lon: float
    ) -> list[SolarIrradiance]:
        """Parse hourly data from Open-Meteo response."""
        records = []
        hourly = data.get("hourly", {})

        timestamps = hourly.get("time", [])
        ghi = hourly.get("shortwave_radiation", [])
        dni = hourly.get("direct_radiation", [])
        dhi = hourly.get("diffuse_radiation", [])
        temp = hourly.get("temperature_2m", [])
        cloud = hourly.get("cloud_cover", [])

        for i, ts in enumerate(timestamps):
            try:
                records.append(
                    SolarIrradiance(
                        timestamp=datetime.fromisoformat(ts),
                        latitude=lat,
                        longitude=lon,
                        ghi_wm2=ghi[i] if i < len(ghi) and ghi[i] is not None else 0.0,
                        dni_wm2=dni[i] if i < len(dni) else None,
                        dhi_wm2=dhi[i] if i < len(dhi) else None,
                        temperature_c=temp[i] if i < len(temp) else None,
                        cloud_cover_pct=cloud[i] if i < len(cloud) else None,
                    )
                )
            except (ValueError, IndexError) as e:
                logger.warning("parse_error", index=i, error=str(e))
                continue

        return records


async def fetch_site_irradiance(
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    days_back: int = 30,
) -> list[SolarIrradiance]:
    """Convenience function to fetch recent solar data for a site.

    Args:
        latitude: Site latitude (default: Oxfordshire)
        longitude: Site longitude
        days_back: Days of historical data

    Returns:
        List of hourly SolarIrradiance records
    """
    from datetime import timedelta

    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    async with OpenMeteoClient() as client:
        return await client.get_historical_solar(latitude, longitude, start, end)

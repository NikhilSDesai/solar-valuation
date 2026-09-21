"""UK National Grid ESO data client.

NGESO provides free access to GB electricity market data including:
- Day-ahead prices
- Imbalance prices (real-time balancing)
- System frequency
- Carbon intensity

Docs: https://api.nationalgrideso.com/
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from .base_client import BaseAPIClient
from .schemas import ElectricityPrice, PriceSource

logger = structlog.get_logger()


class NGESOClient(BaseAPIClient):
    """Client for UK National Grid ESO electricity data."""

    # NGESO Data Portal dataset IDs
    DATASETS = {
        "imbalance_prices": "BMRS/B1770",  # Imbalance prices
        "day_ahead": "BMRS/B1440",  # Day-ahead prices
        "carbon_intensity": "carbon-intensity",
    }

    def __init__(self, base_url: str = "https://api.nationalgrideso.com"):
        super().__init__(base_url)

    async def get_imbalance_prices(
        self,
        start: datetime,
        end: datetime | None = None,
    ) -> list[ElectricityPrice]:
        """Fetch GB imbalance prices (real-time system price).

        Args:
            start: Start datetime (UTC)
            end: End datetime (UTC), defaults to now

        Returns:
            List of ElectricityPrice records
        """
        if end is None:
            end = datetime.utcnow()

        # NGESO uses settlement dates and periods
        params = {
            "settlementDateFrom": start.strftime("%Y-%m-%d"),
            "settlementDateTo": end.strftime("%Y-%m-%d"),
            "format": "json",
        }

        data = await self._get(f"/api/data/{self.DATASETS['imbalance_prices']}", params)
        return self._parse_imbalance_response(data)

    async def get_day_ahead_prices(
        self,
        settlement_date: datetime,
    ) -> list[ElectricityPrice]:
        """Fetch day-ahead auction prices.

        Args:
            settlement_date: The settlement date to fetch prices for

        Returns:
            List of ElectricityPrice records (48 half-hour periods)
        """
        params = {
            "settlementDate": settlement_date.strftime("%Y-%m-%d"),
            "format": "json",
        }

        data = await self._get(f"/api/data/{self.DATASETS['day_ahead']}", params)
        return self._parse_day_ahead_response(data)

    def _parse_imbalance_response(self, data: dict[str, Any]) -> list[ElectricityPrice]:
        """Parse NGESO imbalance price response."""
        prices = []
        records = data.get("data", [])

        for record in records:
            try:
                # NGESO returns settlement date + period, convert to timestamp
                settlement_date = datetime.strptime(record["settlementDate"], "%Y-%m-%d")
                period = int(record["settlementPeriod"])
                # Each period is 30 minutes, period 1 starts at 00:00
                timestamp = settlement_date + timedelta(minutes=30 * (period - 1))

                prices.append(
                    ElectricityPrice(
                        timestamp=timestamp,
                        price_mwh=float(record["imbalancePriceAmount"]),
                        currency="GBP",
                        source=PriceSource.NGESO,
                        region="GB",
                        settlement_period=period,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("parse_error", record=record, error=str(e))
                continue

        return prices

    def _parse_day_ahead_response(self, data: dict[str, Any]) -> list[ElectricityPrice]:
        """Parse NGESO day-ahead price response."""
        prices = []
        records = data.get("data", [])

        for record in records:
            try:
                settlement_date = datetime.strptime(record["settlementDate"], "%Y-%m-%d")
                period = int(record["settlementPeriod"])
                timestamp = settlement_date + timedelta(minutes=30 * (period - 1))

                prices.append(
                    ElectricityPrice(
                        timestamp=timestamp,
                        price_mwh=float(record["marketPriceAmount"]),
                        currency="GBP",
                        source=PriceSource.NGESO,
                        region="GB",
                        settlement_period=period,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("parse_error", record=record, error=str(e))
                continue

        return prices


async def fetch_latest_prices(days_back: int = 7) -> list[ElectricityPrice]:
    """Convenience function to fetch recent GB prices.

    Args:
        days_back: Number of days of historical data to fetch

    Returns:
        List of ElectricityPrice records
    """
    start = datetime.utcnow() - timedelta(days=days_back)

    async with NGESOClient() as client:
        return await client.get_imbalance_prices(start=start)

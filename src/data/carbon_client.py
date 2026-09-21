"""Carbon price data clients.

Supports multiple carbon price data sources:
1. OilPriceAPI - Free tier available, real-time EU ETS prices
2. EMBER Climate - Open data API for carbon intensity (requires key)
3. Manual/CSV import for historical data

EU ETS: European Union Emissions Trading System (~€65-80/tCO2 in 2024)
UK ETS: UK Emissions Trading Scheme (~£35-50/tCO2 in 2024)
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from .base_client import BaseAPIClient
from .schemas import CarbonPrice, CarbonScheme, CarbonSource

logger = structlog.get_logger()


class OilPriceAPIClient(BaseAPIClient):
    """Client for OilPriceAPI carbon price data.

    Free tier available at: https://www.oilpriceapi.com/auth/signup
    Docs: https://docs.oilpriceapi.com/

    Provides daily EU ETS carbon allowance prices.
    """

    # Supported carbon price codes
    CARBON_CODES = {
        CarbonScheme.EU_ETS: "EU_CARBON_EUR",  # EU allowances in EUR
    }

    def __init__(self, api_key: str, base_url: str = "https://api.oilpriceapi.com/v1"):
        super().__init__(base_url)
        self.api_key = api_key

    async def __aenter__(self) -> "OilPriceAPIClient":
        await super().__aenter__()
        # Add auth header
        self.client.headers["Authorization"] = f"Token {self.api_key}"
        return self

    async def get_latest_price(
        self, scheme: CarbonScheme = CarbonScheme.EU_ETS
    ) -> CarbonPrice | None:
        """Fetch the latest carbon price.

        Args:
            scheme: Carbon trading scheme (currently only EU_ETS supported)

        Returns:
            CarbonPrice record or None if scheme not supported
        """
        code = self.CARBON_CODES.get(scheme)
        if not code:
            logger.warning("unsupported_scheme", scheme=scheme)
            return None

        data = await self._get("/prices/latest", params={"by_code": code})
        return self._parse_response(data, scheme)

    async def get_historical_prices(
        self,
        scheme: CarbonScheme = CarbonScheme.EU_ETS,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CarbonPrice]:
        """Fetch historical carbon prices.

        Note: Historical endpoint may require paid tier.

        Args:
            scheme: Carbon trading scheme
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of CarbonPrice records
        """
        code = self.CARBON_CODES.get(scheme)
        if not code:
            return []

        params: dict[str, Any] = {"by_code": code}
        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        try:
            data = await self._get("/prices/past", params=params)
            return self._parse_historical_response(data, scheme)
        except Exception as e:
            # Historical endpoint may not be available on free tier
            logger.warning("historical_fetch_failed", error=str(e))
            return []

    def _parse_response(self, data: dict[str, Any], scheme: CarbonScheme) -> CarbonPrice | None:
        """Parse single price response."""
        try:
            price_data = data.get("data", {})
            return CarbonPrice(
                timestamp=datetime.fromisoformat(
                    price_data["created_at"].replace("Z", "+00:00")
                ),
                price_tonne=float(price_data["price"]),
                currency=price_data.get("currency", "EUR"),
                scheme=scheme,
                source=CarbonSource.OILPRICE_API,
            )
        except (KeyError, ValueError) as e:
            logger.error("parse_error", error=str(e), data=data)
            return None

    def _parse_historical_response(
        self, data: dict[str, Any], scheme: CarbonScheme
    ) -> list[CarbonPrice]:
        """Parse historical prices response."""
        prices = []
        records = data.get("data", {}).get("prices", [])

        for record in records:
            try:
                prices.append(
                    CarbonPrice(
                        timestamp=datetime.fromisoformat(
                            record["created_at"].replace("Z", "+00:00")
                        ),
                        price_tonne=float(record["price"]),
                        currency=record.get("currency", "EUR"),
                        scheme=scheme,
                        source=CarbonSource.OILPRICE_API,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("parse_error", record=record, error=str(e))
                continue

        return prices


class EmberCarbonClient(BaseAPIClient):
    """Client for EMBER Climate carbon intensity data.

    API docs: https://api.ember-climate.org/docs
    Get API key: https://ember-climate.org/data/api/

    Note: EMBER API focuses on carbon intensity of electricity,
    not direct ETS prices. Useful for calculating emissions costs.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.ember-climate.org/v1"):
        super().__init__(base_url)
        self.api_key = api_key

    async def __aenter__(self) -> "EmberCarbonClient":
        await super().__aenter__()
        self.client.headers["api_key"] = self.api_key
        return self

    async def get_carbon_intensity(
        self,
        entity: str = "GBR",  # ISO country code
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch carbon intensity data for a country.

        Args:
            entity: ISO 3-letter country code (e.g., GBR, DEU, FRA)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of carbon intensity records (gCO2/kWh)
        """
        params: dict[str, Any] = {
            "entity": entity,
            "is_aggregate_entity": "false",
        }
        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        data = await self._get("/carbon-intensity/monthly", params=params)
        return data.get("data", [])


def load_historical_carbon_csv(
    filepath: str,
    scheme: CarbonScheme = CarbonScheme.EU_ETS,
    date_col: str = "date",
    price_col: str = "price",
    currency: str = "EUR",
) -> list[CarbonPrice]:
    """Load historical carbon prices from CSV file.

    Useful for importing data from sources like:
    - EMBER data downloads
    - Sandbag Carbon Price Viewer exports
    - ICE historical data

    Args:
        filepath: Path to CSV file
        scheme: Carbon trading scheme
        date_col: Name of date column
        price_col: Name of price column
        currency: Price currency

    Returns:
        List of CarbonPrice records
    """
    import polars as pl

    df = pl.read_csv(filepath)

    prices = []
    for row in df.iter_rows(named=True):
        try:
            # Handle various date formats
            date_val = row[date_col]
            if isinstance(date_val, str):
                timestamp = datetime.fromisoformat(date_val)
            else:
                timestamp = date_val

            prices.append(
                CarbonPrice(
                    timestamp=timestamp,
                    price_tonne=float(row[price_col]),
                    currency=currency,
                    scheme=scheme,
                    source=CarbonSource.MANUAL,
                )
            )
        except (KeyError, ValueError) as e:
            logger.warning("csv_parse_error", row=row, error=str(e))
            continue

    return prices


# Historical EU ETS prices for development/testing (monthly averages)
# Source: https://sandbag.be/carbon-price-viewer/
EU_ETS_HISTORICAL_2024 = [
    ("2024-01-01", 65.50),
    ("2024-02-01", 58.20),
    ("2024-03-01", 60.80),
    ("2024-04-01", 63.40),
    ("2024-05-01", 70.15),
    ("2024-06-01", 68.90),
    ("2024-07-01", 66.30),
    ("2024-08-01", 64.80),
    ("2024-09-01", 63.50),
    ("2024-10-01", 65.20),
    ("2024-11-01", 68.40),
    ("2024-12-01", 71.50),
]


def get_fallback_carbon_prices(
    scheme: CarbonScheme = CarbonScheme.EU_ETS,
    start_date: datetime | None = None,
) -> list[CarbonPrice]:
    """Get hardcoded historical prices as fallback when API unavailable.

    Args:
        scheme: Carbon trading scheme
        start_date: Optional start date filter

    Returns:
        List of monthly CarbonPrice records
    """
    if scheme != CarbonScheme.EU_ETS:
        return []

    prices = []
    for date_str, price in EU_ETS_HISTORICAL_2024:
        timestamp = datetime.fromisoformat(date_str)
        if start_date and timestamp < start_date:
            continue

        prices.append(
            CarbonPrice(
                timestamp=timestamp,
                price_tonne=price,
                currency="EUR",
                scheme=CarbonScheme.EU_ETS,
                source=CarbonSource.MANUAL,
            )
        )

    return prices


async def fetch_carbon_prices(
    api_key: str | None = None,
    scheme: CarbonScheme = CarbonScheme.EU_ETS,
    use_fallback: bool = True,
) -> list[CarbonPrice]:
    """Convenience function to fetch carbon prices.

    Tries OilPriceAPI first, falls back to hardcoded historical data.

    Args:
        api_key: OilPriceAPI key (None to skip API call)
        scheme: Carbon trading scheme
        use_fallback: Whether to use fallback data if API fails

    Returns:
        List of CarbonPrice records
    """
    prices = []

    if api_key:
        try:
            async with OilPriceAPIClient(api_key) as client:
                latest = await client.get_latest_price(scheme)
                if latest:
                    prices.append(latest)

                # Try historical (may fail on free tier)
                historical = await client.get_historical_prices(
                    scheme,
                    start_date=datetime.utcnow() - timedelta(days=365),
                )
                prices.extend(historical)

        except Exception as e:
            logger.warning("api_fetch_failed", error=str(e))

    if not prices and use_fallback:
        logger.info("using_fallback_carbon_prices")
        prices = get_fallback_carbon_prices(scheme)

    return prices

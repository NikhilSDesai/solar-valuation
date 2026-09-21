"""Data ingestion pipeline.

Orchestrates fetching data from multiple sources and storing locally.
Can be run as a one-off backfill or scheduled for continuous updates.
"""

import asyncio
from datetime import datetime, timedelta

import structlog

from .carbon_client import OilPriceAPIClient, fetch_carbon_prices, get_fallback_carbon_prices
from .ngeso_client import NGESOClient
from .openmeteo_client import OpenMeteoClient
from .schemas import CarbonScheme
from .storage import PriceStore

logger = structlog.get_logger()


class DataPipeline:
    """Orchestrates data ingestion from multiple sources."""

    def __init__(
        self,
        store: PriceStore | None = None,
        site_latitude: float = 51.75,  # Oxfordshire
        site_longitude: float = -1.25,
        oilprice_api_key: str | None = None,
    ):
        self.store = store or PriceStore()
        self.site_lat = site_latitude
        self.site_lon = site_longitude
        self.oilprice_api_key = oilprice_api_key

    async def backfill_prices(self, days: int = 30) -> dict[str, int]:
        """Backfill historical electricity prices.

        Args:
            days: Number of days to backfill

        Returns:
            Dict with counts of records inserted by source
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        results = {}

        async with NGESOClient() as client:
            logger.info("backfill_start", source="ngeso", start=start, end=end)
            prices = await client.get_imbalance_prices(start, end)
            inserted = self.store.insert_prices(prices)
            results["ngeso"] = inserted
            logger.info("backfill_complete", source="ngeso", inserted=inserted)

        return results

    async def backfill_irradiance(self, days: int = 365) -> int:
        """Backfill historical solar irradiance data.

        Args:
            days: Number of days to backfill (Open-Meteo supports up to ~2 years)

        Returns:
            Number of records inserted
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        async with OpenMeteoClient() as client:
            logger.info(
                "backfill_start",
                source="openmeteo",
                lat=self.site_lat,
                lon=self.site_lon,
                days=days,
            )

            # Open-Meteo limits historical queries, chunk if needed
            chunk_size = 90  # days per request
            all_records = []

            current_start = start
            while current_start < end:
                current_end = min(current_start + timedelta(days=chunk_size), end)
                records = await client.get_historical_solar(
                    self.site_lat, self.site_lon, current_start, current_end
                )
                all_records.extend(records)
                current_start = current_end + timedelta(days=1)

                # Be nice to the free API
                await asyncio.sleep(0.5)

            inserted = self.store.insert_irradiance(all_records)
            logger.info("backfill_complete", source="openmeteo", inserted=inserted)

        return inserted

    async def backfill_carbon(
        self,
        scheme: CarbonScheme = CarbonScheme.EU_ETS,
        use_api: bool = True,
    ) -> int:
        """Backfill historical carbon prices.

        Uses OilPriceAPI if key is configured, otherwise falls back to
        hardcoded historical data for development/testing.

        Args:
            scheme: Carbon trading scheme (EU_ETS, UK_ETS)
            use_api: Whether to try API first (requires key)

        Returns:
            Number of records inserted
        """
        logger.info("backfill_start", source="carbon", scheme=scheme.value)

        prices = []

        # Try API if key is available
        if use_api and self.oilprice_api_key:
            try:
                async with OilPriceAPIClient(self.oilprice_api_key) as client:
                    # Get latest price
                    latest = await client.get_latest_price(scheme)
                    if latest:
                        prices.append(latest)

                    # Try historical (may require paid tier)
                    historical = await client.get_historical_prices(
                        scheme,
                        start_date=datetime.utcnow() - timedelta(days=365),
                    )
                    prices.extend(historical)

            except Exception as e:
                logger.warning("carbon_api_failed", error=str(e))

        # Fall back to hardcoded data if no API results
        if not prices:
            logger.info("using_fallback_carbon_data")
            prices = get_fallback_carbon_prices(scheme)

        inserted = self.store.insert_carbon(prices)
        logger.info("backfill_complete", source="carbon", inserted=inserted)

        return inserted

    async def fetch_latest_carbon(
        self, scheme: CarbonScheme = CarbonScheme.EU_ETS
    ) -> int:
        """Fetch latest carbon price.

        Args:
            scheme: Carbon trading scheme

        Returns:
            Number of records inserted (0 or 1)
        """
        if not self.oilprice_api_key:
            logger.debug("skipping_carbon_fetch", reason="no_api_key")
            return 0

        try:
            async with OilPriceAPIClient(self.oilprice_api_key) as client:
                price = await client.get_latest_price(scheme)
                if price:
                    return self.store.insert_carbon([price])
        except Exception as e:
            logger.warning("carbon_fetch_failed", error=str(e))

        return 0

    async def fetch_latest(self) -> dict[str, int]:
        """Fetch latest data from all sources.

        Designed to be called on a schedule (e.g., every 5 minutes).

        Returns:
            Dict with counts of new records by source
        """
        results = {}

        # Fetch last 24 hours of prices (handles gaps from downtime)
        start = datetime.utcnow() - timedelta(hours=24)

        async with NGESOClient() as client:
            prices = await client.get_imbalance_prices(start)
            results["prices"] = self.store.insert_prices(prices)

        # Fetch 7-day irradiance forecast
        async with OpenMeteoClient() as client:
            forecast = await client.get_forecast_solar(
                self.site_lat, self.site_lon, days=7
            )
            results["irradiance_forecast"] = self.store.insert_irradiance(forecast)

        # Fetch latest carbon price
        results["carbon"] = await self.fetch_latest_carbon()

        return results

    async def run_full_backfill(self) -> dict[str, int]:
        """Run complete historical backfill.

        Returns:
            Summary of all records inserted
        """
        logger.info("full_backfill_start")

        price_results = await self.backfill_prices(days=30)
        irradiance_count = await self.backfill_irradiance(days=365)
        carbon_count = await self.backfill_carbon()

        results = {
            **price_results,
            "irradiance": irradiance_count,
            "carbon": carbon_count,
        }

        logger.info("full_backfill_complete", results=results)
        return results


async def main() -> None:
    """CLI entrypoint for running the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Solar valuation data pipeline")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run full historical backfill",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Fetch latest data only",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of price data to backfill",
    )

    args = parser.parse_args()

    # Load API key from environment if available
    import os
    oilprice_key = os.environ.get("OILPRICE_API_KEY", "")

    pipeline = DataPipeline(oilprice_api_key=oilprice_key or None)

    if args.stats:
        stats = pipeline.store.get_stats()
        print("\nDatabase Statistics:")
        for table, info in stats.items():
            print(f"\n  {table}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
        return

    if args.backfill:
        results = await pipeline.run_full_backfill()
        print(f"\nBackfill complete: {results}")
    elif args.latest:
        results = await pipeline.fetch_latest()
        print(f"\nFetched latest: {results}")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

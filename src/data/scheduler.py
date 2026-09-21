"""Scheduled data fetching using APScheduler.

Runs continuous data ingestion on configurable intervals.
"""

import asyncio
from datetime import datetime

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .pipeline import DataPipeline

logger = structlog.get_logger()


class DataScheduler:
    """Manages scheduled data fetching jobs."""

    def __init__(
        self,
        pipeline: DataPipeline | None = None,
        price_interval_seconds: int = 300,  # 5 minutes
        irradiance_interval_seconds: int = 3600,  # 1 hour
    ):
        self.pipeline = pipeline or DataPipeline()
        self.price_interval = price_interval_seconds
        self.irradiance_interval = irradiance_interval_seconds
        self.scheduler = AsyncIOScheduler()

    async def _fetch_prices(self) -> None:
        """Job: Fetch latest prices."""
        try:
            from .ngeso_client import NGESOClient
            from datetime import timedelta

            start = datetime.utcnow() - timedelta(hours=2)

            async with NGESOClient() as client:
                prices = await client.get_imbalance_prices(start)
                count = self.pipeline.store.insert_prices(prices)
                logger.info("scheduled_price_fetch", new_records=count)

        except Exception as e:
            logger.error("scheduled_price_fetch_failed", error=str(e))

    async def _fetch_irradiance(self) -> None:
        """Job: Fetch latest irradiance forecast."""
        try:
            from .openmeteo_client import OpenMeteoClient

            async with OpenMeteoClient() as client:
                forecast = await client.get_forecast_solar(
                    self.pipeline.site_lat,
                    self.pipeline.site_lon,
                    days=7,
                )
                count = self.pipeline.store.insert_irradiance(forecast)
                logger.info("scheduled_irradiance_fetch", new_records=count)

        except Exception as e:
            logger.error("scheduled_irradiance_fetch_failed", error=str(e))

    def start(self) -> None:
        """Start the scheduler with configured jobs."""
        # Price fetching job
        self.scheduler.add_job(
            self._fetch_prices,
            trigger=IntervalTrigger(seconds=self.price_interval),
            id="fetch_prices",
            name="Fetch electricity prices",
            replace_existing=True,
        )

        # Irradiance fetching job
        self.scheduler.add_job(
            self._fetch_irradiance,
            trigger=IntervalTrigger(seconds=self.irradiance_interval),
            id="fetch_irradiance",
            name="Fetch solar irradiance",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(
            "scheduler_started",
            price_interval=self.price_interval,
            irradiance_interval=self.irradiance_interval,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("scheduler_stopped")


async def run_scheduler() -> None:
    """Run the data scheduler continuously."""
    scheduler = DataScheduler()

    # Do an initial fetch before starting scheduled jobs
    logger.info("initial_fetch_start")
    await scheduler.pipeline.fetch_latest()

    scheduler.start()

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        scheduler.stop()


if __name__ == "__main__":
    asyncio.run(run_scheduler())

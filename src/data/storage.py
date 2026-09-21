"""Data storage layer using DuckDB and Parquet.

DuckDB provides:
- Fast analytical queries on time-series price data
- Native Parquet support for archival
- Zero-dependency embedded database
"""

from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import structlog

from .schemas import CarbonPrice, ElectricityPrice, SolarIrradiance

logger = structlog.get_logger()


class PriceStore:
    """Storage layer for electricity price data."""

    def __init__(self, db_path: str = "data/prices.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS electricity_prices (
                    timestamp TIMESTAMP NOT NULL,
                    price_mwh DOUBLE NOT NULL,
                    currency VARCHAR(3) NOT NULL,
                    source VARCHAR(20) NOT NULL,
                    region VARCHAR(50) NOT NULL,
                    settlement_period INTEGER,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, source, region)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS solar_irradiance (
                    timestamp TIMESTAMP NOT NULL,
                    latitude DOUBLE NOT NULL,
                    longitude DOUBLE NOT NULL,
                    ghi_wm2 DOUBLE NOT NULL,
                    dni_wm2 DOUBLE,
                    dhi_wm2 DOUBLE,
                    temperature_c DOUBLE,
                    cloud_cover_pct DOUBLE,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, latitude, longitude)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbon_prices (
                    timestamp TIMESTAMP NOT NULL,
                    price_tonne DOUBLE NOT NULL,
                    currency VARCHAR(3) NOT NULL,
                    scheme VARCHAR(20) NOT NULL,
                    source VARCHAR(20) NOT NULL,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, scheme)
                )
            """)

            # Create indexes for common query patterns
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_ts
                ON electricity_prices(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_irradiance_ts
                ON solar_irradiance(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_carbon_ts
                ON carbon_prices(timestamp)
            """)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        return duckdb.connect(str(self.db_path))

    def insert_prices(self, prices: list[ElectricityPrice]) -> int:
        """Insert electricity prices, ignoring duplicates.

        Args:
            prices: List of ElectricityPrice records

        Returns:
            Number of rows inserted
        """
        if not prices:
            return 0

        df = pl.DataFrame([p.model_dump() for p in prices])

        with self._get_connection() as conn:
            # Use INSERT OR IGNORE to handle duplicates
            before = conn.execute("SELECT COUNT(*) FROM electricity_prices").fetchone()[0]

            conn.execute("""
                INSERT OR IGNORE INTO electricity_prices
                (timestamp, price_mwh, currency, source, region, settlement_period)
                SELECT timestamp, price_mwh, currency, source, region, settlement_period
                FROM df
            """)

            after = conn.execute("SELECT COUNT(*) FROM electricity_prices").fetchone()[0]
            inserted = after - before

        logger.info("prices_inserted", count=inserted, total_provided=len(prices))
        return inserted

    def insert_irradiance(self, records: list[SolarIrradiance]) -> int:
        """Insert solar irradiance records, ignoring duplicates.

        Args:
            records: List of SolarIrradiance records

        Returns:
            Number of rows inserted
        """
        if not records:
            return 0

        df = pl.DataFrame([r.model_dump() for r in records])

        with self._get_connection() as conn:
            before = conn.execute("SELECT COUNT(*) FROM solar_irradiance").fetchone()[0]

            conn.execute("""
                INSERT OR IGNORE INTO solar_irradiance
                (timestamp, latitude, longitude, ghi_wm2, dni_wm2, dhi_wm2,
                 temperature_c, cloud_cover_pct)
                SELECT timestamp, latitude, longitude, ghi_wm2, dni_wm2, dhi_wm2,
                       temperature_c, cloud_cover_pct
                FROM df
            """)

            after = conn.execute("SELECT COUNT(*) FROM solar_irradiance").fetchone()[0]
            inserted = after - before

        logger.info("irradiance_inserted", count=inserted, total_provided=len(records))
        return inserted

    def insert_carbon(self, prices: list[CarbonPrice]) -> int:
        """Insert carbon prices, ignoring duplicates.

        Args:
            prices: List of CarbonPrice records

        Returns:
            Number of rows inserted
        """
        if not prices:
            return 0

        df = pl.DataFrame([p.model_dump() for p in prices])

        with self._get_connection() as conn:
            before = conn.execute("SELECT COUNT(*) FROM carbon_prices").fetchone()[0]

            conn.execute("""
                INSERT OR IGNORE INTO carbon_prices
                (timestamp, price_tonne, currency, scheme, source)
                SELECT timestamp, price_tonne, currency, scheme, source
                FROM df
            """)

            after = conn.execute("SELECT COUNT(*) FROM carbon_prices").fetchone()[0]
            inserted = after - before

        logger.info("carbon_inserted", count=inserted, total_provided=len(prices))
        return inserted

    def get_prices(
        self,
        start: datetime,
        end: datetime,
        source: str | None = None,
        region: str | None = None,
    ) -> pl.DataFrame:
        """Query electricity prices.

        Args:
            start: Start timestamp
            end: End timestamp
            source: Optional source filter
            region: Optional region filter

        Returns:
            Polars DataFrame with price data
        """
        query = """
            SELECT timestamp, price_mwh, currency, source, region, settlement_period
            FROM electricity_prices
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params = [start, end]

        if source:
            query += " AND source = ?"
            params.append(source)
        if region:
            query += " AND region = ?"
            params.append(region)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            return conn.execute(query, params).pl()

    def get_irradiance(
        self,
        start: datetime,
        end: datetime,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> pl.DataFrame:
        """Query solar irradiance data.

        Args:
            start: Start timestamp
            end: End timestamp
            latitude: Optional latitude filter
            longitude: Optional longitude filter

        Returns:
            Polars DataFrame with irradiance data
        """
        query = """
            SELECT timestamp, latitude, longitude, ghi_wm2, dni_wm2, dhi_wm2,
                   temperature_c, cloud_cover_pct
            FROM solar_irradiance
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: list[datetime | float] = [start, end]

        if latitude is not None:
            query += " AND ABS(latitude - ?) < 0.01"
            params.append(latitude)
        if longitude is not None:
            query += " AND ABS(longitude - ?) < 0.01"
            params.append(longitude)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            return conn.execute(query, params).pl()

    def get_carbon(
        self,
        start: datetime,
        end: datetime,
        scheme: str | None = None,
    ) -> pl.DataFrame:
        """Query carbon prices.

        Args:
            start: Start timestamp
            end: End timestamp
            scheme: Optional scheme filter (EU_ETS, UK_ETS, etc.)

        Returns:
            Polars DataFrame with carbon price data
        """
        query = """
            SELECT timestamp, price_tonne, currency, scheme, source
            FROM carbon_prices
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: list[datetime | str] = [start, end]

        if scheme:
            query += " AND scheme = ?"
            params.append(scheme)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            return conn.execute(query, params).pl()

    def get_latest_carbon(self, scheme: str = "EU_ETS") -> float | None:
        """Get the most recent carbon price for a scheme.

        Args:
            scheme: Carbon trading scheme

        Returns:
            Latest price per tonne, or None if no data
        """
        with self._get_connection() as conn:
            result = conn.execute(
                """
                SELECT price_tonne
                FROM carbon_prices
                WHERE scheme = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                [scheme],
            ).fetchone()

        return result[0] if result else None

    def export_to_parquet(self, table: str, path: str) -> None:
        """Export table to Parquet file for archival.

        Args:
            table: Table name ('electricity_prices' or 'solar_irradiance')
            path: Output Parquet file path
        """
        with self._get_connection() as conn:
            conn.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        logger.info("parquet_export", table=table, path=path)

    def get_stats(self) -> dict[str, dict[str, int | str | None]]:
        """Get summary statistics for stored data."""
        with self._get_connection() as conn:
            prices_stats = conn.execute("""
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM electricity_prices
            """).fetchone()

            irradiance_stats = conn.execute("""
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM solar_irradiance
            """).fetchone()

            carbon_stats = conn.execute("""
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM carbon_prices
            """).fetchone()

        return {
            "electricity_prices": {
                "count": prices_stats[0],
                "min_timestamp": str(prices_stats[1]) if prices_stats[1] else None,
                "max_timestamp": str(prices_stats[2]) if prices_stats[2] else None,
            },
            "solar_irradiance": {
                "count": irradiance_stats[0],
                "min_timestamp": str(irradiance_stats[1]) if irradiance_stats[1] else None,
                "max_timestamp": str(irradiance_stats[2]) if irradiance_stats[2] else None,
            },
            "carbon_prices": {
                "count": carbon_stats[0],
                "min_timestamp": str(carbon_stats[1]) if carbon_stats[1] else None,
                "max_timestamp": str(carbon_stats[2]) if carbon_stats[2] else None,
            },
        }

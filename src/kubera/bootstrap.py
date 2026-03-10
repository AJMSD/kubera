"""Bootstrap the local Kubera workspace."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from kubera.config import load_settings
from kubera.utils.calendar import build_market_calendar
from kubera.utils.logging import configure_logging
from kubera.utils.paths import PathManager
from kubera.utils.run_context import RunContext, create_run_context
from kubera.utils.serialization import write_settings_snapshot


def bootstrap() -> RunContext:
    """Create the local workspace tree and snapshot the active settings."""

    settings = load_settings()
    path_manager = PathManager(settings.paths)
    path_manager.ensure_managed_directories()

    started_at_utc = datetime.now(timezone.utc)
    run_context = create_run_context(settings, path_manager, started_at=started_at_utc)
    logger = configure_logging(run_context, settings.run.log_level)

    write_settings_snapshot(settings, run_context.config_snapshot_path)
    calendar = build_market_calendar(settings.market)
    override_exists = settings.market.local_holiday_override_path.exists()
    logger.info(
        "Kubera bootstrap ready | ticker=%s | exchange=%s | mode=%s | market_tz=%s | market_open=%s | market_close=%s | local_holiday_override=%s | trading_day_today=%s | run_id=%s",
        settings.ticker.symbol,
        settings.ticker.exchange,
        settings.run.default_prediction_mode,
        settings.market.timezone_name,
        settings.market.market_open.isoformat(timespec="minutes"),
        settings.market.market_close.isoformat(timespec="minutes"),
        "loaded" if override_exists else "not_found",
        calendar.is_trading_day(started_at_utc.astimezone(calendar.timezone).date()),
        run_context.run_id,
    )

    return run_context


def main() -> int:
    """Run the bootstrap entrypoint."""

    bootstrap()
    logging.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

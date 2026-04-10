"""Terminal dashboard views for Kubera pilot runs and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kubera.config import AppSettings, resolve_runtime_settings
from kubera.pilot.live_pilot import load_pilot_log_frame
from kubera.utils.paths import PathManager
from kubera.utils.user_failure import describe_partial_failure_paths, describe_pilot_stage_failure


DEFAULT_DASH_LIMIT = 20
SUPPORTED_DASHBOARD_VIEWS = ("latest", "all", "run")


def export_dashboard_html(
    settings: AppSettings,
    output_path: Path,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    view: str = "latest",
    prediction_key: str | None = None,
    prediction_date: str | None = None,
    prediction_mode: str | None = None,
    limit: int = DEFAULT_DASH_LIMIT,
    console_width: int = 120,
) -> None:
    """Render the same dashboard as the terminal to a static HTML file (Rich record + save_html)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    console = Console(record=True, width=console_width)
    launch_dashboard(
        settings,
        ticker=ticker,
        exchange=exchange,
        view=view,
        prediction_key=prediction_key,
        prediction_date=prediction_date,
        prediction_mode=prediction_mode,
        limit=limit,
        console_obj=console,
    )
    console.save_html(str(output_path))


def launch_dashboard(
    settings: AppSettings,
    *,
    ticker: str | None = None,
    exchange: str | None = None,
    view: str = "latest",
    prediction_key: str | None = None,
    prediction_date: str | None = None,
    prediction_mode: str | None = None,
    limit: int = DEFAULT_DASH_LIMIT,
    console_obj: Console | None = None,
) -> None:
    """Render one Rich dashboard view for the requested ticker and exchange."""

    runtime = resolve_runtime_settings(settings, ticker=ticker, exchange=exchange)
    path_manager = PathManager(runtime.paths)
    console = console_obj or Console()
    normalized_view = view.strip().lower()
    if normalized_view not in SUPPORTED_DASHBOARD_VIEWS:
        raise ValueError(f"Unsupported dashboard view: {view}")

    runs_frame = load_dashboard_run_frame(
        path_manager=path_manager,
        ticker=runtime.ticker.symbol,
        exchange=runtime.ticker.exchange,
    )
    offline_summary = load_dashboard_offline_summary(
        path_manager=path_manager,
        ticker=runtime.ticker.symbol,
        exchange=runtime.ticker.exchange,
    )

    title = (
        f"Kubera Dashboard | {runtime.ticker.symbol} | {runtime.ticker.exchange} | "
        f"view={normalized_view}"
    )
    console.print(Panel(Text(title, style="bold"), box=box.ROUNDED))

    if runs_frame.empty:
        console.print(
            Panel(
                "No pilot runs are stored for this ticker and exchange yet.",
                title="Runs",
                box=box.ROUNDED,
            )
        )
        return

    if normalized_view == "all":
        render_all_runs_view(console=console, runs_frame=runs_frame, limit=limit)
        return

    if normalized_view == "run":
        selected_row = resolve_dashboard_target_row(
            runs_frame=runs_frame,
            prediction_key=prediction_key,
            prediction_date=prediction_date,
            prediction_mode=prediction_mode,
        )
        if selected_row is None:
            raise ValueError("No matching stored run was found for the requested dashboard filter.")
        render_run_detail_view(console=console, row=selected_row)
        return

    render_latest_view(
        console=console,
        runs_frame=runs_frame,
        offline_summary=offline_summary,
    )


def load_dashboard_run_frame(
    *,
    path_manager: PathManager,
    ticker: str,
    exchange: str,
) -> pd.DataFrame:
    """Load all stored pilot rows for one ticker and exchange."""

    frames: list[pd.DataFrame] = []
    for log_path in path_manager.list_existing_pilot_log_paths(ticker, exchange):
        frame = load_pilot_log_frame(log_path)
        if frame.empty:
            continue
        frames.append(frame.copy())

    if not frames:
        return pd.DataFrame(dtype=object)

    combined = pd.concat(frames, ignore_index=True)
    combined["pilot_timestamp_sort_key"] = pd.to_datetime(
        combined.get("pilot_timestamp_utc"),
        errors="coerce",
        utc=True,
    )
    combined["prediction_date_sort_key"] = pd.to_datetime(
        combined.get("prediction_date"),
        errors="coerce",
    )
    combined = combined.sort_values(
        by=["pilot_timestamp_sort_key", "prediction_date_sort_key", "pilot_entry_id"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return combined


def load_dashboard_offline_summary(
    *,
    path_manager: PathManager,
    ticker: str,
    exchange: str,
) -> dict[str, Any]:
    """Load the saved offline-evaluation summary JSON when it exists."""

    summary_path = path_manager.build_offline_evaluation_summary_json_path(ticker, exchange)
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def render_latest_view(
    *,
    console: Console,
    runs_frame: pd.DataFrame,
    offline_summary: dict[str, Any],
) -> None:
    """Render the operator-focused latest view."""

    latest_table = Table(box=box.SIMPLE_HEAVY, title="Latest Stored Runs", expand=True)
    latest_table.add_column("Mode")
    latest_table.add_column("Prediction Date")
    latest_table.add_column("Status")
    latest_table.add_column("Action")
    latest_table.add_column("Cal Prob", justify="right")
    latest_table.add_column("Raw Prob", justify="right")
    latest_table.add_column("Quality")
    latest_table.add_column("News State")
    latest_table.add_column("Actual")

    latest_rows = (
        runs_frame.sort_values("pilot_timestamp_sort_key", ascending=False)
        .drop_duplicates(subset=["prediction_mode"], keep="first")
        .sort_values("prediction_mode")
    )
    for _, row in latest_rows.iterrows():
        latest_table.add_row(
            _clean_string(row.get("prediction_mode")) or "-",
            _clean_string(row.get("prediction_date")) or "-",
            _render_status(_clean_string(row.get("status"))),
            _render_action(row),
            _format_float(
                _resolve_probability(
                    row,
                    "blended_calibrated_predicted_probability_up",
                    "blended_predicted_probability_up",
                )
            ),
            _format_float(
                _resolve_probability(
                    row,
                    "blended_raw_predicted_probability_up",
                    "blended_predicted_probability_up",
                )
            ),
            _format_quality(row),
            _clean_string(row.get("news_signal_state")) or "-",
            _format_actual_outcome(row),
        )
    console.print(latest_table)

    calibration_table = Table(box=box.SIMPLE, title="Offline Calibration", expand=True)
    calibration_table.add_column("Mode")
    calibration_table.add_column("Coverage", justify="right")
    calibration_table.add_column("Raw Brier", justify="right")
    calibration_table.add_column("Cal Brier", justify="right")
    calibration_table.add_column("Log Loss", justify="right")

    mode_summaries = (offline_summary.get("mode_summaries") or {}) if offline_summary else {}
    if mode_summaries:
        for prediction_mode, mode_summary in mode_summaries.items():
            blended_metrics = (
                ((mode_summary.get("metrics_by_subset") or {}).get("all_rows") or {}).get(
                    "blended_enhanced"
                )
                or {}
            )
            calibration_table.add_row(
                prediction_mode,
                _format_float(blended_metrics.get("selective_coverage")),
                _format_float(blended_metrics.get("raw_brier_score")),
                _format_float(blended_metrics.get("brier_score")),
                _format_float(blended_metrics.get("log_loss")),
            )
    else:
        calibration_table.add_row("-", "-", "-", "-", "-")
    console.print(calibration_table)

    latest_row = latest_rows.iloc[0].to_dict() if not latest_rows.empty else None
    if latest_row is not None:
        render_run_detail_view(console=console, row=latest_row, title="Latest Run Detail")


def render_all_runs_view(
    *,
    console: Console,
    runs_frame: pd.DataFrame,
    limit: int,
) -> None:
    """Render the stored-run browser view."""

    table = Table(box=box.SIMPLE_HEAVY, title="Stored Runs", expand=True)
    table.add_column("Prediction Key")
    table.add_column("Mode")
    table.add_column("Prediction Date")
    table.add_column("Status")
    table.add_column("Action")
    table.add_column("Quality")
    table.add_column("News State")
    table.add_column("Cal Prob", justify="right")
    table.add_column("Actual")

    display_frame = runs_frame.head(max(1, limit)).copy()
    for _, row in display_frame.iterrows():
        table.add_row(
            _clean_string(row.get("prediction_key")) or "-",
            _clean_string(row.get("prediction_mode")) or "-",
            _clean_string(row.get("prediction_date")) or "-",
            _render_status(_clean_string(row.get("status"))),
            _render_action(row),
            _format_quality(row),
            _clean_string(row.get("news_signal_state")) or "-",
            _format_float(
                _resolve_probability(
                    row,
                    "blended_calibrated_predicted_probability_up",
                    "blended_predicted_probability_up",
                )
            ),
            _format_actual_outcome(row),
        )
    console.print(table)


def render_run_detail_view(
    *,
    console: Console,
    row: dict[str, Any] | pd.Series,
    title: str = "Run Detail",
) -> None:
    """Render the drilldown view for one stored pilot run."""

    row_mapping = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    snapshot_payload = load_dashboard_snapshot(row_mapping)
    summary = (snapshot_payload.get("summary_context") or {}) if snapshot_payload else {}

    detail_table = Table(box=box.SIMPLE_HEAVY, title=title, expand=True)
    detail_table.add_column("Field")
    detail_table.add_column("Value")
    detail_table.add_row("Prediction key", _clean_string(row_mapping.get("prediction_key")) or "-")
    detail_table.add_row("Mode", _clean_string(row_mapping.get("prediction_mode")) or "-")
    detail_table.add_row(
        "Market session date",
        _clean_string(summary.get("market_session_date"))
        or _clean_string(row_mapping.get("market_session_date"))
        or "-",
    )
    detail_table.add_row(
        "Historical cutoff date",
        _clean_string(summary.get("historical_cutoff_date"))
        or _clean_string(row_mapping.get("historical_cutoff_date"))
        or "-",
    )
    detail_table.add_row("Prediction date", _clean_string(row_mapping.get("prediction_date")) or "-")
    detail_table.add_row(
        "Window resolution",
        _clean_string(summary.get("window_resolution_kind")) or "-",
    )
    detail_table.add_row(
        "Resolution reason",
        _clean_string(summary.get("window_resolution_reason")) or "-",
    )
    detail_table.add_row("Status", _clean_string(row_mapping.get("status")) or "-")
    detail_table.add_row("Action", _resolve_action_label(row_mapping))
    detail_table.add_row("News state", _clean_string(row_mapping.get("news_signal_state")) or "-")
    detail_table.add_row("Data quality", _format_quality(row_mapping))
    detail_table.add_row(
        "Blended calibrated probability",
        _format_float(
            _resolve_probability(
                row_mapping,
                "blended_calibrated_predicted_probability_up",
                "blended_predicted_probability_up",
            )
        ),
    )
    detail_table.add_row(
        "Blended raw probability",
        _format_float(
            _resolve_probability(
                row_mapping,
                "blended_raw_predicted_probability_up",
                "blended_predicted_probability_up",
            )
        ),
    )
    detail_table.add_row(
        "Selective margin",
        _format_float(_safe_float(row_mapping.get("selective_probability_margin"))),
    )
    detail_table.add_row(
        "Required margin",
        _format_float(_safe_float(row_mapping.get("selective_required_margin"))),
    )
    detail_table.add_row("Actual outcome", _format_actual_outcome(row_mapping))
    detail_table.add_row(
        "Pending correctness",
        "yes" if _clean_string(row_mapping.get("actual_outcome_status")) == "pending" else "no",
    )
    detail_table.add_row(
        "Warnings",
        ", ".join(_decode_json_cell(row_mapping.get("warning_codes_json"), default=[])) or "-",
    )
    detail_table.add_row(
        "Abstain reasons",
        ", ".join(_decode_json_cell(row_mapping.get("abstain_reason_codes_json"), default=[]))
        or "-",
    )
    status_raw = (_clean_string(row_mapping.get("status")) or "").strip().lower()
    if status_raw in ("failure", "partial_failure"):
        summary_ctx = (snapshot_payload or {}).get("summary_context") if snapshot_payload else None
        if not isinstance(summary_ctx, dict):
            summary_ctx = {}
        fs = _clean_string(row_mapping.get("failure_stage"))
        fm = _clean_string(row_mapping.get("failure_message"))
        pub = summary_ctx.get("failure_reason_public")
        nxt = summary_ctx.get("failure_next_step")
        if not pub or not nxt:
            pub, nxt = describe_pilot_stage_failure(fs, fm)
        detail_table.add_row("Failure stage", fs or "-")
        detail_table.add_row("What happened", pub or "-")
        detail_table.add_row("Suggested next step", nxt or "-")
        if status_raw == "partial_failure":
            detail_table.add_row("Path status", describe_partial_failure_paths(fs))
        if fm and fm.lower() not in ("n/a", "none", "nan"):
            detail_table.add_row("Details (sanitized)", fm)
    console.print(detail_table)

    driver_table = Table(box=box.SIMPLE, title="Top Drivers", expand=True)
    driver_table.add_column("Side")
    driver_table.add_column("Feature")
    driver_table.add_column("Impact", justify="right")
    for side, feature_name, impact in extract_top_driver_rows(
        row_mapping=row_mapping,
        snapshot_payload=snapshot_payload,
        limit=4,
    ):
        driver_table.add_row(side, feature_name, f"{impact:+.4f}")
    if driver_table.row_count == 0:
        driver_table.add_row("-", "No explanation payload saved", "-")
    console.print(driver_table)

    recent_news = (
        ((summary.get("news_context") or {}).get("recent_news_summaries") or [])
        if summary
        else []
    )
    news_table = Table(box=box.SIMPLE, title="Recent Linked News", expand=True)
    news_table.add_column("Article")
    news_table.add_column("Sentiment")
    news_table.add_column("Relevance", justify="right")
    if recent_news:
        for item in recent_news[:5]:
            news_table.add_row(
                _clean_string(item.get("article_title")) or "-",
                _clean_string(item.get("sentiment_label")) or "-",
                _format_float(_safe_float(item.get("relevance_score"))),
            )
    else:
        news_table.add_row("-", "-", "-")
    console.print(news_table)


def resolve_dashboard_target_row(
    *,
    runs_frame: pd.DataFrame,
    prediction_key: str | None,
    prediction_date: str | None,
    prediction_mode: str | None,
) -> dict[str, Any] | None:
    """Resolve one selected run from the combined pilot-log frame."""

    candidate_frame = runs_frame.copy()
    if prediction_key is not None:
        candidate_frame = candidate_frame.loc[
            candidate_frame["prediction_key"].astype(str) == prediction_key
        ].copy()
    if prediction_date is not None:
        candidate_frame = candidate_frame.loc[
            candidate_frame["prediction_date"].astype(str) == prediction_date
        ].copy()
    if prediction_mode is not None:
        candidate_frame = candidate_frame.loc[
            candidate_frame["prediction_mode"].astype(str) == prediction_mode
        ].copy()
    if candidate_frame.empty:
        return None
    return candidate_frame.iloc[0].to_dict()


def load_dashboard_snapshot(row_mapping: dict[str, Any]) -> dict[str, Any] | None:
    """Load the saved pilot snapshot for one row when it exists."""

    snapshot_path_value = _clean_string(row_mapping.get("pilot_snapshot_path"))
    if snapshot_path_value is None:
        return None
    snapshot_path = Path(snapshot_path_value)
    if not snapshot_path.exists():
        return None
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_top_driver_rows(
    *,
    row_mapping: dict[str, Any],
    snapshot_payload: dict[str, Any] | None,
    limit: int,
) -> list[tuple[str, str, float]]:
    """Return top positive and negative driver rows from one saved SHAP payload."""

    shap_values = _extract_shap_values(row_mapping=row_mapping, snapshot_payload=snapshot_payload)
    if not shap_values:
        return []

    positive = sorted(
        [(feature_name, value) for feature_name, value in shap_values.items() if value > 0],
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]
    negative = sorted(
        [(feature_name, value) for feature_name, value in shap_values.items() if value < 0],
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]

    rows: list[tuple[str, str, float]] = []
    for feature_name, value in positive:
        rows.append(("positive", feature_name, float(value)))
    for feature_name, value in negative:
        rows.append(("negative", feature_name, float(value)))
    return rows


def _extract_shap_values(
    *,
    row_mapping: dict[str, Any],
    snapshot_payload: dict[str, Any] | None,
) -> dict[str, float]:
    if snapshot_payload is not None:
        summary = snapshot_payload.get("summary_context") or {}
        enhanced_prediction = summary.get("enhanced_prediction") or {}
        feature_contributions = enhanced_prediction.get("feature_contributions") or {}
        shap_values = feature_contributions.get("shap_values") if isinstance(feature_contributions, dict) else None
        if isinstance(shap_values, dict):
            return {
                str(feature_name): float(value)
                for feature_name, value in shap_values.items()
                if _safe_float(value) is not None
            }

    raw_contributions = _clean_string(row_mapping.get("enhanced_feature_contributions_json"))
    if raw_contributions is None:
        return {}
    try:
        payload = json.loads(raw_contributions)
    except json.JSONDecodeError:
        return {}
    shap_values = payload.get("shap_values") if isinstance(payload, dict) else None
    if not isinstance(shap_values, dict):
        return {}
    return {
        str(feature_name): float(value)
        for feature_name, value in shap_values.items()
        if _safe_float(value) is not None
    }


def _render_status(status: str | None) -> Text:
    normalized = (status or "-").strip().lower()
    if normalized == "success":
        return Text(status or "-", style="bold green")
    if normalized == "abstain":
        return Text(status or "-", style="bold yellow")
    if normalized == "partial_failure":
        return Text(status or "-", style="bold magenta")
    if normalized == "failure":
        return Text(status or "-", style="bold red")
    return Text(status or "-")


def _render_action(row_mapping: pd.Series | dict[str, Any]) -> Text:
    label = _resolve_action_label(row_mapping.to_dict() if isinstance(row_mapping, pd.Series) else row_mapping)
    if label == "up":
        return Text("up", style="bold green")
    if label == "down":
        return Text("down", style="bold red")
    if label == "abstain":
        return Text("abstain", style="bold yellow")
    return Text(label)


def _resolve_action_label(row_mapping: dict[str, Any]) -> str:
    selected_action = _clean_string(row_mapping.get("selected_action"))
    if selected_action is not None:
        return selected_action
    if _clean_string(row_mapping.get("status")) == "abstain":
        return "abstain"
    blended_direction = _safe_int(row_mapping.get("blended_predicted_next_day_direction"))
    if blended_direction is None:
        return "-"
    return "up" if blended_direction == 1 else "down"


def _format_quality(row_mapping: pd.Series | dict[str, Any]) -> str:
    mapping = row_mapping.to_dict() if isinstance(row_mapping, pd.Series) else row_mapping
    grade = _clean_string(mapping.get("data_quality_grade")) or "-"
    score = _safe_float(mapping.get("data_quality_score"))
    if score is None:
        return grade
    return f"{grade} ({score:.1f})"


def _format_actual_outcome(row_mapping: pd.Series | dict[str, Any]) -> str:
    mapping = row_mapping.to_dict() if isinstance(row_mapping, pd.Series) else row_mapping
    actual_status = _clean_string(mapping.get("actual_outcome_status"))
    actual_direction = _safe_int(mapping.get("actual_next_day_direction"))
    if actual_status == "backfilled" and actual_direction is not None:
        return "up" if actual_direction == 1 else "down"
    if actual_status is None:
        return "-"
    return actual_status


def _resolve_probability(
    row_mapping: pd.Series | dict[str, Any],
    *keys: str,
) -> float | None:
    mapping = row_mapping.to_dict() if isinstance(row_mapping, pd.Series) else row_mapping
    for key in keys:
        value = _safe_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _decode_json_cell(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"

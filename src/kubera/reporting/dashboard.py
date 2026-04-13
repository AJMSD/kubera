"""Terminal dashboard views for Kubera pilot runs and evaluation artifacts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
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
LINKED_NEWS_DASHBOARD_LIMIT = 3
SUPPORTED_DASHBOARD_VIEWS = ("latest", "all", "run")
HTML_EXPORT_CONSOLE_WIDTH = 180
RESPONSIVE_DASHBOARD_STYLE_MARKER = "kubera-dashboard-responsive"
RESPONSIVE_VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
RESPONSIVE_DASHBOARD_STYLE_BLOCK = f"""
<style id="{RESPONSIVE_DASHBOARD_STYLE_MARKER}">
html, body {{
    width: 100%;
    max-width: 100%;
}}
body {{
    margin: 0;
    padding: 0 6px;
}}
pre {{
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    box-sizing: border-box;
    margin: 0;
    padding-bottom: 0.25rem;
    display: block;
}}
pre code {{
    white-space: pre;
}}
@media (max-width: 900px) {{
    body {{
        padding: 0 4px;
    }}
    pre {{
        font-size: 0.88rem;
    }}
}}
</style>
""".strip()
DRIVER_IMPACT_LEGEND_LINES = (
    "Positive impact = pushes model toward UP probability.",
    "Negative impact = pushes toward DOWN probability.",
    "Impact magnitude = relative contribution for this prediction.",
)
DRIVER_HEURISTIC_DISCLAIMER = (
    "Heuristic explanation for operator interpretation; not a causal statement."
)
FEATURE_DESCRIPTION_BY_NAME: dict[str, str] = {
    "ret_1d": "One-day return, representing immediate price momentum.",
    "ret_3d": "Three-day return, representing short-term momentum trend.",
    "ret_5d_lag2": "Lagged five-day return from recent sessions to capture persistent momentum.",
    "volume_change_1d": "One-day volume change, indicating unusual trading participation.",
    "volume_change_1d_lag2": "Lagged one-day volume change that captures delayed participation shifts.",
    "rsi_14": "14-period RSI oscillator where higher values suggest overbought momentum.",
    "stoch_k": "Stochastic %K (0-100) showing where price sits within its recent high-low range.",
    "macd": "MACD spread between short and long moving averages, indicating trend momentum.",
    "news_weighted_sentiment_score": "Weighted aggregate news sentiment mapped to this prediction window.",
}
FEATURE_PREFIX_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("ret_", "Recent return-derived momentum signal."),
    ("volume_change_", "Change in traded volume versus a prior reference period."),
    ("stoch_", "Stochastic oscillator signal derived from recent trading range."),
    ("rsi_", "Relative Strength Index momentum indicator."),
    ("macd", "MACD trend and momentum indicator."),
    ("volatility_", "Volatility feature capturing recent price variation intensity."),
    ("news_", "News-derived signal aggregated for the active prediction window."),
)
ENABLE_LLM_DRIVER_ENRICHMENT_ENV = "KUBERA_DASH_EXPLAIN_DRIVERS_LLM"
LLM_DRIVER_ENRICHMENT_MAX_FEATURES = 4
MODE_FRIENDLY_LABELS = {
    "after_close": "Next-day forecast (made after prior close)",
    "pre_market": "Same-day forecast (made before open)",
}
MODE_LEGEND_NOTE = (
    "Decision session = when the forecast was made; Target date = date being predicted."
)


def _parse_utc_timestamp_series(values: pd.Series | Any) -> pd.Series:
    """Parse UTC timestamps while tolerating mixed ISO precision formats."""

    timestamp_series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=object)
    try:
        parsed = pd.to_datetime(
            timestamp_series,
            errors="coerce",
            utc=True,
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(
            timestamp_series,
            errors="coerce",
            utc=True,
        )

    if not parsed.isna().any():
        return parsed

    for index, raw_value in timestamp_series[parsed.isna()].items():
        if pd.isna(raw_value):
            continue
        raw_text = str(raw_value).strip()
        if not raw_text:
            continue
        if raw_text.endswith("Z"):
            raw_text = f"{raw_text[:-1]}+00:00"
        try:
            parsed.loc[index] = datetime.fromisoformat(raw_text)
        except ValueError:
            continue
    return parsed


def _inject_responsive_dashboard_html(output_path: Path) -> None:
    """Inject lightweight responsive metadata and CSS into Rich-exported HTML."""

    html = output_path.read_text(encoding="utf-8")
    updated = html

    if "name=\"viewport\"" not in updated:
        if "<head>" in updated:
            updated = updated.replace("<head>", f"<head>\n{RESPONSIVE_VIEWPORT_META}", 1)
        else:
            updated = f"{RESPONSIVE_VIEWPORT_META}\n{updated}"

    if RESPONSIVE_DASHBOARD_STYLE_MARKER not in updated:
        if "</head>" in updated:
            updated = updated.replace("</head>", f"{RESPONSIVE_DASHBOARD_STYLE_BLOCK}\n</head>", 1)
        else:
            updated = f"{RESPONSIVE_DASHBOARD_STYLE_BLOCK}\n{updated}"

    if updated != html:
        output_path.write_text(updated, encoding="utf-8")


def _normalize_feature_tokens(feature_name: str) -> list[str]:
    raw_tokens = [token for token in re.split(r"[_\W]+", feature_name.lower()) if token]
    if not raw_tokens:
        return ["feature"]
    return raw_tokens


def _resolve_feature_description(feature_name: str) -> tuple[str, bool]:
    if feature_name in FEATURE_DESCRIPTION_BY_NAME:
        return FEATURE_DESCRIPTION_BY_NAME[feature_name], False
    for prefix, description in FEATURE_PREFIX_DESCRIPTIONS:
        if feature_name.startswith(prefix):
            return description, False
    token_text = " ".join(_normalize_feature_tokens(feature_name)[:6])
    return f"Derived feature from {token_text}.", True


def _resolve_contextual_feature_effect(*, side: str, feature_name: str, impact: float) -> str:
    direction = "UP" if side == "positive" else "DOWN"
    strength = _impact_strength_label(impact)
    if feature_name.startswith("stoch_"):
        return (
            f"{direction}-leaning SHAP ({impact:+.4f}, {strength}) suggests the current stochastic reading aligns with short-term momentum in that direction."
        )
    if feature_name.startswith("rsi_"):
        return (
            f"{direction}-leaning SHAP ({impact:+.4f}, {strength}) suggests the current RSI state is interpreted as momentum support for that direction."
        )
    if feature_name.startswith("macd"):
        return (
            f"{direction}-leaning SHAP ({impact:+.4f}, {strength}) suggests MACD trend slope is reinforcing that directional view."
        )
    return (
        f"{direction}-leaning SHAP ({impact:+.4f}, {strength}) indicates this feature nudges the model toward that direction."
    )


def _impact_strength_label(impact: float) -> str:
    magnitude = abs(impact)
    if magnitude >= 0.08:
        return "strong"
    if magnitude >= 0.04:
        return "moderate"
    return "light"


def _format_human_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return value
    if isinstance(parsed, pd.DatetimeIndex):
        return value
    tz_name = ""
    if parsed.tzinfo is not None:
        tz_name = parsed.tzname() or ""
        if tz_name in {"UTC+05:30", "+05:30"}:
            tz_name = "IST"
        elif tz_name in {"UTC", "+00:00"}:
            tz_name = "UTC"
    human = parsed.strftime("%d %b %Y, %H:%M")
    return f"{human} {tz_name}".strip()


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enrich_unknown_feature_descriptions_with_llm(
    *,
    settings: AppSettings,
    unknown_features: list[str],
) -> dict[str, str]:
    if not unknown_features:
        return {}
    if not _is_truthy_env(os.environ.get(ENABLE_LLM_DRIVER_ENRICHMENT_ENV)):
        return {}
    api_key = _clean_string(settings.providers.llm_api_key)
    if api_key is None:
        return {}

    candidates = unknown_features[:LLM_DRIVER_ENRICHMENT_MAX_FEATURES]
    prompt = (
        "Write one concise plain-English dashboard description per feature.\n"
        "Each line must be exactly: feature|description\n"
        "Rules: <= 12 words, neutral wording, no markdown.\n"
        f"Features: {', '.join(candidates)}\n"
    )
    try:
        from kubera.llm.extract_news import generate_plain_text_with_tiered_models

        text, _model = generate_plain_text_with_tiered_models(
            settings=settings,
            api_key=api_key,
            prompt=prompt,
        )
    except Exception:
        return {}

    resolved: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        feature_name, description = line.split("|", 1)
        key = feature_name.strip()
        value = description.strip()
        if key in candidates and value:
            resolved[key] = value.rstrip(".") + "."
    return resolved


def _build_driver_explanation_rows(
    *,
    top_driver_rows: list[tuple[str, str, float]],
    settings: AppSettings,
) -> list[tuple[str, str, str]]:
    unknown_features: list[str] = []
    base_rows: list[tuple[str, str, str, bool]] = []
    for side, feature_name, impact in top_driver_rows:
        meaning, is_unknown = _resolve_feature_description(feature_name)
        likely_effect = _resolve_contextual_feature_effect(
            side=side,
            feature_name=feature_name,
            impact=impact,
        )
        base_rows.append((feature_name, meaning, likely_effect, is_unknown))
        if is_unknown:
            unknown_features.append(feature_name)

    llm_enriched = _enrich_unknown_feature_descriptions_with_llm(
        settings=settings,
        unknown_features=unknown_features,
    )
    rows: list[tuple[str, str, str]] = []
    for feature_name, meaning, likely_effect, is_unknown in base_rows:
        if is_unknown and feature_name in llm_enriched:
            meaning = llm_enriched[feature_name]
        rows.append((feature_name, meaning, likely_effect))
    return rows


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
    console_width: int = HTML_EXPORT_CONSOLE_WIDTH,
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
    _inject_responsive_dashboard_html(output_path)


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
        render_run_detail_view(settings=settings, console=console, row=selected_row)
        return

    render_latest_view(
        settings=settings,
        console=console,
        runs_frame=runs_frame,
        offline_summary=offline_summary,
        limit=limit,
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
    combined["pilot_timestamp_sort_key"] = _parse_utc_timestamp_series(
        combined.get("pilot_timestamp_utc")
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
    settings: AppSettings,
    console: Console,
    runs_frame: pd.DataFrame,
    offline_summary: dict[str, Any],
    limit: int,
) -> None:
    """Render the latest dashboard view for the default consumer run flow."""

    latest_table = Table(box=box.SIMPLE_HEAVY, title="Latest Stored Runs", expand=True)
    latest_table.add_column("Mode", overflow="fold")
    latest_table.add_column("Decision Session", overflow="fold")
    latest_table.add_column("Prediction Date", overflow="fold", min_width=10)
    latest_table.add_column("Status")
    latest_table.add_column("Action")
    latest_table.add_column("Blended Cal Up", justify="right")
    latest_table.add_column("Blended Raw Up", justify="right")
    latest_table.add_column("Quality")
    latest_table.add_column("News State")
    latest_table.add_column("Actual")

    latest_rows_by_timestamp = (
        runs_frame.sort_values("pilot_timestamp_sort_key", ascending=False)
        .drop_duplicates(subset=["prediction_mode"], keep="first")
    )
    latest_rows = latest_rows_by_timestamp.sort_values("prediction_mode")
    for _, row in latest_rows.iterrows():
        latest_table.add_row(
            _format_prediction_mode(_clean_string(row.get("prediction_mode"))),
            _resolve_decision_session_display(row),
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
    console.print(Panel(MODE_LEGEND_NOTE, title="Mode & Date Semantics", box=box.ROUNDED))

    history_table = Table(box=box.SIMPLE_HEAVY, title="Prediction vs Actual History", expand=True)
    history_table.add_column("Mode", overflow="fold")
    history_table.add_column("Decision Session", overflow="fold")
    history_table.add_column("Target Date", overflow="fold", min_width=10)
    history_table.add_column("Predicted Action")
    history_table.add_column("Actual Outcome")
    history_table.add_column("Correctness")
    history_table.add_column("Blended Cal Up", justify="right")
    history_table.add_column("Quality")
    history_frame = _build_unique_day_history_frame(runs_frame, limit=limit)
    for _, row in history_frame.iterrows():
        history_table.add_row(
            _format_prediction_mode(_clean_string(row.get("prediction_mode"))),
            _resolve_decision_session_display(row),
            _clean_string(row.get("prediction_date")) or "-",
            _resolve_action_label(row.to_dict()),
            _format_actual_outcome(row),
            _format_correctness(row),
            _format_float(
                _resolve_probability(
                    row,
                    "blended_calibrated_predicted_probability_up",
                    "blended_predicted_probability_up",
                )
            ),
            _format_quality(row),
        )
    console.print(history_table)

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
                _format_prediction_mode(prediction_mode),
                _format_float(blended_metrics.get("selective_coverage")),
                _format_float(blended_metrics.get("raw_brier_score")),
                _format_float(blended_metrics.get("brier_score")),
                _format_float(blended_metrics.get("log_loss")),
            )
    else:
        calibration_table.add_row("-", "-", "-", "-", "-")
    console.print(calibration_table)

    latest_row = (
        latest_rows_by_timestamp.iloc[0].to_dict() if not latest_rows_by_timestamp.empty else None
    )
    if latest_row is not None:
        render_run_detail_view(
            settings=settings,
            console=console,
            row=latest_row,
            title="Latest Run Detail",
        )


def render_all_runs_view(
    *,
    console: Console,
    runs_frame: pd.DataFrame,
    limit: int,
) -> None:
    """Render the stored-run browser view."""

    table = Table(box=box.SIMPLE_HEAVY, title="Stored Runs", expand=True)
    table.add_column("Prediction Key", overflow="fold")
    table.add_column("Mode", overflow="fold")
    table.add_column("Prediction Date", overflow="fold", min_width=10)
    table.add_column("Status")
    table.add_column("Action")
    table.add_column("Quality")
    table.add_column("News State")
    table.add_column("Blended Cal Up", justify="right")
    table.add_column("Actual")

    display_frame = runs_frame.head(max(1, limit)).copy()
    for _, row in display_frame.iterrows():
        table.add_row(
            _clean_string(row.get("prediction_key")) or "-",
            _format_prediction_mode(_clean_string(row.get("prediction_mode"))),
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
    settings: AppSettings,
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
    detail_table.add_column("Value", overflow="fold")
    detail_table.add_row("Prediction key", _clean_string(row_mapping.get("prediction_key")) or "-")
    detail_table.add_row("Status", _clean_string(row_mapping.get("status")) or "-")
    detail_table.add_row("Selected action", _resolve_action_label(row_mapping))
    detail_table.add_row(
        "Mode",
        _format_prediction_mode(_clean_string(row_mapping.get("prediction_mode"))),
    )
    detail_table.add_row(
        "Decision timestamp (market)",
        _format_human_timestamp(_clean_string(row_mapping.get("pilot_timestamp_market"))) or "-",
    )
    detail_table.add_row(
        "Decision timestamp (UTC)",
        _format_human_timestamp(_clean_string(row_mapping.get("pilot_timestamp_utc"))) or "-",
    )
    detail_table.add_row(
        "Decision session date",
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
    detail_table.add_row("News state", _clean_string(row_mapping.get("news_signal_state")) or "-")
    detail_table.add_row("Data quality", _format_quality(row_mapping))
    detail_table.add_row(
        "Selective margin",
        _format_float(_safe_float(row_mapping.get("selective_probability_margin"))),
    )
    detail_table.add_row(
        "Required margin",
        _format_float(_safe_float(row_mapping.get("selective_required_margin"))),
    )
    detail_table.add_row("Actual outcome", _format_actual_outcome(row_mapping))
    detail_table.add_row("Correctness", _format_correctness(row_mapping))
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
    console.print(Panel(MODE_LEGEND_NOTE, title="Mode & Date Semantics", box=box.ROUNDED))

    probability_table = Table(box=box.SIMPLE, title="Model Probabilities", expand=True)
    probability_table.add_column("Model")
    probability_table.add_column("Direction")
    probability_table.add_column("Raw up probability", justify="right")
    probability_table.add_column("Calibrated up probability", justify="right")
    for probability_row in _build_model_probability_rows(row_mapping):
        probability_table.add_row(*probability_row)
    console.print(probability_table)
    explanation_panel = _resolve_snapshot_explanation_panel(snapshot_payload)
    if explanation_panel is not None:
        console.print(Panel(explanation_panel[1], title=explanation_panel[0], box=box.ROUNDED))

    driver_table = Table(box=box.SIMPLE, title="Enhanced Model Top Drivers", expand=True)
    driver_table.add_column("Side")
    driver_table.add_column("Feature")
    driver_table.add_column("Impact", justify="right")
    top_driver_rows = extract_top_driver_rows(
        row_mapping=row_mapping,
        snapshot_payload=snapshot_payload,
        limit=4,
    )
    for side, feature_name, impact in top_driver_rows:
        driver_table.add_row(side, feature_name, f"{impact:+.4f}")
    if driver_table.row_count == 0:
        driver_table.add_row("-", "No explanation payload saved", "-")
    console.print(driver_table)
    if top_driver_rows:
        legend_text = "\n".join(DRIVER_IMPACT_LEGEND_LINES)
        console.print(Panel(legend_text, title="Driver Impact Legend", box=box.ROUNDED))
        explanation_table = Table(
            box=box.SIMPLE,
            title="Top Driver Explanations",
            expand=True,
        )
        explanation_table.add_column("Feature")
        explanation_table.add_column("Meaning")
        explanation_table.add_column("Likely effect in this run")
        for feature_name, meaning, likely_effect in _build_driver_explanation_rows(
            top_driver_rows=top_driver_rows,
            settings=settings,
        ):
            explanation_table.add_row(feature_name, meaning, likely_effect)
        console.print(explanation_table)
        console.print(Panel(DRIVER_HEURISTIC_DISCLAIMER, box=box.ROUNDED))

    recent_news = (
        ((summary.get("news_context") or {}).get("recent_news_summaries") or [])
        if summary
        else []
    )
    linked_news_rows = _build_linked_news_rows(
        recent_news,
        limit=LINKED_NEWS_DASHBOARD_LIMIT,
    )
    news_title = "Top linked news for this prediction window"
    if not linked_news_rows:
        console.print(
            Panel(
                "No linked news exists for this prediction window.",
                title=news_title,
                box=box.ROUNDED,
            )
        )
        return

    news_table = Table(box=box.SIMPLE, title=news_title, expand=True)
    news_table.add_column("Article")
    news_table.add_column("Source")
    news_table.add_column("Sentiment")
    news_table.add_column("Relevance", justify="right")
    news_table.add_column("Snippet")
    for linked_news_row in linked_news_rows:
        news_table.add_row(*linked_news_row)
    console.print(news_table)


def _build_model_probability_rows(row_mapping: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for model_key, label in (
        ("baseline", "Baseline"),
        ("enhanced", "Enhanced"),
        ("blended", "Blended"),
    ):
        rows.append(
            (
                label,
                _resolve_model_direction(row_mapping, model_key),
                _format_float(
                    _resolve_probability(
                        row_mapping,
                        f"{model_key}_raw_predicted_probability_up",
                        f"{model_key}_predicted_probability_up",
                    )
                ),
                _format_float(
                    _resolve_probability(
                        row_mapping,
                        f"{model_key}_calibrated_predicted_probability_up",
                        f"{model_key}_predicted_probability_up",
                    )
                ),
            )
        )
    return rows


def _build_linked_news_rows(
    recent_news: Any,
    *,
    limit: int,
) -> list[tuple[str, str, str, str, str]]:
    if not isinstance(recent_news, list):
        return []

    rows: list[tuple[str, str, str, str, str]] = []
    for item in recent_news:
        if len(rows) >= limit:
            break
        if not isinstance(item, dict):
            continue
        title = _clean_string(item.get("article_title")) or _clean_string(item.get("title"))
        source = (
            _clean_string(item.get("provider_source"))
            or _clean_string(item.get("source"))
            or _clean_string(item.get("source_domain"))
            or _clean_string(item.get("provider"))
        )
        sentiment = _clean_string(item.get("sentiment_label"))
        if sentiment is None:
            sentiment = _format_float(_safe_float(item.get("sentiment_score")))
        snippet = (
            _clean_string(item.get("summary_snippet"))
            or _clean_string(item.get("snippet"))
            or _clean_string(item.get("rationale_short"))
        )
        rows.append(
            (
                title or "-",
                source or "-",
                sentiment or "-",
                _format_float(_safe_float(item.get("relevance_score"))),
                snippet or "-",
            )
        )
    return rows


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


def _resolve_snapshot_explanation_panel(
    snapshot_payload: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Resolve a human-readable explanation panel from snapshot metadata."""

    if not isinstance(snapshot_payload, dict):
        return None
    explanation = snapshot_payload.get("explanation")
    if not isinstance(explanation, dict):
        return None
    title = _clean_string(explanation.get("headline")) or "Pilot explanation"
    text = _clean_string(explanation.get("text"))
    if text is not None:
        return title, text
    status = (_clean_string(explanation.get("status")) or "").strip().lower()
    if status in {"skipped", "unavailable", "disabled"}:
        return title, "Explanation unavailable for this run."
    return None


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


def _format_prediction_mode(mode_value: str | None) -> str:
    if mode_value is None:
        return "-"
    return MODE_FRIENDLY_LABELS.get(mode_value, mode_value)


def _resolve_decision_session_display(row_mapping: pd.Series | dict[str, Any]) -> str:
    mapping = row_mapping.to_dict() if isinstance(row_mapping, pd.Series) else row_mapping
    timestamp_market = _clean_string(mapping.get("pilot_timestamp_market"))
    if timestamp_market is not None:
        return _format_human_timestamp(timestamp_market) or timestamp_market
    market_session_date = _clean_string(mapping.get("market_session_date"))
    if market_session_date is not None:
        return market_session_date
    return "-"


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _clean_string(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def _format_correctness(row_mapping: pd.Series | dict[str, Any]) -> str:
    mapping = row_mapping.to_dict() if isinstance(row_mapping, pd.Series) else row_mapping
    for key in ("blended_correct", "enhanced_correct", "baseline_correct"):
        verdict = _safe_bool(mapping.get(key))
        if verdict is True:
            return "correct"
        if verdict is False:
            return "incorrect"
    return "pending"


def _build_unique_day_history_frame(runs_frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    """Return one history row per prediction date, preferring pre_market when both modes exist."""

    working = runs_frame.copy()
    working["mode_priority"] = working["prediction_mode"].astype(str).map(
        {"pre_market": 0, "after_close": 1}
    )
    working["mode_priority"] = working["mode_priority"].fillna(2)
    working = working.sort_values(
        by=[
            "prediction_date_sort_key",
            "mode_priority",
            "pilot_timestamp_sort_key",
            "pilot_entry_id",
        ],
        ascending=[False, True, False, False],
        na_position="last",
    )
    unique_days = working.drop_duplicates(subset=["prediction_date"], keep="first")
    return unique_days.head(max(1, limit)).copy()


def _resolve_model_direction(row_mapping: dict[str, Any], model_key: str) -> str:
    direction_value = _safe_int(row_mapping.get(f"{model_key}_predicted_next_day_direction"))
    if direction_value is None:
        return "-"
    return "up" if direction_value == 1 else "down"


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

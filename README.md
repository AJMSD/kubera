# kubera

Kubera is a local, config-driven system for predicting next-day NSE stock direction from historical market data and company news.

It is built to fetch source data, normalize it into traceable local artifacts, extract structured signals from news articles, generate model-ready features, and support fair comparison between historical-only and news-aware workflows.

## What The Project Does

- Fetches and cleans daily OHLCV market data for a configured NSE ticker.
- Builds historical technical features and next-day direction labels, including MACD, 52-week price ratios, and weekday context.
- Discovers company news across Google News RSS, NSE corporate announcements, and optional provider feeds such as Marketaux or Alpha Vantage `NEWS_SENTIMENT`, normalizes article metadata, deduplicates coverage, and fetches article text with fallback handling.
- Uses the Gemini API with Gemma to turn article text into validated structured signals such as sentiment, event type, severity, horizon, and confidence.
- Aggregates article-level signals into daily pre-market and after-close news feature snapshots with NSE-aware market-time alignment.
- Trains a baseline historical-only model plus separate pre-market and after-close enhanced models that merge historical and news features on the same target rows.
- Runs held-out offline evaluation with naive baselines, feature ablations, and mode-separated evidence summaries.
- Saves per-mode enhanced predictions, metrics, merged Stage 8 datasets, baseline-vs-enhanced comparison reports, and Stage 9 aligned offline evaluation outputs.
- Persists raw snapshots, processed tables, feature tables, metadata, logs, and run config so outputs stay reproducible and traceable.

## Project Characteristics

- Config-driven defaults for ticker, exchange, provider selection, timing rules, and output paths.
- Shared ticker resolution from built-in catalog defaults plus optional local catalog overrides through `KUBERA_TICKER_CATALOG_PATH`.
- Indian market-time handling with IST normalization plus built-in India exchange closed dates and local holiday overrides.
- Cache-aware pipelines so unchanged inputs can reuse prior outputs.
- Stage 2 reuses saved OHLCV coverage or refreshes only the missing tail unless `--full-refresh` is requested.
- Canonical source naming, row-level text-origin tagging, and cached article fetch reuse for Stage 5 news ingestion.
- Historical ingestion now defaults to `60` months and Stage 5 news refresh now defaults to a `90` day lookback window.
- Stage 4 and Stage 8 default to `gradient_boosting` and keep logistic regression and random forest as optional comparison paths.
- Explicit failure logs and warning metadata instead of silent degradation.
- Managed artifact paths, URL validation, and secret-redacted logs are on by default for local runs.
- Local-first workflow with CLI entrypoints and automated tests.

## Local Bootstrap

```powershell
$env:PYTHONPATH='src'
python -m kubera.bootstrap
python -m kubera.ingest.market_data
python -m kubera.features.historical_features
python -m pytest
```

Provider keys are not required for the bootstrap commands above.

## Full Local Workflow

```powershell
$env:PYTHONPATH='src'
python -m kubera.bootstrap
python -m kubera.ingest.market_data
python -m kubera.features.historical_features
python -m kubera.ingest.news_data
python -m kubera.llm.extract_news
python -m kubera.features.news_features
python -m kubera.models.train_baseline
python -m kubera.models.train_enhanced
python -m kubera.reporting.offline_evaluation
python -m pytest
```

News and LLM commands require provider credentials in `.env`.

`python -m kubera.ingest.market_data --full-refresh` is available when you want to bypass Stage 2 reuse and rebuild the saved OHLCV window from scratch.

## Unified CLI

The top-level `kubera` command now matches the operator workflow more closely:

```powershell
$env:PYTHONPATH='src'
python -m kubera.cli pilot --mode after_close --timestamp 2026-03-10T16:15:00+05:30
python -m kubera.cli week-plan --start-date 2026-03-09 --end-date 2026-03-13
python -m kubera.cli due-run --plan-path artifacts/reports/pilot/weeks/INFY/INFY_NSE_2026-03-09_2026-03-13/INFY_NSE_2026-03-09_2026-03-13_pilot_week_plan.json
python -m kubera.cli operate-week --start-date 2026-03-09 --end-date 2026-03-13
python -m kubera.cli backfill --date 2026-03-11 --mode after_close
python -m kubera.cli review --start-date 2026-03-09 --end-date 2026-03-13 --refresh-offline
python -m kubera.cli dash --view latest
python -m kubera.cli dash --view all --limit 20
python -m kubera.cli dash --view run --prediction-key INFY|NSE|after_close|2026-03-11
python -m kubera.cli explain --prediction-key INFY|NSE|after_close|2026-03-11
```

`pilot` and `predict` both run one live Stage 10 prediction. `review` now builds the Stage 11 final review package. `runs` prints recent stored pilot rows when you want a quick log view without opening the dashboard.

## Ticker And Exchange Overrides

Kubera keeps `INFY` on `NSE` as the default config, but the runtime settings now resolve ticker metadata through one shared path across Stage 5 through Stage 11.

- Built-in catalog entries cover `INFY` and `TCS` on `NSE` and `BSE`.
- `KUBERA_TICKER_CATALOG_PATH` can point to a local JSON file with extra ticker metadata.
- `KUBERA_TICKER`, `KUBERA_EXCHANGE`, `KUBERA_COMPANY_NAME`, `KUBERA_NEWS_ALIASES`, and `KUBERA_YAHOO_TICKER` can override catalog values for local runs.
- Invalid ticker or exchange values fail during settings load instead of silently reusing the default ticker metadata.

Example local catalog shape:

```json
{
  "tickers": [
    {
      "symbol": "WIPRO",
      "exchange": "NSE",
      "company_name": "Wipro Limited",
      "search_aliases": ["WIPRO", "Wipro Limited"]
    }
  ]
}
```

## Live Pilot Workflow

Run Stage 2 through Stage 8 at least once before the pilot so the saved baseline and enhanced model artifacts already exist. The pilot reuses those frozen artifacts, refreshes Stage 2, Stage 5, Stage 6, and Stage 7 with an as-of cutoff, and does not retrain models.

```powershell
$env:PYTHONPATH='src'
python -m kubera.pilot.live_pilot plan-week --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13
python -m kubera.pilot.live_pilot run-due --plan-path artifacts/reports/pilot/weeks/INFY/INFY_NSE_2026-03-09_2026-03-13/INFY_NSE_2026-03-09_2026-03-13_pilot_week_plan.json --now 2026-03-10T11:00:00Z
python -m kubera.pilot.live_pilot operate-week --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13 --now 2026-03-10T11:00:00Z --as-of 2026-03-10 --dry-run
python -m kubera.pilot.live_pilot run --prediction-mode pre_market --timestamp 2026-03-10T08:05:00+05:30
python -m kubera.pilot.live_pilot run --prediction-mode after_close --timestamp 2026-03-10T16:15:00+05:30
python -m kubera.pilot.live_pilot run --prediction-mode after_close --timestamp 2026-03-10T16:15:00+05:30 --explain
python -m kubera.pilot.live_pilot backfill-due --pilot-start-date 2026-03-10 --pilot-end-date 2026-03-10 --as-of 2026-03-10
python -m kubera.pilot.live_pilot backfill-actuals --prediction-date 2026-03-11 --prediction-mode after_close
python -m kubera.pilot.live_pilot annotate --prediction-mode after_close --prediction-date 2026-03-11 --news-quality-note "Sparse coverage"
python -m kubera.pilot.live_pilot run --ticker TCS --exchange NSE --prediction-mode after_close --timestamp 2026-03-10T16:15:00+05:30
```

`plan-week` writes the deterministic one-week manifest. `run-due` executes only due, incomplete slots from that manifest. `operate-week` wraps manifest creation, due-slot execution, and eligible backfill work into one scheduler-friendly command while still keeping prediction generation and actual backfill as distinct internal steps. `run` stays available for one-off manual execution. `backfill-due` sweeps one pilot window for any eligible pending rows. `backfill-actuals` updates only realized-outcome and correctness columns for matching pending rows. `annotate` updates only the latest matching row's manual note fields.

Direct pilot runs now print a human-readable terminal summary after the pilot row and snapshot are saved. `plan-week`, `run-due`, `backfill-due`, and `operate-week` also print compact operator summaries so scheduled runs are easier to inspect from terminal or Task Scheduler logs. Repeated runs for the same `prediction_key` remain append-only and are labeled with `prediction_attempt_number`; Stage 11 treats the latest timestamped row as canonical while still surfacing rerun counts and degraded-news notes. `run --explain` also prints a labeled Gemini-generated summary after the normal pilot summary when `KUBERA_LLM_API_KEY` is present. If the key is missing or the explanation call fails, the pilot run still exits cleanly.

Kubera does not manage Windows Task Scheduler or cron for you. The Stage 10 commands are scheduler-friendly, but the actual scheduling remains a local manual choice.

## Pilot Outputs

- `artifacts/reports/pilot/*_pilot_log.csv` stores one append-only log per prediction mode.
- `artifacts/reports/pilot/snapshots/<ticker>/*_pilot_snapshot.json` stores one JSON snapshot per pilot run.
- `artifacts/reports/pilot/weeks/<ticker>/<ticker>_<exchange>_<start>_<end>/` stores week manifests, per-slot status markers, and week status summaries.
- Pilot rows include timestamps, cutoff dates, prediction mode, raw and calibrated probabilities, selective `up` or `down` or `abstain` action state, data quality score and grade, disagreement flags, linked article ids, top event counts, stage artifact references, model artifact references, Stage 5 and Stage 6 retry counters, per-stage durations, total runtime, runtime warnings, fallback-heavy warnings, `prediction_attempt_number`, and actual-outcome fields when backfilled.

## Stage 8 Outputs

- `data/features/merged/*_enhanced_dataset.csv` stores the prediction-date-aligned historical plus news dataset used for Stage 8 training.
- `artifacts/models/enhanced/*_enhanced_model.pkl` stores one enhanced model per prediction mode.
- `artifacts/reports/enhanced/*_enhanced_predictions.csv` and `*_enhanced_metrics.json` store per-mode evaluation outputs, including raw and calibrated probability paths.
- `artifacts/reports/enhanced/*_baseline_comparison.csv` and `*_baseline_comparison.json` store aligned baseline-versus-enhanced comparisons and disagreement summaries.

## Stage 9 Outputs

- `artifacts/reports/evaluation/*_offline_evaluation_predictions.csv` stores one held-out, mode-specific prediction table with baseline, enhanced, blended, naive, and ablation outputs on the same rows.
- `artifacts/reports/evaluation/*_offline_metrics.csv` stores the long-form metrics table for all compared variants across all rows, fresh-news rows, carried-forward rows, zero-news rows, fallback-heavy rows, high-confidence rows, abstain-eliminated rows, and data-quality slices.
- `artifacts/reports/evaluation/*_offline_evaluation_summary.json` and `*_offline_evaluation_summary.md` store the run summary, calibration notes, selective coverage, conservative enhanced-versus-baseline evidence notes, saved input lineage, and explicit diagnostics when news features contributed nothing.

## Final Review Workflow

Stage 11 reuses saved Stage 9 evaluation artifacts only when they still align with the current Stage 3, Stage 7, and Stage 8 inputs. If the saved Stage 9 outputs are missing or stale, rerun Stage 11 with `--refresh-offline-evaluation` so the report is rebuilt intentionally instead of auto-refreshing behind the scenes. It does not fabricate pilot evidence when pilot logs are absent or incomplete.

```powershell
$env:PYTHONPATH='src'
python -m kubera.reporting.final_review --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13
python -m kubera.reporting.final_review --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13 --refresh-offline-evaluation
python -m kubera.reporting.final_review --ticker TCS --exchange NSE --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13
```

The final review summarizes Stage 3 coverage, Stage 5 article volume, Stage 6 extraction behavior, Stage 7 zero-news coverage, Stage 9 per-mode metrics, and Stage 10 pilot evidence for the requested market-session window. It now includes Stage 5 and Stage 6 retry totals, saved pilot runtime notes, and direct diagnostics when the enhanced model tied baseline because news features did not contribute. When expected pilot days or modes are missing, the report marks that gap explicitly instead of claiming complete coverage.

## Stage 11 Outputs

- `artifacts/reports/final_review/*_final_review.json` stores the machine-readable final review payload.
- `artifacts/reports/final_review/*_final_review.md` stores the human-readable final review package.
- Stage 11 outputs include artifact traceability for offline metrics, offline summary artifacts, pilot logs, and saved model metadata.

## Source Notes

- Historical market data defaults to `yfinance` with NSE symbols such as `INFY.NS`.
- Stage 5 news discovery always supports the free Google News RSS and NSE corporate announcement sources.
- Stage 5 also includes `alphavantage` when `KUBERA_NEWS_PROVIDER=alphavantage` and `KUBERA_ALPHAVANTAGE_API_KEY` are configured.
- Stage 5 also includes `marketaux` when `KUBERA_NEWS_PROVIDER=marketaux` and `KUBERA_NEWS_API_KEY` are configured.
- Stage 5 ranks sources before the global article cap in the order `nse_announcements`, `alphavantage`, the configured paid provider, then `google_news_rss`, with company-specificity and recency used as tie-breakers.
- Stage 5 now defaults to a `90` day lookback window.
- Stage 5 now paces provider requests with `KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS` and article fetches with `KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS`.
- Stage 5 rejects malformed or suspicious article URLs and falls back to snippet-only handling instead of fetching unsafe targets.
- Stage 5 records degraded-source warnings when a saved run is Google-only, mostly fallback text, or otherwise low-specificity.
- Stage 6 stays on plain-text extraction first, then routes only weak rows through bounded recovery using URL context by default and Google Search only when `KUBERA_LLM_RECOVERY_GOOGLE_SEARCH_ENABLED=true`.
- `KUBERA_LLM_RECOVERY_MAX_ARTICLES_PER_RUN` bounds weak-row recovery, and `KUBERA_LLM_RECOVERY_MODEL_POOL_JSON` overrides the tool-enabled Gemini recovery pool with explicit capability and quota metadata.
- Logs redact common key and bearer-token patterns before writing to console or files.
- Provider and publisher terms should still be reviewed before using wider or unattended news automation.

## Local Notes

See [`docs/local_notes.md`](docs/local_notes.md) for the repo-level architecture note, NSE timing rules, feature definitions, evaluation methodology, and current limitations.

## Manual Follow-Up

The code, tests, and local report generation are in place, but the real one-week Stage 10 pilot is still manual work:

- Run both `pre_market` and `after_close` pilot commands for five trading sessions.
- Backfill actual outcomes after the relevant closes are available.
- Add outage, sparse-news, or market-shock notes with `annotate`.
- Re-run Stage 11 for that exact pilot window.

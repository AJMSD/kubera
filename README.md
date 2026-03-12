# kubera

Kubera is a local, config-driven system for predicting next-day NSE stock direction from historical market data and company news.

It is built to fetch source data, normalize it into traceable local artifacts, extract structured signals from news articles, generate model-ready features, and support fair comparison between historical-only and news-aware workflows.

## What The Project Does

- Fetches and cleans daily OHLCV market data for a configured NSE ticker.
- Builds historical technical features and next-day direction labels.
- Discovers company news, normalizes article metadata, deduplicates coverage, and fetches article text with fallback handling.
- Uses the Gemini API with Gemma to turn article text into validated structured signals such as sentiment, event type, severity, horizon, and confidence.
- Aggregates article-level signals into daily pre-market and after-close news feature snapshots with NSE-aware market-time alignment.
- Trains a baseline historical-only model plus separate pre-market and after-close enhanced models that merge historical and news features on the same target rows.
- Runs held-out offline evaluation with naive baselines, feature ablations, and mode-separated evidence summaries.
- Saves per-mode enhanced predictions, metrics, merged Stage 8 datasets, baseline-vs-enhanced comparison reports, and Stage 9 aligned offline evaluation outputs.
- Persists raw snapshots, processed tables, feature tables, metadata, logs, and run config so outputs stay reproducible and traceable.

## Project Characteristics

- Config-driven defaults for ticker, exchange, provider selection, timing rules, and output paths.
- Shared ticker resolution from built-in catalog defaults plus optional local catalog overrides through `KUBERA_TICKER_CATALOG_PATH`.
- Indian market-time handling with IST normalization plus local holiday overrides.
- Cache-aware pipelines so unchanged inputs can reuse prior outputs.
- Stage 2 reuses saved OHLCV coverage or refreshes only the missing tail unless `--full-refresh` is requested.
- Canonical source naming, row-level text-origin tagging, and cached article fetch reuse for Stage 5 news ingestion.
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
python -m kubera.pilot.live_pilot run --prediction-mode pre_market --timestamp 2026-03-10T08:05:00+05:30
python -m kubera.pilot.live_pilot run --prediction-mode after_close --timestamp 2026-03-10T16:15:00+05:30
python -m kubera.pilot.live_pilot backfill-due --pilot-start-date 2026-03-10 --pilot-end-date 2026-03-10 --as-of 2026-03-10
python -m kubera.pilot.live_pilot backfill-actuals --prediction-date 2026-03-11 --prediction-mode after_close
python -m kubera.pilot.live_pilot annotate --prediction-mode after_close --prediction-date 2026-03-11 --news-quality-note "Sparse coverage"
python -m kubera.pilot.live_pilot run --ticker TCS --exchange NSE --prediction-mode after_close --timestamp 2026-03-10T16:15:00+05:30
```

`plan-week` writes the deterministic one-week manifest. `run-due` executes only due, incomplete slots from that manifest. `run` stays available for one-off manual execution. `backfill-due` sweeps one pilot window for any eligible pending rows. `backfill-actuals` updates only realized-outcome and correctness columns for matching pending rows. `annotate` updates only the latest matching row's manual note fields.

Kubera does not manage Windows Task Scheduler or cron for you. The Stage 10 commands are scheduler-friendly, but the actual scheduling remains a local manual choice.

## Pilot Outputs

- `artifacts/reports/pilot/*_pilot_log.csv` stores one append-only log per prediction mode.
- `artifacts/reports/pilot/snapshots/<ticker>/*_pilot_snapshot.json` stores one JSON snapshot per pilot run.
- `artifacts/reports/pilot/weeks/<ticker>/<ticker>_<exchange>_<start>_<end>/` stores week manifests, per-slot status markers, and week status summaries.
- Pilot rows include timestamps, cutoff dates, prediction mode, model outputs, disagreement flags, linked article ids, top event counts, stage artifact references, model artifact references, Stage 5 and Stage 6 retry counters, per-stage durations, total runtime, runtime warnings, fallback-heavy warnings, and actual-outcome fields when backfilled.

## Stage 8 Outputs

- `data/features/merged/*_enhanced_dataset.csv` stores the prediction-date-aligned historical plus news dataset used for Stage 8 training.
- `artifacts/models/enhanced/*_enhanced_model.pkl` stores one enhanced model per prediction mode.
- `artifacts/reports/enhanced/*_enhanced_predictions.csv` and `*_enhanced_metrics.json` store per-mode evaluation outputs.
- `artifacts/reports/enhanced/*_baseline_comparison.csv` and `*_baseline_comparison.json` store aligned baseline-versus-enhanced comparisons and disagreement summaries.

## Stage 9 Outputs

- `artifacts/reports/evaluation/*_offline_evaluation_predictions.csv` stores one held-out, mode-specific prediction table with baseline, enhanced, naive, and ablation outputs on the same rows.
- `artifacts/reports/evaluation/*_offline_metrics.csv` stores the long-form metrics table for all compared variants across all rows, news-heavy rows, and zero-news rows.
- `artifacts/reports/evaluation/*_offline_evaluation_summary.json` and `*_offline_evaluation_summary.md` store the run summary, coverage notes, and conservative enhanced-versus-baseline evidence notes.

## Final Review Workflow

Stage 11 reuses saved Stage 9 evaluation artifacts and saved Stage 10 pilot logs when they exist. If the Stage 9 outputs are missing, the command refreshes Stage 9 once before writing the final review. It does not fabricate pilot evidence when pilot logs are absent or incomplete.

```powershell
$env:PYTHONPATH='src'
python -m kubera.reporting.final_review --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13
python -m kubera.reporting.final_review --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13 --refresh-offline-evaluation
python -m kubera.reporting.final_review --ticker TCS --exchange NSE --pilot-start-date 2026-03-09 --pilot-end-date 2026-03-13
```

The final review summarizes Stage 3 coverage, Stage 5 article volume, Stage 6 extraction behavior, Stage 7 zero-news coverage, Stage 9 per-mode metrics, and Stage 10 pilot evidence for the requested market-session window. It now includes Stage 5 and Stage 6 retry totals plus saved pilot runtime notes when those logs exist. When expected pilot days or modes are missing, the report marks that gap explicitly instead of claiming complete coverage.

## Stage 11 Outputs

- `artifacts/reports/final_review/*_final_review.json` stores the machine-readable final review payload.
- `artifacts/reports/final_review/*_final_review.md` stores the human-readable final review package.
- Stage 11 outputs include artifact traceability for offline metrics, offline summary artifacts, pilot logs, and saved model metadata.

## Source Notes

- Historical market data defaults to `yfinance` with NSE symbols such as `INFY.NS`.
- Stage 5 news discovery currently supports `marketaux` when configured.
- Stage 5 now paces provider requests with `KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS` and article fetches with `KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS`.
- Stage 5 rejects malformed or suspicious article URLs and falls back to snippet-only handling instead of fetching unsafe targets.
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

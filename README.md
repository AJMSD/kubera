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
- Indian market-time handling with IST normalization plus local holiday overrides.
- Cache-aware pipelines so unchanged inputs can reuse prior outputs.
- Canonical source naming, row-level text-origin tagging, and cached article fetch reuse for Stage 5 news ingestion.
- Explicit failure logs and warning metadata instead of silent degradation.
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

## Stage 8 Outputs

- `data/features/merged/*_enhanced_dataset.csv` stores the prediction-date-aligned historical plus news dataset used for Stage 8 training.
- `artifacts/models/enhanced/*_enhanced_model.pkl` stores one enhanced model per prediction mode.
- `artifacts/reports/enhanced/*_enhanced_predictions.csv` and `*_enhanced_metrics.json` store per-mode evaluation outputs.
- `artifacts/reports/enhanced/*_baseline_comparison.csv` and `*_baseline_comparison.json` store aligned baseline-versus-enhanced comparisons and disagreement summaries.

## Stage 9 Outputs

- `artifacts/reports/evaluation/*_offline_evaluation_predictions.csv` stores one held-out, mode-specific prediction table with baseline, enhanced, naive, and ablation outputs on the same rows.
- `artifacts/reports/evaluation/*_offline_metrics.csv` stores the long-form metrics table for all compared variants across all rows, news-heavy rows, and zero-news rows.
- `artifacts/reports/evaluation/*_offline_evaluation_summary.json` and `*_offline_evaluation_summary.md` store the run summary, coverage notes, and conservative enhanced-versus-baseline evidence notes.

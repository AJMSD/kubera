# kubera

Kubera is a local, config-driven system for predicting next-day NSE stock direction from historical market data and company news.

It is built to fetch source data, normalize it into traceable local artifacts, extract structured signals from news articles, generate model-ready features, and support fair comparison between historical-only and news-aware workflows.

## What The Project Does

- Fetches and cleans daily OHLCV market data for a configured NSE ticker.
- Builds historical technical features and next-day direction labels.
- Discovers company news, normalizes article metadata, deduplicates coverage, and fetches article text with fallback handling.
- Uses the Gemini API with Gemma to turn article text into validated structured signals such as sentiment, event type, severity, horizon, and confidence.
- Aggregates article-level signals into daily pre-market and after-close news feature snapshots with NSE-aware market-time alignment.
- Trains and evaluates a baseline historical-only model while keeping the news-derived feature pipeline available for comparison work.
- Persists raw snapshots, processed tables, feature tables, metadata, logs, and run config so outputs stay reproducible and traceable.

## Project Characteristics

- Config-driven defaults for ticker, exchange, provider selection, timing rules, and output paths.
- Indian market-time handling with IST normalization plus local holiday overrides.
- Cache-aware pipelines so unchanged inputs can reuse prior outputs.
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
python -m pytest
```

News and LLM commands require provider credentials in `.env`.

# Kubera Local Notes

## Architecture

Kubera is organized as a staged local pipeline:

- Stage 1 sets config, paths, logging, run metadata, timezone helpers, and the NSE trading calendar boundary.
- Stage 2 fetches and validates daily OHLCV history into `data/processed/market_data`.
- Stage 3 turns cleaned market data into historical features and next-day labels in `data/features/historical`.
- Stage 4 trains the historical-only baseline model and saves artifacts under `artifacts/models/baseline` and `artifacts/reports/baseline`.
- Stage 5 discovers company news, deduplicates it, fetches article text when possible, and writes normalized rows plus raw snapshots.
- Stage 6 sends normalized article inputs through the configured Gemini-based extraction path and validates every saved row against the Stage 6 schema.
- Stage 7 maps article timestamps to NSE prediction windows and aggregates article-level signals into daily `pre_market` and `after_close` feature rows.
- Stage 8 merges Stage 3 and Stage 7 rows on shared prediction targets, trains one enhanced model per prediction mode, and saves aligned comparison artifacts.
- Stage 9 evaluates baseline, enhanced, and benchmark variants on the same held-out rows for each prediction mode.
- Stage 10 reuses frozen Stage 4 and Stage 8 model artifacts, refreshes upstream data with an as-of cutoff, and appends pilot rows plus snapshot JSON files.

The main package boundaries follow that flow:

- `src/kubera/config` owns typed settings and environment loading.
- `src/kubera/ingest` owns external data access.
- `src/kubera/llm` owns structured extraction.
- `src/kubera/features` owns historical and news feature engineering.
- `src/kubera/models` owns training and saved-model inference.
- `src/kubera/reporting` owns offline evaluation and summary outputs.
- `src/kubera/pilot` owns live-pilot inference, actual backfill, and manual note updates.
- `src/kubera/utils` owns shared cross-stage helpers such as paths, time normalization, calendars, hashing, serialization, and run metadata.

## Ticker Catalog And Exchange Resolution

- `INFY` on `NSE` stays the default config only.
- Runtime ticker and exchange overrides now resolve through one shared settings path instead of stage-specific overrides.
- Built-in catalog defaults cover `INFY` and `TCS` on `NSE` and `BSE`.
- `KUBERA_TICKER_CATALOG_PATH` can point to a local JSON file with additional ticker metadata.
- `KUBERA_COMPANY_NAME`, `KUBERA_NEWS_ALIASES`, and `KUBERA_YAHOO_TICKER` can override catalog metadata for local runs.
- Invalid ticker strings and unsupported exchange codes fail during settings load before any stage starts writing files.
- Exchange defaults now carry the market calendar name, market hours, and provider symbol suffix rules together.

## NSE Timing Rules

- Market timezone is `Asia/Kolkata`.
- Regular-session open is `09:15`.
- Regular-session close is `15:30`.
- The local holiday override file is `config/market_holidays.local.json`.
- All stored timestamps stay explicit about timezone. Stage logic converts provider timestamps into market time before using them for date mapping.

Stage 7 timing rules:

- An article published on a trading day before the open can affect that same day's `pre_market` row.
- Any article published intraday or after the close maps to the next trading day's `pre_market` row.
- `after_close` rows always target the next trading day after the first trading session that could have known the article.
- Weekend and holiday articles roll forward to the next trading session through the market calendar.

Stage 10 timing rules:

- `pre_market` pilot runs must use a trading-day timestamp before the open.
- A `pre_market` pilot uses the previous completed trading day as the market-data cutoff and predicts the same trading day.
- `after_close` pilot runs must use a trading-day timestamp at or after the close.
- An `after_close` pilot uses the same trading day as the market-data cutoff and predicts the next trading day.
- Non-trading timestamps fail fast instead of guessing.

## Data Sources

Current defaults:

- Historical market data uses `yfinance`.
- The default NSE symbol mapping uses Yahoo Finance style symbols such as `INFY.NS`.
- The default BSE symbol mapping uses Yahoo Finance style symbols such as `INFY.BO`.
- News discovery uses `marketaux` when `KUBERA_NEWS_PROVIDER` and `KUBERA_NEWS_API_KEY` are configured.
- LLM extraction uses the Gemini API path when `KUBERA_LLM_PROVIDER` and `KUBERA_LLM_API_KEY` are configured.

Source behavior notes:

- `yfinance` is practical for local development, but it is still a free market-data source with delayed or corrected rows possible outside exchange-grade feeds.
- Marketaux coverage is good enough for a prototype, but Indian single-company coverage can still be sparse, duplicated, or headline-heavy on quiet days.
- Stage 5 separates provider discovery from direct article-text acquisition so text-fetch failures do not destroy article metadata coverage.
- Stage 5 paces provider requests with `KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS` and article fetches with `KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS`.
- Stage 5 metadata now records the active pacing values and a `source_terms_review_required` flag.
- Provider and publisher terms still need a manual review before broader or unattended automation.

## Feature Definitions

Historical feature families:

- Close-based returns over `1d`, `3d`, and `5d` windows by default.
- Simple moving averages over `5`, `10`, and `20` trading days by default.
- Rolling close-return volatility over `5d` and `10d`.
- One-day volume change plus a volume-to-moving-average ratio.
- Wilder RSI over a `14` day window by default.
- The target label is next-day direction from the current row's close to the next trading row's close.

News feature families:

- Article counts, bullish and bearish counts, neutral counts, warning counts, and extraction-mode counts.
- Mean sentiment, relevance, confidence, severity, and content-quality scores.
- Event counts such as earnings, guidance, partnership, lawsuit, product, macro, and other schema-defined event types.
- Weighted sentiment, bullish, bearish, relevance, and confidence scores.
- Fallback and quality fields such as `news_fallback_article_ratio` and `news_avg_content_quality_score`.

Weighting rules:

- Stage 7 starts with an extraction-mode quality weight.
- Default quality weights are `1.0` for `full_article`, `0.75` for `headline_plus_snippet`, and `0.5` for `headline_only`.
- Article weight is `quality_weight * relevance_score * confidence_score` when confidence weighting is enabled.
- Weighted relevance and weighted confidence use quality weights.
- Weighted sentiment and weighted directional scores use article weights.

No-news defaults:

- Stage 7 writes an explicit zero-filled row when a trading day and prediction mode have no mapped articles.
- Stage 10 reuses the saved Stage 7 row when it exists and synthesizes the same zero row when it does not.
- Zero-news pilot runs still produce enhanced-model inference so long as the saved enhanced artifact exists.

## Model Comparison Methodology

- Stage 4 and Stage 8 both use temporal train, validation, and test splits.
- Stage 8 keeps those split ratios aligned with Stage 4 so baseline and enhanced comparisons stay fair.
- Stage 8 trains separate enhanced models for `pre_market` and `after_close`.
- Stage 9 evaluates every variant on the same held-out rows for one prediction mode at a time.
- Saved Stage 9 outputs include the baseline historical-only model, the full enhanced model, the naive majority-class baseline, the naive previous-day-direction baseline, and several news ablations.
- Stage 9 also breaks metrics out across all held-out rows, `news_heavy_rows`, and `zero_news_rows`.
- The offline evaluation summary is the evidence source for whether news helped, not the live pilot.

## Live Pilot Operation

- Stage 10 uses frozen saved model artifacts from Stage 4 and Stage 8. It does not retrain.
- Pilot commands refresh Stage 2, Stage 5, Stage 6, and Stage 7 artifacts with explicit market-date and publish-time cutoffs.
- Pilot logs are append-only and separated by prediction mode.
- Each pilot row stores timestamps, prediction mode, market cutoffs, baseline and enhanced outputs, disagreement flags, linked article ids, top non-zero event counts, stage artifact references, model artifact references, status, failure details, Stage 5 and Stage 6 retry counters, per-stage durations, total runtime, and note fields.
- `backfill-actuals` updates only actual-outcome and correctness columns for matching pending rows.
- `annotate` updates only the latest matching row's manual note fields.
- `KUBERA_PILOT_FALLBACK_HEAVY_RATIO_THRESHOLD` controls when fallback-heavy warnings are recorded. The default is `0.5`.
- No real one-week pilot notes exist yet in this repo. Add outage, sparse-news, or source-issue notes through `annotate` after live runs happen.

## Final Review Package

- Stage 11 reads the saved Stage 9 offline metrics CSV, offline summary JSON, and offline summary Markdown first.
- If those Stage 9 outputs are missing, Stage 11 runs the offline evaluation once and then reuses the saved outputs it produced.
- Stage 11 reads both Stage 10 pilot logs for an explicit market-session window and compares them to the expected trading days from the market calendar.
- Missing pilot days, missing modes, partial failures, pending actuals, fallback-heavy rows, zero-news rows, and manual notes are reported as gaps or caveats, not smoothed away.
- Stage 11 now surfaces saved pilot retry totals and runtime summaries as operational notes when pilot rows exist.
- The Stage 11 Markdown report is meant to be readable without opening raw run folders first, while the JSON payload keeps the machine-readable summary and artifact traceability.
- Stage 11 does not infer operational reliability from missing pilot evidence, and it does not turn the report into trading advice.

## Assumptions

- `INFY` is the default because the repo is a single-stock NSE-first prototype and `Infosys Limited` provides stable ticker aliases for both market and news paths.
- `yfinance` was chosen for Stage 2 because it keeps local bootstrap and development simple.
- Marketaux was chosen for Stage 5 because it exposes entity-driven company-news discovery without building a custom search adapter first.
- Gemini with schema validation was chosen for Stage 6 because the pipeline depends on structured extraction rather than free-form summaries.
- Default fallback weights intentionally down-rank partial article text instead of dropping it, since sparse news is common and a zero-row-only policy would erase too much signal.

## Current Limitations

- Kubera is still a single-ticker v1 by default, even though most pipeline boundaries are config-driven.
- Stage 12 removed hidden single-ticker assumptions across Stage 5 through Stage 11, but it did not add pooled multi-ticker training.
- Free and aggregator-backed sources can be delayed, incomplete, or noisy.
- Full article text is not guaranteed. Some rows rely on headline plus snippet or headline-only fallback paths.
- LLM extraction is validated, but it can still miss nuance, misread context, or lean on incomplete text.
- The live pilot path exists, but a real one-week pre-market and after-close operating run is still a manual follow-up task.
- Live pilot logs are operational evidence, not scientific proof that the strategy is reliable or tradeable.

## What To Use For A Future Writeup

- The core question is whether company-news features improve next-day direction prediction over a historical-only baseline for one NSE stock.
- The clean comparison is baseline historical-only versus mode-specific enhanced models evaluated on the same held-out rows.
- The market-specific differentiator is the NSE timing logic that separates `pre_market` and `after_close` prediction windows.
- The extraction differentiator is the fallback-aware article pipeline that preserves lower-quality coverage with explicit penalties instead of pretending it is full text.
- Improvement or non-improvement claims should cite the saved Stage 9 offline evaluation outputs first, then use Stage 10 pilot logs only as operating evidence.
- The main tradeoffs today are free-data coverage versus implementation speed, frozen logistic-regression models versus heavier experimentation, and prototype-friendly local automation versus stronger production scheduling.

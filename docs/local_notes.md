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

Stage 2 refresh behavior:

- Saved OHLCV coverage is reused when it already covers the requested end date.
- If only the tail is missing, Stage 2 fetches from a small overlap window, merges, dedupes, and rewrites the cleaned table.
- `python -m kubera.ingest.market_data --full-refresh` bypasses reuse when you want a full refetch.

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
- Built-in exchange closed dates currently supplement Republic Day, Maharashtra Day, Independence Day, and Gandhi Jayanti for supported Indian exchanges.
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
- Historical ingestion defaults to `36` months of daily data.
- The default NSE symbol mapping uses Yahoo Finance style symbols such as `INFY.NS`.
- The default BSE symbol mapping uses Yahoo Finance style symbols such as `INFY.BO`.
- News discovery always includes Google News RSS and NSE corporate announcements.
- News discovery also uses Alpha Vantage `NEWS_SENTIMENT` when `KUBERA_NEWS_PROVIDER=alphavantage` and `KUBERA_ALPHAVANTAGE_API_KEY` are configured.
- News discovery also uses `marketaux` when `KUBERA_NEWS_PROVIDER` and `KUBERA_NEWS_API_KEY` are configured.
- Stage 5 news refresh defaults to a `90` day lookback window.
- LLM extraction uses the Gemini API path when `KUBERA_LLM_PROVIDER` and `KUBERA_LLM_API_KEY` are configured.

Source behavior notes:

- `yfinance` is practical for local development, but it is still a free market-data source with delayed or corrected rows possible outside exchange-grade feeds.
- Marketaux coverage is good enough for a prototype, but Indian single-company coverage can still be sparse, duplicated, or headline-heavy on quiet days.
- Alpha Vantage is supported as an optional ticker-scoped provider, not as the forced default path. It should still be treated as coverage-dependent and potentially throttled on free-tier access.
- Google News RSS helps broaden free discovery, but it still needs dedupe and publisher-level filtering because syndication is common.
- NSE corporate announcements are high-relevance primary-source filings and are prioritized before the global article cap is applied.
- Stage 5 ranks saved candidates in the order `nse_announcements`, `alphavantage`, the configured paid provider, then `google_news_rss`, with company-specificity and recency used to stop generic commentary from crowding out ticker-specific rows.
- Stage 5 separates provider discovery from direct article-text acquisition so text-fetch failures do not destroy article metadata coverage.
- Stage 5 now reuses resolved publisher URLs when a fetch succeeds through a validated redirect and records degraded-source warnings when a run ends up Google-only, mostly fallback text, or otherwise low-specificity.
- Stage 5 paces provider requests with `KUBERA_NEWS_PROVIDER_REQUEST_PAUSE_SECONDS` and article fetches with `KUBERA_NEWS_ARTICLE_REQUEST_PAUSE_SECONDS`.
- Stage 5 metadata now records the active pacing values and a `source_terms_review_required` flag.
- Provider and publisher terms still need a manual review before broader or unattended automation.

Stage 6 recovery behavior:

- Plain-text Gemini extraction remains the first pass for every article row.
- Weak rows can enter a separate recovery path that records `request_mode`, `recovery_reason`, and `recovery_status` per saved row.
- URL context recovery is on by default when Kubera already has a trusted article URL but only weak body text.
- Google Search recovery is off by default and only runs when `KUBERA_LLM_RECOVERY_GOOGLE_SEARCH_ENABLED=true`.
- `KUBERA_LLM_RECOVERY_MAX_ARTICLES_PER_RUN` bounds how many weak rows can trigger tool-aware recovery during one run.
- `KUBERA_LLM_RECOVERY_MODEL_POOL_JSON` defines the ordered recovery-only Gemini pool, including model ids, URL-context support, Google-search support, and per-model RPM or RPD limits.
- Recovery-only citations and retrieved URLs stay in raw snapshots instead of normal logs.

## Feature Definitions

Historical feature families:

- Close-based returns over `1d`, `3d`, and `5d` windows by default.
- Simple moving averages over `5`, `10`, and `20` trading days by default.
- Rolling close-return volatility over `5d` and `10d`.
- One-day volume change plus a volume-to-moving-average ratio.
- MACD as the `12` day EMA minus the `26` day EMA of close, plus a `9` day EMA signal line.
- Close divided by the rolling `252` day high and rolling `252` day low.
- `day_of_week`, based on the source row date rather than the prediction target date.
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
- The default split is `70 / 15 / 15`.
- Stage 8 keeps those split ratios aligned with Stage 4 so baseline and enhanced comparisons stay fair.
- Logistic regression remains the default model path for both Stage 4 and Stage 8.
- Stage 4 and Stage 8 also support `gradient_boosting` with fixed tree parameters and the shared run seed.
- Stage 8 trains separate enhanced models for `pre_market` and `after_close`.
- Stage 9 evaluates every variant on the same held-out rows for one prediction mode at a time.
- Saved Stage 9 outputs include the baseline historical-only model, the full enhanced model, the naive majority-class baseline, the naive previous-day-direction baseline, and several news ablations.
- Stage 9 also breaks metrics out across all held-out rows, `news_heavy_rows`, and `zero_news_rows`.
- The offline evaluation summary is the evidence source for whether news helped, not the live pilot.

## Live Pilot Operation

- Stage 10 uses frozen saved model artifacts from Stage 4 and Stage 8. It does not retrain.
- Pilot commands refresh Stage 2, Stage 5, Stage 6, and Stage 7 artifacts with explicit market-date and publish-time cutoffs.
- `plan-week` writes a deterministic manifest of trading-day slots for the requested pilot window.
- `run-due` executes only due, incomplete manifest slots and writes one per-slot status marker.
- `backfill-due` scans one pilot window and backfills any eligible pending rows.
- `operate-week` ensures the manifest exists, runs due slots, then backfills eligible rows and prints one combined operator summary.
- Pilot logs are append-only and separated by prediction mode.
- Each pilot row stores timestamps, prediction mode, market cutoffs, baseline and enhanced outputs, disagreement flags, linked article ids, top non-zero event counts, stage artifact references, model artifact references, status, failure details, Stage 5 and Stage 6 retry counters, per-stage durations, total runtime, note fields, and `prediction_attempt_number`.
- Direct pilot runs also print a human-readable summary block to stdout after the pilot row and snapshot are written.
- `plan-week`, `run-due`, `backfill-due`, and `operate-week` also print compact summaries for scheduler logs and manual terminal use.
- `python -m kubera.pilot.live_pilot run --explain ...` sends the completed snapshot JSON through the existing Gemini client and prints a labeled generated explanation when `KUBERA_LLM_API_KEY` is present.
- Pilot snapshots and final review outputs surface runtime warnings when a run exceeds the configured threshold.
- `backfill-actuals` updates only actual-outcome and correctness columns for matching pending rows.
- `annotate` updates only the latest matching row's manual note fields.
- Repeated runs for the same `prediction_key` are explicit reruns. Stage 11 collapses append-only logs to the latest row per market session while keeping rerun counts and degraded-news notes visible.
- `KUBERA_PILOT_FALLBACK_HEAVY_RATIO_THRESHOLD` controls when fallback-heavy warnings are recorded. The default is `0.5`.
- Default pilot schedule targets are `08:05` IST for `pre_market` and `16:15` IST for `after_close`.
- Scheduler setup stays outside the repo. Kubera prepares the plan and due-run commands, but Windows Task Scheduler or cron remains your local responsibility.
- No real one-week pilot notes exist yet in this repo. Add outage, sparse-news, or source-issue notes through `annotate` after live runs happen.

## Final Review Package

- Stage 11 reads the saved Stage 9 offline metrics CSV, offline summary JSON, and offline summary Markdown first.
- Stage 11 now fails fast if the saved Stage 9 outputs are missing or stale relative to the current Stage 3, Stage 7, or Stage 8 inputs. Use `--refresh-offline-evaluation` when you want Stage 11 to rebuild them on purpose.
- Stage 11 reads both Stage 10 pilot logs for an explicit market-session window and compares them to the expected trading days from the market calendar.
- Missing pilot days, missing modes, partial failures, pending actuals, fallback-heavy rows, zero-news rows, and manual notes are reported as gaps or caveats, not smoothed away.
- Stage 11 now surfaces saved pilot retry totals and runtime summaries as operational notes when pilot rows exist.
- Stage 11 can also read the optional pilot-week status summary for the exact review window and use it for missing-slot diagnostics.
- The Stage 11 Markdown report is meant to be readable without opening raw run folders first, while the JSON payload keeps the machine-readable summary and artifact traceability.
- Stage 9 and Stage 11 summaries now call out when the enhanced model matched baseline because the saved Stage 8 feature-importance summary showed no news-feature contribution.
- Stage 11 does not infer operational reliability from missing pilot evidence, and it does not turn the report into trading advice.

## Security Boundaries

- Stage 5 URL validation rejects non-HTTP schemes, missing hosts, embedded credentials, localhost targets, and private or reserved IP targets.
- Unsafe article URLs fall back to provider snippet handling instead of attempting a direct fetch.
- Managed path assertions keep repo-owned artifact writes inside the configured workspace directories.
- Logging redacts common API-key, token, and bearer-auth fragments before writing error text to console or log files.
- Stage 6 keeps article text inside explicit delimiters and neutralizes embedded article markers before prompt construction.

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

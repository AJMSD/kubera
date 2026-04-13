# Operator notes: news ingest, Marketaux credits, and Stage 7 rebuilds

## Free sources vs Marketaux

- Enable **`KUBERA_NEWS_ENABLE_GOOGLE_NEWS_RSS`** and **`KUBERA_NEWS_ENABLE_NSE_ANNOUNCEMENTS`** (defaults are true) to merge RSS and NSE data alongside the configured paid provider in Stage 5.
- Enable **`KUBERA_NEWS_ENABLE_NSE_RSS`** and **`KUBERA_NEWS_ENABLE_BSE_RSS`** (defaults are true) to include official exchange RSS feeds in Stage 5.
- Use **`KUBERA_NEWS_PROVIDER_PRIORITY`** to control source precedence (default: `nse_rss,bse_rss,nse_announcements,alphavantage,marketaux,economic_times,google_news_rss`).
- Set **`KUBERA_NEWS_OFFICIAL_ONLY=true`** to keep Stage 5 on official RSS feeds only and skip paid/generic feeds.
- **`KUBERA_NEWS_MARKETAUX_LIMIT_PER_REQUEST`**: raise this only if your Marketaux plan allows more articles per HTTP call; fewer pages means fewer API requests toward your daily quota.
- **`KUBERA_NEWS_MARKETAUX_MAX_NEWS_REQUESTS`**: set to a positive integer to cap how many paginated `/news/all` calls run per ingest (`0` = unlimited). Useful when staying under a daily request budget.

## Stage 2 official market precedence

- Stage 2 supports official bhavcopy providers with **`KUBERA_HISTORICAL_PROVIDER_PRIORITY`** (default: `nse_bhavcopy,bse_bhavcopy,yfinance`).
- Set **`KUBERA_HISTORICAL_OFFICIAL_ONLY=true`** to disable non-official fallback providers.
- Bhavcopy publication can lag the close; if official files are delayed, fallback providers prevent full-pipeline stalls unless official-only mode is enabled.

## Marketaux entity cache

- Responses from **`/v1/entity/search`** are cached under **`data/cache/marketaux_entity/`** (relative to your workspace `data_dir`).
- TTL is **`KUBERA_NEWS_MARKETAUX_ENTITY_CACHE_TTL_HOURS`** (default **168** hours). Set to **`0`** to disable caching.

## Stage 7 formula changes

- News feature schema version is **`FEATURE_FORMULA_VERSION`** in `kubera.features.news_features`. After pulling changes that bump it, regenerate Stage 7 artifacts and retrain Stage 8 models before relying on pilot predictions.

## Pilot LLM input size

- **`KUBERA_LLM_MAX_INPUT_CHARS_PILOT`**: optional lower bound for Stage 6 text when invoked from the live pilot (helps tight Gemini TPM budgets). Unset means use **`KUBERA_LLM_MAX_INPUT_CHARS`** for all runs.

## Timeouts (global vs Marketaux)

- **`KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS`** (default **15**): used for Google News RSS, NSE announcements, Economic Times, Alpha Vantage news discovery, and other non-Marketaux HTTP clients in Stage 5.
- **`KUBERA_NEWS_MARKETAUX_CONNECT_TIMEOUT_SECONDS`** (default **10**): TCP/TLS connect budget for Marketaux only (`/v1/entity/search` and `/v1/news/all`). Shorter connects fail fast when the host is unreachable; this does not shorten the read phase.
- **`KUBERA_NEWS_MARKETAUX_READ_TIMEOUT_SECONDS`**: read budget for Marketaux response bodies. If unset, defaults to the same value as **`KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS`** so tuning the global news timeout still updates Marketaux read unless you override it explicitly. Connect must not exceed read (validated at settings load).
- Stage 5 metadata **`fetch_policy`** in raw news snapshots includes these values for the active run.

## Phase 9A research (background)

- For Marketaux **`Read timed out`**, Stage 5 orchestration, and evidence behind the split timeouts above, see [marketaux_phase9a_research.md](marketaux_phase9a_research.md).

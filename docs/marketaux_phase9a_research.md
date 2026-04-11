# Phase 9A - Marketaux timeouts (research note)

This note satisfies the Phase 9A checklist: Stage 5 orchestration, settings map, empirical probes, ranked hypotheses, decision inputs for Phase 9B, and impact on non-Marketaux feeds. **No production code was changed for 9A.**

## 1. Stage 5 orchestration (code-backed)

### Provider order

[`resolve_configured_news_sources`](src/kubera/ingest/news_data.py) builds a **list** in this order:

1. Primary provider from `resolve_news_provider(settings)` (often Marketaux when configured).
2. Google News RSS (if `enable_google_news_rss`).
3. Economic Times (if `enable_economic_times`).
4. NSE announcements (if `enable_nse_announcements`).

### Independence: Marketaux does not block other feeds

[`collect_news_source_results`](src/kubera/ingest/news_data.py) iterates **sequentially** over that list. Each provider runs inside `try` / `except`. On failure:

- `source_warnings` receives `{provider_name}_failed` (e.g. `marketaux_failed`).
- An empty [`CollectedNewsSource`](src/kubera/ingest/news_data.py) is appended with the error in `warnings`.
- The **loop continues**; RSS, NSE, ET, etc. still run afterward.

So a slow or failing Marketaux call **delays** subsequent providers by wall time spent inside Marketaux (including timeouts and retries), but it does **not** discard or skip the rest of the list unless the whole process aborts elsewhere.

### Marketaux call path

1. [`collect_marketaux_source`](src/kubera/ingest/news_data.py) → [`resolve_provider_entities`](src/kubera/ingest/news_data.py): HTTP GET to **`MARKETAUX_ENTITY_SEARCH_URL`** (`/v1/entity/search`), with optional disk cache under `data/cache/marketaux_entity/` when TTL > 0.
2. [`discover_company_news`](src/kubera/ingest/news_data.py): paginated GET to **`MARKETAUX_NEWS_URL`** (`/v1/news/all`), bounded by `marketaux_max_news_requests` and page size `marketaux_limit_per_request`.

### HTTP client behavior

[`MarketauxNewsProvider._get_json`](src/kubera/ingest/news_data.py) uses:

```text
requests.get(url, params=params, timeout=self._timeout_seconds)
```

With a **scalar** `timeout`, `requests` applies one deadline to the operation (not a separate connect vs read tuple in this code path). Failed attempts retry up to `retry_attempts` with `time.sleep(0.5 * attempt)` between tries. Worst-case wall time per logical GET is roughly **`retry_attempts × timeout_seconds`** plus backoff (plus processing), unless an error returns earlier.

**Implication:** A “read timed out” from `urllib3` usually means the full-operation timeout was hit while waiting for the body (or connection), not a separate tunable read phase in Kubera today.

## 2. Settings map (relevant fields)

| Field / env | Role |
|-------------|------|
| `NewsIngestionSettings.request_timeout_seconds` / `KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS` (default **15**) | Passed into `MarketauxNewsProvider` and other news providers as `timeout_seconds`. |
| `marketaux_limit_per_request` / `KUBERA_NEWS_MARKETAUX_LIMIT_PER_REQUEST` | Page size for `/news/all`. |
| `marketaux_max_news_requests` / `KUBERA_NEWS_MARKETAUX_MAX_NEWS_REQUESTS` | Caps paginated news calls (`0` = unlimited in logic). |
| `marketaux_entity_cache_ttl_hours` / `KUBERA_NEWS_MARKETAUX_ENTITY_CACHE_TTL_HOURS` | Entity response cache TTL; `0` disables. |
| `article_retry_attempts` / `KUBERA_NEWS_ARTICLE_RETRY_ATTEMPTS` | Marketaux provider retry count in `_get_json`. |

Article fetch uses **different** timeouts (`article_fetch_timeout_seconds`) for full-text fetches; that is separate from Marketaux discovery above.

## 3. Empirical observations (this workspace, 2026-04-08)

**Reproduction recipe (outside Kubera, no secrets in logs):**

- Use `requests.get` against `https://api.marketaux.com/v1/news/all` and `/v1/entity/search` with an **invalid** `api_token` to measure latency to a terminal HTTP status without exposing a real key.
- **Do not** paste live API tokens into documentation or tickets.

**Sample one-off timings (invalid token, same machine, `timeout=15`):**

| Endpoint | Observed outcome | Elapsed (single sample) |
|----------|------------------|---------------------------|
| `/v1/news/all` | HTTP 401 | ~0.4 s |
| `/v1/entity/search` | HTTP 401 | ~3.2 s |

**Interpretation:** These samples show that **endpoint and server path can dominate latency** even when no heavy payload is returned. They do **not** reproduce `Read timed out`; they establish that successful/error responses can arrive quickly from this network path.

**Read-timeout symptom (`HTTPSConnectionPool ... Read timed out`):** When it appears in logs, it indicates the client waited up to the configured timeout for a completed response and did not get one in time, consistent with **upstream slowness**, **packet loss**, **regional routing**, or **large/slow responses** on authenticated calls. **We did not reproduce a read timeout** in the short probes above; the checklist-allowed outcome is: **intermittent upstream / not reproduced locally; use the recipe below in the field** with a valid key and representative `limit` / page counts.

**External factors to record when debugging:**

- HTTP **429** vs **timeout** (rate limit vs stall).
- VPN/proxy vs direct network.
- Marketaux status pages or incident history (check vendor status when correlating outages).

## 4. Ranked hypotheses (for timeouts on real workloads)

| Rank | Hypothesis | Evidence | Notes |
|------|------------|----------|--------|
| 1 | **Upstream or network slowness** on `api.marketaux.com` (including TLS + transfer) | Matches generic `Read timed out`; intermittent reports | Validate with timestamps and whether **multiple** retries all fail the same way. |
| 2 | **Heavy payload or many pages** (`marketaux_limit_per_request`, pagination) extending transfer time | Longer responses approach `timeout × retries` | Compare `provider_request_count` / retry summaries in logs vs `marketaux_max_news_requests`. |
| 3 | **Entity search slower or colder than news** | Single-sample 401 latency higher on `/entity/search` than `/news/all` | Cache hits reduce entity calls; cold cache does more entity work first. |
| 4 | Client misconfiguration (proxy, DNS) | Rare if other HTTPS sites work | Isolate with minimal `requests` script on same host. |

**Primary narrative (evidence-based):** Without a captured failing trace, the most defensible statement is that **`Read timed out` is consistent with request-level blocking until `KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS`**, driven by **network or API behavior** on the Marketaux host, not with other feeds being disabled by the same exception handler (they are not).

## 5. Impact on other feeds

- **Independence:** After a Marketaux exception, Stage 5 still collects RSS / NSE / ET per the loop in [`collect_news_source_results`](src/kubera/ingest/news_data.py).
- **Waiting:** The run **does** spend time inside Marketaux (entity + news, retries) **before** later providers run. That is sequential **ordering**, not shared state blocking merge logic.

## 6. Decision inputs for Phase 9B (implementation)

| Topic | Recommendation |
|-------|----------------|
| **Timeout bounds (sanity)** | Keep a **floor** ≥ 5–10 s for WAN APIs; **ceiling** high enough for large JSON pages but avoid multi-minute hangs. Current default **15 s** is a reasonable starting point; tune with real p95 from authenticated traces. |
| **Retries** | Existing **limited retries** on idempotent GETs are appropriate; avoid unbounded loops. Consider **shorter timeout + one extra retry** vs **one long timeout** after measuring. |
| **Marketaux-specific vs global `request_timeout_seconds`** | Justified in 9B if traces show entity vs article fetches need different budgets; requires **separate settings** so RSS/NSE are not blindly tied to Marketaux tuning. |
| **Connect vs read split** | Not exposed today (scalar `timeout`). **9B candidate:** use `(connect, read)` tuple for more precise failure messages and faster fail on dead connections. |

### Go / no-go for 9B

- **Go:** Implement evidence-based tuning (timeouts, optional retries, optional Marketaux-specific settings) **after** capturing at least one authenticated latency profile or agreeing ops-only mitigation (schedule runs off-peak, reduce `marketaux_max_news_requests`).
- **No-go (alone):** Blindly raising **global** `KUBERA_NEWS_REQUEST_TIMEOUT_SECONDS` for all providers without isolating Marketaux risks masking failures and lengthening every feed.

---

*References: [Marketaux API](https://www.marketaux.com/) (vendor documentation), Kubera [`news_data.py`](src/kubera/ingest/news_data.py), [`settings.py`](src/kubera/config/settings.py).*

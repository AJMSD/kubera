# Kubera Implementation Deep Dive

Last updated: 2026-03-13 (local workspace)

## Purpose of this document

This is a local engineering brief for discussing Kubera with an expert and planning accuracy improvements. It explains how each section of the codebase works, what was recently added, and what the latest pilot output means.

This file is local-only documentation in your workspace. Nothing here is pushed automatically.

---

## 1. High-level architecture

Kubera is a staged, file-artifact pipeline with reproducibility and traceability as core design goals.

Core flow:

1. Stage 1 bootstrap: load settings, initialize run context, managed paths, logs
2. Stage 2 market ingest: fetch/refresh daily OHLCV for ticker/exchange
3. Stage 3 historical features: build model-ready technical features + label
4. Stage 4 baseline training: historical-only classifier
5. Stage 5 news ingest: multi-source discovery + normalization + dedupe + article text
6. Stage 6 LLM extraction: structured signal extraction per article
7. Stage 7 news features: aggregate article signals into daily pre_market/after_close rows
8. Stage 8 enhanced training: historical + news features
9. Stage 9 offline evaluation: baseline vs enhanced + ablations + naive baselines
10. Stage 10 live pilot: frozen-model inference with operational logging
11. Stage 11 final review: aggregate offline + pilot evidence

Code organization:

- `src/kubera/config`: typed settings, environment, ticker/exchange resolution
- `src/kubera/ingest`: Stage 2 and Stage 5 providers + normalization logic
- `src/kubera/features`: Stage 3 and Stage 7 feature engineering
- `src/kubera/llm`: Stage 6 extraction and recovery routing
- `src/kubera/models`: Stage 4, Stage 8, model utilities, artifact validation
- `src/kubera/pilot`: Stage 10 orchestration and commands
- `src/kubera/reporting`: Stage 9 and Stage 11 reports
- `src/kubera/utils`: paths, calendar, time utils, logging, serialization

---

## 2. Configuration and runtime control

Main implementation center: `src/kubera/config/settings.py`

Important runtime controls:

- Ticker/exchange: default INFY/NSE, but now runtime-safe resolution for alternates
- Providers: Stage 5 can fan in multiple sources (Google RSS, NSE announcements, optional paid/API providers)
- Historical lookback default: now 36 months
- News lookback default: now 90 days
- Model type support: logistic_regression (default) and gradient_boosting
- Stage 6 recovery controls: tool-aware fallback routing and bounded recovery settings

Environment patterns used in practice:

- `PYTHONPATH=src`
- Provider API keys from `.env`
- Explicit command-line overrides for pilot date/time and mode

Key design choice:

- Config snapshots and metadata are saved with runs, so any output can be traced to the exact settings used.

---

## 3. Stage 2 implementation (historical market data)

Main file: `src/kubera/ingest/market_data.py`

What it does:

- Fetches or reuses OHLCV by ticker/exchange/provider
- Applies refresh strategies:
  - `reuse_existing`: use current processed table if full requested window is covered
  - `incremental_tail`: fetch only missing tail with overlap merge/dedupe
  - `full_refresh`: force complete rebuild via CLI flag
- Validates coverage, deduplicates rows, stores metadata and cleaned CSV

Important recent implementation behavior:

- Reuse now requires full-window coverage, preventing stale/underfilled reuse that previously broke downstream stages.

Why this matters for accuracy:

- Feature quality and model stability depend on consistent, sufficient historical window coverage.

---

## 4. Stage 3 implementation (historical features)

Main file: `src/kubera/features/historical_features.py`

Feature families:

- Returns and moving statistics
- RSI/volatility/volume-derived signals
- Richer additions from recent upgrades:
  - MACD + MACD signal
  - 52-week high/low relative ratios
  - day_of_week context

Output includes:

- Feature columns + target label (`next_day_direction`)
- Metadata with formula/version details to protect lineage

Important implementation property:

- Warmup handling and label alignment are strict to avoid leakage and schema drift.

Accuracy implication:

- Richer features can improve separability, but only if enough training rows and quality labels exist.

---

## 5. Stage 4 and Stage 8 implementation (training)

Main files:

- `src/kubera/models/train_baseline.py`
- `src/kubera/models/train_enhanced.py`
- `src/kubera/models/common.py`
- `src/kubera/models/artifact_validation.py`

Current training stack:

- Baseline: historical-only
- Enhanced: historical + Stage 7 news aggregates
- Model type configurable:
  - logistic regression (default)
  - gradient boosting

Key implementation details:

- Temporal splits are deterministic and non-shuffled
- Baseline/enhanced artifact metadata include source lineage hashes
- Stale artifact rejection checks prevent training/evaluation on mismatched upstream versions

Accuracy implication:

- Preventing stale lineage is critical; otherwise performance numbers can look stable but be invalid.

---

## 6. Stage 5 implementation (news ingestion)

Main file: `src/kubera/ingest/news_data.py`

Current multi-source fan-in:

- Google News RSS (free)
- NSE announcements (free, high-relevance potential)
- Optional Marketaux / Alpha Vantage provider paths

Pipeline behavior:

- Collect per-provider payloads
- Normalize to common article schema
- Canonicalize URLs
- Deduplicate and rank/trim under caps
- Fetch article text with safe URL validation and fallback modes
- Save processed CSV + metadata + raw snapshots

Recent hardening:

- NSE envelope handling fixed (`{"data": ...}` response shape)
- Source hardening and degraded-source warnings improved

Accuracy implication:

- This stage is currently the biggest bottleneck: if article quality/volume is low, enhanced model cannot outperform baseline.

---

## 7. Stage 6 implementation (LLM extraction)

Main file: `src/kubera/llm/extract_news.py`

What it does:

- Converts each normalized article into structured, schema-validated signals
- Fields include sentiment, relevance, event type, severity, directional bias, confidence
- Supports deterministic default extraction and bounded recovery routing

Recent upgrade behavior:

- Added selective recovery routing for weak-input rows
- Added model/tool-aware handling controls in config
- Still preserves reproducibility metadata and failure logs

Accuracy implication:

- When input text is weak (headline-only/sparse), extraction becomes noisy and less useful for Stage 7 aggregates.

---

## 8. Stage 7 implementation (news features)

Main file: `src/kubera/features/news_features.py`

What it does:

- Aligns extracted article signals to trading-day prediction windows
- Produces mode-specific (`pre_market`, `after_close`) daily features
- Aggregates counts, confidence, sentiment/event profiles, and quality/fallback diagnostics
- Generates synthetic zero-news rows when needed for continuity

Why this matters for your current run:

- Monday run had zero usable news for that window, so Stage 7 produced synthetic zero-news context.

Accuracy implication:

- If many rows are zero-news or low-quality, enhanced model tends to collapse toward baseline behavior.

---

## 9. Stage 9 and Stage 11 reporting implementation

Main files:

- `src/kubera/reporting/offline_evaluation.py`
- `src/kubera/reporting/final_review.py`

What they do:

- Stage 9: aligned offline benchmark comparison (baseline, enhanced, ablations, naive baselines)
- Stage 11: combines offline evidence and pilot operations evidence for a selected window

Recent implementation strengths:

- Conservative reporting when news adds little/no signal
- Explicit handling of partial/unavailable pilot evidence
- Better diagnostics around retries, degraded news, and operational quality

Accuracy implication:

- Reporting is now better at revealing when enhanced complexity is not buying predictive lift.

---

## 10. Stage 10 pilot implementation and commands

Main file: `src/kubera/pilot/live_pilot.py`

Core commands:

- `run`: one-off pilot run for mode/timestamp
- `plan-week`: create deterministic slot plan
- `run-due`: execute due incomplete slots
- `operate-week`: orchestration wrapper for plan + run + due processing
- `backfill-due` and `backfill-actuals`: fill realized outcomes
- `annotate`: manual notes for shocks/source issues

Recent implementation upgrades:

- Better terminal summaries (human-readable)
- Optional explanation output support
- Improved week-level operational flow and visibility

Important operational behavior:

- Pilot is append-only evidence logging; repeated runs can exist for the same key.

---

## 11. Latest Monday run (16 March) output and interpretation

Command used:

```powershell
$env:PYTHONPATH='src'; python -m kubera.pilot.live_pilot run --prediction-mode pre_market --timestamp 2026-03-16T08:05:00+05:30
```

Observed output summary:

```text
Historical market data ready ... strategy=reuse_existing ... rows=744 ... coverage=2023-03-09..2026-03-13
Company news ready ... providers=['marketaux', 'google_news_rss', 'nse_announcements'] ... rows=15
LLM extraction ready ... model=gemma-3-27b-it ... source_rows=15 ... successes=15 ... failures=0
News features ready ... source_rows=15 ... feature_rows=14 ... coverage=2026-03-05..2026-03-13
Live pilot row recorded ... mode=pre_market ... prediction_date=2026-03-16 ... status=success

Kubera Live Pilot Summary
Ticker: INFY | Exchange: NSE | Mode: pre_market
Prediction date: 2026-03-16 | Run timestamp (IST): 2026-03-16T08:05:00+05:30
Baseline: UP | prob=0.545
Enhanced: UP | prob=0.545
Model agreement: agree
News context: articles=0 | avg_confidence=0.000 | fallback_ratio=0.000 | top_events=none
Warnings fired: yes | codes=zero_news_available, zero_news_row_synthesized
Prior day outcome: 2026-03-13 | status=pending
Total run duration: 97.187s
Status: success
```

What this means technically:

- The run succeeded end-to-end.
- The prediction window had no usable news rows, so zero-news synthesis kicked in.
- Baseline and enhanced predictions are identical because Stage 7 news signal for this target window is effectively absent.

---

## 12. Why probability is 0.545 and why news is low

### Why probability is only 0.545

Most likely combined reasons in this implementation state:

1. No incremental news signal for the target window
- Enhanced model had no active news features for this prediction row.
- Result: enhanced output collapses toward baseline output.

2. Model confidence is naturally moderate in noisy daily direction tasks
- Daily next-day direction for single equities is hard; many rows sit near decision boundary.
- 0.545 is a weak-to-moderate edge, not a strong conviction prediction.

3. Classifier likely sees mixed evidence in historical feature vector
- Historical features can disagree (trend/mean-reversion/volatility regimes), producing probabilities near 0.5.

### Why we are not getting a lot of useful news

From this run and recent behavior:

1. Target-day window can legitimately be empty
- Even if Stage 5 ingests rows globally, the specific pre-market alignment window for a date can have zero mapped articles.

2. Source quality and specificity are uneven
- Broad market commentary can dominate over company-specific actionable events.

3. Provider degradation or limits can reduce high-quality feed contribution
- When paid/provider quotas are hit, free fallback paths may provide less specific signals.

4. Stage 7 mapping is strict by design
- Articles outside the mode/time window do not influence that row, preserving correctness but reducing per-row coverage.

---

## 13. Expert discussion checklist to improve accuracy

Use this list with an expert to prioritize high-impact work:

1. Data quality and coverage
- Increase company-specific article density per prediction window.
- Improve article-body capture rate and reduce weak headline-only rows.
- Revisit source ranking/scoring and cap allocation toward high-specificity sources.

2. Signal engineering
- Add stronger event recency decay and source reliability weighting.
- Add mode-specific features that better separate pre_market vs after_close behavior.
- Track and penalize generic macro-only article influence.

3. Modeling
- Evaluate gradient boosting vs logistic per mode with calibration checks.
- Tune threshold using validation objective aligned to your business goal (not fixed 0.5 by default).
- Add probability calibration (Platt/isotonic) and compare Brier/log-loss changes.

4. Evaluation rigor
- Slice performance by zero-news vs non-zero-news, high-confidence vs low-confidence, and event type bins.
- Run rolling-window backtests to detect regime drift.

5. Operations
- Maintain explicit degraded-news flags in pilot decisions.
- Avoid over-trusting enhanced predictions on synthetic zero-news rows.

---

## 14. How to reproduce this exact run locally

```powershell
$env:PYTHONPATH='src'
python -m kubera.pilot.live_pilot run --prediction-mode pre_market --timestamp 2026-03-16T08:05:00+05:30
```

Optional follow-up commands:

```powershell
python -m kubera.pilot.live_pilot backfill-actuals --prediction-date 2026-03-16 --prediction-mode pre_market
python -m kubera.reporting.final_review --pilot-start-date 2026-03-10 --pilot-end-date 2026-03-16 --refresh-offline-evaluation
python -m pytest -q
```

---

## 15. Current verification status

- Full test suite currently passes in this workspace: `192 passed`.
- Monday pre-market pilot run for 2026-03-16 completed with `status=success`.
- Key issue remains: low news in the target window produced identical baseline/enhanced probability (`0.545`).

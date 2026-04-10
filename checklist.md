# Consumer Product Checklist

## Product Contract

Kubera must ship a consumer-grade default experience around one command:

- `kubera run` is the primary consumer entrypoint.
- A consumer can run `kubera run` with no CLI flags and receive a usable prediction flow.
- The command must auto-resolve the prediction window instead of requiring a manual `--timestamp`.
- The CLI dashboard is the primary product surface; HTML remains a secondary artifact.
- The CLI dashboard must show the prediction, model probabilities, and top 3 linked news items for the resolved prediction window.
- News must be framed as linked context for the prediction window, not as exact article-level causal attribution.
- The default setup must not require paid providers; free-tier external APIs are acceptable.
- Fully offline or zero-key operation is not a v1 requirement.

## Track 1 - One-command default run

### Goal

Make `kubera run` the single documented consumer path that produces a complete end-to-end experience without requiring the user to understand mode, timestamp, dashboard flags, or multi-command workflows.

### Tasks

- [x] Make `kubera run` the only quickstart command presented as the default consumer flow in product-facing docs.
- [x] Remove default-journey language that implies a consumer must manually pass `--mode` or `--timestamp`.
- [x] Keep `kubera predict` as an advanced or operator-facing command, not the default consumer entrypoint.
- [x] Ensure the final terminal output clearly states the resolved prediction window, prediction date, status, and where the dashboard artifact was written if HTML export is enabled.
- [x] Ensure `kubera run` still completes the current bundled flow: bootstrap, train-if-needed, refresh, predict, backfill, terminal dashboard, and optional HTML artifact.

### Acceptance Criteria

- [x] A new user can follow the quickstart and succeed with `kubera run` alone.
- [x] The default user journey does not require reading internal timing rules before getting a result.
- [x] A successful run ends with a readable terminal summary and dashboard without requiring a second command.
- [x] The docs no longer imply that `predict` is the normal entrypoint for consumers.

### Definition of Done

- [x] README, CLI help, and checklist language consistently describe `kubera run` as the default product path.
- [x] No default-flow doc tells the user to manually choose a timestamp unless they are doing an advanced override.
- [x] The end-of-run output exposes the key outcome fields a consumer needs without additional lookup.

## Track 2 - Auto-detect and window resolution

### Goal

Replace failure-prone "current time plus inferred mode" behavior with deterministic auto-resolution so a bare `kubera run` succeeds across normal consumer usage windows.

### Tasks

- [x] Define and implement one explicit resolved-window policy for bare `kubera run`.
- [x] Resolve to the next trading day's pre-market window on non-trading days.
- [x] Resolve to the same trading day's pre-market window before market open.
- [x] Resolve to the latest completed valid scheduled window during market hours.
- [x] Resolve to the same trading day's after-close window at or after market close.
- [x] Surface whether the run used the current natural window or snapped to the latest available valid window.
- [x] Expose the resolved market session date, prediction mode, and prediction date in the terminal summary and dashboard detail.
- [x] Preserve `--timestamp` and `--mode` as advanced overrides, but do not require them in the default path.

### Acceptance Criteria

- [x] Bare `kubera run` before market open succeeds and labels the same-day pre-market window.
- [x] Bare `kubera run` during market hours succeeds and labels the latest available resolved window.
- [x] Bare `kubera run` after market close succeeds and predicts the next trading day.
- [x] Bare `kubera run` on weekends or market holidays resolves to the next trading-session behavior instead of failing for timing alone.
- [x] The product clearly tells the user what window was resolved and why.

### Definition of Done

- [x] There is one documented and testable resolution policy for pre-market, intraday, after-close, weekend, and holiday runs.
- [x] A consumer does not need to know the internal Stage 10 timing rules to get a result from `kubera run`.
- [x] Timing-related errors are limited to true invalid override cases, not the default consumer path.

## Track 3 - CLI dashboard output

### Goal

Upgrade the CLI dashboard from a mostly blended-action view into a consumer-readable result surface that shows what was predicted and how confident each model was.

### Tasks

- [x] Keep the CLI dashboard as the primary result surface for `kubera run`.
- [x] Show the final selected action and status prominently.
- [x] Show baseline raw probability and calibrated probability.
- [x] Show enhanced raw probability and calibrated probability.
- [x] Show blended raw probability and calibrated probability.
- [x] Present model outputs in a side-by-side, consumer-readable layout instead of requiring the user to infer them from separate logs.
- [x] Preserve top-driver visibility for the enhanced path.
- [x] Make terminology consistent across terminal summary and dashboard views.
- [x] Keep HTML export as a secondary artifact that mirrors the CLI dashboard content as closely as practical.

### Acceptance Criteria

- [x] A user can read the terminal alone and answer: what was predicted, what each model thought, and how confident the system was.
- [x] Baseline, enhanced, and blended probabilities are visually distinct and not collapsed into a blended-only presentation.
- [x] The dashboard still handles abstain and degraded-news cases without hiding probability information.
- [x] The CLI dashboard remains useful even if the user never opens the HTML artifact.

### Definition of Done

- [x] The default `kubera run` output contains a complete CLI dashboard with prediction, probabilities, and top drivers.
- [x] Probability labels are stable and unambiguous across latest-view and detail-view output.
- [x] The HTML artifact is additive, not required to understand the result.

## Track 4 - Linked news context

### Goal

Show the top 3 linked news items for the resolved prediction window so the consumer can see the most relevant news context behind the enhanced path.

### Tasks

- [x] Limit the default dashboard news section to the top 3 linked news items.
- [x] Rank linked items using the saved linked-news relevance ordering for the prediction window.
- [x] Show article title, source, sentiment, relevance, and a short snippet when available.
- [x] Add explicit zero-news messaging when no linked news exists for the resolved window.
- [x] Add clean handling for 1-item and 2-item cases without placeholder noise.
- [x] Use wording such as "Top linked news for this prediction window".
- [x] Explicitly avoid wording that claims these are the exact 3 articles that caused the model output.
- [x] Keep richer article lists or deeper drill-down behavior optional and secondary to the default top-3 summary.

### Acceptance Criteria

- [x] When 3 or more linked articles exist, the dashboard shows exactly 3.
- [x] When only 1 or 2 linked articles exist, the dashboard shows only those items and still renders cleanly.
- [x] When no linked articles exist, the dashboard shows an explicit zero-news state rather than an empty or confusing table.
- [x] The user can see enough context to understand why the enhanced path had news input without reading raw artifacts.
- [x] No product copy claims exact article-level attribution unless stronger attribution is implemented later.

### Definition of Done

- [x] The CLI dashboard defaults to a top-3 linked-news presentation.
- [x] Each rendered news row contains the required context fields when present in saved artifacts.
- [x] Attribution wording is product-safe and consistent across dashboard, summary, and docs.

## Track 5 - Free-tier operation

### Goal

Make the default product story work without paid news providers, while allowing optional paid-provider upgrades for users who want them.

### Tasks

- [x] Define the default product path around free-source news discovery and free-tier-compatible external inference usage.
- [x] Treat paid news providers as optional upgrades, not prerequisites for the default journey.
- [x] Keep free discovery sources in the documented default path.
- [x] Document which credentials are still required for the LLM-enhanced path in the default experience.
- [x] Ensure setup and quickstart copy never imply that Marketaux or another paid provider is mandatory.
- [x] Document free-tier operational caveats honestly, including rate limits, latency, and reduced coverage risk.
- [x] Preserve optional paid-provider configuration as an additive quality or coverage path.

### Acceptance Criteria

- [x] A user can follow the default setup without configuring a paid news provider.
- [x] Default docs clearly separate required free-tier setup from optional paid-provider upgrades.
- [x] The default checklist target does not require Marketaux, Alpha Vantage paid usage, or similar paid-provider access.
- [x] The product story remains honest that the enhanced path still depends on an external LLM path unless changed later.

### Definition of Done

- [x] The default documented product path is paid-provider-free.
- [x] Any paid-provider instructions are explicitly labeled optional.
- [x] Free-tier limits and tradeoffs are documented without weakening the one-command product story.

## Track 6 - Docs, QA, and release readiness

### Goal

Align product copy, CLI behavior, and test coverage so the consumer story is coherent and releaseable.

### Tasks

- [ ] Update README and CLI help text to match the new default consumer contract.
- [ ] Remove stale wording that says or implies a timestamp is required for the default path.
- [ ] Align terminology across CLI summary, dashboard, and docs for resolved window, prediction date, probabilities, and linked news.
- [ ] Add or update tests for auto-window resolution across the major market-time scenarios.
- [ ] Add or update dashboard rendering tests for side-by-side probability output.
- [ ] Add or update dashboard rendering tests for 0, 1, 2, and 3-plus linked-news cases.
- [ ] Add or update tests that confirm the default path works without paid-provider configuration.
- [ ] Review release notes and quickstart copy to ensure the consumer story matches actual behavior.

### Acceptance Criteria

- [ ] Docs and runtime behavior agree on what bare `kubera run` does.
- [ ] Auto-resolution behavior is covered by tests for pre-market, intraday, after-close, weekend, and holiday scenarios.
- [ ] Dashboard behavior is covered by tests for model probabilities and linked-news rendering cases.
- [ ] Default setup docs do not require paid-provider configuration.
- [ ] A release reviewer can validate the new consumer flow from docs and tests without reading source code.

### Definition of Done

- [ ] Product docs, CLI help, and tests all reflect the same default behavior.
- [ ] The release surface is coherent enough that a new user can trust the docs and get the expected result.
- [ ] Regressions in timing resolution, dashboard visibility, and linked-news rendering are test-detectable.

## Global Definition of Done

- [ ] `kubera run` is a reliable no-flag consumer entrypoint.
- [ ] The default path auto-resolves a valid prediction window instead of requiring a manual timestamp.
- [x] The CLI dashboard appears by default and shows the selected action plus baseline, enhanced, and blended probabilities.
- [x] The CLI dashboard shows the top 3 linked news items for the resolved prediction window with safe attribution wording.
- [x] The default product path does not require paid providers.
- [ ] README, CLI help, dashboard wording, and tests all describe the same product contract.

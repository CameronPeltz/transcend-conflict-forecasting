# Discrete Event Forecasting (the actual DARPA-specified task)

Standalone analysis, not yet part of the DARPA proposal draft.

## Why this exists

Reading the real program page directly
(darpa.mil/research/programs/predicting-forecasting-high-confidence)
against the current proposal draft's Part One surfaced a real gap: the
program forecasts **discrete, ACLED-style geopolitical events**
(dated, geolocated, categorized), at specific and growing lead times
(10-day at Month 6, two-week at Month 9) — not the country-week binary
escalation classifier this project has validated everywhere else,
including Criterion 2's 84% precision headline figure.

This folder tests the actual specified task for the first time: real
UCDP GED events assigned to PRIO-GRID cells (the same ~55km grid ViEWS
uses), forecast at both real named horizons, with the same
rolling-origin, never-look-ahead discipline used throughout the rest
of this project.

## What's here

- `scripts/build_discrete_event_dataset.py` — builds the real
  candidate dataset: 186,327 (grid cell, weekly issue date) rows,
  404 active cells across the 19-country set, verified PRIO-GRID
  formula (99.99% match against UCDP's own `priogrid_gid` field).
- `scripts/train_discrete_event_model.py` — the 8-config iteration
  sweep (4 feature sets × 2 horizons), same GBM+RF+logreg ensemble
  used as the "current best approach" elsewhere in this project.
- `scripts/build_visualizations_data.py` — precision-recall curves,
  false-positive-rate-vs-threshold sweep, and a naive frequency-only
  baseline comparison.
- `scripts/make_report.py` — generates the self-contained SVG charts
  (no external chart library) and assembles the final report.
- `data/discrete_event_candidates.csv` — the full derived, model-ready
  dataset (~15MB, real data, reproducible from the download script and
  the already-committed raw UCDP file).
- `data/active_cells.csv` — the 404 active grid cells with real
  centroid coordinates and total event counts, used for the map.
- `results/discrete_event_forecasting_analysis.html` — the full
  write-up: why this analysis, data & method, the 8-config sweep
  results, visualizations (map, iteration comparison, PR curves,
  false-positive-rate curve), and an honest verdict.
- `results/discrete_event_forecasting_results.json` — raw results for
  all 8 configurations.
- `results/visualization_data.json` — raw PR-curve and
  threshold-sweep data behind the charts.

## Headline result

AUC 0.83–0.84 on both real horizons; the best configuration
(cell history + spatial spillover from neighboring grid cells) beats
a naive "this cell has been busy lately" baseline by a real ~14%
relative margin in Average Precision on both horizons. Precision at
this cell/short-horizon granularity (35–43%) is well below the
country-week result (84%) — expected, since this is a genuinely harder
task, not a weaker model. Full discussion in the HTML report.

## Round 2 (tab 6 of `results/event_forecasting_writeup.html`)

All five round-1 recommendations were tried:

- **New real data, both freely downloaded, no account created**:
  UCDP GED v26.1 (`ucdp.uu.se/downloads/ged/ged261-csv.zip`, CC BY 4.0,
  verified live before download — extends real coverage to
  2025-12-31) and ACLED civilian-targeting country-month aggregates
  (`data.humdata.org/organization/acled`, HDX's open files, no login).
  Not pursued, disclosed honestly: PRIO-GRID covariates (download is a
  client-side JS interaction, not a scriptable URL, within this pass's
  budget), NASA Black Marble nighttime lights and full ACLED
  event-level data (both require creating a free third-party account —
  consistent with this project's standing position of not creating new
  accounts on the user's behalf, left for the user's own setup).
- `scripts/build_discrete_event_dataset_v2.py` /
  `train_discrete_event_model_v2.py` / `make_report_v2.py` — the
  round-2 pipeline: 228,851 candidate rows (up from 186,327), the new
  ACLED feature, frozen-threshold holdout validation (mirroring
  Criterion 2's own method exactly), per-country breakdown, and an
  event-type generalization test (train with one-sided violence
  against civilians hidden from training entirely, test whether the
  model still ranks it above real negatives).
- `data/acled_civilian_targeting_country_month.csv`,
  `data/discrete_event_candidates_v2.csv`, `data/active_cells_v2.csv`
  — round-2 derived data.
- `results/round2_results.json` — full raw round-2 results.

**Headline round-2 findings**: the new ACLED feature added essentially
nothing (a real negative result). The frozen-threshold holdout showed
real, honest degradation out of sample (50%→41% precision, 10-day) —
the opposite of Criterion 2's country-week result, traced to the
holdout window containing the real 2021 Afghanistan/Taliban regime
change. Per-country precision ranges from 0.000 (Kyrgyzstan, too few
real cells/events) to 0.47-0.53 (Ecuador). Event-type generalization
is a real positive: a model trained with one-sided violence hidden
entirely from training still ranks real held-out one-sided-violence
events above real negatives 81% of the time (AUC ~0.81).

## Round 3 (`results/round3_optimization_writeup.html`)

Tests the seven concrete levers identified after round 2, each measured
honestly, including — for the first time — retesting the 80% precision
target itself under the same frozen-threshold discipline Criterion 2
uses, not just pooled cross-validation.

- `scripts/build_round3_features.py` — adds momentum/trend features
  (30d/90d deltas, a momentum ratio), a wider spatial radius (Chebyshev
  distance 2, beyond the existing immediate-8-cell ring), and a
  severity-tightened label (>=5 real fatalities), all computed directly
  from the already-downloaded raw UCDP v26.1 events. Output:
  `data/discrete_event_candidates_v3.csv`.
- `scripts/train_discrete_event_model_v3.py` — feature sweep, frozen-
  threshold validation at an 80% target (new) and a 50% target
  (reproducibility check against round 2), and a minimum-support-floor
  comparison. Output: `results/round3_results.json`.
- `scripts/icl_hybrid_test.py` — scoped test of routing ambiguous
  holdout cases through the project's existing local-LLM reasoning
  layer. Output: `results/icl_hybrid_results.json`.

**Headline finding**: honest, frozen-threshold precision at an 80%
target is **72.3% (10-day) / 68.8% (14-day)** on the real holdout —
short of 80%, but meaningfully closer than round 2's 50%-target result
(which fell to ~41%) implied. Momentum and wider-spatial features made
no difference in pooled cross-validation but a small, real gain at the
strict operating point; a minimum-support floor (>=30 historical events
per cell) held precision flat while nearly doubling recall — the best
single lever this round. Severity-tightening the label was a clear
negative result (0% precision/recall on holdout). PRIO-GRID covariates
and a historical GDELT re-pull were investigated and are disclosed as
not pursued this round (a non-scriptable download and a data-coverage
mismatch, respectively — not ruled out on the merits). The ICL/LLM
hybrid test was initially blocked by a diagnosed environment constraint
(3.74GB of 31.15GB RAM free; local model server failed to load) — full
discussion, including next-step priorities, in the round-3 write-up.

**Follow-through, same session, once memory freed up (10.05GB
available)**: both blocked/negative levers were retried and produced
real results.

- `scripts/train_two_stage_severity_model.py` — retries severity as
  P(any event) x P(severe | an event occurs), the second stage trained
  only on the far-less-imbalanced subset of rows where an event actually
  happened. Fixes round 3's single-stage failure (0%/0% holdout
  precision/recall) into a real, working result: 100% precision at 0.1%
  recall (10-day) and 59.2% precision at 3.9% recall (14-day) on the
  frozen holdout. Output: `results/two_stage_severity_results.json`.
- `scripts/icl_hybrid_test.py`, rerun — 30 real holdout cases (10 true
  hits, 10 false alarms, 10 near-threshold): LLM-adjusted accuracy rose
  from 46.7% (ensemble alone) to 53.3%, correcting 6/10 false alarms by
  reading a falling `cell_count_30d_delta` as de-escalation, but losing
  5/10 true hits to the same heuristic where short pauses between
  offensive waves looked like settling down. Net positive on this
  sample, with a clear, addressable failure mode (see the round-3
  write-up's updated Tab 03/05). Output: `results/icl_hybrid_results.json`.
- `scripts/download_gdelt_historical_19country.py` — a real 2015-2025,
  19-country GDELT pull: 29.7 million events across 3,981 days (99.5%
  successfully retrieved). `scripts/build_gdelt_countryweek_features.py`
  aggregates it to a country-week tone/volume panel and joins it onto
  every candidate row using only the completed week strictly before its
  issue date (never-look-ahead; 228,847/228,851 rows, 100.0%, matched).
  `scripts/train_discrete_event_model_v4.py` re-runs the frozen-threshold
  80%-target sweep with it included.

  **Result: the largest real gain of any lever tested across rounds 3-4.**
  10-day holdout precision 72.3% -> **73.9%** (+1.6 points); 14-day 68.8%
  -> **71.2%** (+2.4 points), at a small recall cost. Pooled cross-
  validation barely moved (same pattern the momentum/spatial features
  showed), but the honest, frozen-threshold operating point improved
  measurably. Output: `results/round4_gdelt_results.json`.

  PRIO-GRID covariates were rechecked a second time (rendered-content
  fetch plus five direct endpoint probes) and confirmed still not
  scriptable — needs a manual download; the next recommended step is
  stacking it with GDELT rather than treating them as alternatives.

The write-up's Next Steps tab also includes a freshly researched menu of
four new free datasets not yet tried (WorldPop gridded population,
CHIRPS rainfall, NASA FIRMS active-fire data, the UNHCR displacement
API — each verified live, with real access terms, before listing) and
five structurally new modeling approaches (Hawkes/self-exciting point
processes, explicit changepoint detection for regime shifts, a
graph/hypergraph spatial model reusing the parent project's own
architecture, a joint multi-task occurrence+severity model, and feeding
GDELT context into the ICL/LLM layer), each tied to the specific
weakness it targets rather than proposed generically.

## Round 6 (`results/round6_next_steps_writeup.html`)

Executes all six next-steps items from the grand search's own Tab 05, as 24
real, frozen-threshold-backtested iterations (8.2 minutes total), plus a
dedicated discussion of the gap between this project's two forecasting tasks
and what DARPA's program page actually specifies, quoted directly.

- `scripts/round6_next_steps_search.py` — 6 groups (solo-model controls,
  stacking meta-learner, support-filter push, two-stage severity on the mega
  feature set, WorldPop, and a combined configuration decided programmatically
  from this round's own results). Output: `results/round6_results.json`,
  `results/round6_log.jsonl`.
- `scripts/build_worldpop_features.py` — patched this round to full-band-verify
  each raster before extraction (an earlier windowed-read pass had silently
  "succeeded" against Sudan's partially-downloaded file).

**Headline finding**: a stacking meta-learner (LightGBM + Random Forest +
XGBoost, logistic-regression combiner) beats the prior single-model best at
10-day (76.1% -> 78.8% unfiltered; **79.6% at a ≥100-historical-event support
filter, 25.4% recall**) but not at 14-day, where support-filtering alone
remains the whole gain (75.5% -> **78.1% at ≥100 support, 32.6% recall**).
WorldPop population, tested honestly at its real 46.1% coverage this round
(214/464 cells; the other 8 of 12 countries with active cells, including
Sudan, are still downloading), was net neutral-to-negative — a real result
against the prior round's speculation that it was the most promising untried
lever. The extended two-stage severity model reconfirmed round 3's
single-stage-fails finding on an entirely different, richer feature set, and
found "LightGBM specifically for severity" to be a real precision/recall
trade rather than a clean win.

**Escalation vs. discrete events**: DARPA's program page ("Art of Novel
Signals" SBIR, darpa.mil/research/programs/predicting-forecasting-high-confidence)
specifies forecasting "ACLED style political and conflict events" — dated,
geolocated, individual events — across the same three regions and 10-day/
two-week horizons this folder's discrete-event task targets, not the
country-week statistical-anomaly flag Criterion 2's 84.0%/56.8% headline
figure (currently in the proposal draft) answers. Full quotes and the
precision/recall/base-rate comparison in Tab 05 of the round-6 writeup.

## Round 7 (`results/round7_next_steps_writeup.html`)

Executes round 6's four remaining modeling next steps as 20 more real,
frozen-threshold-backtested iterations (6.8 minutes), then generates a fresh
next-steps list for round 8 from its own results.

- `scripts/round7_next_steps_search.py` — 7 groups: drop LightGBM from
  stacking (round 6 gave it ~zero weight), a finer support-filter sweep
  (75/100/125/150/175), stacking jointly re-optimized on the support-filtered
  population instead of filtered post-hoc, a WorldPop confound-isolation A/B
  test (clean covered-subset comparison + a coverage-indicator feature),
  two-stage severity with an averaged (not single-model) stage-1, and a final
  combined recipe decided programmatically from this round's own results.
  Output: `results/round7_results.json`, `results/round7_log.jsonl`.

**Headline findings**: LightGBM is confirmed redundant in the stacking
ensemble (2-model RF+XGB stack matches the 3-model version, 78.6% vs. 78.8%
at 10-day) -- a real simplification. Jointly re-optimizing stacking on the
support-filtered population (training AND evaluating on well-supported cells
only, not just filtering the evaluation slice) is a **clean, substantial
negative result** -- 6.9 to 14.3 points worse than round 6's train-on-
everything-then-filter approach, at both horizons and both thresholds tested;
this closes that next step. The finer support-filter sweep confirms **≥100
historical events is a real, monotonic 10-day optimum** (79.6% precision,
not a fluke of round 6's coarser grid) and shows 14-day plateaus between
≥100 and ≥150 with no further gain from pushing higher. Most importantly,
a clean covered-subset A/B test (same 214 cells, with vs. without WorldPop
population) **reverses round 6's WorldPop verdict**: the population signal
is real and positive at both horizons (+3.7 points 10-day, +0.5 14-day) --
round 6's negative full-population result was the missing-data confound,
not evidence against the underlying signal. A coverage-indicator feature
only partially recovers this gain (helps at 14-day, not 10-day), pointing
at finishing the real download rather than engineering a better patch.

## Round 8 (`results/round8_next_steps_writeup.html`)

Executes round 7's remaining next steps as 23 more real, frozen-threshold-
backtested iterations (9.3 minutes) -- **explicitly excluding further
WorldPop downloading**, per direct instruction (the downloader had already
run for hours across rounds 6-7). One free bonus: Sudan finished downloading
on its own in the background between rounds 7 and 8 (918.9 MB, verified),
lifting real WorldPop coverage to 254/464 cells (54.7%) at zero additional
active time.

- `scripts/round8_next_steps_search.py` -- 6 groups: WorldPop mean-fill
  imputation (a no-download bridge fix), the finalized 2-model-stack deploy
  recipe, a severity target-precision sweep (50/60/70%, vs. the 80% used
  through round 7), a support-filter-optimum stability check across
  additional holdout split points (50/55/65/70%, vs. the 60% used
  throughout), and a final combined recipe decided programmatically.
  Output: `results/round8_results.json`, `results/round8_log.jsonl`.

**Headline findings**: mean-fill imputation for missing WorldPop cells is a
**clean negative** -- worse than round 7's zero-fill at both horizons, and
worse than not using WorldPop at all, likely because population's
right-skewed distribution makes the covered-cell mean an implausibly high
default for what are probably mostly-rural uncovered cells. Severity
recalibration finds a first genuinely **usable operating point**: a 60%
target at 14-day delivers 40.3% precision at 11.6% recall -- roughly 8x the
recall of the 80%-target result -- while the same recalibration at 10-day
behaves unstably (actual precision doesn't track the target smoothly,
likely due to only 2,395 real positive holdout labels at that horizon). The
support-filter stability check is a genuine methodological first for this
project: re-running the ≥100 support-filter lever across 5 different
chronological holdout splits (not just the one 60% split reported
throughout) finds real variance (78-86% precision depending on the window)
but confirms the lever is robust, not a favorable-window artifact -- the
standard 60% split this project has always reported turns out to be on the
conservative end of that range, not the favorable end.

## Round 9 (`results/round9_next_steps_writeup.html`)

Executes round 8's next steps as 38 more real, frozen-threshold-backtested
iterations (13.3 minutes) -- the most diagnostic round yet rather than a
gains round: three open mysteries get resolved with real mechanisms, the
production recipe stays unchanged.

- `scripts/round9_next_steps_search.py` -- 7 groups: a fine-grained severity
  target-precision sweep with the select-window's own chosen threshold
  reported alongside each result (the diagnostic itself), 2 more support-filter
  stability split points, a clean 3-way WorldPop ablation (population alone /
  history alone / combined) plus a direct correlation check, finalized
  severity deploy configs, a multi-split ensemble (5 models trained through
  different chronological cutoffs, averaged on one common evaluation window),
  and a final combined recipe. Output: `results/round9_results.json`,
  `results/round9_log.jsonl`.

**Headline findings**: 10-day severity's threshold instability (open since
round 8) is now explained, not just reproduced -- the select-window's own
chosen threshold rises smoothly from 0.481 to 0.650 as the target goes from
45% to 60%, then **jumps straight to 0.952 and stays there** for 65% and
70%, revealing a real gap in the score distribution: 10-day severity has
exactly two operating regimes (noisy-low, ~25-35% precision, or
rare-but-reliable, 93.3%/0.6%), not a smooth dial the way 14-day is.
**Multi-split ensembling is a clean negative**: averaging 5 models trained
through different chronological cutoffs, evaluated on one common holdout
window, ties (not beats) a single model trained at the standard split
(83.7% vs. 84.0% at 10-day; 79.3% vs. 79.4% at 14-day) -- the real variance
this project has documented in the support-filter lever reflects genuine
period-to-period differences in the real data, not model instability
averaging could fix. **WorldPop is confirmed genuinely orthogonal, not
redundant** (correlation with existing features r = -0.05 to 0.07) but is
simply a weak standalone predictor (AP 0.09-0.12 vs. 0.39-0.48 for event
history) -- its marginal contribution is a real wash, not evidence of a
confound or a bug, closing three rounds of investigation with a stable
characterization. The support-filter range, now measured across 7
chronological holdout windows, is 75.1-86.3% precision at 10-day and
77.4-84.3% at 14-day -- never bad, genuinely variable, and not a
favorable-window artifact of the standard split this project has always
reported.

## Data note

`data_raw/` (the large raw source files, ~16GB once the WorldPop rasters
and the full GDELT historical pull are downloaded) is gitignored —
regenerate it from the real, verified download URLs in the round-2 and
round-4 build scripts. Within `data/`, the `discrete_event_candidates*.csv`
family (round 1 through the round-6 `_mega` version, 14–106MB — the mega
version alone exceeds GitHub's 100MB hard limit) is also gitignored as of
round 9 and regenerated by `scripts/build_mega_dataset.py` and its
predecessors; this was true from round 6 onward and this note was stale
until round 9 caught it. The small reference tables (`active_cells*.csv`,
`worldpop_cell_population.csv`, the CHIRPS/GDELT/ACLED country-week panels,
all under 5MB) remain committed directly.

# Transcend — Conflict Early-Warning Forecasting

Prototype and evaluation harness for **Transcend**, a conflict early-warning forecasting pipeline built in response to DARPA topic **DPA26BZ04-DV015, "Art of Novel Signals"** (Direct-to-Phase-II SBIR). Developed by **AP Peace Strategies**, with **Rootwise** (radio listening data aggregation) and international evidence-gathering coordination support.

This repository is the evidence base for the Phase II proposal: every number cited in the proposal traces back to a script and a dataset in this repo. Nothing here is simulated — every reported metric comes from a real, out-of-sample rolling-origin backtest against real, publicly sourced conflict-event data.

## What this is

A real-data prototype and a large (1,150+ real backtested configuration) search for the best way to forecast material-conflict escalation one to eight weeks ahead, at country-week granularity, from open-source event data — built as groundwork for the full multilingual radio/ASR (**automatic speech recognition**, converting spoken audio to text) system DARPA's topic actually funds.

## Three data tracks, kept strictly separate

| Track | Source | Scope | Country-weeks | Positives |
|---|---|---|---|---|
| A — original | GDELT 2.0 (**Global Database of Events, Language, and Tone** — a free, public, near-real-time feed of coded political/conflict events extracted from world news) | 6 countries, 180 days | 161 | ~19 |
| B — large scraped | GDELT 2.0, self-scraped this round | 19 countries (3 DARPA-named regions + 2 disclosed extras), 3 years | 2,989 | 358 |
| C — pure UCDP | UCDP GED (**Uppsala Conflict Data Program, Georeferenced Event Dataset** — the real, hand-curated, fatality-coded academic conflict dataset most secondary "conflict prediction" datasets are built from) v25.1 | 19-country subset of the 124-country global release | 29,038 | 4,814 |

Track C is never merged with the GDELT-derived tracks — it exists specifically so results can always be checked against a "pure," independently-produced ground truth, not just an internally-defined proxy label.

## Methods implemented

- **In-context-learning forecaster over a temporal knowledge graph** (`scripts/models.py`), the literal mechanism DARPA's own DP2 evaluation criteria name: retrieval of historical analogs + a language-model call, no gradient training of the forecasting mechanism itself at prediction time. Runs against a free, local, open-source model (Ollama, llama3.1) — no API key, no per-call cost.
- **Hypergraph neural network** (`scripts/hypergraph_model.py`, `hypergraphs_research/`), built from scratch in numpy/scipy and separately validated with the `xgi` hypergraph-network-science library. Hyperedges connect country-weeks by shared country, shared region-week, shared calendar week (global shocks), shared real actors, and shared real conflict/dyad identity.
- **Graph-based semi-supervised label spreading** (`scripts/graph_nlp_features.py`).
- **Graph-based NLP**: a real theme co-occurrence hypergraph built from GDELT's Global Knowledge Graph (**GKG**, GDELT's broader news-theme-tagging layer), embedded via PPMI (**Positive Pointwise Mutual Information**, a standard word/tag-embedding weighting) + truncated SVD (**Singular Value Decomposition**, a matrix factorization method — same embedding family as GloVe/LSA).
- Gradient-boosted trees, random forests, logistic regression, and ensembles as labeled comparison points, all under identical rolling-origin backtesting.

## Key results

- Best real, non-trivial-sample result: **AP 0.871, ROC-AUC 0.932, precision 0.687, recall 0.874** (ensemble, Track C, n=1,245 pooled predictions, 341 real positives, standard 0.5 threshold).
- A temporally out-of-sample threshold-selection test (`scripts/precision_threshold_validation.py`) — see `results_v2/precision_threshold_validation.json` for the full, honest result including the precision/recall tradeoff actually achieved on a strictly later, untouched holdout window.
- Full metrics suite (accuracy, precision, recall, specificity, F1, Brier score, average precision, ROC-AUC, log-loss, MCC) reported for every one of 1,150+ real iterations — see `results_v2/iteration-search-log-v2.html`.

See `results_v2/final-summary-and-case-studies.html` for the full write-up, ranked results, real case studies with genuine LLM reasoning traces, and a plain-English glossary of every metric used.

## Honest limitations (stated plainly, per DARPA's own drafting guidance)

- Metrics are real and reproducible from this repo's own code and data, computed under a rigorous rolling-origin (never-look-ahead) discipline — but have not been audited by an independent third party. Track C's ground truth (UCDP) is independently produced (Uppsala University, not the proposer), which is a real form of label independence; that is distinct from an independent audit of the metrics computation itself, and this README does not conflate the two.
- Multilingual audio ingestion (ASR, radio collection, synthetic-data augmentation) is Phase II future work, not yet built. GDELT's own multilingual source coverage (translated via GDELT Translingual) is used as-is; no original ASR pipeline exists in this repo yet.
- Sample sizes vary a great deal by track and horizon; every ranking in `results_v2/` reports n and n_pos alongside the score for exactly this reason.

## Repository layout

```
scripts/            All data-download, feature-engineering, modeling, and evaluation code
data/                Country-week panels and derived features (large raw event files gitignored; see scripts/download_*.py to regenerate)
results_v2/          Full 1,150-iteration search log, final summary, case studies, dangerous-ideas log, precision validation
hypergraphs_research/  xgi-based hypergraph ablations (edge-type, propagation-mode)
```

## Reproducing a result

```
pip install -r requirements.txt   # pandas, numpy, scikit-learn, xgboost, scipy, xgi
python scripts/download_gdelt_large.py 1095
python scripts/build_ucdp_panel.py
python scripts/grand_search_v2.py 1150
python scripts/render_search_log_v2.py
```

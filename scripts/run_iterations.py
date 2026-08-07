"""
24 real, distinct modeling iterations, each a genuine hypothesis about
what might improve conflict-escalation forecasting on the real 6-country,
180-day GDELT panel -- varying model family/hyperparameters, ensembling,
event-grouping taxonomy (quad-class vs. CAMEO-root-category severity),
feature ablations, time granularity, lag depth, label sensitivity,
forecast horizon (out to 8 weeks -- genuinely "months out"), and
probability calibration. Every number logged below comes from an
actual rolling-origin backtest run against real data.
"""
import json
import time

from iteration_engine import (
    build_panel, FEATURE_SETS, fit_predict, ensemble_predict, run_backtest,
)

print("building panel variants...")
t0 = time.time()
panel_default = build_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8))
panel_biweekly = build_panel(granularity="2W", n_lags=2, label_z=1.0, horizons=(1,))
panel_3lag = build_panel(granularity="W", n_lags=3, label_z=1.0, horizons=(1,))
panel_z075 = build_panel(granularity="W", n_lags=2, label_z=0.75, horizons=(1,))
panel_z15 = build_panel(granularity="W", n_lags=2, label_z=1.5, horizons=(1,))
print(f"  done in {time.time()-t0:.1f}s")

results = []


def log(iter_num, name, hypothesis, panel, feature_set_name, label_col, predictor_fn, min_train=6):
    features = FEATURE_SETS[feature_set_name]
    t = time.time()
    r = run_backtest(panel, features, label_col, predictor_fn, min_train)
    r.update({"iter": iter_num, "name": name, "hypothesis": hypothesis,
              "feature_set": feature_set_name, "label_col": label_col,
              "runtime_s": round(time.time() - t, 2)})
    results.append(r)
    print(f"[{iter_num:2d}] {name:42s} AP={r['ap']:.4f} Brier={r['brier']:.4f} "
          f"Acc={r['accuracy_05']:.3f} P={r['precision_05']:.3f} R={r['recall_05']:.3f} "
          f"n={r['n']} pos={r['n_pos']} folds={r['n_folds']} ({r['runtime_s']}s)")


# --- 1: baseline ---
log(1, "Baseline: GBM, core features, weekly, 1wk", "The starting point every later iteration is measured against.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1"))

# --- 2/3: country / region identity ---
log(2, "+ country identity (categorical)", "GBM never sees WHICH country it's scoring -- giving it a country flag should let it learn country-specific baselines the way the historical-average model does.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1", extra_cat_cols=["country"]))

log(3, "+ region identity (categorical)", "A coarser version of #2 -- does grouping by region instead of country still help, with less risk of overfitting to only 26 weeks per country?",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1", extra_cat_cols=["region"]))

# --- 4/5: tree hyperparameters ---
log(4, "GBM, deeper trees (max_depth=5)", "More capacity to learn feature interactions -- worth checking whether that helps or just overfits a small dataset.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_deep", tr, te, FEATURE_SETS["core"], "label_quad_1"))

log(5, "GBM, shallow + regularized", "The opposite bet -- more trees, shallower, stronger L2 penalty, on the theory that this dataset is too small for a complex model.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_shallow_reg", tr, te, FEATURE_SETS["core"], "label_quad_1"))

# --- 6: different model family ---
log(6, "Random Forest (bagged trees)", "A structurally different ensemble method -- bagging instead of boosting -- as a real second opinion on the same features.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("random_forest", tr, te, FEATURE_SETS["core"], "label_quad_1"))

# --- 7: full-feature logistic regression ---
log(7, "Logistic regression, full feature set", "Earlier rounds only gave logistic regression 3 raw lagged counts. Giving it the same full feature set as GBM is a fairer comparison.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("logreg", tr, te, FEATURE_SETS["core"], "label_quad_1"))

# --- 8/9/10: ensembles ---
log(8, "Ensemble: GBM + logistic regression", "Averaging a nonlinear and a linear model's outputs sometimes cancels out each one's individual blind spots.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: ensemble_predict(["gbm_default", "logreg"], tr, te, FEATURE_SETS["core"], "label_quad_1"))

log(9, "Ensemble: GBM + Random Forest", "Two different tree ensembles averaged -- tests whether boosting and bagging make genuinely different errors worth combining.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: ensemble_predict(["gbm_default", "random_forest"], tr, te, FEATURE_SETS["core"], "label_quad_1"))

log(10, "Ensemble: GBM + RF + logistic regression (3-way)", "The full-diversity ensemble -- boosting, bagging, and a linear model, averaged equally.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: ensemble_predict(["gbm_default", "random_forest", "logreg"], tr, te, FEATURE_SETS["core"], "label_quad_1"))

# --- 11/12: event-grouping taxonomy ---
log(11, "Event grouping by CAMEO root-category severity", "Instead of the 4-bucket quad-class split, group events by real CAMEO root categories into 'severe' (assault/fight/mass violence) and 'coerce-or-protest' shares -- both the FEATURES and the escalation LABEL itself are redefined around this taxonomy.",
    panel_default, "root_taxonomy", "label_root_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["root_taxonomy"], "label_root_1"))

log(12, "Root-category features, quad-class label", "Isolates whether the root-category FEATURES alone help, keeping the original quad-class escalation label fixed so the comparison to the baseline is apples-to-apples on the same target.",
    panel_default, "root_taxonomy", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["root_taxonomy"], "label_quad_1"))

# --- 13/14/15: feature ablations ---
log(13, "Tone-only features", "News-coverage hostility alone, nothing about actual event counts or severity -- tests how much signal is in coverage tone by itself.",
    panel_default, "tone_only", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["tone_only"], "label_quad_1"))

log(14, "Goldstein/conflict-share-only features", "Just the event-severity scale and conflict share, no tone, no actor counts, no raw volume.",
    panel_default, "goldstein_only", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["goldstein_only"], "label_quad_1"))

log(15, "Volume-only features", "Raw event counts and their trend, nothing about what KIND of events they are -- the crudest possible real baseline feature set.",
    panel_default, "volume_only", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["volume_only"], "label_quad_1"))

# --- 16: kitchen sink ---
log(16, "Kitchen sink: all features + country", "Every engineered feature from every family above, plus country identity -- tests whether more is simply better once nothing is held back.",
    panel_default, "kitchen_sink", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["kitchen_sink"], "label_quad_1", extra_cat_cols=["country"]))

# --- 17: coarser time granularity ---
log(17, "Biweekly aggregation instead of weekly", "Coarser time buckets trade temporal resolution for less noisy per-period statistics -- worth testing directly rather than assuming weekly is the right grain.",
    panel_biweekly, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1"), min_train=4)

# --- 18: deeper lag history ---
log(18, "3-week lag history instead of 2", "Gives the model one more week of trailing history per feature -- tests whether the extra lag adds real signal or just noise given how little training data there is.",
    panel_3lag, "core3lag", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core3lag"], "label_quad_1"))

# --- 19/20: label sensitivity ---
log(19, "More sensitive escalation label (z > 0.75)", "A looser definition of 'escalation' catches more real events but also more borderline ones -- tests the precision/recall tradeoff of the label definition itself, not just the model.",
    panel_z075, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1"))

log(20, "Stricter escalation label (z > 1.5)", "The opposite bet -- only the clearest, sharpest spikes count as 'escalation'. Fewer positives, but each one should be a more unambiguous real event.",
    panel_z15, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1"))

# --- 21/22: longer horizons -- genuinely "weeks or months out" ---
log(21, "4-week-ahead forecast (about a month out)", "Same features, same model, but predicting a full month ahead instead of one week -- the real test of whether this feature set carries any signal at longer lead times.",
    panel_default, "core", "label_quad_4",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_4"))

log(22, "8-week-ahead forecast (about two months out)", "Pushing further still -- the outer edge of what this dataset can even be tested on given only ~26 weeks per country.",
    panel_default, "core", "label_quad_8",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_8"))

# --- 23: calibration ---
log(23, "Calibrated GBM (Platt scaling)", "Re-scales the model's raw output so predicted probabilities better match real-world frequencies -- doesn't change rankings, tests whether it improves Brier score specifically, the metric the expert panel flagged as most important for real deployment.",
    panel_default, "core", "label_quad_1",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["core"], "label_quad_1", calibrate=True))

# --- 24: best-feature-set x longer-horizon combination ---
log(24, "Kitchen sink + country, 4-week horizon", "Combines the two individually-promising ideas from #16 and #21 -- the richest feature set, tested at the longer, more operationally useful horizon.",
    panel_default, "kitchen_sink", "label_quad_4",
    lambda tr, te: fit_predict("gbm_default", tr, te, FEATURE_SETS["kitchen_sink"], "label_quad_4", extra_cat_cols=["country"]))


with open("data/iteration_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nwrote data/iteration_results.json ({len(results)} iterations)")

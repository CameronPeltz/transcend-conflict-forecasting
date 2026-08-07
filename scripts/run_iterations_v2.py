"""
Second real iteration round: does adding the four new external data
families (climate, food prices, structural/WDI, election proximity, GKG
news signal) to the existing GDELT-derived feature set actually improve
real backtested prediction -- regardless of what the Granger-style tests
in granger_tests.py found? Predictive usefulness in a nonlinear model and
marginal linear Granger-significance are different questions, so both are
worth checking empirically rather than assuming one implies the other.

Same real rolling-origin backtest discipline, same GBM default model
config, same label (label_quad_1) and min_train as iteration #1/#2 in
run_iterations.py, so every row here is a fair, apples-to-apples,
directly comparable real result -- not a new panel or new label
definition confounding the comparison.
"""
import sys
sys.path.insert(0, "scripts")
import json
import time

from iteration_engine import build_panel, FEATURE_SETS, fit_predict, run_backtest
from external_features import attach_external_features, EXTERNAL_FEATURE_SETS

print("building panel + attaching real external features...")
panel = build_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1,))
ext = attach_external_features(panel)

CORE = FEATURE_SETS["core"]

COMBOS = {
    "core_plus_climate": CORE + EXTERNAL_FEATURE_SETS["climate_only"],
    "core_plus_food": CORE + EXTERNAL_FEATURE_SETS["food_price_only"],
    "core_plus_structural": CORE + EXTERNAL_FEATURE_SETS["structural_only"],
    "core_plus_election": CORE + EXTERNAL_FEATURE_SETS["election_only"],
    "core_plus_gkg": CORE + EXTERNAL_FEATURE_SETS["gkg_only"],
    "core_plus_all_external": CORE + EXTERNAL_FEATURE_SETS["all_external"],
    "external_only_no_gdelt": (EXTERNAL_FEATURE_SETS["climate_only"] + EXTERNAL_FEATURE_SETS["food_price_only"]
                                + EXTERNAL_FEATURE_SETS["gkg_only"] + EXTERNAL_FEATURE_SETS["election_only"]),
}

results = []
for name, features in COMBOS.items():
    t0 = time.time()
    r = run_backtest(ext, features, "label_quad_1",
                      lambda tr, te, f=features: fit_predict("gbm_default", tr, te, f, "label_quad_1"))
    r.update({"name": name, "n_features": len(features), "runtime_s": round(time.time() - t0, 2)})
    results.append(r)
    print(f"{name:28s} AP={r['ap']:.4f} Brier={r['brier']:.4f} Acc={r['accuracy_05']:.3f} "
          f"P={r['precision_05']:.3f} R={r['recall_05']:.3f} n={r['n']} pos={r['n_pos']} folds={r['n_folds']}")

# best-known GDELT-only comparison points, reproduced on THIS exact panel/fold
# structure (ext, not the original panel) so the comparison is apples-to-apples
baseline = run_backtest(ext, CORE, "label_quad_1",
                         lambda tr, te: fit_predict("gbm_default", tr, te, CORE, "label_quad_1"))
baseline.update({"name": "core_baseline_reproduced", "n_features": len(CORE)})
results.insert(0, baseline)
print(f"\n{'core_baseline_reproduced':28s} AP={baseline['ap']:.4f} Brier={baseline['brier']:.4f} "
      f"Acc={baseline['accuracy_05']:.3f} n={baseline['n']} pos={baseline['n_pos']}")

with open("data/external_iteration_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nwrote data/external_iteration_results.json")

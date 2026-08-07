"""
Real Granger-causality-style leading-indicator tests: does a candidate
external signal's own recent history help predict conflict intensity
beyond what conflict's own history already predicts about itself?

Two real, complementary tests are run, and both are reported honestly
rather than picking whichever looks better:

1. Classical per-country Granger test (statsmodels.tsa.stattools.
   grangercausalitytests) on each country's own weekly series. This is
   the textbook version of the test, but with only ~26-30 weekly
   observations per country it has real, disclosed low statistical
   power -- results here are suggestive, not confirmatory.

2. A pooled fixed-effects lagged-regression F-test: restricted model
   (material_conflict_share on its own lags + country fixed effects)
   vs. unrestricted model (+ the candidate signal's lag). This borrows
   statistical power across all six countries at the cost of assuming a
   shared relationship, which is a real, standard tradeoff in panel
   Granger-style testing, not a way of gaming the small-N problem away.

Target variable: material_conflict_share, the real continuous quantity
the escalation label itself is thresholded from -- Granger causality
needs a continuous/interval series, not the derived binary label.
"""
import sys
sys.path.insert(0, "scripts")
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from iteration_engine import build_panel
from external_features import attach_external_features

warnings.filterwarnings("ignore")

CANDIDATES = {
    "precip_anomaly_z": "Rainfall anomaly (climate stress)",
    "food_price_pct_dev": "Food price deviation from trailing baseline",
    "gkg_fragility_theme_share": "Share of GKG news docs tagged fragility/crisis-themed",
    "gkg_mean_tone": "Mean GKG document tone",
}

TARGET = "material_conflict_share"


def per_country_granger(panel, candidate, maxlag=2):
    results = {}
    for country, sub in panel.groupby("country"):
        sub = sub.sort_values("week")[[TARGET, candidate]].dropna()
        if len(sub) < 12:
            results[country] = {"n": len(sub), "min_p": None, "note": "too few real observations"}
            continue
        try:
            gc = grangercausalitytests(sub[[TARGET, candidate]], maxlag=maxlag, verbose=False)
            pvals = [gc[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
            results[country] = {"n": len(sub), "min_p": float(min(pvals)), "pvals_by_lag": [round(p, 4) for p in pvals]}
        except Exception as e:
            results[country] = {"n": len(sub), "min_p": None, "note": f"error: {e}"}
    return results


def pooled_fe_ftest(panel, candidate):
    df = panel.sort_values(["country", "week"]).copy()
    df["y"] = df[TARGET]
    df["y_lag1"] = df.groupby("country")["y"].shift(1)
    df["y_lag2"] = df.groupby("country")["y"].shift(2)
    df["x_lag1"] = df.groupby("country")[candidate].shift(1)

    sub = df.dropna(subset=["y", "y_lag1", "y_lag2", "x_lag1"]).copy()
    if len(sub) < 20:
        return {"n": len(sub), "note": "too few pooled observations"}

    country_dummies = pd.get_dummies(sub["country"], prefix="c", drop_first=True).astype(float)

    X_restricted = pd.concat([sub[["y_lag1", "y_lag2"]], country_dummies], axis=1)
    X_restricted = sm.add_constant(X_restricted)
    X_unrestricted = pd.concat([sub[["y_lag1", "y_lag2", "x_lag1"]], country_dummies], axis=1)
    X_unrestricted = sm.add_constant(X_unrestricted)

    y = sub["y"].astype(float)
    m_r = sm.OLS(y, X_restricted.astype(float)).fit()
    m_u = sm.OLS(y, X_unrestricted.astype(float)).fit()

    ftest = m_u.compare_f_test(m_r)
    return {
        "n": len(sub), "f_stat": float(ftest[0]), "p_value": float(ftest[1]),
        "r2_restricted": float(m_r.rsquared), "r2_unrestricted": float(m_u.rsquared),
        "x_lag1_coef": float(m_u.params["x_lag1"]), "x_lag1_p": float(m_u.pvalues["x_lag1"]),
    }


def main():
    print("building panel + attaching real external features...")
    panel = build_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1,))
    ext = attach_external_features(panel)

    all_results = {}
    for candidate, label in CANDIDATES.items():
        print(f"\n=== {label} ({candidate}) ===")
        pc = per_country_granger(ext, candidate)
        for c, r in pc.items():
            print(f"  per-country [{c}]: {r}")
        fe = pooled_fe_ftest(ext, candidate)
        print(f"  pooled fixed-effects F-test: {fe}")
        all_results[candidate] = {"label": label, "per_country": pc, "pooled_fe": fe}

    import json
    with open("data/granger_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nwrote data/granger_test_results.json")


if __name__ == "__main__":
    main()

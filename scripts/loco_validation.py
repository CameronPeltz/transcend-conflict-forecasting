"""
DARPA proposal Criterion 4: "Generalization to unseen events and
regions." Every backtest run anywhere else in this project so far tests
generalization across TIME within COUNTRIES THE MODEL HAS ALREADY SEEN
(rolling-origin: same 6/19 countries throughout, later weeks held out).
That is a real and necessary test, but it is not a test of the harder,
more DARPA-relevant question: does the model still work in a country it
has never seen a single training example from?

This script runs real leave-one-country-out (LOCO) cross-validation:
for each country in a track, train on every OTHER country's full
history, test only on the held-out country, pool results. Country
identity (the single strongest lever found everywhere else in this
project) is deliberately NOT used as a feature here -- a held-out
country has no trained coefficient/dummy column to fall back on, so
including it would be meaningless at best and misleading at worst.
Only signal that could plausibly transfer to a genuinely new country
(event volume, conflict-share trends, actor/dyad structure, real
fatality counts) is used.

Expect LOCO performance to be lower than the rolling-origin numbers
reported everywhere else in this project -- that gap IS the finding,
and it's reported honestly rather than the harder number being omitted.
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (brier_score_loss, average_precision_score, roc_auc_score,
                              matthews_corrcoef, f1_score)
from xgboost import XGBClassifier

import build_ucdp_panel as up
import large_panel as lp


def loco_fold(panel, feature_cols, label_col, country_col="country"):
    countries = sorted(panel[country_col].unique())
    all_probs, all_labels, all_countries = [], [], []
    per_country = {}

    for held_out in countries:
        train = panel[panel[country_col] != held_out].dropna(subset=feature_cols + [label_col])
        test = panel[panel[country_col] == held_out].dropna(subset=feature_cols + [label_col])
        if len(train) < 30 or len(test) < 5 or test[label_col].nunique() < 2:
            continue

        X_train = train[feature_cols].fillna(0)
        y_train = train[label_col].astype(int)
        X_test = test[feature_cols].fillna(0)
        y_test = test[label_col].astype(int).values

        pos = max(1, y_train.sum()); neg = max(1, len(y_train) - pos)
        model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                               scale_pos_weight=neg / pos, eval_metric="logloss", random_state=0)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        all_probs.extend(probs.tolist())
        all_labels.extend(y_test.tolist())
        all_countries.extend([held_out] * len(y_test))

        p_clip = np.clip(probs, 1e-6, 1 - 1e-6)
        try:
            country_ap = float(average_precision_score(y_test, p_clip)) if y_test.sum() > 0 else None
        except ValueError:
            country_ap = None
        per_country[held_out] = {
            "n": int(len(y_test)), "n_pos": int(y_test.sum()),
            "ap": country_ap,
            "brier": float(brier_score_loss(y_test, p_clip)),
        }

    y = np.array(all_labels)
    p = np.clip(np.array(all_probs), 1e-6, 1 - 1e-6)
    pred05 = (p >= 0.5).astype(int)
    tp = int(((pred05 == 1) & (y == 1)).sum()); fp = int(((pred05 == 1) & (y == 0)).sum())
    fn = int(((pred05 == 0) & (y == 1)).sum()); tn = int(((pred05 == 0) & (y == 0)).sum())
    overall = {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp), "accuracy": (tp + tn) / max(1, len(y)),
        "f1": float(f1_score(y, pred05, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ap": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "mcc": float(matthews_corrcoef(y, pred05)),
        "n_countries_tested": len(per_country),
    }
    return overall, per_country


def main():
    out = {}

    print("LOCO, Track C (pure UCDP)...")
    panel_c, _ = up.build_panel()
    overall_c, by_country_c = loco_fold(panel_c, up.UCDP_FEATURE_SET, "label_1")
    out["C_pure_ucdp"] = {"overall": overall_c, "by_country": by_country_c}
    print(f"  overall: AP={overall_c['ap']:.3f} precision={overall_c['precision']:.3f} "
          f"recall={overall_c['recall']:.3f} n={overall_c['n']} pos={overall_c['n_pos']} "
          f"across {overall_c['n_countries_tested']} held-out countries")

    print("LOCO, Track B (large scraped GDELT)...")
    raw_b = lp.load_raw()
    panel_b = lp.build_panel(raw_df=raw_b)
    overall_b, by_country_b = loco_fold(panel_b, lp.FEATURE_SETS["core"], "label_quad_1")
    out["B_large_scraped_gdelt"] = {"overall": overall_b, "by_country": by_country_b}
    print(f"  overall: AP={overall_b['ap']:.3f} precision={overall_b['precision']:.3f} "
          f"recall={overall_b['recall']:.3f} n={overall_b['n']} pos={overall_b['n_pos']} "
          f"across {overall_b['n_countries_tested']} held-out countries")

    with open("results_v2/loco_generalization_validation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nDONE. Wrote results_v2/loco_generalization_validation.json")


if __name__ == "__main__":
    main()

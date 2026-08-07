"""
Shared engine for the 20+ real iteration experiments: flexible panel
construction (time granularity, event-grouping taxonomy, lag depth,
label sensitivity, label horizon) and a flexible model factory (tree
depth/regularization variants, Random Forest, full-feature logistic
regression, ensembles, calibrated GBM), run through the same real
rolling-origin backtest discipline used in every prior document.

Nothing here is simulated -- every iteration in run_iterations.py
calls into this file and produces real numbers from the real 233,552-
event GDELT dataset already downloaded for the six-country panel.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, average_precision_score
from xgboost import XGBClassifier

TARGET_COUNTRIES = ["SU", "ET", "AF", "BM", "CO", "VE"]
COUNTRY_REGION = {
    "SU": "East/NE Africa", "ET": "East/NE Africa",
    "AF": "Central/SE Asia", "BM": "Central/SE Asia",
    "CO": "South America", "VE": "South America",
}
# real CAMEO root-category numbers (01-20); 18/19/20 = assault, fight,
# mass violence (the most severe root categories); 14/17 = protest, coerce
SEVERE_ROOTS = {"18", "19", "20"}
COERCE_PROTEST_ROOTS = {"14", "17"}

RNG_SEED = 42


def load_raw():
    df = pd.read_csv("data/gdelt_filtered_v2.csv", dtype={"ActionGeo_CountryCode": str, "EventCode": str})
    df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
    df = df[df["ActionGeo_CountryCode"].isin(TARGET_COUNTRIES)]
    df = df[df["date"] >= pd.Timestamp("2026-02-01")]  # same disclosed cleanup as prior rounds
    df["QuadClass"] = pd.to_numeric(df["QuadClass"], errors="coerce")
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")
    df["root"] = df["EventCode"].astype(str).str[:2]
    return df


def build_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8)):
    """granularity: 'W' (weekly) or '2W' (biweekly, via pandas period alias trick)."""
    df = load_raw()
    freq = "W-SUN" if granularity == "W" else "2W-SUN"
    df["period"] = df["date"].dt.to_period(freq).dt.start_time

    rows = []
    for country in TARGET_COUNTRIES:
        sub = df[df.ActionGeo_CountryCode == country]
        g = sub.groupby("period")
        weekly = g.agg(
            n_events=("GlobalEventID", "count"),
            n_material_conflict=("QuadClass", lambda s: (s == 4).sum()),
            n_verbal_conflict=("QuadClass", lambda s: (s == 3).sum()),
            n_severe_root=("root", lambda s: s.isin(SEVERE_ROOTS).sum()),
            n_coerce_protest_root=("root", lambda s: s.isin(COERCE_PROTEST_ROOTS).sum()),
            mean_goldstein=("GoldsteinScale", "mean"),
            mean_tone=("AvgTone", "mean"),
            tone_neg_share=("AvgTone", lambda s: (s < 0).mean()),
            total_mentions=("NumMentions", "sum"),
        ).reset_index().rename(columns={"period": "week"})
        actors = pd.concat([sub["Actor1Code"], sub["Actor2Code"]]).dropna()
        actor_counts = sub.assign(_a1=sub["Actor1Code"], _a2=sub["Actor2Code"]).groupby("period").apply(
            lambda d: pd.concat([d["Actor1Code"], d["Actor2Code"]]).dropna().nunique(), include_groups=False)
        weekly["distinct_actors"] = weekly["week"].map(actor_counts).fillna(0)
        weekly["country"] = country
        weekly["region"] = COUNTRY_REGION[country]
        rows.append(weekly)

    panel = pd.concat(rows, ignore_index=True).sort_values(["country", "week"]).reset_index(drop=True)
    panel["material_conflict_share"] = panel["n_material_conflict"] / panel["n_events"].clip(lower=1)
    panel["severe_root_share"] = panel["n_severe_root"] / panel["n_events"].clip(lower=1)
    panel["coerce_protest_share"] = panel["n_coerce_protest_root"] / panel["n_events"].clip(lower=1)

    def add_baseline(col):
        panel[f"baseline_mean_{col}"] = panel.groupby("country")[col].transform(
            lambda s: s.expanding(min_periods=4).mean().shift(1))
        panel[f"baseline_std_{col}"] = panel.groupby("country")[col].transform(
            lambda s: s.expanding(min_periods=4).std().shift(1))

    for col in ["material_conflict_share", "severe_root_share", "mean_goldstein"]:
        add_baseline(col)

    z_quad = (panel["material_conflict_share"] - panel["baseline_mean_material_conflict_share"]) / panel["baseline_std_material_conflict_share"].replace(0, np.nan)
    z_root = (panel["severe_root_share"] - panel["baseline_mean_severe_root_share"]) / panel["baseline_std_severe_root_share"].replace(0, np.nan)
    goldstein_drop = panel["baseline_mean_mean_goldstein"] - panel["mean_goldstein"]

    panel["escalation_quad"] = ((z_quad > label_z) | (goldstein_drop > 2.0)).astype("Int64")
    panel["escalation_root"] = ((z_root > label_z) | (goldstein_drop > 2.0)).astype("Int64")

    for h in horizons:
        panel[f"label_quad_{h}"] = panel.groupby("country")["escalation_quad"].shift(-h)
        panel[f"label_root_{h}"] = panel.groupby("country")["escalation_root"].shift(-h)

    lag_cols = ["n_events", "material_conflict_share", "severe_root_share", "coerce_protest_share",
                "mean_goldstein", "distinct_actors", "mean_tone", "tone_neg_share"]
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            panel[f"{col}_lag{lag}"] = panel.groupby("country")[col].shift(lag)
        panel[f"{col}_delta"] = panel[col] - panel[f"{col}_lag1"]

    return panel


FEATURE_SETS = {
    "core": ["n_events_lag1", "n_events_lag2", "n_events_delta",
             "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_delta",
             "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
             "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_delta",
             "mean_tone_lag1", "mean_tone_delta"],
    "core3lag": ["n_events_lag1", "n_events_lag2", "n_events_lag3", "n_events_delta",
                 "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_lag3", "material_conflict_share_delta",
                 "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_lag3", "mean_goldstein_delta",
                 "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_lag3", "distinct_actors_delta",
                 "mean_tone_lag1", "mean_tone_delta"],
    "root_taxonomy": ["n_events_lag1", "n_events_lag2", "n_events_delta",
                       "severe_root_share_lag1", "severe_root_share_lag2", "severe_root_share_delta",
                       "coerce_protest_share_lag1", "coerce_protest_share_lag2", "coerce_protest_share_delta",
                       "mean_goldstein_lag1", "mean_goldstein_delta",
                       "distinct_actors_lag1", "distinct_actors_delta",
                       "mean_tone_lag1", "mean_tone_delta"],
    "tone_only": ["mean_tone_lag1", "mean_tone_lag2", "mean_tone_delta", "tone_neg_share_lag1", "tone_neg_share_delta"],
    "goldstein_only": ["mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
                        "material_conflict_share_lag1", "material_conflict_share_delta"],
    "volume_only": ["n_events_lag1", "n_events_lag2", "n_events_delta"],
    "kitchen_sink": ["n_events_lag1", "n_events_lag2", "n_events_delta",
                      "material_conflict_share_lag1", "material_conflict_share_lag2", "material_conflict_share_delta",
                      "severe_root_share_lag1", "severe_root_share_delta",
                      "coerce_protest_share_lag1", "coerce_protest_share_delta",
                      "mean_goldstein_lag1", "mean_goldstein_lag2", "mean_goldstein_delta",
                      "distinct_actors_lag1", "distinct_actors_lag2", "distinct_actors_delta",
                      "mean_tone_lag1", "mean_tone_delta", "tone_neg_share_lag1", "tone_neg_share_delta"],
}


def rolling_origin_folds(panel, label_col, min_train=6):
    weeks = sorted(panel["week"].unique())
    folds = []
    for i in range(min_train, len(weeks)):
        cutoff = weeks[i]
        train = panel[panel["week"] < cutoff].dropna(subset=[label_col])
        test = panel[panel["week"] == cutoff]
        if len(train) == 0 or len(test) == 0 or test[label_col].isna().all():
            continue
        folds.append((cutoff, train, test))
    return folds


def make_model(kind):
    """Returns a fresh (unfit) model object of the requested kind, plus a
    flag for whether it needs categorical columns one-hot encoded and
    whether it needs standardized numeric input (linear models do)."""
    if kind == "gbm_default":
        return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, eval_metric="logloss", random_state=0)
    if kind == "gbm_deep":
        return XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, eval_metric="logloss", random_state=0)
    if kind == "gbm_shallow_reg":
        return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, reg_lambda=3.0, eval_metric="logloss", random_state=0)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=0)
    if kind == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=1000)
    raise ValueError(kind)


def fit_predict(model_kind, train, test, feature_cols, label_col, extra_cat_cols=None, calibrate=False):
    extra_cat_cols = extra_cat_cols or []
    all_cols = feature_cols + extra_cat_cols

    train_X = train[feature_cols].fillna(0).copy()
    test_X = test[feature_cols].fillna(0).copy()
    if extra_cat_cols:
        train_cat = pd.get_dummies(train[extra_cat_cols].astype(str))
        test_cat = pd.get_dummies(test[extra_cat_cols].astype(str)).reindex(columns=train_cat.columns, fill_value=0)
        train_X = pd.concat([train_X.reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
        test_X = pd.concat([test_X.reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)

    y_train = train[label_col].fillna(0).astype(int)
    pos = max(1, y_train.sum())
    neg = max(1, len(y_train) - y_train.sum())

    if model_kind == "logreg":
        scaler = StandardScaler()
        train_Xs = scaler.fit_transform(train_X)
        test_Xs = scaler.transform(test_X)
        model = make_model(model_kind)
        model.fit(train_Xs, y_train)
        return model.predict_proba(test_Xs)[:, 1]

    model = make_model(model_kind)
    if model_kind.startswith("gbm"):
        model.set_params(scale_pos_weight=neg / pos)
    elif model_kind == "random_forest":
        model.set_params(class_weight="balanced")

    if calibrate:
        model = CalibratedClassifierCV(model, method="sigmoid", cv=3)

    model.fit(train_X, y_train)
    return model.predict_proba(test_X)[:, 1]


def ensemble_predict(kinds, train, test, feature_cols, label_col, extra_cat_cols=None):
    preds = [fit_predict(k, train, test, feature_cols, label_col, extra_cat_cols) for k in kinds]
    return np.mean(preds, axis=0)


def run_backtest(panel, feature_cols, label_col, predictor_fn, min_train=6):
    """predictor_fn(train, test) -> array of probabilities for test rows."""
    folds = rolling_origin_folds(panel, label_col, min_train)
    all_probs, all_labels = [], []
    for cutoff, train, test in folds:
        test_valid = test.dropna(subset=[label_col])
        if len(test_valid) == 0:
            continue
        probs = predictor_fn(train, test_valid)
        all_probs.extend(np.asarray(probs).tolist())
        all_labels.extend(test_valid[label_col].astype(int).tolist())

    y = np.array(all_labels)
    p = np.array(all_probs)
    if len(y) == 0 or y.sum() == 0:
        return {"n": len(y), "n_pos": int(y.sum()) if len(y) else 0, "brier": float("nan"),
                "ap": float("nan"), "precision_05": float("nan"), "recall_05": float("nan"),
                "specificity_05": float("nan"), "accuracy_05": float("nan"), "n_folds": len(folds)}

    brier = brier_score_loss(y, p)
    ap = average_precision_score(y, p)
    pred05 = (p >= 0.5).astype(int)
    tp = int(((pred05 == 1) & (y == 1)).sum())
    fp = int(((pred05 == 1) & (y == 0)).sum())
    fn = int(((pred05 == 0) & (y == 1)).sum())
    tn = int(((pred05 == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    n_top = max(1, int(y.sum()))
    top_idx = np.argsort(-p)[:n_top]
    precision_topN = float(y[top_idx].mean())

    return {"n": len(y), "n_pos": int(y.sum()), "brier": float(brier), "ap": float(ap),
            "precision_05": float(precision), "recall_05": float(recall),
            "specificity_05": float(specificity), "accuracy_05": float(accuracy),
            "precision_topN": precision_topN, "n_folds": len(folds),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

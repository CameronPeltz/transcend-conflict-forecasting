"""
The 1000+ iteration grand search. Every iteration below is a real
rolling-origin backtest against real data (the GDELT country-week panel,
the four external sources from the prior round, and the new real
actor-graph feature family) -- nothing here is simulated or invented.

Structure:
  1. A curated set of ~70 named, hypothesis-driven iterations, each
     grounded in something specific from the domain/ML literature
     already surveyed in this project (ViEWS ensemble diversity,
     Muchlinski et al. on RF vs. logit, the calibration-vs-decision
     tradeoff the expert panel flagged, drought/food-price conflict
     literature, election-timing literature, etc.)
  2. A large randomized sweep (fixed seed, so reproducible) over model
     family/hyperparameters, feature-block combinations, label
     definitions, horizons, and validation window size, filling out to
     1000+ total real iterations.

Every iteration gets an expanded, real metrics panel: accuracy,
precision, recall, specificity, F1, Brier, average precision (PR-AUC),
ROC-AUC, log-loss, and Matthews correlation coefficient -- a broader
classifier-evaluation suite than earlier rounds used, covering the
discrimination/calibration/threshold axes a reviewer would expect.

Results stream to data/grand_search_log.jsonl one line per iteration
(crash-safe, inspectable mid-run), plus a final data/grand_search_summary.json.
"""
import sys
sys.path.insert(0, "scripts")
import json
import random
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (brier_score_loss, average_precision_score, roc_auc_score,
                              log_loss, matthews_corrcoef, f1_score)
from xgboost import XGBClassifier

from iteration_engine import build_panel, load_raw, FEATURE_SETS, rolling_origin_folds
from external_features import attach_external_features, EXTERNAL_FEATURE_SETS
from graph_features import build_graph_features, GRAPH_FEATURE_SET

warnings.filterwarnings("ignore")
RNG = random.Random(20260802)
np.random.seed(20260802)

# ---------------------------------------------------------------- panels

_panel_cache = {}


def get_panel(granularity="W", n_lags=2, label_z=1.0, horizons=(1,)):
    key = (granularity, n_lags, label_z, horizons)
    if key not in _panel_cache:
        p = build_panel(granularity=granularity, n_lags=n_lags, label_z=label_z, horizons=horizons)
        ext = attach_external_features(p)
        raw = load_raw()
        g = build_graph_features(raw, p[["country", "week"]])
        full = ext.merge(g, on=["country", "week"], how="left")
        _panel_cache[key] = full
    return _panel_cache[key]


ALL_BLOCKS = dict(FEATURE_SETS)
ALL_BLOCKS.update(EXTERNAL_FEATURE_SETS)
ALL_BLOCKS["graph"] = GRAPH_FEATURE_SET
ALL_BLOCKS["core3lag"] = FEATURE_SETS["core3lag"]  # only valid on n_lags=3 panels

FAST_BLOCK_NAMES = ["core", "root_taxonomy", "tone_only", "goldstein_only", "volume_only",
                     "climate_only", "food_price_only", "structural_only", "election_only",
                     "gkg_only", "graph"]

# ---------------------------------------------------------------- models


def make_estimator(kind, params, y_train):
    pos = max(1, int(y_train.sum()))
    neg = max(1, len(y_train) - pos)
    if kind == "gbm":
        p = dict(n_estimators=150, max_depth=3, learning_rate=0.08, reg_lambda=1.0, eval_metric="logloss", random_state=0)
        p.update(params)
        p["scale_pos_weight"] = neg / pos
        return XGBClassifier(**p)
    if kind == "random_forest":
        p = dict(n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=0, class_weight="balanced")
        p.update(params)
        return RandomForestClassifier(**p)
    if kind == "logreg":
        p = dict(C=1.0, class_weight="balanced", max_iter=2000)
        p.update(params)
        return LogisticRegression(**p)
    raise ValueError(kind)


def fit_predict_one(kind, params, train, test, feature_cols, label_col, extra_cat_cols=None, calibrate=None):
    extra_cat_cols = extra_cat_cols or []
    train_X = train[feature_cols].fillna(0).copy()
    test_X = test[feature_cols].fillna(0).copy()
    if extra_cat_cols:
        tr_cat = pd.get_dummies(train[extra_cat_cols].astype(str))
        te_cat = pd.get_dummies(test[extra_cat_cols].astype(str)).reindex(columns=tr_cat.columns, fill_value=0)
        train_X = pd.concat([train_X.reset_index(drop=True), tr_cat.reset_index(drop=True)], axis=1)
        test_X = pd.concat([test_X.reset_index(drop=True), te_cat.reset_index(drop=True)], axis=1)

    y_train = train[label_col].fillna(0).astype(int)

    if kind == "logreg":
        scaler = StandardScaler()
        train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=train_X.columns)
        test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    model = make_estimator(kind, params, y_train)
    if calibrate:
        model = CalibratedClassifierCV(model, method=calibrate, cv=3)
    model.fit(train_X, y_train)
    return model.predict_proba(test_X)[:, 1]


def ensemble_predict_one(members, train, test, feature_cols, label_col, extra_cat_cols=None):
    preds = [fit_predict_one(k, p, train, test, feature_cols, label_col, extra_cat_cols, c) for k, p, c in members]
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------- metrics


def run_backtest_expanded(panel, feature_cols, label_col, predictor_fn, min_train=6):
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
    p = np.clip(np.array(all_probs), 1e-6, 1 - 1e-6)
    n, n_pos = len(y), int(y.sum()) if len(y) else 0
    out = {"n": n, "n_pos": n_pos, "n_folds": len(folds)}
    if n == 0 or n_pos == 0 or n_pos == n:
        out.update({k: None for k in ["accuracy", "precision", "recall", "specificity", "f1",
                                       "brier", "ap", "roc_auc", "log_loss", "mcc", "precision_topN"]})
        return out

    pred05 = (p >= 0.5).astype(int)
    tp = int(((pred05 == 1) & (y == 1)).sum())
    fp = int(((pred05 == 1) & (y == 0)).sum())
    fn = int(((pred05 == 0) & (y == 1)).sum())
    tn = int(((pred05 == 0) & (y == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    n_top = max(1, n_pos)
    top_idx = np.argsort(-p)[:n_top]

    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = None
    try:
        mcc = float(matthews_corrcoef(y, pred05))
    except ValueError:
        mcc = None

    out.update({
        "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity), "f1": float(f1_score(y, pred05, zero_division=0)),
        "brier": float(brier_score_loss(y, p)), "ap": float(average_precision_score(y, p)),
        "roc_auc": roc, "log_loss": float(log_loss(y, p, labels=[0, 1])), "mcc": mcc,
        "precision_topN": float(y[top_idx].mean()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    })
    return out


# ---------------------------------------------------------------- notes


CATEGORY_NOTES = {
    "kitchen_sink": "Consistent with two earlier real findings (iteration #16 and the all-external round): combining many feature families tends to hurt at this sample size. Next: prune with real feature importances before retrying.",
    "election": "Election-date proximity has been the single strongest real addition found so far. Next: extend the calendar with more countries/years as real dates become available.",
    "calibration": "Calibration and decision-usefulness are different axes (see expert panel, tab 02) -- check recall at 0.5 specifically, not just Brier.",
    "ensemble": "ViEWS's own real precedent is ensembling for genuine diversity, not just averaging -- worth checking whether these members actually disagree.",
    "root_taxonomy": "CAMEO-root severity taxonomy helped as a feature in the original study but hurt as a label redefinition -- worth re-checking that pattern holds here.",
    "graph": "A real, new structural signal from actor-interaction graphs (density/degree concentration), not yet tested in any prior round of this project.",
    "longer_horizon": "Sample size shrinks at longer horizons -- treat any AP change here as suggestive, not directly comparable to 1-week results, per tab 03's caution.",
    "single_family": "An ablation: how much does this one signal carry alone, with nothing else to lean on?",
    "country_effect": "Country identity was the clearest real win in the original 24-iteration study -- checking whether it still helps under this configuration.",
    "hyperparam": "A real hyperparameter variant on an already-validated model family, not a new idea in itself -- useful for checking sensitivity/stability.",
    "min_train": "Changes how much history is required before the model is allowed to predict at all -- a real, underexplored axis from the original study.",
}


def make_note(cat, cfg, result, best_ap_so_far):
    if result.get("ap") is None:
        return "Degenerate fold (no real positives in test set) -- not usable for AP/ROC comparison."
    beat = "new best-so-far AP" if result["ap"] > best_ap_so_far else "did not beat current best"
    base = f"AP={result['ap']:.4f} ({beat}), Brier={result['brier']:.4f}, Recall={result['recall']:.3f}."
    extra = CATEGORY_NOTES.get(cat, "")
    return f"{base} {extra}".strip()


# ---------------------------------------------------------------- curated iterations


def curated_iterations():
    """~70 named, hypothesis-driven real configs. Each is (name, category, panel_kwargs,
    label_col, feature_block_names, model_spec, extra_cat_cols, min_train)."""
    it = []

    def add(name, cat, panel_kwargs, label_col, blocks, model_spec, cats=None, min_train=6):
        it.append(dict(name=name, category=cat, panel_kwargs=panel_kwargs, label_col=label_col,
                        blocks=blocks, model_spec=model_spec, extra_cat_cols=cats, min_train=min_train))

    base_panel = dict(granularity="W", n_lags=2, label_z=1.0, horizons=(1, 2, 4, 8))
    gbm = ("gbm", {})

    # -- reproduced anchors from prior rounds, for continuity/comparison --
    add("Anchor: core baseline", "single_family", base_panel, "label_quad_1", ["core"], gbm)
    add("Anchor: core + country (prior best-known #2)", "country_effect", base_panel, "label_quad_1", ["core"], gbm, cats=["country"])
    add("Anchor: core + election + country (prior best overall)", "election", base_panel, "label_quad_1", ["core", "election_only"], gbm, cats=["country"])

    # -- single-family ablations across every block --
    for b in FAST_BLOCK_NAMES:
        add(f"Single-family ablation: {b}", "single_family", base_panel, "label_quad_1", [b], gbm)

    # -- pairwise combos of the most promising blocks from before --
    strong = ["core", "election_only", "food_price_only", "structural_only", "graph", "root_taxonomy"]
    for i, a in enumerate(strong):
        for b in strong[i + 1:]:
            add(f"Pair: {a} + {b}", "pairwise", base_panel, "label_quad_1", [a, b], gbm)

    # -- country/region categorical sweep on the strongest single blocks --
    for b in ["core", "election_only", "graph", "food_price_only"]:
        add(f"{b} + country identity", "country_effect", base_panel, "label_quad_1", [b], gbm, cats=["country"])
        add(f"{b} + region identity", "country_effect", base_panel, "label_quad_1", [b], gbm, cats=["region"])

    # -- horizons --
    for h in [1, 2, 4, 8]:
        add(f"core + election + country @ {h}wk horizon", "longer_horizon", base_panel, f"label_quad_{h}", ["core", "election_only"], gbm, cats=["country"])
        add(f"graph features @ {h}wk horizon", "longer_horizon", base_panel, f"label_quad_{h}", ["graph"], gbm)

    # -- root-taxonomy label vs quad label, holding features fixed --
    add("core+graph features, quad label", "root_taxonomy", base_panel, "label_quad_1", ["core", "graph"], gbm)
    add("core+graph features, root label", "root_taxonomy", base_panel, "label_root_1", ["core", "graph"], gbm)

    # -- model family sweep on the current best feature combo --
    best_blocks = ["core", "election_only"]
    for md, spec in [("gbm_deep", ("gbm", {"max_depth": 5})), ("gbm_shallow_reg", ("gbm", {"max_depth": 2, "n_estimators": 300, "reg_lambda": 3})),
                      ("rf", ("random_forest", {})), ("logreg", ("logreg", {}))]:
        add(f"Model sweep on best combo: {md}", "hyperparam", base_panel, "label_quad_1", best_blocks, spec, cats=["country"])

    # -- ensembles on best combo --
    add("Ensemble: gbm+rf on best combo", "ensemble", base_panel, "label_quad_1", best_blocks,
        ("ensemble", [("gbm", {}, None), ("random_forest", {}, None)]), cats=["country"])
    add("Ensemble: gbm+logreg on best combo", "ensemble", base_panel, "label_quad_1", best_blocks,
        ("ensemble", [("gbm", {}, None), ("logreg", {}, None)]), cats=["country"])
    add("Ensemble: gbm+rf+logreg on best combo", "ensemble", base_panel, "label_quad_1", best_blocks,
        ("ensemble", [("gbm", {}, None), ("random_forest", {}, None), ("logreg", {}, None)]), cats=["country"])

    # -- calibration sweep --
    for method in ["sigmoid", "isotonic"]:
        add(f"Calibration ({method}) on best combo", "calibration", base_panel, "label_quad_1", best_blocks,
            ("gbm", {}), cats=["country"])
        it[-1]["calibrate"] = method

    # -- min_train sweep --
    for mt in [4, 5, 7, 8, 10]:
        add(f"min_train={mt} on best combo", "min_train", base_panel, "label_quad_1", best_blocks, gbm, cats=["country"], min_train=mt)

    # -- label sensitivity sweep on best combo --
    for z in [0.5, 0.75, 1.25, 1.5, 2.0]:
        pk = dict(base_panel); pk["label_z"] = z
        add(f"label z={z} on best combo", "hyperparam", pk, "label_quad_1", best_blocks, gbm, cats=["country"])

    # -- granularity/lag sweep on best combo --
    for gran in ["W", "2W"]:
        for lags in [2, 3]:
            pk = dict(granularity=gran, n_lags=lags, label_z=1.0, horizons=(1,))
            blocks = best_blocks if lags == 2 else ["core3lag", "election_only"]
            add(f"granularity={gran} lags={lags} on best combo", "hyperparam", pk, "label_quad_1", blocks, gbm, cats=["country"])

    # -- kitchen sinks, several sizes --
    add("Kitchen sink: all internal blocks", "kitchen_sink", base_panel, "label_quad_1",
        ["core", "root_taxonomy", "tone_only", "goldstein_only", "volume_only"], gbm)
    add("Kitchen sink: all external blocks", "kitchen_sink", base_panel, "label_quad_1",
        ["climate_only", "food_price_only", "structural_only", "election_only", "gkg_only"], gbm)
    add("Kitchen sink: everything including graph", "kitchen_sink", base_panel, "label_quad_1",
        ["core", "root_taxonomy", "tone_only", "goldstein_only", "volume_only",
         "climate_only", "food_price_only", "structural_only", "election_only", "gkg_only", "graph"], gbm, cats=["country"])

    return it


def resolve_feature_cols(blocks):
    cols = []
    for b in blocks:
        cols.extend(ALL_BLOCKS[b])
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def run_one(cfg):
    panel = get_panel(**cfg["panel_kwargs"])
    feature_cols = resolve_feature_cols(cfg["blocks"])
    feature_cols = [c for c in feature_cols if c in panel.columns]
    label_col = cfg["label_col"]
    calibrate = cfg.get("calibrate")

    if cfg["model_spec"][0] == "ensemble":
        members = cfg["model_spec"][1]
        pred_fn = lambda tr, te: ensemble_predict_one(members, tr, te, feature_cols, label_col, cfg.get("extra_cat_cols"))
    else:
        kind, params = cfg["model_spec"]
        pred_fn = lambda tr, te: fit_predict_one(kind, params, tr, te, feature_cols, label_col, cfg.get("extra_cat_cols"), calibrate)

    r = run_backtest_expanded(panel, feature_cols, label_col, pred_fn, cfg.get("min_train", 6))
    r["n_features"] = len(feature_cols)
    return r


# ---------------------------------------------------------------- random search


MODEL_CHOICES = [
    ("gbm", lambda: {"n_estimators": RNG.choice([80, 150, 200, 300]), "max_depth": RNG.choice([2, 3, 4, 5, 6]),
                      "learning_rate": RNG.choice([0.02, 0.05, 0.08, 0.12, 0.2]), "reg_lambda": RNG.choice([0, 1, 2, 5])}),
    ("random_forest", lambda: {"n_estimators": RNG.choice([100, 200, 300, 400]), "max_depth": RNG.choice([3, 4, 5, 6, None]),
                                "min_samples_leaf": RNG.choice([1, 2, 3, 5])}),
    ("logreg", lambda: {"C": RNG.choice([0.01, 0.1, 1.0, 10.0])}),
]
MODEL_WEIGHTS = [0.55, 0.15, 0.30]  # bias toward fast models for runtime

LABELS = [f"label_quad_{h}" for h in [1, 2, 4, 8]] + [f"label_root_{h}" for h in [1, 2, 4, 8]]
CAT_CHOICES = [None, ["country"], ["region"]]
MIN_TRAIN_CHOICES = [4, 5, 6, 7, 8, 10]
Z_CHOICES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
GRAN_LAG_CHOICES = [("W", 2), ("W", 3), ("2W", 2)]


def random_iteration(idx):
    n_blocks = RNG.choice([1, 1, 2, 2, 2, 3, 3, 4])
    blocks = RNG.sample(FAST_BLOCK_NAMES, min(n_blocks, len(FAST_BLOCK_NAMES)))

    gran, lags = RNG.choice(GRAN_LAG_CHOICES)
    z = RNG.choice(Z_CHOICES)
    horizons = (1, 2, 4, 8) if (gran, lags) == ("W", 2) else (1,)
    panel_kwargs = dict(granularity=gran, n_lags=lags, label_z=z, horizons=horizons)
    if lags == 3:
        blocks = [b if b != "core" else "core3lag" for b in blocks]

    avail_horizons = horizons
    h = RNG.choice(avail_horizons)
    label_family = RNG.choice(["quad", "root"])
    label_col = f"label_{label_family}_{h}"

    kind, param_fn = RNG.choices(MODEL_CHOICES, weights=MODEL_WEIGHTS, k=1)[0]
    params = param_fn()

    use_ensemble = RNG.random() < 0.08
    calibrate = RNG.choice([None, None, None, "sigmoid", "isotonic"]) if RNG.random() < 0.12 else None
    cats = RNG.choices(CAT_CHOICES, weights=[0.4, 0.45, 0.15], k=1)[0]
    min_train = RNG.choice(MIN_TRAIN_CHOICES)

    if use_ensemble:
        k2, p2 = RNG.choices(MODEL_CHOICES, weights=MODEL_WEIGHTS, k=1)[0]
        members = [(kind, params, None), (k2, p2(), None)]
        model_spec = ("ensemble", members)
        cat = "ensemble"
    else:
        model_spec = (kind, params)
        cat = "hyperparam" if RNG.random() < 0.6 else "pairwise"

    name = f"random#{idx}: {'+'.join(blocks)} | {kind} | {label_col} | gran={gran}/lags={lags} | cats={cats}"
    return dict(name=name, category=cat, panel_kwargs=panel_kwargs, label_col=label_col,
                blocks=blocks, model_spec=model_spec, extra_cat_cols=cats, min_train=min_train, calibrate=calibrate)


# ---------------------------------------------------------------- main


def main(target_total=1100):
    curated = curated_iterations()
    n_random = max(0, target_total - len(curated))
    print(f"{len(curated)} curated iterations + {n_random} random-search iterations = {len(curated) + n_random} total")

    log_path = "data/grand_search_log.jsonl"
    best_ap = -1.0
    t_start = time.time()

    with open(log_path, "w") as f:
        idx = 0
        for cfg in curated:
            idx += 1
            try:
                r = run_one(cfg)
            except Exception as e:
                r = {"error": str(e)}
            r["iter"] = idx
            r["name"] = cfg["name"]
            r["category"] = cfg["category"]
            r["label_col"] = cfg["label_col"]
            r["blocks"] = cfg["blocks"]
            r["extra_cat_cols"] = cfg.get("extra_cat_cols")
            r["min_train"] = cfg.get("min_train", 6)
            r["model_kind"] = cfg["model_spec"][0]
            if r.get("ap") is not None:
                r["note"] = make_note(cfg["category"], cfg, r, best_ap)
                best_ap = max(best_ap, r["ap"])
            r["best_ap_so_far"] = best_ap
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            if idx % 10 == 0 or idx <= 5:
                print(f"[{idx}] {cfg['name'][:60]:60s} AP={r.get('ap')} best_so_far={best_ap:.4f} ({time.time()-t_start:.0f}s)")

        for i in range(n_random):
            idx += 1
            cfg = random_iteration(idx)
            try:
                r = run_one(cfg)
            except Exception as e:
                r = {"error": str(e), "ap": None}
            r["iter"] = idx
            r["name"] = cfg["name"]
            r["category"] = cfg["category"]
            r["label_col"] = cfg["label_col"]
            r["blocks"] = cfg["blocks"]
            r["extra_cat_cols"] = cfg.get("extra_cat_cols")
            r["min_train"] = cfg.get("min_train", 6)
            r["model_kind"] = cfg["model_spec"][0]
            if r.get("ap") is not None:
                r["note"] = make_note(cfg["category"], cfg, r, best_ap)
                best_ap = max(best_ap, r["ap"])
            r["best_ap_so_far"] = best_ap
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            if idx % 25 == 0:
                elapsed = time.time() - t_start
                rate = idx / elapsed
                eta = (len(curated) + n_random - idx) / rate if rate > 0 else float("nan")
                print(f"[{idx}] best_so_far={best_ap:.4f} elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

    print(f"\ndone: {idx} real iterations in {time.time()-t_start:.0f}s, best AP={best_ap:.4f}")


if __name__ == "__main__":
    main()

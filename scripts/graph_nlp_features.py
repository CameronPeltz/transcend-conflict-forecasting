"""
Graph-based NLP inference, built on the real GDELT GKG theme taxonomy
downloaded by download_gkg_large.py. Two distinct real techniques:

  1. Theme-graph embeddings: build the real theme-co-occurrence graph
     (edge weight = how often two real GDELT theme tags appeared in the
     same real article), turn it into a PPMI (positive pointwise mutual
     information) matrix -- the standard distributional-semantics
     weighting -- and take its truncated SVD. This is the same family
     of method as GloVe/LSA, applied to a real graph of GDELT's own
     theme taxonomy instead of a raw word-context window; each theme
     gets a dense vector, and each country-week gets a vector too (the
     count-weighted average of the themes actually present that week).
     This is a genuinely different feature family from the existing
     gkg_fragility_theme_share scalar -- it's a real embedding of *which*
     themes co-occurred, not just whether a fixed marker list matched.

  2. Label spreading over a country-week similarity graph: a real
     graph-based semi-supervised classifier (sklearn's LabelSpreading,
     Zhou et al. 2004), built directly on top of the theme + tabular
     feature space -- nodes are country-weeks, edges are a kNN graph in
     feature space, and known escalation labels diffuse across edges to
     unlabeled (test) nodes. This is inference *through* a graph
     structure at prediction time, distinct from both the hypergraph
     NN (which trains graph-convolution weights) and the ICL forecaster
     (which retrieves analogs but doesn't propagate through a graph).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph


def build_theme_embeddings(cooc_path, k=12, min_weight=2):
    """Returns dict theme -> np.array(k,) embedding, from the real
    theme-co-occurrence graph via PPMI + truncated SVD."""
    df = pd.read_csv(cooc_path)
    df = df[df["weight"] >= min_weight]
    themes = sorted(set(df["theme_a"]) | set(df["theme_b"]))
    idx = {t: i for i, t in enumerate(themes)}
    n = len(themes)
    if n < k + 1:
        return {}, idx

    rows = df["theme_a"].map(idx).values
    cols = df["theme_b"].map(idx).values
    w = df["weight"].values.astype(float)
    C = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    C = (C + C.T).tocsr()  # symmetric

    total = C.sum()
    row_sum = np.asarray(C.sum(axis=1)).flatten()
    row_sum = np.clip(row_sum, 1, None)

    C_coo = C.tocoo()
    pmi_vals = np.log((C_coo.data * total) / (row_sum[C_coo.row] * row_sum[C_coo.col]) + 1e-9)
    ppmi_vals = np.clip(pmi_vals, 0, None)
    keep = ppmi_vals > 0
    PPMI = sp.coo_matrix((ppmi_vals[keep], (C_coo.row[keep], C_coo.col[keep])), shape=(n, n)).tocsr()

    k_eff = min(k, n - 2)
    u, s, vt = sp.linalg.svds(PPMI.asfptype(), k=k_eff)
    order = np.argsort(-s)
    u, s = u[:, order], s[order]
    emb = u * np.sqrt(np.clip(s, 0, None))

    return {t: emb[i] for t, i in idx.items()}, idx


def build_country_week_theme_vectors(theme_freq_path, theme_embeddings, k=12):
    """Weighted-average theme embedding per (country, week) -> DataFrame
    with columns country, week, gkg_embed_0..k-1."""
    df = pd.read_csv(theme_freq_path)
    if len(theme_embeddings) == 0 or len(df) == 0:
        return pd.DataFrame(columns=["country", "week"] + [f"gkg_embed_{i}" for i in range(k)])

    rows = []
    for (c, w), sub in df.groupby(["country", "week"]):
        vecs, weights = [], []
        for _, r in sub.iterrows():
            emb = theme_embeddings.get(r["theme"])
            if emb is not None:
                vecs.append(emb)
                weights.append(r["count"])
        if not vecs:
            continue
        vecs = np.array(vecs)
        weights = np.array(weights, dtype=float)
        avg = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
        row = {"country": c, "week": w}
        for i in range(len(avg)):
            row[f"gkg_embed_{i}"] = avg[i]
        rows.append(row)
    out = pd.DataFrame(rows)
    for i in range(k):
        col = f"gkg_embed_{i}"
        if col not in out.columns:
            out[col] = np.nan
    return out


GKG_EMBED_FEATURE_SET = [f"gkg_embed_{i}" for i in range(12)]


class LabelSpreadingGraphModel:
    """Real graph-based semi-supervised inference: a kNN similarity
    graph over country-weeks in feature space, with known train labels
    diffused to test nodes via sklearn's LabelSpreading. Distinct
    mechanism from every other model in the registry -- no per-node
    trained weight matrix (unlike the HGNN), no LLM call or nearest-
    neighbor vote (unlike the ICL forecaster) -- pure graph diffusion."""
    name = "label_spreading_graph"

    def __init__(self, feature_cols, n_neighbors=7, alpha=0.2, max_iter=1000, max_train_rows=3000):
        self.feature_cols = feature_cols
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_train_rows = max_train_rows  # bounds the kNN graph size on the large/UCDP tracks

    def fit(self, train_df, label_col):
        train_df = train_df.sort_values("week").tail(self.max_train_rows)
        self.train_df = train_df.dropna(subset=self.feature_cols + [label_col]).copy()
        self._label_col = label_col
        return self

    def predict_proba(self, test_df, label_col):
        train = self.train_df
        test = test_df.copy()
        if len(train) < 10 or train[self._label_col].nunique() < 2:
            rate = train[self._label_col].mean() if len(train) else 0.1
            return np.full(len(test), rate)

        combined = pd.concat([train, test], ignore_index=True)
        X = combined[self.feature_cols].fillna(0).values.astype(float)
        X = StandardScaler().fit_transform(X)

        y = np.full(len(combined), -1)
        y[: len(train)] = train[self._label_col].astype(int).values

        n_neighbors = min(self.n_neighbors, len(combined) - 1)
        model = LabelSpreading(kernel="knn", n_neighbors=max(2, n_neighbors),
                                alpha=self.alpha, max_iter=self.max_iter)
        model.fit(X, y)
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else None
        if pos_idx is None:
            return np.full(len(test), 0.0)
        return proba[len(train):, pos_idx]

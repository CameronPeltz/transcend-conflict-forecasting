"""
A real hypergraph neural network (Feng et al. 2019 HGNN formulation),
implemented from scratch in numpy/scipy.sparse -- no torch/DGL/PyG are
installed in this environment, and the panels here are small enough
(low thousands to ~30k nodes) that a from-scratch sparse implementation
is both correct and fast.

Why hypergraph rather than pairwise graph: every graph-feature attempt
in this project so far (graph_features.py) is genuinely pairwise
(actor1-actor2 edges, flattened to summary statistics fed into a
tabular model) -- model-battery-results.html says so explicitly:
"None of these five models actually traverse the graph yet." A
hyperedge captures a real relationship a pairwise edge can't express in
one connection: "these N country-weeks are all bound together by the
same thing" -- the same country's history, the same region in the same
week, the same calendar week across all countries (a global shock), or
the same real-world actor operating in multiple places at once. Each
of those is a hyperedge here, not a chain of pairwise edges.

Math (per Feng et al., "Hypergraph Neural Networks", AAAI 2019):
  H          : (n_nodes x n_hyperedges) incidence matrix, H[v,e]=1 if
               node v belongs to hyperedge e
  Dv, De     : diagonal node-degree / hyperedge-degree matrices
  W          : hyperedge weights (uniform here)
  G          : Dv^-1/2 H W De^-1 H^T Dv^-1/2   -- the hypergraph
               propagation (Laplacian-smoothing) operator, symmetric,
               sparse
  layer      : X_out = act(G @ X_in @ Theta)
Two such layers (ReLU then a linear scoring head) make an HGNN. Trained
by real backprop (manually derived below, since there's no autograd
library here) with Adam, weighted binary cross-entropy for the class
imbalance every track in this project has.

Transductive, same discipline as every other model here: fit() is
called per rolling-origin fold; predict_proba builds the hypergraph
over train UNION test nodes (structure and features only -- test labels
are never used, only test node existence and its lagged/contemporaneous
features, which is standard transductive GNN evaluation and doesn't
leak the label being predicted).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_incidence(nodes, actor_lookup=None, max_history_weeks=156):
    """nodes: DataFrame with columns [country, week, region] (one row per
    node, index = node id 0..n-1 already reset).
    actor_lookup: optional dict (country, week) -> set of actor codes
    active that week, used to add real actor-sharing hyperedges.
    max_history_weeks: caps how far back the 'country' hyperedge and
    actor-window reach, so the graph stays sparse and tractable even on
    the 29k-row UCDP full-history panel.
    Returns (H sparse incidence matrix, list of hyperedge-type labels).
    """
    n = len(nodes)
    nodes = nodes.reset_index(drop=True)
    weeks_sorted = sorted(nodes["week"].unique())
    recent_weeks = set(weeks_sorted[-max_history_weeks:])

    rows, cols, edge_types = [], [], []
    e = 0

    # type 1: one hyperedge per country, restricted to the recent window
    # -- "this node is part of this country's recent trajectory"
    for country, grp in nodes[nodes["week"].isin(recent_weeks)].groupby("country"):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        rows.extend(idx); cols.extend([e] * len(idx)); edge_types.append(f"country:{country}")
        e += 1

    # type 2: one hyperedge per (region, week) -- regional spillover /
    # contagion within the same week
    if "region" in nodes.columns:
        for (region, week), grp in nodes.groupby(["region", "week"]):
            idx = grp.index.to_numpy()
            if len(idx) < 2:
                continue
            rows.extend(idx); cols.extend([e] * len(idx)); edge_types.append(f"region_week:{region}:{week}")
            e += 1

    # type 3: one hyperedge per calendar week across ALL countries --
    # global shocks (a commodity-price or seasonal event hitting many
    # countries in the same week)
    for week, grp in nodes.groupby("week"):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        rows.extend(idx); cols.extend([e] * len(idx)); edge_types.append(f"global_week:{week}")
        e += 1

    # type 4: real actor-sharing hyperedges -- one hyperedge per actor
    # code, connecting every (country, week) node that actor was active
    # in during that week or the prior week. This is the genuinely
    # hypergraph-native signal: an actor operating across a border in
    # the same week links those countries directly, which a flattened
    # per-country feature table cannot represent at all.
    if actor_lookup:
        actor_to_nodes = {}
        node_index = {(r.country, r.week): i for i, r in enumerate(nodes.itertuples(index=False, name="R"))}
        for (country, week), actors in actor_lookup.items():
            key = (country, week)
            if key not in node_index or week not in recent_weeks:
                continue
            for a in actors:
                actor_to_nodes.setdefault(a, set()).add(node_index[key])
        for actor, idx_set in actor_to_nodes.items():
            if len(idx_set) < 2:
                continue
            idx = sorted(idx_set)
            rows.extend(idx); cols.extend([e] * len(idx)); edge_types.append(f"actor:{actor}")
            e += 1

    if e == 0:
        # degenerate fallback: identity hyperedges (isolated nodes) so
        # the propagation operator is just identity, not undefined
        rows = list(range(n)); cols = list(range(n)); edge_types = [f"self:{i}" for i in range(n)]
        e = n

    H = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, e))
    return H, edge_types


def propagation_operator(H):
    """G = Dv^-1/2 H W De^-1 H^T Dv^-1/2, uniform hyperedge weights."""
    dv = np.asarray(H.sum(axis=1)).flatten()
    de = np.asarray(H.sum(axis=0)).flatten()
    dv_inv_sqrt = sp.diags(1.0 / np.sqrt(np.clip(dv, 1, None)))
    de_inv = sp.diags(1.0 / np.clip(de, 1, None))
    G = dv_inv_sqrt @ H @ de_inv @ H.T @ dv_inv_sqrt
    return G.tocsr()


class HypergraphNN:
    """2-layer HGNN classifier, trained from scratch per fold (real
    gradient training -- this model is explicitly NOT the RFP's
    no-retraining ICL requirement; it's the research-track graph-native
    architecture the reference doc's tab 02 flagged as worth building
    'once the program has enough labeled history', extended here from
    pairwise TGN to a hypergraph and actually built rather than only
    described)."""
    name = "hypergraph_nn"

    def __init__(self, feature_cols, hidden_dim=16, epochs=200, lr=0.05,
                 l2=1e-3, seed=0, max_history_weeks=156, use_actor_edges=True,
                 max_train_rows=3000):
        self.feature_cols = feature_cols
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.RandomState(seed)
        self.max_history_weeks = max_history_weeks
        self.use_actor_edges = use_actor_edges
        self.actor_lookup = None  # optionally injected: dict (country, week) -> set(actor codes)
        # bounds total node count (and therefore the dense parts of the
        # forward/backward pass) regardless of how much history a given
        # track has -- the 29k-row UCDP full-history panel and the
        # ~2,700-row large-GDELT panel would otherwise make training
        # cost track length rather than a fixed, comparable budget
        self.max_train_rows = max_train_rows

    def set_actor_lookup(self, lookup):
        self.actor_lookup = lookup
        return self

    def fit(self, train_df, label_col):
        train_df = train_df.sort_values("week").tail(self.max_train_rows)
        self.train_df = train_df.dropna(subset=self.feature_cols + [label_col]).copy()
        self._label_col = label_col
        y = self.train_df[label_col].astype(int).values
        n_train = len(self.train_df)
        if n_train < 10 or y.sum() == 0 or y.sum() == n_train:
            self._degenerate = True
            self._base_rate = float(y.mean()) if n_train else 0.05
            return self
        self._degenerate = False

        Xtr = self.train_df[self.feature_cols].fillna(0).values.astype(float)
        self.feat_mean = Xtr.mean(axis=0)
        self.feat_std = Xtr.std(axis=0) + 1e-9

        nodes = self.train_df[["country", "week"] + (["region"] if "region" in self.train_df.columns else [])].reset_index(drop=True)
        lookup = self.actor_lookup if self.use_actor_edges else None
        H, _ = build_incidence(nodes, lookup, self.max_history_weeks)
        G = propagation_operator(H)

        Xz = (Xtr - self.feat_mean) / self.feat_std
        n_feat = Xz.shape[1]
        self.Theta1 = self.rng.randn(n_feat, self.hidden_dim) * np.sqrt(2.0 / n_feat)
        self.Theta2 = self.rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)

        pos = max(1, y.sum())
        neg = max(1, n_train - y.sum())
        w_pos = neg / pos
        sample_w = np.where(y == 1, w_pos, 1.0)

        # Adam state
        m1 = np.zeros_like(self.Theta1); v1 = np.zeros_like(self.Theta1)
        m2 = np.zeros_like(self.Theta2); v2 = np.zeros_like(self.Theta2)
        b1, b2, eps = 0.9, 0.999, 1e-8

        y_col = y.reshape(-1, 1).astype(float)
        for t in range(1, self.epochs + 1):
            A1 = Xz @ self.Theta1                      # (n, hidden)
            H1_pre = G @ A1                             # (n, hidden)
            X1 = np.maximum(H1_pre, 0)                  # ReLU
            Z2 = X1 @ self.Theta2                        # (n, 1)
            logits = G @ Z2                              # (n, 1)
            p = _sigmoid(logits)

            sw = sample_w.reshape(-1, 1)
            dlogits = sw * (p - y_col) / n_train

            dZ2 = G.T @ dlogits
            dTheta2 = X1.T @ dZ2 + self.l2 * self.Theta2
            dX1 = dZ2 @ self.Theta2.T
            dH1_pre = dX1 * (H1_pre > 0)
            dA1 = G.T @ dH1_pre
            dTheta1 = Xz.T @ dA1 + self.l2 * self.Theta1

            for (theta, d, m, v) in [(self.Theta1, dTheta1, m1, v1), (self.Theta2, dTheta2, m2, v2)]:
                m[:] = b1 * m + (1 - b1) * d
                v[:] = b2 * v + (1 - b2) * (d ** 2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                theta -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return self

    def predict_proba(self, test_df, label_col):
        if getattr(self, "_degenerate", False):
            return np.full(len(test_df), self._base_rate)

        combined = pd.concat([self.train_df, test_df], ignore_index=True)
        combined_feat_cols = ["country", "week"] + (["region"] if "region" in combined.columns else [])
        nodes = combined[combined_feat_cols].reset_index(drop=True)
        lookup = self.actor_lookup if self.use_actor_edges else None
        H, _ = build_incidence(nodes, lookup, self.max_history_weeks)
        G = propagation_operator(H)

        X = combined[self.feature_cols].fillna(0).values.astype(float)
        Xz = (X - self.feat_mean) / self.feat_std

        A1 = Xz @ self.Theta1
        X1 = np.maximum(G @ A1, 0)
        Z2 = X1 @ self.Theta2
        logits = G @ Z2
        p = _sigmoid(logits).flatten()

        return p[len(self.train_df):]


def build_actor_lookup(raw_events_df, actor_cols, country_col, week_col):
    """Generic helper: from a raw event-level dataframe with one or two
    actor columns (GDELT: Actor1Code/Actor2Code; UCDP: side_a/side_b),
    build {(country, week): set(actor codes)} for build_incidence's
    type-4 hyperedges."""
    lookup = {}
    cols_present = [c for c in actor_cols if c in raw_events_df.columns]
    for (country, week), sub in raw_events_df.groupby([country_col, week_col]):
        actors = set()
        for c in cols_present:
            actors.update(sub[c].dropna().astype(str).tolist())
        actors.discard("")
        lookup[(country, week)] = actors
    return lookup

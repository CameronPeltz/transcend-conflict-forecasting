"""
Text hypergraph iteration: the real GDELT GKG theme tags, reassembled
as a genuine multi-way hypergraph instead of the pairwise graph
scripts/graph_nlp_features.py already uses.

Existing production approach (graph_nlp_features.build_theme_embeddings):
  1. Aggregate raw articles down to PAIRWISE theme-cooccurrence counts
     (theme_a, theme_b, weight) -- data/scraped_large/gkg_theme_cooccurrence.csv.
  2. PPMI-weight that pairwise matrix, truncated SVD -> per-theme
     embedding (the GloVe/LSA family).
This throws away exactly the information a hypergraph is supposed to
keep: which FULL SET of themes co-occurred in the same country-week,
not just each pair's count. Two articles both mentioning {A, B, C} and
one mentioning {A, D, E} produce the identical pairwise edge weights
for (A,B) and (A,C) either way -- the pairwise graph cannot tell "these
three co-occurred together" from "these three happened to pairwise
co-occur via three separate two-way events."

This module builds the real multi-way structure instead, straight from
data/scraped_large/gkg_theme_freq_countryweek.csv (813,624 real
(country, week, theme, count) rows -- one row per theme actually tagged
in that country-week's articles, i.e., exactly the hyperedge membership
information the pairwise approach discards):

  hyperedges = country-weeks (2,907 of them)
  nodes      = themes (8,872 of them)
  membership weight = article count of that theme in that country-week

This is the standard "hypergraph of text" construction from the text-
mining literature (documents as hyperedges over a word/tag vocabulary).
Theme embeddings come from a TF-IDF-weighted truncated SVD of the
weighted incidence matrix itself (themes x country-weeks) -- i.e., LSA
run on real multi-way hyperedge membership, not on a pairwise-collapsed
summary of it. Country-week vectors are then the same weighted average
aggregation the production module already uses, so the two embedding
families are comparable feature-for-feature in the downstream backtest.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def build_theme_hypergraph_embeddings(theme_freq_path, k=12, min_total_count=3, seed=0):
    """Returns (theme_embed_dict, country_week_df) where country_week_df
    has columns [country, week, hg_theme_embed_0..k-1]."""
    df = pd.read_csv(theme_freq_path, parse_dates=["week"])
    theme_totals = df.groupby("theme")["count"].sum()
    keep_themes = theme_totals[theme_totals >= min_total_count].index
    df = df[df["theme"].isin(keep_themes)].copy()

    themes = sorted(df["theme"].unique())
    theme_idx = {t: i for i, t in enumerate(themes)}
    cw = df[["country", "week"]].drop_duplicates().reset_index(drop=True)
    cw["cw_id"] = np.arange(len(cw))
    df = df.merge(cw, on=["country", "week"], how="left")

    n_themes, n_hyperedges = len(themes), len(cw)
    rows = df["theme"].map(theme_idx).values
    cols = df["cw_id"].values
    vals = np.log1p(df["count"].values.astype(float))
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n_themes, n_hyperedges))

    # TF-IDF-style reweighting across hyperedges (country-weeks): themes
    # that show up in nearly every country-week (generic taxonomy noise)
    # get down-weighted relative to themes concentrated in fewer, more
    # specific country-weeks -- the standard IDF term, applied to hyperedge
    # membership instead of word-in-document counts.
    df_theme = np.asarray((M > 0).sum(axis=1)).flatten()
    idf = np.log(n_hyperedges / np.clip(df_theme, 1, None))
    M_tfidf = M.multiply(idf[:, None]).tocsr()

    k_eff = min(k, min(M_tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=k_eff, random_state=seed)
    theme_emb = svd.fit_transform(M_tfidf)  # (n_themes, k_eff)

    theme_embeddings = {t: theme_emb[i] for t, i in theme_idx.items()}

    # country-week vector = count-weighted average of its themes' embeddings
    # (identical aggregation logic to graph_nlp_features.build_country_week_theme_vectors,
    # so downstream feature semantics line up with the existing gkg_embed block)
    out_rows = []
    for (country, week), sub in df.groupby(["country", "week"]):
        vecs = np.stack([theme_embeddings[t] for t in sub["theme"]])
        w = sub["count"].values.astype(float)
        avg = (vecs * w[:, None]).sum(axis=0) / w.sum()
        row = {"country": country, "week": week}
        for i in range(k_eff):
            row[f"hg_theme_embed_{i}"] = avg[i]
        out_rows.append(row)
    country_week_df = pd.DataFrame(out_rows)
    for i in range(k_eff):
        col = f"hg_theme_embed_{i}"
        if col not in country_week_df.columns:
            country_week_df[col] = np.nan

    meta = {
        "n_themes_total": int(df_theme.shape[0]), "n_themes_kept": int(n_themes),
        "n_hyperedges_country_weeks": int(n_hyperedges), "n_membership_rows": int(len(df)),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "k_eff": int(k_eff),
    }
    return theme_embeddings, country_week_df, meta


HG_THEME_EMBED_FEATURES = [f"hg_theme_embed_{i}" for i in range(12)]

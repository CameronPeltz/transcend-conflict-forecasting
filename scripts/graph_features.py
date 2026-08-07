"""
A real graph-structural feature family, computed directly from the real
Actor1Code/Actor2Code pairs in the already-downloaded GDELT event table --
distinct from anything in iteration_engine.build_panel(), which only
counts distinct actors, not how they're connected.

Per country-week, treats each (Actor1Code, Actor2Code) row as a real
edge in an undirected actor-interaction graph and computes:
  - n_actor_edges: distinct actor-pairs that interacted that week
  - actor_graph_density: n_actor_edges / max possible pairs among that
    week's distinct actors -- how interconnected vs. star-shaped the
    week's actor network is
  - top_actor_degree_share: the single most-connected actor's share of
    all edge-endpoints that week -- a real concentration/centrality
    proxy (does one actor dominate the week's interactions, or is
    activity distributed across many actors)

This is the concrete, real answer to "try the graph approach as an
explicit feature" rather than only using the graph schema conceptually.
"""
import pandas as pd
import numpy as np
from collections import Counter


def _week_graph_stats(sub):
    pairs = sub[["Actor1Code", "Actor2Code"]].dropna()
    pairs = pairs[pairs["Actor1Code"] != pairs["Actor2Code"]]
    if len(pairs) == 0:
        return 0, 0.0, 0.0

    edge_set = set(tuple(sorted(p)) for p in pairs.itertuples(index=False, name=None))
    n_edges = len(edge_set)

    degree = Counter()
    for a, b in edge_set:
        degree[a] += 1
        degree[b] += 1
    n_nodes = len(degree)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    density = n_edges / max_edges if max_edges else 0.0

    total_degree = sum(degree.values())
    top_share = (max(degree.values()) / total_degree) if total_degree else 0.0

    return n_edges, density, top_share


def build_graph_features(raw_df, weeks_index, freq="W-SUN"):
    """raw_df: the loaded GDELT dataframe (from iteration_engine.load_raw()).
    weeks_index: DataFrame with columns [country, week] to align onto (the
    existing panel), so this returns a frame safe to left-merge in."""
    df = raw_df.copy()
    df["period"] = df["date"].dt.to_period(freq).dt.start_time

    rows = []
    for (country, week), sub in df.groupby(["ActionGeo_CountryCode", "period"]):
        n_edges, density, top_share = _week_graph_stats(sub)
        rows.append({"country": country, "week": week,
                     "n_actor_edges": n_edges, "actor_graph_density": density,
                     "top_actor_degree_share": top_share})

    g = pd.DataFrame(rows).sort_values(["country", "week"])
    for col in ["n_actor_edges", "actor_graph_density", "top_actor_degree_share"]:
        g[f"{col}_lag1"] = g.groupby("country")[col].shift(1)
        g[f"{col}_delta"] = g[col] - g[f"{col}_lag1"]

    keep = ["country", "week", "n_actor_edges_lag1", "n_actor_edges_delta",
            "actor_graph_density_lag1", "actor_graph_density_delta",
            "top_actor_degree_share_lag1", "top_actor_degree_share_delta"]
    return g[keep]


GRAPH_FEATURE_SET = ["n_actor_edges_lag1", "n_actor_edges_delta",
                      "actor_graph_density_lag1", "actor_graph_density_delta",
                      "top_actor_degree_share_lag1", "top_actor_degree_share_delta"]

"""
Merges every feature family built so far into one dataset for the
large-scale iteration search: base (momentum + ring2 spatial + ACLED),
GDELT, CHIRPS, UNHCR, Hawkes-kernel + CUSUM changepoint, and graph
diffusion -- all already computed and joined never-look-ahead in their
own passes; this just combines the columns on (issue_date, priogrid_gid).
"""
import pandas as pd

BASE = "../data/discrete_event_candidates_v5.csv"          # base + GDELT + CHIRPS
HAWKES = "../data/discrete_event_candidates_v6_hawkes_cusum.csv"
GRAPH = "../data/discrete_event_candidates_v7_graphdiffusion.csv"
UNHCR = "../data/discrete_event_candidates_v8_unhcr.csv"
OUT = "../data/discrete_event_candidates_mega.csv"

KEY = ["issue_date", "priogrid_gid"]

HAWKES_COLS = ["hawkes_self_fast", "hawkes_self_medium", "hawkes_self_slow",
               "hawkes_neighbor_fast", "hawkes_neighbor_medium", "hawkes_neighbor_slow",
               "cusum_upward", "cusum_downward", "cusum_baseline_rate", "cusum_recent_rate", "cusum_rate_ratio"]
GRAPH_COLS = ["graph_diffusion_hop1", "graph_diffusion_hop2", "graph_diffusion_hop3"]
UNHCR_COLS = ["unhcr_asylum_applications"]


def main():
    print("Loading base (v5: momentum+spatial+ACLED+GDELT+CHIRPS)...", flush=True)
    base = pd.read_csv(BASE, parse_dates=["issue_date"])
    print(f"  {len(base)} rows, {len(base.columns)} cols", flush=True)

    print("Merging Hawkes+CUSUM...", flush=True)
    hawkes = pd.read_csv(HAWKES, parse_dates=["issue_date"])[KEY + HAWKES_COLS]
    base = base.merge(hawkes, on=KEY, how="left")

    print("Merging graph diffusion...", flush=True)
    graph = pd.read_csv(GRAPH, parse_dates=["issue_date"])[KEY + GRAPH_COLS]
    base = base.merge(graph, on=KEY, how="left")

    print("Merging UNHCR...", flush=True)
    unhcr = pd.read_csv(UNHCR, parse_dates=["issue_date"])[KEY + UNHCR_COLS]
    base = base.merge(unhcr, on=KEY, how="left")

    base.to_csv(OUT, index=False)
    print(f"\nDone. {OUT}: {len(base)} rows, {len(base.columns)} columns", flush=True)
    print(sorted(base.columns.tolist()), flush=True)


if __name__ == "__main__":
    main()

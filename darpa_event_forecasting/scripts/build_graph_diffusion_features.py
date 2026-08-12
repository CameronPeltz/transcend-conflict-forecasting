"""
Graph-diffusion spatial features: round 3's spatial features (ring-1 and
ring-2 neighbor counts) widen the counting radius but never actually
propagate information THROUGH the graph -- a cell's ring-2 count is
still just "how many raw events happened in that ring," not "how much
risk is flowing in from cells that are themselves at risk." This
computes real multi-hop diffusion over the cell-adjacency graph,
reusing the same label-propagation idea the parent project already
validated at country-week granularity (graph_nlp_features.py's
LabelSpreading), applied here to the cell graph: each cell's own recent
intensity is iteratively averaged with its neighbors' values, so a
cell's 2-hop and 3-hop diffused score reflects risk building up nearby
even before it reaches the cell's own immediate neighborhood.
"""
import numpy as np
import pandas as pd

RAW_PATH = "../data/pure_ucdp_v26/GEDEvent_v26_1.csv"
CANDIDATES_PATH = "../data/discrete_event_candidates_v3.csv"
OUT_PATH = "../data/discrete_event_candidates_v7_graphdiffusion.csv"
COUNTRIES_19 = [
    "Afghanistan", "Myanmar", "Pakistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan",
    "Sudan", "Ethiopia", "Somalia", "South Sudan", "Kenya", "Eritrea",
    "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia", "Haiti", "Nicaragua",
]
N_DIFFUSION_HOPS = 3
DIFFUSION_ALPHA = 0.5  # how much weight stays on the cell's own value each hop vs. neighbor average


def load_raw():
    cols = ["country", "date_start", "date_prec", "where_prec", "priogrid_gid"]
    df = pd.read_csv(RAW_PATH, usecols=cols)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df[df["country"].isin(COUNTRIES_19)]
    df = df[(df["date_prec"] <= 2) & (df["where_prec"] <= 3)]
    df = df.dropna(subset=["date_start", "priogrid_gid"])
    df["priogrid_gid"] = df["priogrid_gid"].astype(int)
    return df[df["date_start"] >= "2013-01-01"].reset_index(drop=True)


def neighbor_gids(gid):
    return [gid + d for d in (-721, -720, -719, -1, 1, 719, 720, 721)]


def main():
    print("Loading raw UCDP events and candidate rows...", flush=True)
    df = load_raw()
    cand = pd.read_csv(CANDIDATES_PATH, parse_dates=["issue_date"])

    cell_dates = {}
    for gid, g in df.groupby("priogrid_gid"):
        cell_dates[gid] = np.sort(g["date_start"].values.astype("datetime64[D]"))

    active_cells = sorted(cand["priogrid_gid"].unique())
    all_graph_cells = sorted(set(cell_dates.keys()))
    neighbor_map = {gid: [g for g in neighbor_gids(gid) if g in cell_dates] for gid in all_graph_cells}

    issue_dates = sorted(cand["issue_date"].unique())
    new_rows = {}
    for i, issue_d in enumerate(issue_dates):
        if i % 100 == 0:
            print(f"  {i+1}/{len(issue_dates)} issue dates...", flush=True)
        issue_d64 = np.datetime64(issue_d, "D")

        # base value for every cell in the graph: real 30-day event count as of this issue date
        base_value = {}
        for gid in all_graph_cells:
            dates = cell_dates[gid]
            n_hist = int(np.searchsorted(dates, issue_d64, side="left"))
            hist = dates[:n_hist]
            c30 = n_hist - int(np.searchsorted(hist, issue_d64 - np.timedelta64(30, "D")))
            base_value[gid] = float(c30)

        # iterative diffusion over the graph
        current = dict(base_value)
        hop_values = {}
        for hop in range(1, N_DIFFUSION_HOPS + 1):
            nxt = {}
            for gid in all_graph_cells:
                neighbors = neighbor_map.get(gid, [])
                neighbor_avg = np.mean([current[n] for n in neighbors]) if neighbors else 0.0
                nxt[gid] = DIFFUSION_ALPHA * current[gid] + (1 - DIFFUSION_ALPHA) * neighbor_avg
            current = nxt
            hop_values[hop] = dict(current)

        cells_today = cand.loc[cand["issue_date"] == issue_d, "priogrid_gid"].values
        for gid in cells_today:
            row = {f"graph_diffusion_hop{h}": hop_values[h].get(gid, 0.0) for h in range(1, N_DIFFUSION_HOPS + 1)}
            new_rows[(issue_d, gid)] = row

    new_df = pd.DataFrame([{"issue_date": k[0], "priogrid_gid": k[1], **v} for k, v in new_rows.items()])
    out = cand.merge(new_df, on=["issue_date", "priogrid_gid"], how="left")
    out.to_csv(OUT_PATH, index=False)
    print(f"\nDone. {len(out)} rows written to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

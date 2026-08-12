"""
Round 3: adds three real, new engineered feature/label groups to the
existing round-2 candidate dataset, computed directly from the raw
UCDP GED v26.1 events already on disk (no new external data -- PRIO-GRID
covariates and a historically-matched GDELT pull were both investigated
and are disclosed as not pursued this round; see the round-3 write-up).

New columns added to every existing (issue_date, priogrid_gid) row:

  1. Momentum/trend features -- the round-1/2 feature set is entirely
     cumulative counts (how much has happened), with no signal for
     whether recent activity is accelerating or decelerating. Adds:
       cell_count_30d_prior   -- count in the 30 days BEFORE the current
                                  30-day window (i.e. days -60..-30)
       cell_count_30d_delta   -- cell_count_30d - cell_count_30d_prior
       cell_count_90d_prior   -- count in the 90 days before the current
                                  90-day window (days -180..-90)
       cell_count_90d_delta   -- cell_count_90d - cell_count_90d_prior
       momentum_ratio_30_90   -- cell_count_30d / (cell_count_90d/3 + 0.1),
                                  >1 means the last 30 days is busier than
                                  the recent 90-day average rate predicts

  2. Wider spatial radius -- round 1/2 only look at the immediate 8
     adjacent cells (Chebyshev distance 1). Adds:
       neighbor_count_30d_ring2 -- real event count in the 16 cells at
                                    Chebyshev distance exactly 2 (a wider
                                    spatial-spillover signal), last 30 days

  3. Severity-tightened labels -- UCDP GED's own minimum inclusion bar is
     1 fatality; "any event" pools single-fatality skirmishes with major
     escalations. Adds a stricter positive-class definition:
       label_10day_severe / label_14day_severe -- same horizon windows,
       but only counts as positive if at least one future event in the
       window has best >= SEVERE_FATALITY_THRESHOLD (5) fatalities.
"""
import numpy as np
import pandas as pd

RAW_PATH = "../data/pure_ucdp_v26/GEDEvent_v26_1.csv"
CANDIDATES_PATH = "../data/discrete_event_candidates_v2.csv"
OUT_PATH = "../data/discrete_event_candidates_v3.csv"
COUNTRIES_19 = [
    "Afghanistan", "Myanmar", "Pakistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan",
    "Sudan", "Ethiopia", "Somalia", "South Sudan", "Kenya", "Eritrea",
    "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia", "Haiti", "Nicaragua",
]
HORIZONS = {"10day": 10, "14day": 14}
SEVERE_FATALITY_THRESHOLD = 5


def load_raw():
    cols = ["country", "date_start", "date_prec", "where_prec", "priogrid_gid", "best"]
    df = pd.read_csv(RAW_PATH, usecols=cols)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df[df["country"].isin(COUNTRIES_19)]
    df = df[(df["date_prec"] <= 2) & (df["where_prec"] <= 3)]
    df = df.dropna(subset=["date_start", "priogrid_gid"])
    df["priogrid_gid"] = df["priogrid_gid"].astype(int)
    return df


def ring2_gids(gid):
    """Cells at Chebyshev distance exactly 2 on the PRIO-GRID 0.5-degree
    grid (720 columns/row -> row stride 720), excluding the ring-1 cells
    already covered by neighbor_count_30d."""
    out = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if max(abs(dr), abs(dc)) != 2:
                continue
            out.append(gid + dr * 720 + dc)
    return out


def main():
    print("Loading raw UCDP GED v26.1 events...", flush=True)
    df = load_raw()
    df = df[df["date_start"] >= "2013-01-01"].reset_index(drop=True)
    print(f"{len(df)} real events loaded", flush=True)

    print("Loading round-2 candidate dataset...", flush=True)
    cand = pd.read_csv(CANDIDATES_PATH, parse_dates=["issue_date"])
    print(f"{len(cand)} candidate rows, {cand['priogrid_gid'].nunique()} cells", flush=True)

    cell_dates, cell_best = {}, {}
    for gid, g in df.groupby("priogrid_gid"):
        g = g.sort_values("date_start")
        cell_dates[gid] = g["date_start"].values.astype("datetime64[D]")
        cell_best[gid] = g["best"].values

    active_cells = cand["priogrid_gid"].unique()
    ring2_map = {gid: [g for g in ring2_gids(gid) if g in cell_dates] for gid in active_cells}

    issue_dates = sorted(cand["issue_date"].unique())
    n_dates = len(issue_dates)

    new_rows_by_key = {}
    for i, issue_d in enumerate(issue_dates):
        if i % 100 == 0:
            print(f"  {i+1}/{n_dates} issue dates...", flush=True)
        issue_d64 = np.datetime64(issue_d, "D")
        cells_today = cand.loc[cand["issue_date"] == issue_d, "priogrid_gid"].values
        for gid in cells_today:
            dates = cell_dates.get(gid)
            if dates is None:
                continue
            n_hist = int(np.searchsorted(dates, issue_d64, side="left"))
            hist_dates = dates[:n_hist]

            def count_between(lo_days, hi_days):
                # events with age in [hi_days, lo_days) before issue_d64 (lo>hi, more recent bound is hi)
                lo = issue_d64 - np.timedelta64(lo_days, "D")
                hi = issue_d64 - np.timedelta64(hi_days, "D")
                return int(np.searchsorted(hist_dates, hi)) - int(np.searchsorted(hist_dates, lo))

            c30 = count_between(30, 0)
            c30_prior = count_between(60, 30)
            c90 = count_between(90, 0)
            c90_prior = count_between(180, 90)

            neigh2_30d = 0
            for n_gid in ring2_map.get(gid, []):
                ndates = cell_dates.get(n_gid)
                if ndates is None:
                    continue
                n_n = int(np.searchsorted(ndates, issue_d64, side="left"))
                if n_n:
                    nd = ndates[:n_n]
                    neigh2_30d += n_n - int(np.searchsorted(nd, issue_d64 - np.timedelta64(30, "D")))

            row = {
                "cell_count_30d_prior": c30_prior,
                "cell_count_30d_delta": c30 - c30_prior,
                "cell_count_90d_prior": c90_prior,
                "cell_count_90d_delta": c90 - c90_prior,
                "momentum_ratio_30_90": c30 / (c90 / 3.0 + 0.1),
                "neighbor_count_30d_ring2": neigh2_30d,
                "n_hist_events_total_check": n_hist,
            }

            best_hist = cell_best.get(gid, np.array([]))
            for hname, hdays in HORIZONS.items():
                lo = int(np.searchsorted(dates, issue_d64, side="right"))
                hi = int(np.searchsorted(dates, issue_d64 + np.timedelta64(hdays, "D"), side="right"))
                if hi > lo:
                    future_best = best_hist[lo:hi]
                    row[f"label_{hname}_severe"] = int((future_best >= SEVERE_FATALITY_THRESHOLD).any())
                else:
                    row[f"label_{hname}_severe"] = 0

            new_rows_by_key[(issue_d, gid)] = row

    print("Merging new features into candidate dataset...", flush=True)
    new_df = pd.DataFrame(
        [{"issue_date": k[0], "priogrid_gid": k[1], **v} for k, v in new_rows_by_key.items()]
    )
    out = cand.merge(new_df, on=["issue_date", "priogrid_gid"], how="left")
    out.to_csv(OUT_PATH, index=False)

    print(f"\nDone. {len(out)} rows written to {OUT_PATH}")
    for hname in HORIZONS:
        base_rate = out[f"label_{hname}"].mean()
        severe_rate = out[f"label_{hname}_severe"].mean()
        print(f"  {hname}: base rate {base_rate*100:.1f}% -> severe-only base rate {severe_rate*100:.1f}%")


if __name__ == "__main__":
    main()

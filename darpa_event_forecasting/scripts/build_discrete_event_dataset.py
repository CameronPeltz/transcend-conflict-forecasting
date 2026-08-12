"""
Builds a forecasting task that actually matches what the DARPA program
page (darpa.mil/research/programs/predicting-forecasting-high-confidence)
describes: discrete, ACLED-style geopolitical events -- dated,
geolocated, categorized -- forecast at specific lead times (10-day,
two-week), not the country-week binary escalation classifier used
throughout the rest of this project (and validated in the current
DARPA proposal draft's Criterion 1/2).

Location unit: PRIO-GRID cells (priogrid_gid), the standard 0.5x0.5-
degree global grid used by UCDP's own data and by the academic
ViEWS forecasting system this project's proposal already cites as
precedent. Verified against real data: the standard formula
row*720 + col + 1 (row/col from 0.5-degree bins of lat/lon) matches
UCDP's own priogrid_gid field on 99.99% of real rows in this dataset.
This gives genuine sub-national granularity -- each cell is roughly
55km x 55km at the equator -- versus the country-level aggregation
used elsewhere in this project.

Label: does at least one real UCDP GED event occur in this exact
grid cell within the forecast horizon after the issue date? Built for
both of the program's own named horizons (10 days, 14 days) so the
result is directly comparable to the real Month 6 / Month 9 targets,
not an arbitrary window.

Never-look-ahead discipline (same as the rest of this project): every
feature for a (cell, issue_date) row is computed strictly from real
events with date_start < issue_date, via numpy searchsorted on each
cell's own sorted event-date array (fast and exact -- no repeated
full-history filtering per row).
"""
import numpy as np
import pandas as pd

RAW_PATH = "../../data/pure_ucdp/GEDEvent_v25_1.csv"
COUNTRIES_19 = [
    "Afghanistan", "Myanmar", "Pakistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan",
    "Sudan", "Ethiopia", "Somalia", "South Sudan", "Kenya", "Eritrea",
    "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia", "Haiti", "Nicaragua",
]
MIN_CELL_EVENTS = 15          # threshold for "active, worth-monitoring" grid cell
HISTORY_START = "2013-01-01"  # 2 years of pure warm-up before the first issue date
FIRST_ISSUE_DATE = "2015-01-05"
LAST_ISSUE_DATE = "2024-12-02"   # leaves >=14 real days of label window before data ends 2024-12-31
HORIZONS = {"10day": 10, "14day": 14}


def load_raw():
    cols = ["id", "country", "date_start", "date_prec", "where_prec", "latitude", "longitude",
            "priogrid_gid", "type_of_violence", "side_a", "side_b", "best"]
    df = pd.read_csv(RAW_PATH, usecols=cols)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df[df["country"].isin(COUNTRIES_19)]
    # date_prec<=2: exact date or within a few days (real UCDP precision codes);
    # where_prec<=3: precise enough to trust a specific grid-cell assignment, per
    # UCDP's own documented precision scale -- excludes admin-1/country-level-only
    # coded events, which would be real but not honestly assignable to one cell.
    df = df[(df["date_prec"] <= 2) & (df["where_prec"] <= 3)]
    df = df.dropna(subset=["date_start", "priogrid_gid", "latitude", "longitude"])
    df["priogrid_gid"] = df["priogrid_gid"].astype(int)
    return df


def neighbor_gids(gid):
    """Moore (8-connected) neighborhood on the real global PRIO-GRID (720 cols/row)."""
    return [gid + d for d in (-721, -720, -719, -1, 1, 719, 720, 721)]


def build_dataset():
    print("Loading real UCDP GED events...", flush=True)
    df = load_raw()
    df = df[df["date_start"] >= HISTORY_START].reset_index(drop=True)
    print(f"{len(df)} real events, {df['country'].nunique()} countries, "
          f"{df['priogrid_gid'].nunique()} distinct grid cells with any activity", flush=True)

    cell_counts = df["priogrid_gid"].value_counts()
    active_cells = sorted(cell_counts[cell_counts >= MIN_CELL_EVENTS].index.tolist())
    print(f"{len(active_cells)} active cells (>= {MIN_CELL_EVENTS} real events in the window)", flush=True)

    cell_meta = (df[df["priogrid_gid"].isin(active_cells)]
                 .groupby("priogrid_gid")
                 .agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                      country=("country", lambda s: s.mode().iat[0]),
                      n_events_total=("id", "count"))
                 .reset_index())
    cell_meta.to_csv("../data/active_cells.csv", index=False)

    issue_dates = pd.date_range(FIRST_ISSUE_DATE, LAST_ISSUE_DATE, freq="W-MON")
    print(f"{len(issue_dates)} weekly issue dates, {len(active_cells)} active cells "
          f"-> up to {len(issue_dates) * len(active_cells)} candidate rows", flush=True)

    # Precompute, per cell, a SORTED numpy datetime64 array of that cell's real
    # event dates -- turns "how many events before date X" into one searchsorted
    # call (O(log n)) instead of a full boolean scan repeated per issue date.
    cell_dates = {}
    cell_actor_sets = {}
    cell_type_of_violence = {}
    for gid, g in df.groupby("priogrid_gid"):
        g = g.sort_values("date_start")
        cell_dates[gid] = g["date_start"].values.astype("datetime64[D]")
        cell_actor_sets[gid] = list(zip(g["side_a"].values, g["side_b"].values))
        cell_type_of_violence[gid] = g["type_of_violence"].values

    country_dates = {c: np.sort(g["date_start"].values.astype("datetime64[D]"))
                      for c, g in df.groupby("country")}
    cell_to_country = dict(zip(cell_meta["priogrid_gid"], cell_meta["country"]))
    neighbor_map = {gid: [g for g in neighbor_gids(gid) if g in cell_dates] for gid in active_cells}

    issue_dates_np = issue_dates.values.astype("datetime64[D]")

    rows = []
    for i, (issue_date, issue_d64) in enumerate(zip(issue_dates, issue_dates_np)):
        if i % 100 == 0:
            print(f"  issue_date {i+1}/{len(issue_dates)} ({issue_date.date()})...", flush=True)
        for gid in active_cells:
            dates = cell_dates[gid]
            n_hist = int(np.searchsorted(dates, issue_d64, side="left"))
            if n_hist == 0:
                continue  # no history yet at this point in time -- not a fair candidate

            hist_dates = dates[:n_hist]
            c30 = n_hist - int(np.searchsorted(hist_dates, issue_d64 - np.timedelta64(30, "D")))
            c60 = n_hist - int(np.searchsorted(hist_dates, issue_d64 - np.timedelta64(60, "D")))
            c90 = n_hist - int(np.searchsorted(hist_dates, issue_d64 - np.timedelta64(90, "D")))
            c365 = n_hist - int(np.searchsorted(hist_dates, issue_d64 - np.timedelta64(365, "D")))
            days_since_last = int((issue_d64 - hist_dates[-1]).astype("timedelta64[D]").astype(int))

            neigh_30d = 0
            for n_gid in neighbor_map[gid]:
                ndates = cell_dates[n_gid]
                n_n = int(np.searchsorted(ndates, issue_d64, side="left"))
                if n_n:
                    nd = ndates[:n_n]
                    neigh_30d += n_n - int(np.searchsorted(nd, issue_d64 - np.timedelta64(30, "D")))

            country = cell_to_country[gid]
            cdates = country_dates[country]
            n_c = int(np.searchsorted(cdates, issue_d64, side="left"))
            country_30d = n_c - int(np.searchsorted(cdates[:n_c], issue_d64 - np.timedelta64(30, "D"))) if n_c else 0

            hist_actor_pairs = cell_actor_sets[gid][:n_hist]
            actor_diversity = len(set(a for pair in hist_actor_pairs for a in pair))
            tov_hist = cell_type_of_violence[gid][:n_hist]
            type_share_state_based = float((tov_hist == 1).mean())
            type_share_one_sided = float((tov_hist == 3).mean())

            row = {
                "issue_date": issue_date, "priogrid_gid": gid, "country": country,
                "cell_count_30d": c30, "cell_count_60d": c60,
                "cell_count_90d": c90, "cell_count_365d": c365,
                "days_since_last_event": days_since_last,
                "neighbor_count_30d": neigh_30d,
                "country_count_30d": country_30d,
                "actor_diversity": actor_diversity,
                "type_share_state_based": type_share_state_based,
                "type_share_one_sided": type_share_one_sided,
                "n_hist_events_total": n_hist,
            }
            for hname, hdays in HORIZONS.items():
                lo = int(np.searchsorted(dates, issue_d64, side="right"))
                hi = int(np.searchsorted(dates, issue_d64 + np.timedelta64(hdays, "D"), side="right"))
                row[f"label_{hname}"] = int(hi > lo)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("../data/discrete_event_candidates.csv", index=False)
    print(f"\nDone. {len(out)} candidate (cell, issue_date) rows.")
    for hname in HORIZONS:
        print(f"  {hname}: {out[f'label_{hname}'].sum()} positives "
              f"({out[f'label_{hname}'].mean()*100:.1f}% base rate)")
    return out


if __name__ == "__main__":
    build_dataset()

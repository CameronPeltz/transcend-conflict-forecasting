"""
Round 2: rebuilds the discrete-event candidate dataset with the real
new data acquired for this round:

  - UCDP GED v26.1 (was v25.1) -- a real, free, no-registration update
    (https://ucdp.uu.se/downloads/ged/ged261-csv.zip, verified
    downloadable directly, CC BY 4.0), extending real coverage from
    2024-12-31 to 2025-12-31. Same schema, same country/date/location
    fields, so the rest of the pipeline needed no structural changes.
  - ACLED civilian-targeting country-month event/fatality counts
    (real data, freely downloaded without registration from HDX:
    data.humdata.org/organization/acled -- the aggregated files are
    open; only ACLED's full disaggregated event-level export requires
    a free registered account, which was not created for this pass).
    An independent, differently-sourced cross-check signal at country-
    month granularity, joined in as an additional real feature.

Everything else (PRIO-GRID cell assignment/verification, never-look-
ahead searchsorted feature construction, the four original feature
groups) is unchanged from round 1's build_discrete_event_dataset.py.
"""
import numpy as np
import pandas as pd

RAW_PATH = "../data/pure_ucdp_v26/GEDEvent_v26_1.csv"
ACLED_PATH = "../data/acled_civilian_targeting_country_month.csv"
COUNTRIES_19 = [
    "Afghanistan", "Myanmar", "Pakistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan",
    "Sudan", "Ethiopia", "Somalia", "South Sudan", "Kenya", "Eritrea",
    "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia", "Haiti", "Nicaragua",
]
MIN_CELL_EVENTS = 15
HISTORY_START = "2013-01-01"
FIRST_ISSUE_DATE = "2015-01-05"
LAST_ISSUE_DATE = "2025-11-24"   # extended a real ~1 year vs round 1, leaves 14-day label buffer before v26.1's 2025-12-31 cutoff
HORIZONS = {"10day": 10, "14day": 14}

MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


def load_raw():
    cols = ["id", "country", "date_start", "date_prec", "where_prec", "latitude", "longitude",
            "priogrid_gid", "type_of_violence", "side_a", "side_b", "best"]
    df = pd.read_csv(RAW_PATH, usecols=cols)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df[df["country"].isin(COUNTRIES_19)]
    df = df[(df["date_prec"] <= 2) & (df["where_prec"] <= 3)]
    df = df.dropna(subset=["date_start", "priogrid_gid", "latitude", "longitude"])
    df["priogrid_gid"] = df["priogrid_gid"].astype(int)
    return df


def load_acled_lookup():
    """Real ACLED civilian-targeting Events/Fatalities, keyed by
    (country, year, month) -- a genuinely different, independently
    collected source from UCDP, used as a cross-check feature."""
    a = pd.read_csv(ACLED_PATH)
    a["month_num"] = a["Month"].map(MONTH_NUM)
    return {(row.Country, int(row.Year), int(row.month_num)): (row.Events, row.Fatalities)
            for row in a.itertuples()}


def neighbor_gids(gid):
    return [gid + d for d in (-721, -720, -719, -1, 1, 719, 720, 721)]


def build_dataset():
    print("Loading real UCDP GED v26.1 events...", flush=True)
    df = load_raw()
    df = df[df["date_start"] >= HISTORY_START].reset_index(drop=True)
    print(f"{len(df)} real events (v26.1), {df['country'].nunique()} countries, "
          f"{df['priogrid_gid'].nunique()} distinct grid cells with any activity", flush=True)

    print("Loading real ACLED civilian-targeting country-month lookup...", flush=True)
    acled_lookup = load_acled_lookup()
    print(f"{len(acled_lookup)} real (country, year, month) ACLED entries", flush=True)

    cell_counts = df["priogrid_gid"].value_counts()
    active_cells = sorted(cell_counts[cell_counts >= MIN_CELL_EVENTS].index.tolist())
    print(f"{len(active_cells)} active cells (>= {MIN_CELL_EVENTS} real events in the window)", flush=True)

    cell_meta = (df[df["priogrid_gid"].isin(active_cells)]
                 .groupby("priogrid_gid")
                 .agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                      country=("country", lambda s: s.mode().iat[0]),
                      n_events_total=("id", "count"))
                 .reset_index())
    cell_meta.to_csv("../data/active_cells_v2.csv", index=False)

    issue_dates = pd.date_range(FIRST_ISSUE_DATE, LAST_ISSUE_DATE, freq="W-MON")
    print(f"{len(issue_dates)} weekly issue dates, {len(active_cells)} active cells "
          f"-> up to {len(issue_dates) * len(active_cells)} candidate rows", flush=True)

    cell_dates, cell_actor_sets, cell_type_of_violence = {}, {}, {}
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
        # ACLED signal for "last calendar month" strictly before this issue date
        prev_month_date = issue_date.replace(day=1) - pd.Timedelta(days=1)
        for gid in active_cells:
            dates = cell_dates[gid]
            n_hist = int(np.searchsorted(dates, issue_d64, side="left"))
            if n_hist == 0:
                continue

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

            acled_events, acled_fatalities = acled_lookup.get(
                (country, prev_month_date.year, prev_month_date.month), (0, 0))

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
                "acled_civ_targeting_events_prevmonth": acled_events,
                "acled_civ_targeting_fatalities_prevmonth": acled_fatalities,
                "n_hist_events_total": n_hist,
            }
            for hname, hdays in HORIZONS.items():
                lo = int(np.searchsorted(dates, issue_d64, side="right"))
                hi = int(np.searchsorted(dates, issue_d64 + np.timedelta64(hdays, "D"), side="right"))
                row[f"label_{hname}"] = int(hi > lo)
                # also record the FIRST future event's type_of_violence for the
                # event-type-generalization test (round 2 addition)
                tov = cell_type_of_violence[gid]
                if hi > lo:
                    row[f"label_{hname}_type"] = int(tov[lo])
                else:
                    row[f"label_{hname}_type"] = 0
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("../data/discrete_event_candidates_v2.csv", index=False)
    print(f"\nDone. {len(out)} candidate (cell, issue_date) rows.")
    for hname in HORIZONS:
        print(f"  {hname}: {out[f'label_{hname}'].sum()} positives "
              f"({out[f'label_{hname}'].mean()*100:.1f}% base rate)")
    return out


if __name__ == "__main__":
    build_dataset()

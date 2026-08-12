"""
Two structurally new feature families, computed directly from the real
raw UCDP GED v26.1 event timestamps already on disk (no new external
data):

  1. Hawkes-kernel intensity -- round 1-3's cell_count_30d/60d/90d/365d
     features are all fixed-window box-car counts (every event in the
     window counts equally, then drops to zero the instant it exits the
     window). A self-exciting point process (Hawkes process) instead
     weights each past event by a smooth exponential decay from the
     forecast date -- the academic standard for this exact class of
     problem (Lewis, Mohler et al.'s work on civilian conflict deaths
     and gang violence). This computes the excitation-kernel intensity
     directly -- sum of exp(-beta * age_in_days) over every real past
     event in the cell, at three different decay rates (fast/medium/
     slow, corresponding to roughly 3/14/30-day half-lives) -- plus the
     same kernel applied to neighboring cells' events for spatial
     excitation. This is the Hawkes kernel itself used as an engineered
     feature for the existing ensemble, not a full separately-fit
     Hawkes-process likelihood model; that scoping is disclosed as such
     in the write-up.

  2. CUSUM changepoint statistic -- round 3's momentum feature compares
     two fixed windows (a single potentially noisy comparison). A CUSUM
     (cumulative sum control chart) instead accumulates evidence of a
     sustained shift in a cell's weekly event rate over many weeks,
     which is exactly what's needed to catch the diagnosed round-2/3
     failure mode: a cell whose conflict dynamics genuinely changed
     (the August 2021 Afghanistan regime change), not just a single busy
     or quiet week.
"""
import numpy as np
import pandas as pd

RAW_PATH = "../data/pure_ucdp_v26/GEDEvent_v26_1.csv"
CANDIDATES_PATH = "../data/discrete_event_candidates_v3.csv"
OUT_PATH = "../data/discrete_event_candidates_v6_hawkes_cusum.csv"
COUNTRIES_19 = [
    "Afghanistan", "Myanmar", "Pakistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan",
    "Sudan", "Ethiopia", "Somalia", "South Sudan", "Kenya", "Eritrea",
    "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia", "Haiti", "Nicaragua",
]
DECAY_RATES = {"fast": np.log(2) / 3, "medium": np.log(2) / 14, "slow": np.log(2) / 30}  # half-lives in days
CUSUM_TARGET_MEAN_WEEKS = 12   # baseline window for the CUSUM reference rate
CUSUM_RECENT_WEEKS = 4         # recent window being tested for a shift


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
        g = g.sort_values("date_start")
        cell_dates[gid] = g["date_start"].values.astype("datetime64[D]")

    active_cells = cand["priogrid_gid"].unique()
    neighbor_map = {gid: [g for g in neighbor_gids(gid) if g in cell_dates] for gid in active_cells}

    issue_dates = sorted(cand["issue_date"].unique())
    new_rows = {}
    for i, issue_d in enumerate(issue_dates):
        if i % 100 == 0:
            print(f"  {i+1}/{len(issue_dates)} issue dates...", flush=True)
        issue_d64 = np.datetime64(issue_d, "D")
        cells_today = cand.loc[cand["issue_date"] == issue_d, "priogrid_gid"].values
        for gid in cells_today:
            dates = cell_dates.get(gid)
            if dates is None:
                continue
            n_hist = int(np.searchsorted(dates, issue_d64, side="left"))
            hist_dates = dates[:n_hist]
            # cap history considered to the last 200 days for tractability -- events
            # older than that contribute negligibly to any of these decay rates anyway
            recent_cutoff = issue_d64 - np.timedelta64(200, "D")
            recent_start = int(np.searchsorted(hist_dates, recent_cutoff))
            recent_hist = hist_dates[recent_start:]
            ages_days = (issue_d64 - recent_hist).astype("timedelta64[D]").astype(float)

            row = {}
            for name, beta in DECAY_RATES.items():
                row[f"hawkes_self_{name}"] = float(np.sum(np.exp(-beta * ages_days))) if len(ages_days) else 0.0

            neighbor_ages = []
            for n_gid in neighbor_map.get(gid, []):
                ndates = cell_dates.get(n_gid)
                if ndates is None:
                    continue
                n_n = int(np.searchsorted(ndates, issue_d64, side="left"))
                nd = ndates[:n_n]
                n_start = int(np.searchsorted(nd, recent_cutoff))
                nd_recent = nd[n_start:]
                if len(nd_recent):
                    neighbor_ages.append((issue_d64 - nd_recent).astype("timedelta64[D]").astype(float))
            if neighbor_ages:
                all_neighbor_ages = np.concatenate(neighbor_ages)
                for name, beta in DECAY_RATES.items():
                    row[f"hawkes_neighbor_{name}"] = float(np.sum(np.exp(-beta * all_neighbor_ages)))
            else:
                for name in DECAY_RATES:
                    row[f"hawkes_neighbor_{name}"] = 0.0

            # CUSUM: weekly counts for the baseline window and the recent window
            baseline_lo = issue_d64 - np.timedelta64(7 * (CUSUM_TARGET_MEAN_WEEKS + CUSUM_RECENT_WEEKS), "D")
            baseline_hi = issue_d64 - np.timedelta64(7 * CUSUM_RECENT_WEEKS, "D")
            recent_lo = issue_d64 - np.timedelta64(7 * CUSUM_RECENT_WEEKS, "D")

            baseline_n = int(np.searchsorted(hist_dates, baseline_hi)) - int(np.searchsorted(hist_dates, baseline_lo))
            baseline_weekly_rate = baseline_n / max(CUSUM_TARGET_MEAN_WEEKS, 1)
            recent_n = n_hist - int(np.searchsorted(hist_dates, recent_lo))
            recent_weekly_rate = recent_n / max(CUSUM_RECENT_WEEKS, 1)

            # per-week CUSUM: accumulate (actual - baseline_rate) over the recent weeks
            cusum_pos, cusum_neg, running = 0.0, 0.0, 0.0
            for w in range(CUSUM_RECENT_WEEKS):
                wk_hi = issue_d64 - np.timedelta64(7 * w, "D")
                wk_lo = issue_d64 - np.timedelta64(7 * (w + 1), "D")
                wk_n = int(np.searchsorted(hist_dates, wk_hi)) - int(np.searchsorted(hist_dates, wk_lo))
                running += (wk_n - baseline_weekly_rate)
                cusum_pos = max(cusum_pos, running)
                cusum_neg = min(cusum_neg, running)

            row["cusum_upward"] = cusum_pos
            row["cusum_downward"] = cusum_neg
            row["cusum_baseline_rate"] = baseline_weekly_rate
            row["cusum_recent_rate"] = recent_weekly_rate
            row["cusum_rate_ratio"] = recent_weekly_rate / (baseline_weekly_rate + 0.1)

            new_rows[(issue_d, gid)] = row

    new_df = pd.DataFrame([{"issue_date": k[0], "priogrid_gid": k[1], **v} for k, v in new_rows.items()])
    out = cand.merge(new_df, on=["issue_date", "priogrid_gid"], how="left")
    out.to_csv(OUT_PATH, index=False)
    print(f"\nDone. {len(out)} rows written to {OUT_PATH}", flush=True)
    print(out[["hawkes_self_medium", "hawkes_neighbor_medium", "cusum_upward", "cusum_downward"]].describe(), flush=True)


if __name__ == "__main__":
    main()

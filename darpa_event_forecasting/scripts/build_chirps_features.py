"""
Extracts real CHIRPS monthly rainfall for each of the 464 real
PRIO-GRID cells (windowed mean over each cell's real 0.5-degree
footprint), for every real month January 2015 through November 2025,
then joins onto the candidate dataset using only the calendar month
strictly before each row's own issue_date -- never-look-ahead, same
discipline as every other join in this project.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

CELLS_PATH = "../data/active_cells_v2.csv"
CANDIDATES_PATH = "../data/discrete_event_candidates_v4.csv"
OUT_PANEL_PATH = "../data/chirps_cell_month_panel.csv"
OUT_CANDIDATES_PATH = "../data/discrete_event_candidates_v5.csv"
HALF_DEG = 0.25
MONTHS = [(y, m) for y in range(2015, 2026) for m in range(1, 13) if not (y == 2025 and m > 11)]


def cell_rainfall(src, lat, lon):
    bounds = (lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG)
    try:
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=-9999)
        data = data[data > -1]  # CHIRPS nodata is -9999
        if data.size == 0:
            return np.nan
        return float(np.nanmean(data))
    except Exception:
        return np.nan


def build_panel(cells):
    rows = []
    for year, month in MONTHS:
        path = f"../data_raw/chirps/chirps-v2.0.{year}.{month:02d}.tif"
        try:
            src = rasterio.open(path)
        except Exception as e:
            print(f"  {year}-{month:02d}: could not open ({e})", flush=True)
            continue
        for _, row in cells.iterrows():
            rf = cell_rainfall(src, row["lat"], row["lon"])
            rows.append({"priogrid_gid": row["priogrid_gid"], "year": year, "month": month, "chirps_rainfall_mm": rf})
        src.close()
        if month == 1:
            print(f"  {year}-{month:02d}...", flush=True)
    return pd.DataFrame(rows)


def main():
    cells = pd.read_csv(CELLS_PATH)
    print(f"Building CHIRPS cell-month panel for {len(cells)} cells x {len(MONTHS)} months...", flush=True)
    panel = build_panel(cells)
    panel = panel.sort_values(["priogrid_gid", "year", "month"])
    panel["chirps_rainfall_mm_lag1"] = panel.groupby("priogrid_gid")["chirps_rainfall_mm"].shift(1)
    panel["chirps_rainfall_anomaly"] = panel["chirps_rainfall_mm"] - panel.groupby("priogrid_gid")["chirps_rainfall_mm"].transform("mean")
    panel.to_csv(OUT_PANEL_PATH, index=False)
    print(f"Wrote {OUT_PANEL_PATH}: {len(panel)} cell-month rows", flush=True)

    print("Joining onto candidates (never-look-ahead: prior completed calendar month only)...", flush=True)
    cand = pd.read_csv(CANDIDATES_PATH, parse_dates=["issue_date"])
    prior_month_date = cand["issue_date"] - pd.Timedelta(days=1)
    cand["chirps_join_year"] = prior_month_date.dt.year
    cand["chirps_join_month"] = prior_month_date.dt.month

    join_cols = ["priogrid_gid", "year", "month", "chirps_rainfall_mm", "chirps_rainfall_mm_lag1", "chirps_rainfall_anomaly"]
    panel_join = panel[join_cols].rename(columns={"year": "chirps_join_year", "month": "chirps_join_month"})
    out = cand.merge(panel_join, on=["priogrid_gid", "chirps_join_year", "chirps_join_month"], how="left")
    n_matched = out["chirps_rainfall_mm"].notna().sum()
    print(f"  {n_matched}/{len(out)} rows ({n_matched/len(out)*100:.1f}%) matched to real CHIRPS data", flush=True)
    out.to_csv(OUT_CANDIDATES_PATH, index=False)
    print(f"Wrote {OUT_CANDIDATES_PATH}", flush=True)


if __name__ == "__main__":
    main()

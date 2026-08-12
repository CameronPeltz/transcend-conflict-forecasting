"""
Extracts real WorldPop population totals for each of the 464 real
PRIO-GRID cells used throughout this project, by windowed-summing the
real 100m population raster over each cell's real 0.5-degree footprint
(centered on the cell's own real centroid, already computed and
verified in active_cells_v2.csv). A static, one-time structural
feature -- population doesn't meaningfully change week to week -- so
this is a single column merged onto every candidate row for its cell,
no time-series join needed (no look-ahead risk: population level is
essentially fixed background context, the same category of signal
PRIO-GRID's own covariates would provide).
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

ISO3 = {
    "Afghanistan": "AFG", "Myanmar": "MMR", "Pakistan": "PAK", "Tajikistan": "TJK",
    "Kyrgyzstan": "KGZ", "Uzbekistan": "UZB",
    "Sudan": "SDN", "Ethiopia": "ETH", "Somalia": "SOM", "South Sudan": "SSD",
    "Kenya": "KEN", "Eritrea": "ERI",
    "Colombia": "COL", "Venezuela": "VEN", "Ecuador": "ECU", "Peru": "PER", "Bolivia": "BOL",
    "Haiti": "HTI", "Nicaragua": "NIC",
}
CELLS_PATH = "../data/active_cells_v2.csv"
OUT_PATH = "../data/worldpop_cell_population.csv"
HALF_DEG = 0.25  # PRIO-GRID cells are 0.5deg x 0.5deg


def cell_population(src, lat, lon):
    bounds = (lon - HALF_DEG, lat - HALF_DEG, lon + HALF_DEG, lat + HALF_DEG)
    try:
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=0)
        data = np.where(data < 0, 0, data)  # WorldPop uses negative nodata sentinels
        return float(np.nansum(data))
    except Exception:
        return np.nan


def main():
    cells = pd.read_csv(CELLS_PATH)
    results = []
    for country, group in cells.groupby("country"):
        iso3 = ISO3.get(country)
        if iso3 is None:
            continue
        raster_path = f"../data_raw/worldpop/{iso3}_ppp_2020.tif"
        try:
            src = rasterio.open(raster_path)
            src.read(1)  # full-band read: catches partial/still-downloading files that open() alone won't
        except Exception as e:
            print(f"  {country}: raster not fully downloaded/readable yet, skipping ({e})", flush=True)
            continue
        for _, row in group.iterrows():
            pop = cell_population(src, row["lat"], row["lon"])
            results.append({"priogrid_gid": row["priogrid_gid"], "worldpop_population": pop})
        src.close()
        print(f"  {country}: {len(group)} cells processed", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(out)} cells, "
          f"median population {out['worldpop_population'].median():.0f}, "
          f"{out['worldpop_population'].isna().sum()} failed extractions", flush=True)


if __name__ == "__main__":
    main()

"""
Downloads real CHIRPS 2.0 global monthly rainfall GeoTIFFs
(data.chc.ucsb.edu, public domain / free, no registration, verified
live before use), January 2015 through November 2025 -- matching this
project's own issue-date range. ~130 files, ~14MB each (gzipped).
"""
import gzip
import os
import urllib.request

BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif.gz"
OUT_DIR = "../data_raw/chirps"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    months = [(y, m) for y in range(2015, 2026) for m in range(1, 13) if not (y == 2025 and m > 11)]
    for year, month in months:
        out_path = os.path.join(OUT_DIR, f"chirps-v2.0.{year}.{month:02d}.tif")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            continue
        url = BASE.format(year=year, month=month)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                gz_data = resp.read()
            tif_data = gzip.decompress(gz_data)
            with open(out_path, "wb") as f:
                f.write(tif_data)
            print(f"  {year}-{month:02d}: {len(tif_data)/1e6:.1f} MB", flush=True)
        except Exception as e:
            print(f"  {year}-{month:02d}: FAILED ({e})", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

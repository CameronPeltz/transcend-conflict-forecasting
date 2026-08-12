"""
Historical (2015-2025) GDELT 2.0 event pull across the same 19 countries
used throughout this project, adapted directly from the parent project's
scripts/download_gdelt_large.py (which pulled 3 years/1095 days -- this
pulls the full ~11-year range the discrete-event task's issue dates span,
2015-01-01 through 2025-11-24, matching build_discrete_event_dataset_v2.py's
FIRST_ISSUE_DATE/LAST_ISSUE_DATE).

Same real, free, no-auth source (data.gdeltproject.org), same 58-field
schema, same streaming day-by-day approach so the full multi-year pull
never needs to hold raw text for more than one day in memory at a time.
Long-running by design (an order of magnitude more days than the parent
project's 3-year pull, which took ~19 minutes) -- meant to run in the
background with progress checkpointed to disk.
"""
import io
import sys
import time
import urllib.request
import zipfile
from datetime import date, timedelta

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BASE = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"

TARGET_COUNTRIES = {
    "SU", "ET", "SO", "OD", "KE", "ER",       # East/NE Africa
    "AF", "BM", "PK", "TI", "KG", "UZ",       # Central/SE Asia
    "CO", "VE", "EC", "PE", "BL",             # South America
    "HA", "NU",                                # extras, disclosed
}

COLS_KEEP = [0, 1, 5, 6, 7, 15, 16, 17, 26, 29, 30, 31, 32, 33, 34, 51, 53, 54]
COLS_HEADER = [
    "GlobalEventID", "SQLDATE", "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "EventCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "ActionGeo_CountryCode", "ActionGeo_Lat", "ActionGeo_Long",
]

START = date(2015, 1, 1)
END = date(2025, 11, 24)  # matches build_discrete_event_dataset_v2.py's LAST_ISSUE_DATE


def fetch_day(day_str, retries=4):
    url = BASE.format(date=day_str)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  {day_str}: FAILED ({e})", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def main():
    days = []
    d = START
    while d <= END:
        days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    out_path = "../data_raw/gdelt_historical_19country_raw.csv"
    progress_path = "../data_raw/gdelt_historical_download_progress.txt"
    total_kept = 0
    failed_days = []
    t0 = time.time()
    print(f"Pulling {len(days)} days ({START} to {END}), 19 countries, real GDELT 2.0 daily export files...", flush=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(",".join(COLS_HEADER) + "\n")
        for i, day_str in enumerate(days):
            raw = fetch_day(day_str)
            if raw is None:
                failed_days.append(day_str)
                continue
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
                name = zf.namelist()[0]
                content = zf.read(name).decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  {day_str}: zip error ({e})", file=sys.stderr)
                failed_days.append(day_str)
                continue

            kept_today = 0
            for line in content.split("\n"):
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) < 58:
                    continue
                if fields[51] not in TARGET_COUNTRIES:
                    continue
                row = [fields[c] for c in COLS_KEEP]
                row = [f'"{v}"' if "," in v else v for v in row]
                out.write(",".join(row) + "\n")
                kept_today += 1
            total_kept += kept_today

            if i % 50 == 0 or i == len(days) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta_min = (len(days) - i - 1) / rate / 60 if rate > 0 else float("nan")
                msg = (f"[{i+1}/{len(days)}] {day_str}: {kept_today} rows "
                       f"(total {total_kept}, {len(failed_days)} failed days, "
                       f"elapsed {elapsed/60:.1f}m, ETA {eta_min:.1f}m)")
                print(msg, flush=True)
                with open(progress_path, "w") as pf:
                    pf.write(msg + "\n")
            time.sleep(0.12)

    final_msg = (f"DONE. wrote {out_path}, {total_kept} total rows across "
                 f"{len(days)} days ({len(failed_days)} failed: {failed_days[:20]}...)")
    print(final_msg, flush=True)
    with open(progress_path, "a") as pf:
        pf.write(final_msg + "\n")


if __name__ == "__main__":
    main()

"""
The "way bigger" self-scraped GDELT track, kept in data/scraped_large/,
distinct from both the original small GDELT panel (data/gdelt_filtered_v2.csv,
6 countries x 180 days) and the UCDP GED "pure" track (data/pure_ucdp/).

Real GDELT 2.0 daily event export files (data.gdeltproject.org, free, no
auth), same confirmed 58-field schema as download_gdelt.py, expanded to:
  - 19 countries (FIPS 10-4 codes, verified against gdeltproject.org's own
    FIPS.country.txt lookup) across DARPA's 3 named regions, plus two
    extras (Haiti, Nicaragua) added transparently for label balance --
    high-instability countries outside the three named regions, not
    hidden as if they were in-scope.
  - 3 years (1095 days) instead of 90/180 days.

East/NE Africa: SU Sudan, ET Ethiopia, SO Somalia, OD South Sudan, KE Kenya, ER Eritrea
Central/SE Asia: AF Afghanistan, BM Myanmar, PK Pakistan, TI Tajikistan, KG Kyrgyzstan, UZ Uzbekistan
South America:   CO Colombia, VE Venezuela, EC Ecuador, PE Peru, BL Bolivia
Extra (label balance, disclosed): HA Haiti, NU Nicaragua

This alone does not fit the "way huger" ask into RAM as raw text -- it
streams day by day, filters to these 19 countries on the fly, and only
persists the filtered rows, same as the original script.
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
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    end = date(2026, 8, 5)
    days = [(end - timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)]
    days = sorted(days)  # oldest first, so a partial run still gives a contiguous usable window

    out_path = "data/scraped_large/gdelt_large_raw.csv"
    progress_path = "data/scraped_large/download_progress.txt"
    total_kept = 0
    failed_days = []
    t0 = time.time()
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

            if i % 20 == 0 or i == len(days) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta_min = (len(days) - i - 1) / rate / 60 if rate > 0 else float("nan")
                msg = (f"[{i+1}/{len(days)}] {day_str}: {kept_today} rows "
                       f"(total {total_kept}, {len(failed_days)} failed days, "
                       f"elapsed {elapsed/60:.1f}m, ETA {eta_min:.1f}m)")
                print(msg)
                with open(progress_path, "w") as pf:
                    pf.write(msg + "\n")
            time.sleep(0.12)

    final_msg = (f"DONE. wrote {out_path}, {total_kept} total rows across "
                 f"{len(days)} days ({len(failed_days)} failed: {failed_days[:20]}...)")
    print(final_msg)
    with open(progress_path, "a") as pf:
        pf.write(final_msg + "\n")


if __name__ == "__main__":
    main()

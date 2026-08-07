"""
Real GDELT 2.0 scrape for the Democratic Republic of the Congo (FIPS
code CG -- confirmed against gdeltproject.org's own FIPS.country.txt
lookup; not to be confused with CF, the Republic of the Congo /
Congo-Brazzaville). DRC was not in the original 19-country track --
this is a new country added specifically to test the real Rootwise
DRC radio-transcript data against.

Same free, no-auth GDELT 2.0 daily export pattern as
download_gdelt_large.py, 3 years for consistency with that track.
"""
import io
import sys
import time
import urllib.request
import zipfile
from datetime import date, timedelta

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BASE = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"
TARGET_COUNTRIES = {"CG"}

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
    end = date(2026, 8, 7)
    days = sorted([(end - timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)])

    out_path = "data/drc/gdelt_drc_raw.csv"
    progress_path = "data/drc/download_progress.txt"
    total_kept = 0
    failed = []
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(",".join(COLS_HEADER) + "\n")
        for i, day_str in enumerate(days):
            raw = fetch_day(day_str)
            if raw is None:
                failed.append(day_str)
                continue
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
                content = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  {day_str}: zip error ({e})", file=sys.stderr)
                failed.append(day_str)
                continue
            kept_today = 0
            for line in content.split("\n"):
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) < 58 or fields[51] not in TARGET_COUNTRIES:
                    continue
                row = [fields[c] for c in COLS_KEEP]
                row = [f'"{v}"' if "," in v else v for v in row]
                out.write(",".join(row) + "\n")
                kept_today += 1
            total_kept += kept_today
            if i % 20 == 0 or i == len(days) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(days) - i - 1) / rate / 60 if rate > 0 else float("nan")
                msg = f"[{i+1}/{len(days)}] {day_str}: total {total_kept} rows, {len(failed)} failed, elapsed {elapsed/60:.1f}m ETA {eta:.1f}m"
                print(msg)
                with open(progress_path, "w") as pf:
                    pf.write(msg + "\n")
            time.sleep(0.12)

    final = f"DONE. {total_kept} rows across {len(days)} days ({len(failed)} failed)"
    print(final)
    with open(progress_path, "a") as pf:
        pf.write(final + "\n")


if __name__ == "__main__":
    main()

"""
Download real GDELT 2.0 daily event export files (data.gdeltproject.org,
free, no auth) for a real recent window, filtering on the fly to four
countries spanning DARPA's three named target regions:
  Sudan (SU)       -- East/Northeast Africa
  Myanmar (BM)     -- Southeast Asia
  Afghanistan (AF) -- Central Asia
  Colombia (CO)    -- South America
Confirmed live GDELT 2.0 schema (58 tab-delimited fields, 0-indexed):
  0 GlobalEventID, 1 SQLDATE, ... 29 QuadClass, 30 GoldsteinScale,
  31 NumMentions, 32 NumSources, 33 NumArticles, 34 AvgTone,
  ... 51 ActionGeo_CountryCode, ... 57 SOURCEURL

Round 2: expanded to 2 countries per RFP region (Sudan+Ethiopia for
East/NE Africa, Afghanistan+Myanmar for Central/SE Asia, Colombia+
Venezuela for South America) and 180 days, to get a big enough sample
for accuracy/specificity to mean something.
"""
import io
import sys
import time
import urllib.request
import zipfile
from datetime import date, timedelta

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BASE = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"
TARGET_COUNTRIES = {"SU", "ET", "AF", "BM", "CO", "VE"}

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
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    end = date(2026, 8, 1)
    days = [(end - timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)]

    out_path = "data/gdelt_filtered_v2.csv"
    total_kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(",".join(COLS_HEADER) + "\n")
        for i, day_str in enumerate(days):
            raw = fetch_day(day_str)
            if raw is None:
                continue
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
                name = zf.namelist()[0]
                content = zf.read(name).decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  {day_str}: zip error ({e})", file=sys.stderr)
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
            print(f"  {day_str}: {kept_today} matching rows (running total {total_kept})")
            time.sleep(0.15)

    print(f"DONE. wrote {out_path}, {total_kept} total rows across {len(days)} days")


if __name__ == "__main__":
    main()

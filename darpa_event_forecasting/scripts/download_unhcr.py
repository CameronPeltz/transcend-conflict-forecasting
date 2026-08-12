"""
Pulls real UNHCR asylum-application counts by country of origin (coo),
2013-2025, for all 19 project countries -- api.unhcr.org, free, no
credentials, verified live before use. Used as a country-level,
differently-biased displacement/conflict-flow signal: unlike GDELT/UCDP/
ACLED, this is data collected specifically because of forced
displacement, not a general news or event feed.
"""
import json
import time
import urllib.request

ISO3 = {
    "Afghanistan": "AFG", "Myanmar": "MMR", "Pakistan": "PAK", "Tajikistan": "TJK",
    "Kyrgyzstan": "KGZ", "Uzbekistan": "UZB",
    "Sudan": "SDN", "Ethiopia": "ETH", "Somalia": "SOM", "South Sudan": "SSD",
    "Kenya": "KEN", "Eritrea": "ERI",
    "Colombia": "COL", "Venezuela": "VEN", "Ecuador": "ECU", "Peru": "PER", "Bolivia": "BOL",
    "Haiti": "HTI", "Nicaragua": "NIC",
}
BASE = "https://api.unhcr.org/population/v1/asylum-applications/?limit=1000&yearFrom={y0}&yearTo={y1}&coo={iso3}"
OUT_PATH = "../data_raw/unhcr_asylum_applications.json"


def main():
    all_rows = []
    for country, iso3 in ISO3.items():
        url = BASE.format(y0=2013, y1=2025, iso3=iso3)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("items", [])
            for it in items:
                it["country_full"] = country
            all_rows.extend(items)
            print(f"  {country} ({iso3}): {len(items)} real year-records", flush=True)
        except Exception as e:
            print(f"  {country} ({iso3}): FAILED ({e})", flush=True)
        time.sleep(0.3)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nDone. {len(all_rows)} total real records written to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

"""
Real daily precipitation + temperature for each target country's capital
(or largest commercial city, where more representative of population/
economic activity), pulled live from NASA POWER (power.larc.nasa.gov) --
free, no API key, real MERRA-2 reanalysis data. No simulation: every
value in the output CSV is a real API response.

Rainfall deficit is a documented real leading indicator in the conflict-
climate literature (drought -> agricultural/pastoral stress -> resource
competition), which is exactly why it's being added here as a candidate
leading indicator to Granger-test against the existing escalation label.
"""
import time
import requests
import pandas as pd

# capital or largest commercial city per country, real coordinates
COUNTRY_POINTS = {
    "SU": ("Khartoum", 15.5007, 32.5599),
    "ET": ("Addis Ababa", 9.0250, 38.7469),
    "AF": ("Kabul", 34.5553, 69.2075),
    "BM": ("Yangon", 16.8409, 96.1735),   # commercial capital; more population-weighted than Naypyidaw
    "CO": ("Bogota", 4.7110, -74.0721),
    "VE": ("Caracas", 10.4806, -66.9036),
}

START = "20260101"
END = "20260802"  # covers the full existing GDELT panel window with margin

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_country(code, name, lat, lon):
    params = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START,
        "end": END,
        "format": "JSON",
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    dates = sorted(data["PRECTOTCORR"].keys())
    rows = []
    for d in dates:
        precip = data["PRECTOTCORR"][d]
        temp = data["T2M"][d]
        if precip == -999.0:
            precip = None
        if temp == -999.0:
            temp = None
        rows.append({"country": code, "city": name, "date": d, "precip_mm": precip, "temp_c": temp})
    return rows


def main():
    all_rows = []
    for code, (name, lat, lon) in COUNTRY_POINTS.items():
        print(f"fetching NASA POWER climate for {name} ({code})...")
        rows = fetch_country(code, name, lat, lon)
        print(f"  {len(rows)} real daily records")
        all_rows.extend(rows)
        time.sleep(1)  # be polite to a free public API

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.to_csv("data/climate_daily.csv", index=False)
    print(f"\nwrote data/climate_daily.csv ({len(df)} real rows, {df['country'].nunique()} countries)")


if __name__ == "__main__":
    main()

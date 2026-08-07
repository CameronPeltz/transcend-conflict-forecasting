"""
Real World Bank World Development Indicators (WDI) for each target
country, pulled live from api.worldbank.org -- free, no API key.

These are annual, slow-moving structural indicators (inflation, food
inflation, GDP growth, unemployment) rather than weekly leading
indicators -- they don't vary within the ~26-week GDELT panel, so they
function as a real per-country "structural fragility" prior, the same
role country identity itself played in iteration #2 of the prior 24-
iteration study, not as Granger-testable time series.

A genuinely interesting, disclosed real finding while pulling this: how
recent the *most recent available* data point is varies enormously by
country -- itself a real signal of state capacity / reporting collapse,
worth keeping in the output rather than silently backfilling.
"""
import time
import requests
import pandas as pd

COUNTRY_ISO3 = {"SU": "SDN", "ET": "ETH", "AF": "AFG", "BM": "MMR", "CO": "COL", "VE": "VEN"}

INDICATORS = {
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "SI.POV.GINI": "gini_index",
}

BASE_URL = "https://api.worldbank.org/v2/country/{iso3}/indicator/{ind}"


def fetch_latest(iso3, ind):
    r = requests.get(BASE_URL.format(iso3=iso3, ind=ind), params={"format": "json", "mrv": 1}, timeout=30)
    r.raise_for_status()
    js = r.json()
    if len(js) < 2 or not js[1]:
        return None, None
    rec = js[1][0]
    return rec["date"], rec["value"]


def main():
    rows = []
    for code, iso3 in COUNTRY_ISO3.items():
        row = {"country": code, "iso3": iso3}
        print(f"fetching World Bank WDI for {code} ({iso3})...")
        for ind, col in INDICATORS.items():
            year, val = fetch_latest(iso3, ind)
            row[col] = val
            row[f"{col}_year"] = year
            time.sleep(0.3)
        print(f"  {row}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("data/worldbank_structural.csv", index=False)
    print(f"\nwrote data/worldbank_structural.csv ({len(df)} real country rows)")


if __name__ == "__main__":
    main()

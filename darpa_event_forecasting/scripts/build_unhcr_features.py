"""
Builds a country-year asylum-application feature from the real UNHCR
pull (download_unhcr.py) and joins it onto the candidate dataset using
only the prior COMPLETED year's total -- e.g. any issue_date in 2022
sees only the finalized 2021 total, never same-year or future data, to
stay strictly never-look-ahead given this source's annual granularity.
"""
import json
import pandas as pd

RAW_PATH = "../data_raw/unhcr_asylum_applications.json"
CANDIDATES_PATH = "../data/discrete_event_candidates_v3.csv"
OUT_PATH = "../data/discrete_event_candidates_v8_unhcr.csv"


def main():
    print("Loading real UNHCR data...", flush=True)
    with open(RAW_PATH, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["applied"] = pd.to_numeric(df["applied"], errors="coerce").fillna(0)
    yearly = df.groupby(["country_full", "year"])["applied"].sum().reset_index()
    yearly = yearly.rename(columns={"country_full": "country", "applied": "unhcr_asylum_applications"})
    print(f"{len(yearly)} real country-year records, {yearly['country'].nunique()} countries with data", flush=True)

    cand = pd.read_csv(CANDIDATES_PATH, parse_dates=["issue_date"])
    cand["prior_year"] = cand["issue_date"].dt.year - 1
    yearly_join = yearly.rename(columns={"year": "prior_year"})
    out = cand.merge(yearly_join[["country", "prior_year", "unhcr_asylum_applications"]],
                      on=["country", "prior_year"], how="left")
    n_matched = out["unhcr_asylum_applications"].notna().sum()
    print(f"{n_matched}/{len(out)} rows ({n_matched/len(out)*100:.1f}%) matched to real UNHCR prior-year data "
          f"(remaining rows are countries/years with no UNHCR record returned)", flush=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

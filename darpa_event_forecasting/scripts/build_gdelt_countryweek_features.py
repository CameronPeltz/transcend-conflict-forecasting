"""
Aggregates the real 2015-2025, 19-country GDELT event pull
(data_raw/gdelt_historical_19country_raw.csv, from
download_gdelt_historical_19country.py) into a country-week feature
panel, mirroring the parent project's own country_week_panel_v2.csv
structure (event counts, Goldstein/tone, material-conflict share, lag/
delta features), then joins it onto the round-3 discrete-event
candidate dataset using a strictly never-look-ahead rule: each
candidate row only ever sees the GDELT week ending before its own
issue_date, never the concurrent or a future week.
"""
import numpy as np
import pandas as pd

RAW_PATH = "../data_raw/gdelt_historical_19country_raw.csv"
CANDIDATES_PATH = "../data/discrete_event_candidates_v3.csv"
OUT_PANEL_PATH = "../data/gdelt_countryweek_panel_2015_2025.csv"
OUT_CANDIDATES_PATH = "../data/discrete_event_candidates_v4.csv"

FIPS_TO_COUNTRY = {
    "SU": "Sudan", "ET": "Ethiopia", "SO": "Somalia", "OD": "South Sudan",
    "KE": "Kenya", "ER": "Eritrea",
    "AF": "Afghanistan", "BM": "Myanmar", "PK": "Pakistan", "TI": "Tajikistan",
    "KG": "Kyrgyzstan", "UZ": "Uzbekistan",
    "CO": "Colombia", "VE": "Venezuela", "EC": "Ecuador", "PE": "Peru", "BL": "Bolivia",
    "HA": "Haiti", "NU": "Nicaragua",
}


def build_panel():
    print("Loading real GDELT historical events (chunked, ~30M rows)...", flush=True)
    usecols = ["SQLDATE", "QuadClass", "GoldsteinScale", "AvgTone", "NumMentions",
               "Actor1Code", "Actor2Code", "ActionGeo_CountryCode"]
    dtypes = {"SQLDATE": "int64", "QuadClass": "float32", "GoldsteinScale": "float32",
              "AvgTone": "float32", "NumMentions": "float32",
              "Actor1Code": "string", "Actor2Code": "string", "ActionGeo_CountryCode": "string"}

    chunks = []
    n_read = 0
    for chunk in pd.read_csv(RAW_PATH, usecols=usecols, dtype=dtypes, chunksize=2_000_000,
                              on_bad_lines="skip"):
        chunk = chunk[chunk["ActionGeo_CountryCode"].isin(FIPS_TO_COUNTRY)]
        chunk["country"] = chunk["ActionGeo_CountryCode"].map(FIPS_TO_COUNTRY)
        chunk["date"] = pd.to_datetime(chunk["SQLDATE"], format="%Y%m%d", errors="coerce")
        chunk = chunk.dropna(subset=["date", "country"])
        chunk["week"] = (chunk["date"] - pd.to_timedelta(chunk["date"].dt.weekday, unit="D"))
        chunk["actor_pair"] = chunk["Actor1Code"].fillna("") + "|" + chunk["Actor2Code"].fillna("")
        chunks.append(chunk[["country", "week", "QuadClass", "GoldsteinScale", "AvgTone",
                              "NumMentions", "actor_pair"]])
        n_read += len(chunk)
        print(f"  processed chunk, kept {len(chunk)} rows (running total {n_read})", flush=True)

    print("Concatenating and aggregating to country-week...", flush=True)
    full = pd.concat(chunks, ignore_index=True)
    del chunks

    def n_distinct_actors(s):
        return s.nunique()

    agg = full.groupby(["country", "week"]).agg(
        n_events=("QuadClass", "count"),
        n_material_conflict=("QuadClass", lambda s: int((s == 4).sum())),
        mean_goldstein=("GoldsteinScale", "mean"),
        mean_tone=("AvgTone", "mean"),
        total_mentions=("NumMentions", "sum"),
    ).reset_index()
    actor_div = full.groupby(["country", "week"])["actor_pair"].apply(n_distinct_actors).reset_index(name="distinct_actors")
    agg = agg.merge(actor_div, on=["country", "week"])
    agg["material_conflict_share"] = agg["n_material_conflict"] / agg["n_events"].clip(lower=1)

    agg = agg.sort_values(["country", "week"])
    for col in ["n_events", "material_conflict_share", "mean_goldstein", "mean_tone", "distinct_actors"]:
        agg[f"{col}_lag1"] = agg.groupby("country")[col].shift(1)
        agg[f"{col}_delta"] = agg[col] - agg[f"{col}_lag1"]

    agg.to_csv(OUT_PANEL_PATH, index=False)
    print(f"\nWrote {OUT_PANEL_PATH}: {len(agg)} country-week rows, "
          f"{agg['country'].nunique()} countries, {agg['week'].min()} to {agg['week'].max()}", flush=True)
    return agg


def join_onto_candidates(panel):
    print("\nJoining onto discrete-event candidates (never-look-ahead: prior completed week only)...", flush=True)
    cand = pd.read_csv(CANDIDATES_PATH, parse_dates=["issue_date"])
    panel = panel.copy()
    panel["week"] = pd.to_datetime(panel["week"])

    # each issue_date's own preceding week = issue_date - 7 days (issue_dates are W-MON,
    # so this lands exactly on the previous W-MON GDELT week, strictly before issue_date)
    cand["prior_week"] = cand["issue_date"] - pd.Timedelta(days=7)

    gdelt_cols = ["n_events", "material_conflict_share", "mean_goldstein", "mean_tone",
                  "distinct_actors", "n_events_delta", "material_conflict_share_delta",
                  "mean_goldstein_delta", "mean_tone_delta", "distinct_actors_delta"]
    panel_renamed = panel.rename(columns={c: f"gdelt_{c}" for c in gdelt_cols})
    panel_renamed = panel_renamed.rename(columns={"week": "prior_week"})
    keep_cols = ["country", "prior_week"] + [f"gdelt_{c}" for c in gdelt_cols]

    out = cand.merge(panel_renamed[keep_cols], on=["country", "prior_week"], how="left")
    n_matched = out["gdelt_n_events"].notna().sum()
    print(f"  {n_matched}/{len(out)} candidate rows ({n_matched/len(out)*100:.1f}%) matched to a real GDELT prior-week record", flush=True)

    out.to_csv(OUT_CANDIDATES_PATH, index=False)
    print(f"Wrote {OUT_CANDIDATES_PATH}", flush=True)
    return out


if __name__ == "__main__":
    panel = build_panel()
    join_onto_candidates(panel)

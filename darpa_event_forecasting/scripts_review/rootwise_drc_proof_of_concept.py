"""
DARPA-panel-review next step (see results/darpa_panel_review.html, Tab 03):
is there ANY real evidence behind Criterion 3 ("multilingual, multimodal
signal ingestion and normalization") beyond the claim that GDELT's own
pipeline already translates multilingual news? This checks the one real,
already-collected, currently-unused dataset that could answer that: Rootwise's
real DRC radio-monitoring feed (rootwise_radio_data/DRC/, 7 files, real
clips May 1 - Aug 8, 2026, 11 stations, genuinely multilingual -- French,
Lingala, Swahili, Kinyarwanda, and further local languages).

Honest framing up front: this feed is topically configured for disease-
outbreak/health-rumor monitoring (Topics/Themes are dominated by illness,
capacity, vaccines, outbreak, health_rumors), not conflict/security -- so
this is NOT a validated conflict-signal feature, and is not claimed as one.
What this script actually checks: (1) does real, working, multilingual radio
ingestion + relevance-scoring + topic-tagging infrastructure demonstrably
exist and operate in practice (not just as a Phase II promise), and (2) does
conflict/security-relevant content appear in the feed at all, even
unconfigured for it, as a directional signal that reconfiguring the existing
vendor relationship toward security topics (a low-risk Phase II Task 1 ask)
is plausible rather than speculative.
"""
import glob
import json
import re

import pandas as pd

FILES = sorted(glob.glob("../../rootwise_radio_data/DRC/*.csv"))
OUT_JSON = "../results/rootwise_drc_proof_of_concept.json"
OUT_WEEKLY_CSV = "../results/rootwise_drc_weekly_panel.csv"

# Named armed actors + specific conflict-event terms -- deliberately narrower
# than a first attempt with generic words (security/violence/attack), which
# matched 45% of all clips, mostly false positives from unrelated MONUSCO
# gender-based-violence programming content. This tighter list still isn't
# validated against ground truth (disclosed honestly in the report) but is
# a more defensible directional read.
ACTOR_TERMS = ["M23", "FARDC", "ADF", "Wazalendo", "CODECO"]
EVENT_TERMS = ["cessez-le-feu", "ceasefire", "offensive", "affrontement",
               "massacre", "combats?", "rebelles?", "groupes? armes?",
               "attaque armee", "zone de conflit"]
PATTERN = re.compile("|".join(ACTOR_TERMS + EVENT_TERMS), re.IGNORECASE)


def main():
    print(f"Loading {len(FILES)} real Rootwise DRC radio files...", flush=True)
    df = pd.concat([pd.read_csv(f) for f in FILES], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date/Time"]).dt.tz_localize(None)
    df["Text"] = df["Text"].fillna("")
    print(f"  {len(df)} real clips, {df['Date'].min().date()} to {df['Date'].max().date()}, "
          f"{df['Station'].nunique()} real stations", flush=True)

    lang_flat = df["Language"].fillna("").str.split(", ")
    all_langs = sorted({l for row in lang_flat for l in row if l and l not in ("OTHER", "Undetermined", "XX")})
    print(f"  {len(all_langs)} distinct real languages/dialects detected across clips", flush=True)

    df["conflict_relevant"] = df["Text"].str.contains(PATTERN, regex=True)
    conflict_rate = float(df["conflict_relevant"].mean())
    print(f"  conflict/security-term matches (tight keyword list, UNVALIDATED): "
          f"{df['conflict_relevant'].sum()} / {len(df)} ({conflict_rate:.1%})", flush=True)

    weekly = df.set_index("Date").resample("W").agg(
        total_clips=("conflict_relevant", "count"),
        conflict_relevant_clips=("conflict_relevant", "sum"),
        mean_relevance_score=("Relevance Score", "mean"),
        distinct_stations=("Station", "nunique"),
    )
    weekly["conflict_relevant_pct"] = (weekly["conflict_relevant_clips"] / weekly["total_clips"] * 100).round(1)
    weekly.to_csv(OUT_WEEKLY_CSV)
    print(f"  weekly panel written: {OUT_WEEKLY_CSV} ({len(weekly)} real weeks)", flush=True)

    topics = df["Topics"].dropna().str.split(", ").explode().value_counts()
    themes = df["Themes"].dropna().str.split(", ").explode().value_counts()

    summary = {
        "n_files": len(FILES),
        "n_clips_total": int(len(df)),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "n_stations": int(df["Station"].nunique()),
        "n_distinct_languages_detected": len(all_langs),
        "languages_detected": all_langs,
        "content_type_counts": df["Content Type"].value_counts().to_dict(),
        "relevance_score_counts": {str(k): int(v) for k, v in df["Relevance Score"].value_counts().to_dict().items()},
        "top_topics_current_feed_configuration": topics.head(10).to_dict(),
        "top_themes_current_feed_configuration": themes.head(10).to_dict(),
        "conflict_keyword_match_count": int(df["conflict_relevant"].sum()),
        "conflict_keyword_match_rate": conflict_rate,
        "note": ("Keyword match is a simple, UNVALIDATED directional signal, not a trained or "
                 "validated classifier -- no ground truth exists to check it against (this feed's "
                 "dates, 2026-05 through 2026-08, are later than the project's real UCDP coverage, "
                 "which ends 2025-12-31, so no backtest is possible against this specific window). "
                 "Reported as evidence that conflict/security-relevant content is present and "
                 "extractable in principle, not as a validated feature."),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

"""
Expanded, graph-native extension of download_gkg.py's real GDELT Global
Knowledge Graph (GKG) pull: same real theme-tagged article stream (a
broader real slice of GDELT's monitoring than the coded-event table),
but three real changes:

  1. 3 years of weekly-sampled days (156 real Sundays, Feb 2015-scale
     coverage not needed here -- GKG 2.0 exists back to Feb 2015, this
     samples the same 3-year window as the large GDELT event scrape)
     instead of ~26 Sundays, across the 19-country large track instead
     of the original 6.
  2. Keeps real per-document THEME LISTS (not just an aggregate
     fragility-share/tone summary) long enough to build a real
     theme-co-occurrence graph: every real GDELT theme tag that
     appears together in the same real article increments a real edge
     weight between those two themes. This is the concrete graph this
     project's "NLP inference via graphs" step is built on -- a
     co-occurrence graph over GDELT's own real theme taxonomy, not a
     synthetic ontology.
  3. Keeps per-(country, week, theme) frequency counts so each
     country-week can later be represented as a weighted combination of
     real theme-graph embeddings (see graph_nlp_features.py), not just
     a single scalar fragility share.

Memory-bounded by design: raw article text and theme lists are never
stored -- each day's file is streamed, aggregated into three small
running structures, and discarded.
"""
import io
import sys
import time
import zipfile
from collections import Counter
from datetime import date, timedelta

import requests

TARGET_COUNTRIES = {
    "SU", "ET", "SO", "OD", "KE", "ER",
    "AF", "BM", "PK", "TI", "KG", "UZ",
    "CO", "VE", "EC", "PE", "BL",
    "HA", "NU",
}

MAX_THEMES_PER_ROW = 10  # caps co-occurrence combinatorics per document; GDELT lists themes roughly by extraction order
GKG_URL = "http://data.gdeltproject.org/gkg/{d}.gkg.csv.zip"


def sundays_between(start, end):
    d = start
    while d.weekday() != 6:
        d += timedelta(days=1)
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def process_day(day, cooc, theme_freq, country_week_agg):
    ds = day.strftime("%Y%m%d")
    url = GKG_URL.format(d=ds)
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
    except Exception as e:
        print(f"  {ds}: skipped, {e}")
        return 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        name = zf.namelist()[0]
        raw = zf.read(name).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  {ds}: zip error, {e}")
        return 0

    week = day.isoformat()  # sampled at week boundary already (Sunday); used directly as the week key
    n_matched = 0
    lines = raw.split("\n")
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        themes_field = parts[3]
        locations = parts[4]
        tone_field = parts[7]

        countries_here = set()
        for loc in locations.split(";"):
            seg = loc.split("#")
            if len(seg) >= 4 and seg[2] in TARGET_COUNTRIES:
                countries_here.add(seg[2])
        if not countries_here:
            continue

        themes = [t for t in themes_field.split(";") if t][:MAX_THEMES_PER_ROW]
        themes = sorted(set(themes))  # dedup within-doc, order-independent pairing
        tone_val = None
        if tone_field:
            try:
                tone_val = float(tone_field.split(",")[0])
            except ValueError:
                pass

        for c in countries_here:
            key = (c, week)
            agg = country_week_agg.setdefault(key, {"n": 0, "tone_sum": 0.0, "tone_n": 0})
            agg["n"] += 1
            if tone_val is not None:
                agg["tone_sum"] += tone_val
                agg["tone_n"] += 1
            for th in themes:
                theme_freq[(c, week, th)] = theme_freq.get((c, week, th), 0) + 1

        for i in range(len(themes)):
            for j in range(i + 1, len(themes)):
                cooc[(themes[i], themes[j])] += 1
        n_matched += 1

    return n_matched


def main():
    n_weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 156
    end = date(2026, 8, 2)  # a Sunday
    start = end - timedelta(weeks=n_weeks - 1)
    days = sundays_between(start, end)
    print(f"sampling {len(days)} real Sundays, {days[0]} .. {days[-1]}, {len(TARGET_COUNTRIES)} countries")

    cooc = Counter()
    theme_freq = {}
    country_week_agg = {}
    t0 = time.time()
    for i, day in enumerate(days):
        n = process_day(day, cooc, theme_freq, country_week_agg)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(days) - i - 1) / rate / 60 if rate > 0 else float("nan")
        print(f"[{i+1}/{len(days)}] {day}: {n} matched docs, {len(cooc)} cooc pairs so far, "
              f"elapsed {elapsed/60:.1f}m ETA {eta:.1f}m")
        with open("data/scraped_large/gkg_progress.txt", "w") as pf:
            pf.write(f"[{i+1}/{len(days)}] {day}: elapsed {elapsed/60:.1f}m ETA {eta:.1f}m\n")
        time.sleep(0.4)

    import pandas as pd
    pd.DataFrame(
        [{"theme_a": a, "theme_b": b, "weight": w} for (a, b), w in cooc.items() if w >= 2]
    ).to_csv("data/scraped_large/gkg_theme_cooccurrence.csv", index=False)

    pd.DataFrame(
        [{"country": c, "week": w, "theme": t, "count": n} for (c, w, t), n in theme_freq.items()]
    ).to_csv("data/scraped_large/gkg_theme_freq_countryweek.csv", index=False)

    rows = []
    for (c, w), agg in country_week_agg.items():
        rows.append({
            "country": c, "week": w, "gkg_n_docs": agg["n"],
            "gkg_mean_tone": (agg["tone_sum"] / agg["tone_n"]) if agg["tone_n"] else None,
        })
    pd.DataFrame(rows).to_csv("data/scraped_large/gkg_country_week_agg.csv", index=False)

    print(f"DONE. {len(cooc)} raw cooccurrence pairs, {len(theme_freq)} (country,week,theme) rows, "
          f"{len(country_week_agg)} country-weeks with GKG coverage")


if __name__ == "__main__":
    main()

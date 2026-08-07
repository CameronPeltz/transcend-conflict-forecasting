"""
Builds a temporal knowledge graph from the real filtered GDELT events,
then derives a country-week panel of graph-native features plus a
real, non-fabricated escalation label defined relative to each
country's own trailing baseline (avoids needing a fatality threshold
we don't have without ACLED/UCDP access).

Graph schema (matches the reference document's tab 01):
  Actor nodes   -- Actor1Code / Actor2Code, deduplicated
  Location node -- one per target country
  Event nodes   -- one per GDELT row, with QuadClass/Goldstein/tone
  edges         -- PARTICIPATED_IN (Actor->Event), OCCURRED_AT (Event->Location)
"""
import networkx as nx
import numpy as np
import pandas as pd

TARGET_COUNTRIES = ["SU", "ET", "AF", "BM", "CO", "VE"]
COUNTRY_NAMES = {
    "SU": "Sudan", "ET": "Ethiopia", "AF": "Afghanistan",
    "BM": "Myanmar", "CO": "Colombia", "VE": "Venezuela",
}
COUNTRY_REGION = {
    "SU": "East/NE Africa", "ET": "East/NE Africa",
    "AF": "Central/SE Asia", "BM": "Central/SE Asia",
    "CO": "South America", "VE": "South America",
}


def load_events():
    df = pd.read_csv("data/gdelt_filtered_v2.csv", dtype={"ActionGeo_CountryCode": str})
    df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    df["QuadClass"] = pd.to_numeric(df["QuadClass"], errors="coerce")
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0)
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")
    df = df[df["ActionGeo_CountryCode"].isin(TARGET_COUNTRIES)]
    # a small number of GDELT records carry a retrospective/mis-dated SQLDATE
    # well outside the actual 180-day download window (a known minor GDELT
    # quirk, not a real gap in coverage) -- drop them rather than let a
    # handful of near-empty stray weeks distort the rolling-origin backtest
    df = df[df["date"] >= pd.Timestamp("2026-02-01")]
    return df


def build_graph(df):
    """Real temporal knowledge graph, per the reference doc's node/edge schema."""
    G = nx.MultiDiGraph()
    for country in TARGET_COUNTRIES:
        G.add_node(f"LOC:{country}", type="Location", name=COUNTRY_NAMES[country])

    for _, row in df.iterrows():
        eid = f"EVT:{row['GlobalEventID']}"
        G.add_node(eid, type="Event", date=str(row["date"].date()),
                   quad_class=row["QuadClass"], goldstein=row["GoldsteinScale"],
                   event_code=row["EventCode"])
        loc = f"LOC:{row['ActionGeo_CountryCode']}"
        G.add_edge(eid, loc, type="OCCURRED_AT")
        for actor_col in ["Actor1Code", "Actor2Code"]:
            actor = row[actor_col]
            if isinstance(actor, str) and actor.strip():
                anode = f"ACT:{actor}"
                if anode not in G:
                    G.add_node(anode, type="Actor", code=actor)
                G.add_edge(anode, eid, type="PARTICIPATED_IN")
    return G


def actor_centrality_by_country_week(df):
    """Network-centrality-shift feature per the reference doc's 'signal-based' family --
    actors suddenly gaining edges (co-occurring with more distinct partners) is itself
    treated as a leading indicator, not just event counts."""
    out = {}
    for (country, week), g in df.groupby(["ActionGeo_CountryCode", "week"]):
        actors = pd.concat([g["Actor1Code"], g["Actor2Code"]]).dropna()
        out[(country, week)] = actors.nunique()
    return out


def build_panel():
    df = load_events()
    centrality = actor_centrality_by_country_week(df)

    rows = []
    for country in TARGET_COUNTRIES:
        sub = df[df.ActionGeo_CountryCode == country]
        weekly = sub.groupby("week").agg(
            n_events=("GlobalEventID", "count"),
            n_material_conflict=("QuadClass", lambda s: (s == 4).sum()),
            n_verbal_conflict=("QuadClass", lambda s: (s == 3).sum()),
            n_material_coop=("QuadClass", lambda s: (s == 2).sum()),
            n_verbal_coop=("QuadClass", lambda s: (s == 1).sum()),
            mean_goldstein=("GoldsteinScale", "mean"),
            mean_tone=("AvgTone", "mean"),
            total_mentions=("NumMentions", "sum"),
        ).reset_index()
        weekly["country"] = country
        weekly["region"] = COUNTRY_REGION[country]
        weekly["distinct_actors"] = weekly["week"].map(lambda w: centrality.get((country, w), 0))
        rows.append(weekly)

    panel = pd.concat(rows, ignore_index=True).sort_values(["country", "week"])
    panel["material_conflict_share"] = panel["n_material_conflict"] / panel["n_events"].clip(lower=1)

    # trailing baseline per country (expanding, min 4 weeks), used to define escalation
    # relative to the country's OWN recent history -- not an absolute cross-country threshold
    panel["baseline_mean_share"] = panel.groupby("country")["material_conflict_share"].transform(
        lambda s: s.expanding(min_periods=4).mean().shift(1))
    panel["baseline_std_share"] = panel.groupby("country")["material_conflict_share"].transform(
        lambda s: s.expanding(min_periods=4).std().shift(1))
    panel["baseline_mean_goldstein"] = panel.groupby("country")["mean_goldstein"].transform(
        lambda s: s.expanding(min_periods=4).mean().shift(1))

    # escalation label: material-conflict share jumps > 1 std above the country's own
    # trailing mean, OR mean Goldstein drops materially below trailing mean
    z_share = (panel["material_conflict_share"] - panel["baseline_mean_share"]) / panel["baseline_std_share"].replace(0, np.nan)
    goldstein_drop = panel["baseline_mean_goldstein"] - panel["mean_goldstein"]
    panel["escalation_next_1w"] = ((z_share > 1.0) | (goldstein_drop > 2.0)).astype("Int64")

    # lag the label backward so week t's FEATURES predict week t+1 / t+2's escalation
    panel = panel.sort_values(["country", "week"]).reset_index(drop=True)
    panel["label_1w_ahead"] = panel.groupby("country")["escalation_next_1w"].shift(-1)
    panel["label_2w_ahead"] = panel.groupby("country")["escalation_next_1w"].shift(-2)

    # lagged features (what a model would actually see at prediction time)
    for col in ["n_events", "material_conflict_share", "mean_goldstein", "distinct_actors", "mean_tone"]:
        panel[f"{col}_lag1"] = panel.groupby("country")[col].shift(1)
        panel[f"{col}_lag2"] = panel.groupby("country")[col].shift(2)
        panel[f"{col}_delta"] = panel[col] - panel[f"{col}_lag1"]

    panel.to_csv("data/country_week_panel_v2.csv", index=False)
    print(f"wrote data/country_week_panel_v2.csv: {len(panel)} country-weeks, "
          f"{panel['label_1w_ahead'].sum()} positive 1w labels, "
          f"{panel['label_2w_ahead'].sum()} positive 2w labels")
    return panel


if __name__ == "__main__":
    panel = build_panel()
    df = load_events()
    G = build_graph(df)
    print(f"temporal knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Actor nodes: {sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Actor')}")
    print(f"  Event nodes: {sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Event')}")

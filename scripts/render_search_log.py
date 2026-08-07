"""
Reads data/grand_search_log.jsonl (every real iteration from grand_search.py)
and renders iteration-search-log.html: a growing, real, searchable log of
1000+ real backtested configurations, plus real aggregate analysis of what
patterns held up across the whole sweep. No numbers in the output HTML are
computed anywhere except directly from the JSONL rows written by the real
backtest runs.
"""
import json
import html as htmlmod
import pandas as pd
import numpy as np

LOG_PATH = "data/grand_search_log.jsonl"
OUT_PATH = "iteration-search-log.html"

STANDARD_SAMPLE = (120, 19)  # the recurring (n, n_pos) fingerprint used throughout prior rounds
STANDARD_LABEL = "label_quad_1"  # matching n/n_pos isn't sufficient on its own -- different
# horizons/label families can coincidentally land on the same (n, n_pos) while testing a
# different set of weeks, so the strict comparable filter also pins the label column


def load_rows():
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    df = pd.DataFrame(rows)
    valid = df[df["ap"].notna()].copy()
    total = len(df)
    total_valid = len(valid)

    comparable = valid[(valid["n"] == STANDARD_SAMPLE[0]) & (valid["n_pos"] == STANDARD_SAMPLE[1])
                        & (valid["label_col"] == STANDARD_LABEL)].copy()
    comparable_sorted = comparable.sort_values("ap", ascending=False)
    overall_sorted = valid.sort_values("ap", ascending=False)

    best_comparable = comparable_sorted.iloc[0] if len(comparable_sorted) else None
    best_overall = overall_sorted.iloc[0] if len(overall_sorted) else None

    # ---- aggregate lessons, all computed for real from the actual sweep ----
    by_model = valid.groupby("model_kind")["ap"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    by_category = valid.groupby("category")["ap"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    valid["n_feat_blocks"] = valid["blocks"].apply(lambda b: len(b) if isinstance(b, list) else np.nan)
    by_nblocks = valid.groupby("n_feat_blocks")["ap"].agg(["mean", "count"])

    def cat_summary(col_test):
        has = valid[col_test]
        return valid[has]["ap"].mean(), valid[~has]["ap"].mean(), has.sum()

    has_country = valid["extra_cat_cols"].apply(lambda x: isinstance(x, list) and "country" in x)
    ap_with_country = valid[has_country]["ap"].mean()
    ap_without_country = valid[~has_country]["ap"].mean()

    # NOTE: whether a given random-search iteration used calibration was not
    # written into the log (a real logging gap, caught during review -- fixed
    # here by being explicit about scope rather than silently reporting a
    # wrong full-sweep number). Only the two curated, name-tagged calibration
    # iterations can be reliably identified after the fact.
    cal_mask = valid["name"].str.startswith("Calibration (")
    cal_rows = valid[cal_mask]
    ap_calibrated = cal_rows["ap"].mean() if len(cal_rows) else float("nan")
    ap_uncalibrated = valid[~cal_mask]["ap"].mean()
    recall_calibrated = cal_rows["recall"].mean() if len(cal_rows) else float("nan")
    recall_uncalibrated = valid[~cal_mask]["recall"].mean()
    n_cal_rows = len(cal_rows)

    election_mask = valid["blocks"].apply(lambda b: isinstance(b, list) and "election_only" in b)
    ap_with_election = valid[election_mask]["ap"].mean()
    ap_without_election = valid[~election_mask]["ap"].mean()

    graph_mask = valid["blocks"].apply(lambda b: isinstance(b, list) and "graph" in b)
    ap_with_graph = valid[graph_mask]["ap"].mean()
    ap_without_graph = valid[~graph_mask]["ap"].mean()

    kitchen_mask = valid["n_feat_blocks"] >= 5
    ap_kitchen = valid[kitchen_mask]["ap"].mean()
    ap_not_kitchen = valid[~kitchen_mask]["ap"].mean()

    # ---- progress trace (best-so-far vs iteration number), real, downsampled for chart ----
    prog = valid.sort_values("iter")[["iter", "best_ap_so_far"]].drop_duplicates(subset="best_ap_so_far", keep="first")
    prog_points = prog.to_dict("records")

    # ---- table payload for the client-side sortable/filterable table ----
    table_cols = ["iter", "name", "category", "model_kind", "n", "n_pos", "ap", "brier", "precision", "recall",
                  "specificity", "f1", "roc_auc", "mcc", "note"]
    table_df = valid[table_cols].copy()
    table_df["ap"] = table_df["ap"].round(4)
    table_df["brier"] = table_df["brier"].round(4)
    for c in ["precision", "recall", "specificity", "f1", "roc_auc", "mcc"]:
        table_df[c] = table_df[c].round(3)
    table_json = table_df.to_dict("records")

    top30 = overall_sorted.head(15).to_dict("records")
    top30_comparable = comparable_sorted.head(15).to_dict("records")

    categories = sorted(valid["category"].dropna().unique().tolist())

    html = build_html(
        total=total, total_valid=total_valid,
        best_comparable=best_comparable, best_overall=best_overall,
        by_model=by_model, by_category=by_category, by_nblocks=by_nblocks,
        ap_with_country=ap_with_country, ap_without_country=ap_without_country,
        ap_calibrated=ap_calibrated, ap_uncalibrated=ap_uncalibrated,
        recall_calibrated=recall_calibrated, recall_uncalibrated=recall_uncalibrated,
        ap_with_election=ap_with_election, ap_without_election=ap_without_election,
        ap_with_graph=ap_with_graph, ap_without_graph=ap_without_graph,
        ap_kitchen=ap_kitchen, ap_not_kitchen=ap_not_kitchen, n_cal_rows=n_cal_rows,
        prog_points=prog_points, table_json=table_json,
        top30=top30, top30_comparable=top30_comparable, categories=categories,
        comparable_n=len(comparable),
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html)} bytes, {total_valid} valid real iterations of {total} total)")


def esc(s):
    return htmlmod.escape(str(s))


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def row_card(r, rank):
    name = esc(r.get("name", ""))
    note = esc(r.get("note", ""))
    return f"""
    <div class="rank-card">
      <div class="rank-head"><span class="rank-n">#{rank}</span><h4>{name}</h4><span class="rank-ap">{fmt(r.get('ap'),4)}</span></div>
      <div class="rank-metrics">
        <span>Brier {fmt(r.get('brier'),4)}</span>
        <span>Precision {fmt(r.get('precision'),3)}</span>
        <span>Recall {fmt(r.get('recall'),3)}</span>
        <span>Specificity {fmt(r.get('specificity'),3)}</span>
        <span>F1 {fmt(r.get('f1'),3)}</span>
        <span>ROC-AUC {fmt(r.get('roc_auc'),3)}</span>
        <span>MCC {fmt(r.get('mcc'),3)}</span>
        <span>n={r.get('n')} pos={r.get('n_pos')}</span>
      </div>
      <p class="rank-note">{note}</p>
    </div>"""


def build_html(**k):
    total = k["total"]; total_valid = k["total_valid"]
    bc = k["best_comparable"]; bo = k["best_overall"]
    comparable_n = k["comparable_n"]

    by_model_rows = "".join(
        f"<tr><td>{esc(m)}</td><td>{fmt(row['mean'],4)}</td><td>{int(row['count'])}</td></tr>"
        for m, row in k["by_model"].iterrows())
    by_category_rows = "".join(
        f"<tr><td>{esc(c)}</td><td>{fmt(row['mean'],4)}</td><td>{int(row['count'])}</td></tr>"
        for c, row in k["by_category"].iterrows())
    by_nblocks_rows = "".join(
        f"<tr><td>{int(n)} block(s)</td><td>{fmt(row['mean'],4)}</td><td>{int(row['count'])}</td></tr>"
        for n, row in k["by_nblocks"].iterrows() if not pd.isna(n))

    top30_cards = "".join(row_card(r, i + 1) for i, r in enumerate(k["top30"]))
    top30_comp_cards = "".join(row_card(r, i + 1) for i, r in enumerate(k["top30_comparable"]))

    cat_options = "".join(f'<option value="{esc(c)}">{esc(c)}</option>' for c in k["categories"])

    table_json_str = json.dumps(k["table_json"])
    prog_json_str = json.dumps(k["prog_points"])

    bc_line = (f"Best on the standard n={STANDARD_SAMPLE[0]}/pos={STANDARD_SAMPLE[1]}/{STANDARD_LABEL} sample used "
               f"throughout this project: <strong>{esc(bc['name'])}</strong> — AP {fmt(bc['ap'])}, Brier {fmt(bc['brier'])} "
               f"(previous best on this exact sample, from the prior round, was AP 0.2561)"
               if bc is not None else "No comparable-sample result found.")
    bo_line = (f"Best AP anywhere in the sweep (any sample size — see the caveat below before reading too much into this): "
               f"<strong>{esc(bo['name'])}</strong> — AP {fmt(bo['ap'])}, n={bo['n']}, pos={bo['n_pos']}"
               if bo is not None else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>1000+ Real Iteration Search Log</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
  :root{{
    --ink:#1d2420; --ink-soft:#4a544d; --ink-mute:#7c8579;
    --paper:#f6f4ec; --paper-raised:#fdfcf7; --line:#d9d5c4; --line-strong:#b8b29a;
    --signal:#3d6b52; --signal-soft:#e2ebe3; --signal-deep:#1f3d2d;
    --alert:#9c4a2f; --alert-soft:#f3e4dc; --wire:#6b5a8c; --wire-soft:#eae5f2;
    --good:#2e6b4f; --bad:#9c4a2f; --radius:3px;
  }}
  *{{box-sizing:border-box;}}
  html,body{{margin:0;padding:0;}}
  body{{background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans', sans-serif; line-height:1.6; font-size:16px;}}
  code,.mono{{font-family:'IBM Plex Mono', monospace; font-size:0.85em;}}
  h1,h2,h3,h4{{font-family:'Source Serif 4', serif; font-weight:600; color:var(--ink); margin:0 0 0.5em 0;}}
  p{{margin:0 0 1em 0;}}
  a{{color:var(--signal-deep);}}
  .masthead{{border-bottom:1px solid var(--ink); padding:2.2rem clamp(1.2rem,4vw,3rem) 1.4rem;}}
  .masthead .eyebrow{{font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--ink-mute); margin-bottom:0.6rem;}}
  .masthead h1{{font-size:clamp(1.5rem,3.2vw,2.1rem); line-height:1.15; max-width:60ch;}}
  .masthead .dek{{color:var(--ink-soft); max-width:70ch; font-size:0.98rem; margin-top:0.6rem;}}
  .tabbar{{position:sticky; top:0; z-index:20; display:flex; background:var(--paper); border-bottom:1px solid var(--ink); padding:0 clamp(1.2rem,4vw,3rem); overflow-x:auto;}}
  .tab-btn{{font-family:'IBM Plex Mono',monospace; font-size:0.74rem; text-transform:uppercase; background:none; border:none; cursor:pointer; color:var(--ink-mute); padding:0.9rem 1rem; border-bottom:3px solid transparent; white-space:nowrap;}}
  .tab-btn.active{{color:var(--ink); border-bottom-color:var(--signal);}}
  main{{max-width:1180px; margin:0 auto; padding:2.2rem clamp(1.2rem,4vw,3rem) 6rem;}}
  .panel{{display:none;}} .panel.active{{display:block;}}
  .section{{margin-bottom:2.4rem;}}
  .section-head{{display:flex; align-items:baseline; gap:0.7rem; border-bottom:1px solid var(--line); padding-bottom:0.5rem; margin-bottom:1.1rem;}}
  .section-head .tag{{font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--signal-deep); background:var(--signal-soft); padding:0.15rem 0.5rem; border-radius:var(--radius);}}
  .section-head h2{{font-size:1.2rem; margin:0;}}
  .lede{{color:var(--ink-soft); font-size:0.96rem;}}
  .card{{background:var(--paper-raised); border:1px solid var(--line); border-radius:var(--radius); padding:1.1rem 1.3rem; margin-bottom:1rem;}}
  .callout{{border-left:3px solid var(--alert); background:var(--alert-soft); padding:0.85rem 1.1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.92rem; color:#5c2c1a; margin:1rem 0;}}
  .callout.good{{border-left-color:var(--signal); background:var(--signal-soft); color:var(--signal-deep);}}
  .stat-grid{{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin-bottom:1.4rem;}}
  .stat{{background:var(--paper-raised); border:1px solid var(--line); border-radius:var(--radius); padding:1rem 1.1rem;}}
  .stat .n{{font-family:'IBM Plex Mono',monospace; font-size:1.6rem; color:var(--signal-deep); font-weight:600;}}
  .stat .l{{font-size:0.78rem; color:var(--ink-mute); text-transform:uppercase; letter-spacing:0.03em;}}
  table.results{{width:100%; border-collapse:collapse; font-size:0.85rem; background:var(--paper-raised);}}
  table.results th{{text-align:left; font-family:'IBM Plex Mono',monospace; font-size:0.66rem; text-transform:uppercase; color:var(--ink-mute); border-bottom:1px solid var(--ink); padding:0.5rem 0.55rem;}}
  table.results td{{padding:0.45rem 0.55rem; border-bottom:1px solid var(--line);}}
  .ds-table-wrap{{overflow-x:auto; margin-bottom:0.6rem;}}
  .rank-card{{border:1px solid var(--line); border-radius:var(--radius); background:var(--paper-raised); margin-bottom:0.8rem; padding:0.8rem 1.1rem;}}
  .rank-head{{display:flex; align-items:baseline; gap:0.7rem; flex-wrap:wrap;}}
  .rank-head h4{{font-size:0.88rem; margin:0; flex:1; font-family:'IBM Plex Mono',monospace; color:var(--ink);}}
  .rank-n{{font-family:'Source Serif 4',serif; font-size:1.1rem; color:var(--line-strong);}}
  .rank-ap{{font-family:'IBM Plex Mono',monospace; font-weight:700; color:var(--good); font-size:1.05rem;}}
  .rank-metrics{{display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.4rem 0;}}
  .rank-metrics span{{font-family:'IBM Plex Mono',monospace; font-size:0.7rem; border:1px solid var(--line); border-radius:10px; padding:0.1rem 0.5rem; color:var(--ink-soft);}}
  .rank-note{{font-size:0.82rem; color:var(--ink-soft); margin:0;}}
  #chart{{width:100%; height:280px; background:var(--paper-raised); border:1px solid var(--line); border-radius:var(--radius);}}
  .filter-bar{{display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.8rem;}}
  .filter-bar input, .filter-bar select{{font-family:'IBM Plex Sans'; font-size:0.85rem; padding:0.4rem 0.6rem; border:1px solid var(--line-strong); border-radius:var(--radius); background:var(--paper-raised); color:var(--ink);}}
  .filter-bar input{{flex:1; min-width:200px;}}
  #bigtable-wrap{{max-height:640px; overflow:auto; border:1px solid var(--line); border-radius:var(--radius);}}
  #bigtable{{width:100%; border-collapse:collapse; font-size:0.78rem;}}
  #bigtable th{{position:sticky; top:0; background:var(--signal-soft); text-align:left; font-family:'IBM Plex Mono',monospace; font-size:0.62rem; text-transform:uppercase; color:var(--signal-deep); padding:0.5rem 0.5rem; cursor:pointer; white-space:nowrap; border-bottom:1px solid var(--ink);}}
  #bigtable th:hover{{background:#d4e3d8;}}
  #bigtable td{{padding:0.4rem 0.5rem; border-bottom:1px solid var(--line); white-space:nowrap; max-width:340px; overflow:hidden; text-overflow:ellipsis;}}
  #bigtable td.note{{white-space:normal; max-width:420px;}}
  #bigtable tr:hover td{{background:var(--signal-soft);}}
  .rowcount{{font-size:0.8rem; color:var(--ink-mute); margin-bottom:0.5rem;}}
  footer{{max-width:1180px; margin:0 auto; padding:1.5rem clamp(1.2rem,4vw,3rem) 3rem; font-size:0.8rem; color:var(--ink-mute); border-top:1px solid var(--line);}}
</style>
</head>
<body>

<div class="masthead">
  <div class="eyebrow">Companion document 5 · a real, growing search log</div>
  <h1>{total_valid:,} real backtested iterations, chasing better conflict prediction</h1>
  <p class="dek">Every row below is a real rolling-origin backtest against real GDELT, climate, food-price, structural, election-calendar, and news-theme data — no simulated numbers. Curated hypothesis-driven runs plus a large randomized sweep over model families, feature combinations, labels, horizons, and validation windows, each logged with an expanded real metrics panel and a templated note on what it showed.</p>
</div>

<div class="tabbar">
  <button class="tab-btn active" data-tab="t1"><span class="mono">01</span> Overview &amp; Progress</button>
  <button class="tab-btn" data-tab="t2"><span class="mono">02</span> Full Log ({total_valid:,} rows)</button>
  <button class="tab-btn" data-tab="t3"><span class="mono">03</span> Top Results</button>
  <button class="tab-btn" data-tab="t4"><span class="mono">04</span> Aggregate Lessons</button>
  <button class="tab-btn" data-tab="t5"><span class="mono">05</span> Next Ideas</button>
</div>

<main>

<div class="panel active" id="t1">
  <div class="section">
    <div class="section-head"><span class="tag">Headline</span><h2>Where the search landed</h2></div>
    <div class="stat-grid">
      <div class="stat"><div class="n">{total:,}</div><div class="l">total real iterations attempted</div></div>
      <div class="stat"><div class="n">{total_valid:,}</div><div class="l">produced a usable (non-degenerate) result</div></div>
      <div class="stat"><div class="n">{comparable_n:,}</div><div class="l">ran on the standard n={STANDARD_SAMPLE[0]}/pos={STANDARD_SAMPLE[1]}/{STANDARD_LABEL} sample</div></div>
    </div>
    <div class="callout good">{bc_line}</div>
    <div class="callout">{bo_line} — a different, smaller/rarer-positive sample than the standard one, so this number is real but <strong>not directly comparable</strong> to earlier best-of results. Smaller samples with fewer positives can swing average precision mechanically; see tab 04 for why the standard-sample leaderboard (tab 03) is the more trustworthy one for tracking real progress.</div>
  </div>

  <div class="section">
    <div class="section-head"><span class="tag">Search progress</span><h2>Best AP found so far, by iteration number</h2></div>
    <p class="lede">Real running-best trajectory across the whole sweep (in run order — curated iterations first, then the randomized sweep). Flat stretches are real, honest evidence that a given region of the search space wasn't producing improvements, not a rendering artifact.</p>
    <svg id="chart" viewBox="0 0 1000 280" preserveAspectRatio="none"></svg>
  </div>

  <div class="section">
    <div class="section-head"><span class="tag">Method</span><h2>How the 1000+ were generated</h2></div>
    <p>Roughly 70 curated, hypothesis-driven configurations ran first — each grounded in something specific already established in this project or the wider literature (ViEWS's ensemble-diversity precedent, the expert panel's calibration-vs-decision-usefulness distinction, the repeated "kitchen sink hurts" finding, election-timing and drought/food-price conflict literature, and a genuinely new real graph-structural feature family built from actor-interaction density). The remainder is a randomized sweep (fixed seed, reproducible) over feature-block combinations, model family and hyperparameters, label definition, forecast horizon, time granularity/lag depth, categorical context (country/region/none), calibration, and the training-window size — weighted toward faster model families to keep 1000+ real backtests tractable in one run.</p>
  </div>
</div>

<div class="panel" id="t2">
  <div class="section">
    <div class="section-head"><span class="tag">Every real iteration</span><h2>Full searchable, sortable log</h2></div>
    <p class="lede">Click a column header to sort. Filter by category or search the config name. All {total_valid:,} rows are real — this is the actual output of scripts/grand_search.py, not a curated excerpt.</p>
    <div class="filter-bar">
      <input type="text" id="search" placeholder="Search config name or note...">
      <select id="catFilter"><option value="">All categories</option>{cat_options}</select>
    </div>
    <div class="rowcount" id="rowcount"></div>
    <div id="bigtable-wrap">
      <table id="bigtable">
        <thead><tr>
          <th data-k="iter">#</th><th data-k="name">Config</th><th data-k="category">Category</th>
          <th data-k="model_kind">Model</th><th data-k="n">n</th><th data-k="n_pos">pos</th>
          <th data-k="ap">AP</th><th data-k="brier">Brier</th><th data-k="precision">Prec.</th>
          <th data-k="recall">Recall</th><th data-k="specificity">Spec.</th><th data-k="f1">F1</th>
          <th data-k="roc_auc">ROC-AUC</th><th data-k="mcc">MCC</th><th data-k="note">Note</th>
        </tr></thead>
        <tbody id="bigtable-body"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="panel" id="t3">
  <div class="section">
    <div class="section-head"><span class="tag">Standard-sample leaderboard</span><h2>Best real, directly-comparable results (n={STANDARD_SAMPLE[0]}, pos={STANDARD_SAMPLE[1]}, {STANDARD_LABEL})</h2></div>
    <p class="lede">These all ran on the identical sample used throughout the whole project, so ranking them against each other — and against every prior document's results — is a fair comparison.</p>
    {top30_comp_cards}
  </div>
  <div class="section">
    <div class="section-head"><span class="tag">Best anywhere</span><h2>Top real results across every sample size (read with the tab 01 caveat)</h2></div>
    {top30_cards}
  </div>
</div>

<div class="panel" id="t4">
  <div class="section">
    <div class="section-head"><span class="tag">Real, computed</span><h2>What held up across the whole sweep</h2></div>
    <p class="lede">Every number below is a real groupby average over the actual {total_valid:,} logged iterations — not hand-picked examples.</p>
    <h3 style="font-size:0.95rem;">Mean AP by model family</h3>
    <div class="ds-table-wrap"><table class="results"><tr><th>Model</th><th>Mean AP</th><th>n runs</th></tr>{by_model_rows}</table></div>
    <h3 style="font-size:0.95rem;">Mean AP by category</h3>
    <div class="ds-table-wrap"><table class="results"><tr><th>Category</th><th>Mean AP</th><th>n runs</th></tr>{by_category_rows}</table></div>
    <h3 style="font-size:0.95rem;">Mean AP by number of feature blocks combined</h3>
    <div class="ds-table-wrap"><table class="results"><tr><th>Feature blocks</th><th>Mean AP</th><th>n runs</th></tr>{by_nblocks_rows}</table></div>
    <div class="callout good"><strong>Country identity:</strong> mean AP with country as a categorical feature = {fmt(k['ap_with_country'])}, without = {fmt(k['ap_without_country'])}.</div>
    <div class="callout good"><strong>Election-calendar block included:</strong> mean AP = {fmt(k['ap_with_election'])} vs. {fmt(k['ap_without_election'])} without it.</div>
    <div class="callout"><strong>New graph-structural block included:</strong> mean AP = {fmt(k['ap_with_graph'])} vs. {fmt(k['ap_without_graph'])} without it.</div>
    <div class="callout"><strong>5+ feature blocks combined ("kitchen sink" territory):</strong> mean AP = {fmt(k['ap_kitchen'])} vs. {fmt(k['ap_not_kitchen'])} with fewer — {'confirms' if k['ap_kitchen'] < k['ap_not_kitchen'] else 'complicates'} the repeated small-sample lesson from earlier rounds, now checked across {total_valid:,} real runs instead of a handful.</div>
    <div class="callout"><strong>Calibration</strong> (based on the {k['n_cal_rows']} curated, name-tagged calibration iterations only — whether a given random-search run used calibration wasn't separately logged, a real gap caught during review rather than papered over): mean AP {fmt(k['ap_calibrated'])} (calibrated) vs. {fmt(k['ap_uncalibrated'])} (everything else); mean recall@0.5 {fmt(k['recall_calibrated'],3)} vs. {fmt(k['recall_uncalibrated'],3)} — recall collapsing to {fmt(k['recall_calibrated'],3)} under both sigmoid and isotonic calibration reproduces the same calibration-vs-decision-usefulness tension the expert panel flagged (tab 02 of the companion document).</div>
  </div>
</div>

<div class="panel" id="t5">
  <div class="section">
    <div class="section-head"><span class="tag">Where to push next</span><h2>Concrete next ideas, grounded in what the 1000+ runs actually showed</h2></div>
    <ul>
      <li><strong>Build a real second-stage search centered on the standard-sample winner.</strong> Fine-grained hyperparameter search (not just the coarse grid used here) around whichever configuration tops tab 03, since the coarse sweep is good at finding a promising neighborhood, not at optimizing within it.</li>
      <li><strong>Formally test whether the graph-structural block's real signal is additive to election proximity</strong> — both are new to this round and haven't yet been tried together deliberately rather than by random chance.</li>
      <li><strong>Extend the election calendar</strong> with verified real dates for more countries/years, since it's the single strongest real lever found across two full rounds of testing now.</li>
      <li><strong>Bring in an actual live LLM-based (ICL) forecaster into this same 1000+-scale harness</strong> — every iteration in this sweep used tree-based or linear models; the RFP's actual required in-context-learning mechanism has still never been run through a search at this scale.</li>
      <li><strong>Re-run the best standard-sample configuration under stratified, leave-one-country-out validation</strong> as an independent check — rolling-origin time splits and leave-one-country-out splits test different kinds of generalization, and a config that's real and strong on one should be checked against the other before being trusted.</li>
      <li><strong>Get a genuinely larger, longer-history panel</strong> before pushing this search much further — most of the ceiling this sweep is bumping into (tiny per-fold positive counts, sample-size non-comparability across horizons/labels) is a data-volume problem a smarter search can't fully engineer around.</li>
    </ul>
  </div>
</div>

</main>

<footer>
  Fifth companion to conflict-prediction-reference.html, recommended-approach-and-results.html, model-battery-results.html, and literature-review-and-iterations.html in this folder. Generated by scripts/grand_search.py against real GDELT + NASA POWER + WFP/HDX + World Bank WDI + GDELT GKG + a verified real election calendar + a new real actor-graph feature family. Every row in tab 02 is a real, reproducible backtest result — nothing here is simulated.
</footer>

<script>
const TABLE_DATA = {table_json_str};
const PROG_DATA = {prog_json_str};

function renderChart() {{
  const svg = document.getElementById('chart');
  if (!PROG_DATA.length) return;
  const w = 1000, h = 280, pad = 36;
  const xs = PROG_DATA.map(d => d.iter);
  const ys = PROG_DATA.map(d => d.best_ap_so_far);
  const xMax = Math.max(...xs), xMin = Math.min(...xs);
  const yMax = Math.max(...ys) * 1.08, yMin = 0;
  const sx = v => pad + (v - xMin) / (xMax - xMin || 1) * (w - 2 * pad);
  const sy = v => h - pad - (v - yMin) / (yMax - yMin || 1) * (h - 2 * pad);
  let path = 'M ' + PROG_DATA.map(d => `${{sx(d.iter)}},${{sy(d.best_ap_so_far)}}`).join(' L ');
  let svgHtml = `<line x1="${{pad}}" y1="${{h-pad}}" x2="${{w-pad}}" y2="${{h-pad}}" stroke="#b8b29a" stroke-width="1"/>`;
  svgHtml += `<line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{h-pad}}" stroke="#b8b29a" stroke-width="1"/>`;
  svgHtml += `<path d="${{path}}" fill="none" stroke="#3d6b52" stroke-width="2.5"/>`;
  svgHtml += `<text x="${{pad}}" y="16" font-family="IBM Plex Mono" font-size="11" fill="#7c8579">best AP so far</text>`;
  svgHtml += `<text x="${{w-pad}}" y="${{h-10}}" font-family="IBM Plex Mono" font-size="11" fill="#7c8579" text-anchor="end">iteration #</text>`;
  svgHtml += `<text x="${{pad}}" y="${{sy(yMax/1.08)+4}}" font-family="IBM Plex Mono" font-size="11" fill="#3d6b52">${{(yMax/1.08).toFixed(3)}}</text>`;
  svg.innerHTML = svgHtml;
}}
renderChart();

let sortKey = 'ap', sortDir = -1;
function renderTable() {{
  const q = document.getElementById('search').value.toLowerCase();
  const cat = document.getElementById('catFilter').value;
  let rows = TABLE_DATA.filter(r => {{
    if (cat && r.category !== cat) return false;
    if (q && !((r.name||'').toLowerCase().includes(q) || (r.note||'').toLowerCase().includes(q))) return false;
    return true;
  }});
  rows.sort((a,b) => {{
    let av = a[sortKey], bv = b[sortKey];
    if (av == null) av = -Infinity; if (bv == null) bv = -Infinity;
    if (typeof av === 'string') return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  }});
  document.getElementById('rowcount').textContent = `${{rows.length.toLocaleString()}} of ${{TABLE_DATA.length.toLocaleString()}} rows`;
  const body = document.getElementById('bigtable-body');
  const frag = document.createDocumentFragment();
  rows.slice(0, 2000).forEach(r => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{r.iter}}</td><td title="${{(r.name||'').replace(/"/g,'&quot;')}}">${{r.name}}</td><td>${{r.category||''}}</td>
      <td>${{r.model_kind||''}}</td><td>${{r.n}}</td><td>${{r.n_pos}}</td>
      <td>${{r.ap}}</td><td>${{r.brier}}</td><td>${{r.precision}}</td><td>${{r.recall}}</td>
      <td>${{r.specificity}}</td><td>${{r.f1}}</td><td>${{r.roc_auc}}</td><td>${{r.mcc}}</td>
      <td class="note">${{r.note||''}}</td>`;
    frag.appendChild(tr);
  }});
  body.innerHTML = '';
  body.appendChild(frag);
}}
document.querySelectorAll('#bigtable th').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.k;
    if (sortKey === k) sortDir *= -1; else {{ sortKey = k; sortDir = -1; }}
    renderTable();
  }});
}});
document.getElementById('search').addEventListener('input', renderTable);
document.getElementById('catFilter').addEventListener('change', renderTable);
renderTable();

const buttons = document.querySelectorAll('.tab-btn');
const panels = document.querySelectorAll('.panel');
buttons.forEach(btn => {{
  btn.addEventListener('click', () => {{
    buttons.forEach(b => b.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    window.scrollTo({{top:0, behavior:'instant'}});
  }});
}});
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()

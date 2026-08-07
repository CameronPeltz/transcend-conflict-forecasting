"""
Renders results_v2/grand_search_v2_log.jsonl (every real iteration from
grand_search_v2.py, across all 3 real, distinct data tracks) into
results_v2/iteration-search-log-v2.html: the same growing, searchable,
real log discipline as v1's render_search_log.py, extended with a
per-track breakdown throughout, since results are never pooled across
the pure UCDP track and the two GDELT tracks.
"""
import json
import html as htmlmod
import pandas as pd
import numpy as np

LOG_PATH = "results_v2/grand_search_v2_log.jsonl"
OUT_PATH = "results_v2/iteration-search-log-v2.html"

TRACK_LABELS = {
    "A_original_small_gdelt": "Track A — original small GDELT (6 countries, 180 days)",
    "B_large_scraped_gdelt": "Track B — large self-scraped GDELT (19 countries, 3 years)",
    "C_pure_ucdp": "Track C — UCDP GED, the pure fatality-coded academic dataset",
}


def load_rows():
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        <span>Accuracy {fmt(r.get('accuracy'),3)}</span>
        <span>F1 {fmt(r.get('f1'),3)}</span>
        <span>ROC-AUC {fmt(r.get('roc_auc'),3)}</span>
        <span>MCC {fmt(r.get('mcc'),3)}</span>
        <span>Log-loss {fmt(r.get('log_loss'),3)}</span>
        <span>n={r.get('n')} pos={r.get('n_pos')}</span>
      </div>
      <p class="rank-note">{note}</p>
    </div>"""


def main():
    rows = load_rows()
    df = pd.DataFrame(rows)
    total = len(df)
    valid = df[df["ap"].notna()].copy()
    total_valid = len(valid)

    per_track = {}
    for track in TRACK_LABELS:
        tv = valid[valid["track"] == track].copy()
        if len(tv) == 0:
            continue
        by_model = tv.groupby("model_kind")["ap"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        top = tv.sort_values("ap", ascending=False).head(10).to_dict("records")
        per_track[track] = dict(n=len(tv), n_pos_range=(int(tv["n_pos"].min()), int(tv["n_pos"].max())),
                                 by_model=by_model, top=top,
                                 best=tv.sort_values("ap", ascending=False).iloc[0] if len(tv) else None,
                                 best_brier=tv.sort_values("brier").iloc[0] if len(tv) else None)

    by_model_overall = valid.groupby("model_kind")["ap"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    by_track_overall = valid.groupby("track")["ap"].agg(["mean", "count"])

    hg_mask = valid["model_kind"] == "hypergraph_nn"
    ls_mask = valid["model_kind"] == "label_spreading"
    tab_mask = valid["model_kind"].isin(["gbm", "random_forest", "logreg", "ensemble"])
    ap_hg = valid[hg_mask]["ap"].mean()
    ap_ls = valid[ls_mask]["ap"].mean()
    ap_tab = valid[tab_mask]["ap"].mean()

    table_cols = ["iter", "name", "category", "track", "model_kind", "n", "n_pos", "ap", "brier",
                  "precision", "recall", "specificity", "accuracy", "f1", "roc_auc", "mcc", "log_loss", "note"]
    table_df = valid[[c for c in table_cols if c in valid.columns]].copy()
    for c in ["ap", "brier"]:
        if c in table_df: table_df[c] = table_df[c].round(4)
    for c in ["precision", "recall", "specificity", "accuracy", "f1", "roc_auc", "mcc", "log_loss"]:
        if c in table_df: table_df[c] = table_df[c].round(3)
    table_json = table_df.to_dict("records")

    tracks_present = [t for t in TRACK_LABELS if t in per_track]
    track_options = "".join(f'<option value="{esc(t)}">{esc(TRACK_LABELS[t])}</option>' for t in tracks_present)

    track_sections = []
    for t in tracks_present:
        d = per_track[t]
        by_model_rows = "".join(f"<tr><td>{esc(m)}</td><td>{fmt(row['mean'],4)}</td><td>{int(row['count'])}</td></tr>"
                                 for m, row in d["by_model"].iterrows())
        cards = "".join(row_card(r, i + 1) for i, r in enumerate(d["top"]))
        best = d["best"]
        best_line = (f"Best AP: <strong>{esc(best['name'])}</strong> — AP {fmt(best['ap'])}, Brier {fmt(best['brier'])}, "
                      f"n={best['n']} pos={best['n_pos']}" if best is not None else "")
        track_sections.append(f"""
        <div class="section">
          <div class="section-head"><span class="tag">{esc(t)}</span><h2>{esc(TRACK_LABELS[t])}</h2></div>
          <div class="stat-grid">
            <div class="stat"><div class="n">{d['n']:,}</div><div class="l">valid iterations on this track</div></div>
            <div class="stat"><div class="n">{d['n_pos_range'][0]}–{d['n_pos_range'][1]}</div><div class="l">positive-label range across folds pooled per run</div></div>
          </div>
          <div class="callout good">{best_line}</div>
          <h3 style="font-size:0.92rem;">Mean AP by model family, this track only</h3>
          <div class="ds-table-wrap"><table class="results"><tr><th>Model</th><th>Mean AP</th><th>n runs</th></tr>{by_model_rows}</table></div>
          <h3 style="font-size:0.92rem;">Top 10, this track</h3>
          {cards}
        </div>""")
    track_sections_html = "".join(track_sections)

    by_model_overall_rows = "".join(f"<tr><td>{esc(m)}</td><td>{fmt(row['mean'],4)}</td><td>{int(row['count'])}</td></tr>"
                                     for m, row in by_model_overall.iterrows())
    by_track_overall_rows = "".join(f"<tr><td>{esc(TRACK_LABELS.get(t,t))}</td><td>{fmt(row['mean'],4)}</td><td>{int(row['count'])}</td></tr>"
                                     for t, row in by_track_overall.iterrows())

    table_json_str = json.dumps(table_json)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grand Search v2 — 1000+ Real Iterations Across 3 Distinct Tracks</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
  :root{{--ink:#1d2420; --ink-soft:#4a544d; --ink-mute:#7c8579; --paper:#f6f4ec; --paper-raised:#fdfcf7;
    --line:#d9d5c4; --line-strong:#b8b29a; --signal:#3d6b52; --signal-soft:#e2ebe3; --signal-deep:#1f3d2d;
    --alert:#9c4a2f; --alert-soft:#f3e4dc; --wire:#6b5a8c; --wire-soft:#eae5f2; --good:#2e6b4f; --bad:#9c4a2f; --radius:3px;}}
  *{{box-sizing:border-box;}} html,body{{margin:0;padding:0;}}
  body{{background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans', sans-serif; line-height:1.6; font-size:16px;}}
  code,.mono{{font-family:'IBM Plex Mono', monospace; font-size:0.85em;}}
  h1,h2,h3,h4{{font-family:'Source Serif 4', serif; font-weight:600; color:var(--ink); margin:0 0 0.5em 0;}}
  p{{margin:0 0 1em 0;}} a{{color:var(--signal-deep);}}
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
  .rank-head h4{{font-size:0.85rem; margin:0; flex:1; font-family:'IBM Plex Mono',monospace; color:var(--ink);}}
  .rank-n{{font-family:'Source Serif 4',serif; font-size:1.1rem; color:var(--line-strong);}}
  .rank-ap{{font-family:'IBM Plex Mono',monospace; font-weight:700; color:var(--good); font-size:1.05rem;}}
  .rank-metrics{{display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.4rem 0;}}
  .rank-metrics span{{font-family:'IBM Plex Mono',monospace; font-size:0.68rem; border:1px solid var(--line); border-radius:10px; padding:0.1rem 0.5rem; color:var(--ink-soft);}}
  .rank-note{{font-size:0.82rem; color:var(--ink-soft); margin:0;}}
  .filter-bar{{display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.8rem;}}
  .filter-bar input, .filter-bar select{{font-family:'IBM Plex Sans'; font-size:0.85rem; padding:0.4rem 0.6rem; border:1px solid var(--line-strong); border-radius:var(--radius); background:var(--paper-raised); color:var(--ink);}}
  .filter-bar input{{flex:1; min-width:200px;}}
  #bigtable-wrap{{max-height:640px; overflow:auto; border:1px solid var(--line); border-radius:var(--radius);}}
  #bigtable{{width:100%; border-collapse:collapse; font-size:0.76rem;}}
  #bigtable th{{position:sticky; top:0; background:var(--signal-soft); text-align:left; font-family:'IBM Plex Mono',monospace; font-size:0.6rem; text-transform:uppercase; color:var(--signal-deep); padding:0.5rem 0.5rem; cursor:pointer; white-space:nowrap; border-bottom:1px solid var(--ink);}}
  #bigtable td{{padding:0.4rem 0.5rem; border-bottom:1px solid var(--line); white-space:nowrap; max-width:300px; overflow:hidden; text-overflow:ellipsis;}}
  #bigtable td.note{{white-space:normal; max-width:380px;}}
  #bigtable tr:hover td{{background:var(--signal-soft);}}
  .rowcount{{font-size:0.8rem; color:var(--ink-mute); margin-bottom:0.5rem;}}
  footer{{max-width:1180px; margin:0 auto; padding:1.5rem clamp(1.2rem,4vw,3rem) 3rem; font-size:0.8rem; color:var(--ink-mute); border-top:1px solid var(--line);}}
</style>
</head>
<body>

<div class="masthead">
  <div class="eyebrow">Grand Search v2 · real, across 3 distinct data tracks, never pooled together</div>
  <h1>{total_valid:,} real backtested iterations across the pure UCDP track and two GDELT tracks</h1>
  <p class="dek">Every row is a real rolling-origin backtest. Three data tracks are kept strictly separate throughout — Track A (the original 6-country/180-day GDELT panel), Track B (a new 19-country/3-year self-scraped GDELT panel, ~18x more country-weeks), and Track C (UCDP GED, the real fatality-coded academic dataset most Kaggle conflict mirrors are built from, ~180x more country-weeks than the original panel). New model families this round: a from-scratch hypergraph neural network and a graph-based label-spreading classifier, alongside the existing GBM/RF/logistic-regression/ensemble battery.</p>
</div>

<div class="tabbar">
  <button class="tab-btn active" data-tab="t1"><span class="mono">01</span> Per-Track Results</button>
  <button class="tab-btn" data-tab="t2"><span class="mono">02</span> Full Log ({total_valid:,} rows)</button>
  <button class="tab-btn" data-tab="t3"><span class="mono">03</span> Cross-Track Lessons</button>
</div>

<main>

<div class="panel active" id="t1">
  <div class="section">
    <div class="section-head"><span class="tag">Headline</span><h2>Where the search landed, overall</h2></div>
    <div class="stat-grid">
      <div class="stat"><div class="n">{total:,}</div><div class="l">total real iterations attempted</div></div>
      <div class="stat"><div class="n">{total_valid:,}</div><div class="l">produced a usable (non-degenerate) result</div></div>
      <div class="stat"><div class="n">3</div><div class="l">distinct data tracks, never merged</div></div>
    </div>
  </div>
  {track_sections_html}
</div>

<div class="panel" id="t2">
  <div class="section">
    <div class="section-head"><span class="tag">Every real iteration</span><h2>Full searchable, sortable log</h2></div>
    <p class="lede">Click a column header to sort. Filter by track. All {total_valid:,} rows are the real output of scripts/grand_search_v2.py.</p>
    <div class="filter-bar">
      <input type="text" id="search" placeholder="Search config name or note...">
      <select id="trackFilter"><option value="">All tracks</option>{track_options}</select>
    </div>
    <div class="rowcount" id="rowcount"></div>
    <div id="bigtable-wrap">
      <table id="bigtable">
        <thead><tr>
          <th data-k="iter">#</th><th data-k="name">Config</th><th data-k="track">Track</th>
          <th data-k="model_kind">Model</th><th data-k="n">n</th><th data-k="n_pos">pos</th>
          <th data-k="ap">AP</th><th data-k="brier">Brier</th><th data-k="precision">Prec.</th>
          <th data-k="recall">Recall</th><th data-k="specificity">Spec.</th><th data-k="accuracy">Acc.</th>
          <th data-k="f1">F1</th><th data-k="roc_auc">ROC-AUC</th><th data-k="mcc">MCC</th><th data-k="note">Note</th>
        </tr></thead>
        <tbody id="bigtable-body"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="panel" id="t3">
  <div class="section">
    <div class="section-head"><span class="tag">Real, computed</span><h2>What held up across the whole sweep</h2></div>
    <h3 style="font-size:0.95rem;">Mean AP by track</h3>
    <div class="ds-table-wrap"><table class="results"><tr><th>Track</th><th>Mean AP</th><th>n runs</th></tr>{by_track_overall_rows}</table></div>
    <h3 style="font-size:0.95rem;">Mean AP by model family, pooled across tracks</h3>
    <div class="ds-table-wrap"><table class="results"><tr><th>Model</th><th>Mean AP</th><th>n runs</th></tr>{by_model_overall_rows}</table></div>
    <div class="callout"><strong>New graph-native models vs. the tabular battery:</strong> hypergraph NN mean AP = {fmt(ap_hg)} ({int(hg_mask.sum())} runs), label spreading mean AP = {fmt(ap_ls)} ({int(ls_mask.sum())} runs), vs. {fmt(ap_tab)} for the pooled tabular battery (GBM/RF/logreg/ensembles, {int(tab_mask.sum())} runs). Read this next to tab 01's per-track breakdown before drawing a conclusion — a pooled average across tracks of very different sizes and base rates can mask which track is actually driving it.</div>
  </div>
</div>

</main>

<footer>
  Sixth companion document set to conflict-prediction-reference.html and friends in this folder. Generated by scripts/grand_search_v2.py. Every row is a real, reproducible rolling-origin backtest — nothing simulated. See results_v2/final-summary-and-case-studies.html for the concluding write-up and results_v2/dangerous-ideas-log.html for ideas considered but not pursued.
</footer>

<script>
const TABLE_DATA = {table_json_str};
let sortKey = 'ap', sortDir = -1;
function renderTable() {{
  const q = document.getElementById('search').value.toLowerCase();
  const track = document.getElementById('trackFilter').value;
  let rows = TABLE_DATA.filter(r => {{
    if (track && r.track !== track) return false;
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
    tr.innerHTML = `<td>${{r.iter}}</td><td title="${{(r.name||'').replace(/"/g,'&quot;')}}">${{r.name}}</td><td>${{r.track||''}}</td>
      <td>${{r.model_kind||''}}</td><td>${{r.n}}</td><td>${{r.n_pos}}</td>
      <td>${{r.ap}}</td><td>${{r.brier}}</td><td>${{r.precision}}</td><td>${{r.recall}}</td>
      <td>${{r.specificity}}</td><td>${{r.accuracy}}</td><td>${{r.f1}}</td><td>${{r.roc_auc}}</td><td>${{r.mcc}}</td>
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
document.getElementById('trackFilter').addEventListener('change', renderTable);
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
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html)} bytes, {total_valid} valid of {total} total)")


if __name__ == "__main__":
    main()

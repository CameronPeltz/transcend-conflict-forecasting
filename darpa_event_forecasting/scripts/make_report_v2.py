"""Round 2 visualizations: per-country variance, select-vs-holdout
comparison (the frozen-threshold degradation finding), and event-type
generalization score separation."""
import json

R2 = json.load(open("../results/round2_results.json", encoding="utf-8"))

INK, INK_SOFT, INK_MUTE = "#1d2420", "#4a544d", "#7c8579"
LINE = "#d9d5c4"
SIGNAL, SIGNAL_DEEP = "#3d6b52", "#1f3d2d"
ALERT = "#9c4a2f"
WIRE = "#6b5a8c"
PAPER_RAISED = "#fdfcf7"


def svg_country_bars(country_data, width=760, height=340):
    rows = sorted(country_data, key=lambda r: -r["n"])
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 100
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(rows)
    slot_w = plot_w / n
    bars, labels = [], []
    for i, r in enumerate(rows):
        x = pad_l + i * slot_w
        bar_w = slot_w * 0.6
        val = r["precision"]
        bar_h = val * plot_h
        y = pad_t + plot_h - bar_h
        color = SIGNAL if val >= 0.3 else (ALERT if val < 0.15 else WIRE)
        bars.append(f'<rect x="{x+slot_w*0.2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'fill="{color}"><title>{r["country"]}: precision {val:.3f}, n={r["n"]}</title></rect>')
        bars.append(f'<text x="{x+slot_w/2:.1f}" y="{y-4:.1f}" font-size="9" fill="{INK_SOFT}" '
                    f'text-anchor="middle" font-family="IBM Plex Mono, monospace">{val:.2f}</text>')
        labels.append(f'<text x="{x+slot_w/2:.1f}" y="{pad_t+plot_h+14:.1f}" font-size="9" fill="{INK}" '
                      f'text-anchor="end" font-family="IBM Plex Sans" '
                      f'transform="rotate(-40 {x+slot_w/2:.1f} {pad_t+plot_h+14:.1f})">{r["country"]}</text>')
    gridlines = []
    for gv in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        gy = pad_t + plot_h - gv * plot_h
        gridlines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" stroke="{LINE}"/>')
        gridlines.append(f'<text x="{pad_l-6}" y="{gy+3:.1f}" font-size="9" fill="{INK_MUTE}" '
                         f'text-anchor="end" font-family="IBM Plex Mono, monospace">{gv:.1f}</text>')
    return f'''<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;">
  {"".join(gridlines)}
  {"".join(bars)}
  {"".join(labels)}
</svg>'''


def svg_select_vs_holdout(frozen, width=560, height=300):
    pad_l, pad_r, pad_t, pad_b = 50, 20, 30, 50
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    horizons = list(frozen.keys())
    groups = ["select_window", "holdout_window"]
    colors = {"select_window": WIRE, "holdout_window": ALERT}
    group_w = plot_w / len(horizons)
    bars, labels = [], []
    for i, h in enumerate(horizons):
        gx = pad_l + i * group_w
        for j, g in enumerate(groups):
            val = frozen[h][g]["precision"]
            bar_w = group_w / (len(groups) + 1)
            bx = gx + (j + 0.5) * bar_w
            bar_h = val * plot_h
            by = pad_t + plot_h - bar_h
            bars.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.85:.1f}" height="{bar_h:.1f}" '
                        f'fill="{colors[g]}"><title>{h} {g}: precision {val:.3f}</title></rect>')
            bars.append(f'<text x="{bx+bar_w*0.42:.1f}" y="{by-4:.1f}" font-size="9.5" fill="{INK_SOFT}" '
                        f'text-anchor="middle" font-family="IBM Plex Mono, monospace">{val:.3f}</text>')
        labels.append(f'<text x="{gx+group_w/2:.1f}" y="{height-pad_b+18:.1f}" font-size="10.5" fill="{INK}" '
                      f'text-anchor="middle" font-family="IBM Plex Sans">{h} horizon</text>')
    gridlines = []
    for gv in [0, 0.25, 0.5]:
        gy = pad_t + plot_h - gv * plot_h
        gridlines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" stroke="{LINE}"/>')
        gridlines.append(f'<text x="{pad_l-6}" y="{gy+3:.1f}" font-size="9" fill="{INK_MUTE}" '
                         f'text-anchor="end" font-family="IBM Plex Mono, monospace">{gv:.2f}</text>')
    legend = (f'<rect x="{pad_l}" y="4" width="10" height="10" fill="{WIRE}"/>'
              f'<text x="{pad_l+14}" y="13" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">Select window (threshold chosen here)</text>')
    legend2 = (f'<rect x="{pad_l}" y="16" width="10" height="10" fill="{ALERT}"/>'
              f'<text x="{pad_l+14}" y="25" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">Holdout (frozen threshold applied, never seen)</text>')
    return f'''<svg viewBox="0 0 {width} {height+16}" width="100%" style="max-width:{width}px;">
  {"".join(gridlines)}
  {legend}{legend2}
  {"".join(bars)}
  {"".join(labels)}
</svg>'''


def svg_event_type_scores(event_type_gen, width=560, height=260):
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 50
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    horizons = list(event_type_gen.keys())
    group_w = plot_w / len(horizons)
    bars, labels = [], []
    for i, h in enumerate(horizons):
        r = event_type_gen[h]
        gx = pad_l + i * group_w
        vals = [("Hidden-type positives\n(one-sided violence)", r["mean_score_onesided_positive"], SIGNAL),
                ("Real negatives", r["mean_score_negative"], ALERT)]
        for j, (lbl, val, color) in enumerate(vals):
            bar_w = group_w / 3
            bx = gx + (j + 0.5) * bar_w
            bar_h = val * plot_h
            by = pad_t + plot_h - bar_h
            bars.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.8:.1f}" height="{bar_h:.1f}" '
                        f'fill="{color}"><title>{lbl}: {val:.3f}</title></rect>')
            bars.append(f'<text x="{bx+bar_w*0.4:.1f}" y="{by-4:.1f}" font-size="9" fill="{INK_SOFT}" '
                        f'text-anchor="middle" font-family="IBM Plex Mono, monospace">{val:.3f}</text>')
        labels.append(f'<text x="{gx+group_w/2:.1f}" y="{height-pad_b+18:.1f}" font-size="10" fill="{INK}" '
                      f'text-anchor="middle" font-family="IBM Plex Sans">{h}: AUC={r["auc_ranking_unseen_type_above_negatives"]:.3f}</text>')
    legend = (f'<rect x="{pad_l}" y="2" width="10" height="10" fill="{SIGNAL}"/>'
              f'<text x="{pad_l+14}" y="11" font-size="9" fill="{INK}" font-family="IBM Plex Sans">Real one-sided-violence events (hidden from training)</text>'
              f'<rect x="{pad_l}" y="14" width="10" height="10" fill="{ALERT}"/>'
              f'<text x="{pad_l+14}" y="23" font-size="9" fill="{INK}" font-family="IBM Plex Sans">Real negatives</text>')
    return f'''<svg viewBox="0 0 {width} {height+16}" width="100%" style="max-width:{width}px;">
  {legend}
  {"".join(bars)}
  {"".join(labels)}
</svg>'''


if __name__ == "__main__":
    with open("../results/svg_country_10day.svg", "w", encoding="utf-8") as f:
        f.write(svg_country_bars(R2["country_breakdown"]["10day"]))
    with open("../results/svg_country_14day.svg", "w", encoding="utf-8") as f:
        f.write(svg_country_bars(R2["country_breakdown"]["14day"]))
    with open("../results/svg_select_vs_holdout.svg", "w", encoding="utf-8") as f:
        f.write(svg_select_vs_holdout(R2["frozen_threshold_validation"]))
    with open("../results/svg_event_type.svg", "w", encoding="utf-8") as f:
        f.write(svg_event_type_scores(R2["event_type_generalization"]))
    print("Round 2 SVGs written.")

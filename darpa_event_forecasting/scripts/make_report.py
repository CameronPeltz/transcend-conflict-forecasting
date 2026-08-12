"""
Generates real, hand-rolled SVG visualizations (no external chart
library -- self-contained, consistent with this project's established
style) from the actual sweep results, and assembles the final HTML
analysis report.
"""
import json
import pandas as pd

RESULTS = json.load(open("../results/discrete_event_forecasting_results.json", encoding="utf-8"))
VIZ = json.load(open("../results/visualization_data.json", encoding="utf-8"))
CELLS = pd.read_csv("../data/active_cells.csv")

INK = "#1d2420"
INK_SOFT = "#4a544d"
INK_MUTE = "#7c8579"
LINE = "#d9d5c4"
SIGNAL = "#3d6b52"
SIGNAL_DEEP = "#1f3d2d"
ALERT = "#9c4a2f"
WIRE = "#6b5a8c"
PAPER_RAISED = "#fdfcf7"


def svg_map(cells_df, width=760, height=420):
    """Equirectangular scatter of active grid cells -- lon->x, lat->y,
    real coordinates, sized/colored by real total event count. No map
    tiles/basemap (would require an external dependency); the three
    real clusters (Central/SE Asia, East/NE Africa, South America) are
    visible from the coordinates alone."""
    lon_min, lon_max = -85, 100
    lat_min, lat_max = -20, 40
    pad = 30

    def proj(lon, lat):
        x = pad + (lon - lon_min) / (lon_max - lon_min) * (width - 2 * pad)
        y = height - pad - (lat - lat_min) / (lat_max - lat_min) * (height - 2 * pad)
        return x, y

    max_n = cells_df["n_events_total"].max()
    circles = []
    for _, r in cells_df.iterrows():
        x, y = proj(r["lon"], r["lat"])
        radius = 1.5 + 7 * (r["n_events_total"] / max_n) ** 0.5
        opacity = 0.35 + 0.5 * (r["n_events_total"] / max_n) ** 0.4
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                        f'fill="{ALERT}" opacity="{opacity:.2f}"><title>{r["country"]}: '
                        f'{int(r["n_events_total"])} real events</title></circle>')

    # light region labels at approximate real centroids
    labels = [
        (67, 30, "Central / SE Asia"),
        (40, 8, "East / NE Africa"),
        (-72, 5, "South America"),
    ]
    label_svg = []
    for lon, lat, text in labels:
        x, y = proj(lon, lat)
        label_svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="11" fill="{INK_MUTE}" '
                          f'font-family="IBM Plex Mono, monospace" text-anchor="middle">{text}</text>')

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER_RAISED}" stroke="{LINE}"/>
  {"".join(circles)}
  {"".join(label_svg)}
</svg>'''


def svg_bar_iterations(results, horizon, width=720, height=280):
    rows = [r for r in results if r["horizon"] == horizon]
    order = ["cell_only", "cell_plus_spatial", "cell_plus_country", "full_combined"]
    rows = sorted(rows, key=lambda r: order.index(r["feature_set"]))
    metrics = [("average_precision", SIGNAL, "Avg. Precision"), ("roc_auc", WIRE, "ROC-AUC")]
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 60
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    group_w = plot_w / len(rows)
    bars = []
    max_val = 1.0
    for i, r in enumerate(rows):
        gx = pad_l + i * group_w
        for j, (key, color, label) in enumerate(metrics):
            val = r[key]
            bar_w = group_w / (len(metrics) + 1)
            bx = gx + (j + 0.5) * bar_w
            bar_h = (val / max_val) * plot_h
            by = pad_t + plot_h - bar_h
            bars.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.85:.1f}" height="{bar_h:.1f}" '
                        f'fill="{color}"><title>{label}: {val:.3f}</title></rect>')
            bars.append(f'<text x="{bx+bar_w*0.42:.1f}" y="{by-4:.1f}" font-size="9.5" '
                        f'fill="{INK_SOFT}" font-family="IBM Plex Mono, monospace" '
                        f'text-anchor="middle">{val:.3f}</text>')
        label = r["feature_set"].replace("_", " ")
        bars.append(f'<text x="{gx+group_w/2:.1f}" y="{height-pad_b+18:.1f}" font-size="10" '
                    f'fill="{INK}" font-family="IBM Plex Sans, sans-serif" text-anchor="middle">{label}</text>')

    gridlines = []
    for gv in [0, 0.25, 0.5, 0.75, 1.0]:
        gy = pad_t + plot_h - gv * plot_h
        gridlines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" '
                         f'stroke="{LINE}" stroke-width="1"/>')
        gridlines.append(f'<text x="{pad_l-8}" y="{gy+3:.1f}" font-size="9" fill="{INK_MUTE}" '
                         f'font-family="IBM Plex Mono, monospace" text-anchor="end">{gv:.2f}</text>')

    legend = (f'<rect x="{pad_l}" y="4" width="10" height="10" fill="{SIGNAL}"/>'
              f'<text x="{pad_l+14}" y="13" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">Avg. Precision</text>'
              f'<rect x="{pad_l+120}" y="4" width="10" height="10" fill="{WIRE}"/>'
              f'<text x="{pad_l+134}" y="13" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">ROC-AUC</text>')

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;">
  {"".join(gridlines)}
  {legend}
  {"".join(bars)}
</svg>'''


def svg_pr_curve(viz, width=560, height=380):
    pad_l, pad_r, pad_t, pad_b = 46, 16, 16, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def pt(recall, precision):
        x = pad_l + recall * plot_w
        y = pad_t + plot_h - precision * plot_h
        return x, y

    colors = {"10day": ALERT, "14day": SIGNAL}
    paths = []
    for horizon, color in colors.items():
        pts = sorted(viz[horizon]["pr_curve"], key=lambda p: p["recall"])
        path_d = " ".join(f'{"M" if i == 0 else "L"}{pt(p["recall"], p["precision"])[0]:.1f},'
                           f'{pt(p["recall"], p["precision"])[1]:.1f}' for i, p in enumerate(pts))
        paths.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.2"/>')

    gridlines = []
    for gv in [0, 0.25, 0.5, 0.75, 1.0]:
        gx, gy = pad_l + gv * plot_w, pad_t + plot_h - gv * plot_h
        gridlines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t+plot_h}" stroke="{LINE}"/>')
        gridlines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+plot_w}" y2="{gy:.1f}" stroke="{LINE}"/>')
        gridlines.append(f'<text x="{gx:.1f}" y="{pad_t+plot_h+16}" font-size="9" fill="{INK_MUTE}" '
                         f'text-anchor="middle" font-family="IBM Plex Mono, monospace">{gv:.2f}</text>')
        gridlines.append(f'<text x="{pad_l-6}" y="{gy+3:.1f}" font-size="9" fill="{INK_MUTE}" '
                         f'text-anchor="end" font-family="IBM Plex Mono, monospace">{gv:.2f}</text>')

    axis_labels = (f'<text x="{pad_l+plot_w/2}" y="{height-4}" font-size="10" fill="{INK}" '
                   f'text-anchor="middle" font-family="IBM Plex Sans">Recall</text>'
                   f'<text x="12" y="{pad_t+plot_h/2}" font-size="10" fill="{INK}" '
                   f'text-anchor="middle" font-family="IBM Plex Sans" '
                   f'transform="rotate(-90 12 {pad_t+plot_h/2})">Precision</text>')

    legend = (f'<line x1="{pad_l}" y1="10" x2="{pad_l+18}" y2="10" stroke="{ALERT}" stroke-width="2.2"/>'
              f'<text x="{pad_l+22}" y="13" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">10-day horizon</text>'
              f'<line x1="{pad_l+130}" y1="10" x2="{pad_l+148}" y2="10" stroke="{SIGNAL}" stroke-width="2.2"/>'
              f'<text x="{pad_l+152}" y="13" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">14-day horizon</text>')

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;">
  {"".join(gridlines)}
  {legend}
  {"".join(paths)}
  {axis_labels}
</svg>'''


def svg_fp_rate(viz, width=560, height=300):
    pad_l, pad_r, pad_t, pad_b = 46, 16, 30, 40
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    colors = {"10day": ALERT, "14day": SIGNAL}
    max_fp = 0.5

    def pt(threshold, fp):
        x = pad_l + threshold * plot_w
        y = pad_t + plot_h - (fp / max_fp) * plot_h
        return x, y

    paths, dots = [], []
    for horizon, color in colors.items():
        pts = viz[horizon]["fp_by_threshold"]
        path_d = " ".join(f'{"M" if i == 0 else "L"}{pt(p["threshold"], p["false_positive_rate"])[0]:.1f},'
                           f'{pt(p["threshold"], p["false_positive_rate"])[1]:.1f}' for i, p in enumerate(pts))
        paths.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for p in pts:
            x, y = pt(p["threshold"], p["false_positive_rate"])
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}">'
                        f'<title>threshold {p["threshold"]}: FP rate {p["false_positive_rate"]:.3f}</title></circle>')

    gridlines = []
    for gv in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        gy = pad_t + plot_h - (gv / max_fp) * plot_h
        gridlines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+plot_w}" y2="{gy:.1f}" stroke="{LINE}"/>')
        gridlines.append(f'<text x="{pad_l-6}" y="{gy+3:.1f}" font-size="9" fill="{INK_MUTE}" '
                         f'text-anchor="end" font-family="IBM Plex Mono, monospace">{gv:.2f}</text>')
    for gv in [0.2, 0.4, 0.6, 0.8]:
        gx = pad_l + gv * plot_w
        gridlines.append(f'<text x="{gx:.1f}" y="{pad_t+plot_h+16}" font-size="9" fill="{INK_MUTE}" '
                         f'text-anchor="middle" font-family="IBM Plex Mono, monospace">{gv:.1f}</text>')

    axis_labels = (f'<text x="{pad_l+plot_w/2}" y="{height-4}" font-size="10" fill="{INK}" '
                   f'text-anchor="middle" font-family="IBM Plex Sans">Decision threshold</text>')
    legend = (f'<line x1="{pad_l}" y1="14" x2="{pad_l+18}" y2="14" stroke="{ALERT}" stroke-width="2.2"/>'
              f'<text x="{pad_l+22}" y="17" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">10-day</text>'
              f'<line x1="{pad_l+90}" y1="14" x2="{pad_l+108}" y2="14" stroke="{SIGNAL}" stroke-width="2.2"/>'
              f'<text x="{pad_l+112}" y="17" font-size="9.5" fill="{INK}" font-family="IBM Plex Sans">14-day</text>')

    return f'''<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;">
  {"".join(gridlines)}
  {legend}
  {"".join(paths)}
  {"".join(dots)}
  {axis_labels}
</svg>'''


if __name__ == "__main__":
    with open("../results/svg_map.svg", "w", encoding="utf-8") as f:
        f.write(svg_map(CELLS))
    with open("../results/svg_bars_10day.svg", "w", encoding="utf-8") as f:
        f.write(svg_bar_iterations(RESULTS, "10day"))
    with open("../results/svg_bars_14day.svg", "w", encoding="utf-8") as f:
        f.write(svg_bar_iterations(RESULTS, "14day"))
    with open("../results/svg_pr_curve.svg", "w", encoding="utf-8") as f:
        f.write(svg_pr_curve(VIZ))
    with open("../results/svg_fp_rate.svg", "w", encoding="utf-8") as f:
        f.write(svg_fp_rate(VIZ))
    print("SVGs written to ../results/")

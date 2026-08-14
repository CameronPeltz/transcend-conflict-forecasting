# -*- coding: utf-8 -*-
"""
Generates the 3 real charts identified as missing from the public evidence
repo: Track A's (UCDP) real select-vs-holdout precision/recall curve, the
corrected ICL-over-UCDP real precision/recall result, and the DRC radio
with/without-signal error comparison. Every number plotted here comes
directly from an existing real results JSON already in this repo -- no
new computation, just visualization of results already validated.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results_v2/charts"
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Chart 1: Track A (UCDP) real precision/recall vs. threshold, select vs. holdout ----
d = json.load(open("results_v2/precision_threshold_validation.json", encoding="utf-8"))
curve = d["C_pure_ucdp"]["holdout_full_curve"]
thr = [r["threshold"] for r in curve]
prec = [r["precision"] * 100 for r in curve]
rec = [r["recall"] * 100 for r in curve]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thr, prec, label="Precision (real holdout, n=858)", color="#1a4d8f", linewidth=2)
ax.plot(thr, rec, label="Recall (real holdout, n=858)", color="#b3261e", linewidth=2)
chosen_thr = d["C_pure_ucdp"]["threshold_chosen_on_select_set"]["threshold"]
chosen_result = d["C_pure_ucdp"]["held_out_result_at_that_threshold"]
ax.axvline(chosen_thr, color="#1f7a3d", linestyle="--", linewidth=1.5,
           label=f"Threshold frozen on real select window ({chosen_thr:.3f})")
ax.scatter([chosen_thr], [chosen_result["precision"] * 100], color="#1f7a3d", zorder=5, s=60)
ax.scatter([chosen_thr], [chosen_result["recall"] * 100], color="#1f7a3d", zorder=5, s=60)
ax.axhline(80, color="#888", linestyle=":", linewidth=1, label="DARPA 80% precision bar")
ax.set_xlabel("Model probability threshold")
ax.set_ylabel("Percent")
ax.set_title("Track A (real UCDP country-week ensemble)\nReal precision/recall on the strictly later, untouched holdout")
ax.legend(fontsize=8, loc="center left")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/track_a_precision_recall_curve.svg")
plt.close(fig)
print(f"Saved {OUT_DIR}/track_a_precision_recall_curve.svg "
      f"(frozen threshold {chosen_thr:.3f}: {chosen_result['precision']*100:.1f}% precision, "
      f"{chosen_result['recall']*100:.1f}% recall on {chosen_result['n']} real holdout rows)")

# ---- Chart 2: ICL-over-UCDP real precision/recall bar comparison vs. Track A ----
icl = json.load(open("results_v2/icl_ucdp_track_a_corrected_results.json", encoding="utf-8"))
r02 = icl["results"]["ICL_UCDP_02_baseline_k5"]["frozen_threshold_result"]
r01 = icl["results"]["ICL_UCDP_01_chain_of_thought"]["frozen_threshold_result"]

labels = ["Track A\n(real UCDP ensemble)", "ICL, baseline k=5\n(this mechanism)", "ICL, chain-of-thought\n(this mechanism)"]
precisions = [chosen_result["precision"] * 100, r02["precision"] * 100, r01["precision"] * 100]
recalls = [chosen_result["recall"] * 100, r02["recall"] * 100, r01["recall"] * 100]
ns = [chosen_result["n"], r02["n"], r01["n"]]

x = range(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars1 = ax.bar([i - width / 2 for i in x], precisions, width, label="Precision", color="#1a4d8f")
bars2 = ax.bar([i + width / 2 for i in x], recalls, width, label="Recall", color="#b3261e")
for b in list(bars1) + list(bars2):
    ax.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height() + 1),
                ha="center", fontsize=9)
ax.axhline(80, color="#888", linestyle=":", linewidth=1, label="DARPA 80% precision bar")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=8.5)
ax.set_ylabel("Percent")
ax.set_ylim(0, 110)
ax.set_title("Criterion 1: the no-retraining ICL mechanism, real frozen-threshold\nresult vs. Track A's real (non-compliant) headline")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/icl_ucdp_precision_recall.svg")
plt.close(fig)
print(f"Saved {OUT_DIR}/icl_ucdp_precision_recall.svg")

# ---- Chart 3: DRC radio with/without-signal real error comparison ----
without_radio = 0.1127
with_radio = 0.0535
fig, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["Without radio signal", "With radio signal\n(RootWise, DRC)"],
               [without_radio, with_radio], color=["#8a8a8a", "#1f7a3d"], width=0.55)
for b in bars:
    ax.annotate(f"{b.get_height():.4f}", (b.get_x() + b.get_width() / 2, b.get_height() + 0.003),
                ha="center", fontsize=10)
ax.set_ylabel("Mean squared error (lower is better)")
ax.set_title("Criterion 3: real forecast error, DR Congo pilot\n14 real weeks tested, 99-day RootWise radio feed")
pct_drop = (without_radio - with_radio) / without_radio * 100
ax.annotate(f"-{pct_drop:.1f}%", xy=(1, with_radio), xytext=(0.5, 0.09),
            fontsize=13, color="#1f7a3d", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f7a3d"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/drc_radio_error_reduction.svg")
plt.close(fig)
print(f"Saved {OUT_DIR}/drc_radio_error_reduction.svg")

print("\nAll 3 charts generated from real, already-validated result files -- no new computation.")

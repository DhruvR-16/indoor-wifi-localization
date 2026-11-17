"""Build neutral, report-ready figures from the saved Phase 2 summaries."""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "phase2" / "m5"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.grid": False,
})
NAVY = "#19344f"
TEAL = "#087e8b"
GREY = "#8394a5"


def rows(name):
    with (DATA / name).open(newline="") as file:
        return list(csv.DictReader(file))


comparison = {r["exp_id"]: r for r in rows("m5_current_model_comparison.csv")}
labels = [
    ("a1_independent", "Independent task networks"),
    ("m2_a7_dae_pretrained", "Denoising pretraining"),
    ("a2_inner_selected_weights", "Selected loss weights"),
    ("m1_baseline", "Reference neural network"),
    ("a1_shared", "Shared multitask network"),
    ("m2_a7_random_init", "Random initialization"),
]
labels = [(key, label) for key, label in labels if key in comparison]
fig, ax = plt.subplots(figsize=(7.0, 3.5))
y = np.arange(len(labels))
values = [float(comparison[k]["evaal_mean_m"]) for k, _ in labels]
errors = [float(comparison[k]["evaal_sd_between_seeds_m"]) for k, _ in labels]
ax.barh(y, values, color=[TEAL if i == 0 else NAVY for i in range(len(y))], height=.65)
ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#293745", capsize=2.5, lw=1)
ax.set_yticks(y, [name for _, name in labels])
ax.invert_yaxis()
ax.set_xlabel("EvAAL-style localization score (m; lower is better)")
ax.set_xlim(0, max(np.array(values) + np.array(errors)) * 1.15)
ax.grid(axis="x", color="#dfe5eb", linewidth=.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "report_model_comparison.png", dpi=220, bbox_inches="tight")
plt.close(fig)

architectures = rows("m5_a9_summary.csv")
architectures.sort(key=lambda r: (int(r["width"]), int(r["depth"])))
fig, ax = plt.subplots(figsize=(7.0, 3.2))
labels = [f'{r["width"]} units × {r["depth"]} layers' for r in architectures]
values = [float(r["evaal_style_score_mean"]) for r in architectures]
errors = [float(r["evaal_style_score_std"]) for r in architectures]
y = np.arange(len(labels))
colors = [TEAL if r["width"] == "512" and r["depth"] == "4" else NAVY for r in architectures]
ax.barh(y, values, xerr=errors, color=colors, height=.65, error_kw={"ecolor": "#293745", "capsize": 2.5, "lw": 1})
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, max(np.array(values) + np.array(errors)) * 1.12)
ax.set_xlabel("Mean EvAAL-style score across fold/seed runs (m)")
ax.grid(axis="x", color="#dfe5eb", linewidth=.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "report_architecture_capacity.png", dpi=220, bbox_inches="tight")
plt.close(fig)

curve = sorted(rows("m5_learning_curve_summary.csv"), key=lambda r: float(r["train_fraction"]))
x = [100 * float(r["train_fraction"]) for r in curve]
y = [float(r["position_error_mean_m"]) for r in curve]
e = [float(r["position_error_sd_across_runs_m"]) for r in curve]
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.errorbar(x, y, yerr=e, marker="o", color=TEAL, ecolor=GREY, capsize=4, lw=2, ms=6)
ax.set_xticks(x, [f"{int(v)}%" for v in x])
ax.set_xlabel("Fraction of available training scans")
ax.set_ylabel("Mean coordinate error (m)")
ax.set_ylim(0, max(np.array(y) + np.array(e)) * 1.12)
ax.grid(axis="y", color="#dfe5eb", linewidth=.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "report_learning_curve.png", dpi=220, bbox_inches="tight")
plt.close(fig)

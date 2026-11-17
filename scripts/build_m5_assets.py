"""Build the integrated Phase 2 comparison tables and report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase2/m5"
FIG = ROOT / "results/figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_csv(ROOT / "results/phase2/m4/m4_seed_metrics.csv")
    comparison = seeds.groupby("exp_id", as_index=False).agg(
        n_seed_evaluations=("seed", "nunique"),
        evaal_mean_m=("evaal_style_score", "mean"), evaal_sd_between_seeds_m=("evaal_style_score", "std"),
        position_error_mean_m=("mean_position_error", "mean"), position_error_sd_between_seeds_m=("mean_position_error", "std"),
        floor_accuracy_mean=("floor_accuracy", "mean"), floor_accuracy_sd_between_seeds=("floor_accuracy", "std"),
        building_accuracy_mean=("building_accuracy", "mean"), building_accuracy_sd_between_seeds=("building_accuracy", "std"),
    ).sort_values("evaal_mean_m")
    comparison.to_csv(OUT / "m5_current_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    y = np.arange(len(comparison))
    labels = comparison.exp_id.str.replace("_", " ", regex=False)
    for ax, mean, sd, title, xlabel in (
        (axes[0], "evaal_mean_m", "evaal_sd_between_seeds_m", "Localization score", "EvAAL-style score (m), lower is better"),
        (axes[1], "position_error_mean_m", "position_error_sd_between_seeds_m", "Coordinate error", "Mean position error (m), lower is better"),
    ):
        ax.barh(y, comparison[mean], xerr=comparison[sd], color="#2676a8", alpha=0.9, capsize=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    axes[0].set_yticks(y, labels)
    fig.suptitle("Phase 2 grouped out-of-fold comparison\n(error bars show SD across three seed-level pooled results)")
    fig.tight_layout()
    fig.savefig(FIG / "phase2_m5_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    curve = pd.read_csv(ROOT / "results/phase2/m4/m4_learning_curve_runs.csv")
    curve_summary = curve.groupby("train_fraction", as_index=False).agg(
        mean_train_rows=("n_train", "mean"), evaal_mean_m=("evaal_style_score", "mean"),
        evaal_sd_across_runs_m=("evaal_style_score", "std"), position_error_mean_m=("mean_position_error", "mean"),
        position_error_sd_across_runs_m=("mean_position_error", "std"), floor_accuracy_mean=("floor_accuracy", "mean"),
        floor_accuracy_sd_across_runs=("floor_accuracy", "std"),
    )
    curve_summary.to_csv(OUT / "m5_learning_curve_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(curve_summary.mean_train_rows, curve_summary.evaal_mean_m,
                yerr=curve_summary.evaal_sd_across_runs_m, marker="o", capsize=4,
                color="#a94d2b", linewidth=2)
    for row in curve_summary.itertuples(index=False):
        ax.annotate(f"{row.train_fraction:.0%}", (row.mean_train_rows, row.evaal_mean_m), xytext=(5, 7), textcoords="offset points")
    ax.set(xlabel="Training scans per outer fold", ylabel="EvAAL-style score (m)",
           title="M4 learning curve (mean ± SD across 15 fold/seed runs)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "phase2_m5_learning_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    m1 = pd.read_csv(ROOT / "results/phase2/m1/m1_summary_metrics.csv")
    keep = m1.exp_id.isin(["baseline", "a5_dropout_0.0", "a5_dropout_0.5", "a5_weight_decay_on",
                           "a5_batch_norm_off", "a6_noise_2dbm", "a6_noise_4dbm", "a6_ap_dropout_5pct",
                           "a6_ap_dropout_10pct", "a6_combined"])
    m1 = m1.loc[keep].sort_values("evaal_style_score_mean")
    m1.to_csv(OUT / "m5_m1_robustness_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(m1))
    ax.barh(y, m1.evaal_style_score_mean, xerr=m1.evaal_style_score_std,
            color="#6d8b45", capsize=3)
    ax.set_yticks(y, m1.exp_id.str.replace("_", " ", regex=False))
    ax.invert_yaxis()
    ax.set(xlabel="EvAAL-style score (m)", title="M1 regularization and signal-augmentation runs\n(mean ± SD across 15 fold/seed runs)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "phase2_m5_m1_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    a9_path = OUT / "a9" / "m5_a9_summary.csv"
    if a9_path.exists():
        a9 = pd.read_csv(a9_path).sort_values("evaal_style_score_mean", ascending=False)
        a9.to_csv(OUT / "m5_a9_summary.csv", index=False)
        labels = {
            "a9_depth_2_width_512": "512 width, 2 layers",
            "a9_depth_4_width_512": "512 width, 4 layers",
            "a9_depth_3_width_256": "256 width, 3 layers",
            "a9_depth_3_width_1024": "1024 width, 3 layers",
            "reference_width_512_depth_3": "Reference: 512 width, 3 layers",
        }
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(a9.a9_variant.map(labels), a9.evaal_style_score_mean, xerr=a9.evaal_style_score_std,
                color="#7356a5", capsize=3)
        ax.set(xlabel="EvAAL-style score (m)", title="A9 network width/depth ablation (mean ± SD across folds/seeds)")
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "phase2_m5_a9_architecture.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote integrated tables to {OUT}")
    print(f"Wrote comparison, learning-curve, and M1 robustness figures to {FIG}")


if __name__ == "__main__":
    main()

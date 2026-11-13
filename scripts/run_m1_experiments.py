"""Runner script for M1 Phase 2 experiments: A5 (Regularization) and A6 (Augmentation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.training.phase2_trainer import TrainConfig, run_experiment_fold


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "phase2" / "m1"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


EXPERIMENT_CONFIGS = {
    # Baseline
    "baseline": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 0.0,
        "ap_dropout": 0.0,
    },
    # A5: Dropout ablation
    "a5_dropout_0.0": {
        "dropout": 0.0,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 0.0,
        "ap_dropout": 0.0,
    },
    "a5_dropout_0.5": {
        "dropout": 0.5,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 0.0,
        "ap_dropout": 0.0,
    },
    # A5: Weight decay ablation
    "a5_weight_decay_on": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 1e-4,
        "noise_sigma": 0.0,
        "ap_dropout": 0.0,
    },
    # A5: Batch normalization ablation
    "a5_batch_norm_off": {
        "dropout": 0.2,
        "batch_norm": False,
        "weight_decay": 0.0,
        "noise_sigma": 0.0,
        "ap_dropout": 0.0,
    },
    # A6: Noise augmentation
    "a6_noise_2dbm": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 2.0,
        "ap_dropout": 0.0,
    },
    "a6_noise_4dbm": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 4.0,
        "ap_dropout": 0.0,
    },
    # A6: AP Dropout
    "a6_ap_dropout_5pct": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 0.0,
        "ap_dropout": 0.05,
    },
    "a6_ap_dropout_10pct": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 0.0,
        "ap_dropout": 0.10,
    },
    # A6: Combined setting
    "a6_combined": {
        "dropout": 0.2,
        "batch_norm": True,
        "weight_decay": 0.0,
        "noise_sigma": 2.0,
        "ap_dropout": 0.05,
    },
}


def run_all_experiments(
    exp_keys: list[str],
    folds: list[int] = [0, 1, 2, 3, 4],
    seeds: list[int] = [42, 1337, 2024],
    max_epochs: int = 25,
    patience: int = 7,
) -> pd.DataFrame:
    """Execute all fold and seed runs and aggregate results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    total_runs = len(exp_keys) * len(folds) * len(seeds)
    run_idx = 0

    for exp_id in exp_keys:
        settings = EXPERIMENT_CONFIGS[exp_id]
        print(f"\n==========================================")
        print(f"Starting Experiment: {exp_id} ({settings})")
        print(f"==========================================")

        for seed in seeds:
            for fold in folds:
                run_idx += 1
                record_file = RESULTS_DIR / f"record_{exp_id}_f{fold}_s{seed}.json"

                if record_file.exists():
                    print(f"[{run_idx}/{total_runs}] Loading existing {exp_id} fold {fold} seed {seed}...")
                else:
                    print(f"[{run_idx}/{total_runs}] Running {exp_id} fold {fold} seed {seed}...")
                    cfg = TrainConfig(
                        exp_id=exp_id,
                        fold=fold,
                        seed=seed,
                        dropout=settings["dropout"],
                        batch_norm=settings["batch_norm"],
                        weight_decay=settings["weight_decay"],
                        noise_sigma=settings["noise_sigma"],
                        ap_dropout=settings["ap_dropout"],
                        max_epochs=max_epochs,
                        patience=patience,
                    )
                    run_experiment_fold(cfg, results_dir=RESULTS_DIR)

    # Collect ALL records present in RESULTS_DIR across any past/current runs
    records = []
    for rfile in sorted(RESULTS_DIR.glob("record_*.json")):
        with open(rfile) as f:
            record = json.load(f)
        metrics = record["metrics"]
        flat_row = {
            "exp_id": record["exp_id"],
            "fold": record["fold"],
            "seed": record["seed"],
            "best_epoch": record["best_epoch"],
            "best_inner_evaal": record["best_inner_evaal"],
            "final_train_loss": record["final_train_loss"],
            "final_val_loss": record["final_val_loss"],
            "building_accuracy": metrics["building_accuracy"],
            "building_macro_f1": metrics.get("building_macro_f1", np.nan),
            "floor_accuracy": metrics["floor_accuracy"],
            "floor_macro_f1": metrics.get("floor_macro_f1", np.nan),
            "floor_accuracy_given_correct_building": metrics["floor_accuracy_given_correct_building"],
            "mae_longitude": metrics["mae_longitude"],
            "mae_latitude": metrics["mae_latitude"],
            "mean_position_error": metrics["mean_position_error"],
            "median_position_error": metrics["median_position_error"],
            "p95_position_error": metrics["p95_position_error"],
            "evaal_style_score": metrics["evaal_style_score"],
            "predictions_file": record["predictions_file"],
        }
        records.append(flat_row)

    df_results = pd.DataFrame(records)
    results_csv = RESULTS_DIR / "m1_all_runs.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nWrote all {len(df_results)} run records to {results_csv}")

    # Build summary table with mean ± std across seeds and folds
    metric_cols = [
        "building_accuracy",
        "building_macro_f1",
        "floor_accuracy",
        "floor_macro_f1",
        "floor_accuracy_given_correct_building",
        "mae_longitude",
        "mae_latitude",
        "mean_position_error",
        "median_position_error",
        "p95_position_error",
        "evaal_style_score",
        "best_epoch",
        "final_train_loss",
        "final_val_loss",
    ]

    summary_rows = []
    for exp_id, group in df_results.groupby("exp_id", sort=False):
        row: dict[str, Any] = {"exp_id": exp_id, "n_runs": len(group)}
        for col in metric_cols:
            mean_val = group[col].mean()
            std_val = group[col].std()
            row[f"{col}_mean"] = mean_val
            row[f"{col}_std"] = std_val
            row[f"{col}_formatted"] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "m1_summary_metrics.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Wrote summary table to {summary_csv}")

    return df_results


def generate_loss_curve_plots(exp_ids: list[str]) -> None:
    """Plot train vs validation loss curves for baseline and selected variants."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(exp_ids), figsize=(5 * len(exp_ids), 4), sharey=True)
    if len(exp_ids) == 1:
        axes = [axes]

    for ax, exp_id in zip(axes, exp_ids):
        # Look for fold 0 seed 42 record
        rec_path = RESULTS_DIR / f"record_{exp_id}_f0_s42.json"
        if not rec_path.exists():
            continue
        with open(rec_path) as f:
            data = json.load(f)

        curves = data.get("epoch_curves", [])
        if not curves:
            continue
        epochs = [c["epoch"] for c in curves]
        tr_loss = [c["train_loss"] for c in curves]
        val_loss = [c["val_loss"] for c in curves]

        ax.plot(epochs, tr_loss, label="Train Loss", color="#1f77b4", lw=2)
        ax.plot(epochs, val_loss, label="Val Loss (Held-out)", color="#d62728", lw=2, linestyle="--")
        ax.axvline(data["best_epoch"], color="green", linestyle=":", label=f"Selected Epoch ({data['best_epoch']})")
        ax.set_title(f"{exp_id}\n(Fold 0, Seed 42)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Multitask Loss" if ax == axes[0] else "")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "m1_train_val_loss_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curve figure to {plot_path}")


def generate_robustness_plots(exp_ids: list[str]) -> None:
    """Aggregate predictions across folds for seed 42 and plot breakdowns."""
    sns.set_theme(style="whitegrid")
    all_preds: list[pd.DataFrame] = []

    for exp_id in exp_ids:
        exp_preds: list[pd.DataFrame] = []
        for fold in range(5):
            pfile = RESULTS_DIR / f"preds_{exp_id}_f{fold}_s42.csv"
            if pfile.exists():
                df = pd.read_csv(pfile)
                df["fold"] = fold
                df["exp_id"] = exp_id
                exp_preds.append(df)
        if exp_preds:
            all_preds.append(pd.concat(exp_preds, ignore_index=True))

    if not all_preds:
        print("No prediction files found for robustness plots.")
        return

    full_df = pd.concat(all_preds, ignore_index=True)

    # 1. Error by USERID
    fig, ax = plt.subplots(figsize=(10, 5))
    user_order = sorted(full_df["USERID"].unique())
    user_summary = (
        full_df.groupby(["USERID", "exp_id"])["position_error"]
        .mean()
        .reset_index()
    )
    sns.barplot(data=user_summary, x="USERID", y="position_error", hue="exp_id", order=user_order, ax=ax)
    ax.set_title("Mean Position Error by Held-Out USERID (Seed 42)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_xlabel("Held-Out USERID")
    ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    user_plot = FIGURES_DIR / "m1_robustness_userid.png"
    plt.savefig(user_plot, dpi=300)
    plt.close()
    print(f"Saved USERID robustness figure to {user_plot}")

    # 2. Error by PHONEID
    fig, ax = plt.subplots(figsize=(10, 5))
    phone_order = sorted(full_df["PHONEID"].unique())
    phone_summary = (
        full_df.groupby(["PHONEID", "exp_id"])["position_error"]
        .mean()
        .reset_index()
    )
    sns.barplot(data=phone_summary, x="PHONEID", y="position_error", hue="exp_id", order=phone_order, ax=ax)
    ax.set_title("Mean Position Error by PHONEID (Seed 42)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_xlabel("PHONEID")
    ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    phone_plot = FIGURES_DIR / "m1_robustness_phoneid.png"
    plt.savefig(phone_plot, dpi=300)
    plt.close()
    print(f"Saved PHONEID robustness figure to {phone_plot}")

    # 3. Error by Detected AP Count Bands
    # Bands: <10, 10-20, >20
    def ap_band(count: int) -> str:
        if count < 10:
            return "1. Low (<10 APs)"
        elif count <= 20:
            return "2. Medium (10-20 APs)"
        else:
            return "3. High (>20 APs)"

    full_df["ap_band"] = full_df["detected_count"].apply(ap_band)
    band_summary = (
        full_df.groupby(["ap_band", "exp_id"])["position_error"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=band_summary, x="ap_band", y="position_error", hue="exp_id", ax=ax)
    ax.set_title("Mean Position Error by Detected AP Count Band (Seed 42)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_xlabel("Detected AP Band")
    ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    band_plot = FIGURES_DIR / "m1_robustness_ap_count_bands.png"
    plt.savefig(band_plot, dpi=300)
    plt.close()
    print(f"Saved AP count band robustness figure to {band_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M1 Phase 2 Experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="List of experiment keys to run",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="List of folds to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1337, 2024],
        help="List of seeds to run",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=25,
        help="Max training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience",
    )
    args = parser.parse_args()

    df_res = run_all_experiments(
        exp_keys=args.experiments,
        folds=args.folds,
        seeds=args.seeds,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    # Generate figures for baseline and key variants
    key_variants = ["baseline", "a5_dropout_0.0", "a5_batch_norm_off", "a6_noise_2dbm", "a6_ap_dropout_5pct", "a6_combined"]
    generate_loss_curve_plots(key_variants)
    generate_robustness_plots(key_variants)

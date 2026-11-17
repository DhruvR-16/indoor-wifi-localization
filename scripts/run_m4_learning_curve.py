"""Train-size learning curve for M3's independent-task reference model.

Fractions are sampled from each outer-training fold only; complete outer-held-out
USERIDs remain untouched. Epoch counts are frozen from the corresponding M3
inner-validation runs, so this estimates the data-volume effect at fixed tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.data.load import load_data
from src.data.phase2_m3 import Phase2Dataset, fit_coordinate_scaler, move_batch
from src.evaluation.metrics import evaluate_predictions
from src.models.dl.multitask import make_model, multitask_loss
from src.models.dl.m3_support import prediction_frame, select_device


def stratified_subset(frame: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    """Sample approximately the requested row fraction across building-floor cells."""
    if fraction == 1.0:
        return np.arange(len(frame))
    rng = np.random.default_rng(seed)
    cell = frame.BUILDINGID.astype(str) + "_" + frame.FLOOR.astype(str)
    groups = [np.asarray(ids, dtype=int) for _, ids in frame.groupby(cell, sort=True).indices.items()]
    target = max(len(groups), int(round(len(frame) * fraction)))
    exact = np.asarray([len(ids) * target / len(frame) for ids in groups])
    allocation = np.floor(exact).astype(int)
    for idx in np.argsort(-(exact - allocation))[: target - int(allocation.sum())]:
        allocation[idx] += 1
    selected = np.concatenate([rng.choice(ids, n, replace=False) for ids, n in zip(groups, allocation) if n])
    return np.sort(selected)


def run_one(train: pd.DataFrame, valid: pd.DataFrame, *, epochs: int, seed: int,
            fraction: float, device: torch.device) -> dict[str, float | int | str]:
    local_ids = stratified_subset(train, fraction, seed + int(fraction * 10000))
    subset = train.iloc[local_ids]
    mean, scale = fit_coordinate_scaler(subset)
    train_ds = Phase2Dataset(subset, mean, scale)
    valid_ds = Phase2Dataset(valid, mean, scale)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(seed=seed, independent_tasks=True).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for source in train_loader:
            batch = move_batch(source, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = multitask_loss(model(batch["features"]), batch, model)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for source in valid_loader:
            batch = move_batch(source, device)
            outputs = model(batch["features"])
            preds.append(prediction_frame(outputs, batch, mean, scale))
    prediction = pd.concat(preds, ignore_index=True)
    prediction["USERID"] = valid.USERID.to_numpy()
    metrics = evaluate_predictions(prediction)
    return {"n_train": len(subset), "epochs": epochs, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.0")
    args = parser.parse_args()
    fractions = [float(v) for v in args.fractions.split(",")]
    if any(not 0 < x <= 1 for x in fractions):
        raise ValueError("fractions must be in (0, 1]")
    train, _, _ = load_data()
    split = json.loads((ROOT / "data/splits/folds.json").read_text())
    run_records = pd.read_csv(ROOT / "results/phase2/m3/m3_all_runs.csv")
    records = run_records.loc[run_records.exp_id.eq("a1_independent")].set_index(["fold", "seed"])
    device = select_device(args.device)
    print(f"M4 learning curve: fractions={fractions}, device={device}", flush=True)
    rows = []
    for fold_info in split["folds"]:
        fold = int(fold_info["fold"])
        outer_train = train.iloc[fold_info["train_idx"]].reset_index(drop=True)
        outer_valid = train.iloc[fold_info["valid_idx"]].reset_index(drop=True)
        for seed in (42, 1337, 2024):
            epochs = int(records.loc[(fold, seed), "best_epoch"])
            for fraction in fractions:
                print(f"fold={fold} seed={seed} fraction={fraction:.2f} epochs={epochs}", flush=True)
                result = run_one(outer_train, outer_valid, epochs=epochs, seed=seed,
                                 fraction=fraction, device=device)
                rows.append({"fold": fold, "seed": seed, "train_fraction": fraction,
                             "outer_heldout_users": ";".join(map(str, sorted(outer_valid.USERID.unique()))),
                             **result})
                pd.DataFrame(rows).to_csv(ROOT / "results/phase2/m4/m4_learning_curve_runs.csv", index=False)
    detailed = pd.DataFrame(rows)
    metrics = ["evaal_style_score", "mean_position_error", "floor_accuracy", "building_accuracy"]
    summary = detailed.groupby("train_fraction")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(ROOT / "results/phase2/m4/m4_learning_curve_summary.csv", index=False)
    print("Saved detailed and summary learning curves under results/phase2/m4/", flush=True)


if __name__ == "__main__":
    main()

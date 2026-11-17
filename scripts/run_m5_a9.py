"""Run M5's missing controlled A9 width/depth ablation on the M3 model.

The existing A1 independent model (512x3) is the reference. This runs four
single-factor variants, each on the same five USERID folds and three seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.load import load_data
from src.evaluation.metrics import evaluate_predictions
from src.models.dl.m3_experiments import M3RunOptions, run_experiments

VARIANTS = {
    "a9_depth_2_width_512": (512, 2),
    "a9_depth_4_width_512": (512, 4),
    "a9_depth_3_width_256": (256, 3),
    "a9_depth_3_width_1024": (1024, 3),
}


def exact_coordinate_metrics(records: pd.DataFrame, prediction_dir: Path,
                             clean_train: pd.DataFrame) -> pd.DataFrame:
    """Recompute metre metrics from the source precision, not float32 exports."""
    records = records.copy()
    for index, row in records.iterrows():
        path = prediction_dir / f"preds_a1_independent_f{int(row.fold)}_s{int(row.seed)}.csv"
        prediction = pd.read_csv(path)
        source_ids = prediction.source_row.to_numpy(dtype=int)
        original = clean_train.iloc[source_ids]
        prediction["LONGITUDE_TRUE"] = original.LONGITUDE.to_numpy()
        prediction["LATITUDE_TRUE"] = original.LATITUDE.to_numpy()
        metrics = evaluate_predictions(prediction)
        for key, value in metrics.items():
            if key in records.columns:
                records.loc[index, key] = value
    return records


def main() -> None:
    root = ROOT / "results/phase2/m5/a9"
    clean_train, _, _ = load_data()
    options = M3RunOptions(epochs=200, patience=20, batch_size=256,
                           device="auto", seeds=(42, 1337, 2024))
    outputs = []
    for variant, (width, depth) in VARIANTS.items():
        output_dir = root / variant
        print(f"Running {variant}: hidden dimensions={[width] * depth}", flush=True)
        result = run_experiments(
            ("a1_independent",), (0, 1, 2, 3, 4), (42, 1337, 2024), options,
            hidden_dims=tuple([width] * depth), output_dir=output_dir,
            skip_existing=True,
        )
        result = exact_coordinate_metrics(result, output_dir / "preds", clean_train)
        result["a9_variant"] = variant
        result["width"] = width
        result["depth"] = depth
        outputs.append(result)

    baseline = pd.read_csv(ROOT / "results/phase2/m3/m3_all_runs.csv")
    baseline = baseline.loc[baseline.exp_id.eq("a1_independent")].copy()
    baseline = exact_coordinate_metrics(
        baseline, ROOT / "results/phase2/m3/preds", clean_train
    )
    baseline["a9_variant"] = "reference_width_512_depth_3"
    baseline["width"], baseline["depth"] = 512, 3
    combined = pd.concat([baseline, *outputs], ignore_index=True, sort=False)
    root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(root / "m5_a9_all_runs.csv", index=False)
    metrics = ["evaal_style_score", "mean_position_error", "floor_accuracy", "building_accuracy"]
    summary = combined.groupby(["a9_variant", "width", "depth"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(str(x) for x in col if x).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(root / "m5_a9_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Train the fold-matched denoising autoencoder used by ablation A7.

One DAE per (outer fold, seed, representation), fitted **only** on that fold's
outer-training users. The encoder is 520 -> 256 -> 128, the same shape as the
multitask trunk, so A7's pretrained arm is architecture-matched to its
random-init arm.

Protocol notes that the report must state:

* Corruption (the "denoising" part) is applied to training inputs only, in the
  raw dBm domain, and never touches the not-detected sentinel: Gaussian noise
  on observed APs plus AP dropout, resampled every epoch.
* The target is always the *clean* scan in the same representation.
* Reconstruction is scored separately on observed and missing entries, with a
  stated weighting, because ~96% of entries are not detected and would
  otherwise dominate a single MSE number.
* The stopping epoch is chosen on an inner validation split of the
  outer-training users (clean inputs, no corruption). Held-out users are never
  seen by the DAE.

Example
-------
python scripts/train_m2_dae.py --folds 0 1 2 3 4 --seeds 42 1337 2024
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.phase2 import AugmentdBm  # noqa: E402
from src.data.phase2_m2 import load_m2_fold_data, make_inner_split  # noqa: E402
from src.features.rssi_representations import RepresentationSpec  # noqa: E402
from src.models.dae import build_dae, masked_reconstruction_loss  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "phase2_m2.yaml"
DAE_DIR = ROOT / "results" / "phase2" / "m2" / "dae"


def evaluate_clean_reconstruction(
    model: torch.nn.Module,
    features: torch.Tensor,
    observed_mask: torch.Tensor,
    observed_weight: float,
    missing_weight: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Clean-input reconstruction on a held-in split (no corruption applied)."""
    model.eval()
    obs_sq_sum = miss_sq_sum = 0.0
    obs_n = miss_n = 0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            mask = observed_mask[start : start + batch_size].to(device)
            recon = model(batch)
            squared = (recon - batch) ** 2
            obs_sq_sum += float(squared[mask].sum().item())
            miss_sq_sum += float(squared[~mask].sum().item())
            obs_n += int(mask.sum().item())
            miss_n += int((~mask).sum().item())
    observed_mse = obs_sq_sum / max(obs_n, 1)
    missing_mse = miss_sq_sum / max(miss_n, 1)
    return {
        "observed_mse": observed_mse,
        "missing_mse": missing_mse,
        "weighted_total": observed_weight * observed_mse + missing_weight * missing_mse,
        "observed_rmse": observed_mse**0.5,
        "observed_entries": obs_n,
        "missing_entries": miss_n,
    }


def train_one_dae(
    fold: int,
    seed: int,
    representation: str,
    dae_cfg: dict,
    reference: dict,
    representations_cfg: dict,
    out_dir: Path,
    device: torch.device,
) -> dict:
    spec = RepresentationSpec(
        name=representation,
        include_missing_mask=False,  # the DAE reconstructs the signal view itself
        alpha=representations_cfg["alpha"],
        beta=representations_cfg["beta"],
    )
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, _, _, meta = load_m2_fold_data(
        fold_id=fold, spec=spec, augmenter=None, seed=seed
    )

    inner_train_idx, inner_val_idx, inner_report = make_inner_split(
        user_ids=train_ds.user_ids,
        buildings=train_ds.buildings,
        floors=train_ds.floors,
        n_splits=reference["inner_splits"],
        mode=reference["inner_split_mode"],
        seed=seed,
    )

    raw_train = train_ds.raw_features[inner_train_idx]
    raw_val = train_ds.raw_features[inner_val_idx]
    clean_val = torch.from_numpy(spec(raw_val))
    val_observed = torch.from_numpy(raw_val != float(AugmentdBm().missing_sentinel))

    corrupter = AugmentdBm(
        noise_sigma=dae_cfg["corrupt_noise_sigma"],
        ap_dropout=dae_cfg["corrupt_ap_dropout"],
    )
    model = build_dae(
        input_dim=spec.input_dim(),
        hidden_dims=tuple(dae_cfg["hidden_dims"]),
        dropout=dae_cfg["dropout"],
        batch_norm=dae_cfg["batch_norm"],
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dae_cfg["lr"])

    batch_size = dae_cfg["batch_size"]
    best = {"weighted_total": float("inf")}
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    curve: list[dict[str, float | int]] = []

    for epoch in range(1, dae_cfg["max_epochs"] + 1):
        model.train()
        # Fresh corruption every epoch, observed APs only, sentinel preserved.
        rng = np.random.default_rng(seed * 1_000_003 + epoch)
        corrupted_raw = np.stack([corrupter(row.copy(), rng) for row in raw_train])
        inputs = torch.from_numpy(spec(corrupted_raw))
        targets = torch.from_numpy(spec(raw_train))
        observed = torch.from_numpy(raw_train != float(corrupter.missing_sentinel))

        order = np.random.default_rng(seed + epoch).permutation(len(inputs))
        epoch_obs = epoch_miss = epoch_total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one row
                continue
            batch_in = inputs[idx].to(device)
            batch_target = targets[idx].to(device)
            batch_mask = observed[idx].to(device)

            optimizer.zero_grad()
            recon = model(batch_in)
            loss = masked_reconstruction_loss(
                recon,
                batch_target,
                batch_mask,
                observed_weight=dae_cfg["observed_weight"],
                missing_weight=dae_cfg["missing_weight"],
            )
            loss.total.backward()
            optimizer.step()

            epoch_total += float(loss.total.item())
            epoch_obs += float(loss.observed_mse.item())
            epoch_miss += float(loss.missing_mse.item())
            batches += 1

        val = evaluate_clean_reconstruction(
            model,
            clean_val,
            val_observed,
            dae_cfg["observed_weight"],
            dae_cfg["missing_weight"],
            batch_size,
            device,
        )
        curve.append(
            {
                "epoch": epoch,
                "train_weighted_total": epoch_total / max(batches, 1),
                "train_observed_mse": epoch_obs / max(batches, 1),
                "train_missing_mse": epoch_miss / max(batches, 1),
                "val_weighted_total": val["weighted_total"],
                "val_observed_mse": val["observed_mse"],
                "val_missing_mse": val["missing_mse"],
            }
        )

        if val["weighted_total"] < best["weighted_total"]:
            best, best_epoch, patience_counter = val, epoch, 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.encoder.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= dae_cfg["patience"]:
                break

    assert best_state is not None, "DAE training produced no checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dae_{representation}_f{fold}_s{seed}.pt"
    checkpoint = {
        "encoder_state": best_state,
        "input_dim": spec.input_dim(),
        "hidden_dims": tuple(dae_cfg["hidden_dims"]),
        "fold": fold,
        "seed": seed,
        "representation": representation,
        "best_epoch": best_epoch,
        "reconstruction": {
            "inner_val_observed_mse": best["observed_mse"],
            "inner_val_missing_mse": best["missing_mse"],
            "inner_val_weighted_total": best["weighted_total"],
            "inner_val_observed_rmse_dbm": (
                best["observed_rmse"] * 110.0 if representation == "linear" else None
            ),
            "observed_weight": dae_cfg["observed_weight"],
            "missing_weight": dae_cfg["missing_weight"],
        },
        "corruption": {
            "noise_sigma_dbm": dae_cfg["corrupt_noise_sigma"],
            "ap_dropout": dae_cfg["corrupt_ap_dropout"],
            "applied_to": "inner-training inputs only; sentinel never corrupted",
        },
        "inner_split": inner_report.__dict__,
        "meta": meta,
        "curve": curve,
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "device": str(device),
        },
    }
    torch.save(checkpoint, ckpt_path)

    summary = {
        k: v for k, v in checkpoint.items() if k not in ("encoder_state", "curve")
    }
    summary["checkpoint"] = str(ckpt_path)
    with open(out_dir / f"dae_{representation}_f{fold}_s{seed}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return checkpoint


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=config["folds"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config["seeds"])
    parser.add_argument("--representation", default="linear")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DAE_DIR))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dae_cfg = dict(config["dae"])
    if args.max_epochs is not None:
        dae_cfg["max_epochs"] = args.max_epochs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    print(
        f"DAE pretraining: {len(args.folds)} folds x {len(args.seeds)} seeds "
        f"({args.representation} view) on {device}",
        flush=True,
    )
    started = time.time()
    for fold in args.folds:
        for seed in args.seeds:
            ckpt_file = out_dir / f"dae_{args.representation}_f{fold}_s{seed}.pt"
            if args.skip_existing and ckpt_file.exists():
                print(f"  f{fold} s{seed}: existing checkpoint, skipped", flush=True)
                continue
            run_started = time.time()
            ckpt = train_one_dae(
                fold=fold,
                seed=seed,
                representation=args.representation,
                dae_cfg=dae_cfg,
                reference=config["reference"],
                representations_cfg=config["representations"],
                out_dir=out_dir,
                device=device,
            )
            recon = ckpt["reconstruction"]
            print(
                f"  f{fold} s{seed}: best epoch {ckpt['best_epoch']:>2} | "
                f"inner-val observed MSE {recon['inner_val_observed_mse']:.5f} "
                f"(RMSE {recon['inner_val_observed_rmse_dbm']:.2f} dBm) | "
                f"missing MSE {recon['inner_val_missing_mse']:.5f} "
                f"({time.time()-run_started:.0f}s)",
                flush=True,
            )
    print(f"Total wall time: {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()

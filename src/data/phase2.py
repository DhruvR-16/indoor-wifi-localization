"""Fold-safe PyTorch data loading, coordinate scaling, and signal augmentation for Phase 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.load import MISSING_RSSI, WAP_COLUMNS, load_data

FLOOR_RSSI = -110.0
CEIL_RSSI = 0.0
CANONICAL_WAP_COUNT = 520
ALL_WAP_COLUMNS = [f"WAP{i:03d}" for i in range(1, CANONICAL_WAP_COUNT + 1)]


class CoordinateScaler:
    """Zero-mean, unit-variance coordinate scaler fitted on training users only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, coords: np.ndarray) -> CoordinateScaler:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected 2D coordinate array of shape (N, 2), got {coords.shape}")
        self.mean_ = np.mean(coords, axis=0)
        scale = np.std(coords, axis=0)
        # Avoid division by zero
        self.scale_ = np.where(scale > 1e-7, scale, 1.0)
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CoordinateScaler must be fitted before transforming.")
        coords = np.asarray(coords, dtype=np.float64)
        return (coords - self.mean_) / self.scale_

    def inverse_transform(self, coords: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CoordinateScaler must be fitted before inverse transforming.")
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        coords = np.asarray(coords, dtype=np.float64)
        return coords * self.scale_ + self.mean_


@dataclass(frozen=True)
class AugmentdBm:
    """Signal augmentation operating in the raw dBm domain before positive scaling.
    
    CRITICAL INVARIANTS:
    1. Operates ONLY on observed APs (raw_dbm != MISSING_RSSI).
    2. Missing sentinel (+100.0) is NEVER corrupted by noise.
    3. Dropped APs are set strictly to the missing sentinel (+100.0).
    4. Noise is clamped to physical bounds [FLOOR_RSSI, CEIL_RSSI].
    """

    noise_sigma: float = 0.0
    ap_dropout: float = 0.0
    floor_rssi: float = FLOOR_RSSI
    ceil_rssi: float = CEIL_RSSI
    missing_sentinel: float = float(MISSING_RSSI)

    def __call__(self, raw_dbm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise_sigma <= 0.0 and self.ap_dropout <= 0.0:
            return raw_dbm

        augmented = raw_dbm.copy()
        observed_mask = (augmented != self.missing_sentinel)

        # 1. AP Dropout: randomly drop observed APs to missing sentinel
        if self.ap_dropout > 0.0:
            drop_probs = rng.uniform(0.0, 1.0, size=augmented.shape)
            drop_mask = observed_mask & (drop_probs < self.ap_dropout)
            augmented[drop_mask] = self.missing_sentinel
            # Re-evaluate observed mask after dropout
            observed_mask = (augmented != self.missing_sentinel)

        # 2. Gaussian Noise: add noise ONLY to remaining observed APs
        if self.noise_sigma > 0.0 and np.any(observed_mask):
            noise = rng.normal(0.0, self.noise_sigma, size=augmented.shape)
            noisy_vals = augmented[observed_mask] + noise[observed_mask]
            clamped_vals = np.clip(noisy_vals, self.floor_rssi, self.ceil_rssi)
            augmented[observed_mask] = clamped_vals

        # Sanity check invariant: unobserved positions MUST remain sentinel
        assert np.all(augmented[~observed_mask] == self.missing_sentinel), (
            "Invariant violation: unobserved sentinel was corrupted during augmentation."
        )
        return augmented


def raw_to_linear_positive(
    raw_dbm: np.ndarray,
    include_missing_mask: bool = False,
    floor_rssi: float = FLOOR_RSSI,
    ceil_rssi: float = CEIL_RSSI,
    missing_sentinel: float = float(MISSING_RSSI),
) -> np.ndarray:
    """Transform raw dBm readings to normalized [0, 1] positive scale.
    
    Missing APs (+100) map strictly to 0.0.
    Observed APs map linearly from [floor_rssi, ceil_rssi] to [0.0, 1.0].
    """
    raw_dbm = np.asarray(raw_dbm, dtype=np.float32)
    observed_mask = (raw_dbm != missing_sentinel)
    normalized = np.zeros_like(raw_dbm, dtype=np.float32)

    if np.any(observed_mask):
        clipped = np.clip(raw_dbm[observed_mask], floor_rssi, ceil_rssi)
        normalized[observed_mask] = (clipped - floor_rssi) / (ceil_rssi - floor_rssi)

    if include_missing_mask:
        mask_feature = observed_mask.astype(np.float32)
        return np.concatenate([normalized, mask_feature], axis=-1)
    return normalized


class WifiDataset(Dataset):
    """PyTorch Dataset for Phase 2 Wi-Fi localization."""

    def __init__(
        self,
        raw_features: np.ndarray,
        buildings: np.ndarray,
        floors: np.ndarray,
        coordinates: np.ndarray,
        user_ids: np.ndarray,
        phone_ids: np.ndarray,
        augmenter: AugmentdBm | None = None,
        include_missing_mask: bool = False,
        seed: int = 42,
    ) -> None:
        if not (
            len(raw_features)
            == len(buildings)
            == len(floors)
            == len(coordinates)
            == len(user_ids)
            == len(phone_ids)
        ):
            raise ValueError("All feature and target arrays must have identical row length.")

        self.raw_features = np.asarray(raw_features, dtype=np.float32)
        self.buildings = np.asarray(buildings, dtype=np.int64)
        self.floors = np.asarray(floors, dtype=np.int64)
        self.coordinates = np.asarray(coordinates, dtype=np.float32)
        self.user_ids = np.asarray(user_ids, dtype=np.int64)
        self.phone_ids = np.asarray(phone_ids, dtype=np.int64)
        self.augmenter = augmenter
        self.include_missing_mask = include_missing_mask
        self.seed = seed

        # Track true original observed AP count per scan (for evaluation breakdowns)
        self.detected_ap_counts = np.sum(self.raw_features != float(MISSING_RSSI), axis=1)

        # Pre-convert targets to tensors for zero-overhead batching
        self.buildings_tensor = torch.from_numpy(self.buildings)
        self.floors_tensor = torch.from_numpy(self.floors)
        self.coordinates_tensor = torch.from_numpy(self.coordinates)

        # If no augmentation is configured, pre-compute normalized feature tensor once
        if self.augmenter is None:
            norm_feats = raw_to_linear_positive(
                self.raw_features, include_missing_mask=self.include_missing_mask
            )
            self.precomputed_features: torch.Tensor | None = torch.from_numpy(norm_feats)
        else:
            self.precomputed_features = None

    def __len__(self) -> int:
        return len(self.raw_features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if self.precomputed_features is not None:
            features = self.precomputed_features[idx]
        else:
            raw_scan = self.raw_features[idx].copy()
            # Deterministic, unique RNG sequence per sample index & epoch seed
            rng = np.random.default_rng(self.seed + idx * 10007)
            raw_scan = self.augmenter(raw_scan, rng)
            features = torch.from_numpy(
                raw_to_linear_positive(
                    raw_scan, include_missing_mask=self.include_missing_mask
                )
            )

        return {
            "features": features,
            "building": self.buildings_tensor[idx],
            "floor": self.floors_tensor[idx],
            "coordinates": self.coordinates_tensor[idx],
            "userid": int(self.user_ids[idx]),
            "phoneid": int(self.phone_ids[idx]),
            "detected_count": int(self.detected_ap_counts[idx]),
            "index": int(idx),
        }


def restore_520_waps(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has all 520 WAPs in canonical order WAP001..WAP520."""
    missing_cols = [c for c in ALL_WAP_COLUMNS if c not in frame.columns]
    if missing_cols:
        missing_df = pd.DataFrame(float(MISSING_RSSI), index=frame.index, columns=missing_cols)
        result = pd.concat([frame, missing_df], axis=1)
    else:
        result = frame
    return result[ALL_WAP_COLUMNS]


def load_phase2_fold_data(
    fold_id: int,
    augmenter: AugmentdBm | None = None,
    include_missing_mask: bool = False,
    data_dir: str | Path | None = None,
    splits_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[WifiDataset, WifiDataset, CoordinateScaler, dict]:
    """Load fold-safe train and validation Datasets and fitted CoordinateScaler.
    
    Guarantees:
    - 520 WAP positions restored.
    - Outer validation user partition held out completely.
    - CoordinateScaler fitted ONLY on training users.
    - Validation dataset has NO augmentation (augmenter=None).
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    if splits_path is None:
        splits_path = Path(__file__).resolve().parents[2] / "data" / "splits" / "folds.json"

    train_df, _, quality = load_data(data_dir=data_dir)
    with open(splits_path) as f:
        splits_data = json.load(f)

    fold_info = next(f for f in splits_data["folds"] if f["fold"] == fold_id)
    train_idx = np.array(fold_info["train_idx"], dtype=np.int64)
    valid_idx = np.array(fold_info["valid_idx"], dtype=np.int64)

    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    valid_part = train_df.iloc[valid_idx].reset_index(drop=True)

    # 1. Restore all 520 WAPs
    train_waps_df = restore_520_waps(train_part)
    valid_waps_df = restore_520_waps(valid_part)

    train_raw = train_waps_df.to_numpy(dtype=np.float32)
    valid_raw = valid_waps_df.to_numpy(dtype=np.float32)

    # 2. Fit CoordinateScaler on training fold ONLY
    coord_scaler = CoordinateScaler().fit(train_part[["LONGITUDE", "LATITUDE"]].to_numpy())
    train_coords_norm = coord_scaler.transform(train_part[["LONGITUDE", "LATITUDE"]].to_numpy())
    valid_coords_norm = coord_scaler.transform(valid_part[["LONGITUDE", "LATITUDE"]].to_numpy())

    # 3. Create Datasets (validation strictly has NO augmentation)
    train_dataset = WifiDataset(
        raw_features=train_raw,
        buildings=train_part["BUILDINGID"].to_numpy(),
        floors=train_part["FLOOR"].to_numpy(),
        coordinates=train_coords_norm,
        user_ids=train_part["USERID"].to_numpy(),
        phone_ids=train_part["PHONEID"].to_numpy(),
        augmenter=augmenter,
        include_missing_mask=include_missing_mask,
        seed=seed,
    )

    valid_dataset = WifiDataset(
        raw_features=valid_raw,
        buildings=valid_part["BUILDINGID"].to_numpy(),
        floors=valid_part["FLOOR"].to_numpy(),
        coordinates=valid_coords_norm,
        user_ids=valid_part["USERID"].to_numpy(),
        phone_ids=valid_part["PHONEID"].to_numpy(),
        augmenter=None,  # NEVER augment validation rows
        include_missing_mask=include_missing_mask,
        seed=seed,
    )

    metadata = {
        "fold": fold_id,
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "train_users": sorted(int(u) for u in train_part["USERID"].unique()),
        "valid_users": sorted(int(u) for u in valid_part["USERID"].unique()),
        "input_dim": 1040 if include_missing_mask else 520,
    }
    return train_dataset, valid_dataset, coord_scaler, metadata


def create_phase2_dataloaders(
    train_dataset: WifiDataset,
    valid_dataset: WifiDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible DataLoaders with deterministic generator."""
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, valid_loader

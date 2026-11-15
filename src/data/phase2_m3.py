"""Fold-safe PyTorch data loading for Phase 2 models.

Inputs follow the Phase 1 linear-positive RSSI convention. The 55 WAPs removed
as training-constant by Phase 1 are restored as all-missing zero features so
the MLP and autoencoder keep the guide's stable 520-value input contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.load import MISSING_RSSI, WAP_COLUMNS


@dataclass(frozen=True)
class RSSIAugmentation:
    """Training-only corruption settings in physical RSSI units."""

    noise_std_dbm: float = 0.0
    ap_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.noise_std_dbm < 0:
            raise ValueError("noise_std_dbm must be non-negative")
        if not 0.0 <= self.ap_dropout_probability <= 1.0:
            raise ValueError("ap_dropout_probability must be in [0, 1]")


def raw_rssi_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return all 520 WAPs in canonical order, restoring removed WAPs as missing."""
    raw = np.full((len(frame), len(WAP_COLUMNS)), MISSING_RSSI, dtype=np.float32)
    present = [column for column in WAP_COLUMNS if column in frame.columns]
    if present:
        positions = {column: index for index, column in enumerate(WAP_COLUMNS)}
        raw[:, [positions[column] for column in present]] = frame[present].to_numpy(
            dtype=np.float32
        )
    return raw


def augment_rssi(
    raw_rssi: torch.Tensor,
    augmentation: RSSIAugmentation,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add noise to observed RSSI only, then optionally hide observed APs."""
    values = raw_rssi.clone()
    observed = values.ne(float(MISSING_RSSI))
    if augmentation.noise_std_dbm:
        noise = (
            torch.randn(
                values.shape,
                dtype=values.dtype,
                device=values.device,
                generator=generator,
            )
            * augmentation.noise_std_dbm
        )
        values[observed] = (values + noise)[observed].clamp(min=-110.0, max=0.0)
    if augmentation.ap_dropout_probability:
        keep = (
            torch.rand(
                values.shape,
                dtype=values.dtype,
                device=values.device,
                generator=generator,
            )
            >= augmentation.ap_dropout_probability
        )
        values[observed & ~keep] = float(MISSING_RSSI)
    return values


def linear_positive_rssi(raw_rssi: torch.Tensor) -> torch.Tensor:
    """Match Phase 1 ``linear_positive``: RSSI in [0,1], missing values at zero."""
    observed = raw_rssi.ne(float(MISSING_RSSI))
    dbm = torch.where(observed, raw_rssi, torch.full_like(raw_rssi, -110.0))
    normalized = ((dbm + 110.0) / 110.0).clamp(0.0, 1.0)
    return torch.where(observed, normalized, torch.zeros_like(normalized))


def fit_coordinate_scaler(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit longitude/latitude scaling from training rows only."""
    coordinates = frame[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=np.float32)
    mean = coordinates.mean(axis=0)
    scale = coordinates.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


class Phase2Dataset(Dataset[dict[str, torch.Tensor]]):
    """Multitask data with optional augmentation applied only at item access."""

    def __init__(
        self,
        frame: pd.DataFrame,
        coordinate_mean: np.ndarray,
        coordinate_scale: np.ndarray,
        *,
        augmentation: RSSIAugmentation | None = None,
        include_missingness: bool = False,
    ) -> None:
        self.raw_rssi = torch.from_numpy(raw_rssi_matrix(frame))
        self.building = torch.as_tensor(
            frame["BUILDINGID"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self.floor = torch.as_tensor(
            frame["FLOOR"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        coordinates = frame[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=np.float32)
        self.coordinates = torch.from_numpy(
            (coordinates - coordinate_mean) / coordinate_scale
        )
        self.augmentation = augmentation or RSSIAugmentation()
        self.include_missingness = include_missingness

    def __len__(self) -> int:
        return len(self.raw_rssi)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw = augment_rssi(self.raw_rssi[index], self.augmentation)
        observed = raw.ne(float(MISSING_RSSI))
        features = linear_positive_rssi(raw)
        if self.include_missingness:
            features = torch.cat((features, (~observed).to(features.dtype)))
        return {
            "features": features,
            "building": self.building[index],
            "floor": self.floor[index],
            "coordinates": self.coordinates[index],
        }


def make_phase2_dataloaders(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    *,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0,
    augmentation: RSSIAugmentation | None = None,
    include_missingness: bool = False,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Build loaders and return the train-fold coordinate mean and scale."""
    if train_frame.empty or valid_frame.empty:
        raise ValueError("train_frame and valid_frame must both be non-empty")
    mean, scale = fit_coordinate_scaler(train_frame)
    train_dataset = Phase2Dataset(
        train_frame,
        mean,
        scale,
        augmentation=augmentation,
        include_missingness=include_missingness,
    )
    valid_dataset = Phase2Dataset(
        valid_frame,
        mean,
        scale,
        augmentation=RSSIAugmentation(),
        include_missingness=include_missingness,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_seed_init,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_seed_init,
    )
    return train_loader, valid_loader, mean, scale


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Move every batch tensor to the selected accelerator."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def worker_seed_init(worker_id: int) -> None:
    """Seed NumPy in each worker from PyTorch's per-worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)

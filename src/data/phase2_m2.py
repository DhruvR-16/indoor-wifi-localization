"""Representation-aware, fold-safe Phase 2 data path for A4 / A7 / A8.

Member 1's frozen loader still decides which rows a model may see: the outer
USERID-grouped partition, the 520-WAP restoration and the train-only
CoordinateScaler are reused unchanged. Only two things are Member 2's own:

1. The input view is injected as a `RepresentationSpec`, since A4 compares
   three views. Augmentation still runs in the raw dBm domain before the view
   is applied, as the input contract requires.
2. The inner early-stopping split is label-coverage-checked. Member 1's plain
   `GroupKFold` leaves building 0 out of inner training on folds 0 and 1,
   where it supplies 76% and 56% of inner-validation rows; the inner EvAAL
   signal there is 183 m and 138 m against an outer score near 18 m. Pass
   ``mode="legacy"`` to reproduce that protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from src.data.load import MISSING_RSSI
from src.data.phase2 import AugmentdBm, CoordinateScaler, load_phase2_fold_data
from src.features.rssi_representations import RepresentationSpec

InnerSplitMode = Literal["stratified", "legacy"]


class RepresentedWifiDataset(Dataset):
    """Phase 2 dataset whose input view is a configurable `RepresentationSpec`.

    Emits the same batch keys as `src.data.phase2.WifiDataset`, so Member 1's
    training and metric helpers accept it unchanged.
    """

    def __init__(
        self,
        raw_features: np.ndarray,
        buildings: np.ndarray,
        floors: np.ndarray,
        coordinates: np.ndarray,
        user_ids: np.ndarray,
        phone_ids: np.ndarray,
        spec: RepresentationSpec,
        augmenter: AugmentdBm | None = None,
        seed: int = 42,
    ) -> None:
        lengths = {
            len(raw_features),
            len(buildings),
            len(floors),
            len(coordinates),
            len(user_ids),
            len(phone_ids),
        }
        if len(lengths) != 1:
            raise ValueError(
                "All feature and target arrays must have identical row length."
            )

        self.raw_features = np.asarray(raw_features, dtype=np.float32)
        self.buildings = np.asarray(buildings, dtype=np.int64)
        self.floors = np.asarray(floors, dtype=np.int64)
        self.coordinates = np.asarray(coordinates, dtype=np.float32)
        self.user_ids = np.asarray(user_ids, dtype=np.int64)
        self.phone_ids = np.asarray(phone_ids, dtype=np.int64)
        self.spec = spec
        self.augmenter = augmenter
        self.seed = seed

        self.detected_ap_counts = np.sum(
            self.raw_features != float(MISSING_RSSI), axis=1
        )

        self.buildings_tensor = torch.from_numpy(self.buildings)
        self.floors_tensor = torch.from_numpy(self.floors)
        self.coordinates_tensor = torch.from_numpy(self.coordinates)

        if self.augmenter is None:
            self.precomputed_features: torch.Tensor | None = torch.from_numpy(
                spec(self.raw_features)
            )
        else:
            self.precomputed_features = None

    def __len__(self) -> int:
        return len(self.raw_features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        if self.precomputed_features is not None:
            features = self.precomputed_features[idx]
        else:
            raw_scan = self.raw_features[idx].copy()
            # Same per-index RNG stream as Member 1's dataset, for comparability.
            rng = np.random.default_rng(self.seed + idx * 10007)
            raw_scan = self.augmenter(raw_scan, rng)
            features = torch.from_numpy(self.spec(raw_scan))

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

    def subset(
        self, indices: np.ndarray, augmenter: AugmentdBm | None
    ) -> RepresentedWifiDataset:
        """Row subset carrying its own augmenter (used to build inner partitions)."""
        idx = np.asarray(indices, dtype=np.int64)
        return RepresentedWifiDataset(
            raw_features=self.raw_features[idx],
            buildings=self.buildings[idx],
            floors=self.floors[idx],
            coordinates=self.coordinates[idx],
            user_ids=self.user_ids[idx],
            phone_ids=self.phone_ids[idx],
            spec=self.spec,
            augmenter=augmenter,
            seed=self.seed,
        )


@dataclass(frozen=True)
class InnerSplitReport:
    """Label coverage of an inner split, recorded alongside every run."""

    mode: str
    inner_train_rows: int
    inner_val_rows: int
    inner_train_users: list[int]
    inner_val_users: list[int]
    buildings_missing_from_inner_train: list[int]
    floors_missing_from_inner_train: list[int]
    uncovered_inner_val_row_share: float

    @property
    def is_covered(self) -> bool:
        return not self.buildings_missing_from_inner_train


def _coverage(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    buildings: np.ndarray,
    floors: np.ndarray,
) -> tuple[list[int], list[int], float]:
    missing_buildings = sorted(set(buildings[val_idx]) - set(buildings[train_idx]))
    missing_floors = sorted(set(floors[val_idx]) - set(floors[train_idx]))
    uncovered = (
        float(np.isin(buildings[val_idx], missing_buildings).mean())
        if missing_buildings
        else 0.0
    )
    return (
        [int(b) for b in missing_buildings],
        [int(f) for f in missing_floors],
        uncovered,
    )


def make_inner_split(
    user_ids: np.ndarray,
    buildings: np.ndarray,
    floors: np.ndarray,
    n_splits: int = 4,
    mode: InnerSplitMode = "stratified",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, InnerSplitReport]:
    """Split outer-training rows into inner train/validation, keeping users intact.

    ``legacy`` reproduces Member 1's protocol exactly: plain `GroupKFold`, first
    candidate split, whatever its label coverage.

    ``stratified`` (default) draws candidates from a building-stratified grouped
    splitter and keeps the first candidate whose **inner-training** partition
    covers every building and floor present in inner validation, breaking ties
    toward the most size-balanced candidate. This matters because building 0 is
    visited by only two users (1 and 11): on outer folds 0 and 1 exactly one of
    them remains in outer training, so a careless inner split drops building 0
    from inner training entirely. Only outer-training rows are consulted, so the
    held-out users are untouched by this choice.
    """
    user_ids = np.asarray(user_ids)
    buildings = np.asarray(buildings)
    floors = np.asarray(floors)
    n_rows = len(user_ids)

    if mode == "legacy":
        splitter = GroupKFold(n_splits=n_splits)
        candidates = list(splitter.split(np.zeros(n_rows), groups=user_ids))[:1]
    elif mode == "stratified":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        candidates = list(
            splitter.split(np.zeros(n_rows), y=buildings, groups=user_ids)
        )
    else:
        raise ValueError(f"Unknown inner split mode {mode!r}")

    target_val_share = 1.0 / n_splits
    scored = []
    for cand_train, cand_val in candidates:
        cand_train = np.asarray(cand_train, dtype=np.int64)
        cand_val = np.asarray(cand_val, dtype=np.int64)
        if len(cand_val) == 0 or len(cand_train) == 0:
            continue
        miss_b, miss_f, uncovered = _coverage(cand_train, cand_val, buildings, floors)
        imbalance = abs(len(cand_val) / n_rows - target_val_share)
        # Prefer: no missing building, then no missing floor, then size balance.
        scored.append(((len(miss_b), len(miss_f), imbalance), cand_train, cand_val))

    if not scored:
        raise AssertionError("Inner splitter produced no usable candidate split")
    scored.sort(key=lambda item: item[0])
    _, train_idx, val_idx = scored[0]

    overlap = set(user_ids[train_idx]) & set(user_ids[val_idx])
    if overlap:
        raise AssertionError(
            f"Inner split leaked users across the boundary: {sorted(overlap)}"
        )

    missing_buildings, missing_floors, uncovered = _coverage(
        train_idx, val_idx, buildings, floors
    )
    report = InnerSplitReport(
        mode=mode,
        inner_train_rows=int(len(train_idx)),
        inner_val_rows=int(len(val_idx)),
        inner_train_users=sorted(int(u) for u in set(user_ids[train_idx])),
        inner_val_users=sorted(int(u) for u in set(user_ids[val_idx])),
        buildings_missing_from_inner_train=missing_buildings,
        floors_missing_from_inner_train=missing_floors,
        uncovered_inner_val_row_share=uncovered,
    )
    return train_idx, val_idx, report


def load_m2_fold_data(
    fold_id: int,
    spec: RepresentationSpec,
    augmenter: AugmentdBm | None = None,
    seed: int = 42,
    data_dir: str | Path | None = None,
    splits_path: str | Path | None = None,
) -> tuple[RepresentedWifiDataset, RepresentedWifiDataset, CoordinateScaler, dict]:
    """Outer train/validation datasets under ``spec``, reusing M1's fold partition.

    The outer validation dataset is never augmented, and the returned
    `CoordinateScaler` is the one Member 1's loader fitted on training users only.
    """
    train_ds, valid_ds, coord_scaler, meta = load_phase2_fold_data(
        fold_id=fold_id,
        augmenter=None,  # raw dBm is kept; the view is applied below
        include_missing_mask=False,
        data_dir=data_dir,
        splits_path=splits_path,
        seed=seed,
    )

    m2_train = RepresentedWifiDataset(
        raw_features=train_ds.raw_features,
        buildings=train_ds.buildings,
        floors=train_ds.floors,
        coordinates=train_ds.coordinates,
        user_ids=train_ds.user_ids,
        phone_ids=train_ds.phone_ids,
        spec=spec,
        augmenter=augmenter,
        seed=seed,
    )
    m2_valid = RepresentedWifiDataset(
        raw_features=valid_ds.raw_features,
        buildings=valid_ds.buildings,
        floors=valid_ds.floors,
        coordinates=valid_ds.coordinates,
        user_ids=valid_ds.user_ids,
        phone_ids=valid_ds.phone_ids,
        spec=spec,
        augmenter=None,  # NEVER augment held-out users
        seed=seed,
    )

    meta = dict(meta)
    meta["representation"] = spec.name
    meta["include_missing_mask"] = spec.include_missing_mask
    meta["input_dim"] = spec.input_dim()
    return m2_train, m2_valid, coord_scaler, meta


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    """Deterministic DataLoader (single process, seeded generator)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
    )

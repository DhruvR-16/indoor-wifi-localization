"""A8 hybrid: multitask-trunk embedding plus a routed local coordinate estimator.

Routing reuses Phase 1's `HierarchicalCoordinateRegressor` unchanged, so the
hybrid differs from the Phase 1 coordinate route in exactly one factor - the
feature space - and inherits its route fallbacks and ``min_samples`` rule.

Leakage controls: embeddings come from the fold-specific model; the estimator
is fitted only on outer-training embeddings and coordinates; its settings are
tuned on an inner split using an encoder that never saw those rows; held-out
rows are routed by predicted building/floor, never by true labels except in
the explicitly named oracle diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, Dataset

from src.models.ml.regressors import (
    HierarchicalCoordinateRegressor,
    make_base_regressor,
)

RouteMode = Literal["global", "per_building", "per_building_floor"]

#: Phase 1's kNN grid (configs/phase1_regressors.yaml), reused so the hybrid is
#: tuned over the same candidate settings as the baseline it is compared to.
DEFAULT_KNN_GRID: tuple[dict[str, Any], ...] = tuple(
    {"n_neighbors": k, "weights": w, "p": p}
    for k in (3, 5, 7, 9, 11, 15)
    for w in ("distance",)
    for p in (1, 2)
)


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Run `model.encode` over a dataset, returning the embedding and the labels.

    Also returns the model's own predicted building/floor, which is what the
    end-to-end route must use for held-out rows.
    """
    if not hasattr(model, "encode"):
        raise AttributeError(
            "Multitask model does not expose .encode(features); the A8 hybrid "
            "requires the encode() interface frozen in docs/phase2_input_contract.md"
        )
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings, coords, buildings, floors = [], [], [], []
    predicted_buildings, predicted_floors, indices, users = [], [], [], []
    for batch in loader:
        features = batch["features"].to(device)
        latent = model.encode(features)
        outputs = model(features)
        embeddings.append(latent.cpu().numpy())
        predicted_buildings.append(
            torch.argmax(outputs["building_logits"], dim=1).cpu().numpy()
        )
        predicted_floors.append(
            torch.argmax(outputs["floor_logits"], dim=1).cpu().numpy()
        )
        coords.append(batch["coordinates"].numpy())
        buildings.append(batch["building"].numpy())
        floors.append(batch["floor"].numpy())
        indices.append(batch["index"].numpy())
        users.append(batch["userid"].numpy())

    return {
        "embedding": np.concatenate(embeddings).astype(np.float32),
        "coordinates_norm": np.concatenate(coords).astype(np.float64),
        "building_true": np.concatenate(buildings),
        "floor_true": np.concatenate(floors),
        "building_pred": np.concatenate(predicted_buildings),
        "floor_pred": np.concatenate(predicted_floors),
        "index": np.concatenate(indices),
        "userid": np.concatenate(users),
    }


@dataclass
class RoutedEstimatorSelection:
    """Outcome of tuning a routed estimator inside the outer-training users."""

    estimator_name: str
    route_mode: str
    params: dict[str, Any]
    inner_mean_position_error_m: float
    candidates: list[dict[str, Any]] = field(default_factory=list)
    routed_by: str = "predicted"


def _position_error_metres(
    predicted_norm: np.ndarray,
    true_norm: np.ndarray,
    coord_scaler,
) -> np.ndarray:
    predicted_m = coord_scaler.inverse_transform(predicted_norm)
    true_m = coord_scaler.inverse_transform(true_norm)
    return np.hypot(predicted_m[:, 0] - true_m[:, 0], predicted_m[:, 1] - true_m[:, 1])


def fit_routed_estimator(
    embeddings: np.ndarray,
    coordinates_norm: np.ndarray,
    buildings: np.ndarray,
    floors: np.ndarray,
    estimator: BaseEstimator,
    route_mode: RouteMode,
    min_samples: int = 10,
) -> HierarchicalCoordinateRegressor:
    """Fit Phase 1's routed regressor on embeddings of outer-training rows only."""
    routed = HierarchicalCoordinateRegressor(
        base_estimator=estimator, routing=route_mode, min_samples=min_samples
    )
    routed.fit(embeddings, coordinates_norm, buildings=buildings, floors=floors)
    return routed


def tune_routed_estimator(
    inner_train: dict[str, np.ndarray],
    inner_val: dict[str, np.ndarray],
    coord_scaler,
    estimator_name: str = "knn",
    route_mode: RouteMode = "per_building_floor",
    grid: Sequence[dict[str, Any]] = DEFAULT_KNN_GRID,
    seed: int = 42,
    min_samples: int = 10,
    route_with_predictions: bool = True,
) -> RoutedEstimatorSelection:
    """Pick estimator settings using inner-validation rows from outer training.

    Scored by mean position error in metres, because the estimator under test
    only produces coordinates; the classification terms of the primary metric
    are unaffected by this choice.
    """
    route_key = "pred" if route_with_predictions else "true"
    candidates: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None

    for params in grid:
        estimator = make_base_regressor(estimator_name, seed=seed, **params)
        routed = fit_routed_estimator(
            embeddings=inner_train["embedding"],
            coordinates_norm=inner_train["coordinates_norm"],
            buildings=inner_train["building_true"],
            floors=inner_train["floor_true"],
            estimator=estimator,
            route_mode=route_mode,
            min_samples=min_samples,
        )
        predicted = routed.predict(
            inner_val["embedding"],
            route_buildings=inner_val[f"building_{route_key}"],
            route_floors=inner_val[f"floor_{route_key}"],
        )
        error = float(
            np.mean(
                _position_error_metres(
                    predicted, inner_val["coordinates_norm"], coord_scaler
                )
            )
        )
        candidates.append(
            {"params": dict(params), "inner_mean_position_error_m": error}
        )
        if best is None or error < best[0]:
            best = (error, dict(params))

    assert best is not None, "empty estimator grid"
    return RoutedEstimatorSelection(
        estimator_name=estimator_name,
        route_mode=route_mode,
        params=best[1],
        inner_mean_position_error_m=best[0],
        candidates=candidates,
        routed_by="predicted" if route_with_predictions else "oracle",
    )

"""Unit tests for Member 2's Phase 2 representation, inner-split, and DAE invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.load import MISSING_RSSI, load_data
from src.data.phase2 import AugmentdBm
from src.data.phase2_m2 import RepresentedWifiDataset, make_inner_split
from src.features.rssi_representations import REPRESENTATIONS, RepresentationSpec
from src.models.dae import (
    build_dae,
    load_encoder_into_trunk,
    masked_reconstruction_loss,
)
from src.models.stub_multitask import build_multitask_model

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "splits" / "folds.json"


@pytest.fixture(scope="module")
def sample_scan() -> np.ndarray:
    scan = np.full(520, float(MISSING_RSSI), dtype=np.float32)
    scan[5] = -104.0
    scan[10] = -90.0
    scan[20] = -67.0
    scan[30] = -40.0
    return scan


@pytest.mark.parametrize("name", REPRESENTATIONS)
def test_sentinel_maps_to_zero_and_detected_stays_positive(
    name: str, sample_scan
) -> None:
    """Every representation must honour the frozen input contract."""
    values = RepresentationSpec(name)(sample_scan)
    observed = sample_scan != float(MISSING_RSSI)

    assert values.shape == sample_scan.shape
    assert np.all(
        values[~observed] == 0.0
    ), "not-detected entries must map strictly to 0.0"
    assert np.all(
        values[observed] > 0.0
    ), "detected readings must stay strictly positive"
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert values.dtype == np.float32


@pytest.mark.parametrize("name", REPRESENTATIONS)
def test_representation_is_monotone_in_rssi(name: str) -> None:
    """A stronger reading must never map to a smaller value."""
    ladder = np.array(
        [-104.0, -95.0, -85.0, -70.0, -55.0, -30.0, 0.0], dtype=np.float32
    )
    values = RepresentationSpec(name)(ladder)
    assert np.all(np.diff(values) > 0.0)
    assert values[-1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", REPRESENTATIONS)
def test_representation_is_stateless_and_fold_safe(name: str) -> None:
    """Nothing is fitted, so a row's view cannot depend on the other rows present."""
    rng = np.random.default_rng(0)
    batch = np.where(
        rng.random((40, 520)) < 0.05,
        rng.uniform(-104.0, -30.0, size=(40, 520)),
        float(MISSING_RSSI),
    ).astype(np.float32)
    spec = RepresentationSpec(name)
    full = spec(batch)
    subset = spec(batch[:7])
    np.testing.assert_allclose(full[:7], subset, rtol=0, atol=0)


def test_missing_mask_doubles_width_and_flags_detection(sample_scan) -> None:
    spec = RepresentationSpec("linear", include_missing_mask=True)
    values = spec(sample_scan)
    assert values.shape == (1040,)
    assert spec.input_dim() == 1040
    flags = values[520:]
    np.testing.assert_array_equal(
        flags, (sample_scan != float(MISSING_RSSI)).astype(np.float32)
    )


def test_unknown_representation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown representation"):
        RepresentationSpec("sigmoid")


def test_exponential_and_powed_differ_from_linear(sample_scan) -> None:
    """The A4 arms must actually be different inputs, not relabelled copies."""
    linear = RepresentationSpec("linear")(sample_scan)
    exponential = RepresentationSpec("exponential")(sample_scan)
    powed = RepresentationSpec("powed")(sample_scan)
    observed = sample_scan != float(MISSING_RSSI)
    assert not np.allclose(linear[observed], exponential[observed])
    assert not np.allclose(linear[observed], powed[observed])
    # Both non-linear views compress weak signals relative to the linear one.
    assert np.all(exponential[observed] <= linear[observed] + 1e-6)
    assert np.all(powed[observed] <= linear[observed] + 1e-6)


def test_augmentation_runs_before_representation_and_preserves_sentinel() -> None:
    """Corruption happens in dBm; the sentinel survives into the final view as 0.0."""
    scan = np.full(520, float(MISSING_RSSI), dtype=np.float32)
    scan[:30] = -70.0
    dataset = RepresentedWifiDataset(
        raw_features=scan[None, :],
        buildings=np.array([1]),
        floors=np.array([2]),
        coordinates=np.zeros((1, 2)),
        user_ids=np.array([7]),
        phone_ids=np.array([13]),
        spec=RepresentationSpec("powed"),
        augmenter=AugmentdBm(noise_sigma=4.0, ap_dropout=0.5),
        seed=42,
    )
    features = dataset[0]["features"].numpy()
    assert features.shape == (520,)
    # Entries that were never detected stay exactly zero after augmentation.
    assert np.all(features[30:] == 0.0)
    # Dropout may zero some observed APs, but it cannot invent new detections.
    assert np.count_nonzero(features[:30]) <= 30


def test_dataset_without_augmenter_is_deterministic(sample_scan) -> None:
    def build() -> RepresentedWifiDataset:
        return RepresentedWifiDataset(
            raw_features=np.repeat(sample_scan[None, :], 4, axis=0),
            buildings=np.zeros(4, dtype=np.int64),
            floors=np.zeros(4, dtype=np.int64),
            coordinates=np.zeros((4, 2)),
            user_ids=np.arange(4),
            phone_ids=np.arange(4),
            spec=RepresentationSpec("exponential"),
            augmenter=None,
            seed=42,
        )

    first, second = build()[2]["features"], build()[2]["features"]
    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="requires data/splits/folds.json")
@pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
def test_stratified_inner_split_keeps_users_intact_and_covers_buildings(
    fold: int,
) -> None:
    """The early-stopping split must not drop a building from inner training.

    Member 1's plain GroupKFold does exactly that on folds 0 and 1, where only
    one building-0 user remains in outer training.
    """
    train_df, _, _ = load_data()
    folds = json.loads(SPLITS_PATH.read_text())
    fold_info = next(f for f in folds["folds"] if f["fold"] == fold)
    outer_train = train_df.iloc[fold_info["train_idx"]]

    train_idx, val_idx, report = make_inner_split(
        user_ids=outer_train["USERID"].to_numpy(),
        buildings=outer_train["BUILDINGID"].to_numpy(),
        floors=outer_train["FLOOR"].to_numpy(),
        mode="stratified",
        seed=42,
    )

    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(train_idx) + len(val_idx) == len(outer_train)
    assert not set(report.inner_train_users) & set(report.inner_val_users)
    assert report.buildings_missing_from_inner_train == []
    assert report.floors_missing_from_inner_train == []
    assert report.is_covered
    assert 0.1 < report.inner_val_rows / len(outer_train) < 0.4


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="requires data/splits/folds.json")
def test_legacy_inner_split_reproduces_the_coverage_defect() -> None:
    """Guard the comparison: the legacy mode must stay faithful to M1's protocol."""
    train_df, _, _ = load_data()
    folds = json.loads(SPLITS_PATH.read_text())
    fold_info = next(f for f in folds["folds"] if f["fold"] == 0)
    outer_train = train_df.iloc[fold_info["train_idx"]]

    _, _, report = make_inner_split(
        user_ids=outer_train["USERID"].to_numpy(),
        buildings=outer_train["BUILDINGID"].to_numpy(),
        floors=outer_train["FLOOR"].to_numpy(),
        mode="legacy",
        seed=42,
    )
    assert report.buildings_missing_from_inner_train == [0]
    assert report.uncovered_inner_val_row_share > 0.5


def test_dae_encoder_is_architecture_matched_to_the_trunk() -> None:
    """A7's two arms are only comparable if the encoder transfers exactly."""
    dae = build_dae(input_dim=520, hidden_dims=(256, 128), seed=42)
    model = build_multitask_model(input_dim=520, hidden_dims=(256, 128), seed=7)

    transferred = load_encoder_into_trunk(
        model.trunk, dae.encoder.state_dict(), strict=True
    )
    assert transferred, "no tensors were transferred"
    for key, tensor in dae.encoder.state_dict().items():
        torch.testing.assert_close(model.trunk.state_dict()[key], tensor)


def test_mismatched_encoder_transfer_is_refused_not_reshaped() -> None:
    """The guide's 520-512-256-128 trunk must fail loudly against a 520-256-128 DAE."""
    dae = build_dae(input_dim=520, hidden_dims=(256, 128), seed=42)
    wider = build_multitask_model(input_dim=520, hidden_dims=(512, 256, 128), seed=7)
    with pytest.raises(ValueError, match="architecture-matched"):
        load_encoder_into_trunk(wider.trunk, dae.encoder.state_dict(), strict=True)


def test_masked_reconstruction_reports_observed_and_missing_separately() -> None:
    target = torch.zeros(2, 6)
    target[0, :2] = torch.tensor([0.4, 0.8])
    target[1, 0] = 0.5
    observed = target > 0.0
    reconstruction = target.clone()
    reconstruction[observed] += 0.1  # error only on observed entries

    loss = masked_reconstruction_loss(
        reconstruction, target, observed, observed_weight=1.0, missing_weight=0.1
    )
    assert loss.observed_entries == 3
    assert loss.missing_entries == 9
    assert loss.observed_mse.item() == pytest.approx(0.01, abs=1e-6)
    assert loss.missing_mse.item() == pytest.approx(0.0, abs=1e-9)
    assert loss.total.item() == pytest.approx(0.01, abs=1e-6)


def test_masked_reconstruction_weighting_shifts_with_missing_weight() -> None:
    target = torch.zeros(1, 4)
    target[0, 0] = 0.6
    observed = target > 0.0
    reconstruction = torch.full((1, 4), 0.2)

    low = masked_reconstruction_loss(reconstruction, target, observed, 1.0, 0.1)
    high = masked_reconstruction_loss(reconstruction, target, observed, 1.0, 1.0)
    assert high.total.item() > low.total.item()
    assert low.observed_mse.item() == pytest.approx(high.observed_mse.item())

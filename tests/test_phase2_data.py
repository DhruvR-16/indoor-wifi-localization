"""Unit tests for Phase 2 data loading, augmentation, and coordinate scaling invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.load import MISSING_RSSI
from src.data.phase2 import (
    ALL_WAP_COLUMNS,
    AugmentdBm,
    CoordinateScaler,
    WifiDataset,
    load_phase2_fold_data,
    raw_to_linear_positive,
    restore_520_waps,
)


def test_520_wap_ordering_and_restoration() -> None:
    """Check that all 520 WAPs exist in strict ascending order WAP001..WAP520."""
    assert len(ALL_WAP_COLUMNS) == 520
    assert ALL_WAP_COLUMNS[0] == "WAP001"
    assert ALL_WAP_COLUMNS[-1] == "WAP520"

    # Test restoration from sparse dataframe
    import pandas as pd

    sparse_df = pd.DataFrame(
        {"WAP005": [-75.0], "WAP010": [-80.0], "USERID": [1], "LONGITUDE": [0.0], "LATITUDE": [0.0]}
    )
    restored = restore_520_waps(sparse_df)
    assert restored.shape == (1, 520)
    assert list(restored.columns) == ALL_WAP_COLUMNS
    assert restored["WAP005"].iloc[0] == -75.0
    assert restored["WAP001"].iloc[0] == float(MISSING_RSSI)


def test_sentinel_never_corrupted_by_noise() -> None:
    """Noise must be added ONLY to observed APs; +100 sentinel must never be touched."""
    raw_scan = np.full(520, float(MISSING_RSSI), dtype=np.float32)
    # Observe two APs
    raw_scan[10] = -70.0
    raw_scan[50] = -85.0

    augmenter = AugmentdBm(noise_sigma=10.0, ap_dropout=0.0)
    rng = np.random.default_rng(1234)

    augmented = augmenter(raw_scan, rng)

    # Missing APs must stay exactly MISSING_RSSI
    unobserved_indices = [i for i in range(520) if i not in (10, 50)]
    assert np.all(augmented[unobserved_indices] == float(MISSING_RSSI))

    # Observed APs must be perturbed and clamped within [-110, 0]
    assert augmented[10] != -70.0
    assert -110.0 <= augmented[10] <= 0.0
    assert -110.0 <= augmented[50] <= 0.0


def test_ap_dropout_sets_to_missing_sentinel() -> None:
    """AP dropout must set dropped observed APs strictly to the missing sentinel."""
    raw_scan = np.full(520, -70.0, dtype=np.float32)
    augmenter = AugmentdBm(noise_sigma=0.0, ap_dropout=0.5)
    rng = np.random.default_rng(42)

    augmented = augmenter(raw_scan, rng)
    # Some APs must be dropped to MISSING_RSSI
    dropped = augmented == float(MISSING_RSSI)
    assert np.sum(dropped) > 0
    assert np.sum(~dropped) > 0
    assert np.all(augmented[~dropped] == -70.0)


def test_raw_to_linear_positive_transform() -> None:
    """Verify linear positive transform maps [-110, 0] -> [0, 1] and +100 -> 0.0."""
    raw_dbm = np.array([float(MISSING_RSSI), -110.0, -55.0, 0.0, -120.0, 10.0], dtype=np.float32)
    transformed = raw_to_linear_positive(raw_dbm)

    assert transformed[0] == 0.0  # Missing sentinel maps to 0.0
    assert pytest.approx(transformed[1], abs=1e-5) == 0.0  # Floor maps to 0.0
    assert pytest.approx(transformed[2], abs=1e-5) == 0.5  # Midpoint (-55 dBm) maps to 0.5
    assert pytest.approx(transformed[3], abs=1e-5) == 1.0  # 0 dBm maps to 1.0
    assert transformed[4] == 0.0  # Clamped below floor
    assert transformed[5] == 1.0  # Clamped above ceiling

    # Check 1040 indicator flag version
    with_mask = raw_to_linear_positive(raw_dbm, include_missing_mask=True)
    assert with_mask.shape == (12,)
    assert with_mask[6] == 0.0  # First element was missing -> mask is 0.0
    assert with_mask[7] == 1.0  # Observed -> mask is 1.0


def test_coordinate_scaler_train_only() -> None:
    """Ensure CoordinateScaler is fit on train and correctly reconstructs coordinates."""
    train_coords = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)
    val_coords = np.array([[500.0, 600.0]], dtype=np.float64)

    scaler = CoordinateScaler()
    scaler.fit(train_coords)

    assert pytest.approx(scaler.mean_[0]) == 200.0
    assert pytest.approx(scaler.mean_[1]) == 300.0

    transformed_val = scaler.transform(val_coords)
    reconstructed_val = scaler.inverse_transform(transformed_val)
    np.testing.assert_allclose(val_coords, reconstructed_val, atol=1e-5)


def test_dataset_reproducibility() -> None:
    """Verify dataset augmentation with fixed seed produces identical output."""
    raw = np.full((10, 520), float(MISSING_RSSI), dtype=np.float32)
    raw[:, :50] = -75.0
    targets = np.zeros(10, dtype=np.int64)
    coords = np.zeros((10, 2), dtype=np.float32)

    augmenter = AugmentdBm(noise_sigma=4.0, ap_dropout=0.1)

    ds1 = WifiDataset(raw, targets, targets, coords, targets, targets, augmenter=augmenter, seed=42)
    ds2 = WifiDataset(raw, targets, targets, coords, targets, targets, augmenter=augmenter, seed=42)

    item1 = ds1[0]["features"].numpy()
    item2 = ds2[0]["features"].numpy()

    np.testing.assert_array_equal(item1, item2)


def test_fold_data_loading_isolation() -> None:
    """Test outer fold loading guarantees validation has NO augmentation."""
    train_ds, val_ds, scaler, meta = load_phase2_fold_data(
        fold_id=0,
        augmenter=AugmentdBm(noise_sigma=2.0, ap_dropout=0.05),
        seed=42,
    )

    assert meta["fold"] == 0
    assert meta["input_dim"] == 520
    assert train_ds.augmenter is not None
    assert val_ds.augmenter is None  # Validation MUST NOT have augmenter

    # Check zero user overlap
    train_users = set(meta["train_users"])
    valid_users = set(meta["valid_users"])
    assert len(train_users.intersection(valid_users)) == 0


"""Tests for Member 2's aggregation: pooled coverage, pairing, clustered CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_m2_tables import paired_comparison, pooled_oof_by_seed

PRED_COLUMNS = [
    "index",
    "USERID",
    "PHONEID",
    "detected_count",
    "BUILDINGID_TRUE",
    "BUILDINGID_PRED",
    "FLOOR_TRUE",
    "FLOOR_PRED",
    "LONGITUDE_TRUE",
    "LATITUDE_TRUE",
    "LONGITUDE_PRED",
    "LATITUDE_PRED",
    "position_error",
]


def _write_preds(
    path,
    users: np.ndarray,
    errors: np.ndarray,
    indices: np.ndarray | None = None,
) -> None:
    """Write a predictions CSV whose only varying quantity is position error."""
    n = len(users)
    indices = np.arange(n) if indices is None else indices
    frame = pd.DataFrame(
        {
            "index": indices,
            "USERID": users,
            "PHONEID": np.zeros(n, dtype=int),
            "detected_count": np.full(n, 17),
            "BUILDINGID_TRUE": np.zeros(n, dtype=int),
            "BUILDINGID_PRED": np.zeros(n, dtype=int),
            "FLOOR_TRUE": np.zeros(n, dtype=int),
            "FLOOR_PRED": np.zeros(n, dtype=int),
            "LONGITUDE_TRUE": np.zeros(n),
            "LATITUDE_TRUE": np.zeros(n),
            "LONGITUDE_PRED": errors,
            "LATITUDE_PRED": np.zeros(n),
            "position_error": errors,
        }
    )
    frame[PRED_COLUMNS].to_csv(path, index=False)


def _records(
    tmp_path, arm_errors: dict[str, np.ndarray], users: np.ndarray
) -> list[dict]:
    records = []
    for arm, errors in arm_errors.items():
        path = tmp_path / f"preds_{arm}.csv"
        _write_preds(path, users, errors)
        records.append(
            {
                "exp_id": arm,
                "fold": 0,
                "seed": 42,
                "metrics": {"n_samples": len(users)},
                "predictions_file": str(path),
            }
        )
    return records


def test_uniform_improvement_across_users_is_called_an_improvement(tmp_path) -> None:
    users = np.repeat(np.arange(12), 20)
    reference = np.full(len(users), 20.0)
    better = np.full(len(users), 15.0)
    records = _records(tmp_path, {"ref": reference, "arm": better}, users)

    result = paired_comparison(records, reference="ref", n_boot=200, seed=0)
    row = result.set_index("exp_id").loc["arm"]
    assert row["delta_evaal_mean"] == pytest.approx(-5.0)
    assert row["clusters_userid"] == 12
    assert row["improves_reference"]
    assert not row["degrades_reference"]


def test_difference_driven_by_one_user_is_not_called_an_improvement(tmp_path) -> None:
    """A gain concentrated in a single held-out user must not clear the CI.

    This is the case a row-level iid bootstrap gets wrong: thousands of
    correlated rows from one user look like overwhelming evidence.
    """
    users = np.repeat(np.arange(12), 20)
    reference = np.full(len(users), 20.0)
    arm = reference.copy()
    arm[users == 0] -= 60.0  # one user improves hugely, the other eleven do not
    records = _records(tmp_path, {"ref": reference, "arm": arm}, users)

    result = paired_comparison(records, reference="ref", n_boot=400, seed=0)
    row = result.set_index("exp_id").loc["arm"]
    assert row["delta_evaal_mean"] < 0.0
    assert not row["improves_reference"]
    # The interval must still contain zero. It cannot exceed it here, because
    # no user is worse under the arm, so no resample can produce a positive mean.
    assert row["delta_evaal_ci_high"] >= 0.0


def test_unmatched_rows_are_refused(tmp_path) -> None:
    users = np.repeat(np.arange(4), 10)
    reference_path = tmp_path / "preds_ref.csv"
    arm_path = tmp_path / "preds_arm.csv"
    _write_preds(reference_path, users, np.full(len(users), 20.0))
    # Same row count, different row ids: the arms do not cover the same scans.
    _write_preds(
        arm_path, users, np.full(len(users), 18.0), indices=np.arange(len(users)) + 500
    )
    records = [
        {
            "exp_id": name,
            "fold": 0,
            "seed": 42,
            "metrics": {"n_samples": len(users)},
            "predictions_file": str(path),
        }
        for name, path in (("ref", reference_path), ("arm", arm_path))
    ]

    with pytest.raises(AssertionError, match="identical rows"):
        paired_comparison(records, reference="ref", n_boot=50, seed=0)


def test_pooled_oof_sums_folds_once_per_seed(tmp_path) -> None:
    records = []
    for fold in range(5):
        users = np.repeat([fold * 2, fold * 2 + 1], 10)
        path = tmp_path / f"preds_f{fold}.csv"
        _write_preds(path, users, np.full(len(users), 12.0), indices=np.arange(20))
        records.append(
            {
                "exp_id": "arm",
                "fold": fold,
                "seed": 42,
                "metrics": {"n_samples": 20},
                "predictions_file": str(path),
            }
        )

    pooled = pooled_oof_by_seed(records)
    assert len(pooled) == 1
    assert pooled.iloc[0]["folds_pooled"] == 5
    assert pooled.iloc[0]["pooled_rows"] == 100
    assert pooled.iloc[0]["mean_position_error"] == pytest.approx(12.0)


def test_oof_prediction_archive_round_trips_exactly(tmp_path):
    """The tracked archive must rebuild the per-run prediction files exactly.

    The per-run CSVs are gitignored, so the archive is the only tracked copy of
    the row-level out-of-fold predictions handed to Member 5. A lossy round-trip
    would silently corrupt that handoff.
    """
    import pandas as pd

    from scripts.export_m2_oof_predictions import READ_KWARGS, RUN_COLUMN

    frame = pd.DataFrame(
        {
            "index": [0, 1],
            "USERID": [11, 11],
            "LONGITUDE_TRUE": [-7632.143599578828, -7637.256998033113],
            "LATITUDE_TRUE": [4864982.217099939, 4864949.81429723],
            "position_error": [226.49338626134445, 30.119649975681778],
        }
    )

    packed = frame.copy()
    packed.insert(0, RUN_COLUMN, "a4_linear_f0_s42")
    archive = tmp_path / "archive.csv.gz"
    packed.to_csv(archive, index=False, compression={"method": "gzip"})

    restored = (
        pd.read_csv(archive, **READ_KWARGS)
        .drop(columns=[RUN_COLUMN])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(frame, restored, check_exact=True)

"""Pack Member 2's per-run out-of-fold predictions into one versioned file.

The 180 per-run files under ``results/phase2/m2/preds/`` are ~75 MB of CSV and
are regenerated deterministically by the A4/A7/A8 runners, so they are not
tracked (see ``.gitignore``). This script packs them into a single gzipped CSV
that *is* tracked, so Member 5 gets row-level predictions from a clone and the
audit and tables can be rebuilt without re-running 180 training runs.

Values are written at full precision, so ``--restore`` reproduces the per-run
files exactly and ``scripts/verify_m2_phase2.py`` passes on the result.

    python scripts/export_m2_oof_predictions.py            # preds/ -> archive
    python scripts/export_m2_oof_predictions.py --restore   # archive -> preds/
    python scripts/export_m2_oof_predictions.py --check      # verify round-trip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "phase2" / "m2"
PREDS_DIR = RESULTS_DIR / "preds"
ARCHIVE = RESULTS_DIR / "m2_oof_predictions.csv.gz"

PREFIX = "preds_"
SUFFIX = ".csv"
RUN_COLUMN = "run"
GZIP = {"method": "gzip", "compresslevel": 9}

# pandas' default C float parser (xstrtod) is not round-trip exact and shifts the
# last digit of a float64 by one unit in the last place. Every read here is a
# round-trip of values written by the runners, so it must use the exact parser.
READ_KWARGS = {"float_precision": "round_trip"}


def _run_name(path: Path) -> str:
    return path.name[len(PREFIX) : -len(SUFFIX)]


def _per_run_files() -> list[Path]:
    return sorted(PREDS_DIR.glob(f"{PREFIX}*{SUFFIX}"))


def export(archive: Path = ARCHIVE) -> int:
    """Concatenate every per-run prediction file into one gzipped CSV."""
    files = _per_run_files()
    if not files:
        sys.exit(f"no prediction files in {PREDS_DIR}; run the A4/A7/A8 runners first")

    frames = []
    for path in files:
        frame = pd.read_csv(path, **READ_KWARGS)
        frame.insert(0, RUN_COLUMN, _run_name(path))
        frames.append(frame)

    packed = pd.concat(frames, ignore_index=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    packed.to_csv(archive, index=False, compression=GZIP)

    size_mb = archive.stat().st_size / 1e6
    print(
        f"packed {len(files)} runs / {len(packed):,} rows "
        f"-> {archive.relative_to(ROOT)} ({size_mb:.1f} MB)"
    )
    return len(files)


def restore(archive: Path = ARCHIVE) -> int:
    """Split the archive back into one CSV per run."""
    if not archive.exists():
        sys.exit(f"{archive} not found")

    packed = pd.read_csv(archive, **READ_KWARGS)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for run, group in packed.groupby(RUN_COLUMN, sort=True):
        target = PREDS_DIR / f"{PREFIX}{run}{SUFFIX}"
        group.drop(columns=[RUN_COLUMN]).to_csv(target, index=False)
        written += 1

    print(f"restored {written} runs -> {PREDS_DIR.relative_to(ROOT)}")
    return written


def check(archive: Path = ARCHIVE) -> None:
    """Confirm the archive reproduces every per-run file exactly."""
    if not archive.exists():
        sys.exit(f"{archive} not found; run without --restore first")

    packed = pd.read_csv(archive, **READ_KWARGS)
    files = _per_run_files()
    if not files:
        sys.exit(f"no prediction files in {PREDS_DIR} to check against")

    packed_runs = set(packed[RUN_COLUMN].unique())
    disk_runs = {_run_name(path) for path in files}
    if packed_runs != disk_runs:
        sys.exit(
            f"run sets differ: {len(packed_runs - disk_runs)} only in archive, "
            f"{len(disk_runs - packed_runs)} only on disk"
        )

    groups = {run: group for run, group in packed.groupby(RUN_COLUMN, sort=False)}
    for path in files:
        on_disk = pd.read_csv(path, **READ_KWARGS)
        from_archive = (
            groups[_run_name(path)]
            .drop(columns=[RUN_COLUMN])
            .reset_index(drop=True)[on_disk.columns]
        )
        pd.testing.assert_frame_equal(on_disk, from_archive, check_exact=True)

    print(f"OK  archive reproduces all {len(files)} per-run files exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--restore",
        action="store_true",
        help="unpack the archive into per-run prediction files",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="verify the archive matches the per-run files on disk exactly",
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()

    if args.restore:
        restore(args.archive)
    elif args.check:
        check(args.archive)
    else:
        export(args.archive)
        check(args.archive)


if __name__ == "__main__":
    main()

"""Run M3's A1 shared-vs-independent, A2 loss, and A3 floor-head studies."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.dl.m3_experiments import DEFAULT_EXPERIMENTS, EXPERIMENTS, M3RunOptions, run_experiments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-missingness", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "results/phase2/m3"))
    parser.add_argument("--skip-existing", action="store_true", help="Resume from saved fold/seed records")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()
    options = M3RunOptions(epochs=args.epochs, patience=args.patience,
                           batch_size=args.batch_size, device=args.device,
                           seeds=tuple(args.seeds))
    if args.width is None and args.depth is None:
        hidden_dims = (512, 256, 128)
    else:
        width = 512 if args.width is None else args.width
        depth = 3 if args.depth is None else args.depth
        hidden_dims = tuple([width] * depth)
    result = run_experiments(tuple(args.experiments), tuple(args.folds),
                             tuple(args.seeds), options,
                             hidden_dims=hidden_dims, include_missingness=args.include_missingness,
                             output_dir=args.output_dir, skip_existing=args.skip_existing)
    print(result.groupby("exp_id")["evaal_style_score"].agg(["count", "mean", "std"]).to_string())


if __name__ == "__main__":
    main()

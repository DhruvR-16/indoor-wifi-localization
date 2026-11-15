# Phase 2 M3 Handoff

## Scope

M3 owns the shared multitask neural network and the A1, A2, and A3 comparisons. The shared baseline predicts building, floor, and standardized coordinates from a Wi-Fi scan. The M3 runner uses USERID-grouped outer folds, a grouped inner split for epoch/loss-weight selection, training-only coordinate scaling, and a final refit before scoring the held-out outer users.

## Files delivered by M3

- `src/models/dl/multitask.py`: shared and independent-task models; stable `make_model` factory; named outputs; 128-value default embedding through `encode`; flat and building-probability-conditioned floor heads; fixed and learned uncertainty loss weighting; configurable 2/3/4-layer widths for M4's A9 work.
- `src/models/dl/m3_experiments.py`: nested grouped training, inner-only weight selection, outer-fold scoring, checkpoint/prediction/curve writing, per-task metrics, and resumable experiment execution.
- `src/data/phase2_m3.py` and `src/models/dl/m3_support.py`: the M3-specific data/scoring helpers; these keep the reproduced M3 pipeline separate from M1/M2 data code.
- `scripts/run_m3_ablation.py`: command-line entry point for M3's fold/seed experiments.

## Experiment mapping

| M3 experiment | Comparison |
|---|---|
| `a1_shared` | Shared trunk; this is also the flat-floor, equal-loss control for A2 and A3. |
| `a1_independent` | Three independent task trunks with the same per-task width/depth and training budget. |
| `a2_inner_selected_weights` | Selects among fixed loss-weight candidates using inner validation only. |
| `a2_uncertainty_weights` | Learns task weights through trainable log variances. |
| `a3_hierarchical_floor` | Mixes building-specific floor logits using predicted building probabilities. |

The repeated control labels `a2_equal_weights` and `a3_flat_floor` are intentionally omitted from the default run because both are the same model/configuration as `a1_shared`.

## Model and objectives

The shared baseline is `520 → 512 → 256 → 128`; the first two layers use BatchNorm, ReLU, and dropout 0.3, and the final layer uses ReLU. The heads are building `128 → 3`, floor `128 → 5`, and coordinates `128 → 64 → 2`. The 520 inputs can optionally become 1,040 when a missingness channel is requested. The coordinate head predicts training-standardized UTM coordinates, which are inverse-transformed before metre-based scoring.

The equal-weight objective is `L = CE_building + CE_floor + Huber_coordinates`. A2's learned uncertainty objective is `L = Σ[exp(-s_t) L_t + s_t]`, with one learned `s_t` per task. A2's selected fixed weights are chosen by inner-validation EvAAL from `(0.5, 0.5, 2.0)`, `(1, 1, 1)`, and `(1, 1, 2)`. A3 forms floor logits by taking a building-probability-weighted mixture of the three building-conditioned floor heads, so the validation/test floor prediction uses the model's predicted building probabilities.

## Results and run state

M3's confirmatory sweep is complete: five unique configurations, five USERID outer folds, and seeds 42, 1337, and 2024 (75 fold/seed runs total). For each arm and seed, the five outer-fold prediction files cover all 19,300 cleaned training rows exactly once. Coordinate normalization was fitted on the applicable training users only. The official validation file was not used.

Pooled OOF metrics below show the mean ± sample standard deviation across the three seed-level results. They are descriptive; M4 owns paired uncertainty and significance analysis.

| Experiment | EvAAL-style score (m) | Mean position error (m) | Building accuracy | Floor accuracy | Floor accuracy when building is correct |
|---|---:|---:|---:|---:|---:|
| A1 shared multitask (also A2/A3 control) | 21.47 ± 0.86 | 18.67 ± 0.89 | 95.9% ± 1.6% | 81.9% ± 1.0% | 82.5% ± 1.4% |
| A1 independent task networks | **19.29 ± 0.69** | **15.91 ± 0.12** | 94.7% ± 1.1% | **82.2% ± 1.9%** | **84.4% ± 0.9%** |
| A2 inner-selected fixed weights | 20.11 ± 1.34 | 17.67 ± 1.17 | **96.6% ± 1.2%** | 81.2% ± 0.4% | 81.8% ± 0.5% |
| A2 learned uncertainty weights | 22.25 ± 2.40 | 19.66 ± 2.32 | 96.3% ± 1.3% | 81.0% ± 0.8% | 81.4% ± 1.0% |
| A3 building-conditioned floor head | 22.00 ± 0.91 | 19.36 ± 0.64 | 96.2% ± 0.6% | 81.4% ± 0.3% | 81.5% ± 0.3% |

Descriptively, independent task networks have the lowest mean EvAAL score and position error, while the shared model has higher building accuracy. Inner-selected fixed weights improve on the equal-weight control in this run; learned uncertainty weighting and the hierarchical floor head do not improve the headline score. These results do not establish statistical significance; M4 should perform paired fold/user-level analysis before conclusions are finalized.

Artifacts are in `results/phase2/m3/`: `m3_all_runs.csv` has fold/seed metrics; `m3_pooled_oof_by_seed.csv` and `m3_pooled_oof_summary.csv` hold pooled summaries; `m3_oof_predictions.csv.gz` contains all aligned OOF rows (arm, fold, seed, USERID, PHONEID, true/predicted targets); `records/`, `checkpoints/`, `curves/`, and `preds/` retain run-level provenance.

## Reproduce or resume M3

From the project root, run:

```bash
python scripts/run_m3_ablation.py --skip-existing --device auto
```

The default run uses five unique configurations across five USERID folds and three seeds. The shared, equal-weight, flat-floor control is saved once as `a1_shared`, avoiding duplicate identical runs. `--skip-existing` allows interrupted work to resume by fold/seed record.

Each run writes a JSON record, checkpoint with coordinate scaler values, row-level held-out predictions, and inner/outer loss curves. The aggregate CSV is rebuilt from the selected record set when the command finishes.

## Next handoff

M4 can now start tuning and statistical analysis. Use the aligned OOF rows for paired comparisons, cluster-aware uncertainty, and error analysis; leave M3 model and result files unchanged. M5 can integrate the finalized statistical interpretation after M4 reports it.

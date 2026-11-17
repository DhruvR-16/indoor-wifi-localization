# Phase 2 — Member 4: Paired Evaluation and Statistical Analysis

**Status:** Core analysis completed for available M1, M2 A4/A7, and M3 out-of-fold (OOF) results. M2 A8 is excluded because its current outputs are stub-backed and incomplete. Final model/hyperparameter selection and final report integration remain.

**Comparability note:** the M1 and M2 A4/A7 reference pipelines use a `520 → 256 → 128` temporary interface architecture; M3's reference uses a `520 → 512 → 256 → 128` trunk and a nonlinear coordinate head. The table below compares complete saved pipelines descriptively, not architecture-matched variants. M1 also has the inner-fold building-coverage limitation described in its report. Rerunning M1 with M3's final factory and label-covered inner folds is needed before treating the M1-versus-M3 difference as an architecture-controlled result.

## Scope and protocol

M4 compares saved predictions on the same 19,300 cleaned training scans, five USERID-grouped outer folds, and three seeds (42, 1337, 2024). The official validation file is not used. Valid M2 comparison arms are A4 linear-positive, A7 DAE-pretrained, and A7 random initialization. The incomplete A8 hybrid runs are deliberately excluded.

`scripts/analyze_m4_oof.py` checks row alignment across member outputs, then restores M3 truth coordinates from the cleaned source data using `source_row`. This matters because M3's prediction export stores UTM coordinates as float32, which rounds northings at roughly half-metre intervals. M1 and M2 predictions are paired by fold and seed after their USERID, PHONEID, building, floor, and coordinate ordering is checked.

Uncertainty intervals are paired percentile bootstrap intervals over the 18 USERID clusters (2,000 resamples, fixed seed), with each user's mean score averaged across their scans and the three seeds. This makes the unit of resampling the independent user group rather than treating repeated scans as independent. The exact McNemar test is reported at scan level for seed 42 as a conventional diagnostic; its p-values can be optimistic because they do not model within-user dependence. Interpret those p-values together with the USERID-cluster bootstrap intervals. No multiple-comparison correction is applied, so comparisons are exploratory.

## Current OOF comparison

Means below are pooled across all scans and seeds after restoring original coordinate precision. EvAAL uses `position error + 50 m if building is wrong + 4 m if floor is wrong`.

| Model | EvAAL (m) | Mean position error (m) | Floor accuracy | Building accuracy |
|---|---:|---:|---:|---:|
| M3 independent task networks | **19.29** | **15.91** | 82.18% | 94.66% |
| M2 A7 DAE-pretrained | 19.76 | 17.42 | 82.13% | 96.70% |
| M3 inner-selected fixed loss weights | 20.11 | 17.66 | 81.23% | 96.61% |
| M1 fixed baseline | 21.41 | 19.28 | 81.90% | 97.18% |
| M3 shared multitask | 21.47 | 18.67 | 81.86% | 95.85% |
| M2 A4 linear-positive kNN | 21.95 | 19.40 | 82.50% | 96.30% |
| M2 A7 random initialization | 21.95 | 19.40 | 82.50% | 96.30% |
| M3 hierarchical floor head | 22.00 | 19.36 | 81.42% | 96.20% |
| M3 learned uncertainty weights | 22.25 | 19.66 | 81.03% | 96.34% |

The M3 independent model is descriptively best on EvAAL and mean coordinate error. Its paired USERID-bootstrap EvAAL difference versus the shared M3 control is −1.11 m (95% CI −2.76 to 0.38), which includes zero. Versus M1 baseline, its EvAAL difference is −1.62 m (−3.77 to 0.41), also including zero; the mean position-error difference is −2.80 m (−4.36 to −1.53). These intervals do not support claiming a statistically reliable EvAAL improvement. M1's report documents an inner-split label-coverage limitation on folds 0 and 1, which should also be considered in that comparison.

For M2 A7, DAE pretraining versus random initialization gives a paired user-mean EvAAL difference of −1.32 m (95% CI −2.48 to −0.39) and mean position-error difference of −1.26 m (−2.37 to −0.26). Floor accuracy is lower by 1.1 percentage points (−2.0 to −0.3 points). Treat this as exploratory because the intervals are unadjusted for the several model comparisons.

The seed-42 scan-level McNemar comparison for independent versus shared M3 models has 637 scans correct only under the independent model and 666 correct only under the shared model (exact p=0.438). The p-value is nominal and scan-level; the clustered interval is the primary uncertainty summary.

## Floor calibration

M4 obtained floor probabilities by running each saved M3 fold checkpoint on that fold's held-out rows. The reported ECE is top-label expected calibration error using 15 equal-width confidence bins; Brier and negative log-likelihood provide complementary proper scoring rules.

| M3 arm | Floor ECE | Multiclass Brier | Negative log-likelihood |
|---|---:|---:|---:|
| Shared | 0.139 | 0.322 | 1.261 |
| Independent | 0.139 | 0.319 | 1.503 |
| Inner-selected fixed weights | 0.140 | 0.325 | 1.110 |
| Learned uncertainty weights | 0.146 | 0.333 | 1.253 |
| Hierarchical floor head | 0.145 | 0.330 | 1.295 |

The independent model's higher floor accuracy does not imply better probability calibration: its NLL is higher than the shared model's. Avoid treating ECE alone as decisive; it depends on binning, and results are summarized across three fitted seeds.

## Learning curve

`scripts/run_m4_learning_curve.py` retrains the M3 independent-task architecture at 25%, 50%, 75%, and 100% of each outer-training fold, with training rows stratified over building-floor cells. Every held-out outer USERID remains untouched. The selected epoch count from the matching M3 inner-validation run is frozen at all data fractions; therefore this measures a training-data-volume effect under a fixed training schedule, not the best achievable tuning at each fraction. Mean and standard deviation are across 15 fold/seed runs.

| Training fraction | Mean training rows | EvAAL (m), mean ± SD | Position error (m), mean ± SD | Floor accuracy, mean ± SD |
|---:|---:|---:|---:|---:|
| 25% | 3,860 | 26.41 ± 11.71 | 24.24 ± 12.14 | 78.85% ± 12.81% |
| 50% | 7,720 | 21.01 ± 6.54 | 17.98 ± 5.13 | 81.57% ± 10.43% |
| 75% | 11,580 | 18.96 ± 6.49 | 16.24 ± 4.47 | 80.59% ± 12.06% |
| 100% | 15,440 | 19.29 ± 6.31 | 15.84 ± 2.63 | 81.82% ± 12.60% |

The broad trend is lower localization error with more training rows, while the 75% and 100% results are not strictly monotonic. At 100%, all 15 fold/seed EvAAL and position-error records match the saved M3 independent model, providing a consistency check on the refit implementation.

## Error review and output files

The worst-error export contains the 100 largest position errors for each of the three independent-model seeds (300 rows total), with user, phone, fold, true/predicted labels, coordinates, and EvAAL score. It is a review queue; it should not be used to selectively remove difficult samples.

Generated M4 outputs are under `results/phase2/m4/`:

- `m4_seed_metrics.csv` — pooled metrics by model and seed.
- `m4_per_user_metrics.csv` — per-user averages across rows and seeds.
- `m4_paired_user_bootstrap.csv` — paired 95% user-cluster bootstrap intervals for all model pairs and metrics.
- `m4_floor_mcnemar_seed42.csv` — exact scan-level McNemar diagnostics, with dependence caveat.
- `m4_floor_calibration.csv` — held-out M3 floor ECE, Brier, NLL, and accuracy by seed.
- `m4_learning_curve_runs.csv` and `m4_learning_curve_summary.csv` — 60 fold/seed/fraction refits and aggregate curve.
- `m4_worst_100_scans_per_seed.csv` — worst localization rows for error review.

Reproduce the analyses from the repository root with:

```bash
python scripts/analyze_m4_oof.py --bootstrap 2000
python scripts/run_m4_learning_curve.py --device auto
```

## Remaining M4 work and limitations

- Incorporate M2's A8 hybrid only after M2 replaces the stub, implements all required candidates, and produces valid grouped OOF predictions.
- Agree with the team on the final selection metric and candidate before any official-file evaluation; this analysis uses grouped OOF only.
- Treat bootstrap and McNemar findings as exploratory until the team settles its planned comparison family and multiple-testing policy.
- Preserve M1's inner-split caveat and M2 A8's incomplete status in final conclusions.
- M5 has integrated the available A1–A7 evidence and completed the A9 width/depth sweep in `docs/phase2_integrated_report.tex` (compiled PDF: `output/pdf/phase2_integrated_report.pdf`). Extend the paired user-cluster analysis to A9 before making inferential claims about that sweep.

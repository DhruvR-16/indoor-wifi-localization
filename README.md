# Indoor Wi-Fi Localization

An offline machine-learning project that predicts a scan's building, floor, and indoor coordinates from Wi-Fi access-point signal fingerprints. Phase 1 establishes the classical ML baseline on UJIIndoorLoc. Phase 2 evaluates deep-learning and hybrid approaches against the same user-held-out baseline.

Both phases currently use saved dataset files. The project does not perform live Wi-Fi scanning, connect to a live database, or provide a deployed mobile application.

## Project status

- **Phase 1:** implementation and evaluation artifacts are available: data-quality and EDA results, feature engineering, grouped-CV classifiers and regressors, result tables, figures, and an IEEE-style report.
- **Phase 2:** M1 robustness/regularization, M2 A4/A7, M3 multitask experiments, M4 OOF analysis, and M5's A9 sweep/integrated report are available. Phase 2 is **not complete**: M1's initial A5/A6 runs use the temporary 520→256→128 stub and have an inner-split limitation; M2's A8 hybrid remains incomplete and its stub-backed outputs are not reportable; M4 still needs A9 paired uncertainty and final candidate selection. See the member reports for caveats. Planned improvements are not results.

The consolidated Phase 1 + Phase 2 project report is [`docs/project_report.md`](docs/project_report.md). It presents Phase 2 as current progress and records unfinished work; `main.pdf` remains the Phase 1-only IEEE report.

## Dataset

The project uses the UJIIndoorLoc dataset from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5MS59), licensed under CC BY 4.0. The source files contain 21,048 scans in total:

| File | Scans | Use |
|---|---:|---|
| `trainingData.csv` | 19,937 | Development, grouped cross-validation, and model fitting. |
| `validationData.csv` | 1,111 | Official held-out file. Earlier fixed-setting Phase 1 predictions exist; see the evaluation note below. |

Each source row has 529 columns: 520 WAP signal readings, four targets (`BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE`), and five metadata fields (`SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`). The labels cover three buildings and 13 building-floor combinations. The two CSV files occupy about 43 MB.

Raw data is downloaded locally and excluded from Git. To download it, use the setup steps below. The loader removes exact duplicate rows from training only, identifies constant WAP columns using training only, and drops those columns from both partitions. It flags observed RSSI values in `[-5, 0]` dBm for review but retains them. The current recorded quality checks found 637 duplicate rows, 55 constant WAP columns, and 224 flagged near-zero readings. The cleaned training set has 19,300 rows.

## Setup

Use Python 3.11 and the pinned dependencies in `requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_splits.py
python scripts/export_quality_report.py
```

The download script saves the source CSVs under `data/raw/`. `build_splits.py` creates five deterministic `GroupKFold` splits grouped by `USERID`. Rows for a user stay together, and every cleaned training row appears in one outer validation fold. The 1,111-row official file is not used to fit the folds.

## Phase 1 workflow

The pipeline follows this order:

1. **Explore and audit the data.** EDA examines signal coverage and RSSI, building and floor support, scan counts by user and phone, coordinate maps, and train-versus-official-file signal summaries. It covers signal data and relevant labels and metadata; it is not an exhaustive analysis of every metadata field.
2. **Clean and split.** Remove duplicate training rows, drop training-constant WAPs from both partitions, retain flagged near-zero readings, and make USERID-grouped folds.
3. **Build fold-safe features.** The source uses `+100` to mean “not detected.” The feature pipeline maps it to `-110` dBm for numeric signal views and adds missingness indicators. It compares linear-positive, exponential, and power-based RSSI views, plus per-scan signal summaries. Scaling, PCA, and variance filtering are fitted on the training part of each fold and then applied to that fold's held-out users.
4. **Classify and localize.** M3 compares Random Forest and XGBoost for building/floor labels. M4 compares kNN, Random Forest, and XGBoost for coordinates with global, per-building, and per-building-floor routing. M4 uses out-of-fold M3 predictions for realistic end-to-end routing; oracle routing with true labels is a diagnostic only.
5. **Evaluate consistently.** M5 combines saved predictions and applies shared classification, coordinate, hierarchical, and EvAAL-style metrics. Pooled out-of-fold results are kept distinct from the earlier official-file baseline.

The feature pipeline used by the Phase 1 classifiers has 941 inputs: 465 transformed WAP values, 465 WAP absence indicators, and 11 scan-level summary features. The pipeline and data loader are in `src/features/transform.py` and `src/data/load.py`.

## Phase 1 results

The headline figures below are pooled out-of-fold results from five USERID-grouped folds. The task-best classification figures come from different classifier choices, so the model name belongs with each result.

| Task | Result | Evaluation details |
|---|---:|---|
| Building classification | 99.34% accuracy | Task-best result with Random Forest. |
| Floor classification | 77.10% accuracy | Task-best result with XGBoost. Floor performance is uneven; the least-supported floor remains difficult. |
| Coordinate prediction | 12.21 m mean position error | Tuned per-building kNN using the predicted building; median 7.52 m and 95th-percentile error 37.11 m. |

Other useful context: the training scans contain a mean of 17.88 detected APs per scan versus 15.59 in the official file. The tuned per-building kNN result improves on the global kNN route (about 21.01 m mean error). The per-building-floor route did not improve on per-building routing in the primary comparison.

### Official-file evaluation note

The official 1,111-row file was evaluated earlier with fixed model settings. Those saved predictions are not a fresh evaluation of the corrected tuned pipeline and must not be presented as tuned-model test results or used to choose tuned models. Use the nested USERID-grouped cross-validation results for Phase 1 model comparisons. Preserve this limitation in presentations and reports. For Phase 2, select models using grouped validation and disclose that the official file has already been evaluated by the earlier baseline; do not repeatedly inspect it during tuning.

## Reproducing Phase 1 experiments

After setup and split generation, the main commands are:

```bash
# M2: grouped kNN comparison of RSSI representations and feature reduction
python scripts/run_knn_baselines.py
python scripts/report_feature_reduction.py

# M3: fixed comparison, then nested USERID-grouped tuning
python scripts/train_classifiers.py
python scripts/train_tuned_classifiers.py

# M4: tuned end-to-end coordinate routes using M3's tuned OOF predictions
python scripts/train_regressors.py \
  --upstream-predictions classifiers_tuned_grouped_cv.csv \
  --upstream-features tuned_variance_filtered \
  --output-suffix tuned

# M5: rebuild summary tables after the prediction CSVs have been generated
python scripts/build_results_table.py

# Run project tests
pytest
```

Model scripts create row-level predictions locally under `results/preds/`. Summary and fold-level tables are stored under `results/tables/`. Do not run `scripts/evaluate_official_test.py` or otherwise evaluate the official file during model selection.

## Phase 1 ownership

| Member | Phase 1 responsibility | Main handoff |
|---|---|---|
| M1 | Data setup, quality checks, EDA, and USERID-grouped folds. | Cleaned data and fold definitions. |
| M2 | Missing-signal handling, RSSI representations, engineered features, fold-safe transforms, and kNN feature baselines. | Shared feature builder and feature comparison. |
| M3 | Building/floor classification with Random Forest and XGBoost. | Out-of-fold building/floor predictions for M4 and M5. |
| M4 | Coordinate regression, routing, and end-to-end localization evaluation. | Coordinate predictions, route comparisons, and error analysis. |
| M5 | Shared metrics, result integration, and report-ready analysis. | Pooled and fold-level comparison tables. |

Use the team's agreed roster to map people to M1-M5 in the final contribution statement; the repository role labels alone do not establish who completed each task.

## Phase 2 plan

Phase 2 will test whether learned Wi-Fi representations and joint or hybrid models improve on the Phase 1 classical baseline. Every experiment must reuse the same cleaned data convention and USERID-grouped folds. Tune on grouped training folds, compare models on identical held-out users, and report negative and neutral findings as well as gains.

| Member | Planned work | Evidence and output |
|---|---|---|
| M1 — data and robustness | Initial fold-safe pipeline and A5/A6 sweep completed on the temporary 520→256→128 stub. The plain inner GroupKFold misses building coverage on two outer folds. Rerun with M3's final factory and label-covered inner folds before claiming transfer to the final architecture. | `docs/phase2_m1_report.md`; initial saved run records and predictions. |
| M2 — representations and hybrid method | A4/A7 representation and pretraining work is complete. A8 remains unfinished: connect the real M3 encoder and complete the missing hybrid candidates; current stub-backed A8 results are not reportable. | `docs/phase2_m2_report.md`; finish A8 before making hybrid claims. |
| M3 — multitask model | Completed: shared/independent task networks, fixed and learned loss weighting, hierarchical floor head, and five USERID-fold OOF experiments across three seeds. | `docs/phase2_m3_handoff.md`, model code, run records, checkpoints, and pooled OOF predictions. |
| M4 — tuning and statistical validation | Core paired analysis is available for M1 baseline, valid M2 A4/A7 arms, and M3; M2 A8 is excluded pending its completion. Bootstrap intervals resample USERID groups; learning-curve epochs are fixed from the matching M3 inner selection. Final candidate selection and interpretation remain. | `docs/phase2_m4_report.md`, `scripts/analyze_m4_oof.py`, `scripts/run_m4_learning_curve.py`, and `results/phase2/m4/`. |
| M5 — ablations and integration | A1-A7 results integrated; A9 width/depth sweep run across five folds and three seeds. A8 remains blocked on valid M2 results. | `docs/phase2_integrated_report.md`; `scripts/run_m5_a9.py`; tables and figures under `results/phase2/m5/` and `results/figures/`. |

### Phase 2 evaluation rules

- Keep PyTorch as the deep-learning framework and reuse Phase 1's preprocessing rules and saved folds.
- Fit every scaler, feature transform, feature selector, and target scaler only on the training partition for that fold.
- Use grouped validation for architecture and hyperparameter selection. Freeze the model before any final evaluation on the official file.
- The official file has already been evaluated for the Phase 1 fixed-setting baseline. Report that history and avoid treating it as a completely untouched test set for Phase 2.
- Compare models on the same folds, seeds, targets, and metrics. Report all planned ablations, including results that do not improve on Phase 1.

Remaining Phase 2 deliverables are the architecture-matched M1 rerun, M2's completed and fairly evaluated A8 hybrid, M4's paired uncertainty analysis extended to A9 and final candidate selection, an updated integrated report, and the reproducible demo workflow. The current integrated report and figures summarize available work and identify these pending items.

## Repository layout

```text
configs/                 Phase 1 feature, classifier, and regressor settings
data/raw/                Downloaded UJIIndoorLoc CSVs (ignored by Git)
data/splits/folds.json   Saved USERID-grouped split indices
notebooks/               EDA, feature, classification, and regression notebooks
src/data/                Data loading and quality checks
src/features/            RSSI and fold-safe feature transformations
src/models/ml/           Phase 1 classifier and regressor definitions
src/evaluation/          Shared evaluation metrics
scripts/                 Data, split, model, plotting, and evaluation commands
results/figures/         EDA, model, and routing plots
results/tables/          Quality, grouped-CV, classifier, and regressor summaries
main.tex                 IEEE-style Phase 1 report source
main.pdf                 Compiled Phase 1 report
```

## Data source

UJIIndoorLoc, UCI Machine Learning Repository, DOI [10.24432/C5MS59](https://doi.org/10.24432/C5MS59). Cite the dataset and follow its CC BY 4.0 license terms in submitted work.

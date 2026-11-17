# Indoor Wi-Fi Localization

An offline machine-learning project that predicts a scan's building, floor, and indoor coordinates from Wi-Fi access-point signal fingerprints. Phase 1 establishes the classical ML baseline on UJIIndoorLoc. Phase 2 evaluates deep-learning and hybrid approaches against the same user-held-out baseline.

Both phases currently use saved dataset files. The project does not perform live Wi-Fi scanning, connect to a live database, or provide a deployed mobile application.

## Project status

- **Phase 1:** implementation and evaluation artifacts are available: data-quality and EDA results, feature engineering, grouped-CV classifiers and regressors, result tables, figures, and an IEEE-style report.
- **Phase 2:** grouped experiments evaluate RSSI representations, missingness masks, denoising pretraining, neural task architectures, regularization, data volume, and network capacity. The reports present measured results and their methodological limits.

The consolidated Phase 1 + Phase 2 [project report PDF](output/pdf/project_report.pdf) and [LaTeX source](docs/project_report.tex) document the full evaluated workflow. The separate [Phase 2 report PDF](output/pdf/phase2_integrated_report.pdf) and [LaTeX source](docs/phase2_integrated_report.tex) describe the neural experiments and statistical interpretation. `main.pdf` remains the Phase 1-only IEEE report.

To rebuild either report, install [Tectonic](https://tectonic-typesetting.github.io/) and run `tectonic docs/project_report.tex --outdir output/pdf` or `tectonic docs/phase2_integrated_report.tex --outdir output/pdf` from the repository root.

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

## Phase 2 evaluation

Phase 2 uses the cleaned 19,300 training scans, five `USERID`-grouped outer folds, and three random seeds. It preserves all 520 WAP positions for neural inputs and evaluates these measured comparisons:

- Linear, exponential, and power RSSI views with and without an AP-detection mask.
- Denoising-autoencoder pretraining against random initialization.
- Shared and independent task networks, loss weighting, and building-conditioned floor prediction.
- Regularization and signal perturbation within a reference network.
- Training-data learning curves and network width/depth changes.
- Paired user-cluster uncertainty and floor-probability calibration for the evaluated models.

The [Phase 2 report](output/pdf/phase2_integrated_report.pdf) gives methods, numerical results, figures, and the limits of cross-model comparisons. Its figures can be regenerated with `python scripts/build_publication_figures.py` from the saved result tables. The official validation file is not used for Phase 2 model selection or the results in that report.

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
results/figures/         EDA, model, routing, and report figures
output/pdf/               Compiled complete-project and Phase 2 reports
results/tables/          Quality, grouped-CV, classifier, and regressor summaries
main.tex                 IEEE-style Phase 1 report source
main.pdf                 Compiled Phase 1 report
```

## Data source

UJIIndoorLoc, UCI Machine Learning Repository, DOI [10.24432/C5MS59](https://doi.org/10.24432/C5MS59). Cite the dataset and follow its CC BY 4.0 license terms in submitted work.

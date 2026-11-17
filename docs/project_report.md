# Indoor Wi-Fi Localization
## Project Report — Phase 1 Baseline and Phase 2 Progress

**Project:** Project 9 — Applied Machine Learning

**Team:** Kavya Jain, Naman Jain, Dhruv Ramani, Lakshya Agrawal, Akhilesh Kumar
**Report status:** Consolidated report of completed Phase 1 and available Phase 2 work. Phase 2 deliverables listed in Section 10 remain incomplete; this report does not present them as completed.

---

## Abstract

Indoor Wi-Fi fingerprinting estimates a device's location from the received signal strength (RSSI) pattern of nearby access points. This project predicts building, floor, and two-dimensional coordinates from UJIIndoorLoc. Phase 1 establishes a classical machine-learning baseline using data-quality analysis, fold-safe Wi-Fi representations, Random Forest and XGBoost classification, and routed coordinate regression. The cleaned training set contains 19,300 scans after duplicate removal. Under five USERID-grouped outer folds, tuned Random Forest achieved 99.34% building accuracy, tuned XGBoost achieved 77.10% floor accuracy, and tuned per-building k-nearest neighbors (kNN) achieved 12.21 m mean position error.

Phase 2 investigates robustness, RSSI representations, denoising pretraining, multitask neural networks, statistical evaluation, and architecture width/depth. The available M3 experiments evaluated five model configurations over five user folds and three seeds (75 runs); M4 produced paired out-of-fold analysis and a learning curve; M5 evaluated four additional architecture variants. The independent-task M3 model has the lowest observed mean EvAAL-style score among completed valid arms (19.29 m), but its paired interval versus the shared M3 model includes zero. M2's A8 hybrid is not complete, M1's initial A5/A6 sweep uses a temporary model and has an inner-split limitation, and paired A9 uncertainty and final candidate selection are pending. Consequently, the report presents Phase 2 as current experimental progress rather than a completed final comparison.

The project is an offline research pipeline using saved CSV scans. It does not currently scan live Wi-Fi, deploy a mobile application, or connect to a live database.

## 1. Problem statement and objectives

Satellite navigation is often unavailable or inaccurate indoors. Wi-Fi fingerprinting offers an alternative: record signal strengths at known places, then learn how a new signal pattern corresponds to a building, floor, or coordinate. The localization problem is hierarchical and multi-output:

1. Predict one of three buildings.
2. Predict one of five encoded floor labels (the benchmark contains 13 building-floor combinations).
3. Predict projected horizontal coordinates and report errors in metres.

The project objectives are to establish a reproducible Phase 1 baseline; compare model and feature choices without leaking user information across folds; explore deep and hybrid models in Phase 2; report uncertainty, negative results, and limitations; and preserve a workflow that can be rerun from the repository.

## 2. Dataset and exploratory analysis

### 2.1 Dataset contents

The UJIIndoorLoc dataset contains 19,937 training scans and 1,111 official validation scans (21,048 total). Each row has 529 columns: 520 WAP signal readings, four targets (`BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE`), and five collection metadata fields (`SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`). The locations cover three buildings and 13 building-floor combinations [1].

### 2.2 EDA and data quality

EDA examined signal coverage and observed RSSI, scans per building/floor/user/phone, coordinate maps, class support, and descriptive training-versus-official-file signal summaries. It was performed on signal data and relevant labels/metadata, not as an exhaustive analysis of every metadata field. Important findings:

| Check | Result and treatment |
|---|---|
| Exact duplicate training rows | 637 removed, retaining the first copy. |
| WAP columns constant in training | 55 of 520; detected from training only and removed from both data partitions. |
| Cleaned training scans | 19,300. |
| Observed RSSI in `[-5, 0] dBm` | 224 readings flagged for review and retained. |
| Mean detected APs per scan | 17.88 in training; 15.59 in the official file. |
| Mean observed RSSI | −78.59 dBm in training; −77.45 dBm in the official file. |

The train/test differences are descriptive only; they were not used to fit features, tune models, or select hyperparameters. The source sentinel `+100` means that a WAP was not detected. It is not a physical RSSI measurement and is treated separately from observed signals.

## 3. Evaluation design and metrics

### 3.1 User-grouped validation

The cleaned training data is partitioned into five deterministic outer `GroupKFold` folds grouped by `USERID`. All rows from one user remain in one partition. Across the five folds, each cleaned scan is a held-out scan exactly once. Grouping by person tests generalization to unseen users and avoids the optimistic results that can arise when scans from the same user's route appear in both training and validation.

All transformations that estimate parameters—including scaling, PCA, variance selection, and coordinate target scaling—are fitted on the training side of a fold and then applied unchanged to its held-out users. Model settings and stopping times are selected within outer-training users where the experiment uses nested selection. Out-of-fold (OOF) means a scan's prediction comes from a model that did not train on that scan or its user.

### 3.2 Metrics

- **Accuracy:** proportion of exact class predictions that are correct.
- **Macro-F1:** F1 computed per class and averaged equally, so minority classes contribute more than in raw accuracy alone.
- **Position error:** Euclidean distance between true and predicted coordinates after inverse scaling, in metres. Mean, median, and 95th percentile are reported to describe average, typical, and tail performance.
- **EvAAL-style score:** `position error + 50 m × wrong-building indicator + 4 m × wrong-floor indicator`. Lower is better. It encodes that a wrong building is costlier than a wrong floor; it is a designed penalty score, not physical distance. Classification and unpenalized coordinate metrics are reported beside it.
- **Calibration:** M4 reports floor expected calibration error (ECE), Brier score, and negative log-likelihood (NLL). ECE depends on binning and should not be used alone.
- **Uncertainty:** M4's paired percentile bootstrap resamples USERID clusters so multiple scans from one person are not treated as independent. Current multi-model findings remain exploratory because comparisons are unadjusted for multiple testing.

### 3.3 Official validation file

The official 1,111-row file was evaluated earlier with fixed Phase 1 model settings. It is not a fresh final test set for the tuned Phase 1 or Phase 2 comparisons and was not used for Phase 2 selection. The primary evidence in this report is grouped cross-validation on the training scans. Any future official-file result must disclose this earlier use.

## 4. Phase 1: classical machine-learning baseline

### 4.1 Signal features

The 55 constant WAPs are dropped, leaving 465 signal channels. Phase 1 maps a missing WAP to `−110 dBm`, uses an explicit absence indicator, and compares three signal views:

- Linear-positive: `x = (r + 110) / 110`.
- Exponential: `x = exp(r / 24)` for observed RSSI `r`.
- Power view: normalized positive signal raised to `e`.

Missing values map to zero in each view. Eleven per-scan summaries describe detected AP count, strongest signal and its AP, average/spread of observed signals, and top AP information. Thus the main classifier representation contains 465 transformed readings + 465 missingness indicators + 11 summaries = 941 inputs. PCA and variance filtering are performed inside the training fold; the transform is then reused on held-out users. The representation choices are motivated by Wi-Fi fingerprint similarity research [2, 3].

### 4.2 Building and floor classification

Random Forest and XGBoost classifiers model nonlinear interactions among signal features. Tuning uses nested USERID-grouped splits. Accuracy is interpreted alongside macro-F1 and confusion matrices because class support is imbalanced, particularly for floor 4.

| Task/model | Pooled OOF result |
|---|---:|
| Building — tuned Random Forest | 99.34% accuracy |
| Floor — tuned XGBoost | 77.10% accuracy; outer-fold macro-F1 0.743 |

The best building and floor results come from different classifier choices. The model name must accompany each number. Floor 4 remains difficult due to limited support.

### 4.3 Coordinate regression and routing

The coordinates are predicted with kNN, Random Forest, or XGBoost regressors. Routes are global, per-building, or per-building-floor. End-to-end evaluation uses predicted building/floor labels; a route based on true labels is only a diagnostic because the true labels are unknown at inference. Coordinate scaling is trained on each fold's training users and predictions are converted back to metres before scoring.

| Coordinate route | Mean error | Median | 95th percentile | EvAAL-style |
|---|---:|---:|---:|---:|
| Tuned per-building kNN using predicted building | **12.21 m** | 7.52 m | 37.11 m | 13.55 m |

The global kNN route has about 21.01 m mean error. Restricting the local estimator to the predicted building is beneficial in this dataset; per-building-floor routing was not better than per-building routing in the primary comparison. Better floor classification does not automatically imply lower coordinate error.

### 4.4 Phase 1 member responsibilities

| Member | Phase 1 role |
|---|---|
| M1 | Data setup, quality checks, EDA, and USERID-grouped folds. |
| M2 | Missing-signal handling, RSSI representations, engineered/fold-safe features, PCA/variance and kNN feature comparisons. |
| M3 | Building/floor classification with Random Forest and XGBoost. |
| M4 | Coordinate regression, routing, and end-to-end localization evaluation. |
| M5 | Shared metrics, pooled/fold results integration, and report-ready analysis. |

## 5. Phase 2: deep-learning and hybrid investigation

### 5.1 Research questions and responsibilities

Phase 2 builds on the Phase 1 fingerprint pipeline and asks whether learned representations and neural/hybrid models improve held-out-user localization. The experiments address robustness to signal variation, nonlinear RSSI representations, missingness, unsupervised pretraining, shared versus independent multitask learning, task-loss balance, floor hierarchy, statistical uncertainty, and architecture capacity. Wi-Fi fingerprinting and learned signal representations are motivated by the surveys in [3-5]; those sources provide background, while the grouped-fold artifacts below provide evidence for our specific findings.

| Member | Phase 2 question and responsibility | Main deliverable/status |
|---|---|---|
| M1 | Establish the 520-WAP input/data contract; test regularization and signal perturbations (A5/A6). | Initial 150-run sweep exists; model and inner-split caveats require a final rerun. |
| M2 | Compare RSSI views/missingness (A4), denoising pretraining (A7), and a learned-embedding hybrid (A8). | A4/A7 completed (120 reportable runs); A8 incomplete and excluded. |
| M3 | Compare multitask sharing, loss weighting, and hierarchical floors (A1-A3). | Five unique architectures/configurations, 75 fold/seed runs. |
| M4 | Compare OOF predictions, estimate uncertainty, inspect calibration and errors, measure data-size effects. | Core analysis available; A9 paired uncertainty/final selection pending. |
| M5 | Test network width/depth (A9) and integrate reproducible tables and figures. | Four new architectures and integrated evidence; A9 inference still pending. |

### 5.2 Shared data contract and evaluation protocol

Phase 2 retains the complete 520 ordered WAP columns. Phase 1 removes 55 WAPs constant in the complete training partition; Phase 2 retains 520 positions so each fold and model has a stable input dimension even when an AP happens to be constant in one subset. For detected RSSI `r`, linear-positive encoding is `(clip(r, -110, 0) + 110) / 110`. The source sentinel `+100` maps to exactly zero, while detected readings map to a positive value. In a missingness-mask arm, a 520-bit detection vector is concatenated with the signal vector for 1,040 inputs. Transform/scaler parameters and coordinate scaling are fitted using training users only.

The five outer folds reuse `GroupKFold` grouped by `USERID`; seeds are 42, 1337, and 2024. For each fold, the outer validation users are not used for fitting, augmentation, hyperparameter selection, or stopping selection. Where the method uses nested selection, a grouped inner split selects epoch or fixed loss weights. The selected model is then refit on the full outer-training users, and the held-out outer users are scored. OOF predictions record one held-out prediction for every cleaned scan per complete arm and seed. M2's `verify_m2_phase2.py` checks this protocol and the saved predictions; its current audit reports 47 checks passed and zero failed. Phase 2 avoids the official 1,111-row file because it was evaluated previously for the fixed Phase 1 baseline.

The primary Phase 2 metric is the EvAAL-style score from Section 3.2. Because it adds a 50 m building penalty and a 4 m floor penalty, it weights building correctness more heavily. We therefore report building/floor accuracy and unpenalized Euclidean position error alongside it. Across-run mean +/- standard deviation, pooled scan OOF results, and paired group-level confidence intervals answer different questions and are labeled separately below.

### 5.3 M1 — input contract, robustness, and regularization (A5/A6)

M1's data helpers and trainer are in `src/data/phase2.py` and `src/training/phase2_trainer.py`. `scripts/run_m1_experiments.py` runs the initial baseline plus nine A5/A6 alternatives. The saved reference uses dropout 0.2, batch normalization enabled, no weight decay, Adam with learning rate `1e-3`, batch size 64, maximum 20 epochs, patience 6, and equal weights for the three task losses. Each of the ten configurations has five outer folds and three seeds, giving 15 run summaries per configuration and 150 total runs.

A5 varies regularization: dropout 0.0, dropout 0.5, weight decay `1e-4`, and batch normalization off. A6 varies signal corruption: Gaussian noise with standard deviation 2 or 4 dBm on detected readings, AP dropout at 5% or 10%, and a combined 2 dBm noise + 5% dropout arm. Augmentation is applied to training inputs; missing entries remain missing. The goal is to test whether modest measurement noise or transient AP loss improves robustness, not to claim that synthetic corruption recreates a building remodel or a changed AP installation.

| Configuration | Building acc. (%) | Floor acc. (%) | Mean position error (m) | EvAAL-style score (m) |
|---|---:|---:|---:|---:|
| Baseline | 97.16 +/- 3.62 | 81.60 +/- 10.41 | 19.09 +/- 4.20 | 21.24 +/- 5.49 |
| Dropout 0.0 | 95.29 +/- 7.04 | 79.40 +/- 11.48 | 18.89 +/- 4.82 | 22.06 +/- 7.35 |
| Dropout 0.5 | 95.84 +/- 6.41 | 82.07 +/- 9.47 | 26.90 +/- 8.32 | 29.70 +/- 8.39 |
| Weight decay `1e-4` | 96.24 +/- 5.44 | 83.49 +/- 8.25 | **17.89 +/- 3.68** | **20.43 +/- 5.71** |
| Batch normalization off | 96.55 +/- 3.50 | 81.77 +/- 13.57 | 18.68 +/- 4.07 | 21.13 +/- 4.34 |
| Noise sigma 2 dBm | 96.73 +/- 4.46 | 81.13 +/- 10.59 | 18.72 +/- 4.61 | 21.11 +/- 5.85 |
| Noise sigma 4 dBm | 96.00 +/- 5.40 | 80.20 +/- 12.31 | 18.20 +/- 3.97 | 20.99 +/- 5.57 |
| AP dropout 5% | 95.14 +/- 6.52 | 81.02 +/- 10.28 | 18.46 +/- 3.36 | 21.65 +/- 6.06 |
| AP dropout 10% | 95.51 +/- 5.95 | 80.30 +/- 12.10 | 18.48 +/- 4.13 | 21.52 +/- 6.00 |
| Noise 2 dBm + AP dropout 5% | 95.58 +/- 5.68 | 81.30 +/- 10.39 | 18.68 +/- 3.86 | 21.64 +/- 6.18 |

These are mean +/- standard deviation across 15 outer fold/seed run summaries. Weight decay is the strongest initial M1 setting in this table; the improvement over baseline is modest relative to run-to-run spread. Dropout 0.5 notably harms coordinate regression. Noise at 4 dBm gives a small reduction in mean position error relative to baseline, but the combined augmentation does not beat weight decay. These are within-sweep comparisons on the same initial model.

**Important M1 limitations:** an audit found that the trainer imports `src/models/stub_multitask.py`. Its 150 runs use a `520 → 256 → 128` shared trunk and a linear coordinate head, whereas M3's final model has a `520 → 512 → 256 → 128` trunk and `128 → 64 → 2` coordinate head. Also, plain inner `GroupKFold` can leave building 0 out of inner training on outer folds 0 and 1 because only users 1 and 11 represent that building. The outer users remain held out, but the epoch selection for those two folds is not label-coverage-safe. Thus the initial sweep may be reported as a completed experiment on the temporary architecture, but its conclusions are provisional for M3's final model. An architecture-matched rerun with label-covered inner validation remains required.

Artifacts: `docs/phase2_m1_report.md`; `results/phase2/m1/` run JSON, predictions, and summaries; `results/figures/m1_*.png`; `src/data/phase2.py`; `src/training/phase2_trainer.py`; `scripts/run_m1_experiments.py`.

### 5.4 M2 — input representations (A4), DAE pretraining (A7), and hybrid (A8)

#### A4: RSSI transformations and missingness flags

M2 compares the impact of signal geometry with a matched `520 → 256 → 128` reference network. For an observed reading `r`, let `P = clip(r, -110, 0) + 110` and `R=110`. The views are linear-positive `P/R`, exponential `exp(P/alpha)/exp(R/alpha)` with `alpha=24 dB`, and power `(P/R)^beta` with `beta=e`. Each view is tested with and without a 520-element binary AP detection mask, producing six arms. A4 has 6 × 5 × 3 = 90 fold/seed runs. The transforms are deterministic per reading and require no statistics to be fitted on validation data; their motivation is Wi-Fi fingerprint similarity research [2].

| A4 arm | EvAAL-style (m), mean +/- SD | Position error (m) | Building acc. (%) | Floor acc. (%) |
|---|---:|---:|---:|---:|
| Powed + mask | **18.98 +/- 2.64** | 17.04 | 97.80 | 79.18 |
| Linear + mask | 19.47 +/- 1.66 | 17.56 | 97.92 | 78.28 |
| Exponential + mask | 20.35 +/- 1.47 | 18.38 | 98.04 | 75.21 |
| Linear reference | 21.95 +/- 1.71 | 19.40 | 96.31 | 82.47 |
| Exponential | 25.65 +/- 0.92 | 22.17 | 94.49 | 81.83 |
| Powed | 29.13 +/- 1.38 | 24.66 | 92.50 | 82.03 |

The result is an interaction: missingness masks help the nonlinear views because they distinguish “weak but detected” from “not detected.” Without a mask, the powed transform pushes some weak observations near zero, colliding with the missing code; the exponential transform compresses much of the signal dynamic range. The best EvAAL arm is not best on every output: masked arms improve building accuracy but reduce floor accuracy. Since the metric penalizes a wrong building by 50 m but a wrong floor by 4 m, its apparent gain partly reflects that weighting. In the paired five-fold comparison to unmasked linear, powed-mask has mean fold difference -2.91 +/- 2.28 m, p=0.046, wins all five folds and 12/15 matched runs. Other mask comparisons are inconclusive. This p-value is unadjusted across several arms and should be called exploratory, not definitive proof of universal superiority.

#### A7: denoising autoencoder pretraining

The DAE encoder is `520 → 256 → 128`, with batch normalization, ReLU, and dropout 0.2; the decoder mirrors the hidden layers and ends with sigmoid because the input view is in `[0,1]`. One DAE is trained for each outer fold and seed on outer-training users only. Its input receives 4 dBm Gaussian noise and 10% AP dropout on observed values; its target is the corresponding clean vector, and missingness is preserved. Around 96% of entries are missing, so the code scores reconstruction on observed and missing entries separately and uses `1.0 × observed_MSE + 0.1 × missing_MSE`. This reduces the risk that the model appears successful simply by reconstructing zeros. The encoder is transferred to the downstream MLP only when layer shapes match; mismatches raise an error.

| A7 initialization | EvAAL-style (m), mean +/- SD | Paired fold delta (m) | Fold-level p | Folds won | Matched runs won |
|---|---:|---:|---:|---:|---:|
| Random initialization | 21.95 +/- 1.71 | Reference | - | - | - |
| DAE-pretrained trunk | 19.76 +/- 0.87 | -2.20 +/- 2.37 | 0.107 | 4/5 | 8/15 |

Pretraining reduces the pooled mean and observed seed spread, but wins only 8/15 matched runs; the five-fold test p=0.107 does not establish a reliable gain. It is suggestive evidence. Reconstruction quality is a pretraining diagnostic, not the localization objective: a representation can reconstruct RSSI without preserving location-relevant neighborhoods. A4 has 90 runs and A7 30, for 120 reportable M2 runs.

#### A8: hybrid status and why current values are excluded

The intended A8 goal is to test whether a fold-specific, learned embedding can improve on the MLP and Phase 1 coordinate approach. Its planned candidates are the MLP coordinate head, embedding + distance-weighted kNN, the Phase 1 hierarchical classifier/regressor route under the same folds, and a stacking blend. The M3 encoder must be trained without outer test users; downstream regressors fit only outer-training embeddings; test routing uses predicted building/floor. Stacking requires leakage-safe out-of-fold base predictions within outer training. Oracle labels can appear only in a diagnostic column and cannot be headline predictions.

The current A8 runner imports the temporary stub rather than `src/models/dl/multitask.py`. It implements only the MLP head and embedding+kNN of the required four; an embedding+XGBoost arm and oracle diagnostic do not replace the missing matched Phase 1 hierarchy and stacking methods. Existing A8 runs are tagged `blocked_on_m3` and are excluded. **A8 is incomplete and no current A8 score should be claimed as a valid hybrid result.**

M2 artifacts: `docs/phase2_m2_report.md`, `configs/phase2_m2.yaml`, `src/data/phase2_m2.py`, `src/training/m2_trainer.py`, `src/models/dae.py`, `scripts/run_m2_a4_representations.py`, `scripts/run_m2_a7_pretraining.py`, `scripts/run_m2_a8_hybrid.py`, and `results/phase2/m2/`. The audit `python scripts/verify_m2_phase2.py` currently reports 47 checks passed, 0 failed.

### 5.5 M3 — multitask model architecture and ablations (A1-A3)

#### Network and objective

M3's default shared network accepts 520 RSSI values and uses a `520 → 512 → 256 → 128` trunk. The first two layers use linear transformation, batch normalization, ReLU, and dropout 0.3; the final trunk layer uses linear + ReLU. Its 128-value representation feeds three task heads: building `128 → 3`, floor `128 → 5`, and coordinate `128 → 64 → 2`. The coordinate output is standardized per training fold, then inverse-transformed to metres for scoring. The model has an `encode()` method to expose the learned 128-D representation for a future valid hybrid.

The shared equal-weight objective is `L = CE_building + CE_floor + Huber_coordinates`. Cross-entropy trains class probabilities; Huber is quadratic for small coordinate residuals and linear for large residuals, reducing sensitivity to large outliers compared with squared error. A1 compares the shared trunk with three independent task trunks. A2 chooses among fixed loss weights `(0.5,0.5,2)`, `(1,1,1)`, `(1,1,2)` using inner validation only and compares this with learned uncertainty weights `sum_t[exp(-s_t)L_t+s_t]`. A3 uses three building-conditioned floor heads and combines their logits using predicted building probabilities, so it does not use true building labels at inference.

The M3 trainer uses AdamW (learning rate `1e-3`, weight decay `1e-4`), batch size 256, up to 200 epochs, patience 20, and ReduceLROnPlateau scheduling (factor 0.5 after five non-improving validation epochs). The inner split is grouped by user and checked to cover building labels. The 75-run sweep is five unique configurations × five outer folds × three seeds; identical shared controls are run only once.

| M3 arm | EvAAL-style (m), mean +/- SD | Mean position error (m), mean +/- SD | Building acc. | Floor acc. | Floor acc. given correct building |
|---|---:|---:|---:|---:|---:|
| Shared equal-weight control | 21.47 +/- 0.86 | 18.67 +/- 0.89 | 95.9% +/- 1.6% | 81.9% +/- 1.0% | 82.5% +/- 1.4% |
| Independent task networks | **19.29 +/- 0.69** | **15.91 +/- 0.12** | 94.7% +/- 1.1% | **82.2% +/- 1.9%** | **84.4% +/- 0.9%** |
| Inner-selected fixed weights | 20.11 +/- 1.34 | 17.67 +/- 1.17 | **96.6% +/- 1.2%** | 81.2% +/- 0.4% | 81.8% +/- 0.5% |
| Learned uncertainty weights | 22.25 +/- 2.40 | 19.66 +/- 2.32 | 96.3% +/- 1.3% | 81.0% +/- 0.8% | 81.4% +/- 1.0% |
| Hierarchical floor head | 22.00 +/- 0.91 | 19.36 +/- 0.64 | 96.2% +/- 0.6% | 81.4% +/- 0.3% | 81.5% +/- 0.3% |

The independent-task architecture has the best observed mean EvAAL and coordinate error among completed M3 arms. Shared models are not uniformly worse: the inner-selected fixed weights reach the highest building accuracy. Learned uncertainty and the hierarchical floor head do not improve the headline localization score in this sweep. M4's paired interval for independent versus shared includes zero, so report the first as “best observed mean,” not as a proven statistically reliable win. Artifacts are under `results/phase2/m3/`; source files are `src/models/dl/multitask.py`, `src/models/dl/m3_experiments.py`, `src/data/phase2_m3.py`, and `src/models/dl/m3_support.py`.

### 5.6 M4 — paired statistics, calibration, learning curve, and error analysis

M4 checks that predictions align by source row, user, phone, target labels, fold, and seed before comparing. It restores original coordinate truth from the cleaned dataset using `source_row`, because the M3 prediction export stores coordinates as float32 and can round northings by roughly half a metre. Metrics are computed on the same 19,300 scans, five outer folds, and three seeds whenever an arm is eligible.

For uncertainty, M4 averages each user's score over scans and seeds, samples the 18 USERID groups 2,000 times with replacement, and forms paired percentile 95% intervals. Both models in a comparison use the same sampled users. This respects within-user scan dependence better than resampling thousands of scans independently. M4 also reports the exact McNemar test for seed-42 floor correctness at scan level; this is a secondary diagnostic because repeated scans per user violate the test's independence assumption. No multiple-comparison correction is applied, so the analyses are exploratory.

| Valid OOF model | EvAAL-style score (m) | Mean position error (m) | Floor acc. | Building acc. |
|---|---:|---:|---:|---:|
| M3 independent task networks | **19.29** | **15.91** | 82.18% | 94.66% |
| M2 A7 DAE-pretrained | 19.76 | 17.42 | 82.13% | 96.70% |
| M3 inner-selected fixed weights | 20.11 | 17.66 | 81.23% | 96.61% |
| M1 fixed baseline | 21.41 | 19.28 | 81.90% | 97.18% |
| M3 shared multitask | 21.47 | 18.67 | 81.86% | 95.85% |
| M2 A4 linear / A7 random initialization | 21.95 | 19.40 | 82.50% | 96.30% |

M3 independent versus shared has paired user-bootstrap EvAAL difference -1.11 m (95% CI -2.76 to +0.38); the interval contains zero. Versus the M1 baseline, the EvAAL difference is -1.62 m (-3.77 to +0.41), also containing zero; the position-error difference is -2.80 m (-4.36 to -1.53). M1 architecture and split caveats limit the cross-member interpretation. DAE pretraining versus random initialization has a paired user-mean EvAAL difference -1.32 m (95% CI -2.48 to -0.39) and position-error difference -1.26 m (-2.37 to -0.26); floor accuracy is lower by 1.1 percentage points (-2.0 to -0.3). These are unadjusted multi-model comparisons and should be described as exploratory.

**Floor calibration:** M4 runs each saved M3 checkpoint on its held-out fold, then reports top-label ECE using 15 equal-width bins, multiclass Brier score, and negative log-likelihood. Shared/independent ECE are both 0.139; inner-selected fixed is 0.140, learned uncertainty 0.146, and hierarchical 0.145. Independent NLL is 1.503 compared with 1.261 for shared; higher accuracy therefore does not mean better-calibrated probabilities. ECE changes with binning and should be interpreted alongside proper scoring rules.

**Learning curve:** the independent M3 architecture is retrained at 25%, 50%, 75%, and 100% of outer-training scans, while each fold's held-out users remain fixed. The number of epochs from the corresponding M3 inner-validation selection is frozen across fractions. This measures the effect of training volume under a fixed schedule, not re-tuned best performance at each size.

| Training fraction | Mean train rows | EvAAL (m), mean +/- SD | Position error (m), mean +/- SD | Floor acc., mean +/- SD |
|---:|---:|---:|---:|---:|
| 25% | 3,860 | 26.41 +/- 11.71 | 24.24 +/- 12.14 | 78.85% +/- 12.81% |
| 50% | 7,720 | 21.01 +/- 6.54 | 17.98 +/- 5.13 | 81.57% +/- 10.43% |
| 75% | 11,580 | 18.96 +/- 6.49 | 16.24 +/- 4.47 | 80.59% +/- 12.06% |
| 100% | 15,440 | 19.29 +/- 6.31 | 15.84 +/- 2.63 | 81.82% +/- 12.60% |

The broad trend improves with more data, but the means are not strictly monotonic. The 100% learning-curve predictions match the saved M3 independent runs, which checks the refit path. An error-review export records the 100 largest position errors per seed (300 rows), with user/phone/fold and true/predicted labels; it is for diagnosis and not for excluding difficult examples. Outputs live in `results/phase2/m4/`, generated by `scripts/analyze_m4_oof.py` and `scripts/run_m4_learning_curve.py`. A9 uncertainty and final candidate selection remain pending.

### 5.7 M5 — architecture width/depth sweep (A9) and integration

M5 evaluates width (units per hidden layer) and depth (number of hidden layers) around the M3 independent-task 512-wide, depth-3 reference. It varies one factor at a time: 512×2 and 512×4 change depth; 256×3 and 1024×3 change width. Four new variants each run on five USERID folds and three seeds (60 new fold/seed runs). The 512×3 reference reuses M3's existing 15 runs.

| Width × depth | EvAAL-style (m), mean +/- SD | Position error (m), mean +/- SD | Floor acc., mean +/- SD | Building acc., mean +/- SD |
|---|---:|---:|---:|---:|
| 512 × 2 | 20.95 +/- 7.18 | 17.30 +/- 4.16 | 80.02% +/- 12.87% | 94.29% +/- 8.11% |
| 512 × 3 reference | 19.29 +/- 6.31 | 15.84 +/- 2.63 | 81.82% +/- 12.60% | 94.56% +/- 8.02% |
| 512 × 4 | **16.68 +/- 6.01** | **13.99 +/- 2.95** | 82.01% +/- 11.18% | 96.06% +/- 6.24% |
| 256 × 3 | 21.31 +/- 6.83 | 18.47 +/- 4.51 | 80.96% +/- 12.34% | 95.84% +/- 6.63% |
| 1024 × 3 | 17.71 +/- 6.79 | 14.23 +/- 3.26 | 81.49% +/- 10.23% | 94.53% +/- 8.48% |

The 512×4 architecture has the lowest descriptive mean EvAAL and mean position error in this sweep. Its paired user-cluster uncertainty has not yet been calculated, so this does not establish a statistically reliable gain. M5 integrates output tables and figures while distinguishing pooled scan results from means/standard deviations across fold/seed metrics; these summaries are not interchangeable.

| Ablation | Experiment | Status |
|---|---|---|
| A1 | Shared versus independent task networks | Completed; M3. |
| A2 | Fixed selected versus learned loss weights | Completed; M3. |
| A3 | Flat versus building-conditioned floor head | Completed; M3. |
| A4 | RSSI views and missingness mask | Completed within M2's reference architecture. |
| A5 | Dropout, weight decay, batch normalization | Initial M1 sweep exists; transfer to final M3 model not established. |
| A6 | RSSI noise and AP dropout | Initial M1 sweep exists; transfer to final M3 model not established. |
| A7 | DAE-pretrained versus random trunk initialization | Completed; result is suggestive rather than established. |
| A8 | Learned embedding and hybrid routes | **Incomplete; stub-backed values are excluded.** |
| A9 | Network width and depth | Completed descriptively; paired uncertainty is pending. |

M5 artifacts: `scripts/run_m5_a9.py`, `scripts/build_m5_assets.py`, `results/phase2/m5/`, `results/figures/phase2_m5_*.png`, and `docs/phase2_integrated_report.md`.

### 5.8 Phase 2 evidence map and reproduction

The member reports provide implementation detail, while the files below are the machine-readable evidence used to reproduce the summary:

| Analysis | Main code | Main saved evidence |
|---|---|---|
| M1 A5/A6 | `scripts/run_m1_experiments.py`, `src/training/phase2_trainer.py` | `results/phase2/m1/` records, fold predictions, run summary, and figures. |
| M2 A4/A7 | `scripts/run_m2_a4_representations.py`, `scripts/run_m2_a7_pretraining.py`, `scripts/verify_m2_phase2.py` | `results/phase2/m2/` OOF predictions, run records, DAE checkpoints, and tables. |
| M3 A1-A3 | `scripts/run_m3_ablation.py`, `src/models/dl/m3_experiments.py` | `results/phase2/m3/` pooled OOF rows, records, checkpoints, and curves. |
| M4 analysis | `scripts/analyze_m4_oof.py`, `scripts/run_m4_learning_curve.py` | `results/phase2/m4/` paired metrics, bootstrap, calibration, learning curve, and error queue. |
| M5 A9 | `scripts/run_m5_a9.py`, `scripts/build_m5_assets.py` | `results/phase2/m5/` A9 tables and generated report figures. |

From the repository root, after installing the pinned Python 3.11 requirements, downloading the CSVs, and building folds, the current verification/analysis commands are:

```bash
pytest
python scripts/verify_m2_phase2.py
python scripts/analyze_m4_oof.py --bootstrap 2000
python scripts/run_m4_learning_curve.py --device auto
python scripts/build_m5_assets.py
```

These commands verify or rebuild analyses from saved artifacts; full model training runners are listed in the member reports and can take substantial time. `--skip-existing` can resume several training grids. There is no live Wi-Fi scanner or app command in the project.
## 6. Cross-phase comparison and interpretation

The available reports show that Phase 1's tuned per-building kNN has a lower mean position error (12.21 m) than the current Phase 2 neural models (for example, M3 independent around 15.91 m). These numbers arise from different model families and report aggregation protocols, so comparisons require care. Phase 2 should not be described as having improved over Phase 1 on coordinate accuracy based on the current evidence. Potential advantages of Phase 2 include joint outputs, learned embeddings, and extensible neural ablations; the valid A8 hybrid that was intended to test a learned-space local estimator remains incomplete.

There is also an architecture mismatch in member comparisons: M1 and M2 A4/A7 use a smaller `520 → 256 → 128` interface model, whereas M3 uses `520 → 512 → 256 → 128` with a nonlinear coordinate head. A5/A6 and A4/A7 comparisons are controlled within their own experiment families, but cross-member rows do not isolate architecture causally.

## 7. Reproducibility and how to run

Use Python 3.11 and the versions in `requirements.txt`. From a fresh clone:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_splits.py
python scripts/export_quality_report.py
pytest
```

The raw dataset is downloaded under `data/raw/` and is not stored in Git. Phase 1 and Phase 2 scripts write predictions, run records, aggregate tables, and figures to their documented `results/` subdirectories. Representative reproduction commands are listed in the README and member reports. Full training grids can take substantial time; `--skip-existing` is available on several runners to resume saved fold/seed work. For a demonstration, run an inference/replay example on a saved fingerprint; no live scanner command exists yet.

## 8. Limitations and future work

1. **Complete A8:** connect M2 to M3's actual fold-specific encoder; implement all planned hybrid candidates, including the matched Phase 1 hierarchical route and leakage-safe stacking; rerun and compare on the same USERID folds and seeds.
2. **Correct and repeat M1:** run A5/A6 using the final M3 model factory and an inner grouped split that covers all validation labels; keep initial artifacts for transparent comparison.
3. **Complete M4/M5 inference:** extend paired USERID-cluster uncertainty to A9, agree on the primary final selection metric and candidate, and update the report with the result.
4. **Finalize integration and demo:** regenerate tables/figures after valid A8 and M1 results; provide a reproducible saved-row prediction demonstration and document its input/output.
5. **Plan live localization separately:** live operation needs target-building fingerprint collection, BSSID-to-model-slot mapping, platform-specific scanning, matching preprocessing, packaged inference, and validation on phones/environment not represented by the public data. The current UJIIndoorLoc CSV does not itself supply an arbitrary current-device-to-WAP-slot mapping.
6. **Limit generalization claims:** building 0 has only two users, the outer design has five folds, and class/phone/environment shifts can affect performance. A result on this benchmark is not proof of accuracy in a new building.

## 9. Conclusion

Phase 1 delivers a reproducible and interpretable classical baseline: tuned tree ensembles classify building/floor, and a per-building kNN route gives the strongest reported coordinate result. Phase 2 has completed substantial controlled experiments in robustness, signal representations, pretraining, multitask architecture, statistical diagnostics, learning curves, and width/depth. The best observed Phase 2 mean is the 512×4 A9 result, but its paired uncertainty is pending; M3 independent is the best observed completed neural arm, with no statistically reliable EvAAL gain established versus its shared control. A8 and an architecture-matched M1 rerun remain important outstanding work. Therefore, the project has meaningful Phase 2 results, but the final Phase 2 model comparison and complete project conclusion should be finalized only after the listed work is done.

## References

[1] UCI Machine Learning Repository, “UJIIndoorLoc,” DOI: [10.24432/C5MS59](https://doi.org/10.24432/C5MS59).

[2] S. Torres-Sospedra et al., “Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems,” *Expert Systems with Applications*, 2015, DOI: [10.1016/j.eswa.2015.08.013](https://doi.org/10.1016/j.eswa.2015.08.013).

[3] S. Xia, Y. Liu, G. Yuan, M. Zhu, and Z. Wang, “Indoor Fingerprint Positioning Based on Wi-Fi: An Overview,” *ISPRS International Journal of Geo-Information*, vol. 6, no. 5, article 135, 2017, DOI: [10.3390/ijgi6050135](https://doi.org/10.3390/ijgi6050135).

[4] F. Liu et al., “Survey on WiFi-based indoor positioning techniques,” *IET Communications*, vol. 14, no. 9, pp. 1372–1383, 2020, DOI: [10.1049/iet-com.2019.1059](https://doi.org/10.1049/iet-com.2019.1059).
[5] X. Feng, K. A. Nguyen, and Z. Luo, “A survey of deep learning approaches for WiFi-based indoor positioning,” *Journal of Information and Telecommunication*, vol. 6, no. 2, pp. 163–216, 2022, DOI: [10.1080/24751839.2021.1975425](https://doi.org/10.1080/24751839.2021.1975425).

Additional project citation entries are in [`references_phase1.bib`](references_phase1.bib). The papers provide background and methodological motivation; measured project results come from the repository's saved grouped-fold artifacts.

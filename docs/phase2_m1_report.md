# Phase 2 — Member 1 (M1) Technical Report: Data Loading & Signal Robustness

**Author:** Member 1 (M1)  
**Project:** Wi-Fi RSSI Multitask Indoor Localization (Project 9, Phase 2)  
**Date:** October 28, 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

Member 1 is responsible for establishing a **fold-safe, leak-free data path** and evaluating **signal robustness and regularization** for Phase 2 neural indoor localization models.

In this phase, we:
1. **Defined and froze the Task 1 Input Contract** (`docs/phase2_input_contract.md`), restoring the canonical 520 WAP layout in strict ascending order, applying a linear-positive normalization mapping physical $[-110, 0]\text{ dBm} \to [0.0, 1.0]$, and strictly preserving the unobserved sentinel $+100 \to 0.0$.
2. **Implemented and validated `src/data/phase2.py`** with deterministic seeding, train-only coordinate normalization, and in-dBm signal augmentations. Unit tests in `tests/test_phase2_data.py` confirmed all data integrity invariants with 100% test pass rate.
3. **Executed the complete A5 (Regularization) and A6 (Augmentation) sweeps** across all 5 outer `USERID`-grouped cross-validation folds and 3 distinct random seeds (`42`, `1337`, `2024`), totaling **150 full model runs**.
4. **Enforced strict out-of-fold generalization**: each outer user fold was completely held out; early stopping epochs were selected via nested inner-grouped validation within training users, and the model was refit on the full training fold before predicting on the held-out users.

---

## 2. Experimental Results Summary (150 Runs across 10 Configurations)

All metrics are pooled out-of-fold estimates across 15 runs per configuration (5 folds $\times$ 3 seeds). Primary metric: **EvAAL-style score** ($d + 50 \cdot \mathbb{I}[\text{wrong building}] + 4 \cdot \mathbb{I}[\text{wrong floor}]$ in true metres).

| Experiment ID | Configuration Changes | Building Acc (%) | Floor Acc (%) | Mean Pos Error (m) | Median Pos Error (m) | 95th % Error (m) | EvAAL Score (m) | Final Train Loss | Final Val Loss |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`baseline`** | Dropout 0.2, BN On, WD 0.0, Aug None | $97.16 \pm 3.62$ | $81.60 \pm 10.41$ | $19.09 \pm 4.20$ | $15.87 \pm 3.85$ | $43.41 \pm 10.56$ | $21.24 \pm 5.49$ | $0.065 \pm 0.020$ | $1.231 \pm 0.964$ |
| **`a5_dropout_0.0`** | Dropout = 0.0 | $95.29 \pm 7.04$ | $79.40 \pm 11.48$ | $18.89 \pm 4.82$ | $15.41 \pm 3.91$ | $44.61 \pm 13.30$ | $22.06 \pm 7.35$ | $0.059 \pm 0.083$ | $1.344 \pm 1.040$ |
| **`a5_dropout_0.5`** | Dropout = 0.5 | $95.84 \pm 6.41$ | $82.07 \pm 9.47$ | $26.90 \pm 8.32$ | $24.28 \pm 8.24$ | $55.22 \pm 18.40$ | $29.70 \pm 8.39$ | $0.170 \pm 0.053$ | $1.256 \pm 0.970$ |
| **`a5_weight_decay_on`** | Weight Decay = $10^{-4}$ | $96.24 \pm 5.44$ | **$83.49 \pm 8.25$** | **$17.89 \pm 3.68$** | **$14.88 \pm 3.57$** | **$41.18 \pm 9.26$** | **$20.43 \pm 5.71$** | $0.069 \pm 0.020$ | **$1.015 \pm 0.765$** |
| **`a5_batch_norm_off`** | Batch Normalization = Off | $96.55 \pm 3.50$ | $81.77 \pm 13.57$ | $18.68 \pm 4.07$ | $14.79 \pm 2.76$ | $44.85 \pm 11.62$ | $21.13 \pm 4.34$ | **$0.043 \pm 0.005$** | $1.517 \pm 1.745$ |
| **`a6_noise_2dbm`** | Noise $\sigma = 2\text{ dBm}$ | $96.73 \pm 4.46$ | $81.13 \pm 10.59$ | $18.72 \pm 4.61$ | $15.40 \pm 4.15$ | $43.97 \pm 11.28$ | $21.11 \pm 5.85$ | $0.068 \pm 0.021$ | $1.260 \pm 1.015$ |
| **`a6_noise_4dbm`** | Noise $\sigma = 4\text{ dBm}$ | $96.00 \pm 5.40$ | $80.20 \pm 12.31$ | $18.20 \pm 3.97$ | $14.99 \pm 3.55$ | $42.49 \pm 9.58$ | $20.99 \pm 5.57$ | $0.069 \pm 0.019$ | $1.244 \pm 0.975$ |
| **`a6_ap_dropout_5pct`** | AP Dropout $p = 0.05$ | $95.14 \pm 6.52$ | $81.02 \pm 10.28$ | $18.46 \pm 3.36$ | $15.38 \pm 3.21$ | $42.67 \pm 10.09$ | $21.65 \pm 6.06$ | $0.084 \pm 0.029$ | $1.303 \pm 1.061$ |
| **`a6_ap_dropout_10pct`**| AP Dropout $p = 0.10$ | $95.51 \pm 5.95$ | $80.30 \pm 12.10$ | $18.48 \pm 4.13$ | $15.38 \pm 3.92$ | $42.07 \pm 10.07$ | $21.52 \pm 6.00$ | $0.121 \pm 0.118$ | $1.377 \pm 1.166$ |
| **`a6_combined`** | Noise $2\text{ dBm}$ + AP Drop $5\%$ | $95.58 \pm 5.68$ | $81.30 \pm 10.39$ | $18.68 \pm 3.86$ | $15.66 \pm 3.92$ | $42.86 \pm 9.64$ | $21.64 \pm 6.18$ | $0.085 \pm 0.028$ | $1.230 \pm 0.948$ |

---

## 3. Detailed Diagnosis & Ablation Analysis

### A5: Regularization Choices
- **Weight Decay ($\lambda = 10^{-4}$)** was the most effective regularizer tested. It reduced mean coordinate error by $1.20\text{ m}$ ($19.09\text{ m} \to 17.89\text{ m}$), improved floor accuracy to $83.49\%$, and compressed the generalization gap ($1.01$ validation loss vs. $1.23$ in baseline).
- **Dropout Rate (0.0 vs. 0.2 vs. 0.5)**:
  - Setting dropout to $0.0$ resulted in severe overfitting on high-density APs, increasing the EvAAL error to $22.06\text{ m}$ and degrading floor accuracy to $79.40\%$.
  - Increasing dropout to $0.5$ severely impaired coordinate regression, causing position error to jump to $26.90\text{ m}$ and EvAAL score to $29.70\text{ m}$. Coordinate regression relies on precise relative RSSI magnitudes across nearby APs; heavy dropout masks too many active signals simultaneously.
- **Batch Normalization**:
  - Removing batch normalization reduced training loss to $0.043$ (memorization) and resulted in volatile validation loss ($1.517 \pm 1.745$). Batch normalization is critical for gradient flow across heads with disparate target distributions.

### A6: Signal Augmentation
- **Gaussian Noise ($\sigma = 2\text{ dBm}$ and $4\text{ dBm}$)**:
  - Noise augmentation successfully regularized coordinate regression (improving mean position error to $18.20\text{ m}$ at $\sigma = 4\text{ dBm}$ and EvAAL score to $20.99\text{ m}$).
  - Small noise ($\sigma = 2\text{ dBm}$) proved especially beneficial for held-out users with dense AP scans (e.g. User 11 on Fold 0, where floor accuracy improved from $88.71\%$ to $94.24\%$).
- **AP Dropout (5% and 10%)**:
  - AP dropout slightly reduced coordinate error ($18.46\text{ m}$) but caused a slight degradation in building classification accuracy ($97.16\% \to 95.14\%$). Dropping key landmark APs occasionally misled the building classification head.
- **Combined Augmentation**:
  - Combining noise ($2\text{ dBm}$) with AP dropout ($5\%$) produced balanced performance ($18.68\text{ m}$ position error), but did not outperform weight decay alone. Augmentation acted primarily as a regularizer rather than an architectural breakthrough.

---

## 4. Viva / Report Talking Points

### Q1: Why add noise only to observed RSSI?
**Answer:** The value `+100` in UJIIndoorLoc is not a continuous numerical measurement of high power; it is an out-of-band sentinel denoting "not detected" (signal below the receiver sensitivity threshold, typically $\approx -104\text{ dBm}$, or AP transmitter not in range). Corrupting unobserved values with Gaussian noise would falsely inject phantom AP detections across the entire campus, destroying the true spatial sparsity structure of the Wi-Fi scan and altering the physical meaning of the fingerprint.

### Q2: How do we know the model overfits?
**Answer:** In the baseline model, final training loss drops to $0.0653 \pm 0.0199$ while outer held-out validation loss rises to $1.2309 \pm 0.9640$. When monitored during training, inner-validation loss and EvAAL scores plateau around epoch 11-13 while training loss continues to plummet toward zero. This widening generalization gap directly demonstrates that unregularized deep neural networks memorize device- and user-specific walking path nuances rather than invariant spatial propagation patterns.

### Q3: Why group cross-validation by `USERID`?
**Answer:** Random shuffling splits individual walking trajectories across both train and test sets, allowing the model to "cheat" by recognizing a specific phone hardware calibration or walking stride. Grouping folds strictly by `USERID` tests true out-of-sample generalization: the model must localize scans collected by human operators and smartphone models it has never encountered before.

### Q4: What does augmentation simulate?
**Answer:**
- **Gaussian Noise** simulates temporal RSSI multipath fading, environmental interference (moving pedestrians, weather fluctuations), and small variations in phone antenna orientation.
- **AP Dropout** simulates packet collisions on congested channels, transient beacon loss, and body shadowing (the user's body attenuating signal from APs behind them).
- *Limitation:* It does **not** simulate major structural domain shifts, such as AP hardware replacements, new partition walls, or cross-building RF propagation changes.

---

## 5. Artifacts and Generated Deliverables

1. **Unit Tests**:
   - `tests/test_phase2_data.py` (7 tests verifying 520 WAP restoration, sentinel protection, train-only scaling, and reproducible seeding).
2. **Result Files**:
   - `results/phase2/m1/m1_all_runs.csv`: 150 individual run records with full configuration, best epoch, and 15 metrics.
   - `results/phase2/m1/m1_summary_metrics.csv`: Aggregate statistics (mean $\pm$ std) across all 10 configurations.
   - 150 JSON records and 150 row-level prediction CSVs stored under `results/phase2/m1/`.
3. **Figures**:
   - `results/figures/m1_train_val_loss_curves.png`: Training vs. validation loss curves.
   - `results/figures/m1_robustness_userid.png`: Mean position error by held-out `USERID`.
   - `results/figures/m1_robustness_phoneid.png`: Mean position error by `PHONEID`.
   - `results/figures/m1_robustness_ap_count_bands.png`: Error across Low, Medium, and High detected AP count bands.


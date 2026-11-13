# Phase 2 Input Contract and Architecture Interface Agreement

**Project:** Project 9 (Wi-Fi RSSI Indoor Localization), Phase 2  
**Role:** Member 1 (M1 - Data Loading & Signal Robustness)  
**Stakeholders:** M1 (Data/Robustness), M2 (Hybrid Embeddings), M3 (Multitask Architecture), M4 (Tuning & Validation), M5 (Evaluation & Integration)  
**Status:** FROZEN  

---

## 1. Input Representation & Dimension Contract

1. **Dimensionality & Slot Ordering**:
   - The primary input vector has dimension $D = 520$, corresponding to `WAP001` through `WAP520` sorted in ascending numerical order by WAP index.
   - While Phase 1 dropped 55 training-constant WAPs for classical ML, Phase 2 deep learning standardizes the input width to all 520 WAPs so that architectures remain agnostic to training-subset sparsity patterns.
   - *Optional A4 Variant*: If testing explicit missingness indicators, the input vector is the concatenation of the 520 signal values and 520 binary detection flags, yielding dimension $D = 1{,}040$. This variant is explicitly declared in configuration files.

2. **Physical Units & Sentinel Representation**:
   - Raw scan readings represent received signal strength in dBm ($[-104, 0]\text{ dBm}$ typical).
   - Value `+100` is the dataset sentinel representing "not detected" (missing signal).
   - For neural network inputs, the signal values undergo a **linear-positive normalization**:
     $$\tilde{x}_i = \begin{cases} \text{clip}\left(\frac{x_i - (-110.0)}{0.0 - (-110.0)}, 0.0, 1.0\right) & \text{if } x_i \neq +100 \\ 0.0 & \text{if } x_i = +100 \end{cases}$$
   - Non-detected APs map strictly to $0.0$.
   - Any detected signal is strictly positive in $(0.0, 1.0]$.

---

## 2. Signal Augmentation Contract

1. **Scope and Application Boundary**:
   - Augmentation is applied **strictly during training**.
   - Validation and test scans are **never augmented**; they are evaluated in their clean, unaltered state.
   - The random generator is seeded per run and worker so data shuffling and augmentations are fully reproducible.

2. **Domain of Operation**:
   - All augmentations operate in the **raw dBm domain BEFORE** applying the linear-positive normalization:
     - **Noise Augmentation**: Add Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (e.g., $\sigma \in \{2, 4\}\text{ dBm}$) **only to observed APs** ($x_i \neq +100$). The resulting value is clamped to $[-110.0, 0.0]\text{ dBm}$.
     - **AP Dropout**: Randomly drop observed APs with probability $p \in \{0.05, 0.10\}$ by replacing their value with the sentinel `+100` (which subsequently maps to $0.0$).
     - **Preservation of Missingness**: Unobserved APs ($x_i = +100$) are **never corrupted** with noise or altered. Missingness indicates the AP was out of range or unobserved; treating it as a noisy continuous measurement would corrupt the semantic meaning of the Wi-Fi scan.

---

## 3. Coordinate Normalization Contract

1. **Target Normalization**:
   - Target coordinates (`LONGITUDE`, `LATITUDE`) are continuous values in UTM metres.
   - A `CoordinateScaler` (zero-mean, unit-variance `StandardScaler`) is fitted **strictly on outer training fold users**.
   - The fitted transform is applied unchanged to validation targets during training loss computation.
   - For all downstream evaluation, error metrics, and EvAAL calculation, model coordinate outputs are **inverse-transformed back to true UTM metres**. No metric is ever reported in normalized units.

2. **Categorical Targets**:
   - `BUILDINGID`: Integer in $\{0, 1, 2\}$ (3 classes).
   - `FLOOR`: Integer in $\{0, 1, 2, 3, 4\}$ (5 classes).

---

## 4. Multitask Model Interface (M3 $\to$ M1 / M2 / M5)

The shared multitask model factory must satisfy:

```python
def build_multitask_model(
    input_dim: int = 520,
    dropout: float = 0.2,
    batch_norm: bool = True,
    seed: int = 42,
    **kwargs
) -> nn.Module:
    """Build multitask PyTorch model.
    
    Forward pass takes tensor of shape (batch_size, input_dim).
    Returns dict or tuple:
      - building_logits: (batch_size, 3)
      - floor_logits:    (batch_size, 5)
      - coordinates:     (batch_size, 2)  [normalized]
      
    Must implement .encode(features) returning latent trunk representation (batch_size, hidden_dim).
    """
```

---

## 5. Fold & Data Leakage Rules

1. **Outer Folds**:
   - All models use the shared 5-fold `GroupKFold` split grouped by `USERID` stored in `data/splits/folds.json`.
   - All scans from a single user remain together.
   - The outer validation fold is strictly held out.
2. **Inner Validation for Tuning**:
   - Hyperparameter selection, regularization tuning, and early stopping epoch determination must be conducted inside the training users using inner grouped validation.
   - Outer validation folds are evaluated strictly once per model run for out-of-fold performance estimation.
3. **Official Held-Out File (`validationData.csv`)**:
   - The 1,111-row official file was already evaluated in Phase 1 baseline work. It is never used for tuning or model selection in Phase 2.


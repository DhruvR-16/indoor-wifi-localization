# Phase 2 — Member 2 (M2) Technical Report: Representations, Pretraining, Hybrid

**Role:** Member 2 (M2 — denoising representations and the hybrid model)
**Project:** Wi-Fi RSSI Multitask Indoor Localization (Project 9, Phase 2)
**Status:** A4 and A7 complete; A8 pipeline runs but is missing two of the
four named candidates and is blocked on M3's multitask model

---

## 1. Scope

Member 2 answers two questions with evidence and prepares a third:

| Ablation | Question | Status |
|---|---|---|
| **A4** | Does the RSSI input view (linear-positive vs exponential vs powed, with and without missingness flags) change held-out-user localization? | Complete |
| **A7** | Does fold-matched denoising pretraining of the trunk beat random initialisation? | Complete |
| **A8** | Does a multitask-trunk embedding plus a routed local estimator beat both the Phase 1 ML baseline and the Phase 2 MLP? | Pipeline runs on 2 of 4 named candidates; results blocked on M3 |

A better score is a hypothesis, not a requirement. The primary metric was fixed
before any comparison: the **EvAAL-style score**
(`d + 50·1[wrong building] + 4·1[wrong floor]`, in true metres). Neutral and
negative findings are reported with the same weight as gains.

## 2. Protocol

Everything that decides which rows a model may see is reused from Member 1's
frozen contract (`docs/phase2_input_contract.md`): the 520-WAP canonical
layout, the `+100 → 0.0` sentinel rule, the five `USERID`-grouped outer folds
in `data/splits/folds.json`, and the `CoordinateScaler` fitted on training
users only. Augmentation stays in the raw dBm domain, applies to training
inputs only, and never corrupts the sentinel.

Each arm changes exactly one factor and inherits the rest from
`configs/phase2_m2.yaml:reference` — trunk `520 → 256 → 128`, dropout 0.2,
batch norm on, Adam at 1e-3, batch size 64, at most 20 epochs with patience 6,
equal loss weights. Three seeds (42, 1337, 2024) × five folds per arm.

The stopping epoch `E*` is chosen inside the outer-training users, the model is
then refit from scratch on all outer-training users for `E*` epochs, and the
held-out users are scored exactly once. `scripts/verify_m2_phase2.py` audits
these invariants against the artifacts on disk.

### 2.1 Correction to the inner validation split

Member 1's trainer selects `E*` with `GroupKFold(n_splits=4)` over the
outer-training users. That split is not label-aware, and **building 0 is
visited by only two users (1 and 11)**. Outer folds 0 and 1 each hold one of
them out, leaving a single building-0 user in outer training — whom the plain
split then sends to inner validation. The consequence:

| Outer fold | Buildings missing from inner training | Share of inner-validation rows affected | Recorded inner EvAAL | Outer EvAAL |
|---|---|---:|---:|---:|
| 0 | building 0 | 75.6% | 182.91 m | 18.28 m |
| 1 | building 0 | 55.6% | 137.62 m | 17.51 m |
| 2–4 | none | 0.0% | 16–30 m | 15–27 m |

On folds 0 and 1 the stopping epoch was therefore chosen against a signal
dominated by a class the inner model had never been shown. Member 2's runs use
a building-stratified grouped split that keeps users intact and additionally
requires inner training to cover every label present in inner validation
(`src/data/phase2_m2.py:make_inner_split`); all five folds pass that check.
`mode="legacy"` reproduces the original protocol for comparison.

**This affects Member 1's published A5/A6 table**, whose 150 runs used the
uncorrected split. Whether those runs are repeated is a team decision; the
defect is recorded here so the limitation travels with the numbers.

### 2.2 Known limitation of the stopping rule

Even with coverage fixed, inner EvAAL plateaus by roughly epoch 6 and then
fluctuates within about 3 m, so `E*` is partly selected by noise and runs
frequently reach the 20-epoch ceiling. Member 1's budget was kept for
comparability rather than re-tuned, and the three-seed design is what absorbs
this variance. Seed spread is material — on fold 0 the reference arm scores
15.90 m at seed 42 and 28.15 m at seed 1337 — so no single-seed difference in
this project should be read as a result.

## 3. Methods

### 3.1 A4 — RSSI representations

With `P(x) = clip(x, -110, 0) + 110` for a detected AP and `R = 110`:

| View | Definition | Parameter |
|---|---|---|
| linear-positive | `P(x) / R` | — (the M1 contract view) |
| exponential | `exp(P(x)/α) / exp(R/α)` | α = 24 dB |
| powed | `(P(x) / R)^β` | β = e |

Definitions follow Torres-Sospedra et al. (2015). Every view sends the
not-detected sentinel to exactly 0.0 and detected readings into (0.0, 1.0], and
is a pure per-reading function, so nothing is fitted and no quantity can cross
a fold boundary. The `_mask` arms concatenate 520 binary detection flags,
giving a 1040-wide input.

### 3.2 A7 — denoising pretraining

The DAE encoder is `520 → 256 → 128` with the same per-layer module order as
the multitask trunk (`Linear → BatchNorm → ReLU → Dropout`), so the two A7 arms
are **architecture-matched** and the only difference is the trunk's starting
weights. The guide's `520 → 512 → 256 → 128` trunk would not match, and
`load_encoder_into_trunk` raises on any shape mismatch rather than skipping or
reshaping a layer; a test pins that behaviour.

One DAE is trained per (fold, seed) on that fold's outer-training users only.
Corruption is resampled every epoch and applied to inner-training inputs only
(Gaussian noise σ = 4 dBm on observed APs plus 10% AP dropout, sentinel
preserved); the target is always the clean scan in the same view. Because ~96%
of a scan is not detected, reconstruction is scored separately on observed and
missing entries and combined as `1.0·observed + 0.1·missing`; both components
are recorded per checkpoint. Low reconstruction error is not evidence that the
embedding is useful for localization, which is why A7 is judged on held-out
localization metrics and not on reconstruction.

### 3.3 A8 — multitask-embedding hybrid

The hybrid takes the 128-D trunk embedding via the frozen `encode(features)`
interface and fits a routed coordinate estimator on it, reusing Phase 1's
`HierarchicalCoordinateRegressor` unchanged so that the hybrid differs from the
Phase 1 coordinate route in exactly one factor — the feature space — and
inherits its route fallbacks and `min_samples` rule. Candidates: the multitask
coordinate head, embedding + distance-weighted kNN, embedding + XGBoost, and an
oracle-routed diagnostic.

Leakage controls: the encoder is fold-specific; the estimator is fitted only on
outer-training embeddings and coordinates; its settings are tuned on an inner
split using the inner model, which never saw those rows; and held-out rows are
routed by **predicted** building/floor. Oracle routing is recorded only as a
named diagnostic (`diagnostic_only: true`) and is excluded from headline
comparisons.

**Two of the guide's four named candidates are not yet built.** The guide asks
for four arms on the same held-out rows: the multitask coordinate head,
embedding + kNN, a hierarchical classifier with the Phase 1 coordinate
regressor, and a stacking blend. Only the first two are implemented here;
`a8_embed_xgb` is the guide's second estimator choice and
`a8_embed_knn_oracle` is a diagnostic, so neither substitutes for the two
missing arms. The Phase 1 hierarchical route exists in Phase 1's own results
but has not been re-scored on these fold ids and route mode, and **no stacking
blend has been written** — so the guide's leakage rule for it (base predictions
for meta-training users produced without training a base model on those users,
meta-model and its regularization fitted inside the outer training portion
only) is specified in this section but not yet exercised by any code. Both arms
are pending with A8 itself; the four-way comparison the guide asks for cannot
be closed until they exist.

**A8 is not yet reportable.** M3's multitask model is not in the repository, so
the embedding currently comes from `src/models/stub_multitask.py`. Every A8
record carries `blocked_on_m3: true`, and a stub-backed number must not be
described as "the multitask embedding" or presented as an A8 result.

## 4. Results

180 runs: 90 for A4, 30 for A7, 60 for A8 (four arms per fold and seed — the
head, embedding + kNN, embedding + XGBoost, and the oracle diagnostic; these
are not the guide's four named candidates, see Section 3.3).
Every figure below is generated from the artifacts by
`scripts/build_m2_tables.py --markdown`, which writes
`results/phase2/m2/m2_results_tables.md`; nothing here is transcribed by hand.

Two levels of uncertainty are reported, because they disagree and the weaker
one is the honest one. The **user-clustered** interval resamples held-out
`USERID`s. The **fold-level** figure is the row-weighted difference per outer
fold, with a t interval and p-value over the five folds. Each user is held out
by exactly one fold and shares that fold's model, so users are not independent
replicates and the user interval is optimistic; the five folds are the
independent unit, and an arm is only called robust when both agree.

### 4.1 A4 — RSSI representations

Pooled out-of-fold, 19,300 scans per seed, mean ± std over three seeds:

| Arm | EvAAL (m) | Mean error (m) | Building acc (%) | Floor acc (%) |
|---|---:|---:|---:|---:|
| `a4_powed_mask` | **18.98 ± 2.64** | 17.04 | 97.80 | 79.18 |
| `a4_linear_mask` | 19.47 ± 1.66 | 17.56 | 97.92 | 78.28 |
| `a4_exponential_mask` | 20.35 ± 1.47 | 18.38 | 98.04 | 75.21 |
| `a4_linear` (reference) | 21.95 ± 1.71 | 19.40 | 96.31 | 82.47 |
| `a4_exponential` | 25.65 ± 0.92 | 22.17 | 94.49 | 81.83 |
| `a4_powed` | 29.13 ± 1.38 | 24.66 | 92.50 | 82.03 |

Paired against `a4_linear`:

| Arm | Δ EvAAL per fold (m) | p (5 folds) | Folds won | Runs won | Verdict |
|---|---:|---:|---:|---:|---|
| `a4_powed_mask` | −2.91 ± 2.28 | **0.046** | 5/5 | 12/15 | improves |
| `a4_linear_mask` | −2.29 ± 3.12 | 0.176 | 3/5 | 11/15 | inconclusive |
| `a4_exponential_mask` | −1.39 ± 2.98 | 0.356 | 3/5 | 8/15 | inconclusive |
| `a4_exponential` | +3.51 ± 3.28 | 0.075 | 1/5 | 4/15 | degrades, suggestive |
| `a4_powed` | +6.66 ± 7.63 | 0.123 | 1/5 | 3/15 | degrades, suggestive |

The result is **an interaction, not a ranking**. Both non-linear views are
worse than linear-positive on their own, and both recover once missingness
flags are added; `a4_powed_mask` is the only arm that clears p < 0.05.

The mechanism is measurable in the inputs themselves, and the two failing views
fail for different reasons:

* **`powed` collides with the sentinel.** It encodes 18.1% of all 345,080
  observed readings below 0.01, where "not detected" is exactly 0.0 — a
  −104 dBm detection becomes 0.00037. `linear` and `exponential` encode 0.00%
  of readings that low.
* **`exponential` compresses dynamic range.** No collisions, but the median
  observed reading lands at 0.033 against linear-positive's 0.254, so most of
  the signal is squeezed into the bottom of the input range.

Explicit detection flags separate "is this AP detected" from "how strong is
it", which is exactly the information both views destroy — hence the largest
gain lands on the view that destroys the most.

**A caveat that belongs with the headline.** Every `_mask` arm trades floor
accuracy for building accuracy: +1.5 to +1.7 points of building accuracy
against −3 to −7 points of floor accuracy. EvAAL improves because a wrong
building costs 50 m and a wrong floor only 4 m, so part of this gain is the
metric's weighting rather than better localization. A deployment that cares
about floors should not read `a4_powed_mask` as the better input view.

### 4.2 A7 — denoising pretraining

| Arm | EvAAL pooled (m) | Δ per fold (m) | p (5 folds) | Folds won | Runs won |
|---|---:|---:|---:|---:|---:|
| `a7_random_init` | 21.95 ± 1.71 | — | — | — | — |
| `a7_dae_pretrained` | 19.76 ± 0.87 | −2.20 ± 2.37 | 0.107 | 4/5 | 8/15 |

**Pretraining is not an established gain.** The mean moves in its favour by
2.2 m and four of five folds prefer it, but the fold-level test gives p = 0.107
and it wins only 8 of 15 matched runs. Reported as suggestive. It does
noticeably reduce seed spread (pooled std 0.87 m against 1.71 m), which is a
smaller but more consistent claim than the mean.

The 15 DAE checkpoints reconstruct well in absolute terms — observed-signal
MSE 0.00604 ± 0.00178 (RMSE 8.47 ± 1.20 dBm) and missing-entry MSE
0.00413 ± 0.00060, under the stated 1.0 / 0.1 weighting — but **reconstruction
quality does not predict transfer benefit**: across the 15 runs the correlation
between a checkpoint's observed reconstruction MSE and its localization gain is
r = +0.26 (p = 0.355), and for missing entries r = +0.36 (p = 0.184). The sign
is as expected and the evidence is weak. This is the concrete answer to "why is
reconstruction not enough": a well-reconstructing encoder need not preserve
location-relevant neighbourhoods, and in this project it measurably did not.

### 4.3 A8 — multitask-embedding hybrid (PRELIMINARY)

**These numbers are not an A8 result.** The embedding comes from the M3
interface stub, not M3's multitask model, and every record carries
`blocked_on_m3: true`. They establish that the pipeline runs and that its
leakage controls hold; the comparison must be repeated on M3's model.

| Candidate | EvAAL (m) | Mean error (m) | Δ vs MLP head per fold (m) | p | Folds won |
|---|---:|---:|---:|---:|---:|
| `a8_embed_knn` | 19.03 ± 10.19 | 16.45 | −2.79 ± 7.80 | 0.469 | 4/5 |
| `a8_embed_xgb` | 20.21 ± 9.86 | 17.63 | −1.61 ± 7.72 | 0.666 | 4/5 |
| `a8_mlp_head` | 21.82 ± 5.67 | 19.23 | — | — | — |
| `a8_embed_knn_oracle` | 21.89 ± 12.39 | 19.30 | diagnostic only | — | — |

The hybrid wins four of five folds and is far steadier across seeds inside a
fold (11.6–13.6 m on folds 0–2 where the head ranges 13.9–28.2 m), but **fold 3
reverses it** by +9.3 m, and that single fold is enough to make the mean
difference insignificant (p = 0.469, std across folds 7.80 m). Fold 3 holds out
five users and is the hardest fold for every arm in this report; a local
estimator extrapolates badly there. The classification heads are untouched by
the hybrid, so building and floor accuracy are identical across all four
candidates by construction — only coordinates differ.

**Oracle routing is not an upper bound here**, which is worth stating because
it is counter-intuitive: it is *worse* than predicted routing in three of five
folds and by 2.9 m on average. On fold 4 seed 42, the 76.1% of rows whose
predicted route already equals the true route score identically under both
(13.74 m, as they must). On the 904 mislabelled rows, predicted routing scores
19.11 m and oracle routing 65.20 m — predicted routing wins 95.5% of them, by
46 m on average. When the trunk mislabels a scan, that scan's embedding also
sits among training scans of the *predicted* label, so routing by the true
label hands it to a local model whose neighbourhood does not contain it and the
estimator extrapolates. Routing by predictions is self-consistent with the
embedding geometry. The practical consequence is the opposite of the usual
worry: for a learned-space local estimator, oracle routing can *understate*
rather than flatter the method, so it is not a valid ceiling and is kept out of
the headline comparison.

### 4.4 Summary of findings

1. Input representation matters, and it interacts with missingness flags.
   `a4_powed_mask` is the one arm with a statistically supported gain over the
   frozen contract view (−2.91 m per fold, p = 0.046, 5/5 folds).
2. On their own, both non-linear views are worse than linear-positive, for two
   measurable and different reasons (sentinel collision; range compression).
3. The flag arms' gain is partly an artifact of EvAAL's 50 m / 4 m weighting:
   they buy building accuracy with floor accuracy.
4. DAE pretraining is suggestive, not established (p = 0.107, 8/15 runs); its
   clearer effect is reduced seed variance.
5. Reconstruction quality did not predict transfer benefit (r = +0.26,
   p = 0.355).
6. The A8 pipeline is leak-checked, but it is stub-backed, blocked on M3, and
   covers only two of the four candidates the guide names (no hierarchical
   Phase 1 arm, no stacking blend). Even preliminarily, the hybrid's advantage
   is fold-dependent and not significant.
7. Oracle routing is not an upper bound for an embedding-space local estimator.

### 4.5 Limitations

* Five outer folds is a small number of independent replicates. A 2–3 m
  difference in EvAAL is near the resolution this design can establish, which
  is why several arms land as "suggestive".
* The stopping rule remains noise-driven (Section 2.2); `E*` frequently reaches
  the 20-epoch ceiling.
* Member 1's A5/A6 table was produced with the uncorrected inner split
  (Section 2.1) and has not been regenerated.
* Building 0 is supported by only two users, so any user-grouped result
  involving it rests on a two-person sample.
* A4 and A7 were run on the linear-positive view for A7 and the reference
  capacity throughout; representation and pretraining were not crossed, so an
  interaction between them is untested.
* A8 covers two of the guide's four named candidates (Section 3.3): the
  hierarchical-classifier + Phase 1-regressor arm and the stacking blend are
  not implemented, so the required four-way hybrid comparison is incomplete
  independently of the M3 block.
* The official 1,111-row file was not touched by any of this work.

## 5. Reproducing

```bash
python scripts/train_m2_dae.py                    # 15 fold-matched DAE checkpoints
python scripts/run_m2_a4_representations.py       # A4: 6 arms x 5 folds x 3 seeds
python scripts/run_m2_a7_pretraining.py           # A7: 2 arms x 5 folds x 3 seeds
python scripts/run_m2_a8_hybrid.py                # A8: preliminary until M3 lands
python scripts/build_m2_tables.py --reference a4_linear
python scripts/verify_m2_phase2.py                # audits the invariants above
```

The 180 per-run prediction CSVs are ~75 MB and are regenerated deterministically
by the runners, so they are not tracked. `results/phase2/m2/m2_oof_predictions.csv.gz`
(25 MB, 694,800 rows) is the tracked copy at full precision. From a fresh clone:

```bash
python scripts/export_m2_oof_predictions.py --restore   # rebuilds preds/ exactly
```

The restored files are byte-identical to the originals, so the audit and the
tables above reproduce from the archive alone without re-running any training.
The restore uses pandas' `float_precision="round_trip"` parser, because the
default C parser shifts a float64's last digit by one unit in the last place; a
test pins that.


Each runner is resumable with `--skip-existing`. On macOS, a single process
running dozens of runs on MPS degrades severely (observed: ~35 s per run rising
to ~1400 s after 40 runs) as freed tensors accumulate in the allocator cache
while the machine swaps; `release_accelerator_memory()` is called after every
run, and for a full grid it is safer to invoke the runners one run per process.
`scripts/run_m2_a8_hybrid.py` pins OpenMP to a single thread before importing
torch or xgboost, because their two OpenMP runtimes segfault the interpreter on
macOS when both are loaded multi-threaded.

## 6. Handoffs

| Direction | Contents |
|---|---|
| M3 → M2 | Fold-specific multitask checkpoints, `encode(features)`, out-of-fold classification predictions, training metadata. **Outstanding — blocks A8.** |
| M4 → M2 | Tuning boundaries, selected estimator settings, matched fold ids, uncertainty conventions. |
| M2 → M5 | A4/A7/A8 configs, row-level out-of-fold predictions (`results/phase2/m2/m2_oof_predictions.csv.gz`, one row per scan per arm per seed, keyed by `run`), per-run metrics (`records/`), fold/seed summaries, and the interpretation in this report. |

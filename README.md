# SemMF — Semantically-Regularized Matrix Factorization

LLM-enhanced collaborative filtering on MovieLens-1M. Standard matrix
factorization produces latent item embeddings that have no semantic meaning
— they only capture rating co-occurrence patterns. SemMF anchors those
embeddings to the geometry of a pretrained sentence-transformer space via
a regularization term, producing a structured latent space that
generalizes to **cold items with zero ratings**.

## Concept

Base CF (Matrix Factorization, BPR objective):

```
score(u, i) = u_u · v_i  (+ biases)
```

Two LLM roles:

1. **Semantic regularizer** (training): a learned projection
   `f: R^384 → R^k` maps each item's MiniLM embedding `e_i` into the CF
   latent space, and `||v_i - f(e_i)||²` is added to the loss with a
   per-item adaptive weight `λ_i = λ₀ / (1 + α · n_i)`. Cold items get
   pulled hard toward their semantic anchor; popular items stay free.
2. **Cold-start at inference**: cold items rely on the regularizer-aligned
   `v_i` directly — the regularizer pulls `v_i` toward `f(e_i)` during
   training, so cold V's already encode semantic structure at test time.
   We tested explicit substitution `v̂_i := f(e_i)` (norm-matched) at
   inference and it *hurt* cold NDCG (0.033 → 0.015 on adaptive); the
   regularizer-converged direction is more useful than `f(e_i)` directly,
   because `f` is shared across all items and trained primarily to match
   warm V's. The substitution path is kept in the model code as an option
   but disabled by default.

Two regularizer variants are implemented and ablated:

- **MSE**: `Σ_i λ_i · ||normalize(v_i) − normalize(f(e_i))||²`
- **InfoNCE** (contrastive): `Σ_i λ_i · -log[exp(sim(v_i, f(e_i))/τ) /
  Σ_j exp(sim(v_i, f(e_j))/τ)]` over a sampled batch of items per step.

The InfoNCE variant only requires each `v_i` to be closer to its own LLM
anchor than to other items' anchors — more flexible than strict MSE.

## Dataset

MovieLens-1M. 6,040 users × 3,883 movies × 1M ratings.

**Download and extract before running:**

```bash
# Download MovieLens-1M
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/raw/
```

Or download manually from https://grouplens.org/datasets/movielens/1m/
and extract so that the following files exist:

```
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

After filtering users with `< 5` interactions and removing val/test users
not seen in train, we keep 5,400 users / 3,883 items / ~800k train / ~24k
val / ~80k test. **Temporal global split** (80/10/10 by timestamp): items
that only appear post-cutoff are naturally cold. 536 items have `< 5`
training ratings; 154 cold-positive interactions land in test.

Item text fed to MiniLM uses `Title + Year + Genres`:

```
Toy Story (1995). Genres: Animation, Children's, Comedy.
```

## Headline results

3 seeds, mean ± std, MovieLens-1M temporal split, full-rank evaluation:

| Variant                | NDCG@10           | Cold NDCG@10      |
|------------------------|------------------:|------------------:|
| `mse_mf`               | 0.1795 ± 0.0046   | 0.0225 ± 0.0024   |
| `bpr_mf`               | **0.2324 ± 0.0015** | 0.0299 ± 0.0041 |
| `llm_only`             | 0.1740 ± 0.0062   | 0.0326 ± 0.0048   |
| `llm_init`             | 0.2269 ± 0.0032   | 0.0288 ± 0.0028   |
| `semmf_mse_fixed`      | 0.2201 ± 0.0030   | 0.0279 ± 0.0123   |
| **`semmf_mse_adaptive`** | **0.2296 ± 0.0031** | **0.0358 ± 0.0068** |
| `semmf_infonce` (τ=0.01) | 0.2302 ± 0.0024 | 0.0330 ± 0.0061   |
| `lightgcn` (100 ep, RecBole) | 0.1728 ± 0.0021 | 0.0104 ± 0.0069 |

Full table (NDCG@5/10/20, Recall@10, HR@10, RMSE) in
[results/tables/main_table.md](results/tables/main_table.md).

**Findings:**

1. `semmf_mse_adaptive` is the headline winner on cold-start: **+20%
   over BPR-MF** and **+10% over LLM-Only**, while matching BPR-MF on
   overall NDCG within one std (0.2296 vs 0.2324).
2. `llm_only` (no CF, items = `f(e_i)`) is competitive on cold-start
   but loses ~25% of overall NDCG — pure LLM is not a substitute for CF.
3. `llm_init` (PCA-init then trained freely without regularizer) gets a
   small lift over BPR-MF on cold-start but the lift is fragile: V drifts
   away from semantic init during training. The regularizer is what makes
   it stick.
4. **InfoNCE is competitive once τ is tuned**: at τ=0.1 (initial guess)
   it scored cold NDCG 0.024; a single-seed sweep over τ ∈ {0.01, 0.05,
   0.1, 0.3, 1.0} found τ=0.01 best by ~2× — the contrastive loss needs
   a sharp temperature to make the i=j positives meaningfully outscore
   the off-diagonal negatives. With τ=0.01: cold NDCG 0.0330 ± 0.006,
   variance halved. Still slightly behind MSE-adaptive on cold, but ties
   it on overall NDCG (0.2302 vs 0.2296).
5. `semmf_mse_fixed` is materially worse than adaptive — confirming the
   adaptive λ deviation paid off.
6. **LightGCN underperforms BPR-MF by 26%** on overall NDCG and is
   effectively zero on cold (0.0104). The cause is the temporal split:
   graph propagation memorizes interaction patterns of the training time
   window, but those patterns shift in the temporal test window. SemMF's
   semantic anchor on V[cold] is invariant to the time period — derived
   from item text, not interactions — so it generalizes across the
   temporal gap in a way graph CF cannot.
7. **Negative result: decoupling `f`'s training from V's adaptive pull
   hurt both variants** (MSE-adaptive cold NDCG dropped 29%, InfoNCE
   collapsed to 0.0021). Code path in `src/train.py` behind
   `cfg.reg.decouple_f`, default off.

### How `λ_0` was tuned (single-seed sweep on `semmf_mse_adaptive`, seed 42)

| λ_0 | NDCG@10 | Cold NDCG@10 |
|---:|---:|---:|
| 0.0 (BPR-MF) | 0.2304 | 0.0278 |
| 0.01 | 0.2315 | 0.0113 |
| 0.1 | 0.2278 | 0.0289 |
| **0.5** | **0.2301** | **0.0415** |
| 1.0 | 0.2298 | 0.0326 |
| 10 | 0.2231 | 0.0170 |
| 100 | 0.2186 | 0.0229 |
| 1000 | 0.1993 | 0.0058 |

The curve is non-monotonic with a clear peak at λ_0 = 0.5. Both
overshooting and undershooting hurt: too-large λ corrupts the CF signal
on V[warm]; too-small λ leaves V[cold] dominated by BPR negative-sampling
noise. λ_0 = 0.5 is the sweet spot.

### How `τ` was tuned for InfoNCE (single-seed sweep, seed 42)

| τ | NDCG@10 | Cold NDCG@10 |
|---:|---:|---:|
| **0.01** | 0.2268 | **0.0375** |
| 0.05 | 0.2271 | 0.0123 |
| 0.1 (initial guess) | 0.2262 | 0.0161 |
| 0.3 | 0.2257 | 0.0128 |
| 1.0 | 0.2268 | 0.0186 |

### Negative results (attempts to beat BPR-MF on overall NDCG)

Three additional architectural ideas were tried. All three failed:

1. **Architecture probe** (`k=128, num_negatives=4`): single-seed win
   didn't replicate across seeds. Mean: 0.2299 / 0.0333.
2. **User-side semantic regularization** (`cfg.reg.user_reg=True`):
   geometric mismatch with BPR's discriminative objective. Cold NDCG
   dropped 42% at the weakest tested strength.
3. **Hybrid concat architecture** (`cfg.model.architecture="hybrid"`):
   without an auxiliary regularizer, `f` loses its semantic alignment.
   Overall NDCG dropped to 0.1971 (−14%).

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MovieLens-1M (see Dataset section above)
```

## Reproduce

Run the full ablation grid (8 variants × 3 seeds, ~4 GPU-hours):

```bash
python -m src.baselines
```

Quick smoke run (2 variants, 1 seed, 5 epochs — completes in ~2 min):

```bash
python -m src.baselines --variants bpr_mf,semmf_mse_adaptive --seeds 42 --epochs 5
```

Outputs are written to:

- `results/tables/per_seed.csv` — every (variant, seed) test metric
- `results/tables/aggregated.csv` — mean ± std over seeds
- `results/tables/main_table.md` — headline ablation table
- `results/tables/runs.json` — full per-run history (epoch curves)
- `results/figures/ndcg10.png`, `cold_ndcg10.png`, `recall10.png` — bar charts
- `results/figures/training_curves.png` — val metric vs epoch

## Ablation variants

| Name                  | What it tests                                            |
|-----------------------|----------------------------------------------------------|
| `mse_mf`              | Pure pointwise MF baseline (RMSE/MAE only)               |
| `bpr_mf`              | Pure pairwise MF baseline                                |
| `llm_only`            | Item vectors = `f(e_i)`; no free V                       |
| `llm_init`            | V initialized from PCA of LLM embeddings, then trains    |
| `semmf_mse_fixed`     | SemMF with MSE regularizer, fixed λ for all items        |
| `semmf_mse_adaptive`  | SemMF with MSE regularizer, per-item adaptive λ          |
| `semmf_infonce`       | SemMF with InfoNCE regularizer, per-item adaptive λ      |
| `lightgcn`            | Neural graph-CF baseline (RecBole, 100 epochs, defaults) |
| `semmf_hybrid`        | (negative result) hybrid concat: `v_i = [V_cf ; f(e)]`  |

## File map

```
src/
├── config.py        # all hyperparameters; one preset factory per variant
├── data_loader.py   # parse ML-1M, temporal split, cold flags, BPR/MSE datasets
├── embeddings.py    # sentence-transformer encode + content-hashed cache
├── model.py         # SemMF: U, V, projection f, biases, cold-substitution path
├── train.py         # BPR/MSE loop, adaptive λ, InfoNCE, val early stopping
├── evaluate.py      # full-rank NDCG/Recall/HR, RMSE/MAE, cold-start subset
└── baselines.py     # orchestrates the full ablation grid + tables + figures
experiments/         # hyperparameter sweeps and diagnostic scripts
results/
├── tables/          # CSV and markdown result tables
└── figures/         # plots and bar charts
```

## Key design choices

- **BPR over MSE for SemMF** — ranking metrics dominate the eval;
  pointwise MSE under-performs for top-K. RMSE/MAE reported only on
  the dedicated MSE-MF baseline.
- **Linear projection** for `f` rather than MLP — only ~3,700 items
  provide alignment signal; an MLP overfits and degrades cold-start
  generalization.
- **Adaptive per-item λ at λ_0 = 0.5** — cold items get pulled hard
  toward `f(e_i)`; popular items stay nearly free. The ablation shows
  this matters: `semmf_mse_adaptive` (Cold NDCG 0.0358) beats
  `semmf_mse_fixed` (0.0279) by 28%.
- **Implicit cold-start (regularizer-only)** — the regularizer pulls
  cold V's toward their LLM-projected anchors during training; explicit
  substitution `v̂_i := f(e_i)` at inference underperformed (0.036 →
  0.015) and is disabled by default.
- **Full-rank evaluation** (no sampled negatives) — sampled-eval
  rankings are unreliable (Rendle 2019); with ~3,900 items full-rank
  is tractable.
- **Cold-start subset evaluation** — both the relevance set and
  candidate pool are restricted to cold items only, isolating cold-item
  ranking quality from warm-item dominance.

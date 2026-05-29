"""3-seed validation of (k=128, nneg=4) on semmf_mse_adaptive.

Bar to lock as new default:
  mean ndcg@10  > 0.2324 (BPR-MF mean from current grid)
  mean cold@10  > 0.0322 (current adaptive mean 0.0358 - 10% slack)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.train import train_and_eval


SEEDS = [42, 123, 2024]
K = 128
NNEG = 4


def main() -> None:
    base = PRESETS["semmf_mse_adaptive"]()
    data = build_dataset(base)
    emb = load_item_embeddings(data, base)

    rows = []
    for seed in SEEDS:
        cfg = PRESETS["semmf_mse_adaptive"]()
        cfg.seed = seed
        cfg.model.latent_dim = K
        cfg.train.num_negatives = NNEG
        print(f"\n--- semmf_mse_adaptive  seed={seed}  k={K} nneg={NNEG} ---", flush=True)
        t0 = time.time()
        r = train_and_eval(cfg, data, emb, verbose=False)
        dur = time.time() - t0
        t = r["test"]
        print(f"  TEST ndcg@10={t['ndcg@10']:.4f}  recall@10={t['recall@10']:.4f}  "
              f"hr@10={t['hr@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
              f"({dur:.0f}s)", flush=True)
        rows.append((seed, t["ndcg@10"], t["recall@10"], t["hr@10"], t["cold_ndcg@10"]))

    print(f"\n=== 3-seed summary (k={K}, nneg={NNEG}) ===", flush=True)
    print(f"{'seed':>5s}  {'ndcg@10':>8s}  {'recall@10':>10s}  {'hr@10':>7s}  "
          f"{'cold_ndcg@10':>14s}", flush=True)
    print('-' * 60, flush=True)
    for s, n, r, h, c in rows:
        print(f"{s:>5d}  {n:>8.4f}  {r:>10.4f}  {h:>7.4f}  {c:>14.4f}", flush=True)
    print('-' * 60, flush=True)
    arr = np.array([[n, r, h, c] for _, n, r, h, c in rows])
    means, stds = arr.mean(axis=0), arr.std(axis=0)
    print(f"{'mean':>5s}  {means[0]:>8.4f}  {means[1]:>10.4f}  {means[2]:>7.4f}  "
          f"{means[3]:>14.4f}", flush=True)
    print(f"{'std':>5s}  {stds[0]:>8.4f}  {stds[1]:>10.4f}  {stds[2]:>7.4f}  "
          f"{stds[3]:>14.4f}", flush=True)

    print("\n--- references (3-seed means from current grid) ---", flush=True)
    print("  bpr_mf                          ndcg@10=0.2324  cold_ndcg@10=0.0299", flush=True)
    print("  semmf_mse_adaptive  k=64 nneg=1 ndcg@10=0.2296  cold_ndcg@10=0.0358", flush=True)

    print("\n--- decision ---", flush=True)
    pass_overall = means[0] > 0.2324
    pass_cold = means[3] > 0.0322
    print(f"  ndcg@10 mean = {means[0]:.4f}  >  0.2324 (BPR-MF)?  {'YES' if pass_overall else 'NO'}", flush=True)
    print(f"  cold@10 mean = {means[3]:.4f}  >  0.0322 (adaptive - 10%)?  {'YES' if pass_cold else 'NO'}", flush=True)
    print(f"  LOCK NEW DEFAULTS: {'YES' if (pass_overall and pass_cold) else 'NO'}", flush=True)


if __name__ == "__main__":
    main()

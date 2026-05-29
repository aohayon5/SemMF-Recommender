"""Architecture probe on seed 42 for semmf_mse_adaptive.

Question: do larger k or more BPR negatives push us above BPR-MF on overall NDCG?
Tests:
  baseline:   k=64,  num_negatives=1   (reference: ndcg=0.2301, cold=0.0415)
  big_k:      k=128, num_negatives=1
  more_neg:   k=64,  num_negatives=4
  combined:   k=128, num_negatives=4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.train import train_and_eval


def main() -> None:
    base = PRESETS["semmf_mse_adaptive"]()
    data = build_dataset(base)
    emb = load_item_embeddings(data, base)

    configs = [
        {"name": "k=128 nneg=1", "k": 128, "nneg": 1},
        {"name": "k=64  nneg=4", "k": 64, "nneg": 4},
        {"name": "k=128 nneg=4", "k": 128, "nneg": 4},
    ]

    rows = []
    for c in configs:
        cfg = PRESETS["semmf_mse_adaptive"]()
        cfg.seed = 42
        cfg.model.latent_dim = c["k"]
        cfg.train.num_negatives = c["nneg"]
        print(f"\n--- {c['name']} ---", flush=True)
        t0 = time.time()
        r = train_and_eval(cfg, data, emb, verbose=False)
        dur = time.time() - t0
        t = r["test"]
        print(f"  TEST ndcg@10={t['ndcg@10']:.4f}  recall@10={t['recall@10']:.4f}  "
              f"hr@10={t['hr@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
              f"({dur:.0f}s)", flush=True)
        rows.append((c["name"], t["ndcg@10"], t["recall@10"], t["hr@10"],
                     t["cold_ndcg@10"], dur))

    print("\n=== summary (semmf_mse_adaptive, seed 42) ===", flush=True)
    print(f"{'config':<14s}  {'ndcg@10':>8s}  {'recall@10':>10s}  {'hr@10':>7s}  "
          f"{'cold_ndcg@10':>14s}  {'time':>5s}", flush=True)
    print('-' * 70, flush=True)
    for n, a, b, c_hr, d, dur in rows:
        print(f"{n:<14s}  {a:>8.4f}  {b:>10.4f}  {c_hr:>7.4f}  {d:>14.4f}  {dur:>4.0f}s",
              flush=True)
    print("\n--- references (seed 42) ---", flush=True)
    print("  baseline (k=64 nneg=1)  ndcg@10=0.2301  recall@10=0.0687  "
          "hr@10=0.6747  cold_ndcg@10=0.0415", flush=True)
    print("  bpr_mf (k=64 nneg=1)    ndcg@10=0.2304  recall@10=0.0675  "
          "hr@10=0.6606  cold_ndcg@10=0.0278", flush=True)


if __name__ == "__main__":
    main()

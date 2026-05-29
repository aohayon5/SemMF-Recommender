"""User-side semantic regularization gamma_u sweep on seed 42.

Tests semmf_mse_adaptive + user_reg=True at gamma_u in {0.01, 0.1, 0.5}.

Bar to declare success and proceed to 3-seed validation:
  best ndcg@10 > 0.2301 (current adaptive baseline) by a meaningful margin
  cold_ndcg@10 within ~10% of baseline 0.0415 (no major regression)
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

    rows = []
    for gamma_u in [0.01, 0.1, 0.5]:
        cfg = PRESETS["semmf_mse_adaptive"]()
        cfg.seed = 42
        cfg.reg.user_reg = True
        cfg.reg.gamma_u = gamma_u
        print(f"\n--- semmf_mse_adaptive + user_reg gamma_u={gamma_u} ---", flush=True)
        t0 = time.time()
        r = train_and_eval(cfg, data, emb, verbose=False)
        dur = time.time() - t0
        t = r["test"]
        print(f"  TEST ndcg@10={t['ndcg@10']:.4f}  recall@10={t['recall@10']:.4f}  "
              f"hr@10={t['hr@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
              f"({dur:.0f}s)", flush=True)
        rows.append((gamma_u, t["ndcg@10"], t["recall@10"], t["hr@10"], t["cold_ndcg@10"]))

    print(f"\n=== summary (semmf_mse_adaptive + user_reg, seed 42) ===", flush=True)
    print(f"{'gamma_u':>8s}  {'ndcg@10':>9s}  {'recall@10':>10s}  {'hr@10':>7s}  "
          f"{'cold_ndcg@10':>14s}", flush=True)
    print('-' * 60, flush=True)
    for g, n, r, h, c in rows:
        print(f"{g:>8.3f}  {n:>9.4f}  {r:>10.4f}  {h:>7.4f}  {c:>14.4f}", flush=True)
    print("\n--- references (seed 42) ---", flush=True)
    print("  baseline (no user_reg)  ndcg@10=0.2301  recall@10=0.0687  "
          "hr@10=0.6747  cold_ndcg@10=0.0415", flush=True)
    print("  bpr_mf reference        ndcg@10=0.2304  recall@10=0.0675  "
          "hr@10=0.6606  cold_ndcg@10=0.0278", flush=True)


if __name__ == "__main__":
    main()

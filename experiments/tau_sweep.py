"""Tau sweep for semmf_infonce on seed 42 with current default lambda_0=0.5.

Question: InfoNCE under-performed MSE-adaptive on cold-start (0.024 vs 0.036).
Was this just bad temperature tuning?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.train import train_and_eval


def main() -> None:
    base = PRESETS["semmf_infonce"]()
    data = build_dataset(base)
    emb = load_item_embeddings(data, base)

    results = []
    for tau in [0.01, 0.05, 0.1, 0.3, 1.0]:
        cfg = PRESETS["semmf_infonce"]()
        cfg.seed = 42
        cfg.reg.tau = tau
        print(f"\n--- tau = {tau} (lambda_0={cfg.reg.lambda_0}) ---", flush=True)
        r = train_and_eval(cfg, data, emb, verbose=False)
        t = r["test"]
        print(f"  ndcg@10={t['ndcg@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
              f"cold_recall@10={t['cold_recall@10']:.4f}", flush=True)
        results.append((tau, t["ndcg@10"], t["cold_ndcg@10"], t["cold_recall@10"]))

    print("\n=== summary (semmf_infonce, seed 42, lambda_0=0.5) ===", flush=True)
    print(f"{'tau':>10s}  {'ndcg@10':>9s}  {'cold_ndcg@10':>14s}  {'cold_recall@10':>14s}", flush=True)
    for tau, n, cn, cr in results:
        print(f"{tau:>10.2f}  {n:>9.4f}  {cn:>14.4f}  {cr:>14.4f}", flush=True)
    print("\n--- references (seed 42, current best) ---", flush=True)
    print("  bpr_mf            ndcg@10=0.2304  cold_ndcg@10=0.0278", flush=True)
    print("  semmf_adapt l=0.5 ndcg@10=0.2301  cold_ndcg@10=0.0415", flush=True)
    print("  semmf_infonce(old t=0.1, l=0.5) ndcg@10=0.2262  cold_ndcg@10=0.0161", flush=True)


if __name__ == "__main__":
    main()

"""Low-lambda mini-sweep: lambda_0 in {0.01, 0.1, 0.5}.

The full sweep showed lambda=1 best of {1, 10, 100, 1000} — bigger lambda hurts.
Question: is the optimum even lower than 1?
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
    base = PRESETS["semmf_mse_adaptive"]()
    data = build_dataset(base)
    emb = load_item_embeddings(data, base)

    results = []
    for lam in [0.01, 0.1, 0.5]:
        cfg = PRESETS["semmf_mse_adaptive"]()
        cfg.seed = 42
        cfg.reg.lambda_0 = lam
        print(f"\n--- lambda_0 = {lam} ---", flush=True)
        r = train_and_eval(cfg, data, emb, verbose=False)
        t = r["test"]
        print(f"  ndcg@10={t['ndcg@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
              f"cold_recall@10={t['cold_recall@10']:.4f}", flush=True)
        results.append((lam, t["ndcg@10"], t["cold_ndcg@10"], t["cold_recall@10"]))

    print("\n=== summary (semmf_mse_adaptive, seed 42) ===", flush=True)
    print(f"{'lambda_0':>10s}  {'ndcg@10':>9s}  {'cold_ndcg@10':>14s}  {'cold_recall@10':>14s}", flush=True)
    for lam, n, cn, cr in results:
        print(f"{lam:>10.2f}  {n:>9.4f}  {cn:>14.4f}  {cr:>14.4f}", flush=True)
    print("\n--- references (seed 42, from main grid + earlier sweep) ---", flush=True)
    print("  bpr_mf       ndcg@10=0.2304  cold_ndcg@10=0.0278", flush=True)
    print("  llm_only     ndcg@10=0.1669  cold_ndcg@10=0.0309", flush=True)
    print("  semmf lam=1  ndcg@10=0.2298  cold_ndcg@10=0.0326", flush=True)
    print("  semmf lam=10 ndcg@10=0.2231  cold_ndcg@10=0.0170", flush=True)


if __name__ == "__main__":
    main()

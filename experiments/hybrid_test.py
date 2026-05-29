"""semmf_hybrid single-seed test (seed 42).

Bar to proceed to 3-seed validation:
  ndcg@10 > 0.2324 (BPR-MF mean) by a meaningful margin
  cold_ndcg@10 ideally above ~0.030 (close to or beating semmf_mse_adaptive)
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
    cfg = PRESETS["semmf_hybrid"]()
    cfg.seed = 42
    print(f"semmf_hybrid: arch={cfg.model.architecture}  "
          f"k_cf={cfg.model.latent_dim - cfg.model.k_sem}  k_sem={cfg.model.k_sem}  "
          f"obj={cfg.train.objective}  reg={cfg.reg.mode}", flush=True)

    data = build_dataset(cfg)
    emb = load_item_embeddings(data, cfg)

    t0 = time.time()
    r = train_and_eval(cfg, data, emb, verbose=True)
    dur = time.time() - t0
    t = r["test"]

    print(f"\n=== semmf_hybrid (seed 42) ===", flush=True)
    print(f"  ndcg@5  = {t['ndcg@5']:.4f}", flush=True)
    print(f"  ndcg@10 = {t['ndcg@10']:.4f}", flush=True)
    print(f"  ndcg@20 = {t['ndcg@20']:.4f}", flush=True)
    print(f"  recall@10 = {t['recall@10']:.4f}", flush=True)
    print(f"  hr@10     = {t['hr@10']:.4f}", flush=True)
    print(f"  cold_ndcg@10   = {t['cold_ndcg@10']:.4f}", flush=True)
    print(f"  cold_recall@10 = {t['cold_recall@10']:.4f}", flush=True)
    print(f"  ({dur:.0f}s)", flush=True)

    print(f"\n--- references (seed 42) ---", flush=True)
    print(f"  bpr_mf              ndcg@10=0.2304  cold_ndcg@10=0.0278", flush=True)
    print(f"  semmf_mse_adaptive  ndcg@10=0.2301  cold_ndcg@10=0.0415", flush=True)
    print(f"\n--- references (3-seed means) ---", flush=True)
    print(f"  bpr_mf              ndcg@10=0.2324 +/- 0.0015  cold=0.0299 +/- 0.0041", flush=True)
    print(f"  semmf_mse_adaptive  ndcg@10=0.2296 +/- 0.0031  cold=0.0358 +/- 0.0068", flush=True)

    bar_overall = 0.2324 + 0.0015  # bpr_mf mean + std
    bar_cold = 0.030
    pass_overall = t['ndcg@10'] > bar_overall
    pass_cold = t['cold_ndcg@10'] > bar_cold
    print(f"\n--- decision (single seed; needs 3-seed validation if both pass) ---", flush=True)
    print(f"  ndcg@10 = {t['ndcg@10']:.4f}  >  {bar_overall:.4f} (bpr_mf + std)?  "
          f"{'YES' if pass_overall else 'NO'}", flush=True)
    print(f"  cold@10 = {t['cold_ndcg@10']:.4f}  >  {bar_cold:.4f}?  "
          f"{'YES' if pass_cold else 'NO'}", flush=True)
    print(f"  PROCEED TO 3-SEED VALIDATION: {'YES' if (pass_overall and pass_cold) else 'NO'}",
          flush=True)


if __name__ == "__main__":
    main()

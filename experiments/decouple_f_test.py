"""Decouple-f validation on seed 42.

Tests whether splitting the regularizer into two stop-gradient terms (V trained
adaptively, f trained uniformly on warm items) improves cold NDCG on top of the
already-tuned lambda_0=0.5 (and tau=0.01 for InfoNCE).

If yes, productionize via preset updates and run final 3-seed grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.train import train_and_eval


def run(name: str, **overrides) -> dict:
    """Build a config from a preset, apply overrides, train+eval on seed 42."""
    cfg = PRESETS[overrides.pop("preset")]()
    cfg.seed = 42
    for path, value in overrides.items():
        section, attr = path.split(".")
        setattr(getattr(cfg, section), attr, value)
    print(f"\n--- {name} ---", flush=True)
    print(f"  reg.decouple_f={cfg.reg.decouple_f}  reg.lambda_0={cfg.reg.lambda_0}  "
          f"reg.tau={cfg.reg.tau}  reg.gamma={cfg.reg.gamma}", flush=True)
    return train_and_eval(cfg, DATA, EMB, verbose=False)


def main() -> None:
    global DATA, EMB
    base = PRESETS["semmf_mse_adaptive"]()
    DATA = build_dataset(base)
    EMB = load_item_embeddings(DATA, base)

    configs = [
        {"name": "MSE-adaptive baseline (l=0.5)",
         "preset": "semmf_mse_adaptive"},
        {"name": "MSE-adaptive + decouple_f",
         "preset": "semmf_mse_adaptive", "reg.decouple_f": True},
        {"name": "InfoNCE tau=0.01 baseline",
         "preset": "semmf_infonce", "reg.tau": 0.01},
        {"name": "InfoNCE tau=0.01 + decouple_f",
         "preset": "semmf_infonce", "reg.tau": 0.01, "reg.decouple_f": True},
    ]

    results = []
    for c in configs:
        r = run(**c)
        t = r["test"]
        print(f"  TEST ndcg@10={t['ndcg@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
              f"cold_recall@10={t['cold_recall@10']:.4f}", flush=True)
        results.append((c["name"], t["ndcg@10"], t["cold_ndcg@10"], t["cold_recall@10"]))

    print("\n=== summary (seed 42) ===", flush=True)
    print(f"{'config':>40s}  {'ndcg@10':>9s}  {'cold_ndcg@10':>14s}  {'cold_recall@10':>14s}", flush=True)
    for n, a, b, c in results:
        print(f"{n:>40s}  {a:>9.4f}  {b:>14.4f}  {c:>14.4f}", flush=True)
    print("\n--- references ---", flush=True)
    print("  bpr_mf seed42      ndcg@10=0.2304  cold_ndcg@10=0.0278", flush=True)
    print("  llm_only seed42    ndcg@10=0.1669  cold_ndcg@10=0.0309", flush=True)


if __name__ == "__main__":
    main()

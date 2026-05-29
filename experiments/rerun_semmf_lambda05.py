"""Re-run SemMF variants with the new lambda_0=0.5 default; baselines unchanged.

Loads existing per_seed.csv, drops the SemMF rows, runs SemMF x 3 seeds with
the current config defaults, appends the new rows, re-aggregates, and re-writes
tables and figures via baselines.save_results.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.baselines import save_results
from src.config import Config, PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.evaluate import aggregate_metrics
from src.train import train_and_eval


SEMMF_VARIANTS = ["semmf_mse_fixed", "semmf_mse_adaptive", "semmf_infonce"]
SEEDS = [42, 123, 2024]


def _runs_from_per_seed(per_seed_path: Path,
                        runs_json_path: Path) -> dict:
    """Reconstruct {variant: [run_dict, ...]} from saved CSV+JSON."""
    runs_by_variant: dict = defaultdict(list)
    if runs_json_path.exists():
        data = json.loads(runs_json_path.read_text())
        for variant, runs in data.items():
            for r in runs:
                runs_by_variant[variant].append({
                    "name": variant,
                    "seed": int(r["seed"]),
                    "test": dict(r["test"]),
                    "history": r.get("history", []),
                })
    return runs_by_variant


def main() -> None:
    base = Config()
    data = build_dataset(base)
    emb = load_item_embeddings(data, base)
    print(f"Re-running SemMF with lambda_0={base.reg.lambda_0} on seeds {SEEDS}", flush=True)

    runs_by_variant = _runs_from_per_seed(
        base.paths.results_tables / "per_seed.csv",
        base.paths.results_tables / "runs.json",
    )
    print(f"Loaded existing runs for: {sorted(runs_by_variant.keys())}", flush=True)

    # Drop the old SemMF results — we're replacing them.
    for v in SEMMF_VARIANTS:
        if v in runs_by_variant:
            print(f"  dropping old runs for {v} (n={len(runs_by_variant[v])})", flush=True)
            del runs_by_variant[v]

    # Run new SemMF results.
    for variant in SEMMF_VARIANTS:
        for seed in SEEDS:
            cfg = PRESETS[variant]()
            cfg.seed = seed
            print(f"\n=== {variant}  seed={seed}  lambda_0={cfg.reg.lambda_0} ===", flush=True)
            r = train_and_eval(cfg, data, emb, verbose=False)
            t = r["test"]
            print(f"  TEST: ndcg@10={t['ndcg@10']:.4f}  recall@10={t['recall@10']:.4f}  "
                  f"cold_ndcg@10={t['cold_ndcg@10']:.4f}", flush=True)
            runs_by_variant[variant].append(r)

    aggregated = {v: aggregate_metrics([r["test"] for r in runs])
                  for v, runs in runs_by_variant.items()}

    save_results(dict(runs_by_variant), aggregated, base)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

"""Re-run only semmf_infonce x 3 seeds with the new tau=0.01 preset default.

Drops the existing semmf_infonce rows from runs.json/per_seed.csv, re-runs,
and re-aggregates tables and figures via baselines.save_results.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import save_results
from src.config import Config, PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.evaluate import aggregate_metrics
from src.train import train_and_eval


VARIANT = "semmf_infonce"
SEEDS = [42, 123, 2024]


def _runs_from_json(path: Path) -> dict:
    runs_by_variant: dict = defaultdict(list)
    if path.exists():
        data = json.loads(path.read_text())
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
    cfg_check = PRESETS[VARIANT]()
    print(f"Re-running {VARIANT} with tau={cfg_check.reg.tau}, "
          f"lambda_0={cfg_check.reg.lambda_0} on seeds {SEEDS}", flush=True)

    runs_by_variant = _runs_from_json(base.paths.results_tables / "runs.json")
    if VARIANT in runs_by_variant:
        print(f"  dropping old runs for {VARIANT} (n={len(runs_by_variant[VARIANT])})", flush=True)
        del runs_by_variant[VARIANT]

    for seed in SEEDS:
        cfg = PRESETS[VARIANT]()
        cfg.seed = seed
        print(f"\n=== {VARIANT}  seed={seed} ===", flush=True)
        r = train_and_eval(cfg, data, emb, verbose=False)
        t = r["test"]
        print(f"  TEST: ndcg@10={t['ndcg@10']:.4f}  recall@10={t['recall@10']:.4f}  "
              f"cold_ndcg@10={t['cold_ndcg@10']:.4f}", flush=True)
        runs_by_variant[VARIANT].append(r)

    aggregated = {v: aggregate_metrics([r["test"] for r in runs])
                  for v, runs in runs_by_variant.items()}

    save_results(dict(runs_by_variant), aggregated, base)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

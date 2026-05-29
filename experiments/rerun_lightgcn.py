"""Add LightGCN (100 epochs, 3 seeds) as a new row in the results table.

Loads existing runs.json (with all 7 SemMF/baseline variants), trains LightGCN
x 3 seeds via RecBole at 100 epochs, plugs results in as variant 'lightgcn',
re-aggregates, and re-writes tables/figures via baselines.save_results.

Each seed runs in a subprocess so RecBole/CUDA memory is fully released
between seeds (the in-process version died silently between seed 42 and 123).
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import save_results
from src.config import Config
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.evaluate import aggregate_metrics


SEEDS = [42, 123, 2024]
EPOCHS = 100
PROJECT_ROOT = Path(__file__).resolve().parent.parent


SEED_RUNNER = """
import json, sys
from pathlib import Path
sys.path.insert(0, r'{root}')
from src.config import Config
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from experiments.lightgcn_test import run_one_seed

base = Config()
data = build_dataset(base)
_ = load_item_embeddings(data, base)
metrics = run_one_seed(seed={seed}, data=data, cfg=base,
                       work_dir=Path(r'{work_dir}'), epochs={epochs})
print('__METRICS_JSON__' + json.dumps({{k: float(v) for k, v in metrics.items()}}))
"""


def run_seed_subprocess(seed: int, epochs: int, work_dir: Path) -> dict:
    code = SEED_RUNNER.format(root=str(PROJECT_ROOT), seed=seed,
                              work_dir=str(work_dir), epochs=epochs)
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if line.startswith("__METRICS_JSON__"):
            return json.loads(line[len("__METRICS_JSON__"):])
    raise RuntimeError(
        f"seed {seed} subprocess produced no metrics. "
        f"exit={result.returncode}\nSTDOUT tail:\n{result.stdout[-1000:]}\n"
        f"STDERR tail:\n{result.stderr[-1000:]}"
    )


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
    _ = load_item_embeddings(data, base)  # warm cache (not used by LightGCN itself)
    print(f"Adding lightgcn x {len(SEEDS)} seeds at {EPOCHS} epochs", flush=True)

    runs_by_variant = _runs_from_json(base.paths.results_tables / "runs.json")
    if "lightgcn" in runs_by_variant:
        print(f"  dropping old lightgcn runs (n={len(runs_by_variant['lightgcn'])})",
              flush=True)
        del runs_by_variant["lightgcn"]

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        for seed in SEEDS:
            print(f"\n--- spawning subprocess for seed {seed} ---", flush=True)
            t0 = time.time()
            metrics = run_seed_subprocess(seed, EPOCHS, work_dir)
            dur = time.time() - t0
            print(f"  TEST ndcg@10={metrics['ndcg@10']:.4f}  "
                  f"recall@10={metrics['recall@10']:.4f}  "
                  f"hr@10={metrics['hr@10']:.4f}  "
                  f"cold_ndcg@10={metrics['cold_ndcg@10']:.4f}  ({dur:.0f}s)",
                  flush=True)
            runs_by_variant["lightgcn"].append({
                "name": "lightgcn",
                "seed": seed,
                "test": metrics,
                "history": [],
            })
            # write partial results after each seed so a mid-run failure doesn't
            # lose what we already computed
            aggregated = {v: aggregate_metrics([r["test"] for r in runs])
                          for v, runs in runs_by_variant.items()}
            save_results(dict(runs_by_variant), aggregated, base)
            print(f"  (partial tables saved)", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

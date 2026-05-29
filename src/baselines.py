"""Orchestrator: run all 7 ablation variants x N seeds, save tables and figures."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config, PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.evaluate import aggregate_metrics
from src.train import train_and_eval


def _save_per_seed(runs_by_variant: Dict[str, List[Dict]], out_dir: Path) -> None:
    rows = []
    for variant, runs in runs_by_variant.items():
        for r in runs:
            row = {"variant": variant, "seed": r["seed"]}
            for k, v in r["test"].items():
                row[k] = v
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_seed.csv", index=False)


def _save_aggregated(aggregated: Dict[str, Dict], out_dir: Path) -> None:
    rows = []
    for variant, mdict in aggregated.items():
        row = {"variant": variant}
        for metric, stats in mdict.items():
            if isinstance(stats, dict) and "mean" in stats:
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "aggregated.csv", index=False)


def _save_markdown_table(aggregated: Dict[str, Dict], path: Path) -> None:
    cols = [
        ("ndcg@5", "NDCG@5"),
        ("ndcg@10", "NDCG@10"),
        ("ndcg@20", "NDCG@20"),
        ("recall@10", "Recall@10"),
        ("hr@10", "HR@10"),
        ("cold_ndcg@10", "Cold NDCG@10"),
        ("rmse", "RMSE"),
    ]
    header = "| Variant | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines = [header, sep]
    for variant, m in aggregated.items():
        cells = [variant]
        for key, _ in cols:
            if key in m:
                cells.append(f"{m[key]['mean']:.4f} ± {m[key]['std']:.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_runs_json(runs_by_variant: Dict[str, List[Dict]], out_dir: Path) -> None:
    serializable = {}
    for variant, runs in runs_by_variant.items():
        serializable[variant] = [
            {
                "seed": r["seed"],
                "test": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                         for k, v in r["test"].items()},
                "history": r["history"],
            }
            for r in runs
        ]
    with open(out_dir / "runs.json", "w") as f:
        json.dump(serializable, f, indent=2, default=float)


def _plot_metric_bars(aggregated: Dict[str, Dict], metric: str,
                      title: str, out_path: Path, color: str = "C0") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [v for v in aggregated if metric in aggregated[v]]
    means = [aggregated[v][metric]["mean"] for v in variants]
    stds = [aggregated[v][metric]["std"] for v in variants]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(variants, means, yerr=stds, capsize=3, color=color, alpha=0.85)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_training_curves(runs_by_variant: Dict[str, List[Dict]],
                          out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for variant, runs in runs_by_variant.items():
        if not runs or not runs[0]["history"]:
            continue
        all_curves = []
        max_len = max(len(r["history"]) for r in runs)
        for r in runs:
            curve = [h["val_metric"] for h in r["history"]]
            curve += [curve[-1]] * (max_len - len(curve))
            all_curves.append(curve)
        arr = np.array(all_curves)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, label=variant, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val metric (NDCG@10 / RMSE)")
    ax.set_title("Training curves (mean ± std over seeds)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_results(runs_by_variant: Dict[str, List[Dict]],
                 aggregated: Dict[str, Dict], cfg: Config) -> None:
    cfg.make_dirs()
    tdir = cfg.paths.results_tables
    fdir = cfg.paths.results_figures

    _save_per_seed(runs_by_variant, tdir)
    _save_aggregated(aggregated, tdir)
    _save_markdown_table(aggregated, tdir / "main_table.md")
    _save_runs_json(runs_by_variant, tdir)

    _plot_metric_bars(aggregated, "ndcg@10",
                      "Test NDCG@10 by variant (mean ± std)",
                      fdir / "ndcg10.png", color="C0")
    _plot_metric_bars(aggregated, "cold_ndcg@10",
                      "Test Cold NDCG@10 by variant (mean ± std)",
                      fdir / "cold_ndcg10.png", color="C1")
    _plot_metric_bars(aggregated, "recall@10",
                      "Test Recall@10 by variant (mean ± std)",
                      fdir / "recall10.png", color="C2")
    _plot_training_curves(runs_by_variant, fdir / "training_curves.png")
    print(f"  wrote tables -> {tdir}")
    print(f"  wrote figures -> {fdir}")


def run_grid(variants: Optional[List[str]] = None,
             seeds: Optional[List[int]] = None,
             epochs: Optional[int] = None,
             verbose: bool = True) -> Dict:
    base = Config()
    if variants is None:
        variants = list(PRESETS.keys())
    if seeds is None:
        seeds = list(base.seeds)

    print(f"Grid: {len(variants)} variants x {len(seeds)} seeds = {len(variants)*len(seeds)} runs")
    print(f"  variants: {variants}")
    print(f"  seeds:    {seeds}")

    data = build_dataset(base)
    llm_emb = load_item_embeddings(data, base)
    print(f"  data: n_users={data.n_users} n_items={data.n_items} "
          f"train={len(data.train_df):,}")

    runs_by_variant: Dict[str, List[Dict]] = defaultdict(list)
    for variant in variants:
        for seed in seeds:
            cfg = PRESETS[variant]()
            cfg.seed = seed
            if epochs is not None:
                cfg.train.epochs = epochs
            result = train_and_eval(cfg, data, llm_emb, verbose=verbose)
            runs_by_variant[variant].append(result)

    aggregated = {v: aggregate_metrics([r["test"] for r in runs])
                  for v, runs in runs_by_variant.items()}

    save_results(runs_by_variant, aggregated, base)
    return {"runs": dict(runs_by_variant), "aggregated": aggregated}


def main() -> None:
    p = argparse.ArgumentParser(description="Run SemMF ablation grid.")
    p.add_argument("--variants", type=str, default="all",
                   help="comma-separated variant names, or 'all'")
    p.add_argument("--seeds", type=str, default="42,123,2024",
                   help="comma-separated seeds")
    p.add_argument("--epochs", type=int, default=None,
                   help="override train.epochs (useful for quick smoke runs)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    variants = list(PRESETS.keys()) if args.variants == "all" \
        else [v.strip() for v in args.variants.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    run_grid(variants=variants, seeds=seeds, epochs=args.epochs,
             verbose=not args.quiet)


if __name__ == "__main__":
    main()

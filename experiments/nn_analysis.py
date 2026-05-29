"""Nearest-neighbor analysis on the trained SemMF-MSE-Adaptive model (seed 42).

Retrains seed 42 (no checkpoint is saved), then reports top-5 cosine neighbors
of the V (item) embeddings for three anchor movies. For comparison, also
reports the neighbors in the raw LLM embedding space and in a freshly trained
BPR-MF baseline — this helps separate "the LLM put these next to each other"
from "SemMF's regularizer kept them close after CF training."
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.train import train


ANCHORS = [
    "Toy Story (1995)",
    "Pulp Fiction (1994)",
    "Matrix, The (1999)",  # ML-1M uses comma-prefix "Matrix, The" convention
]
K = 5


def _find_idx(item_text, anchor: str) -> int:
    for i, t in enumerate(item_text):
        if t.startswith(anchor):
            return i
    raise KeyError(f"no item starts with {anchor!r}")


def _topk_neighbors(sims: torch.Tensor, idx: int, k: int, item_text):
    row = sims[idx].clone()
    row[idx] = float("-inf")
    top = torch.topk(row, k=k)
    return [(float(s), item_text[int(i)]) for s, i in zip(top.values, top.indices)]


def main() -> None:
    cfg = PRESETS["semmf_mse_adaptive"]()
    cfg.seed = 42
    data = build_dataset(cfg)
    emb = load_item_embeddings(data, cfg)

    # also train BPR-MF for the baseline column
    bpr_cfg = PRESETS["bpr_mf"]()
    bpr_cfg.seed = 42

    print(f"Training semmf_mse_adaptive (seed 42)...", flush=True)
    t0 = time.time()
    sem_model, _ = train(cfg, data, emb, verbose=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print(f"Training bpr_mf (seed 42)...", flush=True)
    t0 = time.time()
    bpr_model, _ = train(bpr_cfg, data, emb, verbose=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    # Cosine-sim matrices
    V_sem = sem_model.V.weight.detach()
    V_bpr = bpr_model.V.weight.detach()
    E_llm = torch.from_numpy(emb).to(V_sem.device)

    sims_sem = F.normalize(V_sem, dim=-1) @ F.normalize(V_sem, dim=-1).t()
    sims_bpr = F.normalize(V_bpr, dim=-1) @ F.normalize(V_bpr, dim=-1).t()
    sims_llm = F.normalize(E_llm, dim=-1) @ F.normalize(E_llm, dim=-1).t()

    for anchor in ANCHORS:
        idx = _find_idx(data.item_text, anchor)
        print(f"\n\n=== {data.item_text[idx]} (idx {idx}, "
              f"{data.train_item_counts[idx]} train ratings) ===", flush=True)

        print("\n  [SemMF-MSE-Adaptive V]  top-5 by cosine:", flush=True)
        for s, t in _topk_neighbors(sims_sem, idx, K, data.item_text):
            print(f"    {s:+.4f}  {t}", flush=True)

        print("\n  [BPR-MF V]  top-5 by cosine:", flush=True)
        for s, t in _topk_neighbors(sims_bpr, idx, K, data.item_text):
            print(f"    {s:+.4f}  {t}", flush=True)

        print("\n  [Raw LLM e]  top-5 by cosine:", flush=True)
        for s, t in _topk_neighbors(sims_llm, idx, K, data.item_text):
            print(f"    {s:+.4f}  {t}", flush=True)


if __name__ == "__main__":
    main()

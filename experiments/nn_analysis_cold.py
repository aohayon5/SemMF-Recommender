"""Cold-anchor nearest-neighbor analysis.

Picks 3 cold items (train_count < 5) that appear in the test set, with distinct
primary genres, and reports top-5 cosine neighbors in:
  - SemMF-MSE-Adaptive V (regularizer-aligned)
  - BPR-MF V (regularizer-free: V[cold] should be ~random init)
  - Raw LLM e (just MiniLM)

This is the visualization that should belong in the paper — on cold items,
BPR-MF's V has no training signal, so its "neighbors" are essentially random
clusters formed by initialization noise. SemMF's regularizer pulls V[cold]
toward f(e[cold]), making its neighbors semantically meaningful.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.train import train


K = 5


def _primary_genre(text: str) -> str:
    if "Genres:" not in text:
        return ""
    after = text.split("Genres:", 1)[1].strip().rstrip(".")
    return after.split(",")[0].strip()


def _pick_cold_anchors(data, n: int = 3) -> list:
    """Cold items in test set with distinct primary genres.

    Prefers items with >=1 train rating (so V exists but is barely trained)
    and at least 1 positive in test (so it's a real cold-start case).
    """
    cold_idx = np.where(data.cold_item_mask)[0]
    test_items = set(int(x) for x in data.test_df["item"].unique())

    pos_test = data.test_df[data.test_df["rating"] >= 4.0]
    cold_with_pos_test = set(int(x) for x in pos_test["item"].unique())

    candidates = sorted(
        [int(i) for i in cold_idx if int(i) in test_items and int(i) in cold_with_pos_test],
        key=lambda i: (-data.train_item_counts[i], i),  # prefer 4>3>2>1>0, then lowest idx
    )

    seen_genres = set()
    picked = []
    for i in candidates:
        g = _primary_genre(data.item_text[i])
        if g and g not in seen_genres:
            seen_genres.add(g)
            picked.append(i)
            if len(picked) == n:
                break
    return picked


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

    bpr_cfg = PRESETS["bpr_mf"]()
    bpr_cfg.seed = 42

    print("Training semmf_mse_adaptive (seed 42)...", flush=True)
    t0 = time.time()
    sem_model, _ = train(cfg, data, emb, verbose=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    print("Training bpr_mf (seed 42)...", flush=True)
    t0 = time.time()
    bpr_model, _ = train(bpr_cfg, data, emb, verbose=False)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    V_sem = sem_model.V.weight.detach()
    V_bpr = bpr_model.V.weight.detach()
    E_llm = torch.from_numpy(emb).to(V_sem.device)

    sims_sem = F.normalize(V_sem, dim=-1) @ F.normalize(V_sem, dim=-1).t()
    sims_bpr = F.normalize(V_bpr, dim=-1) @ F.normalize(V_bpr, dim=-1).t()
    sims_llm = F.normalize(E_llm, dim=-1) @ F.normalize(E_llm, dim=-1).t()

    anchors = _pick_cold_anchors(data, n=3)
    print(f"\nPicked cold anchors: {anchors}", flush=True)

    for idx in anchors:
        n_train = int(data.train_item_counts[idx])
        n_pos_test = int(((data.test_df["item"] == idx) &
                          (data.test_df["rating"] >= 4.0)).sum())
        print(f"\n\n=== {data.item_text[idx]} (idx {idx}, "
              f"{n_train} train ratings, {n_pos_test} positive test ratings) ===",
              flush=True)

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

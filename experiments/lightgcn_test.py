"""LightGCN baseline via RecBole, evaluated through our pipeline.

Trains RecBole's stock LightGCN on our train split, extracts the propagated
user/item embeddings, wraps them in a SemMF-shaped adapter, and runs them
through our evaluate_all so the metrics are apples-to-apples with the rest of
the grid (same temporal test split, same full-rank eval, same cold-pool
restriction).
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import PRESETS
from src.data_loader import build_dataset
from src.embeddings import load_item_embeddings
from src.evaluate import evaluate_all

# Quiet RecBole/TF logs
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("recbole").setLevel(logging.WARNING)


def _write_inter_file(train_df: pd.DataFrame, path: Path) -> None:
    """Atomic-file format: header `field:type` then TSV rows."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for u, i, r, t in zip(
            train_df["user"].to_numpy(),
            train_df["item"].to_numpy(),
            train_df["rating"].to_numpy(),
            train_df["timestamp"].to_numpy(),
        ):
            f.write(f"{int(u)}\t{int(i)}\t{float(r)}\t{int(t)}\n")


def train_lightgcn(data, seed: int, epochs: int, work_dir: Path):
    """Train RecBole's LightGCN; return (model, recbole_dataset)."""
    # scipy >= 1.16 removed dok_matrix._update and blocks bulk .update(). RecBole 1.2.1
    # uses _update to populate the adjacency. Patch it back with explicit per-key assign.
    import scipy.sparse

    def _dok_update_compat(self, data_dict):
        for key, value in data_dict.items():
            self[key] = value

    scipy.sparse.dok_matrix._update = _dok_update_compat

    from recbole.config import Config as RBConfig
    from recbole.data import create_dataset, data_preparation
    from recbole.model.general_recommender import LightGCN
    from recbole.trainer import Trainer

    dataset_name = "semmf_ml1m"
    ds_path = work_dir / dataset_name
    ds_path.mkdir(parents=True, exist_ok=True)
    _write_inter_file(data.train_df, ds_path / f"{dataset_name}.inter")

    config_dict = {
        "dataset": dataset_name,
        "data_path": str(work_dir),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "eval_args": {
            "split": {"RS": [0.9, 0.05, 0.05]},
            "order": "RO",
            "group_by": "user",
            "mode": "full",
        },
        "epochs": epochs,
        "stopping_step": 5,
        "train_batch_size": 1024,
        "eval_batch_size": 4096,
        "seed": seed,
        "reproducibility": True,
        "state": "WARNING",
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "checkpoint_dir": str(work_dir / "ckpt"),
    }

    rb_cfg = RBConfig(model="LightGCN", config_dict=config_dict)
    dataset = create_dataset(rb_cfg)
    train_data, valid_data, _ = data_preparation(rb_cfg, dataset)
    model = LightGCN(rb_cfg, train_data._dataset).to(rb_cfg["device"])
    trainer = Trainer(rb_cfg, model)
    trainer.fit(train_data, valid_data, saved=False, show_progress=False, verbose=False)
    return model, dataset


def extract_embeddings(model, dataset, n_users: int, n_items: int):
    """Pull propagated user/item embeddings, remap RecBole indices -> our indices.

    RecBole reserves index 0 for [PAD]; valid indices start at 1. Our user/item
    integers are written into the .inter as tokens, so id2token at RecBole index
    `r` is the string of our integer index.
    """
    model.eval()
    with torch.no_grad():
        user_e, item_e = model.forward()
    user_e = user_e.cpu().numpy()
    item_e = item_e.cpu().numpy()
    dim = user_e.shape[1]

    U = np.zeros((n_users, dim), dtype=np.float32)
    V = np.zeros((n_items, dim), dtype=np.float32)
    for rb_idx, tok in enumerate(dataset.field2id_token[dataset.uid_field]):
        if tok == "[PAD]":
            continue
        U[int(tok)] = user_e[rb_idx]
    for rb_idx, tok in enumerate(dataset.field2id_token[dataset.iid_field]):
        if tok == "[PAD]":
            continue
        V[int(tok)] = item_e[rb_idx]
    return U, V


class _LightGCNAdapter(nn.Module):
    """Mimics SemMF's score_users_full / score_pairs for evaluate_all."""

    def __init__(self, U: np.ndarray, V: np.ndarray):
        super().__init__()
        self.U = nn.Parameter(torch.from_numpy(U), requires_grad=False)
        self.V = nn.Parameter(torch.from_numpy(V), requires_grad=False)

    def score_users_full(self, users: torch.Tensor, cold_mask=None) -> torch.Tensor:
        u = self.U[users]
        return u @ self.V.t()

    def score_pairs(self, users, items):
        return (self.U[users] * self.V[items]).sum(-1)


def run_one_seed(seed: int, data, cfg, work_dir: Path, epochs: int = 30) -> dict:
    print(f"\n=== LightGCN  seed={seed} (epochs<= {epochs}) ===", flush=True)
    t0 = time.time()
    model, ds = train_lightgcn(data, seed=seed, epochs=epochs, work_dir=work_dir)
    train_dur = time.time() - t0
    U, V = extract_embeddings(model, ds, data.n_users, data.n_items)
    device = cfg.resolve_device()
    adapter = _LightGCNAdapter(U, V).to(device)
    metrics = evaluate_all(adapter, data, cfg, split="test",
                           use_cold_substitution=False, include_rating=False)
    total_dur = time.time() - t0
    t = metrics
    print(f"  TEST ndcg@10={t['ndcg@10']:.4f}  recall@10={t['recall@10']:.4f}  "
          f"hr@10={t['hr@10']:.4f}  cold_ndcg@10={t['cold_ndcg@10']:.4f}  "
          f"(train {train_dur:.0f}s, total {total_dur:.0f}s)", flush=True)
    return metrics


def main() -> None:
    cfg = PRESETS["bpr_mf"]()
    cfg.seed = 42
    data = build_dataset(cfg)
    _ = load_item_embeddings(data, cfg)  # warm cache (not used by LightGCN itself)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        m = run_one_seed(seed=42, data=data, cfg=cfg, work_dir=work_dir, epochs=30)

    print(f"\n--- references (seed-42 single-seed; 3-seed mean) ---", flush=True)
    print(f"  bpr_mf              ndcg@10=0.2304 (s42); 0.2324 +/- 0.0015 (3s)", flush=True)
    print(f"  semmf_mse_adaptive  ndcg@10=0.2301 (s42); 0.2296 +/- 0.0031 (3s)", flush=True)
    print(f"  semmf_infonce       ndcg@10=0.2262 (s42); 0.2302 +/- 0.0024 (3s)", flush=True)


if __name__ == "__main__":
    main()

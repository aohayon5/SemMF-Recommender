"""Evaluation: full-rank NDCG/Recall/HR, RMSE/MAE, cold-start variant, multi-seed aggregation."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import Config
from src.data_loader import MovieLensData
from src.model import SemMF


def _ndcg_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    h = hits[:k]
    dcg = float((h / np.log2(np.arange(2, k + 2))).sum())
    n = min(k, n_relevant)
    if n == 0:
        return 0.0
    idcg = float((1.0 / np.log2(np.arange(2, n + 2))).sum())
    return dcg / idcg


@torch.no_grad()
def evaluate_ranking(
    model: SemMF,
    data: MovieLensData,
    cfg: Config,
    split: str = "test",
    cold_only: bool = False,
    use_cold_substitution: bool = False,
) -> Dict[str, float]:
    """Full-rank NDCG/Recall/HR over the user pool with positives in `split`.

    Train-observed items are masked from the ranking. If `cold_only`, the
    relevance set is restricted to cold items and users with no cold positives
    are skipped (so the metric is computed on a smaller subpopulation).
    `use_cold_substitution` swaps V[i] for f(e[i]) on cold items at scoring time.
    """
    model.eval()
    device = next(model.parameters()).device

    pos_dict = data.user_test_pos if split == "test" else data.user_val_pos
    obs_dict = data.user_train_obs

    cold_set: Optional[set] = None
    warm_mask_t: Optional[torch.Tensor] = None
    if cold_only:
        cold_set = set(int(i) for i in np.where(data.cold_item_mask)[0])
        warm_mask_t = torch.from_numpy(~data.cold_item_mask).to(device)

    cold_mask_t: Optional[torch.Tensor] = None
    if use_cold_substitution:
        cold_mask_t = torch.from_numpy(data.cold_item_mask).to(device)

    eval_users: List[int] = []
    rel_per_user: List[set] = []
    for u, pos in pos_dict.items():
        rel = set(int(p) for p in pos)
        if cold_set is not None:
            rel = rel & cold_set
        if rel:
            eval_users.append(int(u))
            rel_per_user.append(rel)

    K_values = list(cfg.eval.k_values)
    K_max = max(K_values)
    sums = {f"ndcg@{k}": 0.0 for k in K_values}
    sums.update({f"recall@{k}": 0.0 for k in K_values})
    sums.update({f"hr@{k}": 0.0 for k in K_values})
    n_eval = len(eval_users)
    if n_eval == 0:
        sums["n_users"] = 0
        return sums

    batch = cfg.eval.eval_batch_users
    for start in range(0, n_eval, batch):
        u_batch = eval_users[start:start + batch]
        users_t = torch.tensor(u_batch, dtype=torch.long, device=device)
        scores = model.score_users_full(users_t, cold_mask=cold_mask_t)

        for ridx, u in enumerate(u_batch):
            obs = obs_dict.get(u)
            if obs is not None and len(obs) > 0:
                scores[ridx, torch.from_numpy(obs).to(device)] = float("-inf")

        if warm_mask_t is not None:
            scores[:, warm_mask_t] = float("-inf")

        topk_idx = torch.topk(scores, k=K_max, dim=1).indices.cpu().numpy()

        for ridx, u in enumerate(u_batch):
            rel = rel_per_user[start + ridx]
            ranked = topk_idx[ridx]
            hits = np.fromiter((1 if int(i) in rel else 0 for i in ranked),
                               dtype=np.float32, count=len(ranked))
            n_rel = len(rel)
            for k in K_values:
                n_hits = int(hits[:k].sum())
                sums[f"recall@{k}"] += n_hits / n_rel
                sums[f"hr@{k}"] += 1.0 if n_hits > 0 else 0.0
                sums[f"ndcg@{k}"] += _ndcg_at_k(hits, n_rel, k)

    out = {k: v / n_eval for k, v in sums.items()}
    out["n_users"] = n_eval
    return out


@torch.no_grad()
def evaluate_rating(
    model: SemMF, data: MovieLensData, split: str = "test", chunk: int = 65536,
) -> Dict[str, float]:
    """RMSE / MAE over (user, item, rating) triples in `split`. For MSE-MF only."""
    model.eval()
    device = next(model.parameters()).device
    df = data.test_df if split == "test" else data.val_df
    users = torch.tensor(df["user"].to_numpy(), dtype=torch.long, device=device)
    items = torch.tensor(df["item"].to_numpy(), dtype=torch.long, device=device)
    ratings = torch.tensor(df["rating"].to_numpy(), dtype=torch.float32, device=device)
    preds = torch.empty_like(ratings)
    for s in range(0, len(users), chunk):
        e = s + chunk
        preds[s:e] = model.score_pairs(users[s:e], items[s:e])
    err = preds - ratings
    return {
        "rmse": float(torch.sqrt((err ** 2).mean()).item()),
        "mae": float(err.abs().mean().item()),
        "n": int(len(ratings)),
    }


def evaluate_all(
    model: SemMF, data: MovieLensData, cfg: Config,
    split: str = "test", use_cold_substitution: bool = False,
    include_rating: bool = False,
) -> Dict[str, float]:
    """One-call eval: ranking + cold-start ranking + (optional) rating metrics."""
    out: Dict[str, float] = {}
    rk = evaluate_ranking(model, data, cfg, split=split,
                          cold_only=False,
                          use_cold_substitution=use_cold_substitution)
    out.update({k: v for k, v in rk.items() if k != "n_users"})
    out["n_users"] = rk["n_users"]

    cold = evaluate_ranking(model, data, cfg, split=split,
                            cold_only=True,
                            use_cold_substitution=use_cold_substitution)
    out[f"cold_ndcg@{cfg.eval.cold_k}"] = cold[f"ndcg@{cfg.eval.cold_k}"]
    out[f"cold_recall@{cfg.eval.cold_k}"] = cold[f"recall@{cfg.eval.cold_k}"]
    out[f"cold_hr@{cfg.eval.cold_k}"] = cold[f"hr@{cfg.eval.cold_k}"]
    out["cold_n_users"] = cold["n_users"]

    if include_rating:
        rt = evaluate_rating(model, data, split=split)
        out["rmse"] = rt["rmse"]
        out["mae"] = rt["mae"]
    return out


def aggregate_metrics(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """List of per-seed metric dicts -> {metric: {'mean': m, 'std': s, 'n': n}}."""
    if not runs:
        return {}
    keys = set()
    for r in runs:
        keys.update(r.keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in runs if k in r], dtype=np.float64)
        if len(vals) == 0:
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "n": int(len(vals)),
        }
    return out

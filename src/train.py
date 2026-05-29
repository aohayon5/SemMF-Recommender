"""Training loop for SemMF: BPR/MSE objectives, semantic regularization (MSE / InfoNCE),
adaptive per-item lambda, early stopping on validation."""
from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from src.config import Config
from src.data_loader import MovieLensData, build_train_loader, user_positive_centroids
from src.embeddings import load_item_embeddings
from src.evaluate import evaluate_all, evaluate_rating
from src.model import SemMF, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: SemMF, cfg: Config):
    cls = AdamW if cfg.train.optimizer == "adamw" else Adam
    return cls(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def _lambda_per_item(cfg: Config, train_item_counts: np.ndarray,
                     device: torch.device) -> torch.Tensor:
    n = torch.from_numpy(train_item_counts).float().to(device)
    if cfg.reg.schedule == "fixed":
        return torch.full_like(n, cfg.reg.lambda_0)
    return cfg.reg.lambda_0 / (1.0 + cfg.reg.alpha * n)


def _semantic_loss(
    model: SemMF, cfg: Config, lambda_per_item: torch.Tensor,
    device: torch.device, infonce_batch: int = 512,
    warm_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Semantic regularizer over items. Cosine-normalized so lambda is scale-stable.

    If cfg.reg.decouple_f, splits the loss into two stop-gradient terms:
      L_v: V trained, f detached, adaptive lambda per item.
      L_f: f trained, V detached, uniform on warm items only, weight = gamma.
    Decoupling lets f learn from well-trained V[warm] without being dominated
    by V[cold] noise (which adaptive lambda over-weights).
    """
    f_e = model.project()
    if model.mcfg.item_vector_source == "free":
        V = model.V.weight
    else:
        V = f_e
    decouple = cfg.reg.decouple_f and model.mcfg.item_vector_source == "free"

    if cfg.reg.mode == "mse":
        Vn = F.normalize(V, dim=-1)
        fn = F.normalize(f_e, dim=-1)
        if not decouple:
            per_item = ((Vn - fn) ** 2).sum(-1)
            return (lambda_per_item * per_item).mean()
        # V-side: V trains toward sg(f), adaptive lambda
        per_v = ((Vn - fn.detach()) ** 2).sum(-1)
        loss_v = (lambda_per_item * per_v).mean()
        # f-side: f trains toward sg(V), warm items only, uniform
        per_f = ((fn - Vn.detach()) ** 2).sum(-1)
        if warm_mask is not None and warm_mask.any():
            loss_f = per_f[warm_mask].mean()
        else:
            loss_f = per_f.mean()
        return loss_v + cfg.reg.gamma * loss_f

    # InfoNCE: contrastive on a sampled item batch
    n_items = V.shape[0]
    B = min(infonce_batch, n_items)
    idx = torch.randperm(n_items, device=device)[:B]
    Vb = V[idx]
    fb = f_e[idx]
    if cfg.reg.infonce_normalize:
        Vb = F.normalize(Vb, dim=-1)
        fb = F.normalize(fb, dim=-1)
    targets = torch.arange(B, device=device)

    if not decouple:
        sims = (Vb @ fb.t()) / cfg.reg.tau
        per_item = F.cross_entropy(sims, targets, reduction="none")
        return (lambda_per_item[idx] * per_item).mean()

    # Decoupled InfoNCE
    sims_v = (Vb @ fb.detach().t()) / cfg.reg.tau
    per_v = F.cross_entropy(sims_v, targets, reduction="none")
    loss_v = (lambda_per_item[idx] * per_v).mean()
    sims_f = (fb @ Vb.detach().t()) / cfg.reg.tau
    per_f = F.cross_entropy(sims_f, targets, reduction="none")
    if warm_mask is not None:
        warm_in_batch = warm_mask[idx]
        if warm_in_batch.any():
            loss_f = per_f[warm_in_batch].mean()
        else:
            loss_f = per_f.mean()
    else:
        loss_f = per_f.mean()
    return loss_v + cfg.reg.gamma * loss_f


def _user_semantic_loss(
    model: SemMF, cfg: Config,
    centroids_llm: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Pull normalize(U_u) toward normalize(f(centroid_u).detach()) over valid users."""
    with torch.no_grad():
        target = model.f(centroids_llm)  # (n_users, k); f frozen for this loss
    Un = F.normalize(model.U.weight, dim=-1)
    Tn = F.normalize(target, dim=-1)
    per_user = ((Un - Tn) ** 2).sum(-1)
    return cfg.reg.gamma_u * per_user[valid_mask].mean()


def _val_metric(model: SemMF, data: MovieLensData, cfg: Config) -> Tuple[float, Dict[str, float]]:
    """Returns (early-stopping metric value, full metrics dict)."""
    name = cfg.train.early_stopping_metric
    if name == "rmse":
        m = evaluate_rating(model, data, split="val")
        return m["rmse"], m
    full = evaluate_all(model, data, cfg, split="val", use_cold_substitution=False)
    return full[name], full


def train(cfg: Config, data: MovieLensData,
          llm_emb: np.ndarray, verbose: bool = True) -> Tuple[SemMF, List[Dict]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.resolve_device())

    model = build_model(data.n_users, data.n_items, llm_emb, cfg).to(device)
    opt = _build_optimizer(model, cfg)
    lambda_per_item = _lambda_per_item(cfg, data.train_item_counts, device)
    use_reg = cfg.reg.mode != "none"
    warm_mask = torch.from_numpy(~data.cold_item_mask).to(device) if cfg.reg.decouple_f else None
    user_centroids_t: Optional[torch.Tensor] = None
    user_valid_t: Optional[torch.Tensor] = None
    if cfg.reg.user_reg:
        c_np, v_np = user_positive_centroids(data, llm_emb)
        user_centroids_t = torch.from_numpy(c_np).to(device)
        user_valid_t = torch.from_numpy(v_np).to(device)

    loader = build_train_loader(data, cfg, seed=cfg.seed)

    minimize = cfg.train.early_stopping_metric == "rmse"
    best = float("inf") if minimize else -float("inf")
    best_state = None
    patience_left = cfg.train.early_stopping_patience
    history: List[Dict] = []

    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        running, running_sem, n = 0.0, 0.0, 0
        for batch in loader:
            opt.zero_grad()
            if cfg.train.objective == "bpr":
                u, pos, neg = (t.to(device) for t in batch)
                pos_s = model.score_pairs(u, pos)
                neg_s = model.score_pairs(u, neg)
                main_loss = -F.logsigmoid(pos_s - neg_s).mean()
            else:
                u, i, r = (t.to(device) for t in batch)
                pred = model.score_pairs(u, i)
                main_loss = F.mse_loss(pred, r)

            if use_reg:
                sem = _semantic_loss(model, cfg, lambda_per_item, device,
                                     warm_mask=warm_mask)
                loss = main_loss + sem
                running_sem += float(sem.item())
            else:
                loss = main_loss

            if cfg.reg.user_reg and user_centroids_t is not None:
                user_sem = _user_semantic_loss(
                    model, cfg, user_centroids_t, user_valid_t,
                )
                loss = loss + user_sem
                running_sem += float(user_sem.item())

            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            running += float(main_loss.item())
            n += 1

        train_loss = running / max(n, 1)
        sem_loss = running_sem / max(n, 1)
        epoch_time = time.time() - t0

        if (epoch + 1) % cfg.train.eval_every == 0:
            metric, val_full = _val_metric(model, data, cfg)
            entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "sem_loss": sem_loss,
                "val_metric": metric,
                "epoch_time": epoch_time,
            }
            entry.update({f"val_{k}": v for k, v in val_full.items()
                          if isinstance(v, (int, float))})
            history.append(entry)

            improved = (metric < best) if minimize else (metric > best)
            if improved:
                best = metric
                best_state = copy.deepcopy(model.state_dict())
                patience_left = cfg.train.early_stopping_patience
            else:
                patience_left -= 1

            if verbose:
                tag = "*" if improved else " "
                print(f"  ep{epoch+1:>3d} {tag}  main={train_loss:.4f}  "
                      f"sem={sem_loss:.4f}  val_{cfg.train.early_stopping_metric}={metric:.4f}  "
                      f"({epoch_time:.1f}s, pat={patience_left})")

            if patience_left <= 0:
                if verbose:
                    print(f"  early stop at epoch {epoch+1}, best={best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_and_eval(cfg: Config, data: MovieLensData,
                   llm_emb: np.ndarray, verbose: bool = True) -> Dict:
    """Full pipeline: train, then evaluate on test with cold-substitution where applicable."""
    if verbose:
        print(f"\n=== {cfg.name}  seed={cfg.seed} ===")
    model, history = train(cfg, data, llm_emb, verbose=verbose)

    # We rely on the regularizer (during training) to align V[cold] with f(e[cold]).
    # Explicit substitution at inference was tested and hurt cold NDCG — V[cold]'s
    # regularizer-converged direction is more useful than f(e[cold])'s direction
    # because f is a shared projection trained to match warm V's, not cold V's.
    test_metrics = evaluate_all(model, data, cfg, split="test",
                                use_cold_substitution=False,
                                include_rating=(cfg.train.objective == "mse"))
    if verbose:
        print(f"  TEST: ndcg@10={test_metrics.get('ndcg@10', 0):.4f}  "
              f"recall@10={test_metrics.get('recall@10', 0):.4f}  "
              f"cold_ndcg@10={test_metrics.get(f'cold_ndcg@{cfg.eval.cold_k}', 0):.4f}")
    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "test": test_metrics,
        "history": history,
        "best_val": history[-1]["val_metric"] if history else None,
    }

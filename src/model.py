"""SemMF: Matrix Factorization with optional semantic regularization via LLM projection.

Single nn.Module that supports all 7 ablation variants through config flags:
  - free V (BPR-MF, MSE-MF, SemMF-*) vs projected V (LLM-Only)
  - random init vs LLM-PCA init (LLM-Init)
  - regularizer applied externally in train.py
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.config import Config, ModelConfig


class Projection(nn.Module):
    """f: R^embed_dim -> R^k. Linear or 2-layer MLP, with optional LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, mcfg: ModelConfig):
        super().__init__()
        layers: list[nn.Module] = []
        if mcfg.projection_type == "linear":
            layers.append(nn.Linear(in_dim, out_dim))
        elif mcfg.projection_type == "mlp":
            h = mcfg.projection_hidden
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            if mcfg.projection_dropout > 0:
                layers.append(nn.Dropout(mcfg.projection_dropout))
            layers.append(nn.Linear(h, out_dim))
        else:
            raise ValueError(f"Unknown projection_type: {mcfg.projection_type}")
        if mcfg.use_layernorm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemMF(nn.Module):
    def __init__(
        self, n_users: int, n_items: int,
        llm_embeddings: np.ndarray, cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        self.mcfg = cfg.model
        self.n_users = n_users
        self.n_items = n_items
        k = cfg.model.latent_dim

        self.U = nn.Embedding(n_users, k)
        nn.init.normal_(self.U.weight, std=cfg.model.init_std)

        if cfg.model.architecture == "hybrid":
            self.k_sem = cfg.model.k_sem
            self.k_cf = k - self.k_sem
            if self.k_cf <= 0:
                raise ValueError(
                    f"hybrid architecture needs k_cf > 0; got latent_dim={k}, "
                    f"k_sem={self.k_sem}"
                )
            self.V_cf = nn.Embedding(n_items, self.k_cf)
            nn.init.normal_(self.V_cf.weight, std=cfg.model.init_std)
            self.V = None  # not used in hybrid mode
        else:
            self.V = nn.Embedding(n_items, k)
            nn.init.normal_(self.V.weight, std=cfg.model.init_std)

        self.b_u = nn.Parameter(torch.zeros(n_users)) if cfg.model.use_user_bias else None
        self.b_i = nn.Parameter(torch.zeros(n_items)) if cfg.model.use_item_bias else None
        self.mu = nn.Parameter(torch.zeros(())) if cfg.model.use_global_mean else None

        llm = torch.as_tensor(llm_embeddings, dtype=torch.float32)
        self.register_buffer("llm_emb", llm, persistent=False)
        # f's output dim depends on architecture: hybrid uses k_sem, others use k.
        f_out_dim = self.k_sem if cfg.model.architecture == "hybrid" else k
        self.f = Projection(llm.shape[1], f_out_dim, cfg.model)

        if cfg.model.init_from_llm:
            self._init_v_from_llm(llm_embeddings, cfg.model.init_std)

    @torch.no_grad()
    def _init_v_from_llm(self, llm_np: np.ndarray, target_std: float) -> None:
        from sklearn.decomposition import PCA
        k = self.mcfg.latent_dim
        pca = PCA(n_components=k, random_state=0)
        v_init = pca.fit_transform(llm_np).astype(np.float32)
        scale = target_std / (v_init.std() + 1e-12)
        v_init *= scale
        self.V.weight.data.copy_(torch.from_numpy(v_init))

    def project(self) -> torch.Tensor:
        return self.f(self.llm_emb)

    def item_matrix(self, cold_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Effective (n_items, k) matrix for scoring.

        Hybrid: returns [V_cf ; f(e)] (cold_mask ignored — semantic half handles cold).
        Projection: returns f(e) for all items.
        Free: returns V, optionally substituting f(e) where cold_mask is True.
        """
        if self.mcfg.architecture == "hybrid":
            return torch.cat([self.V_cf.weight, self.project()], dim=-1)
        if self.mcfg.item_vector_source == "projection":
            return self.project()
        V = self.V.weight
        if cold_mask is None:
            return V
        f_e = self.project()
        warm = ~cold_mask
        if warm.any():
            target = V[warm].norm(dim=-1).median().clamp_min(1e-6)
            f_norms = f_e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            f_e = f_e / f_norms * target
        return torch.where(cold_mask.unsqueeze(-1), f_e, V)

    def _user_vec(self, users: torch.Tensor) -> torch.Tensor:
        return self.U(users)

    def _item_vec(self, items: torch.Tensor) -> torch.Tensor:
        if self.mcfg.architecture == "hybrid":
            v_cf = self.V_cf(items)
            v_sem = self.f(self.llm_emb[items])
            return torch.cat([v_cf, v_sem], dim=-1)
        if self.mcfg.item_vector_source == "projection":
            return self.f(self.llm_emb[items])
        return self.V(items)

    def _add_biases(self, scores: torch.Tensor,
                    users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if self.mu is not None:
            scores = scores + self.mu
        if self.b_u is not None:
            scores = scores + self.b_u[users]
        if self.b_i is not None:
            scores = scores + self.b_i[items]
        return scores

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self._user_vec(users)
        v = self._item_vec(items)
        s = (u * v).sum(-1)
        return self._add_biases(s, users, items)

    def score_users_full(
        self, users: torch.Tensor, cold_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score given users against ALL items. Returns (len(users), n_items)."""
        u = self._user_vec(users)
        V = self.item_matrix(cold_mask)
        scores = u @ V.t()
        if self.mu is not None:
            scores = scores + self.mu
        if self.b_u is not None:
            scores = scores + self.b_u[users].unsqueeze(-1)
        if self.b_i is not None:
            scores = scores + self.b_i.unsqueeze(0)
        return scores


def build_model(n_users: int, n_items: int, llm_embeddings: np.ndarray,
                cfg: Config) -> SemMF:
    return SemMF(n_users, n_items, llm_embeddings, cfg)

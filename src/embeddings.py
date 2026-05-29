"""Sentence-transformer pipeline for item embeddings, with disk cache.

Cache key includes model name and an MD5 of all item texts, so changing the
text template or item set automatically invalidates the cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import numpy as np

from src.config import Config
from src.data_loader import MovieLensData


def _cache_path(cfg: Config, item_text: List[str]) -> Path:
    h = hashlib.md5()
    h.update(cfg.embedding.model_name.encode())
    for t in item_text:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    key = h.hexdigest()[:12]
    safe_model = cfg.embedding.model_name.replace("/", "__")
    return cfg.paths.embeddings / f"{safe_model}_{key}.npy"


def encode_texts(texts: List[str], cfg: Config) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = cfg.resolve_device()
    model = SentenceTransformer(cfg.embedding.model_name, device=device)
    model.max_seq_length = cfg.embedding.max_seq_length
    safe_texts = [t if t.strip() else "movie" for t in texts]
    emb = model.encode(
        safe_texts,
        batch_size=cfg.embedding.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)


def load_item_embeddings(
    data: MovieLensData, cfg: Config, force: bool = False
) -> np.ndarray:
    """Returns shape (n_items, embed_dim) float32, indexed by item_idx."""
    cfg.make_dirs()
    path = _cache_path(cfg, data.item_text)
    if path.exists() and not force:
        emb = np.load(path)
        if emb.shape == (data.n_items, cfg.embedding.embed_dim):
            return emb

    emb = encode_texts(data.item_text, cfg)
    if emb.shape[1] != cfg.embedding.embed_dim:
        raise ValueError(
            f"Encoder returned dim {emb.shape[1]}, "
            f"config expects {cfg.embedding.embed_dim}. "
            f"Update cfg.embedding.embed_dim to match the model."
        )
    np.save(path, emb)
    return emb


def enrich_with_tmdb(item_text: List[str], movies, api_key: str) -> List[str]:
    """Stretch goal: append TMDB plot summary to each item's text. Not implemented."""
    raise NotImplementedError(
        "TMDB enrichment is a Day-3 stretch goal; not implemented in v1."
    )

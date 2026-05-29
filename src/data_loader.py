"""MovieLens-1M loader: parse, temporal split, cold-item flags, BPR/MSE datasets."""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config, DataConfig

_TITLE_YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


def _split_title(raw: str) -> Tuple[str, str]:
    m = _TITLE_YEAR_RE.match(raw)
    if m:
        return m.group(1).strip(), m.group(2)
    return raw.strip(), ""


def _read_dat(path: Path, columns: List[str]) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) == len(columns):
                rows.append(parts)
    return pd.DataFrame(rows, columns=columns)


def load_raw(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = _read_dat(raw_dir / "ratings.dat",
                        ["user_id", "movie_id", "rating", "timestamp"])
    ratings = ratings.astype({"user_id": int, "movie_id": int,
                              "rating": float, "timestamp": int})

    movies = _read_dat(raw_dir / "movies.dat", ["movie_id", "title", "genres"])
    movies["movie_id"] = movies["movie_id"].astype(int)
    titles, years = zip(*movies["title"].map(_split_title))
    movies["title_clean"] = titles
    movies["year"] = years
    movies["genres_list"] = movies["genres"].str.split("|")

    users = _read_dat(raw_dir / "users.dat",
                      ["user_id", "gender", "age", "occupation", "zip"])
    users["user_id"] = users["user_id"].astype(int)
    return ratings, movies, users


def temporal_split(
    ratings: pd.DataFrame, train_frac: float, val_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(ratings)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = ratings.iloc[:n_train].copy()
    val = ratings.iloc[n_train:n_train + n_val].copy()
    test = ratings.iloc[n_train + n_val:].copy()
    return train, val, test


def build_text(movies: pd.DataFrame, n_items: int, template: str) -> List[str]:
    text = [""] * n_items
    for _, row in movies.iterrows():
        idx = row["item_idx"]
        if pd.isna(idx):
            continue
        idx = int(idx)
        genres = ", ".join(g for g in row["genres_list"] if g)
        text[idx] = template.format(
            title=row["title_clean"], year=row["year"], genres=genres
        )
    return text


@dataclass
class MovieLensData:
    n_users: int
    n_items: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame
    item_text: List[str]
    train_item_counts: np.ndarray
    cold_item_mask: np.ndarray
    user_train_obs: Dict[int, np.ndarray] = field(repr=False)
    user_train_pos: Dict[int, np.ndarray] = field(repr=False)
    user_val_pos: Dict[int, np.ndarray] = field(repr=False)
    user_test_pos: Dict[int, np.ndarray] = field(repr=False)
    user_id_map: Dict[int, int] = field(repr=False)
    item_id_map: Dict[int, int] = field(repr=False)


def _group_to_dict(df: pd.DataFrame, key: str, val: str) -> Dict[int, np.ndarray]:
    return {int(k): g[val].to_numpy(dtype=np.int64)
            for k, g in df.groupby(key, sort=False)}


def _cache_key(cfg: DataConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def build_dataset(cfg: Config, force_rebuild: bool = False) -> MovieLensData:
    cfg.make_dirs()
    cache = cfg.paths.processed / f"movielens_{_cache_key(cfg.data)}.pkl"
    if cache.exists() and not force_rebuild:
        with open(cache, "rb") as f:
            return pickle.load(f)

    ratings, movies, users = load_raw(cfg.paths.raw_data)

    counts = ratings.groupby("user_id").size()
    keep_users = counts[counts >= cfg.data.min_user_interactions].index
    ratings = ratings[ratings["user_id"].isin(keep_users)].reset_index(drop=True)

    train_raw, val_raw, test_raw = temporal_split(
        ratings, cfg.data.train_frac, cfg.data.val_frac
    )

    train_users = set(train_raw["user_id"].unique())
    val_raw = val_raw[val_raw["user_id"].isin(train_users)].reset_index(drop=True)
    test_raw = test_raw[test_raw["user_id"].isin(train_users)].reset_index(drop=True)

    user_id_map = {uid: i for i, uid in enumerate(sorted(train_users))}
    all_movie_ids = sorted(movies["movie_id"].unique())
    item_id_map = {mid: i for i, mid in enumerate(all_movie_ids)}
    n_users = len(user_id_map)
    n_items = len(item_id_map)

    movies = movies.copy()
    movies["item_idx"] = movies["movie_id"].map(item_id_map)
    users = users.copy()
    users["user_idx"] = users["user_id"].map(user_id_map)

    def _index(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "user": df["user_id"].map(user_id_map).astype("int64"),
            "item": df["movie_id"].map(item_id_map).astype("int64"),
            "rating": df["rating"].astype("float32"),
            "timestamp": df["timestamp"].astype("int64"),
        })
        return out.dropna().reset_index(drop=True)

    train_df = _index(train_raw)
    val_df = _index(val_raw)
    test_df = _index(test_raw)

    train_item_counts = np.zeros(n_items, dtype=np.int64)
    vc = train_df["item"].value_counts()
    train_item_counts[vc.index.to_numpy()] = vc.to_numpy()
    cold_item_mask = train_item_counts < cfg.data.cold_item_threshold

    thr = cfg.data.implicit_threshold
    user_train_obs = _group_to_dict(train_df, "user", "item")
    user_train_pos = _group_to_dict(train_df[train_df["rating"] >= thr], "user", "item")
    user_val_pos = _group_to_dict(val_df[val_df["rating"] >= thr], "user", "item")
    user_test_pos = _group_to_dict(test_df[test_df["rating"] >= thr], "user", "item")

    item_text = build_text(movies, n_items, cfg.embedding.text_template)

    data = MovieLensData(
        n_users=n_users,
        n_items=n_items,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        movies=movies,
        users=users,
        item_text=item_text,
        train_item_counts=train_item_counts,
        cold_item_mask=cold_item_mask,
        user_train_obs=user_train_obs,
        user_train_pos=user_train_pos,
        user_val_pos=user_val_pos,
        user_test_pos=user_test_pos,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
    )
    with open(cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


# === PyTorch datasets ===

class BPRDataset(Dataset):
    """Yields (user, pos_item, neg_item). Negatives sampled with rejection per-call."""

    def __init__(self, data: MovieLensData, num_negatives: int = 1, seed: int = 0):
        self.n_items = data.n_items
        self.num_negatives = num_negatives
        self._obs_sets: Dict[int, set] = {
            u: set(items.tolist()) for u, items in data.user_train_obs.items()
        }
        users, items = [], []
        for u, pos in data.user_train_pos.items():
            users.extend([u] * len(pos))
            items.extend(int(p) for p in pos)
        self.users = np.asarray(users, dtype=np.int64)
        self.pos_items = np.asarray(items, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users) * self.num_negatives

    def __getitem__(self, idx: int):
        i = idx % len(self.users)
        u = int(self.users[i])
        pos = int(self.pos_items[i])
        obs = self._obs_sets[u]
        neg = int(self.rng.integers(0, self.n_items))
        while neg in obs:
            neg = int(self.rng.integers(0, self.n_items))
        return u, pos, neg


class MSEDataset(Dataset):
    """Yields (user, item, rating) triples for explicit-feedback training."""

    def __init__(self, data: MovieLensData):
        self.users = data.train_df["user"].to_numpy(dtype=np.int64)
        self.items = data.train_df["item"].to_numpy(dtype=np.int64)
        self.ratings = data.train_df["rating"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.ratings[idx]


def user_positive_centroids(
    data: MovieLensData, llm_emb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-user mean LLM embedding over training positives.

    Returns (centroids, valid_mask):
      centroids: (n_users, embed_dim) — mean of llm_emb over user's training positives.
                 Rows for users with no positives are zero (and masked out).
      valid_mask: (n_users,) bool — True for users with at least one training positive.
    """
    n_users = data.n_users
    dim = llm_emb.shape[1]
    centroids = np.zeros((n_users, dim), dtype=np.float32)
    valid = np.zeros(n_users, dtype=bool)
    for u, pos in data.user_train_pos.items():
        if len(pos) == 0:
            continue
        centroids[u] = llm_emb[pos].mean(axis=0)
        valid[u] = True
    return centroids, valid


def build_train_loader(data: MovieLensData, cfg: Config, seed: int = 0) -> DataLoader:
    if cfg.train.objective == "bpr":
        ds: Dataset = BPRDataset(data, num_negatives=cfg.train.num_negatives, seed=seed)
    else:
        ds = MSEDataset(data)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

"""Central configuration for SemMF. All hyperparameters live here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    project_root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "raw" / "ml-1m"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    embeddings: Path = PROJECT_ROOT / "data" / "embeddings"
    results_tables: Path = PROJECT_ROOT / "results" / "tables"
    results_figures: Path = PROJECT_ROOT / "results" / "figures"
    checkpoints: Path = PROJECT_ROOT / "results" / "checkpoints"


@dataclass
class DataConfig:
    implicit_threshold: float = 4.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_strategy: Literal["temporal_global", "temporal_user"] = "temporal_global"
    cold_item_threshold: int = 5
    min_user_interactions: int = 5


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    batch_size: int = 64
    max_seq_length: int = 128
    text_template: str = "{title} ({year}). Genres: {genres}."
    use_tmdb: bool = False


@dataclass
class ModelConfig:
    latent_dim: int = 64
    projection_type: Literal["linear", "mlp"] = "linear"
    projection_hidden: int = 256
    projection_dropout: float = 0.0
    use_layernorm: bool = True
    use_user_bias: bool = True
    use_item_bias: bool = True
    use_global_mean: bool = True
    init_std: float = 0.01
    item_vector_source: Literal["free", "projection"] = "free"
    init_from_llm: bool = False
    # Hybrid architecture: v_i = [v_cf_i (k_cf dims) ; f(e_i) (k_sem dims)],
    # total scoring dim = latent_dim, k_cf = latent_dim - k_sem.
    architecture: Literal["mf", "hybrid"] = "mf"
    k_sem: int = 32


@dataclass
class TrainConfig:
    objective: Literal["mse", "bpr"] = "bpr"
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: Literal["adam", "adamw"] = "adam"
    num_negatives: int = 1
    early_stopping_patience: int = 5
    early_stopping_metric: str = "ndcg@10"
    grad_clip: float = 5.0
    eval_every: int = 1


@dataclass
class RegConfig:
    mode: Literal["none", "mse", "infonce"] = "mse"
    schedule: Literal["fixed", "adaptive"] = "adaptive"
    lambda_0: float = 0.5  # tuned on seed 42; sweep showed 0.5 best of {0.01, 0.1, 0.5, 1, 10, 100, 1000}
    alpha: float = 0.1
    tau: float = 0.1
    infonce_normalize: bool = True
    # When True, V's pull (adaptive lambda) and f's training (uniform on warm items)
    # are decoupled via stop-gradient. Lets f learn from well-trained V[warm] without
    # being dominated by V[cold] noise (which adaptive lambda over-weights).
    decouple_f: bool = False
    gamma: float = 1.0
    # User-side semantic regularization: pulls U_u toward f(centroid of u's positive
    # items in LLM space). f is detached so this loss only updates U.
    user_reg: bool = False
    gamma_u: float = 0.1


@dataclass
class EvalConfig:
    k_values: Tuple[int, ...] = (5, 10, 20)
    cold_k: int = 10
    full_rank: bool = True
    eval_batch_users: int = 256
    primary_metric: str = "ndcg@10"


@dataclass
class Config:
    name: str = "semmf"
    seed: int = 42
    seeds: Tuple[int, ...] = (42, 123, 2024)
    device: str = "auto"
    num_workers: int = 0  # Windows: process spawn cost outweighs gain at this scale

    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def make_dirs(self) -> None:
        for p in [
            self.paths.processed,
            self.paths.embeddings,
            self.paths.results_tables,
            self.paths.results_figures,
            self.paths.checkpoints,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


# === Experiment presets — one per ablation row ===

def preset_mse_mf() -> Config:
    cfg = Config(name="mse_mf")
    cfg.train.objective = "mse"
    cfg.train.early_stopping_metric = "rmse"
    cfg.reg.mode = "none"
    return cfg


def preset_bpr_mf() -> Config:
    cfg = Config(name="bpr_mf")
    cfg.train.objective = "bpr"
    cfg.reg.mode = "none"
    return cfg


def preset_llm_only() -> Config:
    """Item vector = f(e_i); no free V. Only U and the projection train."""
    cfg = Config(name="llm_only")
    cfg.train.objective = "bpr"
    cfg.reg.mode = "none"
    cfg.model.item_vector_source = "projection"
    return cfg


def preset_llm_init() -> Config:
    """V initialized from PCA-projected LLM embeddings, then trains freely."""
    cfg = Config(name="llm_init")
    cfg.train.objective = "bpr"
    cfg.reg.mode = "none"
    cfg.model.init_from_llm = True
    return cfg


def preset_semmf_mse_fixed() -> Config:
    cfg = Config(name="semmf_mse_fixed")
    cfg.train.objective = "bpr"
    cfg.reg.mode = "mse"
    cfg.reg.schedule = "fixed"
    return cfg


def preset_semmf_mse_adaptive() -> Config:
    cfg = Config(name="semmf_mse_adaptive")
    cfg.train.objective = "bpr"
    cfg.reg.mode = "mse"
    cfg.reg.schedule = "adaptive"
    return cfg


def preset_semmf_infonce() -> Config:
    cfg = Config(name="semmf_infonce")
    cfg.train.objective = "bpr"
    cfg.reg.mode = "infonce"
    cfg.reg.schedule = "adaptive"
    cfg.reg.tau = 0.01  # tuned on seed 42; sweep showed 0.01 best of {0.01, 0.05, 0.1, 0.3, 1.0}
    return cfg


def preset_semmf_hybrid() -> Config:
    """Concatenated latent space: v_i = [V_cf[i] ; f(e_i)], score = u . v.

    BPR trains everything: U, V_cf, and f (via the f(e_i) entries flowing into the
    score). No separate regularizer — semantic info enters through v_sem dims, and
    the user vector self-partitions to use both halves.
    """
    cfg = Config(name="semmf_hybrid")
    cfg.train.objective = "bpr"
    cfg.model.architecture = "hybrid"
    cfg.model.k_sem = 32  # k_cf = latent_dim - k_sem = 32
    cfg.reg.mode = "none"
    return cfg


PRESETS = {
    "mse_mf": preset_mse_mf,
    "bpr_mf": preset_bpr_mf,
    "llm_only": preset_llm_only,
    "llm_init": preset_llm_init,
    "semmf_mse_fixed": preset_semmf_mse_fixed,
    "semmf_mse_adaptive": preset_semmf_mse_adaptive,
    "semmf_infonce": preset_semmf_infonce,
    "semmf_hybrid": preset_semmf_hybrid,
}

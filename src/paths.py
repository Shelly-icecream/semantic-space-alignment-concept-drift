"""
Central layout: all artifact paths are relative to the repository root.

Override the root with environment variable HDSAD_ROOT if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- relative segments (repository root is the anchor) ---
REL_ALIGNMENT = "result/alignment"
REL_WORD = "result/word"
REL_JOINT_PCA = "result/joint_pca"
REL_CLUSTER = "result/cluster"
REL_GEOMETRIC = "result/geometric"

REL_FIG_ALIGNMENT = "figures/alignment"
REL_FIG_WORD = "figures/word"
REL_FIG_CLUSTER = "figures/cluster"
REL_FIG_GEOMETRIC = "figures/geometric"

# --- KeyedVectors filenames (under result/alignment/) ---
KV_RENMIN = "renmin_fast.kv"
KV_WEIBO_RAW = "weibo_fast.kv"
KV_WEIBO_ALIGNED = "weibo_aligned_v2.kv"
KV_ANCHOR_RENMIN_SORTED = "renmin_anchor_sorted.kv"

# --- common data filenames ---
SEMANTIC_SPACE_PCA_NPZ = "semantic_space_pca.npz"
SEMANTIC_EMBEDDING_CLUSTER_NPZ = "semantic_embedding_cluster.npz"
CLUSTER_WORDS_CSV = "cluster_words_by_category.csv"
SHIFT_RESULTS_CSV = "shift_results_v2.csv"
SHIFT_META_JSON = "shift_meta.json"


def repo_root() -> Path:
    env = os.environ.get("HDSAD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def alignment_dir() -> Path:
    return _mkdir(repo_root() / REL_ALIGNMENT)


def word_dir() -> Path:
    return _mkdir(repo_root() / REL_WORD)


def joint_pca_dir() -> Path:
    return _mkdir(repo_root() / REL_JOINT_PCA)


def cluster_dir() -> Path:
    return _mkdir(repo_root() / REL_CLUSTER)


def geometric_dir() -> Path:
    return _mkdir(repo_root() / REL_GEOMETRIC)


def fig_alignment_dir() -> Path:
    return _mkdir(repo_root() / REL_FIG_ALIGNMENT)


def fig_word_dir() -> Path:
    return _mkdir(repo_root() / REL_FIG_WORD)


def fig_cluster_dir() -> Path:
    return _mkdir(repo_root() / REL_FIG_CLUSTER)


def fig_geometric_dir() -> Path:
    return _mkdir(repo_root() / REL_FIG_GEOMETRIC)


def alignment_kv(name: str) -> Path:
    return alignment_dir() / name


def joint_pca_npz() -> Path:
    return joint_pca_dir() / SEMANTIC_SPACE_PCA_NPZ


def cluster_npz() -> Path:
    return cluster_dir() / SEMANTIC_EMBEDDING_CLUSTER_NPZ


def cluster_csv(name: str) -> Path:
    return cluster_dir() / name


def geometric_csv(name: str) -> Path:
    return geometric_dir() / name


def geometric_npz(name: str) -> Path:
    return geometric_dir() / name


def word_csv(name: str) -> Path:
    return word_dir() / name

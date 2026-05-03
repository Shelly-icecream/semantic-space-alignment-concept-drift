"""Geometric level — orthogonal Procrustes residual on PCA-50 (intersection words)."""

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv

import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

import paths

KV_WEIBO = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)
KV_RENMIN = paths.alignment_kv(paths.KV_RENMIN)
OUT_SUMMARY = paths.geometric_csv("geometric_space_procrustes_summary.csv")
OUT_NPZ = paths.geometric_npz("geometric_space_procrustes_residual_heatmap.npz")

RANDOM_STATE = 42
MAX_WORDS = 12_000
HEATMAP_ROWS = 400


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + eps)


def orthogonal_procrustes_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = a.T @ b
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def main() -> None:
    paths.geometric_dir()
    rng = np.random.default_rng(RANDOM_STATE)

    print("Loading KeyedVectors (mmap)...")
    kv_w = KeyedVectors.load(str(KV_WEIBO), mmap="r")
    kv_r = KeyedVectors.load(str(KV_RENMIN), mmap="r")

    sw = set(kv_w.key_to_index.keys())
    sr = set(kv_r.key_to_index.keys())
    common = sorted(sw & sr)
    if len(common) < 10:
        raise ValueError("Too few intersection words for Procrustes.")

    n_take = min(len(common), MAX_WORDS)
    if len(common) > n_take:
        idx = rng.choice(len(common), size=n_take, replace=False)
        words = [common[i] for i in idx]
    else:
        words = common

    print(f"Intersection |V|: {len(common)}; using n={len(words)} for Procrustes.")

    W = np.stack([kv_w[w] for w in words], axis=0).astype(np.float64)
    Rm = np.stack([kv_r[w] for w in words], axis=0).astype(np.float64)
    W = l2_normalize_rows(W)
    Rm = l2_normalize_rows(Rm)

    print("Fitting joint PCA(50) on full corpora (same pipeline as joint_pca/semantic_space_pca.py)...")
    W_all = l2_normalize_rows(kv_w.vectors.astype(np.float64))
    R_all = l2_normalize_rows(kv_r.vectors.astype(np.float64))
    X_all = np.concatenate([W_all, R_all], axis=0)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=RANDOM_STATE)
    pca.fit(X_all)

    Zw = pca.transform(W)
    Zr = pca.transform(Rm)

    Zw_c = Zw - Zw.mean(axis=0, keepdims=True)
    Zr_c = Zr - Zr.mean(axis=0, keepdims=True)

    orth = orthogonal_procrustes_r(Zw_c, Zr_c)
    aligned = Zw_c @ orth
    resid = Zr_c - aligned

    frob_res = float(np.sum(resid**2))
    frob_b = float(np.sum(Zr_c**2))
    rel = frob_res / max(frob_b, 1e-30)

    mean_abs_per_dim = np.mean(np.abs(resid), axis=0).astype(np.float32)

    row_l2 = np.linalg.norm(resid, axis=1)
    order = np.argsort(-row_l2)
    top = order[:HEATMAP_ROWS]
    heat = np.abs(resid[top])

    np.savez_compressed(
        str(OUT_NPZ),
        residual_abs=heat.astype(np.float32),
        row_l2_top=row_l2[top].astype(np.float32),
        words=np.array([words[i] for i in top], dtype=object),
        pca_dim=np.int32(Zw.shape[1]),
        mean_abs_residual_per_dim=mean_abs_per_dim,
    )

    summary = {
        "space": "pca50_joint_fit",
        "n_intersection_vocab": len(common),
        "n_words_used": len(words),
        "heatmap_rows": HEATMAP_ROWS,
        "pca_dim": Zw.shape[1],
        "random_state": RANDOM_STATE,
        "frobenius_residual_sq": frob_res,
        "relative_frobenius_residual_sq": rel,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("Wrote:", OUT_SUMMARY)
    print("Wrote:", OUT_NPZ)


if __name__ == "__main__":
    main()

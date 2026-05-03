from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

import paths


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + eps)


def main() -> None:
    kv_weibo = KeyedVectors.load(
        str(paths.alignment_kv(paths.KV_WEIBO_ALIGNED)), mmap="r"
    )
    kv_renmin = KeyedVectors.load(
        str(paths.alignment_kv(paths.KV_RENMIN)), mmap="r"
    )

    word_matrix_weibo = l2_normalize_rows(kv_weibo.vectors.astype(np.float64))
    word_matrix_renmin = l2_normalize_rows(kv_renmin.vectors.astype(np.float64))
    word_matrix = np.concatenate([word_matrix_weibo, word_matrix_renmin], axis=0)

    n_weibo = kv_weibo.vectors.shape[0]
    n_renmin = kv_renmin.vectors.shape[0]

    pca = PCA(n_components=50, svd_solver="randomized", random_state=42)
    pca.fit(word_matrix)
    Z_all = pca.transform(word_matrix)

    Z_weibo = pca.transform(word_matrix_weibo)
    Z_renmin = pca.transform(word_matrix_renmin)
    head = Z_all[:n_weibo]
    tail = Z_all[n_weibo : n_weibo + n_renmin]

    match_weibo = np.allclose(head, Z_weibo, rtol=1e-5, atol=1e-6)
    match_renmin = np.allclose(tail, Z_renmin, rtol=1e-5, atol=1e-6)
    err_weibo = np.max(np.abs(head - Z_weibo)) if head.size else 0.0
    err_renmin = np.max(np.abs(tail - Z_renmin)) if tail.size else 0.0

    print("PCA self-check (concat transform vs split transform):")
    print(f"  weibo block match: {match_weibo}, max |diff| = {err_weibo:.3e}")
    print(f"  renmin block match: {match_renmin}, max |diff| = {err_renmin:.3e}")
    if not (match_weibo and match_renmin):
        raise AssertionError("Self-check failed: Z_all slices != Z_weibo / Z_renmin")

    out = paths.joint_pca_npz()
    paths.joint_pca_dir()
    np.savez(
        str(out),
        Z_all=Z_all,
        n_weibo=n_weibo,
        n_renmin=n_renmin,
    )
    print("Wrote:", out)


if __name__ == "__main__":
    main()

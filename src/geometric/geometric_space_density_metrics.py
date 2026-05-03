"""Geometric level — space density (PCA-50 pairwise distances in a subsample)."""

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv

import numpy as np

import paths


def pdist_euclidean_upper(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    parts = []
    for i in range(n - 1):
        d = np.sqrt(np.sum((X[i + 1 :] - X[i]) ** 2, axis=1))
        parts.append(d.astype(np.float64, copy=False))
    return np.concatenate(parts, axis=0)


PCA_NPZ_PATH = paths.joint_pca_npz()
OUT_HIST = paths.geometric_csv("geometric_space_density_pairwise_histogram.csv")
OUT_SUMMARY = paths.geometric_csv("geometric_space_density_summary.csv")

SAMPLE_N = 5000
RANDOM_STATE = 42
N_BINS = 60


def main() -> None:
    paths.geometric_dir()
    rng = np.random.default_rng(RANDOM_STATE)

    data = np.load(str(PCA_NPZ_PATH), allow_pickle=True)
    Z_all = data["Z_all"].astype(np.float64, copy=False)
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])
    N = Z_all.shape[0]
    assert N == n_weibo + n_renmin

    Z_weibo = Z_all[:n_weibo]
    Z_renmin = Z_all[n_weibo:]

    n_w = min(SAMPLE_N, n_weibo)
    n_r = min(SAMPLE_N, n_renmin)
    idx_w = rng.choice(n_weibo, size=n_w, replace=False)
    idx_r = rng.choice(n_renmin, size=n_r, replace=False)

    Sw = Z_weibo[idx_w]
    Sr = Z_renmin[idx_r]

    print(f"Computing pairwise distances (Weibo n={n_w}, dim={Sw.shape[1]})...")
    dist_w = pdist_euclidean_upper(Sw)
    print(f"Computing pairwise distances (Renmin n={n_r}, dim={Sr.shape[1]})...")
    dist_r = pdist_euclidean_upper(Sr)

    d_max = float(max(dist_w.max(), dist_r.max()))
    if d_max <= 0:
        raise ValueError("Degenerate distances (max<=0).")

    bin_edges = np.linspace(0.0, d_max, N_BINS + 1)
    counts_w, _ = np.histogram(dist_w, bins=bin_edges)
    counts_r, _ = np.histogram(dist_r, bins=bin_edges)

    sum_w = counts_w.sum()
    sum_r = counts_r.sum()
    dens_w = counts_w.astype(np.float64) / max(sum_w, 1)
    dens_r = counts_r.astype(np.float64) / max(sum_r, 1)

    with open(OUT_HIST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bin_lower",
                "bin_upper",
                "count_weibo",
                "count_renmin",
                "density_weibo",
                "density_renmin",
            ]
        )
        for i in range(N_BINS):
            w.writerow(
                [
                    f"{bin_edges[i]:.8f}",
                    f"{bin_edges[i + 1]:.8f}",
                    int(counts_w[i]),
                    int(counts_r[i]),
                    f"{dens_w[i]:.10f}",
                    f"{dens_r[i]:.10f}",
                ]
            )

    summary = {
        "space": "pca50",
        "metric": "euclidean_pairwise",
        "sample_n_weibo": n_w,
        "sample_n_renmin": n_r,
        "dim": Sw.shape[1],
        "n_bins": N_BINS,
        "random_state": RANDOM_STATE,
        "mean_pairwise_dist_weibo": float(np.mean(dist_w)),
        "mean_pairwise_dist_renmin": float(np.mean(dist_r)),
        "n_pairs_weibo": int(dist_w.size),
        "n_pairs_renmin": int(dist_r.size),
        "bin_max": d_max,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("Wrote:", OUT_HIST)
    print("Wrote:", OUT_SUMMARY)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv
import os

import numpy as np
from sklearn.metrics import silhouette_samples

import paths

NPZ_PATH = paths.cluster_npz()
OUT_DIR = paths.cluster_dir()
OUT_DIST_WEIBO = OUT_DIR / "separability_intercluster_weibo.csv"
OUT_DIST_RENMIN = OUT_DIR / "separability_intercluster_renmin.csv"
OUT_SIL_SAMPLES = OUT_DIR / "separability_silhouette_samples.csv"
OUT_SIL_BY_CLUSTER = OUT_DIR / "separability_silhouette_by_cluster_source.csv"

SIL_SAMPLE_CAP = int(os.environ.get("SIL_SAMPLE_CAP", "25000"))


def source_cluster_centroids(
    Z: np.ndarray,
    labels: np.ndarray,
    mask_source: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    d = Z.shape[1]
    mus = np.full((n_clusters, d), np.nan, dtype=np.float64)
    for c in range(n_clusters):
        m = mask_source & (labels == c)
        if np.any(m):
            mus[c] = Z[m].mean(axis=0)
    return mus


def pairwise_centroid_distances(mus: np.ndarray) -> np.ndarray:
    k = mus.shape[0]
    D = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        D[i, i] = 0.0
        for j in range(i + 1, k):
            if np.all(np.isfinite(mus[i])) and np.all(np.isfinite(mus[j])):
                d_ij = float(np.linalg.norm(mus[i] - mus[j]))
                D[i, j] = D[j, i] = d_ij
            else:
                D[i, j] = D[j, i] = np.nan
    return D


def save_distance_csv(path: Path, D: np.ndarray, n_clusters: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["cluster_i"] + [str(j) for j in range(n_clusters)]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_clusters):
            row = [i] + [("" if np.isnan(D[i, j]) else f"{D[i, j]:.8f}") for j in range(n_clusters)]
            w.writerow(row)


def safe_silhouette_samples(Z_sub: np.ndarray, lab_sub: np.ndarray):
    if Z_sub.shape[0] < 2:
        return None
    uniq = np.unique(lab_sub)
    if uniq.size < 2:
        return None
    try:
        return silhouette_samples(Z_sub, lab_sub, metric="euclidean")
    except ValueError:
        return None


def subsample_for_silhouette(
    Z_sub: np.ndarray,
    lab_sub: np.ndarray,
    global_idx: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    n = Z_sub.shape[0]
    if cap <= 0 or n <= cap:
        return Z_sub, lab_sub, global_idx, False
    rel = rng.choice(n, size=cap, replace=False)
    rel.sort()
    return Z_sub[rel], lab_sub[rel], global_idx[rel], True


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    data = np.load(str(NPZ_PATH), allow_pickle=True)
    Z = data["Z_all"].astype(np.float64, copy=False)
    labels = data["labels"].astype(np.int64, copy=False)
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])
    if "n_clusters" in data.files:
        n_clusters = int(np.asarray(data["n_clusters"]).item())
    else:
        n_clusters = int(labels.max()) + 1

    N = Z.shape[0]
    assert N == n_weibo + n_renmin

    is_weibo = np.zeros(N, dtype=bool)
    is_weibo[:n_weibo] = True
    is_renmin = ~is_weibo

    print("Computing inter-cluster centroid distance matrices...")
    mus_w = source_cluster_centroids(Z, labels, is_weibo, n_clusters)
    mus_r = source_cluster_centroids(Z, labels, is_renmin, n_clusters)
    D_w = pairwise_centroid_distances(mus_w)
    D_r = pairwise_centroid_distances(mus_r)

    save_distance_csv(OUT_DIST_WEIBO, D_w, n_clusters)
    save_distance_csv(OUT_DIST_RENMIN, D_r, n_clusters)
    print("Wrote:", OUT_DIST_WEIBO)
    print("Wrote:", OUT_DIST_RENMIN)

    sample_rows = []
    by_cluster_rows = []

    for source_name, mask in (("weibo", is_weibo), ("renmin", is_renmin)):
        Z_sub = Z[mask]
        lab_sub = labels[mask]
        global_idx = np.where(mask)[0]
        n_full = Z_sub.shape[0]

        Z_use, lab_use, g_use, did_sub = subsample_for_silhouette(
            Z_sub, lab_sub, global_idx, SIL_SAMPLE_CAP, rng
        )
        print(
            f"Silhouette [{source_name}]: n_full={n_full}, "
            f"n_used={Z_use.shape[0]}, subsampled={did_sub} (cap={SIL_SAMPLE_CAP})"
        )

        sil = safe_silhouette_samples(Z_use, lab_use)
        if sil is None:
            print(f"  Warning: silhouette skipped for {source_name}.")
            continue

        for gi, c, s in zip(g_use, lab_use, sil):
            sample_rows.append(
                {
                    "global_index": int(gi),
                    "source": source_name,
                    "cluster_id": int(c),
                    "silhouette": float(s),
                    "silhouette_subsampled": int(did_sub),
                }
            )

        for c in range(n_clusters):
            m = lab_use == c
            if not np.any(m):
                continue
            vals = sil[m]
            n_full_c = int(np.sum(lab_sub == c))
            by_cluster_rows.append(
                {
                    "cluster_id": c,
                    "source": source_name,
                    "n_in_source_cluster": n_full_c,
                    "n_in_silhouette_sample": int(np.sum(m)),
                    "mean_silhouette": float(np.mean(vals)),
                    "std_silhouette": float(np.std(vals)),
                    "silhouette_subsampled": int(did_sub),
                }
            )

    if sample_rows:
        keys = list(sample_rows[0].keys())
        with open(OUT_SIL_SAMPLES, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sample_rows)
        print("Wrote:", OUT_SIL_SAMPLES, f"({len(sample_rows)} rows)")

    if by_cluster_rows:
        keys = list(by_cluster_rows[0].keys())
        with open(OUT_SIL_BY_CLUSTER, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(by_cluster_rows)
        print("Wrote:", OUT_SIL_BY_CLUSTER)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv

import numpy as np

import paths


def mean_sq_dist_to_centroid(Z_sub: np.ndarray) -> float:
    if Z_sub.shape[0] == 0:
        return float("nan")
    mu = Z_sub.mean(axis=0)
    d2 = np.sum((Z_sub - mu) ** 2, axis=1)
    return float(np.mean(d2))


def main() -> None:
    npz_path = paths.cluster_npz()
    out_path = paths.cluster_csv("cluster_intra_compactness.csv")

    data = np.load(str(npz_path), allow_pickle=True)
    Z = data["Z_all"].astype(np.float64, copy=False)
    labels = data["labels"].astype(np.int64, copy=False)
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])

    N = Z.shape[0]
    assert N == n_weibo + n_renmin

    is_weibo = np.zeros(N, dtype=bool)
    is_weibo[:n_weibo] = True
    is_renmin = ~is_weibo

    rows = []
    for c in np.sort(np.unique(labels)):
        m = labels == c
        Zw = Z[m & is_weibo]
        Zr = Z[m & is_renmin]
        rows.append(
            {
                "cluster_id": int(c),
                "n_weibo_in_cluster": Zw.shape[0],
                "n_renmin_in_cluster": Zr.shape[0],
                "compactness_mean_sq_weibo": mean_sq_dist_to_centroid(Zw),
                "compactness_mean_sq_renmin": mean_sq_dist_to_centroid(Zr),
            }
        )

    paths.cluster_dir()
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

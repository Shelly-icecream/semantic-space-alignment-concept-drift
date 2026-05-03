from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv

import numpy as np

import paths


def main() -> None:
    npz_path = paths.cluster_npz()
    out_path = paths.cluster_csv("cluster_centroid_drift.csv")

    data = np.load(str(npz_path), allow_pickle=True)
    Z = data["Z_all"].astype(np.float64, copy=False)
    labels = data["labels"].astype(np.int64, copy=False)
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])

    N = Z.shape[0]
    assert N == n_weibo + n_renmin
    assert Z.ndim == 2

    is_weibo = np.zeros(N, dtype=bool)
    is_weibo[:n_weibo] = True
    is_renmin = ~is_weibo

    cluster_ids = np.sort(np.unique(labels))
    rows = []

    for c in cluster_ids:
        mask_c = labels == c
        mask_w = mask_c & is_weibo
        mask_r = mask_c & is_renmin
        n_w = int(np.sum(mask_w))
        n_r = int(np.sum(mask_r))

        if n_w == 0 or n_r == 0:
            drift = float("nan")
        else:
            mu_w = Z[mask_w].mean(axis=0)
            mu_r = Z[mask_r].mean(axis=0)
            drift = float(np.linalg.norm(mu_w - mu_r))

        rows.append(
            {
                "cluster_id": int(c),
                "n_weibo_in_cluster": n_w,
                "n_renmin_in_cluster": n_r,
                "n_total_in_cluster": n_w + n_r,
                "centroid_drift_l2": drift,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    paths.cluster_dir()
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

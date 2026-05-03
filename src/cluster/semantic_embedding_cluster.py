from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from sklearn.cluster import KMeans

import paths


def main() -> None:
    data = np.load(str(paths.joint_pca_npz()), allow_pickle=True)
    Z_all = data["Z_all"]
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])

    assert Z_all.shape[0] == n_weibo + n_renmin

    k_final = 20
    kmeans_final = KMeans(
        n_clusters=k_final,
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    labels = kmeans_final.fit_predict(Z_all)

    out = paths.cluster_npz()
    paths.cluster_dir()
    np.savez(
        str(out),
        labels=labels,
        n_clusters=k_final,
        Z_all=Z_all,
        n_weibo=n_weibo,
        n_renmin=n_renmin,
        inertia=float(kmeans_final.inertia_),
    )
    print("Wrote:", out)


if __name__ == "__main__":
    main()

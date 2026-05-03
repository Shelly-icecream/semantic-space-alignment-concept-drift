from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

import paths

DATA_PATH = paths.cluster_npz()
WEIBO_KV_PATH = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)
RENMIN_KV_PATH = paths.alignment_kv(paths.KV_RENMIN)
CSV_PATH = paths.cluster_csv(paths.CLUSTER_WORDS_CSV)
GLOBAL_SAMPLE = 8000
PER_CLUSTER_SAMPLE = 1200
CLUSTERS_PER_FIG = 10
COLOR_WEIBO = "#0b4f8a"
COLOR_RENMIN = "#8b1a1a"


def sample_indices(rng: np.random.Generator, idx: np.ndarray, max_n: int) -> np.ndarray:
    if idx.size <= max_n:
        return idx
    return rng.choice(idx, size=max_n, replace=False)


def export_cluster_words_csv(labels: np.ndarray, n_weibo: int, n_renmin: int) -> None:
    kv_weibo = KeyedVectors.load(str(WEIBO_KV_PATH), mmap="r")
    kv_renmin = KeyedVectors.load(str(RENMIN_KV_PATH), mmap="r")

    weibo_words = kv_weibo.index_to_key
    renmin_words = kv_renmin.index_to_key

    if len(weibo_words) < n_weibo or len(renmin_words) < n_renmin:
        raise ValueError("KV vocabulary size is smaller than split sizes in clustering result.")

    if labels.shape[0] != n_weibo + n_renmin:
        raise ValueError("labels length does not match n_weibo + n_renmin.")

    rows = []
    for i in range(n_weibo):
        rows.append((int(labels[i]), "weibo", weibo_words[i], i))
    for j in range(n_renmin):
        idx = n_weibo + j
        rows.append((int(labels[idx]), "renmin", renmin_words[j], idx))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    paths.cluster_dir()
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "source", "word", "global_index"])
        writer.writerows(rows)

    print(f"Saved cluster-word table: {CSV_PATH}")


def main() -> None:
    data = np.load(str(DATA_PATH))
    Z_all = data["Z_all"]
    labels = data["labels"]
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])

    assert Z_all.shape[0] == n_weibo + n_renmin
    N = Z_all.shape[0]

    is_weibo = np.zeros(N, dtype=bool)
    is_weibo[:n_weibo] = True
    is_renmin = ~is_weibo

    Z2 = PCA(n_components=2, random_state=42).fit_transform(Z_all)
    cluster_ids = np.sort(np.unique(labels))

    rng = np.random.default_rng(42)

    idx_w_all = np.where(is_weibo)[0]
    idx_r_all = np.where(is_renmin)[0]
    idx_w_plot = sample_indices(rng, idx_w_all, GLOBAL_SAMPLE)
    idx_r_plot = sample_indices(rng, idx_r_all, GLOBAL_SAMPLE)

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.scatter(
        Z2[idx_w_plot, 0],
        Z2[idx_w_plot, 1],
        s=5,
        alpha=0.32,
        c=COLOR_WEIBO,
        edgecolors="none",
        rasterized=True,
        label=f"Weibo (sample={idx_w_plot.size})",
    )
    ax1.scatter(
        Z2[idx_r_plot, 0],
        Z2[idx_r_plot, 1],
        s=5,
        alpha=0.32,
        c=COLOR_RENMIN,
        edgecolors="none",
        rasterized=True,
        label=f"Renmin (sample={idx_r_plot.size})",
    )
    ax1.set_title("Global 2D projection: source distribution")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    n_clusters = len(cluster_ids)
    n_pages = int(np.ceil(n_clusters / CLUSTERS_PER_FIG))
    n_cols = 5
    n_rows = int(np.ceil(CLUSTERS_PER_FIG / n_cols))

    for page in range(n_pages):
        start = page * CLUSTERS_PER_FIG
        end = min((page + 1) * CLUSTERS_PER_FIG, n_clusters)
        page_cluster_ids = cluster_ids[start:end]

        fig2, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.8 * n_cols, 3.4 * n_rows), sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()

        for i, c in enumerate(page_cluster_ids):
            ax = axes[i]
            mask_c = labels == c

            idx_weibo = np.where(mask_c & is_weibo)[0]
            idx_renmin = np.where(mask_c & is_renmin)[0]
            idx_weibo_plot = sample_indices(rng, idx_weibo, PER_CLUSTER_SAMPLE)
            idx_renmin_plot = sample_indices(rng, idx_renmin, PER_CLUSTER_SAMPLE)

            if idx_weibo_plot.size > 0:
                pts_w = Z2[idx_weibo_plot]
                ax.scatter(
                    pts_w[:, 0],
                    pts_w[:, 1],
                    s=6,
                    alpha=0.35,
                    c=COLOR_WEIBO,
                    edgecolors="none",
                    rasterized=True,
                    label="Weibo",
                )
                center_w = Z2[idx_weibo].mean(axis=0)
                ax.scatter(center_w[0], center_w[1], marker="x", s=80, c=COLOR_WEIBO, linewidths=2)

            if idx_renmin_plot.size > 0:
                pts_r = Z2[idx_renmin_plot]
                ax.scatter(
                    pts_r[:, 0],
                    pts_r[:, 1],
                    s=6,
                    alpha=0.35,
                    c=COLOR_RENMIN,
                    edgecolors="none",
                    rasterized=True,
                    label="Renmin",
                )
                center_r = Z2[idx_renmin].mean(axis=0)
                ax.scatter(center_r[0], center_r[1], marker="x", s=80, c=COLOR_RENMIN, linewidths=2)

            if idx_weibo.size > 0 and idx_renmin.size > 0:
                center_w = Z2[idx_weibo].mean(axis=0)
                center_r = Z2[idx_renmin].mean(axis=0)
                ax.plot(
                    [center_w[0], center_r[0]],
                    [center_w[1], center_r[1]],
                    "--",
                    c="gray",
                    linewidth=1.2,
                    alpha=0.8,
                )

            ax.set_title(f"C{int(c)} | W:{idx_weibo.size} R:{idx_renmin.size}", fontsize=10)
            ax.grid(alpha=0.25)

        for j in range(len(page_cluster_ids), axes.size):
            axes[j].axis("off")

        handles, legend_labels = axes[0].get_legend_handles_labels()
        if handles:
            fig2.legend(
                handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=2,
                frameon=False,
            )
        fig2.suptitle(
            f"Per-cluster 2D overlays with source centroids (page {page + 1}/{n_pages})",
            y=0.955,
            fontsize=14,
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.88])
        plt.show()

    weibo_counts = []
    renmin_counts = []
    for c in cluster_ids:
        mask_c = labels == c
        weibo_counts.append(np.sum(mask_c & is_weibo))
        renmin_counts.append(np.sum(mask_c & is_renmin))

    weibo_counts = np.array(weibo_counts)
    renmin_counts = np.array(renmin_counts)
    total_counts = weibo_counts + renmin_counts
    np.divide(
        weibo_counts,
        total_counts,
        out=np.zeros_like(weibo_counts, dtype=float),
        where=total_counts > 0,
    )
    np.divide(
        renmin_counts,
        total_counts,
        out=np.zeros_like(renmin_counts, dtype=float),
        where=total_counts > 0,
    )

    x = np.arange(n_clusters)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4.8))

    ax3.bar(x, weibo_counts, color=COLOR_WEIBO, label="Weibo")
    ax3.bar(x, renmin_counts, bottom=weibo_counts, color=COLOR_RENMIN, label="Renmin")
    ax3.set_xticks(x, [str(int(c)) for c in cluster_ids])
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Count")
    ax3.set_title("Cluster composition (absolute counts)")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(frameon=False)

    fig3.suptitle("Source dominance diagnostics by cluster", y=0.98, fontsize=13)
    fig3.tight_layout()
    plt.show()

    export_cluster_words_csv(labels=labels, n_weibo=n_weibo, n_renmin=n_renmin)


if __name__ == "__main__":
    main()

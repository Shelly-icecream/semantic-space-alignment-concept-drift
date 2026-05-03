"""
1) Drift + intra-compactness (two panels)
2) Separability: inter-cluster distance heatmaps + silhouette violin
"""

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths

DRIFT_CSV = paths.cluster_csv("cluster_centroid_drift.csv")
COMPACT_CSV = paths.cluster_csv("cluster_intra_compactness.csv")
OUT_COMBINED = paths.fig_cluster_dir() / "cluster_drift_visualization_side_by_side.png"
OUT_DRIFT_ONLY = paths.fig_cluster_dir() / "cluster_centroid_drift_panel.png"
OUT_COMPACT_ONLY = paths.fig_cluster_dir() / "cluster_intra_compactness_panel.png"

RESULT_DIR = paths.cluster_dir()
SEP_DIST_WEIBO = RESULT_DIR / "separability_intercluster_weibo.csv"
SEP_DIST_RENMIN = RESULT_DIR / "separability_intercluster_renmin.csv"
SEP_SIL_SAMPLES = RESULT_DIR / "separability_silhouette_samples.csv"
OUT_SEP_HEATMAP = paths.fig_cluster_dir() / "separability_intercluster_heatmap.png"
OUT_SEP_VIOLIN = paths.fig_cluster_dir() / "separability_silhouette_violin.png"

COLOR_DRIFT_LINE = "#1b3a5f"
COLOR_BUBBLE = "#7eb6d9"
COLOR_WEIBO = "#2f6f8c"
COLOR_RENMIN = "#8c3a3a"


def bubble_point_sizes(n: pd.Series, s_min: float = 36.0, s_max: float = 200.0) -> np.ndarray:
    t = np.sqrt(np.maximum(n.astype(float).values, 1.0))
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return np.full_like(t, (s_min + s_max) / 2.0)
    u = (t - t_min) / (t_max - t_min)
    return s_min + u * (s_max - s_min)


def _set_academic_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
        }
    )


def load_merged() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    df_d = pd.read_csv(DRIFT_CSV)
    df_c = pd.read_csv(COMPACT_CSV)
    df = df_d.merge(df_c, on="cluster_id", how="inner", suffixes=("", "_y"))
    dup_cols = [c for c in df.columns if c.endswith("_y")]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    x = df["cluster_id"].to_numpy()
    drift = df["centroid_drift_l2"].to_numpy()
    n_total = df["n_total_in_cluster"]
    return x, drift, bubble_point_sizes(n_total), df


def plot_drift_panel(ax, x, drift, sizes) -> None:
    ax.plot(x, drift, "-", color=COLOR_DRIFT_LINE, lw=2.2, label="Centroid drift (L2)", zorder=3)
    ax.scatter(
        x,
        drift,
        s=sizes,
        alpha=0.45,
        c=COLOR_BUBBLE,
        edgecolors=COLOR_DRIFT_LINE,
        linewidths=0.25,
        zorder=4,
        label="Marker area ∝ √(N_weibo + N_renmin)",
    )
    ax.set_xlabel("Cluster id (KMeans)")
    ax.set_ylabel("L2 distance between subset means\n(PCA-50 embedding)")
    ax.set_title("(A) Cross-source centroid drift")
    ax.set_xticks(x)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#cccccc")


def plot_compactness_panel(ax, df) -> None:
    x = df["cluster_id"].to_numpy()
    ax.plot(
        x,
        df["compactness_mean_sq_weibo"],
        "o-",
        color=COLOR_WEIBO,
        lw=1.7,
        ms=5,
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="Weibo: mean ‖z − μ_wb‖²",
    )
    ax.plot(
        x,
        df["compactness_mean_sq_renmin"],
        "s-",
        color=COLOR_RENMIN,
        lw=1.7,
        ms=5,
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="Renmin: mean ‖z − μ_rm‖²",
    )
    ax.set_xlabel("Cluster id (KMeans)")
    ax.set_ylabel("Mean squared distance to subset mean\n(same PCA-50 embedding)")
    ax.set_title("(B) Intra-source compactness")
    ax.set_xticks(x)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    print("Saved:", path)


def read_distance_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    return df.astype(float).to_numpy()


def plot_separability_heatmaps() -> None:
    if not (SEP_DIST_WEIBO.is_file() and SEP_DIST_RENMIN.is_file()):
        print("Skip separability heatmaps: run cluster_separability_metrics.py first.")
        return

    D_w = read_distance_matrix(SEP_DIST_WEIBO)
    D_r = read_distance_matrix(SEP_DIST_RENMIN)
    k = D_w.shape[0]

    finite = np.concatenate([D_w[np.isfinite(D_w)], D_r[np.isfinite(D_r)]])
    if finite.size == 0:
        print("Skip separability heatmaps: no finite distances.")
        return
    vmax = float(np.nanmax(finite))
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, D, title in (
        (axes[0], D_w, "Weibo: inter-cluster centroid distances"),
        (axes[1], D_r, "Renmin: inter-cluster centroid distances"),
    ):
        im = ax.imshow(D, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Cluster j")
        ax.set_ylabel("Cluster i")
        ax.set_xticks(np.arange(k))
        ax.set_yticks(np.arange(k))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L2 distance (PCA-50)")

    fig.suptitle(
        "Separability: pairwise distances between source-specific cluster centroids",
        fontsize=12,
    )
    save_figure(fig, OUT_SEP_HEATMAP)
    plt.close(fig)


def plot_separability_violin() -> None:
    if not SEP_SIL_SAMPLES.is_file():
        print("Skip separability violin: no", SEP_SIL_SAMPLES)
        return

    df_s = pd.read_csv(SEP_SIL_SAMPLES)
    if df_s.empty or "source" not in df_s.columns:
        return

    data_w = df_s.loc[df_s["source"] == "weibo", "silhouette"].to_numpy()
    data_r = df_s.loc[df_s["source"] == "renmin", "silhouette"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
    parts = ax.violinplot(
        [data_w, data_r],
        positions=[0, 1],
        showmeans=True,
        showmedians=False,
        widths=0.65,
    )
    for b in parts["bodies"]:
        b.set_alpha(0.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Weibo", "Renmin"])
    ax.set_ylabel("Silhouette coefficient (per word, joint KMeans labels)")
    ax.set_title("Boundary clarity within each source (higher = clearer boundaries)")
    ax.axhline(0.0, color="#666666", lw=0.8, ls="--", alpha=0.8)
    save_figure(fig, OUT_SEP_VIOLIN)
    plt.close(fig)


def main() -> None:
    _set_academic_style()

    x, drift, sizes, df = load_merged()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True, constrained_layout=True)
    plot_drift_panel(axes[0], x, drift, sizes)
    plot_compactness_panel(axes[1], df)
    fig.suptitle(
        "Per-cluster metrics in PCA-50 space (same KMeans labels)",
        fontsize=12,
    )
    save_figure(fig, OUT_COMBINED)
    plt.close(fig)

    fig_d, ax_d = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    plot_drift_panel(ax_d, x, drift, sizes)
    fig_d.suptitle("Cross-source centroid drift", fontsize=11)
    save_figure(fig_d, OUT_DRIFT_ONLY)
    plt.close(fig_d)

    fig_c, ax_c = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    plot_compactness_panel(ax_c, df)
    fig_c.suptitle("Intra-source compactness", fontsize=11)
    save_figure(fig_c, OUT_COMPACT_ONLY)
    plt.close(fig_c)

    plot_separability_heatmaps()
    plot_separability_violin()

    plt.show()


if __name__ == "__main__":
    main()

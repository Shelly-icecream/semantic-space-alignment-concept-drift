"""Geometric figures for the HDSAD pipeline (reads result/geometric, writes figures/geometric)."""

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
from matplotlib.colors import PowerNorm

import paths

DENSITY_HIST_CSV = paths.geometric_csv("geometric_space_density_pairwise_histogram.csv")
DENSITY_SUMMARY_CSV = paths.geometric_csv("geometric_space_density_summary.csv")
OUT_FIG_DENSITY = paths.fig_geometric_dir() / "geometric_space_density_pairwise_hist.png"
SPECTRUM_CSV = paths.geometric_csv("geometric_space_covariance_spectrum.csv")
SUMMARY_CSV = paths.geometric_csv("geometric_space_covariance_summary.csv")
OUT_FIG_ISOTROPY = paths.fig_geometric_dir() / "geometric_space_covariance_isotropy_spectrum.png"
OUT_FIG_BREADTH = paths.fig_geometric_dir() / "geometric_space_covariance_logdet_breadth.png"

PROCRUSTES_NPZ = paths.geometric_npz("geometric_space_procrustes_residual_heatmap.npz")
PROCRUSTES_SUMMARY_CSV = paths.geometric_csv("geometric_space_procrustes_summary.csv")
OUT_FIG_PROCRUSTES = (
    paths.fig_geometric_dir() / "geometric_space_topology_procrustes_residual_heatmap.png"
)


def _set_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
        }
    )


def plot_procrustes_residual_heatmap(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_axisbelow(False)

    z = np.load(str(PROCRUSTES_NPZ), allow_pickle=True)
    H = np.asarray(z["residual_abs"], dtype=float)
    Ht = H.T
    n_dim, n_word = Ht.shape

    vmax = float(np.quantile(H, 0.995))
    vmax = max(vmax, 1e-5)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)

    im = ax.imshow(
        Ht,
        aspect="auto",
        origin="upper",
        cmap="inferno",
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("Word rank within subsample (1 = largest L2 residual; left = worst words)")
    ax.set_ylabel("PCA dimension (joint PCA-50)")
    ax.set_title("Topology: |residual| after orthogonal map (Weibo → Renmin)")

    ax.set_yticks(np.arange(0, n_dim, 5))
    step = max(1, n_word // 12)
    ax.set_xticks(np.arange(0, n_word, step))
    ax.tick_params(axis="x", rotation=0, labelsize=8)

    if PROCRUSTES_SUMMARY_CSV.is_file():
        s = pd.read_csv(PROCRUSTES_SUMMARY_CSV).iloc[0].to_dict()
        fr = s.get("frobenius_residual_sq", float("nan"))
        rel = s.get("relative_frobenius_residual_sq", float("nan"))
        note = (
            f"Frobenius squared residual: {fr:.4e}; rel. to ||Z_renmin||_F^2: {rel:.4e}\n"
            f"Color cap: 99.5% quantile = {vmax:.4g} (PowerNorm γ=0.5)"
        )
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.5", "alpha": 0.94},
        )

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend="max")
    cbar.set_label("|residual|")


def plot_procrustes_mean_residual_by_dim(ax: plt.Axes) -> None:
    z = np.load(str(PROCRUSTES_NPZ), allow_pickle=True)
    H = np.asarray(z["residual_abs"], dtype=float)
    if "mean_abs_residual_per_dim" in z.files:
        m = np.asarray(z["mean_abs_residual_per_dim"], dtype=float).ravel()
    else:
        m = np.mean(np.abs(H), axis=0)
    d = m.size
    x = np.arange(1, d + 1)
    ax.fill_between(x, 0.0, m, color="#4c1d4f", alpha=0.35, linewidth=0)
    ax.plot(x, m, color="#f98e09", lw=2.0, marker="o", ms=4, mfc="white", mec="#f98e09")
    ax.set_xlabel("PCA dimension index (joint PCA-50, 1 = leading component)")
    ax.set_ylabel("Mean absolute residual (all subsampled intersection words)")
    ax.set_title(
        "Same Procrustes map: average |residual| per dimension (summarizes heatmap structure)"
    )
    ax.set_xlim(0.5, d + 0.5)
    ax.set_ylim(0.0, max(float(np.max(m)) * 1.08, 1e-6))
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", linewidth=0.7)


def main() -> None:
    missing_p = [p for p in (PROCRUSTES_NPZ, PROCRUSTES_SUMMARY_CSV) if not p.is_file()]
    if missing_p:
        raise FileNotFoundError(
            "Missing: " + ", ".join(str(p) for p in missing_p) + ". Run geometric_space_procrustes_metrics.py first."
        )

    _set_style()
    paths.fig_geometric_dir().mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20.0, 12.0),
        constrained_layout=True,
        height_ratios=[1.35, 1.0],
    )
    plot_procrustes_residual_heatmap(axes[0])
    plot_procrustes_mean_residual_by_dim(axes[1])
    fig.savefig(OUT_FIG_PROCRUSTES, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", OUT_FIG_PROCRUSTES)
    plt.close(fig)


if __name__ == "__main__":
    main()

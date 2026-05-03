"""Geometric level — covariance spectrum and generalized variance (PCA-50)."""

from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import csv

import numpy as np

import paths

PCA_NPZ_PATH = paths.joint_pca_npz()
OUT_SPECTRUM = paths.geometric_csv("geometric_space_covariance_spectrum.csv")
OUT_SUMMARY = paths.geometric_csv("geometric_space_covariance_summary.csv")

EIGENVALUE_FLOOR = 1e-18
LOGDET_EXP_CLIP = 700.0


def _spectrum_scores(evals: np.ndarray) -> dict:
    lam = np.clip(evals.astype(np.float64), EIGENVALUE_FLOOR, None)
    d = lam.size
    s = lam.sum()
    q = lam / s
    h = float(-(q * np.log(q)).sum())
    h_max = float(np.log(d)) if d > 1 else 1.0
    isotropy_entropy_ratio = h / h_max if h_max > 0 else float("nan")

    lam_max = float(lam[0])
    lam_min = float(lam[-1])
    cond = lam_max / max(lam_min, lam_max * 1e-15)

    log_gvar = float(np.log(lam).sum())
    gvar = float(np.exp(np.clip(log_gvar, -LOGDET_EXP_CLIP, LOGDET_EXP_CLIP)))

    return {
        "isotropy_entropy_ratio": isotropy_entropy_ratio,
        "condition_number": cond,
        "log_generalized_variance": log_gvar,
        "generalized_variance": gvar,
    }


def cov_eigenvalues_desc(Z: np.ndarray) -> np.ndarray:
    X = Z - Z.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two rows to form a covariance.")
    c = (X.T @ X) / (n - 1)
    w, _ = np.linalg.eigh(c)
    w = np.clip(w[::-1], 0.0, None)
    return w


def main() -> None:
    paths.geometric_dir()

    data = np.load(str(PCA_NPZ_PATH), allow_pickle=True)
    Z_all = data["Z_all"].astype(np.float64, copy=False)
    n_weibo = int(data["n_weibo"])
    n_renmin = int(data["n_renmin"])
    assert Z_all.shape[0] == n_weibo + n_renmin

    Z_weibo = Z_all[:n_weibo]
    Z_renmin = Z_all[n_weibo:]

    print("Eigen-decomposition of Weibo covariance...")
    ev_w = cov_eigenvalues_desc(Z_weibo)
    print("Eigen-decomposition of Renmin covariance...")
    ev_r = cov_eigenvalues_desc(Z_renmin)

    d = ev_w.size
    if ev_r.size != d:
        raise ValueError("PCA dimension mismatch between sources.")

    share_w = ev_w / ev_w.sum()
    share_r = ev_r / ev_r.sum()

    with open(OUT_SPECTRUM, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "eigenvalue_weibo",
                "eigenvalue_renmin",
                "variance_share_weibo",
                "variance_share_renmin",
            ]
        )
        for k in range(d):
            w.writerow(
                [
                    k + 1,
                    f"{ev_w[k]:.10e}",
                    f"{ev_r[k]:.10e}",
                    f"{share_w[k]:.10e}",
                    f"{share_r[k]:.10e}",
                ]
            )

    sc_w = _spectrum_scores(ev_w)
    sc_r = _spectrum_scores(ev_r)

    summary = {
        "space": "pca50",
        "dim": d,
        "n_weibo": n_weibo,
        "n_renmin": n_renmin,
        "isotropy_entropy_ratio_weibo": sc_w["isotropy_entropy_ratio"],
        "isotropy_entropy_ratio_renmin": sc_r["isotropy_entropy_ratio"],
        "condition_number_weibo": sc_w["condition_number"],
        "condition_number_renmin": sc_r["condition_number"],
        "log_generalized_variance_weibo": sc_w["log_generalized_variance"],
        "log_generalized_variance_renmin": sc_r["log_generalized_variance"],
        "generalized_variance_weibo": sc_w["generalized_variance"],
        "generalized_variance_renmin": sc_r["generalized_variance"],
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("Wrote:", OUT_SPECTRUM)
    print("Wrote:", OUT_SUMMARY)


if __name__ == "__main__":
    main()

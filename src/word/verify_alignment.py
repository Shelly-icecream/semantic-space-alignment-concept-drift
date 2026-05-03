import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from gensim.models import KeyedVectors

import paths


def main() -> None:
    rm_path = paths.alignment_kv(paths.KV_RENMIN)
    wb_path = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)
    anchor_path = paths.alignment_kv(paths.KV_ANCHOR_RENMIN_SORTED)

    print("加载数据...")
    model_rm = KeyedVectors.load(str(rm_path), mmap="r")
    model_wb = KeyedVectors.load(str(wb_path), mmap="r")

    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"缺少锚点词向量文件: {anchor_path}（请置于 result/alignment/ 下后再运行本脚本）"
        )

    model_anchor = KeyedVectors.load(str(anchor_path), mmap="r")
    anchor_words = list(model_anchor.key_to_index.keys())

    X = np.array([model_rm[w] for w in anchor_words])
    Y = np.array([model_wb[w] for w in anchor_words])

    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt

    X_aligned = X @ Q

    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    X_aligned_norm = X_aligned / np.linalg.norm(X_aligned, axis=1, keepdims=True)

    I_approx = Q.T @ Q
    ortho_error = np.max(np.abs(I_approx - np.eye(Q.shape[0])))
    print(f"验证1 - 正交性: 最大误差 = {ortho_error:.2e}")
    print(f"  结论: Q {'是' if ortho_error < 1e-5 else '不是'} 正交矩阵")

    recon_error = np.linalg.norm(X_aligned - Y, "fro")
    relative_error = recon_error / np.linalg.norm(Y, "fro")
    print(f"\n验证2 - 锚点重建误差:")
    print(f"  绝对误差 = {recon_error:.4f}")
    print(f"  相对误差 = {relative_error:.4%}")

    dists_before = np.linalg.norm(X_norm - Y_norm, axis=1)
    dists_after = np.linalg.norm(X_aligned_norm - Y_norm, axis=1)
    improvement = (dists_before - dists_after) / dists_before * 100

    print(f"\n验证3 - 对齐前后距离对比（前10个锚点词，归一化后）:")
    print(f"{'词':<6} {'对齐前':>8} {'对齐后':>8} {'改善率':>8}")
    print("-" * 35)
    for i, w in enumerate(anchor_words[:10]):
        print(f"{w:<6} {dists_before[i]:>8.4f} {dists_after[i]:>8.4f} {improvement[i]:>7.1f}%")

    avg_improve = np.mean(improvement)
    print(f"\n平均改善率: {avg_improve:.1f}%")

    cos_before = np.mean(np.sum(X_norm * Y_norm, axis=1))
    cos_after = np.mean(np.sum(X_aligned_norm * Y_norm, axis=1))
    dist_before_avg = np.mean(dists_before)
    dist_after_avg = np.mean(dists_after)

    print(f"\n===== A要的四个数 =====")
    print(f"对齐前平均余弦: {cos_before:.4f}")
    print(f"对齐后平均余弦: {cos_after:.4f}")
    print(f"对齐前平均欧氏距离（归一化后）: {dist_before_avg:.4f}")
    print(f"对齐后平均欧氏距离（归一化后）: {dist_after_avg:.4f}")

    if cos_after > cos_before:
        print("对齐后平均余弦 > 对齐前（符合预期）")
    else:
        print("对齐后平均余弦 <= 对齐前（异常）")
    if dist_after_avg < dist_before_avg:
        print("对齐后平均距离 < 对齐前（符合预期）")
    else:
        print("对齐后平均距离 >= 对齐前（异常）")


if __name__ == "__main__":
    main()

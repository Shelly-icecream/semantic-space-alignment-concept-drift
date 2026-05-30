import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from gensim.models import KeyedVectors

import paths
from word.vocab import ANCHOR_SIZE, TOP_N, load_anchor_words, select_top_non_anchor_words

JACCARD_SAMPLE_N = 1000


def main() -> None:
    print("加载数据...")
    model_rm = KeyedVectors.load(str(paths.alignment_kv(paths.KV_RENMIN)), mmap="r")
    model_wb = KeyedVectors.load(str(paths.alignment_kv(paths.KV_WEIBO_ALIGNED)), mmap="r")

    wb_raw_path = paths.alignment_kv(paths.KV_WEIBO_RAW)
    model_wb_raw = (
        KeyedVectors.load(str(wb_raw_path), mmap="r") if wb_raw_path.is_file() else model_wb
    )
    anchor_words = load_anchor_words(model_rm, model_wb_raw, n=ANCHOR_SIZE)
    top_words = select_top_non_anchor_words(
        model_rm, model_wb, top_n=TOP_N, anchor_words=anchor_words
    )

    n_words = min(JACCARD_SAMPLE_N, len(top_words))
    sample_words = top_words[:n_words]
    print(f"用于 Jaccard 的词数（排除锚点后 Top-{n_words}，与 Shift 同排序）: {n_words}")

    X = np.array([model_rm[w] for w in sample_words])
    Y = np.array([model_wb[w] for w in sample_words])

    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    X_sim = X_norm @ X_norm.T
    Y_sim = Y_norm @ Y_norm.T

    def get_knn(sim: np.ndarray, k: int) -> np.ndarray:
        sim = sim.copy()
        np.fill_diagonal(sim, -1)
        return np.argsort(sim, axis=1)[:, -k:]

    for k in [10, 50]:
        X_knn = get_knn(X_sim, k)
        Y_knn = get_knn(Y_sim, k)

        jaccards = []
        for i in range(n_words):
            inter = len(set(X_knn[i]) & set(Y_knn[i]))
            union = len(set(X_knn[i]) | set(Y_knn[i]))
            jaccards.append(inter / union if union > 0 else 0)

        avg_j = float(np.mean(jaccards))
        print(f"\nk={k}: 平均 Jaccard = {avg_j:.4f}")
        print(f"  前10个词的值: {[f'{jaccards[i]:.4f}' for i in range(min(10, n_words))]}")

    print("\n完成！")


if __name__ == "__main__":
    main()
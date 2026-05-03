import json
import re
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from gensim.models import KeyedVectors

import paths


def main() -> None:
    paths.word_dir()
    rm_path = paths.alignment_kv(paths.KV_RENMIN)
    wb_path = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)

    print("Loading models...")
    model_rm = KeyedVectors.load(str(rm_path), mmap="r")
    model_wb = KeyedVectors.load(str(wb_path), mmap="r")

    common_raw = list(set(model_rm.key_to_index.keys()) & set(model_wb.key_to_index.keys()))
    print(f"Common words: {len(common_raw)}")

    def is_valid_word(w):
        return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,}", w))

    common_filtered = [w for w in common_raw if is_valid_word(w)]
    print(f"After filter: {len(common_filtered)}")

    anchor_words = [
        "的",
        "一",
        "是",
        "了",
        "我",
        "不",
        "在",
        "人",
        "他",
        "有",
        "这",
        "个",
        "们",
        "中",
        "来",
        "上",
        "大",
        "为",
        "就",
        "和",
        "说",
        "地",
        "也",
        "对",
        "到",
        "要",
        "下",
        "会",
        "时",
        "出",
        "那",
        "过",
        "你",
        "她",
        "能",
        "前",
        "它",
        "所",
        "都",
        "后",
    ]

    anchor_words = [w for w in anchor_words if w in common_raw]
    non_anchor_all = [w for w in common_filtered if w not in anchor_words]
    non_anchor = non_anchor_all[:5000]

    print(f"Anchor: {len(anchor_words)}, Non-anchor: {len(non_anchor)}")

    X_all = np.array([model_rm[w] for w in non_anchor])
    Y_all = np.array([model_wb[w] for w in non_anchor])
    X_anchor = np.array([model_rm[w] for w in anchor_words])
    Y_anchor = np.array([model_wb[w] for w in anchor_words])

    X_all_n = X_all / np.linalg.norm(X_all, axis=1, keepdims=True)
    Y_all_n = Y_all / np.linalg.norm(Y_all, axis=1, keepdims=True)
    X_anchor_n = X_anchor / np.linalg.norm(X_anchor, axis=1, keepdims=True)
    Y_anchor_n = Y_anchor / np.linalg.norm(Y_anchor, axis=1, keepdims=True)

    anchor_shifts = 1 - np.sum(X_anchor_n * Y_anchor_n, axis=1)
    all_shifts = 1 - np.sum(X_all_n * Y_all_n, axis=1)

    anchor_mean = float(np.mean(anchor_shifts))
    anchor_std = float(np.std(anchor_shifts))
    threshold = anchor_mean + 2 * anchor_std

    results = sorted(zip(non_anchor, all_shifts), key=lambda x: x[1], reverse=True)

    out_csv = paths.word_csv(paths.SHIFT_RESULTS_CSV)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("word,shift,significant\n")
        for word, shift in results:
            sig = "True" if shift > threshold else "False"
            f.write(f"{word},{shift:.6f},{sig}\n")

    meta = {
        "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "threshold": float(threshold),
    }
    out_meta = paths.word_csv(paths.SHIFT_META_JSON)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAnchor mean: {anchor_mean:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("CSV saved:", out_csv)
    print("Meta saved:", out_meta)
    print("Done.")


if __name__ == "__main__":
    main()

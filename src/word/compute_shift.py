import json
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from gensim.models import KeyedVectors

import paths
from word.vocab import (
    ANCHOR_SIZE,
    TOP_N,
    _get_count,
    load_anchor_words,
    select_top_non_anchor_words,
)


def distribution_stats(shifts: np.ndarray) -> dict:
    s = np.asarray(shifts, dtype=np.float64)
    mean = float(np.mean(s))
    std = float(np.std(s))
    p25, p75 = np.percentile(s, [25, 75])
    p90, p95, p99 = np.percentile(s, [90, 95, 99])
    centered = s - mean
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    skew = m3 / (m2**1.5) if m2 > 0 else 0.0
    m4 = float(np.mean(centered**4))
    excess_kurtosis = m4 / (m2**2) - 3.0 if m2 > 0 else 0.0
    return {
        "n": int(s.size),
        "mean": mean,
        "std": std,
        "median": float(np.median(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "skew": float(skew),
        "excess_kurtosis": float(excess_kurtosis),
        "frac_above_mean": float(np.mean(s > mean)),
    }


def main() -> None:
    paths.word_dir()
    rm_path = paths.alignment_kv(paths.KV_RENMIN)
    wb_path = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)

    print("Loading models...")
    model_rm = KeyedVectors.load(str(rm_path), mmap="r")
    model_wb = KeyedVectors.load(str(wb_path), mmap="r")

    wb_raw_path = paths.alignment_kv(paths.KV_WEIBO_RAW)
    model_wb_raw = (
        KeyedVectors.load(str(wb_raw_path), mmap="r") if wb_raw_path.is_file() else model_wb
    )
    anchor_words = load_anchor_words(model_rm, model_wb_raw, n=ANCHOR_SIZE)
    print(f"Anchor words (excluded): {len(anchor_words)}")

    top_words = select_top_non_anchor_words(
        model_rm, model_wb, top_n=TOP_N, anchor_words=anchor_words
    )
    print(f"Top-{TOP_N} non-anchor by min frequency: {len(top_words)}")

    X_all = np.array([model_rm[w] for w in top_words])
    Y_all = np.array([model_wb[w] for w in top_words])
    X_all_n = X_all / np.linalg.norm(X_all, axis=1, keepdims=True)
    Y_all_n = Y_all / np.linalg.norm(Y_all, axis=1, keepdims=True)
    all_shifts = 1 - np.sum(X_all_n * Y_all_n, axis=1)

    meta = distribution_stats(all_shifts)
    meta["anchors_excluded"] = len(anchor_words)
    meta["top_n"] = TOP_N
    wb_has_count = _get_count(model_wb, top_words[0]) is not None if top_words else False
    if wb_has_count:
        meta["freq_sort"] = "min(count_rm, count_wb) descending, non-anchor"
    else:
        meta["freq_sort"] = "count_rm descending, non-anchor (aligned weibo kv has no count)"

    ranked = sorted(zip(top_words, all_shifts), key=lambda x: x[1], reverse=True)
    rank_by_freq = {w: i + 1 for i, w in enumerate(top_words)}

    out_csv = paths.word_csv(paths.SHIFT_RESULTS_CSV)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("word,shift,rank\n")
        for word, shift in ranked:
            f.write(f"{word},{shift:.6f},{rank_by_freq[word]}\n")

    out_meta = paths.word_csv(paths.SHIFT_META_JSON)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n--- Shift distribution (Top-{TOP_N} non-anchor) ---")
    for key in (
        "n",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "p25",
        "p75",
        "p90",
        "p95",
        "p99",
        "skew",
        "excess_kurtosis",
        "frac_above_mean",
    ):
        val = meta[key]
        print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")
    print("CSV saved:", out_csv)
    print("Meta saved:", out_meta)
    print("Done.")


if __name__ == "__main__":
    main()

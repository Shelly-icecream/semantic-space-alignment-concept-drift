"""Shared vocabulary selection for word-level metrics (anchors excluded)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gensim.models import KeyedVectors

import paths

ANCHOR_SIZE = 2500


def is_good_anchor(word: str) -> bool:
    if len(word) < 2:
        return False
    if word.isdigit():
        return False
    if re.search(r"[a-zA-Z]", word):
        return False
    if re.search(r"[^\u4e00-\u9fff]", word):
        return False
    return True


TOP_N = 5000


def is_valid_word(w: str) -> bool:
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,}", w))


def _get_count(model: KeyedVectors, w: str):
    try:
        return model.get_vecattr(w, "count")
    except KeyError:
        return None


def word_freq(model_rm: KeyedVectors, model_wb: KeyedVectors, w: str) -> float:
    c_rm = _get_count(model_rm, w)
    c_wb = _get_count(model_wb, w)
    counts = [c for c in (c_rm, c_wb) if c is not None]
    if counts:
        return float(min(counts))
    idx_rm = model_rm.key_to_index.get(w)
    idx_wb = model_wb.key_to_index.get(w)
    if idx_rm is not None and idx_wb is not None:
        return -float(min(idx_rm, idx_wb))
    return 0.0


def load_anchor_words(
    model_rm: KeyedVectors,
    model_wb: KeyedVectors,
    *,
    n: int = ANCHOR_SIZE,
) -> set[str]:
    """Return anchor set used for Q (file if present, else same rule as spatial_alignment_v2)."""
    anchor_path = paths.alignment_kv(paths.KV_ANCHOR_RENMIN_SORTED)
    if anchor_path.is_file():
        model_anchor = KeyedVectors.load(str(anchor_path), mmap="r")
        return set(model_anchor.key_to_index.keys())

    common_words = set(model_rm.key_to_index.keys()) & set(model_wb.key_to_index.keys())
    word_scores: list[tuple[str, float]] = []
    for word in common_words:
        if not is_good_anchor(word):
            continue
        c_rm = _get_count(model_rm, word)
        c_wb = _get_count(model_wb, word)
        counts = [c for c in (c_rm, c_wb) if c is not None]
        if not counts:
            continue
        word_scores.append((word, float(min(counts))))

    if word_scores:
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return {w for w, _ in word_scores[:n]}

    anchor: list[str] = []
    for word in common_words:
        if is_good_anchor(word):
            anchor.append(word)
        if len(anchor) >= n:
            break
    return set(anchor)


def select_top_non_anchor_words(
    model_rm: KeyedVectors,
    model_wb: KeyedVectors,
    *,
    top_n: int = TOP_N,
    anchor_words: set[str] | None = None,
) -> list[str]:
    """Morphology filter, exclude anchors, freq-sort, take top_n."""
    if anchor_words is None:
        wb_for_anchor = model_wb
        wb_raw = paths.alignment_kv(paths.KV_WEIBO_RAW)
        if wb_raw.is_file():
            wb_for_anchor = KeyedVectors.load(str(wb_raw), mmap="r")
        anchor_words = load_anchor_words(model_rm, wb_for_anchor)

    common_raw = set(model_rm.key_to_index.keys()) & set(model_wb.key_to_index.keys())
    candidates = [w for w in common_raw if is_valid_word(w) and w not in anchor_words]
    candidates.sort(key=lambda w: word_freq(model_rm, model_wb, w), reverse=True)
    return candidates[:top_n]

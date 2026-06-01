import sys
from pathlib import Path

_align = Path(__file__).resolve().parent
_src = _align.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_align))

from utils import *  # noqa: E402,F403
import paths  # noqa: E402

import random
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors  # noqa: E402

RENMIN_KV = paths.alignment_kv(paths.KV_RENMIN)
WEIBO_KV = paths.alignment_kv(paths.KV_WEIBO_RAW)
WEIBO_ALIGNED_KV = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)

OUT_CSV = paths.alignment_dir() / "reverse_alignment_check_from_saved_vectors.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading People's Daily vectors...")
model_renmin = KeyedVectors.load(str(RENMIN_KV), mmap="r")

print("Loading original Weibo vectors...")
model_weibo = KeyedVectors.load(str(WEIBO_KV), mmap="r")

print("Loading aligned Weibo vectors...")
model_weibo_aligned = KeyedVectors.load(str(WEIBO_ALIGNED_KV), mmap="r")

print("People's Daily vocab size:", len(model_renmin.index_to_key))
print("Weibo vocab size:", len(model_weibo.index_to_key))
print("Aligned Weibo vocab size:", len(model_weibo_aligned.index_to_key))
print("Vector dim:", model_renmin.vector_size)

def get_good_vocab_by_count(model, topn=50000):
    vocab_scores = []

    for word in model.index_to_key:
        if not is_good_anchor(word):
            continue

        try:
            count = model.get_vecattr(word, "count")
        except KeyError:
            count = 1

        vocab_scores.append((word, count))

    vocab_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in vocab_scores[:topn]]


def build_normalized_matrix(model, words, device):
    vecs = np.array([model[w] for w in words], dtype=np.float32)

    vecs = torch.from_numpy(vecs).to(device).float()
    vecs = vecs / (torch.norm(vecs, dim=1, keepdim=True) + 1e-10)

    return vecs

def find_unaligned_high_cos_pairs(
    A_vecs,
    B_vecs,
    A_words,
    B_words,
    topk=20,
    b_chunk_size=10000,
    exclude_same_word=True,
):

    all_pairs = []

    num_A = A_vecs.shape[0]
    num_B = B_vecs.shape[0]

    for i in range(num_A):
        a_vec = A_vecs[i:i + 1]
        a_word = A_words[i]

        local_scores = []
        local_indices = []

        for start in range(0, num_B, b_chunk_size):
            end = min(start + b_chunk_size, num_B)
            b_chunk = B_vecs[start:end]

            sims = torch.matmul(a_vec, b_chunk.T).squeeze(0)

            chunk_topk = min(topk, sims.shape[0])
            vals, idx = torch.topk(sims, k=chunk_topk)

            local_scores.append(vals.detach().cpu())
            local_indices.append(idx.detach().cpu() + start)

        local_scores = torch.cat(local_scores)
        local_indices = torch.cat(local_indices)

        final_k = min(topk * 5, local_scores.shape[0])
        vals, idx = torch.topk(local_scores, k=final_k)

        kept = 0

        for score, b_idx in zip(vals.tolist(), local_indices[idx].tolist()):
            b_word = B_words[b_idx]

            if exclude_same_word and a_word == b_word:
                continue

            all_pairs.append({
                "a_word": a_word,
                "b_word": b_word,
                "raw_cos": float(score),
                "a_index": i,
                "b_index": b_idx,
            })

            kept += 1

            if kept >= topk:
                break

    return all_pairs

def evaluate_with_saved_aligned_vectors(
    pairs,
    A_vecs,
    B_aligned_vecs,
):
    results = []

    for pair in pairs:
        a_idx = pair["a_index"]
        b_idx = pair["b_index"]

        raw_cos = pair["raw_cos"]
        aligned_cos = torch.sum(A_vecs[a_idx] * B_aligned_vecs[b_idx]).item()
        drop = raw_cos - aligned_cos

        results.append({
            "a_word": pair["a_word"],
            "b_word": pair["b_word"],
            "raw_cos": raw_cos,
            "aligned_cos": aligned_cos,
            "drop": drop,
        })

    return pd.DataFrame(results)


# A：People's Daily
# B：Weibo
# B_aligned：aligned Weibo

A_pool = get_good_vocab_by_count(model_renmin, topn=20000)
B_pool = get_good_vocab_by_count(model_weibo, topn=50000)

B_pool = [w for w in B_pool if w in model_weibo_aligned]

random.seed(42)

sample_A_num = 1000
A_words = random.sample(A_pool, min(sample_A_num, len(A_pool)))
B_words = B_pool

print("\nA query words:", len(A_words))
print("B candidate words:", len(B_words))

A_vecs = build_normalized_matrix(model_renmin, A_words, device)
B_vecs = build_normalized_matrix(model_weibo, B_words, device)
B_aligned_vecs = build_normalized_matrix(model_weibo_aligned, B_words, device)

raw_pairs = find_unaligned_high_cos_pairs(
    A_vecs=A_vecs,
    B_vecs=B_vecs,
    A_words=A_words,
    B_words=B_words,
    topk=20,
    b_chunk_size=10000,
    exclude_same_word=True,
)

df_raw = pd.DataFrame(raw_pairs)

if df_raw.empty:
    raise ValueError("No raw pairs found. Please check A_words, B_words, or topk.")

min_raw_cos = 0.2

df_raw = df_raw[df_raw["raw_cos"] >= min_raw_cos]
df_raw = df_raw.sort_values("raw_cos", ascending=False).head(500)

if df_raw.empty:
    raise ValueError(f"No pairs selected after applying min_raw_cos = {min_raw_cos}.")

print("\nSelected high-raw-cos pairs:", len(df_raw))
print(df_raw[["a_word", "b_word", "raw_cos"]].head(20))

df_eval = evaluate_with_saved_aligned_vectors(
    pairs=df_raw.to_dict("records"),
    A_vecs=A_vecs,
    B_aligned_vecs=B_aligned_vecs,
)

df_eval = df_eval.sort_values("drop", ascending=False)

print("\n========== Reverse Alignment Check ==========")
print("Mean raw cosine:", df_eval["raw_cos"].mean())
print("Mean aligned cosine:", df_eval["aligned_cos"].mean())
print("Mean drop:", df_eval["drop"].mean())
print("Drop ratio:", (df_eval["drop"] > 0).mean())

print("\nTop dropped pairs:")
print(df_eval[["a_word", "b_word", "raw_cos", "aligned_cos", "drop"]].head(30))

df_eval.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("\nSaved:", OUT_CSV)
from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import re

import numpy as np
import torch
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans  # noqa: F401 — used by spatial_alignment_v2 via import *
from sklearn.decomposition import PCA  # noqa: F401

import paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def is_good_anchor(word):
    if len(word) < 2:
        return False
    if word.isdigit():
        return False
    if re.search(r"[a-zA-Z]", word):
        return False
    if re.search(r"[^\u4e00-\u9fff]", word):
        return False
    return True


def torch_normalize(vectors):
    norms = torch.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + 1e-10)


def compute_s(source_vectors, target_vectors, k=10, batch_size=5000):
    num_source = source_vectors.shape[0]
    r_values = torch.zeros(num_source, device=device)

    for i in range(0, num_source, batch_size):
        end_idx = min(i + batch_size, num_source)
        batch = source_vectors[i:end_idx]

        similarities = torch.matmul(batch, target_vectors.T)
        vals, _ = torch.topk(similarities, k, dim=1)
        r_values[i:end_idx] = vals.mean(dim=1)

    return r_values


paths.alignment_dir()
model_renmin = KeyedVectors.load(str(paths.alignment_kv(paths.KV_RENMIN)), mmap="r")
model_weibo = KeyedVectors.load(str(paths.alignment_kv(paths.KV_WEIBO_RAW)), mmap="r")

krenmin = list(model_renmin.key_to_index)
kweibo = list(model_weibo.key_to_index)

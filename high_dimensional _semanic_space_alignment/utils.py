from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import re

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体，Windows一般有
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def is_good_anchor(word):
    # 去掉长度太短的词：单字一般不要
    if len(word) < 2:
        return False
    # 去掉纯数字
    if word.isdigit():
        return False
    # 去掉包含英文字母的词
    if re.search(r"[a-zA-Z]", word):
        return False
    # 去掉包含明显符号/标点的词
    if re.search(r"[^\u4e00-\u9fff]", word):
        return False
    return True

# 归一化函数
def torch_normalize(vectors):
    norms = torch.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + 1e-10)

# 分批算相似度函数
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

model_renmin = KeyedVectors.load(r"D:\高代大作业\renmin_fast.kv")
model_weibo = KeyedVectors.load(r"D:\高代大作业\weibo_fast.kv")

krenmin = list(model_renmin.key_to_index)
kweibo = list(model_weibo.key_to_index)
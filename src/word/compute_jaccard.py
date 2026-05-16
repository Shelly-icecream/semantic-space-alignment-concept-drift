import numpy as np
import re
from gensim.models import KeyedVectors

print("加载数据...")
model_rm = KeyedVectors.load("renmin_fast.kv")
model_wb = KeyedVectors.load("weibo_aligned_v2.kv")

# 找共有词并清洗
common_raw = list(set(model_rm.key_to_index.keys()) & set(model_wb.key_to_index.keys()))
print(f"原始共有词: {len(common_raw)}")

def is_valid_word(w):
    return bool(re.fullmatch(r'[\u4e00-\u9fff]{2,}', w))

common_filtered = [w for w in common_raw if is_valid_word(w)]
print(f"过滤后: {len(common_filtered)}")

# 分离锚点词
anchor_words = ['的','一','是','了','我','不','在','人','他','有','这','个','们','中','来','上','大','为','就','和','说','地','也','对','到','要','下','会','时','出','那','过','你','她','能','前','它','所','都','后']
anchor_words = [w for w in anchor_words if w in common_raw]
non_anchor = [w for w in common_filtered if w not in anchor_words]

# 取前 1000 个高频词算 Jaccard（多了太慢）
n_words = min(1000, len(non_anchor))
non_anchor = non_anchor[:n_words]
print(f"用于 Jaccard 的词数: {n_words}")

# 提取向量
X = np.array([model_rm[w] for w in non_anchor])
Y = np.array([model_wb[w] for w in non_anchor])

# 归一化
X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)

# 相似度矩阵
X_sim = X_norm @ X_norm.T
Y_sim = Y_norm @ Y_norm.T

# 找 k 个最近邻
def get_knn(sim, k):
    np.fill_diagonal(sim, -1)
    return np.argsort(sim, axis=1)[:, -k:]

for k in [10, 50]:
    X_knn = get_knn(X_sim.copy(), k)
    Y_knn = get_knn(Y_sim.copy(), k)

    jaccards = []
    for i in range(n_words):
        inter = len(set(X_knn[i]) & set(Y_knn[i]))
        union = len(set(X_knn[i]) | set(Y_knn[i]))
        jaccards.append(inter / union if union > 0 else 0)

    avg_j = np.mean(jaccards)
    print(f"\nk={k}: 平均 Jaccard = {avg_j:.4f}")
    print(f"  前10个词的值: {[f'{jaccards[i]:.4f}' for i in range(10)]}")

print("\n完成！")
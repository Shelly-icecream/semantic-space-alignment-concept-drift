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

def debug_word(word):
    print(f"'{word}' 在人民日报的邻居: {model_renmin.most_similar(word, topn=5)}")
    print(f"'{word}' 在微博的邻居: {model_weibo.most_similar(word, topn=5)}")

model_renmin = KeyedVectors.load(r"D:\高代大作业\renmin_fast.kv")
model_weibo = KeyedVectors.load(r"D:\高代大作业\weibo_fast.kv")

krenmin = list(model_renmin.key_to_index)
kweibo = list(model_weibo.key_to_index)

# ==================================================
# 选取 2500 个公共高频词作为锚点词
# ==================================================

common_words = set(krenmin) & set(kweibo)

word_scores = []

for word in common_words:
    if not is_good_anchor(word):
        continue
    try:
        count_r = model_renmin.get_vecattr(word, "count")
        count_w = model_weibo.get_vecattr(word, "count")
        score = min(count_r, count_w)
        word_scores.append((word, score))
    except KeyError:
        continue

word_scores.sort(key=lambda x: x[1], reverse=True)
anchor = [word for word, score in word_scores[:2500]]

if len(anchor) == 0:
    print("警告：模型中没有 count 属性，改用人民日报词表顺序选取公共词。")
    anchor = []
    for word in common_words:
        if is_good_anchor(word):
            anchor.append(word)
        if len(anchor) == 2500:
            break

print("锚点数", len(anchor))
print("特征维度", len(model_renmin[krenmin[0]]))
print("词数", len(krenmin))

# 锚点词的索引列表
lren = [model_renmin.key_to_index[r] for r in anchor]
lwei = [model_weibo.key_to_index[r] for r in anchor]

X = model_renmin.vectors[lren]
Y = model_weibo.vectors[lwei]

X_t = torch.from_numpy(X).to(device).float()
Y_t = torch.from_numpy(Y).to(device).float()

X0 = torch_normalize(X_t)
Y0 = torch_normalize(Y_t)

# --- 初始设置 ---
max_iter = 5
epsilon = 1e-4
last_avg_sim = 0

X = X0.clone()
Y = Y0.clone()

for i in range(max_iter):

    M = torch.matmul(Y.T, X)

    # 奇异值分解，计算旋转矩阵
    U, sigma, VT = torch.linalg.svd(M.cpu())
    Q = torch.matmul(U, VT).to(device)

    # 剔除万金油词
    P = torch.matmul(Y0, Q)
    current_sims = torch.sum(X0 * P, dim=1)
    r_Y_x = compute_s(X0, P)
    r_X_y = compute_s(P, X0)
    CSLS = 2 * current_sims - r_Y_x - r_X_y
    CSLS = CSLS.cpu().numpy()
    threshold = np.percentile(CSLS, 20)
    keep_idx = np.where(CSLS > threshold)[0]

    avg_sim = np.mean(CSLS[keep_idx])
    avg_cos = torch.mean(current_sims[keep_idx]).item()
    print(f"第 {i + 1} 次迭代, 保留词平均相似度: {avg_sim:.4f}")
    print(f"  保留词平均余弦相似度: {avg_cos:.4f} ")

    if abs(avg_sim - last_avg_sim) < epsilon:
        print("算法已收敛。")
        break

    last_avg_sim = avg_sim

    # 根据当前 Q 在全部初始锚点中重新筛选可靠锚点，用于下一轮训练
    X = X0[keep_idx]
    Y = Y0[keep_idx]

# ==================================================
# 导出
# ==================================================
final_Q = Q.cpu().numpy()

all_weibo_vectors = model_weibo.vectors
all_weibo_vectors_norm = all_weibo_vectors / (
    np.linalg.norm(all_weibo_vectors, axis=1, keepdims=True) + 1e-10
)

aligned_weibo_vectors = all_weibo_vectors_norm @ final_Q

weibo_vocab = model_weibo.index_to_key

aligned_kv = KeyedVectors(vector_size=model_weibo.vector_size)
aligned_kv.add_vectors(weibo_vocab, aligned_weibo_vectors)

aligned_kv.save(r"D:\高代大作业\weibo_aligned_v2.kv")

print("对齐后的微博词向量已保存。")


# ==================================================
# 误差分析
# ==================================================
# 1 平均词级残差
n=len(anchor)
residuals = torch.norm(Y0 @ Q - X0, p='fro')
avg_residual = residuals / np.sqrt(n)
print("平均词级残差：", avg_residual)

# 2 坏锚点词
idx = np.argsort(CSLS)[:50]
for i in idx:
    print(anchor[i], "分数：", CSLS[i])

# 3 好锚点词
idx=np.argsort(-CSLS)[:50]
for i in idx:
    debug_word(anchor[i])

# 4 KMeans
model_weibo_aligned = KeyedVectors.load(r"D:\高代大作业\weibo_aligned_v2.kv")
print("模型加载完成")
probe_words = [
    # 政治/社会稳定词
    "中国", "政府", "人民", "国家", "社会", "政策",

    # 经济/发展稳定词
    "经济", "发展", "资本", "市场", "企业", "金融",

    # 教育/科技/文化
    "科技", "教育", "语文", "艺术", "文化", "创新",

    # 微博/网络文化词
    "微博", "博主", "转发", "红包", "豆瓣", "奥特曼", "绿茶", "百合"
]
all_vectors = []
all_labels = []
for word in probe_words:
    # 人民日报中的词
    if word in model_renmin:
        vec_r = model_renmin[word]
        all_vectors.append(vec_r)
        all_labels.append(f"{word}_人民日报")
    # 对齐后的微博中的词
    if word in model_weibo_aligned:
        vec_w = model_weibo_aligned[word]
        all_vectors.append(vec_w)
        all_labels.append(f"{word}_微博")
all_vectors = np.array(all_vectors)
all_vectors = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-10)
print("参与聚类的词数量：", len(all_labels))
print(all_labels[:10])

n_clusters = 4 # 类别数量
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(all_vectors)
cluster_result = {}
for label, cluster_id in zip(all_labels, clusters):
    if cluster_id not in cluster_result:
        cluster_result[cluster_id] = []
    cluster_result[cluster_id].append(label)

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(all_vectors)
plt.figure(figsize=(14, 10))
for i, label in enumerate(all_labels):
    x, y = vectors_2d[i]
    cluster_id = clusters[i]
    if label.endswith("_人民日报"):
        marker = "o"
        short_label = label.replace("_人民日报", "_R")
    else:
        marker = "^"
        short_label = label.replace("_微博", "_W")
    plt.scatter(
        x,
        y,
        s=90,
        marker=marker,
        label=f"C{cluster_id + 1}" if i == 0 else None
    )
    plt.text(
        x + 0.008,
        y + 0.008,
        short_label,
        fontsize=9
    )
print("\n========== 聚类结果 ==========")
for cluster_id in sorted(cluster_result.keys()):
    print(f"Cluster {cluster_id + 1}:")
    for word in cluster_result[cluster_id]:
        print("   ", word)
    print()
plt.title("人民日报词向量与对齐后微博词向量 KMeans 聚类可视化")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"D:\高代大作业\cluster_visualization_clear.png", dpi=300)
plt.show()




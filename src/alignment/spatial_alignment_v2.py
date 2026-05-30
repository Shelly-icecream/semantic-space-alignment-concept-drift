import sys
from pathlib import Path
from utils import *  # noqa: E402,F403
import paths  # noqa: E402
from gensim.models import KeyedVectors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_align = Path(__file__).resolve().parent
_src = _align.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_align))

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def debug_word(word):
    print(f"'{word}' Neighbors on People's Daily: {model_renmin.most_similar(word, topn=5)}")
    print(f"'{word}' Neighbors on Weibo: {model_weibo.most_similar(word, topn=5)}")


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
    print("Warning: The model does not have the count attribute. Common words are selected in the "
          "order of the People's Daily vocabulary list instead.")
    anchor = []
    for word in common_words:
        if is_good_anchor(word):
            anchor.append(word)
        if len(anchor) == 2500:
            break

print("Anchor points", len(anchor))
print("Feature Dimension", len(model_renmin[krenmin[0]]))
print("Word count", len(krenmin))

lren = [model_renmin.key_to_index[r] for r in anchor]
lwei = [model_weibo.key_to_index[r] for r in anchor]

X = model_renmin.vectors[lren]
Y = model_weibo.vectors[lwei]

X_t = torch.from_numpy(X).to(device).float()
Y_t = torch.from_numpy(Y).to(device).float()

X0 = torch_normalize(X_t)
Y0 = torch_normalize(Y_t)

max_iter = 5
epsilon = 1e-4
last_avg_sim = 0

X = X0.clone()
Y = Y0.clone()

for i in range(max_iter):
    M = torch.matmul(Y.T, X)
    U, sigma, VT = torch.linalg.svd(M.cpu())
    Q = torch.matmul(U, VT).to(device)

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
    print(f"Iteration {i + 1}, average similarity of retained words: {avg_sim:.4f}")
    print(f"Average Cosine Similarity of Reserved Words: {avg_cos:.4f} ")

    if abs(avg_sim - last_avg_sim) < epsilon:
        print("The algorithm has converged.")
        break

    last_avg_sim = avg_sim
    X = X0[keep_idx]
    Y = Y0[keep_idx]

final_Q = Q.cpu().numpy()

all_weibo_vectors = model_weibo.vectors
all_weibo_vectors_norm = all_weibo_vectors / (
    np.linalg.norm(all_weibo_vectors, axis=1, keepdims=True) + 1e-10
)

aligned_weibo_vectors = all_weibo_vectors_norm @ final_Q
weibo_vocab = model_weibo.index_to_key

aligned_kv = KeyedVectors(vector_size=model_weibo.vector_size)
aligned_kv.add_vectors(weibo_vocab, aligned_weibo_vectors)

out_kv = paths.alignment_kv(paths.KV_WEIBO_ALIGNED)
aligned_kv.save(str(out_kv))
print("Aligned Weibo word vectors have been saved:", out_kv)

n = len(anchor)
residuals = torch.norm(Y0 @ Q - X0, p="fro")
avg_residual = residuals / np.sqrt(n)
print("Average word-level residual:", avg_residual)

idx = np.argsort(CSLS)[:50]
for j in idx:
    print(anchor[j], "Score:", CSLS[j])

idx = np.argsort(-CSLS)[:50]
for j in idx:
    debug_word(anchor[j])

model_weibo_aligned = KeyedVectors.load(str(out_kv), mmap="r")
print("Model loading completed")
probe_words = [
    "中国",
    "政府",
    "人民",
    "国家",
    "社会",
    "政策",
    "经济",
    "发展",
    "资本",
    "市场",
    "企业",
    "金融",
    "科技",
    "教育",
    "语文",
    "艺术",
    "文化",
    "创新",
    "微博",
    "博主",
    "转发",
    "红包",
    "豆瓣",
    "奥特曼",
    "绿茶",
    "百合",
]
all_vectors = []
all_labels = []
for word in probe_words:
    if word in model_renmin:
        vec_r = model_renmin[word]
        all_vectors.append(vec_r)
        all_labels.append(f"{word}_People's Daily")
    if word in model_weibo_aligned:
        vec_w = model_weibo_aligned[word]
        all_vectors.append(vec_w)
        all_labels.append(f"{word}_Weibo")
all_vectors = np.array(all_vectors)
all_vectors = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-10)
print("Number of words involved in clustering:", len(all_labels))
print(all_labels[:10])

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
    if label.endswith("_People's Daily"):
        marker = "o"
        short_label = label.replace("_People's Daily", "_R")
    else:
        marker = "^"
        short_label = label.replace("_Weibo", "_W")
    plt.scatter(x, y, s=90, marker=marker, label=f"C{cluster_id + 1}" if i == 0 else None)
    plt.text(x + 0.008, y + 0.008, short_label, fontsize=9)

print("\n========== Clustering Results ==========")
for cluster_id in sorted(cluster_result.keys()):
    print(f"Cluster {cluster_id + 1}:")
    for word in cluster_result[cluster_id]:
        print("   ", word)
    print()

plt.title("Visualization of KMeans Clustering for People's Daily Word Vectors and Aligned Weibo Word Vectors")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.tight_layout()
fig_path = paths.fig_alignment_dir() / "cluster_visualization_clear.png"
plt.savefig(str(fig_path), dpi=300)
print("Saved figure:", fig_path)
plt.show()

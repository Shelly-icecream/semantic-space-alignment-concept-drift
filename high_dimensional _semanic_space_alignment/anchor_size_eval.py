from utils import *

def train_alignment_Q(X0, Y0, max_iter=5, epsilon=1e-4):
    X = X0.clone()
    Y = Y0.clone()
    last_avg_sim = 0
    for i in range(max_iter):
        M = Y.T@ X
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
        print(f"第 {i + 1} 次迭代, 保留词平均相似度: {avg_sim:.4f}")

        if abs(avg_sim - last_avg_sim) < epsilon:
            print("算法已收敛。")
            break

        last_avg_sim = avg_sim
        X = X0[keep_idx]
        Y = Y0[keep_idx]

    return Q

# ==================================================
# 训练候选：5000 测试集：1000
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
anchor = [word for word, score in word_scores[:6000]]

if len(anchor) == 0:
    print("警告：模型中没有 count 属性，改用人民日报词表顺序选取公共词。")
    anchor = []
    for word in common_words:
        if is_good_anchor(word):
            anchor.append(word)
        if len(anchor) == 6000:
            break

# 固定测试集，保证公平
rng = np.random.default_rng(42)
indices = rng.permutation(len(anchor))
test_words = [anchor[i] for i in indices[:1000]]

# 训练集和测试集不重合
test_set = set(test_words)
train_pool = [word for word in anchor if word not in test_set]

lttr = [model_renmin.key_to_index[r] for r in test_words]
lttw = [model_weibo.key_to_index[r] for r in test_words]
x = model_renmin.vectors[lttr]
y = model_weibo.vectors[lttw]
x_t = torch.from_numpy(x).to(device).float()
y_t = torch.from_numpy(y).to(device).float()

x0 = torch_normalize(x_t)
y0 = torch_normalize(y_t)
anchor_nums = []
cos = []
avg_residuals = []
for t in range(500,5001,500):
    lren = [model_renmin.key_to_index[r] for r in train_pool[:t]]
    lwei = [model_weibo.key_to_index[r] for r in train_pool[:t]]

    X = model_renmin.vectors[lren]
    Y = model_weibo.vectors[lwei]

    X_t = torch.from_numpy(X).to(device).float()
    Y_t = torch.from_numpy(Y).to(device).float()

    X0 = torch_normalize(X_t)
    Y0 = torch_normalize(Y_t)

    Q = train_alignment_Q(X0, Y0)
    # 在固定测试集上评估
    P = torch.matmul(y0, Q)

    current_sims = torch.sum(x0 * P, dim=1)
    avg_cos = torch.mean(current_sims).item()

    residual = torch.norm(P - x0, p="fro")
    avg_residual = (residual / np.sqrt(len(test_words))).item()

    print("测试集平均余弦：", avg_cos)
    print("测试集平均词级残差：", avg_residual)

    anchor_nums.append(t)
    cos.append(avg_cos)
    avg_residuals.append(avg_residual)

plt.figure(figsize=(10, 6))
plt.plot(anchor_nums, cos, marker="o")
plt.xlabel("训练锚点数量")
plt.ylabel("测试集平均余弦相似度")
plt.title("不同锚点数量下的测试集平均余弦相似度")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(r"D:\高代大作业\anchor_size_test_cos_curve.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(anchor_nums, avg_residuals, marker="o")
plt.xlabel("训练锚点数量")
plt.ylabel("测试集平均词级残差")
plt.title("不同锚点数量下的测试集平均词级残差")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(r"D:\高代大作业\anchor_size_test_residual_curve.png", dpi=300)
plt.show()
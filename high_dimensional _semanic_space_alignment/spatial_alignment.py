from gensim.models import KeyedVectors
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model_renmin= KeyedVectors.load(r"D:\高代大作业\renmin_fast.kv")
model_weibo= KeyedVectors.load(r"D:\高代大作业\weibo_fast.kv")
krenmin=list(model_renmin.key_to_index)
kweibo=list(model_weibo.key_to_index)
anchor=list(set(krenmin)&set(kweibo))
"""
print("锚点数",len(anchor))
print("特征维度",len(model_renmin[krenmin[0]]))
print(len(model_weibo[krenmin[0]]))
print("词数",len(krenmin))
print(len(kweibo))
"""
#锚点词的索引列表
lren=[model_renmin.key_to_index[r] for r in anchor]
lwei=[model_weibo.key_to_index[r] for r in anchor]

X=model_renmin.vectors[lren]
Y=model_weibo.vectors[lwei]

# 归一化函数
def torch_normalize(vectors):
    # 使用 torch 的范数计算，保持在 GPU 上
    norms = torch.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + 1e-10)

# 分批算相似度函数
def compute_s(source_vectors, target_vectors, k=10, batch_size=5000):

    num_source = source_vectors.shape[0] # 总词数，约11万
    r_values = torch.zeros(num_source,device=device) # 存相似度的矩阵

    for i in range(0, num_source, batch_size):
        end_idx = min(i + batch_size, num_source)
        batch = source_vectors[i:end_idx]
        similarities=torch.matmul(batch,target_vectors.T)
        vals, _ = torch.topk(similarities, k, dim=1)
        r_values[i:end_idx] = vals.mean(dim=1)


    return r_values


def debug_word(word):
    # 打印该词在人民日报里的邻居
    print(f"'{word}' 在人民日报的邻居: {model_renmin.most_similar(word, topn=5)}")
    # 打印该词在微博里的邻居
    print(f"'{word}' 在微博的邻居: {model_weibo.most_similar(word, topn=5)}")


X_t = torch.from_numpy(X).to(device).float()
Y_t = torch.from_numpy(Y).to(device).float()

X0 = torch_normalize(X_t)
Y0 = torch_normalize(Y_t)

# --- 初始设置 ---
max_iter = 5          # 最大迭代次数
epsilon = 1e-4        # 收敛阈值
last_avg_sim = 0      # 记录上一次的平均相似度

X=X0.clone()
Y=Y0.clone()

for i in range(max_iter):
    """
    print(X.shape)
    print(Y.shape)
    """
    M=torch.matmul(Y.T ,X)
    """
    print(M.shape)
    """
    # 奇异值分解，计算旋转矩阵
    U, sigma, VT = torch.linalg.svd(M.cpu())
    Q=torch.matmul(U , VT).to(device)

    # 对于X空间中的每一个向量xi，计算它与Y空间中所有向量的内积。找出这些相似度中最大的k个值。
    # 剔除万金油词
    P=torch.matmul(Y0 , Q)
    # 1. 计算同词相似度
    current_sims = torch.sum(X0 * P, dim=1)
    # 2. 计算 r_Y(x): X 空间中的词在 Y 空间里的平均近邻距离
    r_Y_x=compute_s(X0, P)
    # 3. 计算 r_X(y): Y 空间中的词在 X 空间里的平均近邻距离
    r_X_y=compute_s(P, X0)
    # 4. 计算 CSLS 相似度
    CSLS= 2 * current_sims - r_Y_x - r_X_y
    CSLS=CSLS.cpu().numpy()
    threshold = np.percentile(CSLS, 20)  # 相似度最小的 20% 的“疑似偏移词”
    keep_idx = np.where(CSLS > threshold)[0]
    avg_sim = np.mean(CSLS[keep_idx]) # 计算平均值，它代表了当前旋转矩阵Q对这批保留锚点词的全局对齐效果
    avg_cos = torch.mean(current_sims[keep_idx]).item()  # 原始余弦相似度
    print(f"第 {i + 1} 次迭代, 保留词平均相似度: {avg_sim:.4f}")
    print(f"  保留词平均余弦相似度: {avg_cos:.4f} ")
    if abs(avg_sim - last_avg_sim) < epsilon:
        print("算法已收敛。")
        break
    last_avg_sim = avg_sim

    # 剔除可能发生了概念漂移的词
    X = X0[keep_idx]
    Y = Y0[keep_idx]
    """
    print(X.shape)
    """
"""误差分析
residuals=torch.norm(Y0@Q-X0,p='fro')
print("残差：",residuals)
idx=torch.argsort(current_sims)[:50]
for i in idx:
    print(anchor[i],"分数：",current_sims[i])
debug_word("马伊")
debug_word("公社")
"""
#导出
final_Q = Q.cpu().numpy()
all_weibo_vectors = model_weibo.vectors
all_weibo_vectors_norm = all_weibo_vectors / (np.linalg.norm(all_weibo_vectors, axis=1, keepdims=True) + 1e-10)
aligned_weibo_vectors = all_weibo_vectors_norm @ final_Q

weibo_vocab = model_weibo.index_to_key # 获取原有的词表顺序
aligned_kv = KeyedVectors(vector_size=model_weibo.vector_size) # 创建一个新的 KeyedVectors 实例
aligned_kv.add_vectors(weibo_vocab, aligned_weibo_vectors)
aligned_kv.save(r"D:\高代大作业\weibo_aligned.kv")









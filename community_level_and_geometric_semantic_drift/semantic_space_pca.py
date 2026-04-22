import numpy as np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# load word embedding

kv_weibo=KeyedVectors.load("/home/trojan/workspace/sample/HDSAD/weibo_embedding.kv",mmap="r")
kv_renmin=KeyedVectors.load("/home/trojan/workspace/sample/HDSAD/renmin_embedding.kv",mmap="r")

# get word matrix

word_matrix_weibo=kv_weibo.vectors
word_matrix_renmin=kv_renmin.vectors

# concat word matrix

word_matrix=np.concatenate([word_matrix_weibo,word_matrix_renmin],axis=0)

# get word index

n_weibo=kv_weibo.vectors.shape[0]
n_renmin=kv_renmin.vectors.shape[0]

pca=PCA(n_components=50,svd_solver='randomized',random_state=42)
pca.fit(word_matrix)

# get PCA transformed matrix

Z_all = pca.transform(word_matrix)


# Self-check

Z_weibo=pca.transform(word_matrix_weibo)
Z_renmin=pca.transform(word_matrix_renmin)
head = Z_all[:n_weibo]
tail = Z_all[n_weibo : n_weibo + n_renmin]

match_weibo = np.allclose(head, Z_weibo, rtol=1e-5, atol=1e-8)
match_renmin = np.allclose(tail, Z_renmin, rtol=1e-5, atol=1e-8)
err_weibo = np.max(np.abs(head - Z_weibo)) if head.size else 0.0
err_renmin = np.max(np.abs(tail - Z_renmin)) if tail.size else 0.0

# print self-check result

print("PCA self-check (concat transform vs split transform):")
print(f"  weibo block match: {match_weibo}, max |diff| = {err_weibo:.3e}")
print(f"  renmin block match: {match_renmin}, max |diff| = {err_renmin:.3e}")
if not (match_weibo and match_renmin):
    raise AssertionError("Self-check failed: Z_all slices != Z_weibo / Z_renmin")

# save PCA information

np.savez("/home/trojan/workspace/sample/HDSAD/semantic_space_pca.npz",Z_all=Z_all,n_weibo=n_weibo,n_renmin=n_renmin)
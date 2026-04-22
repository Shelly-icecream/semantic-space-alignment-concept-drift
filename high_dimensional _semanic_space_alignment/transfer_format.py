from gensim.models import KeyedVectors

path_official_txt = r"E:\sgns.renmin.word"
print("第一次加载 txt，请耐心等待...")
model_official = KeyedVectors.load_word2vec_format(path_official_txt, binary=False)

print("正在转换为 gensim 格式...")
model_official.save(r"D:\高代大作业\renmin_fast.kv")

path_official_txt = r"E:\sgns.weibo.word"
print("第一次加载 txt，请耐心等待...")
model_official = KeyedVectors.load_word2vec_format(path_official_txt, binary=False)

print("正在转换为 gensim 格式...")
model_official.save(r"D:\高代大作业\weibo_fast.kv")

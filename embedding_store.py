# embedding_store.py

"""
这个文件负责“检索”这一层能力。

当前版本做了 3 件事：

1. 本地 embedding 向量检索（语义检索）
2. TF-IDF 关键词检索
3. 混合检索（把两种结果合并）

为什么要这样做？

- 向量检索擅长找“意思接近”的内容
- 关键词检索擅长找“字面命中”的内容

两者结合后，RAG 的召回效果通常会更稳。
"""

import numpy as np
import faiss

# 本地 embedding 模型
from sentence_transformers import SentenceTransformer

# TF-IDF 关键词检索
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载本地 embedding 模型
# 第一次运行时会自动下载模型，以后就会直接本地加载
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_index(chunk_items):
    """
    作用：
        用本地 embedding 模型，把所有 chunk 文本转成向量，
        然后建立 FAISS 向量索引。

    参数：
        chunk_items : 列表，每个元素是一个字典，例如：
            {
                "chunk_id": 0,
                "text": "这是一段文本",
                "page": 1,
                "source": "xxx.pdf"
            }

    返回：
        index      : FAISS 向量索引
        embeddings : 所有 chunk 对应的向量
    """
    # 取出所有 chunk 的文本
    texts = [item["text"] for item in chunk_items]

    # 把文本转成 embedding 向量
    embeddings = model.encode(texts)

    # 转成 numpy 数组，并指定为 float32
    # FAISS 对数据类型有要求
    embeddings = np.array(embeddings).astype("float32")

    # 获取向量维度
    dimension = embeddings.shape[1]

    # 创建 FAISS 索引
    index = faiss.IndexFlatL2(dimension)

    # 把所有向量加入索引
    index.add(embeddings)

    return index, embeddings


def search_by_vector(query, index, chunk_items, top_k=3):
    """
    作用：
        用“向量检索”的方式，根据用户问题找到最相似的 chunk。

    参数：
        query       : 用户问题
        index       : FAISS 索引
        chunk_items : 原始 chunk 列表
        top_k       : 返回最相关的前几个结果

    返回：
        results : 检索结果列表
    """
    # 把用户问题转成 embedding 向量
    query_embedding = model.encode([query])

    # 转成 numpy 数组
    query_embedding = np.array(query_embedding).astype("float32")

    # 在 FAISS 索引中搜索最相近的 top_k 个向量
    distances, indices = index.search(query_embedding, top_k)

    results = []

    # indices[0] 是当前 query 对应的结果下标列表
    for rank, idx in enumerate(indices[0]):
        if idx < len(chunk_items):
            item = chunk_items[idx].copy()

            # 这里把向量检索分数存进去，方便后续调试
            # 注意：FAISS 的 L2 距离是“越小越相似”
            item["vector_score"] = float(distances[0][rank])

            results.append(item)

    return results


def build_keyword_index(chunk_items):
    """
    作用：
        用 TF-IDF 为所有 chunk 建立关键词检索索引。

    参数：
        chunk_items : 原始 chunk 列表

    返回：
        vectorizer    : TF-IDF 向量器
        tfidf_matrix  : 所有 chunk 的 TF-IDF 特征矩阵
    """
    # 取出所有 chunk 文本
    texts = [item["text"] for item in chunk_items]

    # 创建 TF-IDF 向量器
    # token_pattern 用默认就行，这里先保持简单
    vectorizer = TfidfVectorizer()

    # 对所有文本做 fit_transform
    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


def search_by_keyword(query, vectorizer, tfidf_matrix, chunk_items, top_k=3):
    """
    作用：
        用“关键词检索”的方式，根据用户问题找到最相关的 chunk。

    参数：
        query        : 用户问题
        vectorizer   : TF-IDF 向量器
        tfidf_matrix : chunk 的 TF-IDF 特征矩阵
        chunk_items  : 原始 chunk 列表
        top_k        : 返回最相关的前几个结果

    返回：
        results : 检索结果列表
    """
    # 把 query 转成 TF-IDF 向量
    query_vector = vectorizer.transform([query])

    # 计算 query 和所有 chunk 的相似度
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # 从大到小排序，取最相似的 top_k 个下标
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        if idx < len(chunk_items):
            item = chunk_items[idx].copy()

            # TF-IDF 相似度越大越相关
            item["keyword_score"] = float(similarities[idx])

            results.append(item)

    return results


def hybrid_search(query, vector_index, vectorizer, tfidf_matrix, chunk_items, top_k=4):
    """
    作用：
        同时做：
        1. 向量检索
        2. 关键词检索

        然后把两边结果合并、去重，形成混合检索结果。

    参数：
        query         : 用户问题
        vector_index  : FAISS 向量索引
        vectorizer    : TF-IDF 向量器
        tfidf_matrix  : TF-IDF 特征矩阵
        chunk_items   : 原始 chunk 列表
        top_k         : 最终返回多少条结果

    返回：
        merged_results : 混合检索后的结果列表
    """
    # 向量检索先取多一点，便于后面融合
    vector_results = search_by_vector(query, vector_index, chunk_items, top_k=top_k)

    # 关键词检索也取同样数量
    keyword_results = search_by_keyword(
        query,
        vectorizer,
        tfidf_matrix,
        chunk_items,
        top_k=top_k
    )

    # 用字典做去重，key 选择 chunk_id
    merged_dict = {}

    # 先放入向量检索结果
    for item in vector_results:
        chunk_id = item["chunk_id"]
        merged_dict[chunk_id] = item

    # 再放入关键词检索结果
    # 如果已经有同一个 chunk，就把分数信息补进去
    for item in keyword_results:
        chunk_id = item["chunk_id"]

        if chunk_id in merged_dict:
            merged_dict[chunk_id]["keyword_score"] = item.get("keyword_score", 0.0)
        else:
            merged_dict[chunk_id] = item

    # 转回列表
    merged_results = list(merged_dict.values())

    # 为了方便排序，我们给每条结果一个“混合排序分数”
    # 这里只做一个很简单、初学者友好的排序策略：
    #
    # - keyword_score 越大越好
    # - vector_score 是距离，越小越好
    #
    # 所以我们把 vector_score 变成一个“越大越好”的值：
    # vector_rank_score = 1 / (1 + vector_score)
    #
    # 然后再和 keyword_score 相加
    for item in merged_results:
        vector_score = item.get("vector_score", None)
        keyword_score = item.get("keyword_score", 0.0)

        if vector_score is not None:
            vector_rank_score = 1 / (1 + vector_score)
        else:
            vector_rank_score = 0.0

        item["hybrid_score"] = vector_rank_score + keyword_score

    # 按 hybrid_score 从大到小排序
    merged_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # 最终只返回前 top_k 个
    return merged_results[:top_k]
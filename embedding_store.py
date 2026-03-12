import os
import pickle
import hashlib
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 全局配置
# =========================
DATA_DIR = "data"

# 本地 embedding 模型
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 工具函数
# =========================
def ensure_data_dir():
    """
    确保 data 目录存在
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def get_file_hash(file_bytes):
    """
    根据 PDF 二进制内容生成 hash
    用来区分是不是同一个文件
    """
    return hashlib.md5(file_bytes).hexdigest()


def get_index_paths(file_hash):
    """
    根据 file_hash 生成一组索引文件路径
    """
    ensure_data_dir()

    return {
        "faiss": os.path.join(DATA_DIR, f"{file_hash}_faiss.index"),
        "chunks": os.path.join(DATA_DIR, f"{file_hash}_chunks.pkl"),
        "vectorizer": os.path.join(DATA_DIR, f"{file_hash}_vectorizer.pkl"),
        "tfidf_matrix": os.path.join(DATA_DIR, f"{file_hash}_tfidf.npy"),
        "meta": os.path.join(DATA_DIR, f"{file_hash}_meta.pkl"),
    }


# =========================
# 向量索引构建
# =========================
def build_vector_index(chunk_items):
    """
    把 chunk 文本转成 embedding，并建立 FAISS 索引
    """
    texts = [item["text"] for item in chunk_items]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


# =========================
# 关键词索引构建
# =========================
def build_keyword_index(chunk_items):
    """
    建立 TF-IDF 关键词索引
    """
    texts = [item["text"] for item in chunk_items]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


# =========================
# 索引保存
# =========================
def save_all_indexes(file_hash, vector_index, chunk_items, vectorizer, tfidf_matrix):
    """
    保存：
    1. FAISS 索引
    2. chunk_items
    3. TF-IDF vectorizer
    4. TF-IDF matrix
    """
    paths = get_index_paths(file_hash)

    # 保存 FAISS index
    faiss.write_index(vector_index, paths["faiss"])

    # 保存 chunk_items
    with open(paths["chunks"], "wb") as f:
        pickle.dump(chunk_items, f)

    # 保存 vectorizer
    with open(paths["vectorizer"], "wb") as f:
        pickle.dump(vectorizer, f)

    # 保存 tfidf_matrix
    # 注意：这里把稀疏矩阵转成 dense 再存，简单直观
    np.save(paths["tfidf_matrix"], tfidf_matrix.toarray())

    # 额外保存一个小 meta 文件，方便调试
    meta_info = {
        "chunk_count": len(chunk_items),
        "file_hash": file_hash,
    }
    with open(paths["meta"], "wb") as f:
        pickle.dump(meta_info, f)


# =========================
# 索引加载
# =========================
def load_all_indexes(file_hash):
    """
    加载：
    1. FAISS 索引
    2. chunk_items
    3. TF-IDF vectorizer
    4. TF-IDF matrix
    """
    paths = get_index_paths(file_hash)

    # 只要有一个文件缺失，就说明不能完整加载
    required_files = [
        paths["faiss"],
        paths["chunks"],
        paths["vectorizer"],
        paths["tfidf_matrix"],
    ]

    if not all(os.path.exists(path) for path in required_files):
        return None

    # 读取 FAISS index
    vector_index = faiss.read_index(paths["faiss"])

    # 读取 chunk_items
    with open(paths["chunks"], "rb") as f:
        chunk_items = pickle.load(f)

    # 读取 vectorizer
    with open(paths["vectorizer"], "rb") as f:
        vectorizer = pickle.load(f)

    # 读取 tfidf_matrix
    tfidf_dense = np.load(paths["tfidf_matrix"])
    tfidf_matrix = tfidf_dense

    return vector_index, chunk_items, vectorizer, tfidf_matrix


def indexes_exist(file_hash):
    """
    判断这一份 PDF 的索引是否已经存在
    """
    paths = get_index_paths(file_hash)

    required_files = [
        paths["faiss"],
        paths["chunks"],
        paths["vectorizer"],
        paths["tfidf_matrix"],
    ]

    return all(os.path.exists(path) for path in required_files)


# =========================
# 向量检索
# =========================
def search_by_vector(query, index, chunk_items, top_k=3):
    """
    根据用户问题做向量检索
    """
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):
        if idx < len(chunk_items):
            item = chunk_items[idx].copy()
            item["vector_score"] = float(distances[0][rank])
            results.append(item)

    return results


# =========================
# 关键词检索
# =========================
def search_by_keyword(query, vectorizer, tfidf_matrix, chunk_items, top_k=3):
    """
    根据用户问题做 TF-IDF 关键词检索
    """
    query_vector = vectorizer.transform([query])

    # 如果 tfidf_matrix 是 ndarray，也能算 cosine_similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        if idx < len(chunk_items):
            item = chunk_items[idx].copy()
            item["keyword_score"] = float(similarities[idx])
            results.append(item)

    return results


# =========================
# 混合检索
# =========================
def hybrid_search(query, vector_index, vectorizer, tfidf_matrix, chunk_items, top_k=4):
    """
    同时做：
    1. 向量检索
    2. 关键词检索
    然后合并结果并排序
    """
    vector_results = search_by_vector(query, vector_index, chunk_items, top_k=top_k)

    keyword_results = search_by_keyword(
        query=query,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        chunk_items=chunk_items,
        top_k=top_k
    )

    merged_dict = {}

    for item in vector_results:
        merged_dict[item["chunk_id"]] = item

    for item in keyword_results:
        chunk_id = item["chunk_id"]

        if chunk_id in merged_dict:
            merged_dict[chunk_id]["keyword_score"] = item.get("keyword_score", 0.0)
        else:
            merged_dict[chunk_id] = item

    merged_results = list(merged_dict.values())

    for item in merged_results:
        vector_score = item.get("vector_score", None)
        keyword_score = item.get("keyword_score", 0.0)

        if vector_score is not None:
            vector_rank_score = 1 / (1 + vector_score)
        else:
            vector_rank_score = 0.0

        item["hybrid_score"] = vector_rank_score + keyword_score

    merged_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return merged_results[:top_k]
def save_all_indexes(vector_index, chunk_items, vectorizer, tfidf_matrix):
    """
    保存所有检索相关对象到本地 data/ 目录
    """
    os.makedirs("data", exist_ok=True)

    # 1. 保存 FAISS 索引
    faiss.write_index(vector_index, "data/faiss_index.bin")

    # 2. 保存 chunk_items
    with open("data/chunk_items.pkl", "wb") as f:
        pickle.dump(chunk_items, f)

    # 3. 保存 TF-IDF vectorizer
    with open("data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # 4. 保存 TF-IDF matrix
    with open("data/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)


def load_all_indexes():
    """
    从本地 data/ 目录加载所有检索相关对象
    如果文件不完整，就返回 None
    """
    required_files = [
        "data/faiss_index.bin",
        "data/chunk_items.pkl",
        "data/vectorizer.pkl",
        "data/tfidf_matrix.pkl",
    ]

    if not all(os.path.exists(path) for path in required_files):
        return None

    vector_index = faiss.read_index("data/faiss_index.bin")

    with open("data/chunk_items.pkl", "rb") as f:
        chunk_items = pickle.load(f)

    with open("data/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("data/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return vector_index, chunk_items, vectorizer, tfidf_matrix


def indexes_exist():
    """
    判断本地索引是否已经存在
    """
    required_files = [
        "data/faiss_index.bin",
        "data/chunk_items.pkl",
        "data/vectorizer.pkl",
        "data/tfidf_matrix.pkl",
    ]

    return all(os.path.exists(path) for path in required_files)
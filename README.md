# AI Study Copilot

一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的 PDF 学习助手。  
用户上传 PDF 后，系统会自动解析文档、建立检索索引，并支持文档总结与基于文档内容的问答。

---

## 项目简介

AI Study Copilot 是一个面向学习场景的 AI 文档助手。  
系统会对 PDF 文档进行文本提取、清洗与切块，然后建立检索索引，通过检索增强生成（RAG）技术，让大语言模型基于文档内容回答问题。

系统同时结合：

- **向量检索（Vector Retrieval）**：查找语义相似的内容
- **关键词检索（Keyword Retrieval）**：查找字面匹配的内容

通过 **Hybrid Search（混合检索）** 提高文档问答的准确性。

---

## 系统流程

```
PDF
↓
Text Extraction（文本提取）
↓
Text Cleaning（文本清洗）
↓
Chunking（文本切块）
↓
Embedding（文本向量化）
↓
FAISS Vector Search（向量检索）
+
TF-IDF Keyword Search（关键词检索）
↓
Hybrid Retrieval（混合检索）
↓
LLM Generation（大语言模型生成）
↓
Answer + Citation（回答 + 引用）
```

---

## 当前功能

- PDF 上传
- 文本清洗
- 文本切块（Chunking）
- 本地 Embedding（文本向量化）
- FAISS 向量检索
- TF-IDF 关键词检索
- Hybrid Search（混合检索）
- AI 文档总结
- AI 文档问答
- 对话记忆
- 引用溯源（页码 / chunk / source）
- 运行日志（Observability，可观测性）
- 性能指标展示（检索耗时 / LLM 耗时 / 总耗时）
- Streamlit 网页界面

---

## 项目结构

```
ai-study-copilot/
├── app.py                # Streamlit 主程序，负责界面与整体流程
├── text_chunker.py       # 文本清洗与切块模块
├── embedding_store.py    # 检索相关模块（向量检索 / 关键词检索 / 混合检索）
├── requirements.txt      # 项目依赖
├── README.md             # 项目说明文档
└── data/                 # 本地索引缓存目录（运行后生成）
```

---

## 技术栈

- Python
- Streamlit（Web 应用框架）
- PyMuPDF（PDF 文本提取）
- Sentence Transformers（文本向量化模型）
- FAISS（向量检索）
- scikit-learn（TF-IDF 关键词检索）
- DeepSeek API（大语言模型）

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 运行项目

```bash
streamlit run app.py
```

运行后浏览器打开：

```
http://localhost:8501
```

---

## 使用方法

1. 运行项目
2. 输入 DeepSeek API Key
3. 上传 PDF 文件
4. 查看 PDF 内容预览
5. 点击“让 AI 总结这篇 PDF”生成文档总结
6. 在问答框输入问题进行文档问答
7. 查看引用溯源与运行日志

---

## 项目特点

### Hybrid Search（混合检索）

系统结合：

- 向量检索（语义相似）
- 关键词检索（文本匹配）

两种方式融合后，可以提升 RAG 系统的召回能力。

### Citation（引用溯源）

回答结果会显示来源信息：

- source（文件名）
- page（页码）
- chunk_id（文本块编号）

方便验证答案是否来自文档。

### Observability（可观测性）

系统记录：

- 检索耗时
- LLM 耗时
- 总耗时
- 命中的 chunks

方便调试和分析 RAG 系统表现。

---

## 后续可扩展方向

- 多文档知识库
- 更适合中文的 Embedding 模型
- Reranker（重排序模块）
- Web 部署（Streamlit Cloud / Docker）

---

## 项目定位

该项目是一个 **RAG（Retrieval-Augmented Generation）应用系统示例**，重点展示：

- 文档处理能力
- 检索增强生成流程
- 向量检索与关键词检索结合
- LLM 应用集成
- 基础工程结构
import time
import streamlit as st
import fitz
from openai import OpenAI
from text_chunker import extract_pdf_chunks_with_pages
from embedding_store import (
    build_vector_index,
    build_keyword_index,
    hybrid_search,
    save_all_indexes,
    load_all_indexes,
    indexes_exist
)

# =========================
# 页面基础设置
# =========================
st.set_page_config(page_title="AI Study Copilot", page_icon="📘")

st.title("AI Study Copilot")
st.write("上传 PDF，让 AI 帮你总结内容，并支持基于文档问答。")
st.write("当前版本：RAG + 混合检索 + 对话记忆 + 索引缓存 + 可观测性")

# =========================
# 初始化 session_state
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "chunk_items" not in st.session_state:
    st.session_state.chunk_items = None

if "rag_logs" not in st.session_state:
    st.session_state.rag_logs = []

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "full_text_preview" not in st.session_state:
    st.session_state.full_text_preview = ""

# =========================
# 输入 DeepSeek API Key
# =========================
deepseek_api_key = st.text_input("请输入你的 DeepSeek API Key", type="password")

# =========================
# 上传 PDF
# =========================
uploaded_file = st.file_uploader("上传一个 PDF 文件", type=["pdf"])

if uploaded_file is not None:
    # 只读一次 PDF
    pdf_bytes = uploaded_file.read()
    st.session_state.pdf_bytes = pdf_bytes

    # 如果用户上传了新文件，就重新处理
    if st.session_state.last_file_name != uploaded_file.name:
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.chat_history = []
        st.session_state.rag_logs = []

        # 做全文预览
        preview_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in preview_pdf:
            full_text += page.get_text()
        st.session_state.full_text_preview = full_text

        # 优先尝试加载本地索引
        if indexes_exist():
            loaded = load_all_indexes()

            if loaded is not None:
                vector_index, chunk_items, vectorizer, tfidf_matrix = loaded

                st.session_state.vector_index = vector_index
                st.session_state.vectorizer = vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.chunk_items = chunk_items

                st.success("检测到本地索引，已直接加载")
            else:
                st.warning("本地索引加载失败，将重新构建")

                pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                chunk_items = extract_pdf_chunks_with_pages(
                    pdf,
                    chunk_size=2000,
                    overlap=200,
                    source_name=uploaded_file.name
                )

                vector_index, _ = build_vector_index(chunk_items)
                vectorizer, tfidf_matrix = build_keyword_index(chunk_items)

                save_all_indexes(
                    vector_index=vector_index,
                    chunk_items=chunk_items,
                    vectorizer=vectorizer,
                    tfidf_matrix=tfidf_matrix
                )

                st.session_state.vector_index = vector_index
                st.session_state.vectorizer = vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.chunk_items = chunk_items
        else:
            st.info("未找到本地索引，正在首次构建索引...")

            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            chunk_items = extract_pdf_chunks_with_pages(
                pdf,
                chunk_size=2000,
                overlap=200,
                source_name=uploaded_file.name
            )

            vector_index, _ = build_vector_index(chunk_items)
            vectorizer, tfidf_matrix = build_keyword_index(chunk_items)

            save_all_indexes(
                vector_index=vector_index,
                chunk_items=chunk_items,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix
            )

            st.session_state.vector_index = vector_index
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.chunk_items = chunk_items

    st.success("文件上传成功")
    st.write("文件名：", uploaded_file.name)

    chunk_items = st.session_state.chunk_items
    st.write("可用于检索的 chunk 数量：", len(chunk_items))

    # =========================
    # PDF 内容预览
    # =========================
    st.subheader("PDF 内容预览")
    st.write(st.session_state.full_text_preview[:1000])

    if deepseek_api_key:
        chat_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )

        st.divider()

        # =========================
        # 功能 1：总结 PDF
        # =========================
        st.subheader("文档总结")

        if st.button("让 AI 总结这篇 PDF"):
            with st.spinner("AI 正在阅读并总结，请稍等..."):
                summaries = []

                for i, item in enumerate(chunk_items):
                    response = chat_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {
                                "role": "system",
                                "content": "你是一个擅长总结文档的学习助手。"
                            },
                            {
                                "role": "user",
                                "content": f"请用简单清晰的中文总结这段内容：\n\n{item['text']}"
                            }
                        ]
                    )

                    summary = response.choices[0].message.content
                    summaries.append(summary)

                    st.markdown(f"### 第 {i + 1} 段总结（第 {item['page']} 页）")
                    st.write(summary)

                # 再做一次总总结
                combined_summaries = "\n\n".join(summaries)

                final_response = chat_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个擅长总结文档的学习助手。请把多个分段总结整理成一份结构清晰、语言简洁的整篇文档总结。"
                        },
                        {
                            "role": "user",
                            "content": f"下面是这篇 PDF 的分段总结，请你整合成一份最终总结：\n\n{combined_summaries}"
                        }
                    ]
                )

                final_summary = final_response.choices[0].message.content

                st.subheader("最终总结")
                st.write(final_summary)

        st.divider()

        # =========================
        # 对话历史
        # =========================
        st.subheader("对话历史")

        if not st.session_state.chat_history:
            st.write("暂无历史对话。")
        else:
            for i, item in enumerate(st.session_state.chat_history, start=1):
                st.markdown(f"**第 {i} 轮**")
                st.markdown(f"**你：** {item['question']}")
                st.markdown(f"**AI：** {item['answer']}")

        st.divider()

        # =========================
        # 用户提问
        # =========================
        st.subheader("基于文档提问")
        user_question = st.text_input("请输入你的问题")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("清空对话记忆"):
                st.session_state.chat_history = []
                st.success("对话记忆已清空")

        with col2:
            if st.button("清空运行日志"):
                st.session_state.rag_logs = []
                st.success("运行日志已清空")

        if st.button("开始问答"):

            if not user_question.strip():
                st.warning("请输入问题")
            else:
                total_start_time = time.time()

                with st.spinner("检索文档并生成回答..."):

                    # =========================
                    # 第一步：混合检索
                    # =========================
                    retrieval_start_time = time.time()

                    retrieved_chunks = hybrid_search(
                        query=user_question,
                        vector_index=st.session_state.vector_index,
                        vectorizer=st.session_state.vectorizer,
                        tfidf_matrix=st.session_state.tfidf_matrix,
                        chunk_items=chunk_items,
                        top_k=4
                    )

                    retrieval_end_time = time.time()
                    retrieval_time = retrieval_end_time - retrieval_start_time

                    context = "\n\n".join(
                        [
                            f"[来源: {item['source']} | 第 {item['page']} 页 | chunk_id: {item['chunk_id']}]\n{item['text']}"
                            for item in retrieved_chunks
                        ]
                    )

                    # =========================
                    # 第二步：取最近几轮对话作为记忆
                    # =========================
                    recent_history = st.session_state.chat_history[-3:]

                    history_messages = []
                    for item in recent_history:
                        history_messages.append(
                            {"role": "user", "content": item["question"]}
                        )
                        history_messages.append(
                            {"role": "assistant", "content": item["answer"]}
                        )

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "你是一个基于文档回答问题的 AI 助手。"
                                "优先根据文档回答。"
                                "如果文档没有答案，请明确说明“根据当前文档内容无法确定”。"
                                "不要编造文档中不存在的信息。"
                                "如果当前问题依赖前面对话，请结合最近几轮对话一起理解。"
                            )
                        }
                    ]

                    messages.extend(history_messages)

                    messages.append(
                        {
                            "role": "user",
                            "content": f"文档内容如下：\n\n{context}\n\n问题：{user_question}"
                        }
                    )

                    # =========================
                    # 第三步：调用 LLM
                    # =========================
                    llm_start_time = time.time()

                    response = chat_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages
                    )

                    llm_end_time = time.time()
                    llm_time = llm_end_time - llm_start_time

                    answer = response.choices[0].message.content

                    total_end_time = time.time()
                    total_time = total_end_time - total_start_time

                    # =========================
                    # 保存对话记忆
                    # =========================
                    st.session_state.chat_history.append(
                        {
                            "question": user_question,
                            "answer": answer
                        }
                    )

                    # =========================
                    # 保存运行日志
                    # =========================
                    log_item = {
                        "question": user_question,
                        "answer": answer,
                        "retrieval_time": retrieval_time,
                        "llm_time": llm_time,
                        "total_time": total_time,
                        "retrieved_chunks": [
                            {
                                "chunk_id": item["chunk_id"],
                                "page": item["page"],
                                "source": item["source"],
                                "vector_score": item.get("vector_score"),
                                "keyword_score": item.get("keyword_score"),
                                "hybrid_score": item.get("hybrid_score")
                            }
                            for item in retrieved_chunks
                        ]
                    }

                    st.session_state.rag_logs.append(log_item)

                    # =========================
                    # 展示回答
                    # =========================
                    st.subheader("回答结果")
                    st.write(answer)

                    # =========================
                    # 展示性能指标
                    # =========================
                    st.subheader("本次运行指标")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("检索耗时", f"{retrieval_time:.3f} 秒")

                    with metric_col2:
                        st.metric("LLM耗时", f"{llm_time:.3f} 秒")

                    with metric_col3:
                        st.metric("总耗时", f"{total_time:.3f} 秒")

                    # =========================
                    # 展示引用溯源
                    # =========================
                    st.subheader("引用溯源")

                    for i, item in enumerate(retrieved_chunks, start=1):
                        st.markdown(
                            f"**引用{i}**：{item['source']} | 第 {item['page']} 页 | chunk_id: {item['chunk_id']}"
                        )

                        vector_score = item.get("vector_score", None)
                        keyword_score = item.get("keyword_score", None)
                        hybrid_score = item.get("hybrid_score", None)

                        score_text = "检索分数："
                        if vector_score is not None:
                            score_text += f" 向量距离={vector_score:.4f}"
                        if keyword_score is not None:
                            score_text += f" | 关键词相似度={keyword_score:.4f}"
                        if hybrid_score is not None:
                            score_text += f" | 混合分数={hybrid_score:.4f}"

                        st.caption(score_text)

                        snippet = item["text"][:500]
                        if len(item["text"]) > 500:
                            snippet += "..."

                        st.write(snippet)

        st.divider()

        # =========================
        # 运行日志面板
        # =========================
        st.subheader("运行日志（Observability）")

        if not st.session_state.rag_logs:
            st.write("暂无运行日志。")
        else:
            for idx, log in enumerate(reversed(st.session_state.rag_logs), start=1):
                with st.expander(f"第 {len(st.session_state.rag_logs) - idx + 1} 次问答日志：{log['question']}"):
                    st.markdown(f"**问题：** {log['question']}")
                    st.markdown(f"**回答：** {log['answer']}")
                    st.markdown(f"**检索耗时：** {log['retrieval_time']:.3f} 秒")
                    st.markdown(f"**LLM耗时：** {log['llm_time']:.3f} 秒")
                    st.markdown(f"**总耗时：** {log['total_time']:.3f} 秒")

                    st.markdown("**命中的 chunks：**")
                    for item in log["retrieved_chunks"]:
                        st.markdown(
                            f"- source={item['source']} | page={item['page']} | chunk_id={item['chunk_id']} | "
                            f"vector_score={item['vector_score']} | "
                            f"keyword_score={item['keyword_score']} | "
                            f"hybrid_score={item['hybrid_score']}"
                        )

    else:
        st.warning("请输入 DeepSeek API Key")
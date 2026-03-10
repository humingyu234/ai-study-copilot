import re


def clean_text(text):
    """
    作用：
        清洗原始文本，去掉多余空白、杂乱换行和多余空格。

    参数：
        text : 原始字符串

    返回：
        清洗后的字符串
    """
    # 如果传进来的文本为空，直接返回空字符串
    if not text:
        return ""

    # 统一换行符
    # Windows 里常见 \r\n
    # 老式格式里可能有 \r
    # 这里统一替换成 \n，方便后续处理
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 去掉每一行前后的空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 连续多个换行，压缩成最多两个换行
    # 避免文本中出现太多空行
    text = re.sub(r"\n{2,}", "\n\n", text)

    # 连续多个空格或制表符，压缩成一个空格
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def split_text_into_chunks(text, chunk_size=2000, overlap=200):
    """
    作用：
        把长文本切成多个 chunk。

    参数：
        text       : 需要切块的文本
        chunk_size : 每个 chunk 的最大长度
        overlap    : 相邻 chunk 的重叠长度

    返回：
        chunks : 字符串列表
    """
    # 如果文本为空，直接返回空列表
    if not text:
        return []

    # overlap 不能大于等于 chunk_size
    # 否则会导致切块逻辑出问题
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    # 不断向后切块，直到切完整篇文本
    while start < text_length:
        end = start + chunk_size

        # 取出当前 chunk
        chunk = text[start:end].strip()

        # 如果 chunk 不为空，就加入结果列表
        if chunk:
            chunks.append(chunk)

        # 下一次切块时，保留 overlap 的重叠部分
        start += chunk_size - overlap

    return chunks


def extract_pdf_chunks_with_pages(pdf, chunk_size=2000, overlap=200, source_name="uploaded.pdf"):
    """
    作用：
        逐页读取 PDF，并为每个 chunk 保留元信息。

    每个 chunk 会包含：
        - chunk_id : 当前 chunk 的编号
        - text     : chunk 的文本内容
        - page     : 来自 PDF 的哪一页
        - source   : 文件名

    参数：
        pdf         : 用 fitz.open() 打开的 PDF 对象
        chunk_size  : 每个 chunk 的最大长度
        overlap     : chunk 之间的重叠长度
        source_name : 来源文件名

    返回：
        all_chunks : 列表，里面每个元素都是一个字典
    """
    all_chunks = []
    chunk_id = 0

    # enumerate(pdf) 会拿到两样东西：
    # page_index : 页码下标（从 0 开始）
    # page       : 当前页对象
    for page_index, page in enumerate(pdf):
        # 读取当前页文本
        raw_text = page.get_text()

        # 清洗当前页文本
        cleaned_text = clean_text(raw_text)

        # 如果这一页没有有效文字，直接跳过
        if not cleaned_text:
            continue

        # 把当前页切成多个 chunk
        page_chunks = split_text_into_chunks(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # 给每个 chunk 补充元信息
        for chunk_text in page_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": page_index + 1,   # 页码从 1 开始，更符合阅读习惯
                "source": source_name
            })

            chunk_id += 1

    return all_chunks
from typing import List
from langchain_core.documents import Document


def load_text_to_documents(text: str, source: str = "manual_input") -> List[Document]:
    """
    将原始文本转换成 LangChain 的 Document 列表。
    """
    doc = Document(
        page_content=text,
        metadata={"source": source}
    )
    return [doc]
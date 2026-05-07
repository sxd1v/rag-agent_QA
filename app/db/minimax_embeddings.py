"""
MiniMax Embedding 适配器（适配 MiniMax API 参数格式）。

标准 OpenAI Embedding API 用 input=...，
MiniMax 用 texts=[...] + type="db"。
"""

from typing import List
import requests
from langchain_core.embeddings import Embeddings


class MiniMaxEmbeddings(Embeddings):
    """MiniMax Embedding 适配器"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.chat/v1",
        model: str = "embo-01",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _embed(self, texts: List[str], embed_type: str) -> List[List[float]]:
        """调用 MiniMax embedding API"""
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "texts": texts,
            "type": embed_type,
        }
        resp = self._session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("vectors") is None:
            raise ValueError(f"MiniMax embedding API error: {data.get('base_resp', {}).get('status_msg')}")

        return data["vectors"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表做 embedding（用于建库）"""
        return self._embed(texts, embed_type="db")

    def embed_query(self, text: str) -> List[float]:
        """对单条 query 做 embedding（用于检索）"""
        vectors = self._embed([text], embed_type="query")
        return vectors[0]

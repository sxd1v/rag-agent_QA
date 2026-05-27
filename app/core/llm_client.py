"""LLM 客户端工厂（公共模块，避免循环导入）"""
from contextvars import ContextVar

from langchain_openai import ChatOpenAI
from app.core.config import SILICONFLOW_API_KEY

_llm_call_count: ContextVar[int] = ContextVar("llm_call_count", default=0)


class CountingChatModel:
    """对 invoke 调用计数的轻量代理，用于请求级成本统计。"""

    def __init__(self, client):
        self._client = client

    def invoke(self, *args, **kwargs):
        _llm_call_count.set(_llm_call_count.get() + 1)
        return self._client.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


def reset_llm_call_count():
    _llm_call_count.set(0)


def get_llm_call_count() -> int:
    return _llm_call_count.get()


def get_chat_llm():
    """获取 SiliconFlow Chat LLM 实例（OpenAI 兼容）"""
    return CountingChatModel(ChatOpenAI(
        model="Pro/zai-org/GLM-4.7",
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0.7,
    ))

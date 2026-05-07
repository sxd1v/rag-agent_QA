"""LLM 客户端工厂（公共模块，避免循环导入）"""
from langchain_openai import ChatOpenAI
from app.core.config import SILICONFLOW_API_KEY


def get_chat_llm():
    """获取 SiliconFlow Chat LLM 实例（OpenAI 兼容）"""
    return ChatOpenAI(
        model="Pro/zai-org/GLM-4.7",
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0.7,
    )

"""
结构化日志：记录请求耗时、检索次数、LLM 调用次数、错误信息。
"""
import time
import logging
import json
from functools import wraps

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_agent")


def log_request(question: str):
    """装饰器：自动记录请求耗时和结果"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(json.dumps({
                    "event": "request_complete",
                    "question": question[:100],
                    "elapsed_ms": round(elapsed * 1000),
                    "retrieval_attempts": result.get("retrieval_attempts", 0),
                    "answer_len": len(result.get("answer", "")),
                }, ensure_ascii=False))
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(json.dumps({
                    "event": "request_error",
                    "question": question[:100],
                    "elapsed_ms": round(elapsed * 1000),
                    "error": str(e),
                }, ensure_ascii=False))
                raise
        return wrapper
    return decorator


def log_cache_event(event_type: str, key: str):
    """记录缓存事件"""
    logger.info(json.dumps({
        "event": f"cache_{event_type}",
        "key": key,
    }, ensure_ascii=False))


def log_circuit_breaker(error_count: int, last_error: str):
    """记录熔断事件"""
    logger.warning(json.dumps({
        "event": "circuit_breaker",
        "error_count": error_count,
        "last_error": last_error,
    }, ensure_ascii=False))


def log_retry(action_name: str, attempt: int, error: str):
    """记录重试事件"""
    logger.warning(json.dumps({
        "event": "retry",
        "action": action_name,
        "attempt": attempt,
        "error": str(error)[:200],
    }, ensure_ascii=False))


def log_agent_trace(question: str, result: dict):
    """记录一次 Agent 的动作、召回和最终引用，供回归与排障使用。"""
    actions = []
    for item in result.get("history", []):
        actions.append({
            "step": item.get("step"),
            "action": item.get("action"),
            "query": item.get("query"),
            "retrieved_chunk_ids": item.get("retrieved_chunk_ids", []),
            "new_chunk_ids": item.get("new_chunk_ids", []),
        })
    logger.info(json.dumps({
        "event": "agent_trace",
        "question": question[:100],
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "actions": actions,
        "citations": result.get("citations", []),
        "abstained": result.get("abstained", False),
    }, ensure_ascii=False))

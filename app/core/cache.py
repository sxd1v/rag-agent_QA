"""
缓存层：Redis 优先，不可用时降级到内存字典。

面试要点：展示"优雅降级"设计模式——外部依赖不可用时系统不崩溃，
自动切换到备用方案，而非直接报错。
"""

import json
import pickle
from contextvars import ContextVar

# 尝试连接 Redis，失败不报错
_redis_client = None
try:
    import redis
    _redis_client = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=1)
    _redis_client.ping()
    print("[Cache] Redis 已连接")
except Exception:
    print("[Cache] Redis 不可用，降级到内存缓存")

# 内存缓存兜底
_memory_cache: dict = {}
_cache_stats = ContextVar("cache_stats", default=None)


def reset_stats():
    """重置当前请求/评估任务的缓存统计。"""
    _cache_stats.set({"gets": 0, "hits": 0, "sets": 0})


def get_stats() -> dict:
    """返回当前上下文的缓存统计和命中率。"""
    stats = _cache_stats.get() or {"gets": 0, "hits": 0, "sets": 0}
    gets = stats["gets"]
    return {
        **stats,
        "hit_rate": round(stats["hits"] / gets, 4) if gets else 0.0,
    }


def _record_stat(field: str):
    stats = _cache_stats.get()
    if stats is not None:
        stats[field] += 1


def get(key: str):
    """读取缓存，Redis 优先"""
    _record_stat("gets")
    if _redis_client:
        try:
            data = _redis_client.get(key)
            if data:
                _record_stat("hits")
                return pickle.loads(data)
        except Exception:
            pass  # Redis 异常，降级到内存
    value = _memory_cache.get(key)
    if value is not None:
        _record_stat("hits")
    return value


def set(key: str, value, ttl: int = 3600):
    """写入缓存，Redis 优先。ttl 单位秒（Redis 专用）"""
    _record_stat("sets")
    if _redis_client:
        try:
            _redis_client.setex(key, ttl, pickle.dumps(value))
            return
        except Exception:
            pass
    _memory_cache[key] = value


def delete(key: str):
    """删除缓存"""
    if _redis_client:
        try:
            _redis_client.delete(key)
        except Exception:
            pass
    _memory_cache.pop(key, None)


def clear():
    """清空所有缓存"""
    if _redis_client:
        try:
            _redis_client.flushdb()
        except Exception:
            pass
    _memory_cache.clear()
    print("[Cache] 缓存已清空")


def clear_prefix(prefix: str):
    """仅清理给定命名空间，避免索引刷新影响 session 等无关数据。"""
    if _redis_client:
        try:
            keys = list(_redis_client.scan_iter(match=f"{prefix}*"))
            if keys:
                _redis_client.delete(*keys)
        except Exception:
            pass
    for key in list(_memory_cache):
        if key.startswith(prefix):
            _memory_cache.pop(key, None)

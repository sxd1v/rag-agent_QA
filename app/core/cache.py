"""
缓存层：Redis 优先，不可用时降级到内存字典。

面试要点：展示"优雅降级"设计模式——外部依赖不可用时系统不崩溃，
自动切换到备用方案，而非直接报错。
"""

import json
import pickle

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


def get(key: str):
    """读取缓存，Redis 优先"""
    if _redis_client:
        try:
            data = _redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception:
            pass  # Redis 异常，降级到内存
    return _memory_cache.get(key)


def set(key: str, value, ttl: int = 3600):
    """写入缓存，Redis 优先。ttl 单位秒（Redis 专用）"""
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

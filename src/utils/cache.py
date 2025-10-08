"""
缓存工具模块
提供Redis缓存客户端和相关工具函数
"""

import json
from typing import Any, Optional, Union, List, Dict
import aioredis
from aioredis import Redis

from ..config.settings import get_settings
from .logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# 全局缓存客户端实例
_cache_client: Optional[Redis] = None


async def init_cache_client() -> Redis:
    """
    初始化Redis缓存客户端
    
    Returns:
        Redis: Redis客户端实例
    """
    global _cache_client
    
    if _cache_client is None:
        try:
            # 构建Redis连接URL
            redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
            if settings.redis_password:
                redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
            
            _cache_client = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 测试连接
            await _cache_client.ping()
            logger.info(f"Redis缓存客户端连接成功: {settings.redis_host}:{settings.redis_port}")
            
        except Exception as e:
            logger.error(f"Redis缓存客户端连接失败: {e}")
            _cache_client = None
            raise
    
    return _cache_client


def get_cache_client() -> Optional[Redis]:
    """
    获取缓存客户端实例（同步版本）
    
    Returns:
        Optional[Redis]: Redis客户端实例，如果未初始化返回None
    """
    return _cache_client


async def close_cache_client() -> None:
    """
    关闭缓存客户端连接
    """
    global _cache_client
    
    if _cache_client:
        try:
            await _cache_client.close()
            logger.info("Redis缓存客户端连接已关闭")
        except Exception as e:
            logger.error(f"关闭Redis缓存客户端失败: {e}")
        finally:
            _cache_client = None


class CacheManager:
    """
    缓存管理器类
    提供高级缓存操作接口
    """
    
    def __init__(self, client: Optional[Redis] = None):
        """
        初始化缓存管理器
        
        Args:
            client: Redis客户端实例，如果为None则使用全局客户端
        """
        self.client = client or get_cache_client()
    
    async def get(self, key: str) -> Optional[str]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[str]: 缓存值，如果不存在返回None
        """
        if not self.client:
            return None
        
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"获取缓存失败 {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: str, 
        expire: Optional[int] = None
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒），None表示不过期
            
        Returns:
            bool: 是否设置成功
        """
        if not self.client:
            return False
        
        try:
            if expire:
                await self.client.setex(key, expire, value)
            else:
                await self.client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        if not self.client:
            return False
        
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 缓存是否存在
        """
        if not self.client:
            return False
        
        try:
            result = await self.client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"检查缓存存在性失败 {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """
        设置缓存过期时间
        
        Args:
            key: 缓存键
            seconds: 过期时间（秒）
            
        Returns:
            bool: 是否设置成功
        """
        if not self.client:
            return False
        
        try:
            result = await self.client.expire(key, seconds)
            return result
        except Exception as e:
            logger.error(f"设置缓存过期时间失败 {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        获取缓存剩余过期时间
        
        Args:
            key: 缓存键
            
        Returns:
            int: 剩余过期时间（秒），-1表示永不过期，-2表示不存在
        """
        if not self.client:
            return -2
        
        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error(f"获取缓存TTL失败 {key}: {e}")
            return -2
    
    async def get_json(self, key: str) -> Optional[Any]:
        """
        获取JSON格式的缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 解析后的JSON对象，如果不存在或解析失败返回None
        """
        value = await self.get(key)
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败 {key}: {e}")
            return None
    
    async def set_json(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[int] = None
    ) -> bool:
        """
        设置JSON格式的缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的对象
            expire: 过期时间（秒），None表示不过期
            
        Returns:
            bool: 是否设置成功
        """
        try:
            json_value = json.dumps(value, ensure_ascii=False)
            return await self.set(key, json_value, expire)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON序列化失败 {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        递增缓存值
        
        Args:
            key: 缓存键
            amount: 递增量
            
        Returns:
            Optional[int]: 递增后的值，如果失败返回None
        """
        if not self.client:
            return None
        
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"递增缓存失败 {key}: {e}")
            return None
    
    async def get_keys(self, pattern: str) -> List[str]:
        """
        根据模式获取缓存键列表
        
        Args:
            pattern: 键模式（支持通配符）
            
        Returns:
            List[str]: 匹配的键列表
        """
        if not self.client:
            return []
        
        try:
            return await self.client.keys(pattern)
        except Exception as e:
            logger.error(f"获取缓存键列表失败 {pattern}: {e}")
            return []
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        清除匹配模式的所有缓存
        
        Args:
            pattern: 键模式（支持通配符）
            
        Returns:
            int: 删除的键数量
        """
        if not self.client:
            return 0
        
        try:
            keys = await self.get_keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"清除缓存模式失败 {pattern}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.client:
            return {"error": "缓存客户端未连接"}
        
        try:
            info = await self.client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "redis_version": info.get("redis_version", "unknown")
            }
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {e}")
            return {"error": str(e)}


# 创建全局缓存管理器实例
cache_manager = CacheManager()
"""
监控指标模块
提供Prometheus指标收集和导出功能
"""

import time
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
import psutil

from ..config.settings import get_settings
from .logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# 创建自定义注册表
registry = CollectorRegistry()

# 定义指标
# 请求计数器
request_count = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

# 请求耗时直方图
request_duration = Histogram(
    'rag_request_duration_seconds',
    'RAG request duration in seconds',
    ['endpoint', 'method'],
    registry=registry
)

# 文档处理指标
document_processing_count = Counter(
    'rag_documents_processed_total',
    'Total number of documents processed',
    ['status'],
    registry=registry
)

document_processing_duration = Histogram(
    'rag_document_processing_duration_seconds',
    'Document processing duration in seconds',
    registry=registry
)

# 问答处理指标
qa_processing_count = Counter(
    'rag_qa_requests_total',
    'Total number of QA requests',
    ['status'],
    registry=registry
)

qa_processing_duration = Histogram(
    'rag_qa_processing_duration_seconds',
    'QA processing duration in seconds',
    registry=registry
)

qa_retrieval_documents = Histogram(
    'rag_qa_retrieved_documents',
    'Number of documents retrieved for QA',
    registry=registry
)

# 缓存指标
cache_operations = Counter(
    'rag_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status'],
    registry=registry
)

cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate',
    registry=registry
)

# 向量数据库指标
vector_db_operations = Counter(
    'rag_vector_db_operations_total',
    'Total number of vector database operations',
    ['operation', 'status'],
    registry=registry
)

vector_db_documents = Gauge(
    'rag_vector_db_documents_total',
    'Total number of documents in vector database',
    registry=registry
)

# 系统资源指标
system_cpu_usage = Gauge(
    'rag_system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

system_memory_usage = Gauge(
    'rag_system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=registry
)

system_disk_usage = Gauge(
    'rag_system_disk_usage_bytes',
    'System disk usage in bytes',
    ['path'],
    registry=registry
)

# 应用信息
app_info = Info(
    'rag_app_info',
    'RAG application information',
    registry=registry
)

# 设置应用信息
app_info.info({
    'version': settings.app_version,
    'name': settings.app_name,
    'ollama_model': settings.ollama_model,
    'embedding_model': settings.embedding_model
})


class MetricsCollector:
    """
    指标收集器类
    提供指标收集和更新的便捷方法
    """
    
    def __init__(self):
        """初始化指标收集器"""
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("指标收集器初始化完成")
    
    def record_request(
        self, 
        endpoint: str, 
        method: str, 
        status: str, 
        duration: float
    ) -> None:
        """
        记录请求指标
        
        Args:
            endpoint: 端点名称
            method: HTTP方法
            status: 响应状态
            duration: 请求耗时
        """
        request_count.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
        
        request_duration.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    def record_document_processing(
        self, 
        status: str, 
        duration: float,
        count: int = 1
    ) -> None:
        """
        记录文档处理指标
        
        Args:
            status: 处理状态
            duration: 处理耗时
            count: 处理文档数量
        """
        document_processing_count.labels(status=status).inc(count)
        document_processing_duration.observe(duration)
    
    def record_qa_processing(
        self, 
        status: str, 
        duration: float,
        retrieved_docs: int = 0
    ) -> None:
        """
        记录问答处理指标
        
        Args:
            status: 处理状态
            duration: 处理耗时
            retrieved_docs: 检索到的文档数量
        """
        qa_processing_count.labels(status=status).inc()
        qa_processing_duration.observe(duration)
        
        if retrieved_docs > 0:
            qa_retrieval_documents.observe(retrieved_docs)
    
    def record_cache_operation(
        self, 
        operation: str, 
        status: str,
        is_hit: Optional[bool] = None
    ) -> None:
        """
        记录缓存操作指标
        
        Args:
            operation: 操作类型 (get, set, delete)
            status: 操作状态
            is_hit: 是否命中缓存（仅对get操作有效）
        """
        cache_operations.labels(
            operation=operation,
            status=status
        ).inc()
        
        if operation == 'get' and is_hit is not None:
            if is_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # 更新缓存命中率
            total = self.cache_hits + self.cache_misses
            if total > 0:
                hit_rate = self.cache_hits / total
                cache_hit_rate.set(hit_rate)
    
    def record_vector_db_operation(
        self, 
        operation: str, 
        status: str
    ) -> None:
        """
        记录向量数据库操作指标
        
        Args:
            operation: 操作类型
            status: 操作状态
        """
        vector_db_operations.labels(
            operation=operation,
            status=status
        ).inc()
    
    def update_vector_db_documents(self, count: int) -> None:
        """
        更新向量数据库文档数量
        
        Args:
            count: 文档数量
        """
        vector_db_documents.set(count)
    
    def update_system_metrics(self) -> None:
        """
        更新系统资源指标
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.used)
            
            # 磁盘使用情况
            disk_usage = psutil.disk_usage('/')
            system_disk_usage.labels(path='/').set(disk_usage.used)
            
        except Exception as e:
            logger.error(f"更新系统指标失败: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Returns:
            Dict[str, Any]: 指标摘要
        """
        try:
            return {
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "system_cpu_percent": psutil.cpu_percent(),
                "system_memory_percent": psutil.virtual_memory().percent,
                "system_disk_percent": psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {"error": str(e)}


def metrics_middleware(func):
    """
    指标收集装饰器
    自动收集函数执行的指标
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            
            # 根据函数名确定指标类型
            if 'process_question' in func.__name__:
                metrics_collector.record_qa_processing(status, duration)
            elif 'process_file' in func.__name__ or 'process_document' in func.__name__:
                metrics_collector.record_document_processing(status, duration)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            
            # 根据函数名确定指标类型
            if 'process_question' in func.__name__:
                metrics_collector.record_qa_processing(status, duration)
            elif 'process_file' in func.__name__ or 'process_document' in func.__name__:
                metrics_collector.record_document_processing(status, duration)
    
    # 检查是否为异步函数
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def get_metrics() -> str:
    """
    获取Prometheus格式的指标数据
    
    Returns:
        str: Prometheus格式的指标数据
    """
    try:
        # 更新系统指标
        metrics_collector.update_system_metrics()
        
        # 生成指标数据
        return generate_latest(registry)
    except Exception as e:
        logger.error(f"生成指标数据失败: {e}")
        return ""


def get_content_type() -> str:
    """
    获取指标数据的Content-Type
    
    Returns:
        str: Content-Type
    """
    return CONTENT_TYPE_LATEST


# 创建全局指标收集器实例
metrics_collector = MetricsCollector()
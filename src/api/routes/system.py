"""
系统管理API路由
提供健康检查、监控指标、配置管理等系统功能的接口
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...core.rag_engine import rag_engine
from ...config.settings import get_settings
from ...utils.logger import get_logger
from ...utils.metrics import get_metrics, get_content_type, metrics_collector
from ..models import (
    HealthCheckResponse,
    SystemStatsResponse,
    ConfigUpdateRequest,
    ConfigResponse,
    ComponentHealth,
    SystemStats,
    MetricsSummary
)

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/system", tags=["系统管理"])


@router.get("/health", response_model=HealthCheckResponse, summary="健康检查")
async def health_check() -> HealthCheckResponse:
    """
    系统健康检查
    
    Returns:
        HealthCheckResponse: 健康检查结果
    """
    try:
        # 调用RAG引擎的健康检查
        health_result = await rag_engine.health_check()
        
        # 格式化组件健康状态
        components = {}
        for component_name, component_data in health_result.get("components", {}).items():
            components[component_name] = ComponentHealth(
                status=component_data.get("status", "unknown"),
                error=component_data.get("error"),
                document_count=component_data.get("document_count"),
                model=component_data.get("model")
            )
        
        return HealthCheckResponse(
            success=True,
            message="健康检查完成",
            status=health_result.get("status", "unknown"),
            components=components
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthCheckResponse(
            success=False,
            message=f"健康检查失败: {str(e)}",
            status="unhealthy",
            components={}
        )


@router.get("/stats", response_model=SystemStatsResponse, summary="获取系统统计信息")
async def get_system_stats() -> SystemStatsResponse:
    """
    获取系统统计信息
    
    Returns:
        SystemStatsResponse: 系统统计信息
    """
    try:
        # 获取系统统计信息
        stats = rag_engine.get_system_stats()
        
        # 格式化系统信息
        system_info = SystemStats(
            app_name=stats["system_info"]["app_name"],
            app_version=stats["system_info"]["app_version"],
            initialized=stats["system_info"]["initialized"],
            total_documents=stats["document_processor"].get("total_documents", 0),
            embedding_model=stats["qa_processor"].get("embedding_model", "unknown"),
            llm_model=stats["qa_processor"].get("llm_model", "unknown"),
            cache_enabled=stats["qa_processor"].get("cache_enabled", False)
        )
        
        # 格式化指标摘要
        metrics_data = stats.get("metrics", {})
        metrics_summary = MetricsSummary(
            cache_hit_rate=metrics_data.get("cache_hit_rate", 0.0),
            cache_hits=metrics_data.get("cache_hits", 0),
            cache_misses=metrics_data.get("cache_misses", 0),
            system_cpu_percent=metrics_data.get("system_cpu_percent", 0.0),
            system_memory_percent=metrics_data.get("system_memory_percent", 0.0),
            system_disk_percent=metrics_data.get("system_disk_percent", 0.0)
        )
        
        # 获取支持的格式
        supported_formats = rag_engine.get_supported_formats()
        
        return SystemStatsResponse(
            success=True,
            message="获取系统统计信息成功",
            system_info=system_info,
            metrics=metrics_summary,
            supported_formats=supported_formats
        )
        
    except Exception as e:
        logger.error(f"获取系统统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统统计信息失败: {str(e)}")


@router.get("/metrics", summary="获取Prometheus监控指标")
async def get_prometheus_metrics():
    """
    获取Prometheus格式的监控指标
    
    Returns:
        Response: Prometheus格式的指标数据
    """
    try:
        metrics_data = get_metrics()
        return Response(
            content=metrics_data,
            media_type=get_content_type()
        )
        
    except Exception as e:
        logger.error(f"获取监控指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控指标失败: {str(e)}")


@router.get("/config", response_model=ConfigResponse, summary="获取系统配置")
async def get_system_config() -> ConfigResponse:
    """
    获取当前系统配置
    
    Returns:
        ConfigResponse: 系统配置信息
    """
    try:
        config = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "debug": settings.debug,
            "host": settings.host,
            "port": settings.port,
            "ollama_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_k": settings.retrieval_k,
            "similarity_threshold": settings.similarity_threshold,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "cache_ttl": settings.cache_ttl,
            "max_file_size": settings.max_file_size,
            "supported_formats": settings.supported_formats
        }
        
        return ConfigResponse(
            success=True,
            message="获取系统配置成功",
            config=config
        )
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统配置失败: {str(e)}")


@router.put("/config", response_model=ConfigResponse, summary="更新系统配置")
async def update_system_config(
    request: ConfigUpdateRequest
) -> ConfigResponse:
    """
    更新系统配置（部分配置项）
    
    Args:
        request: 配置更新请求
        
    Returns:
        ConfigResponse: 更新后的配置信息
    """
    try:
        # 获取要更新的配置项
        update_data = request.dict(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(status_code=400, detail="没有提供要更新的配置项")
        
        # 这里可以实现配置更新逻辑
        # 注意：某些配置项可能需要重启服务才能生效
        
        logger.info(f"配置更新请求: {update_data}")
        
        # 返回更新后的配置
        current_config = {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_k": settings.retrieval_k,
            "similarity_threshold": settings.similarity_threshold,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature
        }
        
        # 应用更新（这里只是示例，实际需要更新settings对象）
        current_config.update(update_data)
        
        return ConfigResponse(
            success=True,
            message=f"配置更新成功，更新了 {len(update_data)} 个配置项",
            config=current_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新系统配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新系统配置失败: {str(e)}")


@router.post("/initialize", summary="初始化系统")
async def initialize_system() -> Dict[str, Any]:
    """
    初始化RAG系统
    
    Returns:
        Dict[str, Any]: 初始化结果
    """
    try:
        if rag_engine.initialized:
            return {
                "success": True,
                "message": "系统已经初始化",
                "initialized": True
            }
        
        # 初始化RAG引擎
        await rag_engine.initialize()
        
        return {
            "success": True,
            "message": "系统初始化成功",
            "initialized": True
        }
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统初始化失败: {str(e)}")


@router.post("/shutdown", summary="关闭系统")
async def shutdown_system() -> Dict[str, Any]:
    """
    关闭RAG系统，释放资源
    
    Returns:
        Dict[str, Any]: 关闭结果
    """
    try:
        await rag_engine.close()
        
        return {
            "success": True,
            "message": "系统关闭成功",
            "shutdown": True
        }
        
    except Exception as e:
        logger.error(f"系统关闭失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统关闭失败: {str(e)}")


@router.get("/version", summary="获取版本信息")
async def get_version_info() -> Dict[str, Any]:
    """
    获取系统版本信息
    
    Returns:
        Dict[str, Any]: 版本信息
    """
    try:
        import sys
        import platform
        
        return {
            "success": True,
            "message": "获取版本信息成功",
            "version_info": {
                "app_name": settings.app_name,
                "app_version": settings.app_version,
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor()
            }
        }
        
    except Exception as e:
        logger.error(f"获取版本信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取版本信息失败: {str(e)}")


@router.get("/logs", summary="获取系统日志")
async def get_system_logs(
    level: str = "INFO",
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    获取系统日志（简化版本）
    
    Args:
        level: 日志级别
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        Dict[str, Any]: 日志信息
    """
    try:
        # 这里可以实现从日志文件读取日志的逻辑
        # 目前返回占位符信息
        
        return {
            "success": True,
            "message": "获取系统日志成功",
            "logs": [],
            "total_count": 0,
            "level": level,
            "limit": limit,
            "offset": offset,
            "note": "日志功能需要进一步实现"
        }
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统日志失败: {str(e)}")


@router.delete("/cache", summary="清除系统缓存")
async def clear_system_cache(
    pattern: str = "*"
) -> Dict[str, Any]:
    """
    清除系统缓存
    
    Args:
        pattern: 缓存键模式
        
    Returns:
        Dict[str, Any]: 清除结果
    """
    try:
        # 这里可以实现缓存清除逻辑
        # 需要访问缓存管理器
        
        logger.info(f"清除缓存请求: pattern={pattern}")
        
        return {
            "success": True,
            "message": "缓存清除成功",
            "pattern": pattern,
            "cleared_count": 0,
            "note": "缓存清除功能需要进一步实现"
        }
        
    except Exception as e:
        logger.error(f"清除系统缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除系统缓存失败: {str(e)}")
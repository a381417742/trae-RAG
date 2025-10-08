"""
RAG知识库问答系统 - FastAPI应用主文件
基于LangChain + FastAPI + Chroma架构的智能问答系统
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn

from .config.settings import get_settings
from .utils.logger import get_logger
from .core.rag_engine import rag_engine
from .api.middleware import setup_middleware
from .api.routes import documents, qa, system

# 获取配置和日志
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    处理启动和关闭事件
    """
    # 启动事件
    logger.info("🚀 RAG系统启动中...")
    
    try:
        # 初始化RAG引擎
        await rag_engine.initialize()
        logger.info("✅ RAG引擎初始化完成")
        
        # 应用启动完成
        logger.info(f"🎉 RAG系统启动完成 - {settings.app_name} v{settings.app_version}")
        logger.info(f"📡 服务地址: http://{settings.host}:{settings.port}")
        logger.info(f"📚 API文档: http://{settings.host}:{settings.port}/docs")
        
    except Exception as e:
        logger.error(f"❌ RAG系统启动失败: {e}")
        raise
    
    yield
    
    # 关闭事件
    logger.info("🛑 RAG系统关闭中...")
    
    try:
        # 关闭RAG引擎
        await rag_engine.close()
        logger.info("✅ RAG引擎已关闭")
        
    except Exception as e:
        logger.error(f"❌ RAG系统关闭失败: {e}")
    
    logger.info("👋 RAG系统已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    description="""
    ## RAG知识库问答系统

    基于LangChain + FastAPI + Chroma架构的智能问答系统，支持：

    ### 🎯 核心功能
    - **文档处理**: 支持PDF、TXT、MD、DOCX等格式文档上传和处理
    - **智能问答**: 基于向量检索和大语言模型的智能问答
    - **批量处理**: 支持批量文档处理和批量问答
    - **缓存加速**: Redis缓存提升问答响应速度

    ### 🛠️ 技术特性
    - **异步处理**: 全面支持异步操作，提升并发性能
    - **监控指标**: Prometheus指标收集，支持性能监控
    - **健康检查**: 完整的系统健康检查机制
    - **安全认证**: 支持API密钥认证

    ### 📊 系统架构
    - **语言模型**: Qwen3:30B (Ollama)
    - **嵌入模型**: BGE-Large-ZH-v1.5
    - **向量数据库**: Chroma
    - **缓存系统**: Redis
    - **监控系统**: Prometheus + Grafana

    ### 🚀 快速开始
    1. 上传文档: `POST /documents/upload`
    2. 处理文档: `POST /documents/process`
    3. 开始问答: `POST /qa/ask`
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 设置中间件
setup_middleware(app)

# 注册路由
app.include_router(documents.router, prefix="/api")
app.include_router(qa.router, prefix="/api")
app.include_router(system.router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root():
    """
    根路径重定向到API文档
    """
    return RedirectResponse(url="/docs")


@app.get("/api", tags=["基础接口"])
async def api_info():
    """
    API基础信息
    """
    return {
        "success": True,
        "message": "RAG知识库问答系统API",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "documents": "/api/documents",
            "qa": "/api/qa",
            "system": "/api/system"
        }
    }


@app.get("/api/health", tags=["基础接口"])
async def simple_health_check():
    """
    简单健康检查（快速响应）
    """
    return {
        "success": True,
        "status": "healthy",
        "message": "服务正常运行",
        "app_name": settings.app_name,
        "version": settings.app_version
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    404错误处理
    """
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "请求的资源不存在",
            "error": "Not Found",
            "path": str(request.url.path),
            "available_endpoints": [
                "/api/documents",
                "/api/qa", 
                "/api/system",
                "/docs"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    500错误处理
    """
    logger.error(f"内部服务器错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "内部服务器错误",
            "error": "Internal Server Error"
        }
    )


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    
    Returns:
        FastAPI: 应用实例
    """
    return app


def main():
    """
    应用入口点
    """
    try:
        logger.info(f"启动RAG系统服务器...")
        logger.info(f"配置信息:")
        logger.info(f"  - 应用名称: {settings.app_name}")
        logger.info(f"  - 版本: {settings.app_version}")
        logger.info(f"  - 监听地址: {settings.host}:{settings.port}")
        logger.info(f"  - 调试模式: {settings.debug}")
        logger.info(f"  - 工作进程: {settings.workers}")
        
        # 启动服务器
        uvicorn.run(
            "src.main:app",
            host=settings.host,
            port=settings.port,
            workers=settings.workers,
            log_level=settings.log_level.lower(),
            reload=settings.debug,
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        raise


if __name__ == "__main__":
    main()
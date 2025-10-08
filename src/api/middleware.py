"""
API中间件
处理CORS、请求日志、指标收集、错误处理等
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json

from ..config.settings import get_settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics_collector

logger = get_logger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录所有API请求的详细信息
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求并记录日志
        
        Args:
            request: HTTP请求
            call_next: 下一个处理函数
            
        Returns:
            Response: HTTP响应
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取客户端信息
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # 记录请求信息
        logger.info(
            f"请求开始 - ID: {request_id}, "
            f"方法: {request.method}, "
            f"路径: {request.url.path}, "
            f"客户端: {client_ip}, "
            f"User-Agent: {user_agent}"
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            logger.info(
                f"请求完成 - ID: {request_id}, "
                f"状态码: {response.status_code}, "
                f"处理时间: {process_time:.3f}s"
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录错误信息
            logger.error(
                f"请求失败 - ID: {request_id}, "
                f"错误: {str(e)}, "
                f"处理时间: {process_time:.3f}s"
            )
            
            # 返回错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "内部服务器错误",
                    "request_id": request_id,
                    "error": str(e)
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": f"{process_time:.3f}"
                }
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    指标收集中间件
    自动收集API请求的指标数据
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求并收集指标
        
        Args:
            request: HTTP请求
            call_next: 下一个处理函数
            
        Returns:
            Response: HTTP响应
        """
        start_time = time.time()
        
        # 获取端点信息
        endpoint = request.url.path
        method = request.method
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            duration = time.time() - start_time
            
            # 确定状态
            if response.status_code < 400:
                status = "success"
            elif response.status_code < 500:
                status = "client_error"
            else:
                status = "server_error"
            
            # 记录指标
            metrics_collector.record_request(endpoint, method, status, duration)
            
            return response
            
        except Exception as e:
            # 计算处理时间
            duration = time.time() - start_time
            
            # 记录错误指标
            metrics_collector.record_request(endpoint, method, "error", duration)
            
            raise


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    安全中间件
    处理API密钥验证、请求限制等安全功能
    """
    
    def __init__(self, app, api_key: str = None):
        """
        初始化安全中间件
        
        Args:
            app: FastAPI应用实例
            api_key: API密钥
        """
        super().__init__(app)
        self.api_key = api_key
        self.public_paths = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/system/health",
            "/system/version"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求安全验证
        
        Args:
            request: HTTP请求
            call_next: 下一个处理函数
            
        Returns:
            Response: HTTP响应
        """
        # 检查是否为公开路径
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # 如果设置了API密钥，进行验证
        if self.api_key:
            auth_header = request.headers.get("Authorization")
            api_key_header = request.headers.get("X-API-Key")
            
            provided_key = None
            if auth_header and auth_header.startswith("Bearer "):
                provided_key = auth_header[7:]
            elif api_key_header:
                provided_key = api_key_header
            
            if not provided_key or provided_key != self.api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "message": "未授权访问",
                        "error": "无效的API密钥"
                    }
                )
        
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    错误处理中间件
    统一处理未捕获的异常
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求并捕获异常
        
        Args:
            request: HTTP请求
            call_next: 下一个处理函数
            
        Returns:
            Response: HTTP响应
        """
        try:
            return await call_next(request)
            
        except HTTPException:
            # FastAPI的HTTP异常直接抛出
            raise
            
        except Exception as e:
            # 记录未捕获的异常
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.error(f"未捕获的异常 - 请求ID: {request_id}, 错误: {str(e)}", exc_info=True)
            
            # 返回统一的错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "内部服务器错误",
                    "error": "服务器处理请求时发生错误",
                    "request_id": request_id
                }
            )


def setup_cors_middleware(app):
    """
    设置CORS中间件
    
    Args:
        app: FastAPI应用实例
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"]
    )


def setup_middleware(app):
    """
    设置所有中间件
    
    Args:
        app: FastAPI应用实例
    """
    # 错误处理中间件（最外层）
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 安全中间件
    app.add_middleware(SecurityMiddleware, api_key=settings.api_key)
    
    # 指标收集中间件
    app.add_middleware(MetricsMiddleware)
    
    # 请求日志中间件
    app.add_middleware(RequestLoggingMiddleware)
    
    # CORS中间件（最内层）
    setup_cors_middleware(app)
    
    logger.info("所有中间件设置完成")
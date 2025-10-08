"""
日志工具模块
提供统一的日志配置和管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger

from ..config.settings import get_settings

settings = get_settings()


class InterceptHandler(logging.Handler):
    """
    拦截标准库日志并转发到loguru
    """
    
    def emit(self, record):
        # 获取对应的loguru级别
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 查找调用者
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """
    设置日志配置
    """
    # 移除默认的loguru处理器
    loguru_logger.remove()
    
    # 添加控制台输出
    loguru_logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # 添加文件输出
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 应用日志
    loguru_logger.add(
        log_dir / "app.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    # 错误日志
    loguru_logger.add(
        log_dir / "error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # 拦截标准库日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 设置第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> "loguru.Logger":
    """
    获取logger实例
    
    Args:
        name: logger名称，默认为调用模块名
        
    Returns:
        loguru.Logger: logger实例
    """
    if name is None:
        # 获取调用者的模块名
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return loguru_logger.bind(name=name)


# 初始化日志配置
setup_logging()
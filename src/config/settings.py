"""
系统配置管理
定义所有系统配置项和环境变量
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """
    系统配置类
    使用pydantic进行配置验证和类型检查
    """
    
    # 应用基础配置
    app_name: str = Field(default="RAG知识库问答系统", description="应用名称")
    app_version: str = Field(default="2.0.0", description="应用版本")
    debug: bool = Field(default=False, description="调试模式")
    
    # 服务配置
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8000, description="服务端口")
    workers: int = Field(default=1, description="工作进程数")
    
    # Ollama模型配置
    ollama_base_url: str = Field(default="http://ollama:11434", description="Ollama服务地址")
    ollama_model: str = Field(default="qwen2.5:7b-instruct", description="使用的语言模型")
    ollama_timeout: int = Field(default=300, description="模型请求超时时间(秒)")
    
    # 向量数据库配置
    chroma_host: str = Field(default="chroma", description="Chroma服务地址")
    chroma_port: int = Field(default=8000, description="Chroma服务端口")
    chroma_collection: str = Field(default="rag_documents", description="向量集合名称")
    
    # 嵌入模型配置
    embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5", description="嵌入模型名称")
    embedding_device: str = Field(default="cuda", description="嵌入模型运行设备")
    
    # Redis缓存配置
    redis_host: str = Field(default="redis", description="Redis服务地址")
    redis_port: int = Field(default=6379, description="Redis服务端口")
    redis_db: int = Field(default=0, description="Redis数据库编号")
    redis_password: Optional[str] = Field(default=None, description="Redis密码")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    # 文档处理配置
    max_file_size: int = Field(default=50 * 1024 * 1024, description="最大文件大小(字节)")
    supported_formats: List[str] = Field(
        default=["pdf", "txt", "md", "docx"], 
        description="支持的文档格式"
    )
    chunk_size: int = Field(default=1000, description="文档分块大小")
    chunk_overlap: int = Field(default=200, description="分块重叠大小")
    
    # RAG检索配置
    retrieval_k: int = Field(default=5, description="检索返回的文档数量")
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")
    
    # 生成配置
    max_tokens: int = Field(default=2000, description="生成的最大token数")
    temperature: float = Field(default=0.7, description="生成温度")
    
    # 监控配置
    enable_metrics: bool = Field(default=True, description="启用指标监控")
    metrics_port: int = Field(default=8001, description="指标服务端口")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    
    # 安全配置
    api_key: Optional[str] = Field(default=None, description="API访问密钥")
    cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    
    class Config:
        """pydantic配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """
    获取配置实例
    
    Returns:
        Settings: 配置实例
    """
    return settings


def update_settings(**kwargs) -> None:
    """
    更新配置项
    
    Args:
        **kwargs: 要更新的配置项
    """
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


def validate_settings() -> bool:
    """
    验证配置的有效性
    
    Returns:
        bool: 配置是否有效
    """
    try:
        # 验证必要的配置项
        required_configs = [
            'ollama_base_url',
            'chroma_host',
            'redis_host'
        ]
        
        for config in required_configs:
            if not getattr(settings, config):
                raise ValueError(f"必要配置项 {config} 未设置")
        
        # 验证数值范围
        if settings.port < 1 or settings.port > 65535:
            raise ValueError("端口号必须在1-65535范围内")
            
        if settings.chunk_size < 100:
            raise ValueError("文档分块大小不能小于100")
            
        if settings.similarity_threshold < 0 or settings.similarity_threshold > 1:
            raise ValueError("相似度阈值必须在0-1范围内")
        
        return True
        
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False


if __name__ == "__main__":
    # 配置验证测试
    print("配置验证结果:", validate_settings())
    print("当前配置:")
    print(f"  应用名称: {settings.app_name}")
    print(f"  服务地址: {settings.host}:{settings.port}")
    print(f"  Ollama地址: {settings.ollama_base_url}")
    print(f"  模型名称: {settings.ollama_model}")
    print(f"  向量数据库: {settings.chroma_host}:{settings.chroma_port}")
    print(f"  缓存服务: {settings.redis_host}:{settings.redis_port}")
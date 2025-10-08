"""
配置模块单元测试
测试系统配置加载和验证功能
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.config.settings import get_settings, Settings


class TestSettings:
    """配置设置测试类"""
    
    def test_default_settings(self):
        """测试默认配置加载"""
        settings = get_settings()
        
        assert settings.app_name == "RAG知识库问答系统"
        assert settings.app_version == "2.0.0"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.retrieval_k == 5
        assert settings.similarity_threshold == 0.7
        assert settings.max_tokens == 2000
        assert settings.temperature == 0.7
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖配置"""
        with patch.dict(os.environ, {
            'APP_NAME': 'Test RAG System',
            'PORT': '9000',
            'CHUNK_SIZE': '500',
            'TEMPERATURE': '0.5'
        }):
            settings = Settings()
            
            assert settings.app_name == 'Test RAG System'
            assert settings.port == 9000
            assert settings.chunk_size == 500
            assert settings.temperature == 0.5
    
    def test_invalid_configuration(self):
        """测试无效配置验证"""
        with pytest.raises(ValidationError):
            Settings(
                port=-1,  # 无效端口
                chunk_size=0,  # 无效块大小
                temperature=3.0  # 无效温度
            )
    
    def test_ollama_configuration(self):
        """测试Ollama配置"""
        settings = get_settings()
        
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "qwen3:30b"
        assert settings.ollama_timeout == 300
    
    def test_chroma_configuration(self):
        """测试Chroma配置"""
        settings = get_settings()
        
        assert settings.chroma_host == "localhost"
        assert settings.chroma_port == 8002
        assert settings.chroma_collection == "rag_documents"
    
    def test_redis_configuration(self):
        """测试Redis配置"""
        settings = get_settings()
        
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.redis_db == 0
        assert settings.cache_ttl == 3600
    
    def test_document_processing_configuration(self):
        """测试文档处理配置"""
        settings = get_settings()
        
        assert settings.max_file_size == 52428800  # 50MB
        assert "pdf" in settings.supported_formats
        assert "txt" in settings.supported_formats
        assert "md" in settings.supported_formats
        assert "docx" in settings.supported_formats
    
    def test_embedding_configuration(self):
        """测试嵌入模型配置"""
        settings = get_settings()
        
        assert settings.embedding_model == "BAAI/bge-large-zh-v1.5"
        assert settings.embedding_device == "cuda"
    
    def test_logging_configuration(self):
        """测试日志配置"""
        settings = get_settings()
        
        assert settings.log_level == "INFO"
        assert "%(asctime)s" in settings.log_format
        assert "%(name)s" in settings.log_format
        assert "%(levelname)s" in settings.log_format
        assert "%(message)s" in settings.log_format
    
    def test_cors_configuration(self):
        """测试CORS配置"""
        settings = get_settings()
        
        assert isinstance(settings.cors_origins, list)
        assert "*" in settings.cors_origins
    
    @patch.dict(os.environ, {'DEBUG': 'true'})
    def test_debug_mode(self):
        """测试调试模式"""
        settings = Settings()
        assert settings.debug is True
    
    @patch.dict(os.environ, {'DEBUG': 'false'})
    def test_production_mode(self):
        """测试生产模式"""
        settings = Settings()
        assert settings.debug is False
    
    def test_settings_singleton(self):
        """测试配置单例模式"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # 应该是同一个实例
        assert settings1 is settings2
    
    def test_metrics_configuration(self):
        """测试监控配置"""
        settings = get_settings()
        
        assert settings.enable_metrics is True
        assert settings.metrics_port == 8001
    
    def test_security_configuration(self):
        """测试安全配置"""
        settings = get_settings()
        
        # API密钥可能为空（测试环境）
        assert isinstance(settings.api_key, (str, type(None)))
        assert isinstance(settings.cors_origins, list)


class TestConfigurationValidation:
    """配置验证测试类"""
    
    def test_port_validation(self):
        """测试端口验证"""
        # 有效端口
        settings = Settings(port=8080)
        assert settings.port == 8080
        
        # 无效端口
        with pytest.raises(ValidationError):
            Settings(port=0)
        
        with pytest.raises(ValidationError):
            Settings(port=65536)
    
    def test_chunk_size_validation(self):
        """测试分块大小验证"""
        # 有效大小
        settings = Settings(chunk_size=1500)
        assert settings.chunk_size == 1500
        
        # 无效大小
        with pytest.raises(ValidationError):
            Settings(chunk_size=50)  # 太小
        
        with pytest.raises(ValidationError):
            Settings(chunk_size=10000)  # 太大
    
    def test_temperature_validation(self):
        """测试温度参数验证"""
        # 有效温度
        settings = Settings(temperature=0.8)
        assert settings.temperature == 0.8
        
        # 无效温度
        with pytest.raises(ValidationError):
            Settings(temperature=-0.1)  # 太小
        
        with pytest.raises(ValidationError):
            Settings(temperature=2.1)  # 太大
    
    def test_similarity_threshold_validation(self):
        """测试相似度阈值验证"""
        # 有效阈值
        settings = Settings(similarity_threshold=0.8)
        assert settings.similarity_threshold == 0.8
        
        # 无效阈值
        with pytest.raises(ValidationError):
            Settings(similarity_threshold=-0.1)
        
        with pytest.raises(ValidationError):
            Settings(similarity_threshold=1.1)
    
    def test_retrieval_k_validation(self):
        """测试检索数量验证"""
        # 有效数量
        settings = Settings(retrieval_k=10)
        assert settings.retrieval_k == 10
        
        # 无效数量
        with pytest.raises(ValidationError):
            Settings(retrieval_k=0)
        
        with pytest.raises(ValidationError):
            Settings(retrieval_k=101)


@pytest.fixture
def mock_env_vars():
    """模拟环境变量的fixture"""
    return {
        'APP_NAME': 'Test RAG System',
        'DEBUG': 'true',
        'PORT': '9000',
        'OLLAMA_MODEL': 'test-model',
        'CHUNK_SIZE': '800',
        'TEMPERATURE': '0.5'
    }


def test_configuration_with_mock_env(mock_env_vars):
    """使用模拟环境变量测试配置"""
    with patch.dict(os.environ, mock_env_vars):
        settings = Settings()
        
        assert settings.app_name == 'Test RAG System'
        assert settings.debug is True
        assert settings.port == 9000
        assert settings.ollama_model == 'test-model'
        assert settings.chunk_size == 800
        assert settings.temperature == 0.5
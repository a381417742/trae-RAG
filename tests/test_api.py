"""
API接口单元测试
测试FastAPI应用的各个端点功能
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from src.main import app
from src.core.rag_engine import rag_engine


class TestAPIEndpoints:
    """API端点测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_engine(self):
        """模拟RAG引擎"""
        with patch.object(rag_engine, 'initialized', True), \
             patch.object(rag_engine, 'process_document'), \
             patch.object(rag_engine, 'answer_question'), \
             patch.object(rag_engine, 'get_system_stats'), \
             patch.object(rag_engine, 'health_check'):
            yield rag_engine
    
    def test_root_redirect(self, client):
        """测试根路径重定向"""
        response = client.get("/", allow_redirects=False)
        assert response.status_code == 302
        assert "/docs" in response.headers["location"]
    
    def test_api_info(self, client):
        """测试API基础信息"""
        response = client.get("/api")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "RAG知识库问答系统" in data["message"]
        assert "endpoints" in data
    
    def test_simple_health_check(self, client):
        """测试简单健康检查"""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"


class TestDocumentAPI:
    """文档管理API测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_engine(self):
        with patch.object(rag_engine, 'initialized', True):
            yield rag_engine
    
    def test_upload_document_success(self, client, mock_rag_engine):
        """测试成功上传文档"""
        # 创建测试文件
        test_content = "这是一个测试文档的内容。"
        test_file = io.BytesIO(test_content.encode())
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test.txt"
        assert "file_path" in data
    
    def test_upload_unsupported_format(self, client, mock_rag_engine):
        """测试上传不支持的文件格式"""
        test_content = "测试内容"
        test_file = io.BytesIO(test_content.encode())
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.xyz", test_file, "application/octet-stream")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "不支持的文件格式" in data["detail"]
    
    def test_upload_large_file(self, client, mock_rag_engine):
        """测试上传超大文件"""
        # 创建超过限制的文件内容
        large_content = "x" * (50 * 1024 * 1024 + 1)  # 超过50MB
        test_file = io.BytesIO(large_content.encode())
        
        response = client.post(
            "/api/documents/upload",
            files={"file": ("large.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "文件大小超过限制" in data["detail"]
    
    def test_process_document_success(self, client, mock_rag_engine):
        """测试成功处理文档"""
        # 模拟处理结果
        mock_rag_engine.process_document = AsyncMock(return_value={
            "success": True,
            "message": "处理完成",
            "file_path": "/test/path.txt",
            "chunks_created": 5,
            "stored_count": 5
        })
        
        response = client.post(
            "/api/documents/process",
            json={"file_path": "/test/path.txt"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] == 5
    
    def test_process_document_not_found(self, client, mock_rag_engine):
        """测试处理不存在的文档"""
        response = client.post(
            "/api/documents/process",
            json={"file_path": "/nonexistent/file.txt"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "文件不存在" in data["detail"]
    
    def test_process_document_invalid_request(self, client, mock_rag_engine):
        """测试无效的处理请求"""
        response = client.post(
            "/api/documents/process",
            json={}  # 缺少必要参数
        )
        
        assert response.status_code == 422  # 验证错误
    
    def test_delete_document_success(self, client, mock_rag_engine):
        """测试成功删除文档"""
        mock_rag_engine.delete_document = AsyncMock(return_value={
            "success": True,
            "message": "删除完成",
            "deleted_count": 3
        })
        
        response = client.delete(
            "/api/documents/delete",
            json={"file_path": "/test/path.txt"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 3
    
    def test_list_documents(self, client, mock_rag_engine):
        """测试获取文档列表"""
        response = client.get("/api/documents/list")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "files" in data
        assert "total_count" in data
    
    def test_get_supported_formats(self, client, mock_rag_engine):
        """测试获取支持的文档格式"""
        mock_rag_engine.get_supported_formats = Mock(return_value=[".pdf", ".txt", ".md", ".docx"])
        
        response = client.get("/api/documents/formats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "supported_formats" in data
        assert ".pdf" in data["supported_formats"]
    
    def test_batch_upload_documents(self, client, mock_rag_engine):
        """测试批量上传文档"""
        # 创建多个测试文件
        files = []
        for i in range(3):
            content = f"测试文档 {i} 的内容"
            file_obj = io.BytesIO(content.encode())
            files.append(("files", (f"test_{i}.txt", file_obj, "text/plain")))
        
        response = client.post("/api/documents/batch-upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_files"] == 3
        assert data["success_count"] >= 0
    
    def test_get_document_stats(self, client, mock_rag_engine):
        """测试获取文档统计信息"""
        mock_rag_engine.get_system_stats = Mock(return_value={
            "document_processor": {"total_documents": 100},
            "system_info": {"app_name": "RAG系统"}
        })
        
        response = client.get("/api/documents/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "document_stats" in data


class TestQAAPI:
    """问答API测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_engine(self):
        with patch.object(rag_engine, 'initialized', True):
            yield rag_engine
    
    def test_ask_question_success(self, client, mock_rag_engine):
        """测试成功问答"""
        mock_rag_engine.answer_question = AsyncMock(return_value={
            "success": True,
            "question": "什么是人工智能？",
            "answer": "人工智能是计算机科学的一个分支。",
            "context_documents": [
                {
                    "content": "AI相关内容",
                    "metadata": {"source": "ai.txt"},
                    "similarity_score": 0.9,
                    "rank": 1
                }
            ],
            "generation_time": 2.5,
            "total_time": 3.0,
            "from_cache": False,
            "token_count": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "retrieval_stats": {
                "retrieved_count": 1,
                "similarity_threshold": 0.7,
                "avg_similarity": 0.9
            }
        })
        
        response = client.post(
            "/api/qa/ask",
            json={
                "question": "什么是人工智能？",
                "k": 5,
                "similarity_threshold": 0.7,
                "use_cache": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["answer"] == "人工智能是计算机科学的一个分支。"
        assert len(data["context_documents"]) == 1
        assert data["from_cache"] is False
    
    def test_ask_question_invalid_request(self, client, mock_rag_engine):
        """测试无效问答请求"""
        response = client.post(
            "/api/qa/ask",
            json={"question": ""}  # 空问题
        )
        
        assert response.status_code == 422  # 验证错误
    
    def test_ask_question_no_documents(self, client, mock_rag_engine):
        """测试未找到相关文档的问答"""
        mock_rag_engine.answer_question = AsyncMock(return_value={
            "success": False,
            "message": "未找到相关文档",
            "answer": "抱歉，我在知识库中没有找到相关信息。",
            "question": "未知问题",
            "context_documents": []
        })
        
        response = client.post(
            "/api/qa/ask",
            json={"question": "未知问题"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "未找到相关文档" in data["detail"]
    
    def test_batch_ask_questions(self, client, mock_rag_engine):
        """测试批量问答"""
        mock_rag_engine.batch_answer_questions = AsyncMock(return_value=[
            {
                "success": True,
                "question": "问题1",
                "answer": "答案1",
                "context_documents": []
            },
            {
                "success": True,
                "question": "问题2", 
                "answer": "答案2",
                "context_documents": []
            }
        ])
        
        response = client.post(
            "/api/qa/batch-ask",
            json={
                "questions": ["问题1", "问题2"],
                "k": 5,
                "use_cache": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_questions"] == 2
        assert len(data["results"]) == 2
    
    def test_get_qa_history(self, client, mock_rag_engine):
        """测试获取问答历史"""
        response = client.get("/api/qa/history?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "history" in data
    
    def test_get_question_suggestions(self, client, mock_rag_engine):
        """测试获取问题建议"""
        response = client.get("/api/qa/suggestions?query=机器学习&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
    
    def test_submit_feedback(self, client, mock_rag_engine):
        """测试提交问答反馈"""
        response = client.post(
            "/api/qa/feedback",
            params={
                "question": "测试问题",
                "answer": "测试答案",
                "rating": 5,
                "feedback": "很好的回答"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "feedback_id" in data
    
    def test_submit_invalid_feedback(self, client, mock_rag_engine):
        """测试提交无效反馈"""
        response = client.post(
            "/api/qa/feedback",
            params={
                "question": "测试问题",
                "answer": "测试答案",
                "rating": 10,  # 无效评分
                "feedback": "反馈"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "评分必须在1-5之间" in data["detail"]
    
    def test_get_qa_stats(self, client, mock_rag_engine):
        """测试获取问答统计"""
        mock_rag_engine.get_system_stats = Mock(return_value={
            "qa_processor": {"total_documents": 100},
            "metrics": {"cache_hit_rate": 0.8}
        })
        
        response = client.get("/api/qa/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "qa_stats" in data
        assert "settings" in data


class TestSystemAPI:
    """系统管理API测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_engine(self):
        with patch.object(rag_engine, 'initialized', True):
            yield rag_engine
    
    def test_health_check(self, client, mock_rag_engine):
        """测试健康检查"""
        mock_rag_engine.health_check = AsyncMock(return_value={
            "status": "healthy",
            "components": {
                "vector_database": {"status": "healthy", "document_count": 100},
                "llm": {"status": "healthy", "model": "qwen3:30b"},
                "cache": {"status": "healthy"}
            }
        })
        
        response = client.get("/api/system/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_get_system_stats(self, client, mock_rag_engine):
        """测试获取系统统计"""
        mock_rag_engine.get_system_stats = Mock(return_value={
            "system_info": {
                "app_name": "RAG知识库问答系统",
                "app_version": "2.0.0",
                "initialized": True
            },
            "document_processor": {"total_documents": 100},
            "qa_processor": {
                "embedding_model": "BAAI/bge-large-zh-v1.5",
                "llm_model": "qwen3:30b",
                "cache_enabled": True
            },
            "metrics": {
                "cache_hit_rate": 0.8,
                "cache_hits": 80,
                "cache_misses": 20,
                "system_cpu_percent": 45.2,
                "system_memory_percent": 60.1,
                "system_disk_percent": 30.5
            }
        })
        
        mock_rag_engine.get_supported_formats = Mock(return_value=[".pdf", ".txt", ".md", ".docx"])
        
        response = client.get("/api/system/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["system_info"]["app_name"] == "RAG知识库问答系统"
        assert data["metrics"]["cache_hit_rate"] == 0.8
        assert len(data["supported_formats"]) == 4
    
    def test_get_prometheus_metrics(self, client, mock_rag_engine):
        """测试获取Prometheus指标"""
        with patch('src.api.routes.system.get_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = "# HELP test_metric Test metric\ntest_metric 1.0"
            
            response = client.get("/api/system/metrics")
            
            assert response.status_code == 200
            assert "test_metric" in response.text
    
    def test_get_system_config(self, client, mock_rag_engine):
        """测试获取系统配置"""
        response = client.get("/api/system/config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "config" in data
        assert "app_name" in data["config"]
    
    def test_update_system_config(self, client, mock_rag_engine):
        """测试更新系统配置"""
        response = client.put(
            "/api/system/config",
            json={
                "chunk_size": 1200,
                "temperature": 0.8,
                "retrieval_k": 8
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "配置更新成功" in data["message"]
    
    def test_initialize_system(self, client, mock_rag_engine):
        """测试初始化系统"""
        mock_rag_engine.initialized = False
        mock_rag_engine.initialize = AsyncMock()
        
        response = client.post("/api/system/initialize")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["initialized"] is True
    
    def test_initialize_already_initialized(self, client, mock_rag_engine):
        """测试初始化已初始化的系统"""
        mock_rag_engine.initialized = True
        
        response = client.post("/api/system/initialize")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "已经初始化" in data["message"]
    
    def test_shutdown_system(self, client, mock_rag_engine):
        """测试关闭系统"""
        mock_rag_engine.close = AsyncMock()
        
        response = client.post("/api/system/shutdown")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["shutdown"] is True
    
    def test_get_version_info(self, client, mock_rag_engine):
        """测试获取版本信息"""
        response = client.get("/api/system/version")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "version_info" in data
        assert "app_name" in data["version_info"]
    
    def test_clear_system_cache(self, client, mock_rag_engine):
        """测试清除系统缓存"""
        response = client.delete("/api/system/cache?pattern=*")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "缓存清除成功" in data["message"]


class TestAPIMiddleware:
    """API中间件测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_cors_headers(self, client):
        """测试CORS头设置"""
        response = client.options("/api/health")
        
        # 检查CORS相关头部
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_request_id_header(self, client):
        """测试请求ID头部"""
        response = client.get("/api/health")
        
        assert "x-request-id" in response.headers
        assert "x-process-time" in response.headers
    
    def test_404_error_handling(self, client):
        """测试404错误处理"""
        response = client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "请求的资源不存在" in data["message"]
        assert "available_endpoints" in data


class TestAPIValidation:
    """API数据验证测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_question_validation(self, client):
        """测试问题验证"""
        # 空问题
        response = client.post("/api/qa/ask", json={"question": ""})
        assert response.status_code == 422
        
        # 超长问题
        long_question = "x" * 1001
        response = client.post("/api/qa/ask", json={"question": long_question})
        assert response.status_code == 422
    
    def test_parameter_validation(self, client):
        """测试参数验证"""
        # 无效的k值
        response = client.post("/api/qa/ask", json={
            "question": "测试问题",
            "k": 0  # 无效值
        })
        assert response.status_code == 422
        
        # 无效的相似度阈值
        response = client.post("/api/qa/ask", json={
            "question": "测试问题",
            "similarity_threshold": 1.5  # 超出范围
        })
        assert response.status_code == 422
    
    def test_batch_questions_validation(self, client):
        """测试批量问题验证"""
        # 空问题列表
        response = client.post("/api/qa/batch-ask", json={"questions": []})
        assert response.status_code == 422
        
        # 超过限制的问题数量
        too_many_questions = ["问题"] * 11
        response = client.post("/api/qa/batch-ask", json={"questions": too_many_questions})
        assert response.status_code == 422
"""
pytest配置文件
定义全局fixtures和测试配置
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# 设置异步测试事件循环
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def temp_file():
    """创建临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("这是一个测试文档的内容。\n包含多行文本用于测试。")
        temp_path = f.name
    
    yield temp_path
    
    # 清理
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_pdf_file():
    """创建示例PDF文件路径（模拟）"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # 写入一些假的PDF内容
        f.write(b'%PDF-1.4\n%fake pdf content for testing')
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_documents():
    """创建示例文档数据"""
    from langchain.schema import Document
    
    return [
        Document(
            page_content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            metadata={
                "source": "ai_intro.txt",
                "page": 1,
                "file_hash": "abc123",
                "chunk_id": "abc123_0"
            }
        ),
        Document(
            page_content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
            metadata={
                "source": "ml_basics.txt", 
                "page": 1,
                "file_hash": "def456",
                "chunk_id": "def456_0"
            }
        ),
        Document(
            page_content="深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。",
            metadata={
                "source": "dl_overview.txt",
                "page": 1,
                "file_hash": "ghi789",
                "chunk_id": "ghi789_0"
            }
        )
    ]


@pytest.fixture
def mock_settings():
    """模拟配置设置"""
    from src.config.settings import Settings
    
    return Settings(
        app_name="Test RAG System",
        debug=True,
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=3,
        similarity_threshold=0.7,
        max_tokens=1000,
        temperature=0.5,
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        chroma_host="localhost",
        chroma_port=8002,
        redis_host="localhost",
        redis_port=6379,
        max_file_size=10485760,  # 10MB for testing
        supported_formats=["pdf", "txt", "md", "docx"]
    )


@pytest.fixture
def mock_embedding_model():
    """模拟嵌入模型"""
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    mock_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0]
    ]
    return mock_model


@pytest.fixture
def mock_chroma_collection():
    """模拟Chroma集合"""
    mock_collection = Mock()
    mock_collection.count.return_value = 100
    mock_collection.get.return_value = {'ids': []}
    mock_collection.add.return_value = None
    mock_collection.delete.return_value = None
    mock_collection.query.return_value = {
        'documents': [['测试文档内容1', '测试文档内容2']],
        'metadatas': [[{'source': 'test1.txt'}, {'source': 'test2.txt'}]],
        'distances': [[0.1, 0.2]]
    }
    return mock_collection


@pytest.fixture
def mock_cache_client():
    """模拟缓存客户端"""
    mock_client = AsyncMock()
    mock_client.get.return_value = None
    mock_client.setex.return_value = True
    mock_client.delete.return_value = 1
    mock_client.ping.return_value = True
    return mock_client


@pytest.fixture
def mock_http_client():
    """模拟HTTP客户端"""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "这是一个测试回答。",
        "prompt_eval_count": 100,
        "eval_count": 50
    }
    mock_client.post.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_document_processor():
    """模拟文档处理器"""
    processor = Mock()
    processor.load_document = AsyncMock(return_value=[])
    processor.split_documents = Mock(return_value=[])
    processor.store_documents = AsyncMock(return_value={
        'stored_count': 1,
        'skipped_count': 0
    })
    processor.process_file = AsyncMock(return_value={
        'success': True,
        'message': '处理完成',
        'chunks_created': 1,
        'stored_count': 1
    })
    processor.delete_document = AsyncMock(return_value={
        'success': True,
        'deleted_count': 1
    })
    processor.get_collection_stats = Mock(return_value={
        'total_documents': 100,
        'collection_name': 'test_collection'
    })
    return processor


@pytest.fixture
def mock_qa_processor():
    """模拟问答处理器"""
    processor = Mock()
    processor.process_question = AsyncMock(return_value={
        'success': True,
        'question': '测试问题',
        'answer': '测试答案',
        'context_documents': [],
        'generation_time': 1.0,
        'total_time': 2.0,
        'from_cache': False
    })
    processor.batch_process_questions = AsyncMock(return_value=[])
    processor.get_stats = Mock(return_value={
        'total_documents': 100,
        'cache_enabled': True
    })
    return processor


@pytest.fixture
def mock_rag_engine():
    """模拟RAG引擎"""
    with patch('src.core.rag_engine.rag_engine') as mock_engine:
        mock_engine.initialized = True
        mock_engine.initialize = AsyncMock()
        mock_engine.close = AsyncMock()
        mock_engine.process_document = AsyncMock(return_value={
            'success': True,
            'message': '处理完成'
        })
        mock_engine.answer_question = AsyncMock(return_value={
            'success': True,
            'answer': '测试答案'
        })
        mock_engine.get_system_stats = Mock(return_value={
            'system_info': {'app_name': 'Test RAG'},
            'document_processor': {'total_documents': 100},
            'qa_processor': {'cache_enabled': True},
            'metrics': {'cache_hit_rate': 0.8}
        })
        mock_engine.health_check = AsyncMock(return_value={
            'status': 'healthy',
            'components': {}
        })
        mock_engine.get_supported_formats = Mock(return_value=['.pdf', '.txt'])
        yield mock_engine


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "unit: 单元测试"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试"
    )
    config.addinivalue_line(
        "markers", "performance: 性能测试"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试"
    )


# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试项目"""
    for item in items:
        # 为异步测试添加标记
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # 为慢速测试添加标记
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# 环境变量设置
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEBUG", "true")


# 清理fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后清理"""
    yield
    # 这里可以添加测试后的清理逻辑
    pass


# 数据库相关fixtures
@pytest.fixture
def mock_database_connection():
    """模拟数据库连接"""
    with patch('src.core.document_processor.chromadb.HttpClient') as mock_client:
        mock_instance = Mock()
        mock_instance.get_collection.return_value = Mock()
        mock_instance.create_collection.return_value = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


# 文件系统相关fixtures
@pytest.fixture
def mock_file_system():
    """模拟文件系统操作"""
    with patch('os.path.exists') as mock_exists, \
         patch('os.path.getsize') as mock_getsize, \
         patch('pathlib.Path.glob') as mock_glob:
        
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        mock_glob.return_value = []
        
        yield {
            'exists': mock_exists,
            'getsize': mock_getsize,
            'glob': mock_glob
        }


# 网络相关fixtures
@pytest.fixture
def mock_network():
    """模拟网络请求"""
    with patch('httpx.AsyncClient') as mock_client:
        mock_instance = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.status_code = 200
        mock_instance.post.return_value = mock_response
        mock_instance.get.return_value = mock_response
        mock_client.return_value = mock_instance
        yield mock_instance
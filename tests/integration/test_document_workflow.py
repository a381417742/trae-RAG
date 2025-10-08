"""
文档处理工作流集成测试
测试从文档上传到向量化存储的完整流程
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.core.rag_engine import RAGEngine
from src.core.document_processor import DocumentProcessor
from langchain.schema import Document


class TestDocumentWorkflowIntegration:
    """文档处理工作流集成测试"""
    
    @pytest.fixture
    async def rag_engine(self):
        """创建RAG引擎实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.document_processor.chromadb.HttpClient'), \
             patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'), \
             patch('src.utils.cache.init_cache_client'):
            
            engine = RAGEngine()
            
            # 模拟初始化
            engine.document_processor = Mock()
            engine.qa_processor = Mock()
            engine.initialized = True
            
            yield engine
    
    @pytest.fixture
    def test_documents_dir(self):
        """创建测试文档目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建不同格式的测试文件
            files = {}
            
            # TXT文件
            txt_file = Path(temp_dir) / "test_document.txt"
            txt_file.write_text("这是一个测试文档。\n包含人工智能相关内容。\n机器学习是AI的重要分支。")
            files['txt'] = str(txt_file)
            
            # Markdown文件
            md_file = Path(temp_dir) / "test_guide.md"
            md_file.write_text("""# 人工智能指南

## 什么是人工智能
人工智能（AI）是计算机科学的一个分支。

## 机器学习
机器学习是AI的核心技术之一。

### 深度学习
深度学习使用神经网络进行学习。
""")
            files['md'] = str(md_file)
            
            # 创建一个不支持的格式文件
            unsupported_file = Path(temp_dir) / "test.xyz"
            unsupported_file.write_text("不支持的格式")
            files['unsupported'] = str(unsupported_file)
            
            yield temp_dir, files
    
    @pytest.mark.asyncio
    async def test_single_document_processing_workflow(self, rag_engine, test_documents_dir):
        """测试单个文档处理完整工作流"""
        temp_dir, files = test_documents_dir
        txt_file = files['txt']
        
        # 模拟文档处理器的各个步骤
        mock_documents = [
            Document(
                page_content="这是一个测试文档。包含人工智能相关内容。",
                metadata={
                    'file_path': txt_file,
                    'file_name': 'test_document.txt',
                    'file_hash': 'abc123',
                    'chunk_id': 'abc123_0'
                }
            )
        ]
        
        # 模拟处理结果
        rag_engine.document_processor.process_file = AsyncMock(return_value={
            'success': True,
            'message': '文档处理完成',
            'file_path': txt_file,
            'original_docs': 1,
            'chunks_created': 2,
            'stored_count': 2,
            'processing_time': 1.5
        })
        
        # 执行处理
        result = await rag_engine.process_document(txt_file)
        
        # 验证结果
        assert result['success'] is True
        assert result['file_path'] == txt_file
        assert result['chunks_created'] == 2
        assert result['stored_count'] == 2
        
        # 验证调用
        rag_engine.document_processor.process_file.assert_called_once_with(txt_file)
    
    @pytest.mark.asyncio
    async def test_directory_processing_workflow(self, rag_engine, test_documents_dir):
        """测试目录处理完整工作流"""
        temp_dir, files = test_documents_dir
        
        # 模拟目录处理结果
        rag_engine.document_processor.process_directory = AsyncMock(return_value={
            'success_count': 2,
            'error_count': 1,  # 不支持的格式文件
            'total_files': 3,
            'total_chunks': 5,
            'processed_files': [
                {
                    'success': True,
                    'file_path': files['txt'],
                    'chunks_created': 2
                },
                {
                    'success': True,
                    'file_path': files['md'],
                    'chunks_created': 3
                }
            ],
            'errors': [
                {
                    'success': False,
                    'file_path': files['unsupported'],
                    'error': '不支持的文件格式'
                }
            ]
        })
        
        # 执行目录处理
        result = await rag_engine.process_directory(temp_dir)
        
        # 验证结果
        assert result['success_count'] == 2
        assert result['error_count'] == 1
        assert result['total_files'] == 3
        assert result['total_chunks'] == 5
        assert len(result['processed_files']) == 2
        assert len(result['errors']) == 1
    
    @pytest.mark.asyncio
    async def test_document_processing_error_handling(self, rag_engine):
        """测试文档处理错误处理"""
        nonexistent_file = "/nonexistent/file.txt"
        
        # 模拟文件不存在错误
        rag_engine.document_processor.process_file = AsyncMock(return_value={
            'success': False,
            'message': '文件不存在',
            'file_path': nonexistent_file,
            'error': 'FileNotFoundError'
        })
        
        result = await rag_engine.process_document(nonexistent_file)
        
        assert result['success'] is False
        assert 'file_path' in result
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_document_deletion_workflow(self, rag_engine, test_documents_dir):
        """测试文档删除工作流"""
        temp_dir, files = test_documents_dir
        txt_file = files['txt']
        
        # 模拟删除结果
        rag_engine.document_processor.delete_document = AsyncMock(return_value={
            'success': True,
            'message': '文档删除成功',
            'file_path': txt_file,
            'deleted_count': 2
        })
        
        result = await rag_engine.delete_document(txt_file)
        
        assert result['success'] is True
        assert result['file_path'] == txt_file
        assert result['deleted_count'] == 2
    
    @pytest.mark.asyncio
    async def test_document_processing_with_metrics(self, rag_engine, test_documents_dir):
        """测试带指标收集的文档处理"""
        temp_dir, files = test_documents_dir
        txt_file = files['txt']
        
        # 模拟处理结果和统计信息
        rag_engine.document_processor.process_file = AsyncMock(return_value={
            'success': True,
            'chunks_created': 3,
            'stored_count': 3
        })
        
        rag_engine.document_processor.get_collection_stats = Mock(return_value={
            'total_documents': 103,  # 原有100 + 新增3
            'collection_name': 'test_collection'
        })
        
        # 模拟指标收集器
        with patch('src.core.rag_engine.metrics_collector') as mock_metrics:
            result = await rag_engine.process_document(txt_file)
            
            # 验证指标记录
            mock_metrics.record_document_processing.assert_called()
            mock_metrics.update_vector_db_documents.assert_called_with(103)
    
    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, rag_engine, test_documents_dir):
        """测试并发文档处理"""
        temp_dir, files = test_documents_dir
        
        # 模拟并发处理多个文件
        async def mock_process_file(file_path):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return {
                'success': True,
                'file_path': file_path,
                'chunks_created': 2,
                'stored_count': 2
            }
        
        rag_engine.document_processor.process_file = mock_process_file
        
        # 并发处理多个文件
        tasks = [
            rag_engine.process_document(files['txt']),
            rag_engine.process_document(files['md'])
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证所有文件都处理成功
        assert all(result['success'] for result in results)
        assert len(results) == 2


class TestDocumentProcessorIntegration:
    """文档处理器集成测试"""
    
    @pytest.fixture
    def processor(self):
        """创建文档处理器实例"""
        with patch('src.core.document_processor.SentenceTransformer') as mock_st, \
             patch('src.core.document_processor.chromadb.HttpClient') as mock_chroma:
            
            # 模拟嵌入模型
            mock_embedding_model = Mock()
            mock_embedding_model.encode.return_value = Mock()
            mock_embedding_model.encode.return_value.tolist.return_value = [
                [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
            ]
            mock_st.return_value = mock_embedding_model
            
            # 模拟Chroma客户端
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_collection.get.return_value = {'ids': []}
            mock_collection.add.return_value = None
            
            mock_client = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client
            
            processor = DocumentProcessor()
            processor.collection = mock_collection
            
            yield processor
    
    @pytest.mark.asyncio
    async def test_full_document_processing_pipeline(self, processor):
        """测试完整的文档处理管道"""
        # 创建测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("人工智能是计算机科学的一个重要分支。\n")
            f.write("机器学习是人工智能的核心技术。\n")
            f.write("深度学习是机器学习的一个子领域。")
            temp_file = f.name
        
        try:
            # 模拟文档加载器
            with patch('src.core.document_processor.TextLoader') as mock_loader:
                mock_instance = Mock()
                mock_instance.load.return_value = [
                    Document(
                        page_content="人工智能是计算机科学的一个重要分支。机器学习是人工智能的核心技术。深度学习是机器学习的一个子领域。",
                        metadata={"source": temp_file}
                    )
                ]
                mock_loader.return_value = mock_instance
                
                # 执行完整处理流程
                result = await processor.process_file(temp_file)
                
                # 验证结果
                assert result['success'] is True
                assert result['file_path'] == temp_file
                assert 'chunks_created' in result
                assert 'stored_count' in result
                
                # 验证各个步骤都被调用
                mock_loader.assert_called_once_with(temp_file)
                processor.collection.add.assert_called_once()
        
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_document_chunking_integration(self, processor):
        """测试文档分块集成"""
        # 创建长文档
        long_content = "人工智能。" * 200  # 创建一个需要分块的长文档
        
        documents = [
            Document(
                page_content=long_content,
                metadata={"source": "long_doc.txt", "file_hash": "test123"}
            )
        ]
        
        # 执行分块
        chunks = processor.split_documents(documents)
        
        # 验证分块结果
        assert len(chunks) > 1  # 应该被分成多个块
        
        # 验证每个块都有正确的元数据
        for i, chunk in enumerate(chunks):
            assert 'chunk_id' in chunk.metadata
            assert 'chunk_index' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['file_hash'] == "test123"
    
    @pytest.mark.asyncio
    async def test_embedding_and_storage_integration(self, processor):
        """测试嵌入生成和存储集成"""
        # 准备测试文档
        documents = [
            Document(
                page_content="人工智能测试内容",
                metadata={"chunk_id": "test_1", "source": "test.txt"}
            ),
            Document(
                page_content="机器学习测试内容", 
                metadata={"chunk_id": "test_2", "source": "test.txt"}
            )
        ]
        
        # 执行存储
        result = await processor.store_documents(documents)
        
        # 验证结果
        assert result['stored_count'] == 2
        assert result['skipped_count'] == 0
        
        # 验证嵌入生成被调用
        processor.embedding_model.encode.assert_called_once()
        
        # 验证存储被调用
        processor.collection.add.assert_called_once()
        
        # 验证调用参数
        call_args = processor.collection.add.call_args
        embeddings, metadatas, texts, ids = call_args[0]
        
        assert len(embeddings) == 2
        assert len(metadatas) == 2
        assert len(texts) == 2
        assert len(ids) == 2
        assert ids == ["test_1", "test_2"]


class TestDocumentWorkflowErrorScenarios:
    """文档工作流错误场景测试"""
    
    @pytest.fixture
    def rag_engine(self):
        """创建RAG引擎实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.document_processor.chromadb.HttpClient'), \
             patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'):
            
            engine = RAGEngine()
            engine.document_processor = Mock()
            engine.initialized = True
            yield engine
    
    @pytest.mark.asyncio
    async def test_processing_corrupted_file(self, rag_engine):
        """测试处理损坏文件"""
        corrupted_file = "/path/to/corrupted.pdf"
        
        # 模拟文件损坏错误
        rag_engine.document_processor.process_file = AsyncMock(return_value={
            'success': False,
            'message': '文件损坏无法读取',
            'file_path': corrupted_file,
            'error': 'CorruptedFileError'
        })
        
        result = await rag_engine.process_document(corrupted_file)
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_storage_failure_handling(self, rag_engine):
        """测试存储失败处理"""
        test_file = "/path/to/test.txt"
        
        # 模拟存储失败
        rag_engine.document_processor.process_file = AsyncMock(return_value={
            'success': False,
            'message': '向量数据库连接失败',
            'file_path': test_file,
            'error': 'DatabaseConnectionError'
        })
        
        result = await rag_engine.process_document(test_file)
        
        assert result['success'] is False
        assert 'DatabaseConnectionError' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, rag_engine):
        """测试嵌入生成失败"""
        test_file = "/path/to/test.txt"
        
        # 模拟嵌入生成失败
        rag_engine.document_processor.process_file = AsyncMock(return_value={
            'success': False,
            'message': '嵌入模型加载失败',
            'file_path': test_file,
            'error': 'EmbeddingModelError'
        })
        
        result = await rag_engine.process_document(test_file)
        
        assert result['success'] is False
        assert 'EmbeddingModelError' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_partial_directory_processing_failure(self, rag_engine):
        """测试目录部分处理失败"""
        test_dir = "/path/to/test_dir"
        
        # 模拟部分文件处理失败
        rag_engine.document_processor.process_directory = AsyncMock(return_value={
            'success_count': 2,
            'error_count': 1,
            'total_files': 3,
            'total_chunks': 4,
            'processed_files': [
                {'success': True, 'file_path': 'file1.txt', 'chunks_created': 2},
                {'success': True, 'file_path': 'file2.txt', 'chunks_created': 2}
            ],
            'errors': [
                {'success': False, 'file_path': 'file3.pdf', 'error': 'PDF解析失败'}
            ]
        })
        
        result = await rag_engine.process_directory(test_dir)
        
        # 验证部分成功的结果
        assert result['success_count'] == 2
        assert result['error_count'] == 1
        assert len(result['processed_files']) == 2
        assert len(result['errors']) == 1


class TestDocumentWorkflowPerformance:
    """文档工作流性能测试"""
    
    @pytest.fixture
    def rag_engine(self):
        """创建RAG引擎实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.document_processor.chromadb.HttpClient'):
            
            engine = RAGEngine()
            engine.document_processor = Mock()
            engine.initialized = True
            yield engine
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_document_processing_performance(self, rag_engine):
        """测试大文档处理性能"""
        large_file = "/path/to/large_document.txt"
        
        # 模拟大文档处理（模拟较长处理时间）
        async def slow_process_file(file_path):
            await asyncio.sleep(0.5)  # 模拟处理时间
            return {
                'success': True,
                'file_path': file_path,
                'original_docs': 1,
                'chunks_created': 100,  # 大量分块
                'stored_count': 100,
                'processing_time': 0.5
            }
        
        rag_engine.document_processor.process_file = slow_process_file
        
        import time
        start_time = time.time()
        result = await rag_engine.process_document(large_file)
        end_time = time.time()
        
        # 验证处理成功且在合理时间内完成
        assert result['success'] is True
        assert result['chunks_created'] == 100
        assert end_time - start_time < 1.0  # 应在1秒内完成
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_processing_performance(self, rag_engine):
        """测试并发处理性能"""
        files = [f"/path/to/file_{i}.txt" for i in range(5)]
        
        # 模拟并发处理
        async def mock_process_file(file_path):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return {
                'success': True,
                'file_path': file_path,
                'chunks_created': 5,
                'stored_count': 5
            }
        
        rag_engine.document_processor.process_file = mock_process_file
        
        # 测试并发处理
        import time
        start_time = time.time()
        
        tasks = [rag_engine.process_document(file_path) for file_path in files]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # 验证并发处理效果
        assert len(results) == 5
        assert all(result['success'] for result in results)
        # 并发处理应该比顺序处理快
        assert end_time - start_time < 0.5  # 应该远少于5 * 0.1 = 0.5秒
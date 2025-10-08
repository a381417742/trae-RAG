"""
文档处理器单元测试
测试文档加载、分块、向量化和存储功能
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

from src.core.document_processor import DocumentProcessor
from langchain.schema import Document


class TestDocumentProcessor:
    """文档处理器测试类"""
    
    @pytest.fixture
    def processor(self):
        """创建文档处理器实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.document_processor.chromadb.HttpClient'):
            processor = DocumentProcessor()
            processor.embedding_model = Mock()
            processor.collection = Mock()
            return processor
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        return [
            Document(
                page_content="这是第一个测试文档的内容。",
                metadata={"source": "test1.txt", "page": 1}
            ),
            Document(
                page_content="这是第二个测试文档的内容，包含更多信息。",
                metadata={"source": "test2.txt", "page": 1}
            )
        ]
    
    def test_processor_initialization(self, processor):
        """测试处理器初始化"""
        assert processor is not None
        assert processor.text_splitter is not None
        assert processor.embedding_model is not None
        assert processor.collection is not None
        assert processor.supported_formats is not None
    
    def test_supported_formats(self, processor):
        """测试支持的文件格式"""
        expected_formats = {'.pdf', '.txt', '.md', '.docx'}
        actual_formats = set(processor.supported_formats.keys())
        assert expected_formats.issubset(actual_formats)
    
    def test_get_file_hash(self, processor):
        """测试文件哈希计算"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            hash1 = processor._get_file_hash(temp_file)
            hash2 = processor._get_file_hash(temp_file)
            
            # 相同文件应该有相同哈希
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5哈希长度
        finally:
            os.unlink(temp_file)
    
    def test_is_file_processed(self, processor):
        """测试文件处理状态检查"""
        # 模拟未处理的文件
        processor.collection.get.return_value = {'ids': []}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            result = processor._is_file_processed(temp_file)
            assert result is False
            
            # 模拟已处理的文件
            processor.collection.get.return_value = {'ids': ['test_id']}
            result = processor._is_file_processed(temp_file)
            assert result is True
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_load_document_txt(self, processor):
        """测试加载TXT文档"""
        # 创建临时TXT文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("这是一个测试文档的内容。\n包含多行文本。")
            temp_file = f.name
        
        try:
            # 模拟文件未处理
            processor._is_file_processed = Mock(return_value=False)
            
            with patch('src.core.document_processor.TextLoader') as mock_loader:
                mock_instance = Mock()
                mock_instance.load.return_value = [
                    Document(page_content="测试内容", metadata={})
                ]
                mock_loader.return_value = mock_instance
                
                documents = await processor.load_document(temp_file)
                
                assert len(documents) == 1
                assert documents[0].metadata['file_path'] == temp_file
                assert 'file_hash' in documents[0].metadata
                assert 'processed_at' in documents[0].metadata
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_load_document_already_processed(self, processor):
        """测试加载已处理的文档"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            # 模拟文件已处理
            processor._is_file_processed = Mock(return_value=True)
            
            documents = await processor.load_document(temp_file)
            assert len(documents) == 0
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_load_document_unsupported_format(self, processor):
        """测试加载不支持的文件格式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                await processor.load_document(temp_file)
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_load_document_not_found(self, processor):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            await processor.load_document("/nonexistent/file.txt")
    
    def test_split_documents(self, processor, sample_documents):
        """测试文档分块"""
        chunks = processor.split_documents(sample_documents)
        
        assert len(chunks) >= len(sample_documents)
        
        # 检查块的元数据
        for i, chunk in enumerate(chunks):
            assert 'chunk_id' in chunk.metadata
            assert 'chunk_index' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
            assert chunk.metadata['chunk_index'] == i
            assert chunk.metadata['chunk_size'] == len(chunk.page_content)
    
    def test_generate_embeddings(self, processor):
        """测试生成嵌入向量"""
        texts = ["这是第一个文本", "这是第二个文本"]
        
        # 模拟嵌入模型返回
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        processor.embedding_model.encode.return_value = Mock()
        processor.embedding_model.encode.return_value.tolist.return_value = mock_embeddings
        
        embeddings = processor.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        assert embeddings == mock_embeddings
        processor.embedding_model.encode.assert_called_once_with(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
    
    @pytest.mark.asyncio
    async def test_store_documents(self, processor, sample_documents):
        """测试存储文档"""
        # 添加必要的元数据
        for i, doc in enumerate(sample_documents):
            doc.metadata['chunk_id'] = f"test_chunk_{i}"
        
        # 模拟嵌入生成
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        processor.generate_embeddings = Mock(return_value=mock_embeddings)
        
        # 模拟集合添加
        processor.collection.add = Mock()
        
        result = await processor.store_documents(sample_documents)
        
        assert result['stored_count'] == len(sample_documents)
        assert result['skipped_count'] == 0
        processor.collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_empty_documents(self, processor):
        """测试存储空文档列表"""
        result = await processor.store_documents([])
        
        assert result['stored_count'] == 0
        assert result['skipped_count'] == 0
    
    @pytest.mark.asyncio
    async def test_process_file_success(self, processor):
        """测试成功处理文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试文档内容")
            temp_file = f.name
        
        try:
            # 模拟各个步骤
            processor.load_document = AsyncMock(return_value=[
                Document(page_content="测试内容", metadata={'chunk_id': 'test_1'})
            ])
            processor.split_documents = Mock(return_value=[
                Document(page_content="测试内容", metadata={'chunk_id': 'test_1'})
            ])
            processor.store_documents = AsyncMock(return_value={
                'stored_count': 1, 'skipped_count': 0
            })
            
            result = await processor.process_file(temp_file)
            
            assert result['success'] is True
            assert result['file_path'] == temp_file
            assert result['stored_count'] == 1
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_process_file_already_processed(self, processor):
        """测试处理已处理的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            # 模拟文件已处理
            processor.load_document = AsyncMock(return_value=[])
            
            result = await processor.process_file(temp_file)
            
            assert result['success'] is True
            assert result['message'] == "文件已处理，跳过"
            assert result['stored_count'] == 0
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_process_directory(self, processor):
        """测试处理目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            test_files = []
            for i in range(3):
                file_path = os.path.join(temp_dir, f"test_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"测试文档 {i}")
                test_files.append(file_path)
            
            # 模拟处理结果
            processor.process_file = AsyncMock(return_value={
                'success': True, 'chunks_created': 1
            })
            
            result = await processor.process_directory(temp_dir)
            
            assert result['total_files'] == 3
            assert result['success_count'] == 3
            assert result['error_count'] == 0
            assert result['total_chunks'] == 3
    
    @pytest.mark.asyncio
    async def test_process_directory_not_found(self, processor):
        """测试处理不存在的目录"""
        with pytest.raises(FileNotFoundError, match="目录不存在"):
            await processor.process_directory("/nonexistent/directory")
    
    def test_get_collection_stats(self, processor):
        """测试获取集合统计信息"""
        processor.collection.count.return_value = 100
        
        stats = processor.get_collection_stats()
        
        assert stats['total_documents'] == 100
        assert 'collection_name' in stats
        assert 'embedding_model' in stats
    
    @pytest.mark.asyncio
    async def test_delete_document(self, processor):
        """测试删除文档"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            # 模拟查找结果
            processor.collection.get.return_value = {
                'ids': ['doc1', 'doc2']
            }
            processor.collection.delete = Mock()
            
            result = await processor.delete_document(temp_file)
            
            assert result['success'] is True
            assert result['deleted_count'] == 2
            processor.collection.delete.assert_called_once_with(ids=['doc1', 'doc2'])
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, processor):
        """测试删除不存在的文档"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            # 模拟未找到文档
            processor.collection.get.return_value = {'ids': []}
            
            result = await processor.delete_document(temp_file)
            
            assert result['success'] is True
            assert result['deleted_count'] == 0
            assert result['message'] == "未找到相关文档"
        finally:
            os.unlink(temp_file)


class TestDocumentProcessorIntegration:
    """文档处理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_document_processing_pipeline(self):
        """测试完整的文档处理流程"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.document_processor.chromadb.HttpClient'):
            
            processor = DocumentProcessor()
            processor.embedding_model = Mock()
            processor.collection = Mock()
            
            # 创建测试文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("这是一个完整的测试文档。\n包含多行内容用于测试分块功能。")
                temp_file = f.name
            
            try:
                # 模拟各个组件
                processor._is_file_processed = Mock(return_value=False)
                processor.embedding_model.encode.return_value = Mock()
                processor.embedding_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
                processor.collection.add = Mock()
                processor.collection.count.return_value = 1
                
                with patch('src.core.document_processor.TextLoader') as mock_loader:
                    mock_instance = Mock()
                    mock_instance.load.return_value = [
                        Document(page_content="测试内容", metadata={})
                    ]
                    mock_loader.return_value = mock_instance
                    
                    # 执行完整流程
                    result = await processor.process_file(temp_file)
                    
                    # 验证结果
                    assert result['success'] is True
                    assert 'chunks_created' in result
                    assert 'stored_count' in result
                    
                    # 验证统计信息
                    stats = processor.get_collection_stats()
                    assert stats['total_documents'] == 1
            finally:
                os.unlink(temp_file)
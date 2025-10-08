"""
问答处理器单元测试
测试问题检索、答案生成和缓存功能
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import httpx

from src.core.qa_processor import QAProcessor


class TestQAProcessor:
    """问答处理器测试类"""
    
    @pytest.fixture
    def processor(self):
        """创建问答处理器实例"""
        with patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'), \
             patch('src.core.qa_processor.get_cache_client'):
            processor = QAProcessor()
            processor.embedding_model = Mock()
            processor.collection = Mock()
            processor.http_client = AsyncMock()
            processor.cache_client = Mock()
            return processor
    
    @pytest.fixture
    def sample_question(self):
        """示例问题"""
        return "什么是人工智能？"
    
    @pytest.fixture
    def sample_documents(self):
        """示例检索文档"""
        return [
            {
                'content': '人工智能是计算机科学的一个分支。',
                'metadata': {'source': 'ai_intro.txt', 'page': 1},
                'similarity_score': 0.9,
                'rank': 1
            },
            {
                'content': '机器学习是人工智能的重要组成部分。',
                'metadata': {'source': 'ml_basics.txt', 'page': 2},
                'similarity_score': 0.8,
                'rank': 2
            }
        ]
    
    def test_processor_initialization(self, processor):
        """测试处理器初始化"""
        assert processor is not None
        assert processor.embedding_model is not None
        assert processor.collection is not None
        assert processor.http_client is not None
        assert processor.system_prompt is not None
    
    def test_generate_cache_key(self, processor, sample_question):
        """测试缓存键生成"""
        key1 = processor._generate_cache_key(sample_question, k=5)
        key2 = processor._generate_cache_key(sample_question, k=5)
        key3 = processor._generate_cache_key(sample_question, k=10)
        
        # 相同参数应生成相同键
        assert key1 == key2
        # 不同参数应生成不同键
        assert key1 != key3
        # 键应以"qa:"开头
        assert key1.startswith("qa:")
    
    @pytest.mark.asyncio
    async def test_get_cached_answer_hit(self, processor, sample_question):
        """测试缓存命中"""
        cache_key = "test_key"
        cached_data = {"answer": "缓存的答案", "from_cache": True}
        
        processor.cache_client.get.return_value = json.dumps(cached_data)
        
        result = await processor._get_cached_answer(cache_key)
        
        assert result == cached_data
        processor.cache_client.get.assert_called_once_with(cache_key)
    
    @pytest.mark.asyncio
    async def test_get_cached_answer_miss(self, processor):
        """测试缓存未命中"""
        cache_key = "test_key"
        processor.cache_client.get.return_value = None
        
        result = await processor._get_cached_answer(cache_key)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_cached_answer(self, processor):
        """测试设置缓存"""
        cache_key = "test_key"
        answer_data = {"answer": "测试答案"}
        
        processor.cache_client.setex = AsyncMock()
        
        await processor._set_cached_answer(cache_key, answer_data)
        
        processor.cache_client.setex.assert_called_once()
        call_args = processor.cache_client.setex.call_args
        assert call_args[0][0] == cache_key
        assert json.loads(call_args[0][2]) == answer_data
    
    def test_generate_question_embedding(self, processor, sample_question):
        """测试问题嵌入向量生成"""
        mock_embedding = [[0.1, 0.2, 0.3]]
        processor.embedding_model.encode.return_value = mock_embedding
        
        result = processor.generate_question_embedding(sample_question)
        
        assert result == [0.1, 0.2, 0.3]
        processor.embedding_model.encode.assert_called_once_with([sample_question])
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self, processor, sample_question):
        """测试成功检索文档"""
        # 模拟嵌入向量生成
        processor.generate_question_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # 模拟检索结果
        mock_results = {
            'documents': [['文档1内容', '文档2内容']],
            'metadatas': [[{'source': 'doc1.txt'}, {'source': 'doc2.txt'}]],
            'distances': [[0.1, 0.2]]
        }
        processor.collection.query.return_value = mock_results
        
        documents = await processor.retrieve_documents(sample_question, k=2)
        
        assert len(documents) == 2
        assert documents[0]['content'] == '文档1内容'
        assert documents[0]['similarity_score'] == 0.9  # 1 - 0.1
        assert documents[1]['similarity_score'] == 0.8  # 1 - 0.2
        assert documents[0]['rank'] == 1
        assert documents[1]['rank'] == 2
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_with_threshold(self, processor, sample_question):
        """测试带相似度阈值的文档检索"""
        processor.generate_question_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        # 模拟检索结果，包含低相似度文档
        mock_results = {
            'documents': [['高相似度文档', '低相似度文档']],
            'metadatas': [[{'source': 'doc1.txt'}, {'source': 'doc2.txt'}]],
            'distances': [[0.1, 0.6]]  # 第二个文档相似度为0.4，低于阈值0.5
        }
        processor.collection.query.return_value = mock_results
        
        documents = await processor.retrieve_documents(
            sample_question, k=2, similarity_threshold=0.5
        )
        
        # 只应返回高相似度文档
        assert len(documents) == 1
        assert documents[0]['content'] == '高相似度文档'
        assert documents[0]['similarity_score'] == 0.9
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_empty_result(self, processor, sample_question):
        """测试检索空结果"""
        processor.generate_question_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        
        mock_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        processor.collection.query.return_value = mock_results
        
        documents = await processor.retrieve_documents(sample_question)
        
        assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_generate_answer_success(self, processor, sample_question, sample_documents):
        """测试成功生成答案"""
        # 模拟Ollama API响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "人工智能是一门计算机科学分支。",
            "prompt_eval_count": 100,
            "eval_count": 50
        }
        processor.http_client.post.return_value = mock_response
        
        result = await processor.generate_answer(sample_question, sample_documents)
        
        assert result['answer'] == "人工智能是一门计算机科学分支。"
        assert result['question'] == sample_question
        assert len(result['context_documents']) == 2
        assert 'generation_time' in result
        assert 'token_count' in result
        assert result['token_count']['prompt_tokens'] == 100
        assert result['token_count']['completion_tokens'] == 50
    
    @pytest.mark.asyncio
    async def test_generate_answer_api_error(self, processor, sample_question, sample_documents):
        """测试API调用错误"""
        # 模拟HTTP错误
        processor.http_client.post.side_effect = httpx.HTTPError("API错误")
        
        with pytest.raises(httpx.HTTPError):
            await processor.generate_answer(sample_question, sample_documents)
    
    @pytest.mark.asyncio
    async def test_process_question_success(self, processor, sample_question):
        """测试成功处理问题"""
        # 模拟检索文档
        processor.retrieve_documents = AsyncMock(return_value=[
            {
                'content': '人工智能相关内容',
                'metadata': {'source': 'ai.txt'},
                'similarity_score': 0.9,
                'rank': 1
            }
        ])
        
        # 模拟生成答案
        processor.generate_answer = AsyncMock(return_value={
            'answer': '人工智能是计算机科学分支',
            'question': sample_question,
            'context_documents': [],
            'generation_time': 1.5,
            'model': 'qwen3:30b',
            'token_count': {'total_tokens': 150}
        })
        
        # 模拟缓存操作
        processor._get_cached_answer = AsyncMock(return_value=None)
        processor._set_cached_answer = AsyncMock()
        
        result = await processor.process_question(sample_question)
        
        assert result['success'] is True
        assert result['answer'] == '人工智能是计算机科学分支'
        assert result['from_cache'] is False
        assert 'total_time' in result
        assert 'retrieval_stats' in result
    
    @pytest.mark.asyncio
    async def test_process_question_from_cache(self, processor, sample_question):
        """测试从缓存返回答案"""
        cached_answer = {
            'success': True,
            'answer': '缓存的答案',
            'question': sample_question,
            'from_cache': True
        }
        
        processor._get_cached_answer = AsyncMock(return_value=cached_answer)
        
        result = await processor.process_question(sample_question, use_cache=True)
        
        assert result['success'] is True
        assert result['answer'] == '缓存的答案'
        assert result['from_cache'] is True
    
    @pytest.mark.asyncio
    async def test_process_question_no_documents(self, processor, sample_question):
        """测试未找到相关文档"""
        processor.retrieve_documents = AsyncMock(return_value=[])
        processor._get_cached_answer = AsyncMock(return_value=None)
        
        result = await processor.process_question(sample_question)
        
        assert result['success'] is False
        assert result['message'] == "未找到相关文档"
        assert "未找到与您问题相关的信息" in result['answer']
    
    @pytest.mark.asyncio
    async def test_process_question_error(self, processor, sample_question):
        """测试处理问题时发生错误"""
        processor.retrieve_documents = AsyncMock(side_effect=Exception("检索错误"))
        processor._get_cached_answer = AsyncMock(return_value=None)
        
        result = await processor.process_question(sample_question)
        
        assert result['success'] is False
        assert "问答处理失败" in result['message']
        assert "处理您的问题时出现了错误" in result['answer']
    
    @pytest.mark.asyncio
    async def test_batch_process_questions(self, processor):
        """测试批量处理问题"""
        questions = ["问题1", "问题2", "问题3"]
        
        # 模拟单个问题处理结果
        processor.process_question = AsyncMock(return_value={
            'success': True,
            'answer': '答案',
            'question': '问题'
        })
        
        results = await processor.batch_process_questions(questions)
        
        assert len(results) == 3
        assert all(result['success'] for result in results)
        assert processor.process_question.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_process_questions_with_errors(self, processor):
        """测试批量处理包含错误的问题"""
        questions = ["问题1", "问题2"]
        
        # 模拟一个成功一个失败
        side_effects = [
            {'success': True, 'answer': '答案1'},
            Exception("处理错误")
        ]
        processor.process_question = AsyncMock(side_effect=side_effects)
        
        results = await processor.batch_process_questions(questions)
        
        assert len(results) == 2
        assert results[0]['success'] is True
        assert results[1]['success'] is False
        assert "处理失败" in results[1]['message']
    
    def test_get_stats(self, processor):
        """测试获取统计信息"""
        processor.collection.count.return_value = 100
        
        stats = processor.get_stats()
        
        assert stats['total_documents'] == 100
        assert 'embedding_model' in stats
        assert 'llm_model' in stats
        assert 'cache_enabled' in stats
    
    @pytest.mark.asyncio
    async def test_close(self, processor):
        """测试关闭资源"""
        processor.http_client.aclose = AsyncMock()
        processor.cache_client.close = AsyncMock()
        
        await processor.close()
        
        processor.http_client.aclose.assert_called_once()
        processor.cache_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, processor):
        """测试异步上下文管理器"""
        processor.close = AsyncMock()
        
        async with processor as p:
            assert p is processor
        
        processor.close.assert_called_once()


class TestQAProcessorIntegration:
    """问答处理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_qa_pipeline(self):
        """测试完整的问答流程"""
        with patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'), \
             patch('src.core.qa_processor.get_cache_client'):
            
            processor = QAProcessor()
            processor.embedding_model = Mock()
            processor.collection = Mock()
            processor.http_client = AsyncMock()
            processor.cache_client = Mock()
            
            # 模拟完整流程
            question = "什么是机器学习？"
            
            # 1. 缓存未命中
            processor._get_cached_answer = AsyncMock(return_value=None)
            
            # 2. 生成问题嵌入
            processor.embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
            
            # 3. 检索文档
            processor.collection.query.return_value = {
                'documents': [['机器学习是人工智能的分支']],
                'metadatas': [[{'source': 'ml.txt'}]],
                'distances': [[0.1]]
            }
            
            # 4. 生成答案
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "机器学习是人工智能的重要分支。",
                "prompt_eval_count": 80,
                "eval_count": 40
            }
            processor.http_client.post.return_value = mock_response
            
            # 5. 设置缓存
            processor._set_cached_answer = AsyncMock()
            
            # 执行完整流程
            result = await processor.process_question(question)
            
            # 验证结果
            assert result['success'] is True
            assert result['answer'] == "机器学习是人工智能的重要分支。"
            assert result['from_cache'] is False
            assert len(result['context_documents']) == 1
            assert result['retrieval_stats']['retrieved_count'] == 1
            assert result['token_count']['total_tokens'] == 120
            
            # 验证缓存设置
            processor._set_cached_answer.assert_called_once()


class TestQAProcessorEdgeCases:
    """问答处理器边界情况测试"""
    
    @pytest.fixture
    def processor(self):
        """创建问答处理器实例"""
        with patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'), \
             patch('src.core.qa_processor.get_cache_client'):
            processor = QAProcessor()
            processor.embedding_model = Mock()
            processor.collection = Mock()
            processor.http_client = AsyncMock()
            processor.cache_client = None  # 无缓存客户端
            return processor
    
    @pytest.mark.asyncio
    async def test_no_cache_client(self, processor):
        """测试无缓存客户端的情况"""
        question = "测试问题"
        
        # 模拟检索和生成
        processor.retrieve_documents = AsyncMock(return_value=[{
            'content': '测试内容',
            'metadata': {},
            'similarity_score': 0.9,
            'rank': 1
        }])
        processor.generate_answer = AsyncMock(return_value={
            'answer': '测试答案',
            'question': question
        })
        
        result = await processor.process_question(question)
        
        assert result['success'] is True
        assert result['from_cache'] is False
    
    @pytest.mark.asyncio
    async def test_empty_question(self, processor):
        """测试空问题"""
        result = await processor.process_question("")
        
        # 应该正常处理，但可能返回无意义结果
        assert 'success' in result
    
    @pytest.mark.asyncio
    async def test_very_long_question(self, processor):
        """测试超长问题"""
        long_question = "这是一个非常长的问题。" * 100
        
        processor.retrieve_documents = AsyncMock(return_value=[])
        
        result = await processor.process_question(long_question)
        
        assert result['success'] is False  # 未找到文档
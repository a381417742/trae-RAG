"""
问答工作流集成测试
测试从问题输入到答案生成的完整流程
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

from src.core.rag_engine import RAGEngine
from src.core.qa_processor import QAProcessor


class TestQAWorkflowIntegration:
    """问答工作流集成测试"""
    
    @pytest.fixture
    async def rag_engine(self):
        """创建RAG引擎实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.document_processor.chromadb.HttpClient'), \
             patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'), \
             patch('src.utils.cache.init_cache_client'):
            
            engine = RAGEngine()
            engine.document_processor = Mock()
            engine.qa_processor = Mock()
            engine.initialized = True
            
            yield engine
    
    @pytest.fixture
    def sample_qa_data(self):
        """示例问答数据"""
        return {
            'question': '什么是人工智能？',
            'expected_context': [
                {
                    'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
                    'metadata': {'source': 'ai_intro.txt', 'page': 1},
                    'similarity_score': 0.95,
                    'rank': 1
                },
                {
                    'content': '机器学习是人工智能的一个重要子领域，专注于开发能够从数据中学习的算法。',
                    'metadata': {'source': 'ml_basics.txt', 'page': 2},
                    'similarity_score': 0.88,
                    'rank': 2
                }
            ],
            'expected_answer': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。它包括机器学习、深度学习等多个子领域。'
        }
    
    @pytest.mark.asyncio
    async def test_single_question_workflow(self, rag_engine, sample_qa_data):
        """测试单个问题处理完整工作流"""
        question = sample_qa_data['question']
        expected_context = sample_qa_data['expected_context']
        expected_answer = sample_qa_data['expected_answer']
        
        # 模拟问答处理结果
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': True,
            'question': question,
            'answer': expected_answer,
            'context_documents': expected_context,
            'generation_time': 2.5,
            'total_time': 3.2,
            'from_cache': False,
            'model': 'qwen3:30b',
            'token_count': {
                'prompt_tokens': 150,
                'completion_tokens': 80,
                'total_tokens': 230
            },
            'retrieval_stats': {
                'retrieved_count': 2,
                'similarity_threshold': 0.7,
                'avg_similarity': 0.915
            }
        })
        
        # 执行问答
        result = await rag_engine.answer_question(question)
        
        # 验证结果
        assert result['success'] is True
        assert result['question'] == question
        assert result['answer'] == expected_answer
        assert len(result['context_documents']) == 2
        assert result['from_cache'] is False
        assert result['retrieval_stats']['retrieved_count'] == 2
        
        # 验证调用
        rag_engine.qa_processor.process_question.assert_called_once_with(
            question=question,
            k=None,
            similarity_threshold=None,
            use_cache=True
        )
    
    @pytest.mark.asyncio
    async def test_cached_question_workflow(self, rag_engine, sample_qa_data):
        """测试缓存命中的问答工作流"""
        question = sample_qa_data['question']
        
        # 模拟缓存命中
        cached_result = {
            'success': True,
            'question': question,
            'answer': '这是缓存的答案',
            'context_documents': [],
            'from_cache': True,
            'total_time': 0.1
        }
        
        rag_engine.qa_processor.process_question = AsyncMock(return_value=cached_result)
        
        result = await rag_engine.answer_question(question, use_cache=True)
        
        assert result['success'] is True
        assert result['from_cache'] is True
        assert result['total_time'] < 1.0  # 缓存应该很快
    
    @pytest.mark.asyncio
    async def test_batch_questions_workflow(self, rag_engine):
        """测试批量问答工作流"""
        questions = [
            '什么是人工智能？',
            '机器学习有哪些类型？',
            '深度学习的应用领域有哪些？'
        ]
        
        # 模拟批量处理结果
        batch_results = []
        for i, question in enumerate(questions):
            batch_results.append({
                'success': True,
                'question': question,
                'answer': f'这是问题{i+1}的答案',
                'context_documents': [
                    {
                        'content': f'相关内容{i+1}',
                        'metadata': {'source': f'doc{i+1}.txt'},
                        'similarity_score': 0.9 - i*0.1,
                        'rank': 1
                    }
                ],
                'generation_time': 2.0,
                'total_time': 2.5,
                'from_cache': False
            })
        
        rag_engine.qa_processor.batch_process_questions = AsyncMock(return_value=batch_results)
        
        results = await rag_engine.batch_answer_questions(questions)
        
        # 验证结果
        assert len(results) == 3
        assert all(result['success'] for result in results)
        assert all(result['question'] in questions for result in results)
        
        # 验证调用
        rag_engine.qa_processor.batch_process_questions.assert_called_once_with(
            questions, k=None, similarity_threshold=None, use_cache=True
        )
    
    @pytest.mark.asyncio
    async def test_question_with_custom_parameters(self, rag_engine, sample_qa_data):
        """测试带自定义参数的问答"""
        question = sample_qa_data['question']
        
        # 模拟自定义参数处理
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': True,
            'question': question,
            'answer': '自定义参数的答案',
            'context_documents': sample_qa_data['expected_context'][:1],  # 只返回1个文档
            'retrieval_stats': {
                'retrieved_count': 1,
                'similarity_threshold': 0.8,
                'avg_similarity': 0.95
            }
        })
        
        result = await rag_engine.answer_question(
            question,
            k=1,  # 只检索1个文档
            similarity_threshold=0.8,  # 更高的相似度阈值
            use_cache=False
        )
        
        assert result['success'] is True
        assert len(result['context_documents']) == 1
        assert result['retrieval_stats']['similarity_threshold'] == 0.8
        
        # 验证参数传递
        rag_engine.qa_processor.process_question.assert_called_once_with(
            question=question,
            k=1,
            similarity_threshold=0.8,
            use_cache=False
        )
    
    @pytest.mark.asyncio
    async def test_question_no_relevant_documents(self, rag_engine):
        """测试未找到相关文档的问答"""
        question = "这是一个完全无关的问题"
        
        # 模拟未找到相关文档
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': False,
            'message': '未找到相关文档',
            'question': question,
            'answer': '抱歉，我在知识库中没有找到与您问题相关的信息。',
            'context_documents': [],
            'total_time': 1.0
        })
        
        result = await rag_engine.answer_question(question)
        
        assert result['success'] is False
        assert '未找到相关文档' in result['message']
        assert len(result['context_documents']) == 0
        assert '没有找到相关信息' in result['answer']


class TestQAProcessorIntegration:
    """问答处理器集成测试"""
    
    @pytest.fixture
    def qa_processor(self):
        """创建问答处理器实例"""
        with patch('src.core.qa_processor.SentenceTransformer') as mock_st, \
             patch('src.core.qa_processor.chromadb.HttpClient') as mock_chroma, \
             patch('src.core.qa_processor.get_cache_client') as mock_cache:
            
            # 模拟嵌入模型
            mock_embedding_model = Mock()
            mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            mock_st.return_value = mock_embedding_model
            
            # 模拟Chroma集合
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['人工智能是计算机科学的分支', '机器学习是AI的子领域']],
                'metadatas': [[{'source': 'ai.txt'}, {'source': 'ml.txt'}]],
                'distances': [[0.1, 0.2]]
            }
            
            mock_client = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client
            
            # 模拟缓存客户端
            mock_cache_client = AsyncMock()
            mock_cache_client.get.return_value = None
            mock_cache_client.setex.return_value = True
            mock_cache.return_value = mock_cache_client
            
            processor = QAProcessor()
            processor.collection = mock_collection
            processor.cache_client = mock_cache_client
            
            yield processor
    
    @pytest.mark.asyncio
    async def test_full_qa_processing_pipeline(self, qa_processor):
        """测试完整的问答处理管道"""
        question = "什么是人工智能？"
        
        # 模拟HTTP客户端响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "人工智能是计算机科学的一个分支，致力于创建智能系统。",
            "prompt_eval_count": 120,
            "eval_count": 60
        }
        qa_processor.http_client.post.return_value = mock_response
        
        # 执行完整问答流程
        result = await qa_processor.process_question(question)
        
        # 验证结果
        assert result['success'] is True
        assert result['question'] == question
        assert '人工智能是计算机科学的一个分支' in result['answer']
        assert len(result['context_documents']) == 2
        assert result['from_cache'] is False
        
        # 验证各个步骤
        qa_processor.embedding_model.encode.assert_called_once()
        qa_processor.collection.query.assert_called_once()
        qa_processor.http_client.post.assert_called_once()
        qa_processor.cache_client.setex.assert_called_once()  # 设置缓存
    
    @pytest.mark.asyncio
    async def test_retrieval_with_similarity_filtering(self, qa_processor):
        """测试带相似度过滤的检索"""
        question = "什么是深度学习？"
        
        # 模拟检索结果，包含不同相似度的文档
        qa_processor.collection.query.return_value = {
            'documents': [['高相似度文档', '中等相似度文档', '低相似度文档']],
            'metadatas': [[{'source': 'high.txt'}, {'source': 'medium.txt'}, {'source': 'low.txt'}]],
            'distances': [[0.1, 0.4, 0.7]]  # 对应相似度: 0.9, 0.6, 0.3
        }
        
        # 模拟LLM响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "基于高质量文档的回答",
            "prompt_eval_count": 80,
            "eval_count": 40
        }
        qa_processor.http_client.post.return_value = mock_response
        
        # 使用较高的相似度阈值
        result = await qa_processor.process_question(
            question, 
            similarity_threshold=0.7
        )
        
        # 验证只返回高相似度文档
        assert result['success'] is True
        assert len(result['context_documents']) == 2  # 只有前两个文档满足阈值
        assert result['context_documents'][0]['similarity_score'] == 0.9
        assert result['context_documents'][1]['similarity_score'] == 0.6
    
    @pytest.mark.asyncio
    async def test_caching_mechanism_integration(self, qa_processor):
        """测试缓存机制集成"""
        question = "测试缓存问题"
        
        # 第一次调用 - 缓存未命中
        qa_processor.cache_client.get.return_value = None
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "第一次生成的答案",
            "prompt_eval_count": 100,
            "eval_count": 50
        }
        qa_processor.http_client.post.return_value = mock_response
        
        result1 = await qa_processor.process_question(question)
        
        assert result1['success'] is True
        assert result1['from_cache'] is False
        assert result1['answer'] == "第一次生成的答案"
        
        # 验证缓存设置
        qa_processor.cache_client.setex.assert_called_once()
        
        # 第二次调用 - 缓存命中
        cached_data = {
            'success': True,
            'question': question,
            'answer': '缓存的答案',
            'from_cache': True,
            'total_time': 0.1
        }
        qa_processor.cache_client.get.return_value = json.dumps(cached_data)
        
        result2 = await qa_processor.process_question(question)
        
        assert result2['success'] is True
        assert result2['from_cache'] is True
        assert result2['answer'] == '缓存的答案'
        assert result2['total_time'] == 0.1
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, qa_processor):
        """测试批量处理集成"""
        questions = [
            "什么是AI？",
            "什么是ML？", 
            "什么是DL？"
        ]
        
        # 模拟单个问题处理
        async def mock_process_question(question, **kwargs):
            return {
                'success': True,
                'question': question,
                'answer': f'{question}的答案',
                'context_documents': [],
                'total_time': 1.0
            }
        
        qa_processor.process_question = mock_process_question
        
        results = await qa_processor.batch_process_questions(questions)
        
        assert len(results) == 3
        assert all(result['success'] for result in results)
        assert results[0]['answer'] == '什么是AI？的答案'
        assert results[1]['answer'] == '什么是ML？的答案'
        assert results[2]['answer'] == '什么是DL？的答案'


class TestQAWorkflowErrorScenarios:
    """问答工作流错误场景测试"""
    
    @pytest.fixture
    def rag_engine(self):
        """创建RAG引擎实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'):
            
            engine = RAGEngine()
            engine.qa_processor = Mock()
            engine.initialized = True
            yield engine
    
    @pytest.mark.asyncio
    async def test_llm_service_unavailable(self, rag_engine):
        """测试LLM服务不可用"""
        question = "测试问题"
        
        # 模拟LLM服务不可用
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': False,
            'message': 'LLM服务连接失败',
            'question': question,
            'answer': '抱歉，服务暂时不可用，请稍后重试。',
            'error': 'ConnectionError'
        })
        
        result = await rag_engine.answer_question(question)
        
        assert result['success'] is False
        assert 'LLM服务连接失败' in result['message']
        assert '服务暂时不可用' in result['answer']
    
    @pytest.mark.asyncio
    async def test_vector_database_error(self, rag_engine):
        """测试向量数据库错误"""
        question = "测试问题"
        
        # 模拟向量数据库错误
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': False,
            'message': '向量数据库查询失败',
            'question': question,
            'answer': '抱歉，检索服务出现问题。',
            'error': 'VectorDatabaseError'
        })
        
        result = await rag_engine.answer_question(question)
        
        assert result['success'] is False
        assert '向量数据库查询失败' in result['message']
    
    @pytest.mark.asyncio
    async def test_embedding_model_error(self, rag_engine):
        """测试嵌入模型错误"""
        question = "测试问题"
        
        # 模拟嵌入模型错误
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': False,
            'message': '嵌入模型处理失败',
            'question': question,
            'answer': '抱歉，问题理解出现错误。',
            'error': 'EmbeddingError'
        })
        
        result = await rag_engine.answer_question(question)
        
        assert result['success'] is False
        assert '嵌入模型处理失败' in result['message']
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, rag_engine):
        """测试超时处理"""
        question = "复杂的问题需要长时间处理"
        
        # 模拟处理超时
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': False,
            'message': '处理超时',
            'question': question,
            'answer': '抱歉，问题处理时间过长，请尝试简化问题。',
            'error': 'TimeoutError'
        })
        
        result = await rag_engine.answer_question(question)
        
        assert result['success'] is False
        assert '处理超时' in result['message']
    
    @pytest.mark.asyncio
    async def test_batch_processing_partial_failure(self, rag_engine):
        """测试批量处理部分失败"""
        questions = ["正常问题", "错误问题", "另一个正常问题"]
        
        # 模拟部分成功的批量处理
        batch_results = [
            {
                'success': True,
                'question': "正常问题",
                'answer': "正常答案"
            },
            {
                'success': False,
                'question': "错误问题",
                'message': "处理失败",
                'error': "ProcessingError"
            },
            {
                'success': True,
                'question': "另一个正常问题",
                'answer': "另一个正常答案"
            }
        ]
        
        rag_engine.qa_processor.batch_process_questions = AsyncMock(return_value=batch_results)
        
        results = await rag_engine.batch_answer_questions(questions)
        
        assert len(results) == 3
        assert results[0]['success'] is True
        assert results[1]['success'] is False
        assert results[2]['success'] is True
        assert 'ProcessingError' in results[1].get('error', '')


class TestQAWorkflowPerformance:
    """问答工作流性能测试"""
    
    @pytest.fixture
    def rag_engine(self):
        """创建RAG引擎实例"""
        with patch('src.core.document_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.SentenceTransformer'), \
             patch('src.core.qa_processor.chromadb.HttpClient'):
            
            engine = RAGEngine()
            engine.qa_processor = Mock()
            engine.initialized = True
            yield engine
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_response_time_performance(self, rag_engine):
        """测试响应时间性能"""
        question = "什么是人工智能？"
        
        # 模拟快速响应
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': True,
            'question': question,
            'answer': '人工智能是计算机科学分支',
            'context_documents': [],
            'generation_time': 1.5,
            'total_time': 2.0,
            'from_cache': False
        })
        
        import time
        start_time = time.time()
        result = await rag_engine.answer_question(question)
        end_time = time.time()
        
        # 验证响应时间
        assert result['success'] is True
        assert end_time - start_time < 3.0  # 应在3秒内完成
        assert result['total_time'] == 2.0
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_questions_performance(self, rag_engine):
        """测试并发问答性能"""
        questions = [
            "什么是AI？",
            "什么是ML？",
            "什么是DL？",
            "什么是NLP？",
            "什么是CV？"
        ]
        
        # 模拟并发处理
        async def mock_process_question(question, **kwargs):
            await asyncio.sleep(0.2)  # 模拟处理时间
            return {
                'success': True,
                'question': question,
                'answer': f'{question}的答案',
                'total_time': 0.2
            }
        
        rag_engine.qa_processor.process_question = mock_process_question
        
        # 测试并发处理
        import time
        start_time = time.time()
        
        tasks = [rag_engine.answer_question(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # 验证并发效果
        assert len(results) == 5
        assert all(result['success'] for result in results)
        # 并发处理应该比顺序处理快
        assert end_time - start_time < 1.0  # 应该远少于5 * 0.2 = 1.0秒
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_cache_performance_impact(self, rag_engine):
        """测试缓存对性能的影响"""
        question = "缓存性能测试问题"
        
        # 第一次调用 - 无缓存
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': True,
            'question': question,
            'answer': '第一次答案',
            'from_cache': False,
            'total_time': 2.0
        })
        
        import time
        start_time = time.time()
        result1 = await rag_engine.answer_question(question)
        first_call_time = time.time() - start_time
        
        # 第二次调用 - 缓存命中
        rag_engine.qa_processor.process_question = AsyncMock(return_value={
            'success': True,
            'question': question,
            'answer': '缓存答案',
            'from_cache': True,
            'total_time': 0.1
        })
        
        start_time = time.time()
        result2 = await rag_engine.answer_question(question)
        cached_call_time = time.time() - start_time
        
        # 验证缓存性能提升
        assert result1['from_cache'] is False
        assert result2['from_cache'] is True
        assert cached_call_time < first_call_time  # 缓存调用应该更快
        assert result2['total_time'] < result1['total_time']
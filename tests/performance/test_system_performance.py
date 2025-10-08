"""
系统性能测试
测试RAG系统在各种负载条件下的性能表现
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock

from src.core.rag_engine import RAGEngine


class TestSystemPerformance:
    """系统性能测试类"""
    
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
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_single_question_response_time(self, rag_engine):
        """测试单个问题响应时间"""
        question = "什么是人工智能？"
        
        # 模拟正常响应时间
        async def mock_answer_question(q, **kwargs):
            await asyncio.sleep(0.5)  # 模拟500ms处理时间
            return {
                'success': True,
                'question': q,
                'answer': '人工智能是计算机科学分支',
                'generation_time': 0.3,
                'total_time': 0.5
            }
        
        rag_engine.qa_processor.process_question = mock_answer_question
        
        # 测试响应时间
        start_time = time.time()
        result = await rag_engine.answer_question(question)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # 验证性能指标
        assert result['success'] is True
        assert response_time < 1.0  # 响应时间应小于1秒
        assert result['total_time'] == 0.5
        
        print(f"单问题响应时间: {response_time:.3f}秒")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_questions_throughput(self, rag_engine):
        """测试并发问答吞吐量"""
        questions = [f"问题{i}" for i in range(20)]
        
        # 模拟并发处理
        async def mock_answer_question(q, **kwargs):
            await asyncio.sleep(0.2)  # 模拟200ms处理时间
            return {
                'success': True,
                'question': q,
                'answer': f'{q}的答案',
                'total_time': 0.2
            }
        
        rag_engine.qa_processor.process_question = mock_answer_question
        
        # 测试并发处理
        start_time = time.time()
        
        # 创建并发任务
        tasks = [rag_engine.answer_question(q) for q in questions]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算吞吐量
        throughput = len(questions) / total_time
        
        # 验证结果
        assert len(results) == 20
        assert all(result['success'] for result in results)
        assert throughput > 10  # 每秒处理超过10个问题
        
        print(f"并发吞吐量: {throughput:.2f} 问题/秒")
        print(f"总处理时间: {total_time:.3f}秒")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_processing_performance(self, rag_engine):
        """测试批量处理性能"""
        batch_sizes = [1, 5, 10, 20]
        performance_results = {}
        
        for batch_size in batch_sizes:
            questions = [f"批量问题{i}" for i in range(batch_size)]
            
            # 模拟批量处理
            async def mock_batch_process(qs, **kwargs):
                await asyncio.sleep(0.1 * len(qs))  # 批量处理时间与数量成正比
                return [
                    {
                        'success': True,
                        'question': q,
                        'answer': f'{q}的答案',
                        'total_time': 0.1
                    }
                    for q in qs
                ]
            
            rag_engine.qa_processor.batch_process_questions = mock_batch_process
            
            # 测试批量处理
            start_time = time.time()
            results = await rag_engine.batch_answer_questions(questions)
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = batch_size / processing_time
            
            performance_results[batch_size] = {
                'processing_time': processing_time,
                'throughput': throughput
            }
            
            assert len(results) == batch_size
            assert all(result['success'] for result in results)
        
        # 分析性能趋势
        print("批量处理性能分析:")
        for size, metrics in performance_results.items():
            print(f"批量大小 {size}: {metrics['processing_time']:.3f}秒, "
                  f"吞吐量 {metrics['throughput']:.2f} 问题/秒")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_document_processing_performance(self, rag_engine):
        """测试文档处理性能"""
        file_sizes = [1, 5, 10, 20]  # 模拟不同大小的文档（KB）
        performance_results = {}
        
        for size_kb in file_sizes:
            file_path = f"/test/document_{size_kb}kb.txt"
            
            # 模拟文档处理时间与大小成正比
            async def mock_process_document(path):
                processing_time = size_kb * 0.1  # 每KB需要100ms
                await asyncio.sleep(processing_time)
                return {
                    'success': True,
                    'file_path': path,
                    'chunks_created': size_kb * 2,  # 每KB产生2个块
                    'stored_count': size_kb * 2,
                    'processing_time': processing_time
                }
            
            rag_engine.document_processor.process_file = mock_process_document
            
            # 测试文档处理
            start_time = time.time()
            result = await rag_engine.process_document(file_path)
            end_time = time.time()
            
            actual_time = end_time - start_time
            throughput = size_kb / actual_time  # KB/秒
            
            performance_results[size_kb] = {
                'processing_time': actual_time,
                'throughput': throughput,
                'chunks_created': result['chunks_created']
            }
            
            assert result['success'] is True
        
        # 分析文档处理性能
        print("文档处理性能分析:")
        for size, metrics in performance_results.items():
            print(f"文档大小 {size}KB: {metrics['processing_time']:.3f}秒, "
                  f"处理速度 {metrics['throughput']:.2f} KB/秒, "
                  f"生成块数 {metrics['chunks_created']}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage_under_load(self, rag_engine):
        """测试负载下的内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 模拟高负载处理
        questions = [f"内存测试问题{i}" for i in range(100)]
        
        async def mock_answer_question(q, **kwargs):
            # 模拟一些内存使用
            data = [i for i in range(1000)]  # 创建一些数据
            await asyncio.sleep(0.01)
            return {
                'success': True,
                'question': q,
                'answer': f'{q}的答案',
                'data': data  # 保持引用
            }
        
        rag_engine.qa_processor.process_question = mock_answer_question
        
        # 分批处理以监控内存使用
        memory_usage = []
        batch_size = 20
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            tasks = [rag_engine.answer_question(q) for q in batch]
            await asyncio.gather(*tasks)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(memory_usage)
        memory_increase = final_memory - initial_memory
        
        print(f"初始内存: {initial_memory:.2f} MB")
        print(f"最大内存: {max_memory:.2f} MB")
        print(f"最终内存: {final_memory:.2f} MB")
        print(f"内存增长: {memory_increase:.2f} MB")
        
        # 验证内存使用合理
        assert memory_increase < 100  # 内存增长应小于100MB
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_cache_performance_impact(self, rag_engine):
        """测试缓存对性能的影响"""
        question = "缓存性能测试"
        iterations = 10
        
        # 测试无缓存性能
        async def mock_no_cache(q, **kwargs):
            await asyncio.sleep(0.5)  # 模拟500ms处理时间
            return {
                'success': True,
                'question': q,
                'answer': '无缓存答案',
                'from_cache': False,
                'total_time': 0.5
            }
        
        rag_engine.qa_processor.process_question = mock_no_cache
        
        start_time = time.time()
        for _ in range(iterations):
            await rag_engine.answer_question(question, use_cache=False)
        no_cache_time = time.time() - start_time
        
        # 测试有缓存性能
        async def mock_with_cache(q, **kwargs):
            await asyncio.sleep(0.05)  # 模拟50ms缓存查询时间
            return {
                'success': True,
                'question': q,
                'answer': '缓存答案',
                'from_cache': True,
                'total_time': 0.05
            }
        
        rag_engine.qa_processor.process_question = mock_with_cache
        
        start_time = time.time()
        for _ in range(iterations):
            await rag_engine.answer_question(question, use_cache=True)
        cache_time = time.time() - start_time
        
        # 计算性能提升
        performance_improvement = (no_cache_time - cache_time) / no_cache_time * 100
        
        print(f"无缓存总时间: {no_cache_time:.3f}秒")
        print(f"有缓存总时间: {cache_time:.3f}秒")
        print(f"性能提升: {performance_improvement:.1f}%")
        
        assert cache_time < no_cache_time
        assert performance_improvement > 80  # 缓存应提升80%以上性能
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_system_stability_under_load(self, rag_engine):
        """测试系统在负载下的稳定性"""
        # 模拟长时间运行
        duration_seconds = 10
        questions_per_second = 5
        total_questions = duration_seconds * questions_per_second
        
        success_count = 0
        error_count = 0
        response_times = []
        
        async def mock_answer_with_occasional_errors(q, **kwargs):
            # 模拟偶发错误（5%概率）
            import random
            if random.random() < 0.05:
                raise Exception("模拟系统错误")
            
            processing_time = random.uniform(0.1, 0.8)  # 随机处理时间
            await asyncio.sleep(processing_time)
            return {
                'success': True,
                'question': q,
                'answer': f'{q}的答案',
                'total_time': processing_time
            }
        
        rag_engine.qa_processor.process_question = mock_answer_with_occasional_errors
        
        # 模拟持续负载
        start_time = time.time()
        
        async def process_question(i):
            nonlocal success_count, error_count
            question = f"稳定性测试问题{i}"
            
            try:
                question_start = time.time()
                result = await rag_engine.answer_question(question)
                question_end = time.time()
                
                if result['success']:
                    success_count += 1
                    response_times.append(question_end - question_start)
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        
        # 创建任务队列
        tasks = []
        for i in range(total_questions):
            tasks.append(process_question(i))
            
            # 控制请求速率
            if (i + 1) % questions_per_second == 0:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(1)  # 等待1秒
        
        # 处理剩余任务
        if tasks:
            await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算统计指标
        success_rate = success_count / total_questions * 100
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0
        
        print(f"稳定性测试结果:")
        print(f"总请求数: {total_questions}")
        print(f"成功数: {success_count}")
        print(f"失败数: {error_count}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均响应时间: {avg_response_time:.3f}秒")
        print(f"P95响应时间: {p95_response_time:.3f}秒")
        print(f"总运行时间: {total_time:.1f}秒")
        
        # 验证稳定性指标
        assert success_rate > 90  # 成功率应大于90%
        assert avg_response_time < 1.0  # 平均响应时间应小于1秒
        assert p95_response_time < 2.0  # P95响应时间应小于2秒


class TestPerformanceBenchmarks:
    """性能基准测试"""
    
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
    async def test_response_time_percentiles(self, rag_engine):
        """测试响应时间百分位数"""
        import random
        
        # 模拟不同响应时间
        async def mock_variable_response_time(q, **kwargs):
            # 模拟真实的响应时间分布
            response_time = random.lognormvariate(0, 0.5)  # 对数正态分布
            await asyncio.sleep(min(response_time, 2.0))  # 最大2秒
            return {
                'success': True,
                'question': q,
                'answer': f'{q}的答案',
                'total_time': response_time
            }
        
        rag_engine.qa_processor.process_question = mock_variable_response_time
        
        # 收集响应时间数据
        response_times = []
        num_requests = 100
        
        for i in range(num_requests):
            start_time = time.time()
            await rag_engine.answer_question(f"基准测试问题{i}")
            end_time = time.time()
            response_times.append(end_time - start_time)
        
        # 计算百分位数
        response_times.sort()
        p50 = response_times[49]  # 中位数
        p90 = response_times[89]  # 90百分位
        p95 = response_times[94]  # 95百分位
        p99 = response_times[98]  # 99百分位
        
        print(f"响应时间百分位数:")
        print(f"P50 (中位数): {p50:.3f}秒")
        print(f"P90: {p90:.3f}秒")
        print(f"P95: {p95:.3f}秒")
        print(f"P99: {p99:.3f}秒")
        
        # 验证性能基准
        assert p50 < 1.0  # 50%的请求应在1秒内完成
        assert p90 < 2.0  # 90%的请求应在2秒内完成
        assert p95 < 3.0  # 95%的请求应在3秒内完成
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_scalability_limits(self, rag_engine):
        """测试系统扩展性限制"""
        concurrent_levels = [1, 5, 10, 20, 50]
        scalability_results = {}
        
        for concurrency in concurrent_levels:
            # 模拟固定处理时间
            async def mock_fixed_processing(q, **kwargs):
                await asyncio.sleep(0.2)  # 200ms固定处理时间
                return {
                    'success': True,
                    'question': q,
                    'answer': f'{q}的答案'
                }
            
            rag_engine.qa_processor.process_question = mock_fixed_processing
            
            # 测试并发处理
            questions = [f"扩展性测试{i}" for i in range(concurrency)]
            
            start_time = time.time()
            tasks = [rag_engine.answer_question(q) for q in questions]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = concurrency / total_time
            
            scalability_results[concurrency] = {
                'total_time': total_time,
                'throughput': throughput,
                'success_rate': sum(1 for r in results if r['success']) / len(results) * 100
            }
        
        # 分析扩展性
        print("扩展性测试结果:")
        for concurrency, metrics in scalability_results.items():
            print(f"并发数 {concurrency}: "
                  f"总时间 {metrics['total_time']:.3f}秒, "
                  f"吞吐量 {metrics['throughput']:.2f} 请求/秒, "
                  f"成功率 {metrics['success_rate']:.1f}%")
        
        # 验证扩展性
        for concurrency, metrics in scalability_results.items():
            assert metrics['success_rate'] > 95  # 成功率应保持在95%以上
            if concurrency <= 20:
                assert metrics['throughput'] > concurrency * 0.8  # 吞吐量应接近理论值
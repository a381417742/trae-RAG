"""
问答处理器模块
负责基于RAG的问答处理：检索相关文档、生成回答
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime

import httpx
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

from ..config.settings import get_settings
from ..utils.logger import get_logger
from ..utils.cache import get_cache_client

logger = get_logger(__name__)
settings = get_settings()


class QAProcessor:
    """
    问答处理器类
    负责RAG问答的完整流程：问题理解、文档检索、答案生成
    """
    
    def __init__(self):
        """
        初始化问答处理器
        """
        # 初始化嵌入模型
        self.embedding_model = None
        self._init_embedding_model()
        
        # 初始化向量数据库连接
        self.chroma_client = None
        self.collection = None
        self._init_chroma_client()
        
        # 初始化HTTP客户端用于调用Ollama
        self.http_client = httpx.AsyncClient(timeout=settings.ollama_timeout)
        
        # 初始化缓存客户端
        self.cache_client = None
        self._init_cache_client()
        
        # 系统提示词模板
        self.system_prompt = """你是一个专业的知识库问答助手。请基于提供的相关文档内容来回答用户的问题。

回答要求：
1. 仅基于提供的文档内容进行回答，不要添加文档中没有的信息
2. 如果文档内容不足以回答问题，请明确说明
3. 回答要准确、简洁、有条理
4. 如果可能，请引用具体的文档片段
5. 使用中文回答

相关文档内容：
{context}

用户问题：{question}

请基于上述文档内容回答用户问题："""
        
        logger.info("问答处理器初始化完成")
    
    def _init_embedding_model(self) -> None:
        """
        初始化嵌入模型
        """
        try:
            self.embedding_model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device
            )
            logger.info(f"嵌入模型加载成功: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _init_chroma_client(self) -> None:
        """
        初始化Chroma向量数据库连接
        """
        try:
            self.chroma_client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=Settings(allow_reset=True)
            )
            
            # 获取集合
            self.collection = self.chroma_client.get_collection(
                name=settings.chroma_collection
            )
            logger.info(f"连接到向量数据库集合: {settings.chroma_collection}")
                
        except Exception as e:
            logger.error(f"Chroma数据库连接失败: {e}")
            raise
    
    def _init_cache_client(self) -> None:
        """
        初始化缓存客户端
        """
        try:
            self.cache_client = get_cache_client()
            logger.info("缓存客户端初始化成功")
        except Exception as e:
            logger.warning(f"缓存客户端初始化失败: {e}")
            self.cache_client = None
    
    def _generate_cache_key(self, question: str, k: int = None) -> str:
        """
        生成缓存键
        
        Args:
            question: 用户问题
            k: 检索数量
            
        Returns:
            str: 缓存键
        """
        k = k or settings.retrieval_k
        import hashlib
        content = f"{question}_{k}_{settings.similarity_threshold}"
        return f"qa:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def _get_cached_answer(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取答案
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Dict[str, Any]]: 缓存的答案，如果不存在返回None
        """
        if not self.cache_client:
            return None
        
        try:
            cached_data = await self.cache_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
        
        return None
    
    async def _set_cached_answer(self, cache_key: str, answer_data: Dict[str, Any]) -> None:
        """
        设置缓存答案
        
        Args:
            cache_key: 缓存键
            answer_data: 答案数据
        """
        if not self.cache_client:
            return
        
        try:
            await self.cache_client.setex(
                cache_key,
                settings.cache_ttl,
                json.dumps(answer_data, ensure_ascii=False)
            )
        except Exception as e:
            logger.warning(f"设置缓存失败: {e}")
    
    def generate_question_embedding(self, question: str) -> List[float]:
        """
        生成问题的嵌入向量
        
        Args:
            question: 用户问题
            
        Returns:
            List[float]: 问题的嵌入向量
        """
        try:
            embedding = self.embedding_model.encode([question])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"生成问题嵌入向量失败: {e}")
            raise
    
    async def retrieve_documents(
        self, 
        question: str, 
        k: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            question: 用户问题
            k: 检索数量
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Dict[str, Any]]: 相关文档列表
        """
        k = k or settings.retrieval_k
        similarity_threshold = similarity_threshold or settings.similarity_threshold
        
        try:
            # 生成问题嵌入向量
            loop = asyncio.get_event_loop()
            question_embedding = await loop.run_in_executor(
                None, self.generate_question_embedding, question
            )
            
            # 检索相关文档
            results = await loop.run_in_executor(
                None,
                self.collection.query,
                question_embedding,
                k
            )
            
            # 处理检索结果
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 计算相似度分数 (距离越小，相似度越高)
                    similarity_score = 1 - distance
                    
                    # 过滤低相似度文档
                    if similarity_score >= similarity_threshold:
                        documents.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i + 1
                        })
            
            logger.info(f"检索到{len(documents)}个相关文档 (阈值: {similarity_threshold})")
            return documents
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            raise
    
    async def generate_answer(
        self, 
        question: str, 
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        基于检索到的文档生成答案
        
        Args:
            question: 用户问题
            context_documents: 上下文文档列表
            
        Returns:
            Dict[str, Any]: 生成的答案和相关信息
        """
        try:
            # 构建上下文
            context_parts = []
            for i, doc in enumerate(context_documents, 1):
                context_parts.append(
                    f"文档片段{i} (相似度: {doc['similarity_score']:.3f}):\n"
                    f"{doc['content']}\n"
                )
            
            context = "\n".join(context_parts)
            
            # 构建完整提示词
            prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            # 调用Ollama生成答案
            start_time = time.time()
            
            payload = {
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens
                }
            }
            
            response = await self.http_client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            generation_time = time.time() - start_time
            
            answer_data = {
                "answer": result.get("response", "").strip(),
                "question": question,
                "context_documents": context_documents,
                "generation_time": generation_time,
                "model": settings.ollama_model,
                "timestamp": datetime.now().isoformat(),
                "token_count": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
            
            logger.info(f"答案生成完成，耗时: {generation_time:.2f}秒")
            return answer_data
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            raise
    
    async def process_question(
        self, 
        question: str,
        k: int = None,
        similarity_threshold: float = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        处理用户问题的完整流程
        
        Args:
            question: 用户问题
            k: 检索文档数量
            similarity_threshold: 相似度阈值
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        try:
            start_time = time.time()
            
            # 检查缓存
            cache_key = self._generate_cache_key(question, k)
            if use_cache:
                cached_answer = await self._get_cached_answer(cache_key)
                if cached_answer:
                    cached_answer["from_cache"] = True
                    cached_answer["total_time"] = time.time() - start_time
                    logger.info("从缓存返回答案")
                    return cached_answer
            
            # 1. 检索相关文档
            logger.info(f"开始处理问题: {question}")
            documents = await self.retrieve_documents(
                question, k, similarity_threshold
            )
            
            if not documents:
                result = {
                    "success": False,
                    "message": "未找到相关文档",
                    "answer": "抱歉，我在知识库中没有找到与您问题相关的信息。请尝试重新表述您的问题或联系管理员添加相关文档。",
                    "question": question,
                    "context_documents": [],
                    "total_time": time.time() - start_time,
                    "from_cache": False
                }
                return result
            
            # 2. 生成答案
            answer_data = await self.generate_answer(question, documents)
            
            # 3. 构建最终结果
            result = {
                "success": True,
                "message": "问答处理完成",
                "from_cache": False,
                "total_time": time.time() - start_time,
                "retrieval_stats": {
                    "retrieved_count": len(documents),
                    "similarity_threshold": similarity_threshold or settings.similarity_threshold,
                    "avg_similarity": sum(doc['similarity_score'] for doc in documents) / len(documents)
                },
                **answer_data
            }
            
            # 4. 缓存结果
            if use_cache:
                await self._set_cached_answer(cache_key, result)
            
            logger.info(f"问答处理完成，总耗时: {result['total_time']:.2f}秒")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"问答处理失败: {str(e)}",
                "answer": "抱歉，处理您的问题时出现了错误。请稍后重试或联系管理员。",
                "question": question,
                "error": str(e),
                "total_time": time.time() - start_time,
                "from_cache": False
            }
            logger.error(f"问答处理失败: {error_result}")
            return error_result
    
    async def batch_process_questions(
        self, 
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量处理问题
        
        Args:
            questions: 问题列表
            **kwargs: 传递给process_question的参数
            
        Returns:
            List[Dict[str, Any]]: 批量处理结果
        """
        try:
            logger.info(f"开始批量处理{len(questions)}个问题")
            
            # 并发处理问题
            tasks = [
                self.process_question(question, **kwargs)
                for question in questions
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "message": f"处理失败: {str(result)}",
                        "question": questions[i],
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            logger.info(f"批量处理完成: {len(processed_results)}个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取问答处理器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            collection_count = self.collection.count()
            
            return {
                "collection_name": settings.chroma_collection,
                "total_documents": collection_count,
                "embedding_model": settings.embedding_model,
                "llm_model": settings.ollama_model,
                "retrieval_k": settings.retrieval_k,
                "similarity_threshold": settings.similarity_threshold,
                "cache_enabled": self.cache_client is not None
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    async def close(self) -> None:
        """
        关闭资源连接
        """
        try:
            await self.http_client.aclose()
            if self.cache_client:
                await self.cache_client.close()
            logger.info("问答处理器资源已关闭")
        except Exception as e:
            logger.error(f"关闭资源失败: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
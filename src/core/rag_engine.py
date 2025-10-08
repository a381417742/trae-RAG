"""
RAG引擎主模块
整合文档处理和问答功能，提供统一的RAG服务接口
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import time

from .document_processor import DocumentProcessor
from .qa_processor import QAProcessor
from ..config.settings import get_settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics_collector, metrics_middleware
from ..utils.cache import init_cache_client

logger = get_logger(__name__)
settings = get_settings()


class RAGEngine:
    """
    RAG引擎主类
    提供文档处理和问答的统一接口
    """
    
    def __init__(self):
        """
        初始化RAG引擎
        """
        self.document_processor = None
        self.qa_processor = None
        self.initialized = False
        
        logger.info("RAG引擎初始化开始")
    
    async def initialize(self) -> None:
        """
        异步初始化RAG引擎组件
        """
        try:
            logger.info("开始初始化RAG引擎组件...")
            
            # 初始化缓存客户端
            try:
                await init_cache_client()
                logger.info("缓存客户端初始化成功")
            except Exception as e:
                logger.warning(f"缓存客户端初始化失败，将在无缓存模式下运行: {e}")
            
            # 初始化文档处理器
            logger.info("初始化文档处理器...")
            self.document_processor = DocumentProcessor()
            
            # 初始化问答处理器
            logger.info("初始化问答处理器...")
            self.qa_processor = QAProcessor()
            
            self.initialized = True
            logger.info("RAG引擎初始化完成")
            
        except Exception as e:
            logger.error(f"RAG引擎初始化失败: {e}")
            raise
    
    def _check_initialized(self) -> None:
        """
        检查引擎是否已初始化
        
        Raises:
            RuntimeError: 如果引擎未初始化
        """
        if not self.initialized:
            raise RuntimeError("RAG引擎未初始化，请先调用initialize()方法")
    
    @metrics_middleware
    async def process_document(
        self, 
        file_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理单个文档
        
        Args:
            file_path: 文档文件路径
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self._check_initialized()
        
        try:
            logger.info(f"开始处理文档: {file_path}")
            result = await self.document_processor.process_file(file_path)
            
            # 记录指标
            if result.get("success"):
                metrics_collector.record_document_processing(
                    "success", 
                    0,  # 耗时已在装饰器中记录
                    result.get("chunks_created", 0)
                )
                
                # 更新向量数据库文档数量
                stats = self.document_processor.get_collection_stats()
                if "total_documents" in stats:
                    metrics_collector.update_vector_db_documents(stats["total_documents"])
            else:
                metrics_collector.record_document_processing("error", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            metrics_collector.record_document_processing("error", 0)
            return {
                "success": False,
                "message": f"文档处理失败: {str(e)}",
                "file_path": file_path,
                "error": str(e)
            }
    
    @metrics_middleware
    async def process_directory(
        self, 
        directory_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理目录中的所有文档
        
        Args:
            directory_path: 目录路径
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        self._check_initialized()
        
        try:
            logger.info(f"开始处理目录: {directory_path}")
            result = await self.document_processor.process_directory(directory_path)
            
            # 记录指标
            metrics_collector.record_document_processing(
                "success", 
                0,  # 耗时已在装饰器中记录
                result.get("total_chunks", 0)
            )
            
            # 更新向量数据库文档数量
            stats = self.document_processor.get_collection_stats()
            if "total_documents" in stats:
                metrics_collector.update_vector_db_documents(stats["total_documents"])
            
            return result
            
        except Exception as e:
            logger.error(f"目录处理失败: {e}")
            metrics_collector.record_document_processing("error", 0)
            return {
                "success": False,
                "message": f"目录处理失败: {str(e)}",
                "directory_path": directory_path,
                "error": str(e)
            }
    
    @metrics_middleware
    async def answer_question(
        self,
        question: str,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            k: 检索文档数量
            similarity_threshold: 相似度阈值
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        self._check_initialized()
        
        try:
            logger.info(f"开始处理问题: {question}")
            result = await self.qa_processor.process_question(
                question=question,
                k=k,
                similarity_threshold=similarity_threshold,
                use_cache=use_cache
            )
            
            # 记录指标
            if result.get("success"):
                metrics_collector.record_qa_processing(
                    "success",
                    0,  # 耗时已在装饰器中记录
                    len(result.get("context_documents", []))
                )
                
                # 记录缓存操作
                if result.get("from_cache"):
                    metrics_collector.record_cache_operation("get", "success", True)
                else:
                    metrics_collector.record_cache_operation("get", "success", False)
            else:
                metrics_collector.record_qa_processing("error", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            metrics_collector.record_qa_processing("error", 0)
            return {
                "success": False,
                "message": f"问答处理失败: {str(e)}",
                "question": question,
                "error": str(e)
            }
    
    async def batch_answer_questions(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            **kwargs: 传递给answer_question的参数
            
        Returns:
            List[Dict[str, Any]]: 批量问答结果
        """
        self._check_initialized()
        
        try:
            logger.info(f"开始批量处理{len(questions)}个问题")
            
            # 使用QAProcessor的批量处理方法
            results = await self.qa_processor.batch_process_questions(
                questions, **kwargs
            )
            
            # 记录指标
            for result in results:
                if result.get("success"):
                    metrics_collector.record_qa_processing(
                        "success",
                        result.get("total_time", 0),
                        len(result.get("context_documents", []))
                    )
                else:
                    metrics_collector.record_qa_processing("error", 0)
            
            return results
            
        except Exception as e:
            logger.error(f"批量问答处理失败: {e}")
            return [{
                "success": False,
                "message": f"批量问答处理失败: {str(e)}",
                "question": q,
                "error": str(e)
            } for q in questions]
    
    async def delete_document(self, file_path: str) -> Dict[str, Any]:
        """
        删除文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        self._check_initialized()
        
        try:
            logger.info(f"开始删除文档: {file_path}")
            result = await self.document_processor.delete_document(file_path)
            
            # 记录指标
            if result.get("success"):
                metrics_collector.record_vector_db_operation("delete", "success")
                
                # 更新向量数据库文档数量
                stats = self.document_processor.get_collection_stats()
                if "total_documents" in stats:
                    metrics_collector.update_vector_db_documents(stats["total_documents"])
            else:
                metrics_collector.record_vector_db_operation("delete", "error")
            
            return result
            
        except Exception as e:
            logger.error(f"文档删除失败: {e}")
            metrics_collector.record_vector_db_operation("delete", "error")
            return {
                "success": False,
                "message": f"文档删除失败: {str(e)}",
                "file_path": file_path,
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            Dict[str, Any]: 系统统计信息
        """
        self._check_initialized()
        
        try:
            # 获取各组件统计信息
            doc_stats = self.document_processor.get_collection_stats()
            qa_stats = self.qa_processor.get_stats()
            metrics_summary = metrics_collector.get_metrics_summary()
            
            return {
                "system_info": {
                    "app_name": settings.app_name,
                    "app_version": settings.app_version,
                    "initialized": self.initialized
                },
                "document_processor": doc_stats,
                "qa_processor": qa_stats,
                "metrics": metrics_summary,
                "settings": {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "retrieval_k": settings.retrieval_k,
                    "similarity_threshold": settings.similarity_threshold,
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统统计信息失败: {e}")
            return {"error": str(e)}
    
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的文档格式
        
        Returns:
            List[str]: 支持的文档格式列表
        """
        self._check_initialized()
        return list(self.document_processor.supported_formats.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # 检查初始化状态
            health_status["components"]["initialization"] = {
                "status": "healthy" if self.initialized else "unhealthy",
                "initialized": self.initialized
            }
            
            if not self.initialized:
                health_status["status"] = "unhealthy"
                return health_status
            
            # 检查向量数据库连接
            try:
                stats = self.document_processor.get_collection_stats()
                health_status["components"]["vector_database"] = {
                    "status": "healthy",
                    "document_count": stats.get("total_documents", 0)
                }
            except Exception as e:
                health_status["components"]["vector_database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "unhealthy"
            
            # 检查LLM连接
            try:
                # 简单的测试问题
                test_result = await self.qa_processor.process_question(
                    "测试连接",
                    k=1,
                    use_cache=False
                )
                health_status["components"]["llm"] = {
                    "status": "healthy",
                    "model": settings.ollama_model
                }
            except Exception as e:
                health_status["components"]["llm"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "unhealthy"
            
            # 检查缓存连接
            try:
                if self.qa_processor.cache_client:
                    await self.qa_processor.cache_client.ping()
                    health_status["components"]["cache"] = {
                        "status": "healthy"
                    }
                else:
                    health_status["components"]["cache"] = {
                        "status": "disabled"
                    }
            except Exception as e:
                health_status["components"]["cache"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                # 缓存失败不影响整体健康状态
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
    
    async def close(self) -> None:
        """
        关闭RAG引擎，释放资源
        """
        try:
            if self.qa_processor:
                await self.qa_processor.close()
            
            logger.info("RAG引擎资源已释放")
            
        except Exception as e:
            logger.error(f"关闭RAG引擎失败: {e}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 创建全局RAG引擎实例
rag_engine = RAGEngine()
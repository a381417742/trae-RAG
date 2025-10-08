"""
文档处理器模块
负责文档的加载、预处理、分块和向量化存储
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import aiofiles
import logging

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentProcessor:
    """
    文档处理器类
    负责文档的完整处理流程：加载、预处理、分块、向量化和存储
    """
    
    def __init__(self):
        """
        初始化文档处理器
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 初始化嵌入模型
        self.embedding_model = None
        self._init_embedding_model()
        
        # 初始化向量数据库连接
        self.chroma_client = None
        self.collection = None
        self._init_chroma_client()
        
        # 支持的文件格式
        self.supported_formats = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': Docx2txtLoader
        }
        
        logger.info("文档处理器初始化完成")
    
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
            
            # 获取或创建集合
            try:
                self.collection = self.chroma_client.get_collection(
                    name=settings.chroma_collection
                )
                logger.info(f"连接到现有集合: {settings.chroma_collection}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=settings.chroma_collection,
                    metadata={"description": "RAG知识库文档向量集合"}
                )
                logger.info(f"创建新集合: {settings.chroma_collection}")
                
        except Exception as e:
            logger.error(f"Chroma数据库连接失败: {e}")
            raise
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件的MD5哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_file_processed(self, file_path: str) -> bool:
        """
        检查文件是否已经处理过
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 文件是否已处理
        """
        try:
            file_hash = self._get_file_hash(file_path)
            results = self.collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            logger.warning(f"检查文件处理状态失败: {e}")
            return False
    
    async def load_document(self, file_path: str) -> List[Document]:
        """
        异步加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 加载的文档列表
            
        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        # 检查文件是否已处理
        if self._is_file_processed(file_path):
            logger.info(f"文件已处理，跳过: {file_path}")
            return []
        
        try:
            loader_class = self.supported_formats[file_ext]
            loader = loader_class(file_path)
            
            # 在事件循环中运行同步加载操作
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(None, loader.load)
            
            # 添加文件元数据
            file_hash = self._get_file_hash(file_path)
            for doc in documents:
                doc.metadata.update({
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'file_hash': file_hash,
                    'file_size': os.path.getsize(file_path),
                    'processed_at': datetime.now().isoformat()
                })
            
            logger.info(f"文档加载成功: {file_path}, 共{len(documents)}个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"文档加载失败 {file_path}: {e}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档为小块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 分割后的文档块列表
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # 为每个块添加唯一ID和块索引
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"{chunk.metadata['file_hash']}_{i}"
                chunk.metadata['chunk_index'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            logger.info(f"文档分块完成: {len(documents)}个文档 -> {len(chunks)}个块")
            return chunks
            
        except Exception as e:
            logger.error(f"文档分块失败: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            raise
    
    async def store_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        异步存储文档到向量数据库
        
        Args:
            documents: 文档块列表
            
        Returns:
            Dict[str, Any]: 存储结果统计
        """
        if not documents:
            return {"stored_count": 0, "skipped_count": 0}
        
        try:
            # 准备数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [doc.metadata['chunk_id'] for doc in documents]
            
            # 生成嵌入向量
            logger.info(f"开始生成{len(texts)}个文档块的嵌入向量...")
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.generate_embeddings, texts
            )
            
            # 存储到向量数据库
            logger.info("开始存储到向量数据库...")
            await loop.run_in_executor(
                None,
                self.collection.add,
                embeddings,
                metadatas,
                texts,
                ids
            )
            
            result = {
                "stored_count": len(documents),
                "skipped_count": 0,
                "collection_name": settings.chroma_collection
            }
            
            logger.info(f"文档存储完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"文档存储失败: {e}")
            raise
    
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理单个文件的完整流程
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            
            # 1. 加载文档
            documents = await self.load_document(file_path)
            if not documents:
                return {
                    "success": True,
                    "message": "文件已处理，跳过",
                    "file_path": file_path,
                    "stored_count": 0
                }
            
            # 2. 分割文档
            chunks = self.split_documents(documents)
            
            # 3. 存储文档
            store_result = await self.store_documents(chunks)
            
            result = {
                "success": True,
                "message": "文件处理完成",
                "file_path": file_path,
                "original_docs": len(documents),
                "chunks_created": len(chunks),
                **store_result
            }
            
            logger.info(f"文件处理完成: {result}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"文件处理失败: {str(e)}",
                "file_path": file_path,
                "error": str(e)
            }
            logger.error(f"文件处理失败: {error_result}")
            return error_result
    
    async def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        处理目录中的所有支持文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            Dict[str, Any]: 处理结果统计
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        results = {
            "success_count": 0,
            "error_count": 0,
            "total_files": 0,
            "total_chunks": 0,
            "processed_files": [],
            "errors": []
        }
        
        try:
            # 获取所有支持的文件
            files_to_process = []
            for ext in self.supported_formats.keys():
                pattern = f"**/*{ext}"
                files_to_process.extend(Path(directory_path).glob(pattern))
            
            results["total_files"] = len(files_to_process)
            logger.info(f"找到{len(files_to_process)}个待处理文件")
            
            # 处理每个文件
            for file_path in files_to_process:
                file_result = await self.process_file(str(file_path))
                
                if file_result["success"]:
                    results["success_count"] += 1
                    results["total_chunks"] += file_result.get("chunks_created", 0)
                    results["processed_files"].append(file_result)
                else:
                    results["error_count"] += 1
                    results["errors"].append(file_result)
            
            logger.info(f"目录处理完成: {results}")
            return results
            
        except Exception as e:
            logger.error(f"目录处理失败: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取向量数据库集合统计信息
        
        Returns:
            Dict[str, Any]: 集合统计信息
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": settings.chroma_collection,
                "total_documents": count,
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {"error": str(e)}
    
    async def delete_document(self, file_path: str) -> Dict[str, Any]:
        """
        删除指定文件的所有文档块
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            file_hash = self._get_file_hash(file_path)
            
            # 查找相关文档
            results = self.collection.get(
                where={"file_hash": file_hash}
            )
            
            if not results['ids']:
                return {
                    "success": True,
                    "message": "未找到相关文档",
                    "deleted_count": 0
                }
            
            # 删除文档
            self.collection.delete(ids=results['ids'])
            
            result = {
                "success": True,
                "message": "文档删除成功",
                "deleted_count": len(results['ids']),
                "file_path": file_path
            }
            
            logger.info(f"文档删除完成: {result}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"文档删除失败: {str(e)}",
                "file_path": file_path,
                "error": str(e)
            }
            logger.error(f"文档删除失败: {error_result}")
            return error_result
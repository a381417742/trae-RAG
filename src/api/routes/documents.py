"""
文档处理API路由
提供文档上传、处理、删除等功能的RESTful接口
"""

import os
import uuid
from typing import List, Dict, Any
from pathlib import Path
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ...core.rag_engine import rag_engine
from ...config.settings import get_settings
from ...utils.logger import get_logger
from ...utils.metrics import metrics_collector
from ..models import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    BatchDocumentProcessResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    FileUploadResponse,
    FileListResponse,
    ErrorResponse
)

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["文档管理"])


def get_upload_dir() -> Path:
    """
    获取上传目录路径
    
    Returns:
        Path: 上传目录路径
    """
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def validate_file_type(filename: str) -> bool:
    """
    验证文件类型是否支持
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否支持该文件类型
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in settings.supported_formats


async def save_uploaded_file(file: UploadFile, upload_dir: Path) -> Dict[str, Any]:
    """
    保存上传的文件
    
    Args:
        file: 上传的文件
        upload_dir: 上传目录
        
    Returns:
        Dict[str, Any]: 文件信息
    """
    # 生成唯一文件名
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = upload_dir / unique_filename
    
    # 保存文件
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return {
        "original_filename": file.filename,
        "saved_filename": unique_filename,
        "file_path": str(file_path),
        "file_size": len(content),
        "content_type": file.content_type
    }


@router.post("/upload", response_model=FileUploadResponse, summary="上传文档文件")
async def upload_document(
    file: UploadFile = File(..., description="要上传的文档文件"),
    auto_process: bool = False
) -> FileUploadResponse:
    """
    上传文档文件
    
    Args:
        file: 上传的文件
        auto_process: 是否自动处理文档
        
    Returns:
        FileUploadResponse: 上传结果
    """
    try:
        # 验证文件类型
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式。支持的格式: {', '.join(settings.supported_formats)}"
            )
        
        # 验证文件大小
        content = await file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制 ({settings.max_file_size} 字节)"
            )
        
        # 重置文件指针
        await file.seek(0)
        
        # 保存文件
        upload_dir = get_upload_dir()
        file_info = await save_uploaded_file(file, upload_dir)
        
        # 记录指标
        metrics_collector.record_request("upload", "POST", "success", 0)
        
        response = FileUploadResponse(
            success=True,
            message="文件上传成功",
            filename=file_info["original_filename"],
            file_path=file_info["file_path"],
            file_size=file_info["file_size"],
            file_type=file_info["content_type"] or "unknown",
            upload_time=None  # 会自动设置为当前时间
        )
        
        # 如果启用自动处理，在后台处理文档
        if auto_process:
            # 这里可以添加后台任务处理逻辑
            logger.info(f"将在后台自动处理文档: {file_info['file_path']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        metrics_collector.record_request("upload", "POST", "error", 0)
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.post("/process", response_model=DocumentProcessResponse, summary="处理文档")
async def process_document(
    request: DocumentProcessRequest
) -> DocumentProcessResponse:
    """
    处理单个文档或目录中的文档
    
    Args:
        request: 文档处理请求
        
    Returns:
        DocumentProcessResponse: 处理结果
    """
    try:
        if request.file_path:
            # 处理单个文件
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail="文件不存在")
            
            result = await rag_engine.process_document(request.file_path)
            
        elif request.directory_path:
            # 处理目录
            if not os.path.exists(request.directory_path):
                raise HTTPException(status_code=404, detail="目录不存在")
            
            result = await rag_engine.process_directory(request.directory_path)
            
        else:
            raise HTTPException(status_code=400, detail="必须提供file_path或directory_path")
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("message", "处理失败"))
        
        # 记录指标
        metrics_collector.record_request("process", "POST", "success", 0)
        
        return DocumentProcessResponse(
            success=True,
            message=result.get("message", "处理完成"),
            file_path=result.get("file_path"),
            directory_path=result.get("directory_path"),
            original_docs=result.get("original_docs"),
            chunks_created=result.get("chunks_created"),
            stored_count=result.get("stored_count"),
            processing_time=result.get("processing_time")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        metrics_collector.record_request("process", "POST", "error", 0)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.delete("/delete", response_model=DocumentDeleteResponse, summary="删除文档")
async def delete_document(
    request: DocumentDeleteRequest
) -> DocumentDeleteResponse:
    """
    删除文档及其相关的向量数据
    
    Args:
        request: 文档删除请求
        
    Returns:
        DocumentDeleteResponse: 删除结果
    """
    try:
        result = await rag_engine.delete_document(request.file_path)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("message", "删除失败"))
        
        # 记录指标
        metrics_collector.record_request("delete", "DELETE", "success", 0)
        
        return DocumentDeleteResponse(
            success=True,
            message=result.get("message", "删除完成"),
            file_path=request.file_path,
            deleted_count=result.get("deleted_count", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档删除失败: {e}")
        metrics_collector.record_request("delete", "DELETE", "error", 0)
        raise HTTPException(status_code=500, detail=f"文档删除失败: {str(e)}")


@router.get("/list", response_model=FileListResponse, summary="获取文档列表")
async def list_documents(
    directory: str = "data/uploads",
    include_processed: bool = False
) -> FileListResponse:
    """
    获取文档列表
    
    Args:
        directory: 目录路径
        include_processed: 是否包含已处理状态
        
    Returns:
        FileListResponse: 文档列表
    """
    try:
        if not os.path.exists(directory):
            return FileListResponse(
                success=True,
                message="目录不存在",
                files=[],
                total_count=0,
                total_size=0
            )
        
        files = []
        total_size = 0
        
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file() and validate_file_type(file_path.name):
                file_stat = file_path.stat()
                file_info = {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_stat.st_size,
                    "modified_time": file_stat.st_mtime,
                    "file_type": file_path.suffix.lower()
                }
                
                # 如果需要包含处理状态，可以在这里查询
                if include_processed:
                    # 这里可以添加查询文档是否已处理的逻辑
                    file_info["processed"] = False
                
                files.append(file_info)
                total_size += file_stat.st_size
        
        # 按修改时间排序
        files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        # 记录指标
        metrics_collector.record_request("list", "GET", "success", 0)
        
        return FileListResponse(
            success=True,
            message=f"找到 {len(files)} 个文档",
            files=files,
            total_count=len(files),
            total_size=total_size
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        metrics_collector.record_request("list", "GET", "error", 0)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.get("/formats", summary="获取支持的文档格式")
async def get_supported_formats() -> Dict[str, List[str]]:
    """
    获取支持的文档格式列表
    
    Returns:
        Dict[str, List[str]]: 支持的格式列表
    """
    try:
        formats = rag_engine.get_supported_formats()
        
        return {
            "success": True,
            "message": "获取支持格式成功",
            "supported_formats": formats,
            "format_descriptions": {
                ".pdf": "PDF文档",
                ".txt": "纯文本文件",
                ".md": "Markdown文档",
                ".docx": "Word文档"
            }
        }
        
    except Exception as e:
        logger.error(f"获取支持格式失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取支持格式失败: {str(e)}")


@router.post("/batch-upload", summary="批量上传文档")
async def batch_upload_documents(
    files: List[UploadFile] = File(..., description="要上传的文档文件列表"),
    auto_process: bool = False
) -> Dict[str, Any]:
    """
    批量上传文档文件
    
    Args:
        files: 上传的文件列表
        auto_process: 是否自动处理文档
        
    Returns:
        Dict[str, Any]: 批量上传结果
    """
    try:
        if len(files) > 10:  # 限制批量上传数量
            raise HTTPException(status_code=400, detail="批量上传文件数量不能超过10个")
        
        upload_dir = get_upload_dir()
        results = []
        success_count = 0
        error_count = 0
        
        for file in files:
            try:
                # 验证文件类型
                if not validate_file_type(file.filename):
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": f"不支持的文件格式: {Path(file.filename).suffix}"
                    })
                    error_count += 1
                    continue
                
                # 验证文件大小
                content = await file.read()
                if len(content) > settings.max_file_size:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "文件大小超过限制"
                    })
                    error_count += 1
                    continue
                
                # 重置文件指针
                await file.seek(0)
                
                # 保存文件
                file_info = await save_uploaded_file(file, upload_dir)
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "file_path": file_info["file_path"],
                    "file_size": file_info["file_size"]
                })
                success_count += 1
                
            except Exception as e:
                logger.error(f"上传文件失败 {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
                error_count += 1
        
        # 记录指标
        metrics_collector.record_request("batch-upload", "POST", "success", 0)
        
        return {
            "success": True,
            "message": f"批量上传完成: 成功{success_count}个，失败{error_count}个",
            "total_files": len(files),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量上传失败: {e}")
        metrics_collector.record_request("batch-upload", "POST", "error", 0)
        raise HTTPException(status_code=500, detail=f"批量上传失败: {str(e)}")


@router.get("/stats", summary="获取文档统计信息")
async def get_document_stats() -> Dict[str, Any]:
    """
    获取文档统计信息
    
    Returns:
        Dict[str, Any]: 统计信息
    """
    try:
        stats = rag_engine.get_system_stats()
        
        return {
            "success": True,
            "message": "获取统计信息成功",
            "document_stats": stats.get("document_processor", {}),
            "system_info": stats.get("system_info", {})
        }
        
    except Exception as e:
        logger.error(f"获取文档统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档统计失败: {str(e)}")
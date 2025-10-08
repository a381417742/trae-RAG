"""
API数据模型
定义FastAPI接口的请求和响应数据结构
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class DocumentFormat(str, Enum):
    """支持的文档格式"""
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"


# ==================== 基础响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = Field(default=False, description="操作失败")
    error: Optional[str] = Field(None, description="错误详情")
    error_code: Optional[str] = Field(None, description="错误代码")


# ==================== 文档处理相关模型 ====================

class DocumentMetadata(BaseModel):
    """文档元数据"""
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小(字节)")
    file_hash: str = Field(..., description="文件哈希值")
    processed_at: datetime = Field(..., description="处理时间")
    chunk_count: Optional[int] = Field(None, description="分块数量")


class DocumentProcessRequest(BaseModel):
    """文档处理请求"""
    file_path: Optional[str] = Field(None, description="文件路径")
    directory_path: Optional[str] = Field(None, description="目录路径")
    overwrite: bool = Field(default=False, description="是否覆盖已存在的文档")
    
    @validator('file_path', 'directory_path')
    def validate_paths(cls, v, values):
        """验证路径参数"""
        if not values.get('file_path') and not values.get('directory_path'):
            raise ValueError("必须提供file_path或directory_path中的一个")
        return v


class DocumentProcessResponse(BaseResponse):
    """文档处理响应"""
    file_path: Optional[str] = Field(None, description="处理的文件路径")
    directory_path: Optional[str] = Field(None, description="处理的目录路径")
    original_docs: Optional[int] = Field(None, description="原始文档数量")
    chunks_created: Optional[int] = Field(None, description="创建的文档块数量")
    stored_count: Optional[int] = Field(None, description="存储的文档数量")
    processing_time: Optional[float] = Field(None, description="处理耗时(秒)")


class BatchDocumentProcessResponse(BaseResponse):
    """批量文档处理响应"""
    total_files: int = Field(..., description="总文件数")
    success_count: int = Field(..., description="成功处理数量")
    error_count: int = Field(..., description="处理失败数量")
    total_chunks: int = Field(..., description="总文档块数量")
    processed_files: List[DocumentProcessResponse] = Field(default=[], description="处理成功的文件")
    errors: List[Dict[str, Any]] = Field(default=[], description="处理失败的文件")


class DocumentDeleteRequest(BaseModel):
    """文档删除请求"""
    file_path: str = Field(..., description="要删除的文件路径")


class DocumentDeleteResponse(BaseResponse):
    """文档删除响应"""
    file_path: str = Field(..., description="删除的文件路径")
    deleted_count: int = Field(..., description="删除的文档块数量")


# ==================== 问答相关模型 ====================

class QuestionRequest(BaseModel):
    """问答请求"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    k: Optional[int] = Field(default=None, ge=1, le=20, description="检索文档数量")
    similarity_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="相似度阈值"
    )
    use_cache: bool = Field(default=True, description="是否使用缓存")
    
    @validator('question')
    def validate_question(cls, v):
        """验证问题内容"""
        if not v.strip():
            raise ValueError("问题不能为空")
        return v.strip()


class ContextDocument(BaseModel):
    """上下文文档"""
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(..., description="文档元数据")
    similarity_score: float = Field(..., description="相似度分数")
    rank: int = Field(..., description="检索排名")


class TokenCount(BaseModel):
    """Token统计"""
    prompt_tokens: int = Field(default=0, description="提示词Token数")
    completion_tokens: int = Field(default=0, description="生成Token数")
    total_tokens: int = Field(default=0, description="总Token数")


class RetrievalStats(BaseModel):
    """检索统计信息"""
    retrieved_count: int = Field(..., description="检索到的文档数量")
    similarity_threshold: float = Field(..., description="使用的相似度阈值")
    avg_similarity: float = Field(..., description="平均相似度")


class QuestionResponse(BaseResponse):
    """问答响应"""
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="生成的答案")
    context_documents: List[ContextDocument] = Field(default=[], description="上下文文档")
    generation_time: Optional[float] = Field(None, description="生成耗时(秒)")
    total_time: Optional[float] = Field(None, description="总耗时(秒)")
    model: Optional[str] = Field(None, description="使用的模型")
    from_cache: bool = Field(default=False, description="是否来自缓存")
    token_count: Optional[TokenCount] = Field(None, description="Token统计")
    retrieval_stats: Optional[RetrievalStats] = Field(None, description="检索统计")


class BatchQuestionRequest(BaseModel):
    """批量问答请求"""
    questions: List[str] = Field(..., min_items=1, max_items=10, description="问题列表")
    k: Optional[int] = Field(default=None, ge=1, le=20, description="检索文档数量")
    similarity_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="相似度阈值"
    )
    use_cache: bool = Field(default=True, description="是否使用缓存")
    
    @validator('questions')
    def validate_questions(cls, v):
        """验证问题列表"""
        if not v:
            raise ValueError("问题列表不能为空")
        
        cleaned_questions = []
        for q in v:
            if not q.strip():
                raise ValueError("问题不能为空")
            cleaned_questions.append(q.strip())
        
        return cleaned_questions


class BatchQuestionResponse(BaseResponse):
    """批量问答响应"""
    results: List[QuestionResponse] = Field(..., description="批量问答结果")
    total_questions: int = Field(..., description="总问题数")
    success_count: int = Field(..., description="成功处理数量")
    error_count: int = Field(..., description="处理失败数量")
    total_time: float = Field(..., description="总处理时间")


# ==================== 系统状态相关模型 ====================

class SystemStats(BaseModel):
    """系统统计信息"""
    app_name: str = Field(..., description="应用名称")
    app_version: str = Field(..., description="应用版本")
    initialized: bool = Field(..., description="是否已初始化")
    total_documents: int = Field(..., description="文档总数")
    embedding_model: str = Field(..., description="嵌入模型")
    llm_model: str = Field(..., description="语言模型")
    cache_enabled: bool = Field(..., description="缓存是否启用")


class ComponentHealth(BaseModel):
    """组件健康状态"""
    status: str = Field(..., description="健康状态")
    error: Optional[str] = Field(None, description="错误信息")
    document_count: Optional[int] = Field(None, description="文档数量")
    model: Optional[str] = Field(None, description="模型名称")


class HealthCheckResponse(BaseResponse):
    """健康检查响应"""
    status: str = Field(..., description="整体健康状态")
    components: Dict[str, ComponentHealth] = Field(..., description="各组件健康状态")


class MetricsSummary(BaseModel):
    """指标摘要"""
    cache_hit_rate: float = Field(..., description="缓存命中率")
    cache_hits: int = Field(..., description="缓存命中次数")
    cache_misses: int = Field(..., description="缓存未命中次数")
    system_cpu_percent: float = Field(..., description="CPU使用率")
    system_memory_percent: float = Field(..., description="内存使用率")
    system_disk_percent: float = Field(..., description="磁盘使用率")


class SystemStatsResponse(BaseResponse):
    """系统统计响应"""
    system_info: SystemStats = Field(..., description="系统信息")
    metrics: MetricsSummary = Field(..., description="指标摘要")
    supported_formats: List[str] = Field(..., description="支持的文档格式")


# ==================== 文件上传相关模型 ====================

class FileUploadResponse(BaseResponse):
    """文件上传响应"""
    filename: str = Field(..., description="上传的文件名")
    file_path: str = Field(..., description="文件保存路径")
    file_size: int = Field(..., description="文件大小")
    file_type: str = Field(..., description="文件类型")
    upload_time: datetime = Field(..., description="上传时间")


class FileListResponse(BaseResponse):
    """文件列表响应"""
    files: List[Dict[str, Any]] = Field(..., description="文件列表")
    total_count: int = Field(..., description="文件总数")
    total_size: int = Field(..., description="总文件大小")


# ==================== 配置相关模型 ====================

class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    chunk_size: Optional[int] = Field(None, ge=100, le=5000, description="文档分块大小")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=1000, description="分块重叠大小")
    retrieval_k: Optional[int] = Field(None, ge=1, le=50, description="检索文档数量")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    max_tokens: Optional[int] = Field(None, ge=100, le=8000, description="最大生成Token数")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="生成温度")


class ConfigResponse(BaseResponse):
    """配置响应"""
    config: Dict[str, Any] = Field(..., description="当前配置")


# ==================== WebSocket相关模型 ====================

class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")


class ProcessingProgress(BaseModel):
    """处理进度"""
    task_id: str = Field(..., description="任务ID")
    status: ProcessingStatus = Field(..., description="处理状态")
    progress: float = Field(..., ge=0.0, le=1.0, description="进度百分比")
    current_step: str = Field(..., description="当前步骤")
    total_steps: int = Field(..., description="总步骤数")
    completed_steps: int = Field(..., description="已完成步骤数")
    message: Optional[str] = Field(None, description="进度消息")
    error: Optional[str] = Field(None, description="错误信息")
"""
问答API路由
提供基于RAG的问答功能的RESTful接口
"""

import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json

from ...core.rag_engine import rag_engine
from ...config.settings import get_settings
from ...utils.logger import get_logger
from ...utils.metrics import metrics_collector
from ..models import (
    QuestionRequest,
    QuestionResponse,
    BatchQuestionRequest,
    BatchQuestionResponse,
    ContextDocument,
    TokenCount,
    RetrievalStats,
    ErrorResponse
)

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/qa", tags=["问答服务"])


def format_question_response(result: Dict[str, Any]) -> QuestionResponse:
    """
    格式化问答响应
    
    Args:
        result: RAG引擎返回的结果
        
    Returns:
        QuestionResponse: 格式化后的响应
    """
    # 格式化上下文文档
    context_documents = []
    for doc in result.get("context_documents", []):
        context_documents.append(ContextDocument(
            content=doc.get("content", ""),
            metadata=doc.get("metadata", {}),
            similarity_score=doc.get("similarity_score", 0.0),
            rank=doc.get("rank", 0)
        ))
    
    # 格式化Token统计
    token_count = None
    if "token_count" in result:
        token_data = result["token_count"]
        token_count = TokenCount(
            prompt_tokens=token_data.get("prompt_tokens", 0),
            completion_tokens=token_data.get("completion_tokens", 0),
            total_tokens=token_data.get("total_tokens", 0)
        )
    
    # 格式化检索统计
    retrieval_stats = None
    if "retrieval_stats" in result:
        stats_data = result["retrieval_stats"]
        retrieval_stats = RetrievalStats(
            retrieved_count=stats_data.get("retrieved_count", 0),
            similarity_threshold=stats_data.get("similarity_threshold", 0.0),
            avg_similarity=stats_data.get("avg_similarity", 0.0)
        )
    
    return QuestionResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        question=result.get("question", ""),
        answer=result.get("answer", ""),
        context_documents=context_documents,
        generation_time=result.get("generation_time"),
        total_time=result.get("total_time"),
        model=result.get("model"),
        from_cache=result.get("from_cache", False),
        token_count=token_count,
        retrieval_stats=retrieval_stats
    )


@router.post("/ask", response_model=QuestionResponse, summary="问答接口")
async def ask_question(
    request: QuestionRequest
) -> QuestionResponse:
    """
    基于RAG的问答接口
    
    Args:
        request: 问答请求
        
    Returns:
        QuestionResponse: 问答结果
    """
    try:
        start_time = time.time()
        
        # 调用RAG引擎处理问题
        result = await rag_engine.answer_question(
            question=request.question,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            use_cache=request.use_cache
        )
        
        # 记录指标
        processing_time = time.time() - start_time
        if result.get("success"):
            metrics_collector.record_request("ask", "POST", "success", processing_time)
        else:
            metrics_collector.record_request("ask", "POST", "error", processing_time)
        
        # 格式化响应
        response = format_question_response(result)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        metrics_collector.record_request("ask", "POST", "error", time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@router.post("/batch-ask", response_model=BatchQuestionResponse, summary="批量问答接口")
async def batch_ask_questions(
    request: BatchQuestionRequest
) -> BatchQuestionResponse:
    """
    批量问答接口
    
    Args:
        request: 批量问答请求
        
    Returns:
        BatchQuestionResponse: 批量问答结果
    """
    try:
        start_time = time.time()
        
        # 调用RAG引擎批量处理问题
        results = await rag_engine.batch_answer_questions(
            questions=request.questions,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            use_cache=request.use_cache
        )
        
        # 格式化响应
        formatted_results = []
        success_count = 0
        error_count = 0
        
        for result in results:
            formatted_result = format_question_response(result)
            formatted_results.append(formatted_result)
            
            if formatted_result.success:
                success_count += 1
            else:
                error_count += 1
        
        total_time = time.time() - start_time
        
        # 记录指标
        metrics_collector.record_request("batch-ask", "POST", "success", total_time)
        
        return BatchQuestionResponse(
            success=True,
            message=f"批量问答完成: 成功{success_count}个，失败{error_count}个",
            results=formatted_results,
            total_questions=len(request.questions),
            success_count=success_count,
            error_count=error_count,
            total_time=total_time
        )
        
    except Exception as e:
        logger.error(f"批量问答处理失败: {e}")
        metrics_collector.record_request("batch-ask", "POST", "error", time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"批量问答处理失败: {str(e)}")


@router.post("/ask-stream", summary="流式问答接口")
async def ask_question_stream(
    request: QuestionRequest
):
    """
    流式问答接口（实验性功能）
    
    Args:
        request: 问答请求
        
    Returns:
        StreamingResponse: 流式响应
    """
    async def generate_stream():
        """生成流式响应"""
        try:
            # 发送开始信号
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理问题'}, ensure_ascii=False)}\n\n"
            
            # 发送检索阶段信号
            yield f"data: {json.dumps({'type': 'retrieval', 'message': '正在检索相关文档'}, ensure_ascii=False)}\n\n"
            
            # 调用RAG引擎处理问题
            result = await rag_engine.answer_question(
                question=request.question,
                k=request.k,
                similarity_threshold=request.similarity_threshold,
                use_cache=request.use_cache
            )
            
            if result.get("success"):
                # 发送生成阶段信号
                yield f"data: {json.dumps({'type': 'generation', 'message': '正在生成答案'}, ensure_ascii=False)}\n\n"
                
                # 发送最终结果
                response_data = {
                    'type': 'answer',
                    'question': result.get("question", ""),
                    'answer': result.get("answer", ""),
                    'context_count': len(result.get("context_documents", [])),
                    'from_cache': result.get("from_cache", False),
                    'generation_time': result.get("generation_time"),
                    'total_time': result.get("total_time")
                }
                yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
            else:
                # 发送错误信息
                error_data = {
                    'type': 'error',
                    'message': result.get("message", "处理失败"),
                    'error': result.get("error", "")
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            
            # 发送结束信号
            yield f"data: {json.dumps({'type': 'end', 'message': '处理完成'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"流式问答处理失败: {e}")
            error_data = {
                'type': 'error',
                'message': f"处理失败: {str(e)}",
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.get("/history", summary="获取问答历史")
async def get_qa_history(
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """
    获取问答历史记录（如果实现了历史记录功能）
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
        
    Returns:
        Dict[str, Any]: 历史记录
    """
    try:
        # 这里可以实现从数据库或缓存中获取历史记录的逻辑
        # 目前返回空列表作为占位符
        
        return {
            "success": True,
            "message": "获取历史记录成功",
            "history": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"获取问答历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问答历史失败: {str(e)}")


@router.get("/suggestions", summary="获取问题建议")
async def get_question_suggestions(
    query: str = "",
    limit: int = 5
) -> Dict[str, Any]:
    """
    根据输入获取问题建议
    
    Args:
        query: 查询关键词
        limit: 返回数量限制
        
    Returns:
        Dict[str, Any]: 问题建议
    """
    try:
        # 这里可以实现基于文档内容的问题建议逻辑
        # 目前返回一些示例建议
        
        sample_suggestions = [
            "这个文档的主要内容是什么？",
            "有哪些重要的概念或术语？",
            "文档中提到了哪些关键信息？",
            "能否总结一下主要观点？",
            "有什么需要注意的地方？"
        ]
        
        # 如果有查询关键词，可以基于关键词过滤建议
        if query:
            suggestions = [s for s in sample_suggestions if query.lower() in s.lower()]
        else:
            suggestions = sample_suggestions
        
        return {
            "success": True,
            "message": "获取问题建议成功",
            "suggestions": suggestions[:limit],
            "query": query,
            "total_count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"获取问题建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问题建议失败: {str(e)}")


@router.get("/stats", summary="获取问答统计信息")
async def get_qa_stats() -> Dict[str, Any]:
    """
    获取问答统计信息
    
    Returns:
        Dict[str, Any]: 统计信息
    """
    try:
        stats = rag_engine.get_system_stats()
        
        return {
            "success": True,
            "message": "获取统计信息成功",
            "qa_stats": stats.get("qa_processor", {}),
            "metrics": stats.get("metrics", {}),
            "settings": {
                "retrieval_k": settings.retrieval_k,
                "similarity_threshold": settings.similarity_threshold,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature
            }
        }
        
    except Exception as e:
        logger.error(f"获取问答统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取问答统计失败: {str(e)}")


@router.post("/feedback", summary="提交问答反馈")
async def submit_feedback(
    question: str,
    answer: str,
    rating: int,
    feedback: str = ""
) -> Dict[str, Any]:
    """
    提交问答反馈（用于改进系统）
    
    Args:
        question: 原问题
        answer: 系统回答
        rating: 评分 (1-5)
        feedback: 反馈内容
        
    Returns:
        Dict[str, Any]: 提交结果
    """
    try:
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="评分必须在1-5之间")
        
        # 这里可以实现反馈存储逻辑
        # 例如存储到数据库或日志文件
        
        feedback_data = {
            "question": question,
            "answer": answer,
            "rating": rating,
            "feedback": feedback,
            "timestamp": time.time()
        }
        
        logger.info(f"收到问答反馈: {feedback_data}")
        
        return {
            "success": True,
            "message": "反馈提交成功，感谢您的反馈！",
            "feedback_id": f"fb_{int(time.time())}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")
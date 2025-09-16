"""
微信聊天记录向量数据库API服务
提供问题查询接口，返回相关的聊天记录内容
"""

import os
import sys
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import socket

# 设置API密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-bae62c151c524da4b4ee5f04e4e19a3f"
import dashscope
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

from langchain_chroma import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

# 请求和响应模型
class QueryRequest(BaseModel):
    question: str
    max_results: int = 5
    similarity_threshold: float = 0.0

class ChatRecord(BaseModel):
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

class QueryResponse(BaseModel):
    question: str
    related_records: List[ChatRecord]
    total_found: int
    status: str
    message: str

# 初始化FastAPI应用
app = FastAPI(
    title="微信聊天记录向量数据库API",
    description="基于向量数据库的聊天记录检索服务",
    version="1.0.0"
)

# 添加CORS中间件，允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_local_ip():
    """获取本机局域网IP地址"""
    try:
        # 创建临时socket连接来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# 全局变量存储向量数据库
vectorstore = None

def load_vectorstore():
    """加载向量数据库"""
    global vectorstore

    try:
        db_path = "./chroma_wechat_db"

        if not os.path.exists(db_path):
            raise Exception("向量数据库不存在，请先运行 test_csv_final.py 创建数据库")

        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

        # 测试数据库是否可用
        test_results = vectorstore.similarity_search("测试", k=1)
        print(f"✅ 成功加载向量数据库，测试查询返回 {len(test_results)} 条结果")

        return True

    except Exception as e:
        print(f"❌ 加载向量数据库失败: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """应用启动时加载向量数据库"""
    print("🚀 正在启动微信聊天记录API服务...")

    # 显示网络信息
    local_ip = get_local_ip()
    print(f"📍 本机IP地址: {local_ip}")
    print(f"🌐 局域网访问地址: http://{local_ip}:8000")
    print(f"🏠 本地访问地址: http://localhost:8000")
    print(f"📖 API文档地址: http://{local_ip}:8000/docs")

    success = load_vectorstore()
    if not success:
        print("❌ 向量数据库加载失败，API服务可能无法正常工作")
    else:
        print("✅ API服务启动成功")
        print("🔗 其他设备可通过以下地址访问:")
        print(f"   - http://{local_ip}:8000")
        print("⚠️  请确保防火墙允许8000端口访问")

@app.get("/")
async def root():
    """API根路径，返回服务信息"""
    return {
        "service": "微信聊天记录向量数据库API",
        "version": "1.0.0",
        "status": "运行中" if vectorstore is not None else "数据库未加载",
        "endpoints": {
            "查询": "POST /query",
            "健康检查": "GET /health",
            "统计信息": "GET /stats"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="向量数据库未加载")

    try:
        # 简单测试查询
        test_results = vectorstore.similarity_search("测试", k=1)
        return {
            "status": "健康",
            "database": "已连接",
            "test_query": f"成功返回 {len(test_results)} 条结果"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"数据库连接异常: {str(e)}")

@app.get("/stats")
async def get_stats():
    """获取数据库统计信息"""
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="向量数据库未加载")

    try:
        # 获取向量数据库的实际统计信息
        # Chroma数据库的集合信息
        collection = vectorstore._collection
        total_count = collection.count()

        # 获取样本来分析发送者和消息类型
        sample_results = vectorstore.similarity_search("", k=min(200, total_count))

        # 统计发送者和消息类型
        senders = set()
        msg_types = set()
        time_range = {"earliest": None, "latest": None}

        for doc in sample_results:
            if 'sender' in doc.metadata:
                senders.add(doc.metadata['sender'])
            if 'msg_type' in doc.metadata:
                msg_types.add(doc.metadata['msg_type'])
            if 'chat_time' in doc.metadata and doc.metadata['chat_time']:
                chat_time = doc.metadata['chat_time']
                if time_range["earliest"] is None or chat_time < time_range["earliest"]:
                    time_range["earliest"] = chat_time
                if time_range["latest"] is None or chat_time > time_range["latest"]:
                    time_range["latest"] = chat_time

        return {
            "total_records": total_count,
            "sample_analyzed": len(sample_results),
            "unique_senders": list(senders),
            "message_types": list(msg_types),
            "time_range": time_range,
            "database_path": "./chroma_wechat_db"
        }

    except Exception as e:
        # 如果获取总数失败，回退到原方法
        try:
            sample_results = vectorstore.similarity_search("", k=100)
            senders = set()
            msg_types = set()

            for doc in sample_results:
                if 'sender' in doc.metadata:
                    senders.add(doc.metadata['sender'])
                if 'msg_type' in doc.metadata:
                    msg_types.add(doc.metadata['msg_type'])

            return {
                "total_records": f"约 {len(sample_results)}+ (无法获取精确总数)",
                "sample_analyzed": len(sample_results),
                "unique_senders": list(senders),
                "message_types": list(msg_types),
                "note": f"统计获取部分失败: {str(e)}",
                "database_path": "./chroma_wechat_db"
            }
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e2)}")

@app.post("/query", response_model=QueryResponse)
async def query_records(request: QueryRequest):
    """查询相关聊天记录"""

    if vectorstore is None:
        raise HTTPException(status_code=503, detail="向量数据库未加载")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        # 在向量数据库中搜索相关内容
        results = vectorstore.similarity_search_with_score(
            request.question,
            k=request.max_results
        )

        # 过滤相似度阈值
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= request.similarity_threshold
        ]

        # 格式化结果
        chat_records = []
        for doc, score in filtered_results:
            chat_record = ChatRecord(
                content=doc.page_content,
                metadata=doc.metadata,
                similarity_score=float(score)
            )
            chat_records.append(chat_record)

        return QueryResponse(
            question=request.question,
            related_records=chat_records,
            total_found=len(chat_records),
            status="success",
            message=f"找到 {len(chat_records)} 条相关记录"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.post("/query_simple")
async def query_simple(question: str, max_results: int = 5):
    """简化的查询接口，直接接受字符串参数"""

    if vectorstore is None:
        return {"error": "向量数据库未加载"}

    if not question.strip():
        return {"error": "问题不能为空"}

    try:
        # 搜索相关内容
        results = vectorstore.similarity_search_with_score(question, k=max_results)

        # 简化的返回格式
        records = []
        for doc, score in results:
            records.append({
                "content": doc.page_content,
                "sender": doc.metadata.get('sender', '未知'),
                "time": doc.metadata.get('chat_time', '未知时间'),
                "similarity": float(score)
            })

        return {
            "question": question,
            "records": records,
            "count": len(records)
        }

    except Exception as e:
        return {"error": f"查询失败: {str(e)}"}

if __name__ == "__main__":
    local_ip = get_local_ip()

    print("🚀 启动微信聊天记录API服务器...")
    print("=" * 60)
    print(f"🏠 本地访问: http://localhost:8000")
    print(f"🌐 局域网访问: http://{local_ip}:8000")
    print(f"📖 API文档: http://{local_ip}:8000/docs")
    print("=" * 60)
    print("📋 其他设备访问步骤:")
    print(f"1. 确保设备在同一局域网")
    print(f"2. 访问地址: http://{local_ip}:8000")
    print(f"3. 如连接失败，请检查防火墙设置")
    print("=" * 60)

    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
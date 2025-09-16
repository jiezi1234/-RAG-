# 微信聊天记录RAG系统

基于向量数据库的微信聊天记录检索问答系统，支持语义搜索和智能问答。

## 🚀 项目简介

本项目是一个完整的RAG（Retrieval-Augmented Generation）系统，可以将微信聊天记录转换为向量数据库，并提供API服务进行智能查询。系统支持语义相似度搜索，能够根据用户问题找到最相关的聊天记录。

## 📁 项目结构

```
rag测试/
├── csv/                          # 存放微信聊天记录CSV文件
├── api/
│   ├── api_service.py           # 完整版API服务
│   └── api_service_test.py      # 测试版API服务（小规模）
├── core/
│   ├── test_csv_final.py        # 完整版向量数据库构建
│   ├── test_csv_small.py        # 测试版向量数据库构建（100条记录）
│   └── rebuild_full_database.py # 数据库重建工具
├── clients/
│   ├── external_client.py       # 外部设备客户端
│   └── api_client_test.py       # API测试客户端
├── chroma_wechat_db/            # 完整版向量数据库目录
├── chroma_wechat_db_test/       # 测试版向量数据库目录
└── README.md                    # 本文件
```

## 🛠️ 技术栈

- **向量数据库**: ChromaDB
- **文本嵌入**: 阿里云DashScope Text-Embedding-V3
- **大语言模型**: 通义千问 (Qwen-Plus)
- **API框架**: FastAPI
- **文档处理**: LangChain
- **HTTP客户端**: Requests

## 📋 环境要求

1. Python 3.8+
2. 阿里云DashScope API密钥

## 🔧 安装步骤

### 1. 克隆项目并安装依赖

```bash
# 安装依赖包
pip install langchain-community langchain-chroma langchain-core
pip install fastapi uvicorn requests tqdm
pip install dashscope
pip install sentence-transformers  # 可选，用于本地embedding
```

### 2. 准备数据

将微信聊天记录的CSV文件放入 `csv/` 目录中。CSV文件应包含以下字段：
- `msg`: 消息内容
- `CreateTime`: 创建时间
- `talker`: 发送者
- `type_name`: 消息类型
- `room_name`: 房间名称
- `is_sender`: 是否为自己发送

### 3. 配置API密钥

在相关文件中设置你的DashScope API密钥：
```python
os.environ["DASHSCOPE_API_KEY"] = "your-api-key-here"
```

## 🚀 使用指南

### 方案一：测试版快速体验（推荐新手）

**适用场景**: 快速测试功能，验证系统工作正常

#### 第1步：创建测试向量数据库
```bash
python core/test_csv_small.py
```
- 仅处理前100条聊天记录
- 创建时间：10-30秒
- 生成测试数据库：`chroma_wechat_db_test/`

#### 第2步：启动测试API服务
```bash
python api/api_service_test.py
```
- 服务地址：http://localhost:8001
- API文档：http://localhost:8001/docs

#### 第3步：客户端测试
```bash
# 方式1：本地测试客户端
python clients/api_client_test.py
# 选择测试服务器地址：localhost:8001

# 方式2：外部设备客户端
python clients/external_client.py
# 输入服务器IP地址
```

### 方案二：完整版生产环境

**适用场景**: 处理完整聊天记录，生产环境使用

#### 第1步：创建完整向量数据库
```bash
python core/test_csv_final.py
```
- 处理所有CSV文件中的聊天记录
- 创建时间：根据记录数量，通常3-10分钟
- 生成数据库：`chroma_wechat_db/`

#### 第2步：启动完整API服务
```bash
python api/api_service.py
```
- 服务地址：http://localhost:8000
- API文档：http://localhost:8000/docs

#### 第3步：客户端连接
```bash
# 本地客户端
python clients/api_client_test.py

# 外部设备客户端
python clients/external_client.py
```

## 🌐 API接口说明

### 主要端点

- `GET /`: 服务信息
- `GET /health`: 健康检查
- `GET /stats`: 数据库统计信息
- `POST /query_simple`: 简化查询接口

### 查询示例

```bash
# 使用curl测试
curl -X POST "http://localhost:8000/query_simple" \
     -d "question=张宏哲说了什么&max_results=5"

# 响应格式
{
  "question": "张宏哲说了什么",
  "records": [
    {
      "content": "聊天记录内容...",
      "sender": "张宏哲",
      "time": "2024-08-29 12:42:29",
      "similarity": 0.85
    }
  ],
  "count": 3
}
```

## 🔄 系统架构

### 1. 数据处理流程
```
CSV文件 → 数据清洗 → 格式化 → 向量化 → ChromaDB存储
```

### 2. 查询流程
```
用户问题 → 向量化 → 相似度搜索 → 返回相关记录
```

### 3. 服务架构
```
客户端 ← HTTP API ← FastAPI服务 ← ChromaDB ← 向量数据
```

## 🎯 核心功能

### 1. 向量数据库构建
- **智能数据清洗**: 自动过滤无效消息（表情、系统消息等）
- **批处理优化**: 支持大规模数据的分批处理
- **错误重试**: 自动重试机制，提高成功率

### 2. 语义搜索
- **相似度搜索**: 基于向量相似度的智能匹配
- **多结果返回**: 返回最相关的多条记录
- **元数据过滤**: 支持按发送者、时间等条件过滤

### 3. API服务
- **跨域支持**: 支持前端跨域访问
- **健康监控**: 提供服务状态检查
- **统计分析**: 数据库统计和分析功能

### 4. 多客户端支持
- **本地客户端**: 命令行交互式客户端
- **远程客户端**: 支持局域网内其他设备访问
- **自动测试**: 内置测试脚本验证功能

## ⚡ 性能优化

### 向量数据库创建优化
- 批处理大小：200条/批次
- API调用延迟：0.1秒/批次
- 重试机制：最多3次重试
- 错误处理：跳过失败批次，继续处理

### 查询性能
- 默认返回：5条最相关记录
- 最大支持：20条记录查询
- 响应时间：通常0.3-1秒

## 🐛 常见问题

### 1. 向量数据库创建失败
```bash
# 检查API密钥是否正确
echo $DASHSCOPE_API_KEY

# 检查网络连接
ping dashscope.aliyuncs.com

# 重新创建数据库
rm -rf chroma_wechat_db/
python core/test_csv_final.py
```

### 2. API服务连接失败
```bash
# 检查服务是否启动
curl http://localhost:8000/health

# 检查端口占用
netstat -an | grep 8000

# 防火墙设置（Windows）
netsh advfirewall firewall add rule name="RAG API" dir=in action=allow protocol=TCP localport=8000
```

### 3. 查询结果重复
已在最新版本中修复，通过移除文档分割避免重复记录。

### 4. 查询结果太少
- 检查 `max_results` 参数设置
- 检查相似度阈值设置
- 确认向量数据库数据完整性

## 📊 测试数据

### 测试版数据量
- 处理记录：100条
- 创建时间：10-30秒
- 数据库大小：约10MB

### 完整版数据量（示例）
- 处理记录：12,000+ 条
- 创建时间：3-5分钟
- 数据库大小：约100-500MB

## 🔮 未来改进

- [ ] 支持更多消息类型（图片、语音等）
- [ ] 添加时间范围过滤
- [ ] 实现聊天上下文关联
- [ ] 支持多语言查询
- [ ] 添加数据可视化界面
- [ ] 支持增量数据更新

## 📝 更新日志

### v1.2.0 (当前版本)
- ✅ 修复查询结果重复问题
- ✅ 优化向量数据库创建性能
- ✅ 添加测试版快速体验功能
- ✅ 改进错误处理和重试机制
- ✅ 增加客户端查询记录数限制修复

### v1.1.0
- ✅ 添加本地embedding支持
- ✅ 实现API服务和客户端
- ✅ 支持跨域访问

### v1.0.0
- ✅ 基础RAG系统实现
- ✅ 向量数据库构建
- ✅ 简单查询功能

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目仅供学习和研究使用。

---

**快速开始建议**: 新用户建议先使用测试版（方案一）快速体验功能，确认系统正常后再使用完整版处理大量数据。
# 网络复现并修改的代码，从csv文件中加载微信聊天记录，创建向量数据库。输入问题时，先从向量数据库查到相关问题，然后将问题和查到的内容
# 一并通过api给大模型，大模型返回回答。受算力限制并未成功。
import getpass
import os
import glob
from pathlib import Path

# 优先使用环境变量，缺失时再交互式输入
os.environ["DASHSCOPE_API_KEY"] = "sk-bae62c151c524da4b4ee5f04e4e19a3f"
import dashscope
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(model="qwen-plus")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class WeChatCSVLoader:
    """自定义微信聊天记录CSV加载器"""

    def __init__(self, csv_folder_path, encoding="utf-8"):
        self.csv_folder_path = Path(csv_folder_path)
        self.encoding = encoding

    def load(self):
        """加载所有CSV文件并返回文档列表"""
        documents = []

        # 查找所有CSV文件
        csv_files = list(self.csv_folder_path.glob("**/*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")

        for csv_file in csv_files:
            print(f"正在处理: {csv_file.name}")

            # 使用CSVLoader加载单个文件
            loader = CSVLoader(
                file_path=str(csv_file),
                encoding=self.encoding,
                csv_args={'delimiter': ','}
            )

            try:
                file_docs = loader.load()

                # 处理每个文档，格式化微信聊天内容
                for doc in file_docs:
                    # 解析CSV行数据
                    content_parts = doc.page_content.split('\n')
                    chat_data = {}

                    for part in content_parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            chat_data[key.strip()] = value.strip()

                    # 格式化聊天内容
                    if 'msg' in chat_data and chat_data['msg']:
                        formatted_content = f"""
聊天记录:
时间: {chat_data.get('CreateTime', '未知时间')}
发送者: {chat_data.get('talker', '未知用户')}
消息类型: {chat_data.get('type_name', '未知类型')}
内容: {chat_data.get('msg', '无内容')}
房间: {chat_data.get('room_name', '私聊')}
是否自己发送: {'是' if chat_data.get('is_sender') == '1' else '否'}
"""

                        # 创建新文档
                        new_doc = Document(
                            page_content=formatted_content.strip(),
                            metadata={
                                "source": str(csv_file),
                                "chat_time": chat_data.get('CreateTime', ''),
                                "sender": chat_data.get('talker', ''),
                                "msg_type": chat_data.get('type_name', ''),
                                "room": chat_data.get('room_name', ''),
                                "is_sender": chat_data.get('is_sender', '0')
                            }
                        )
                        documents.append(new_doc)

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        return documents

# 加载微信聊天记录CSV数据
print("正在加载微信聊天记录CSV文件...")
csv_loader = WeChatCSVLoader("csv")
docs = csv_loader.load()
print(f"已加载 {len(docs)} 条聊天记录")

# 过滤掉空消息和无意义消息
filtered_docs = []
for doc in docs:
    content = doc.page_content
    msg_content = ""

    # 提取消息内容
    for line in content.split('\n'):
        if line.startswith('内容:'):
            msg_content = line.replace('内容:', '').strip()
            break

    # 过滤条件：过滤掉表情、空消息、系统消息等
    if (msg_content and
        len(msg_content) > 2 and
        not msg_content.startswith('[') and
        not msg_content.startswith('表情') and
        '动画表情' not in doc.metadata.get('msg_type', '') and
        msg_content != 'I\'ve accepted your friend request. Now let\'s chat!'):
        filtered_docs.append(doc)

print(f"过滤后剩余 {len(filtered_docs)} 条有效聊天记录")

# 文本分割 - 针对聊天记录调整参数
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # 聊天记录相对较短，使用较小的chunk
    chunk_overlap=100,  # 适当重叠
    separators=["\n聊天记录:", "\n", "。", "！", "？", "；", " "]  # 中文分隔符
)

print("正在分割文档...")
splits = text_splitter.split_documents(filtered_docs)
print(f"已分割为 {len(splits)} 个片段")

print("正在创建向量数据库...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="./chroma_wechat_db"  # 持久化存储
)
print("向量数据库创建完成")

# 构建RAG链
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # 检索更多相关聊天记录
)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 测试查询
print("\n" + "="*50)
print("微信聊天记录RAG系统已就绪！")
print("="*50)

while True:
    query = input("\n请输入您的问题（输入'quit'退出）: ")
    if query.lower() == 'quit':
        break

    print(f"\n正在查询: {query}")
    print("-" * 30)

    try:
        result = rag_chain.invoke(query)
        print(f"回答: {result}")
    except Exception as e:
        print(f"查询出错: {e}")
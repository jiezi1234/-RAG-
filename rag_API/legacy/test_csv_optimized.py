# 利用小部分聊天记录创建向量数据库，并进行测试。测试成功
import getpass
import os
import glob
from pathlib import Path
import time
from tqdm import tqdm

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

    def __init__(self, csv_folder_path, encoding="utf-8", max_files=2):  # 限制处理文件数
        self.csv_folder_path = Path(csv_folder_path)
        self.encoding = encoding
        self.max_files = max_files

    def load(self):
        """加载CSV文件并返回文档列表"""
        documents = []

        # 查找所有CSV文件，但只处理前几个
        csv_files = list(self.csv_folder_path.glob("**/*.csv"))[:self.max_files]
        print(f"找到 {len(list(self.csv_folder_path.glob('**/*.csv')))} 个CSV文件，将处理前 {len(csv_files)} 个")

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
                processed_count = 0

                # 处理每个文档，但限制数量
                for doc in file_docs[:500]:  # 每个文件最多处理500条
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
                        processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录")

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        return documents

def create_vectorstore_with_progress(documents, embeddings):
    """分批创建向量数据库，显示进度"""

    # 检查是否已存在向量数据库
    db_path = "./chroma_wechat_db"
    if os.path.exists(db_path):
        print("发现已存在的向量数据库，正在加载...")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        return vectorstore

    print(f"开始创建向量数据库，总共 {len(documents)} 个文档...")

    # 分批处理，每批50个文档
    batch_size = 50
    vectorstore = None

    for i in tqdm(range(0, len(documents), batch_size), desc="创建向量数据库"):
        batch = documents[i:i + batch_size]

        try:
            if vectorstore is None:
                # 创建第一批
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=db_path
                )
            else:
                # 添加后续批次
                vectorstore.add_documents(batch)

            # 添加小延迟避免API限流
            time.sleep(0.5)

        except Exception as e:
            print(f"批次 {i//batch_size + 1} 处理失败: {e}")
            continue

    return vectorstore

# 主程序
def main():
    try:
        # 加载微信聊天记录CSV数据（限制数量）
        print("正在加载微信聊天记录CSV文件...")
        csv_loader = WeChatCSVLoader("csv", max_files=2)  # 只处理2个文件测试
        docs = csv_loader.load()
        print(f"已加载 {len(docs)} 条聊天记录")

        if not docs:
            print("未找到有效的聊天记录，请检查CSV文件")
            return

        # 过滤掉空消息和无意义消息
        print("正在过滤无效消息...")
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

        # 限制处理数量进行测试
        test_docs = filtered_docs[:200]  # 只处理前200条
        print(f"测试模式：处理前 {len(test_docs)} 条记录")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n聊天记录:", "\n", "。", "！", "？", "；", " "]
        )

        print("正在分割文档...")
        splits = text_splitter.split_documents(test_docs)
        print(f"已分割为 {len(splits)} 个片段")

        # 创建向量数据库（带进度条）
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        vectorstore = create_vectorstore_with_progress(splits, embeddings)

        if vectorstore is None:
            print("向量数据库创建失败")
            return

        print("向量数据库创建完成！")

        # 构建RAG链
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
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

        # 先做一个测试查询
        test_query = "聊天记录中都有哪些人？"
        print(f"\n测试查询: {test_query}")
        try:
            result = rag_chain.invoke(test_query)
            print(f"测试结果: {result}")
        except Exception as e:
            print(f"测试查询失败: {e}")

        # 交互式查询
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

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
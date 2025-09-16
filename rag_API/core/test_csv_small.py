# 小规模测试版本 - 仅使用前100条记录快速测试
import getpass
import os
import glob
from pathlib import Path
import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Processing", total=None):
        print(f"{desc}...")
        return iterable

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
    """自定义微信聊天记录CSV加载器 - 小规模测试版"""

    def __init__(self, csv_folder_path, encoding="utf-8", max_records=100):
        self.csv_folder_path = Path(csv_folder_path)
        self.encoding = encoding
        self.max_records = max_records  # 限制记录数量

    def load(self):
        """加载CSV文件并返回有限数量的文档列表"""
        documents = []
        total_processed = 0

        csv_files = list(self.csv_folder_path.glob("**/*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件，将加载前 {self.max_records} 条有效记录")

        for csv_file in csv_files:
            if total_processed >= self.max_records:
                break

            print(f"正在处理: {csv_file.name}")

            loader = CSVLoader(
                file_path=str(csv_file),
                encoding=self.encoding,
                csv_args={'delimiter': ','}
            )

            try:
                file_docs = loader.load()
                processed_count = 0
                valid_count = 0

                for doc in file_docs:
                    if total_processed >= self.max_records:
                        break

                    try:
                        content_parts = doc.page_content.split('\n')
                        chat_data = {}

                        for part in content_parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                chat_data[key.strip()] = value.strip()

                        msg_content = chat_data.get('msg', '').strip()
                        if not msg_content:
                            continue

                        # 过滤无意义消息
                        if (len(msg_content) <= 2 or
                            msg_content.startswith('[') or
                            msg_content.startswith('表情') or
                            '动画表情' in chat_data.get('type_name', '') or
                            msg_content == "I've accepted your friend request. Now let's chat!" or
                            '<msg>' in msg_content):
                            continue

                        # 格式化聊天内容
                        formatted_content = f"""聊天记录:
时间: {chat_data.get('CreateTime', '未知时间')}
发送者: {chat_data.get('talker', '未知用户')}
消息类型: {chat_data.get('type_name', '文本')}
内容: {msg_content}
房间: {chat_data.get('room_name', '私聊')}
是否自己发送: {'是' if chat_data.get('is_sender') == '1' else '否'}"""

                        new_doc = Document(
                            page_content=formatted_content,
                            metadata={
                                "source": str(csv_file.name),
                                "chat_time": chat_data.get('CreateTime', ''),
                                "sender": chat_data.get('talker', ''),
                                "msg_type": chat_data.get('type_name', ''),
                                "room": chat_data.get('room_name', ''),
                                "is_sender": chat_data.get('is_sender', '0'),
                                "msg_content": msg_content[:200]
                            }
                        )
                        documents.append(new_doc)
                        valid_count += 1
                        total_processed += 1

                    except Exception as e:
                        continue

                    processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录，有效记录 {valid_count} 条")

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        print(f"✅ 总共加载了 {len(documents)} 条有效记录用于测试")
        return documents

def create_small_vectorstore(documents, embeddings):
    """创建小规模测试向量数据库"""

    db_path = "./chroma_wechat_db_test"

    # 删除旧的测试数据库
    if os.path.exists(db_path):
        print("删除旧的测试数据库...")
        try:
            import shutil
            shutil.rmtree(db_path)
        except Exception as e:
            print(f"删除失败: {e}")

    print(f"创建测试向量数据库，文档数量: {len(documents)}")

    try:
        # 一次性创建所有文档，无需分批
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_path
        )
        print("✅ 测试向量数据库创建成功")
        return vectorstore

    except Exception as e:
        print(f"❌ 向量数据库创建失败: {e}")
        return None

def test_query(vectorstore):
    """测试查询功能"""
    print("\n" + "="*50)
    print("🔍 测试查询功能")
    print("="*50)

    test_queries = [
        "你好",
        "时间",
        "学校",
        "什么",
    ]

    for query in test_queries:
        print(f"\n🔍 测试查询: '{query}'")
        try:
            results = vectorstore.similarity_search_with_score(query, k=3)
            print(f"找到 {len(results)} 条结果:")

            for i, (doc, score) in enumerate(results, 1):
                print(f"\n📝 结果 {i} (相似度: {score:.3f}):")
                print(f"发送者: {doc.metadata.get('sender', '未知')}")
                print(f"时间: {doc.metadata.get('chat_time', '未知')}")
                print(f"内容: {doc.metadata.get('msg_content', '')[:50]}...")
                print("-" * 30)

        except Exception as e:
            print(f"❌ 查询失败: {e}")

    # 交互式测试
    print("\n🎮 交互式测试 (输入 'quit' 退出)")
    while True:
        query = input("\n❓ 输入查询: ").strip()
        if query.lower() in ['quit', 'q', '退出']:
            break

        if not query:
            continue

        try:
            results = vectorstore.similarity_search_with_score(query, k=3)
            print(f"\n找到 {len(results)} 条结果:")

            for i, (doc, score) in enumerate(results, 1):
                print(f"\n📝 结果 {i} (相似度: {score:.3f}):")
                print(f"发送者: {doc.metadata.get('sender', '未知')}")
                print(f"时间: {doc.metadata.get('chat_time', '未知')}")
                print(f"完整内容:\n{doc.page_content}")
                print("-" * 40)

        except Exception as e:
            print(f"❌ 查询失败: {e}")

def main():
    """主程序"""
    try:
        print("🚀 启动小规模测试版微信聊天记录RAG系统...")
        print("📋 测试版特点：")
        print("✅ 仅加载前100条有效记录")
        print("✅ 快速创建向量数据库")
        print("✅ 验证查询结果唯一性")
        print("="*50)

        if not os.path.exists("csv"):
            print("❌ 未找到csv文件夹")
            return

        # 加载少量CSV数据
        print("\n📂 正在加载少量CSV数据...")
        csv_loader = WeChatCSVLoader("csv", max_records=100)
        docs = csv_loader.load()

        if not docs:
            print("❌ 未找到有效聊天记录")
            return

        print(f"✅ 成功加载 {len(docs)} 条有效聊天记录")

        # 跳过文本分割，直接使用
        splits = docs
        print(f"✅ 使用 {len(splits)} 条记录创建向量数据库")

        # 创建向量数据库
        print("\n🔧 正在创建测试向量数据库...")
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        vectorstore = create_small_vectorstore(splits, embeddings)

        if vectorstore is None:
            print("❌ 向量数据库创建失败")
            return

        print("✅ 测试向量数据库准备完成！")

        # 测试查询
        test_query(vectorstore)

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
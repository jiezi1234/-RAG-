# 修改后的代码，成功利用全部聊天记录并创建向量数据库。
import getpass
import os
import glob
from pathlib import Path
import time
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的替代
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

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
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
                valid_count = 0

                # 处理每个文档
                for doc in file_docs:
                    try:
                        # 解析CSV行数据
                        content_parts = doc.page_content.split('\n')
                        chat_data = {}

                        for part in content_parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                chat_data[key.strip()] = value.strip()

                        # 检查是否有有效的消息内容
                        msg_content = chat_data.get('msg', '').strip()
                        if not msg_content:
                            continue

                        # 过滤无意义消息
                        if (len(msg_content) <= 2 or
                            msg_content.startswith('[') or
                            msg_content.startswith('表情') or
                            '动画表情' in chat_data.get('type_name', '') or
                            msg_content == "I've accepted your friend request. Now let's chat!" or
                            '<msg>' in msg_content):  # 过滤XML格式的系统消息
                            continue

                        # 格式化聊天内容
                        formatted_content = f"""聊天记录:
时间: {chat_data.get('CreateTime', '未知时间')}
发送者: {chat_data.get('talker', '未知用户')}
消息类型: {chat_data.get('type_name', '文本')}
内容: {msg_content}
房间: {chat_data.get('room_name', '私聊')}
是否自己发送: {'是' if chat_data.get('is_sender') == '1' else '否'}"""

                        # 创建新文档
                        new_doc = Document(
                            page_content=formatted_content,
                            metadata={
                                "source": str(csv_file.name),
                                "chat_time": chat_data.get('CreateTime', ''),
                                "sender": chat_data.get('talker', ''),
                                "msg_type": chat_data.get('type_name', ''),
                                "room": chat_data.get('room_name', ''),
                                "is_sender": chat_data.get('is_sender', '0'),
                                "msg_content": msg_content[:200]  # 截取前200字符用于检索
                            }
                        )
                        documents.append(new_doc)
                        valid_count += 1

                    except Exception as e:
                        continue  # 跳过有问题的记录

                    processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录，有效记录 {valid_count} 条")

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        return documents

def create_vectorstore_with_progress(documents, embeddings, batch_size=100):
    """分批创建向量数据库，显示进度"""

    # 检查是否已存在向量数据库，由于修改了数据处理逻辑，需要重新创建
    db_path = "./chroma_wechat_db"
    if os.path.exists(db_path) and os.listdir(db_path):
        print("检测到旧的向量数据库，由于数据处理逻辑已更新，需要重新创建...")
        try:
            import shutil
            shutil.rmtree(db_path)
            print("已删除旧数据库")
        except Exception as e:
            print(f"删除旧数据库失败: {e}")
            print("请手动关闭相关服务后重试，或重命名数据库目录")
            return None

    print(f"开始创建向量数据库，总共 {len(documents)} 个文档...")

    # 分批处理
    vectorstore = None
    failed_batches = 0
    max_retries = 3

    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(documents), batch_size), desc="创建向量数据库", total=total_batches):
        batch = documents[i:i + batch_size]

        retry_count = 0
        while retry_count < max_retries:
            try:
                if vectorstore is None:
                    # 创建第一批
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=db_path
                    )
                    print(f"成功创建向量数据库，第一批 {len(batch)} 个文档")
                else:
                    # 添加后续批次
                    vectorstore.add_documents(batch)

                # 在新版本中，数据会自动持久化，无需手动调用persist()
                # vectorstore.persist()  # 已移除，因为新版本不支持
                break  # 成功处理，跳出重试循环

            except Exception as e:
                retry_count += 1
                print(f"批次 {i//batch_size + 1} 处理失败 (重试 {retry_count}/{max_retries}): {e}")

                if retry_count >= max_retries:
                    failed_batches += 1
                    print(f"批次 {i//batch_size + 1} 最终失败，跳过")
                    break

                # 等待后重试
                time.sleep(2 ** retry_count)  # 指数退避

        # 减少延迟以加快处理速度
        if (i // batch_size + 1) % 20 == 0:  # 每20批休息长一点
            time.sleep(1)  # 减少休息时间
        else:
            time.sleep(0.1)  # 大幅减少每批的延迟

    if failed_batches > 0:
        print(f"警告: {failed_batches} 个批次处理失败")

    return vectorstore

def interactive_chat(rag_chain):
    """交互式聊天函数"""
    print("\n" + "="*60)
    print("WeChat Chat RAG System Ready!")
    print("="*60)
    print("Tip: You can ask these types of questions:")
    print("   - 某个人说了什么？")
    print("   - 关于某个话题的聊天内容")
    print("   - 某个时间段的对话")
    print("   - 聊天记录的统计信息")
    print("="*60)

    # 先做一个测试查询
    test_queries = [
        "聊天记录中都有哪些人参与了对话？",
        "最近在聊什么话题？"
    ]

    for test_query in test_queries:
        print(f"\nTest query: {test_query}")
        try:
            result = rag_chain.invoke(test_query)
            print(f"Result: {result}")
            break  # 成功一个就够了
        except Exception as e:
            print(f"Error: Test query failed: {e}")
            continue

    # 交互式查询
    while True:
        try:
            query = input("\n❓ 请输入您的问题（输入'quit'、'exit'或'q'退出）: ").strip()

            if query.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 再见！")
                break

            if not query:
                print("Warning: Please enter a valid question")
                continue

            print(f"\nQuerying: {query}")
            print("-" * 40)

            start_time = time.time()
            result = rag_chain.invoke(query)
            end_time = time.time()

            print(f"Answer: {result}")
            print(f"Time: {end_time - start_time:.2f} seconds")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"Error: Query failed: {e}")
            print("Tip: Please try rephrasing your question")

def main():
    """主程序"""
    try:
        print("Starting WeChat Chat RAG System...")

        # 检查CSV文件夹
        if not os.path.exists("csv"):
            print("Error: csv folder not found, please ensure CSV files are in csv directory")
            return

        # 加载微信聊天记录CSV数据
        print("\nLoading WeChat CSV files...")
        csv_loader = WeChatCSVLoader("csv")
        docs = csv_loader.load()

        if not docs:
            print("Error: No valid chat records found, please check CSV file format")
            return

        print(f"Success: Loaded {len(docs)} valid chat records")

        # 对于聊天记录，每条已经是独立完整的单元，跳过文本分割避免重复
        print("\nSkipping document splitting (chat records are already atomic units)...")
        splits = docs  # 直接使用原始文档，不进行分割
        print(f"Using {len(splits)} chat records as-is")

        # 创建向量数据库
        print("\nCreating/loading vector database...")
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        vectorstore = create_vectorstore_with_progress(splits, embeddings, batch_size=200)  # 增大批次大小

        if vectorstore is None:
            print("Error: Vector database creation failed")
            return

        print("Success: Vector database ready!")

        # 构建RAG链
        print("\nBuilding RAG retrieval chain...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 检索5个最相关的片段
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

        print("Success: RAG system build complete!")

        # 开始交互式对话
        interactive_chat(rag_chain)

    except Exception as e:
        print(f"Error: Program execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# 简化版本 - 重新创建包含全部数据的向量数据库
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

from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
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

        csv_files = list(self.csv_folder_path.glob("**/*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")

        for csv_file in tqdm(csv_files, desc="处理CSV文件"):
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

                        # 过滤无效消息
                        if (len(msg_content) <= 2 or
                            msg_content.startswith('[') or
                            msg_content.startswith('表情') or
                            '动画表情' in chat_data.get('type_name', '') or
                            msg_content == "I've accepted your friend request. Now let's chat!" or
                            '<msg>' in msg_content):
                            continue

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

                    except Exception as e:
                        continue

                    processed_count += 1

                print(f"  - 处理了 {processed_count} 条记录，有效记录 {valid_count} 条")

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

        return documents

def create_full_vectorstore(documents, embeddings, batch_size=100):
    """创建包含全部数据的向量数据库"""

    db_path = "./chroma_full_db"

    # 删除旧数据库
    if os.path.exists(db_path):
        import shutil
        print(f"删除旧数据库: {db_path}")
        shutil.rmtree(db_path)

    print(f"开始创建完整向量数据库，总共 {len(documents)} 个文档...")

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
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=db_path
                    )
                    print(f"✅ 成功创建向量数据库，第一批 {len(batch)} 个文档")
                else:
                    vectorstore.add_documents(batch)

                vectorstore.persist()
                break

            except Exception as e:
                retry_count += 1
                print(f"批次 {i//batch_size + 1} 处理失败 (重试 {retry_count}/{max_retries}): {e}")

                if retry_count >= max_retries:
                    failed_batches += 1
                    print(f"批次 {i//batch_size + 1} 最终失败，跳过")
                    break

                time.sleep(1)

        # 显示进度
        if (i // batch_size + 1) % 10 == 0:
            completed = i + batch_size
            progress = min(completed / len(documents) * 100, 100)
            print(f"进度: {progress:.1f}% ({completed}/{len(documents)})")

        time.sleep(0.2)  # 稍微慢一点避免过载

    if failed_batches > 0:
        print(f"⚠️ 警告: {failed_batches} 个批次处理失败")

    return vectorstore

def simple_query_system(vectorstore):
    """简单的查询系统"""
    print("\n" + "="*60)
    print("🤖 简单查询系统 (不使用大语言模型)")
    print("✅ 直接返回最相关的聊天记录")
    print("="*60)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    while True:
        try:
            query = input("\n❓ 请输入查询问题（输入'quit'退出）: ").strip()

            if query.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 再见！")
                break

            if not query:
                print("⚠️ 请输入有效问题")
                continue

            print(f"\n🔍 正在查询: {query}")
            print("-" * 50)

            start_time = time.time()
            # 使用similarity_search_with_score获取相似度
            results = vectorstore.similarity_search_with_score(query, k=5)
            end_time = time.time()

            print(f"📊 找到 {len(results)} 条相关记录 (耗时: {end_time - start_time:.2f}秒):")

            for i, (doc, score) in enumerate(results, 1):
                print(f"\n📝 记录 {i} (相似度: {score:.3f}):")
                print(doc.page_content)

                # 显示元数据
                if 'sender' in doc.metadata:
                    print(f"💬 发送者: {doc.metadata['sender']}")
                if 'chat_time' in doc.metadata:
                    print(f"⏰ 时间: {doc.metadata['chat_time']}")

                print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 查询出错: {e}")

def main():
    """主程序"""
    try:
        print("🚀 重新创建完整向量数据库系统")
        print("📋 本次将处理全部聊天记录数据")
        print("=" * 60)

        if not os.path.exists("csv"):
            print("❌ 未找到csv文件夹")
            return

        # 检查是否已安装sentence-transformers
        try:
            import sentence_transformers
            print("✅ sentence-transformers 已安装")
        except ImportError:
            print("❌ 缺少 sentence-transformers")
            print("💡 请运行: pip install sentence-transformers")
            return

        # 加载CSV数据
        print("\n📂 正在加载所有CSV文件...")
        csv_loader = WeChatCSVLoader("csv")
        docs = csv_loader.load()

        if not docs:
            print("❌ 未找到有效聊天记录")
            return

        print(f"✅ 成功加载 {len(docs):,} 条有效聊天记录")

        # 文本分割
        print("\n📝 正在分割文档...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n聊天记录:", "\n", "。", "！", "？", "；", "，", " "]
        )

        splits = text_splitter.split_documents(docs)
        print(f"✅ 已分割为 {len(splits):,} 个文本片段")

        # 创建本地embedding
        print("\n🔧 正在加载本地embedding模型...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            print("✅ 本地embedding模型加载成功")
        except Exception as e:
            print(f"❌ embedding模型加载失败: {e}")
            return

        # 创建完整向量数据库
        print(f"\n🔧 正在创建包含全部 {len(splits):,} 条记录的向量数据库...")
        print("⚠️ 这可能需要较长时间，请耐心等待...")

        vectorstore = create_full_vectorstore(splits, embeddings, batch_size=50)

        if vectorstore is None:
            print("❌ 向量数据库创建失败")
            return

        print("✅ 完整向量数据库创建完成！")

        # 验证数据库
        try:
            total_count = vectorstore._collection.count()
            print(f"📊 数据库包含 {total_count:,} 条记录")
        except:
            print("📊 数据库创建完成，记录数验证失败但数据库可用")

        # 启动简单查询系统
        simple_query_system(vectorstore)

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🤖 完整数据向量数据库创建系统")
    print("=" * 60)
    main()
# 网络复现并修改的代码，从网页爬取内容创建向量数据库。输入问题时，先从向量数据库查到相关问题，然后将问题和查到的内容
# 一并通过api给大模型，大模型返回回答。
import getpass
import os

# 优先使用环境变量，缺失时再交互式输入
os.environ["DASHSCOPE_API_KEY"] = "sk-bae62c151c524da4b4ee5f04e4e19a3f"
import dashscope
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(model="qwen-plus")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
print("正在加载网页内容...")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    requests_kwargs={
        "headers": {
            "User-Agent": os.getenv("USER_AGENT", "rag-test/1.0 (+https://example.com/contact)")
        }
    },
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print(f"已加载 {len(docs)} 个文档")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
print("正在分割文档...")
splits = text_splitter.split_documents(docs)
print(f"已分割为 {len(splits)} 个片段")
print("正在创建向量数据库...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=DashScopeEmbeddings(model="text-embedding-v3"),
)
print("向量数据库创建完成")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("正在执行 RAG 查询...")
result = rag_chain.invoke("思维链是什么?")
print("\n查询结果:")
print(result)
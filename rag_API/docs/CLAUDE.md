# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) testing project using LangChain with DashScope (Alibaba Cloud) and Chroma vector database. The main script demonstrates a simple RAG pipeline that loads web content, creates embeddings, and answers questions based on the retrieved context.

## Development Setup

### Environment Setup
```bash
# Activate the virtual environment
.venv\Scripts\activate

# Check installed packages
.venv\Scripts\pip.exe list
```

### Running the Application
```bash
# Run the main RAG script
.venv\Scripts\python.exe teat.py
```

## Architecture

### Core Components
- **LLM**: ChatTongyi (qwen-plus model) via DashScope API
- **Embeddings**: DashScopeEmbeddings (text-embedding-v3 model)
- **Vector Store**: Chroma for document storage and retrieval
- **Document Processing**: WebBaseLoader for web scraping, RecursiveCharacterTextSplitter for chunking
- **RAG Chain**: LangChain pipeline combining retrieval and generation

### Key Dependencies
- `langchain` and related packages for RAG pipeline
- `dashscope` for Alibaba Cloud AI services
- `chromadb` for vector database
- `beautifulsoup4` for web scraping
- `langchain-chroma` for Chroma integration

### Current Implementation
The project loads content from a web page (https://lilianweng.github.io/posts/2023-06-23-agent/), splits it into chunks, creates embeddings, stores in Chroma, and answers questions using retrieved context.

## Environment Variables
- `DASHSCOPE_API_KEY`: Required for DashScope API access (currently hardcoded in teat.py)
- `USER_AGENT`: Optional for web scraping requests

## Notes
- API key is currently hardcoded and should be moved to environment variables for security
- No test suite or build process currently configured
- Single Python file structure - consider modularizing for larger applications
AI-RAG-Assistant

AI-RAG-Assistant is a production-ready AI application that combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide context-aware, knowledge-grounded responses.

The system supports both local and cloud-based language models through Ollama and OpenAI integrations. It includes document ingestion, semantic search, vector storage, conversational memory, structured outputs, and a REST API for building scalable AI-powered applications.

Overview

This project was designed to demonstrate modern AI engineering practices and end-to-end LLM application development. The architecture separates model providers, retrieval pipelines, API services, and conversation management into modular components that can be extended and maintained easily.

Key Features

• Multi-provider LLM support (OpenAI and Ollama)

• Retrieval-Augmented Generation (RAG)

• Semantic search using vector embeddings

• Local vector database powered by ChromaDB

• Document ingestion and automatic text chunking

• Conversational memory management

• Structured output generation

• FastAPI REST API

• Streaming response support

• Sentiment analysis

• Text summarization

• Modular and extensible architecture

Technology Stack

Programming Language:
Python

Backend Framework:
FastAPI

LLM Providers:
OpenAI
Ollama

Vector Database:
ChromaDB

Embeddings:
Sentence Transformers

Document Processing:
PyPDF
Python-Docx

Validation and Configuration:
Pydantic
Pydantic Settings

Project Architecture

api/
REST API endpoints and application startup

cli/
Interactive command-line interface

config/
Application settings and prompts

core/
LLM abstractions, providers, and conversation management

llm/
Schemas and structured output generation

rag/
Document loading, embeddings, vector storage, and retrieval pipeline

data/
Knowledge base documents and vector storage

tests/
Unit and integration tests

Core Workflow

1. Documents are loaded and processed.

2. Documents are cleaned and split into chunks.

3. Embeddings are generated for each chunk.

4. Embeddings are stored in ChromaDB.

5. User queries are converted into embeddings.

6. Semantic search retrieves the most relevant chunks.

7. Retrieved context is combined with the user question.

8. The language model generates a grounded response.

9. The response is returned through the CLI or API.

Supported Modes

Demo Mode

Demonstrates the main capabilities of the system.

CLI Mode

Interactive conversational assistant running in the terminal.

RAG Mode

Question-answering over custom knowledge bases.

API Mode

REST API server for integration with external applications.

Installation

Clone the repository:

git clone https://github.com/MohamedElzoka/AI-RAG-Assistant.git

cd AI-RAG-Assistant

Create a virtual environment:

python -m venv venv

Activate the environment:

Windows:
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Configuration

Create a .env file and configure the required settings.

Example:

LLM_PROVIDER=ollama

OLLAMA_MODEL=llama3.2

OLLAMA_BASE_URL=http://localhost:11434

OPENAI_API_KEY=your_api_key

Running the Application

CLI Mode:

python main.py --mode cli

Demo Mode:

python main.py --mode demo

RAG Mode:

python main.py --mode rag

API Mode:

python main.py --mode api

API Documentation

After starting the API server:

http://localhost:8000/docs

Available Endpoints

POST /chat

POST /rag/query

POST /rag/add-text

POST /analyze/sentiment

POST /analyze/summarize

GET /health

GET /stats

Future Improvements

• User authentication and authorization

• Hybrid search (keyword + semantic search)

• Advanced conversation memory

• Docker deployment

• Kubernetes support

• Monitoring and observability

• Multi-agent workflows

Author

Mohamed Ayman Elzoka

Machine Learning Engineer and Artificial Intelligence Researcher

Master's Student in Artificial Intelligence

Benha University

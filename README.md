# AI-RAG-Assistant

AI-RAG-Assistant is a production-ready Artificial Intelligence application that combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to deliver context-aware, knowledge-grounded responses.

The system supports both local and cloud-based models through Ollama and OpenAI integrations and provides document processing, semantic search, conversational memory, structured outputs, and REST API capabilities.

---

## Project Overview

This project demonstrates modern AI engineering practices and end-to-end LLM application development. The architecture follows a modular design that separates model providers, retrieval pipelines, APIs, conversation management, and structured output generation.

---

## Key Features

| Feature                              | Description                                              |
| ------------------------------------ | -------------------------------------------------------- |
| Multi-Provider LLM Support           | Works with OpenAI and Ollama                             |
| Retrieval-Augmented Generation (RAG) | Retrieves relevant knowledge before generating answers   |
| Semantic Search                      | Searches documents based on meaning rather than keywords |
| Vector Database                      | Stores embeddings using ChromaDB                         |
| Conversational Memory                | Maintains chat history and context                       |
| Structured Outputs                   | Generates validated structured responses                 |
| REST API                             | FastAPI-powered backend                                  |
| Streaming Responses                  | Supports token-by-token generation                       |
| Sentiment Analysis                   | AI-powered sentiment classification                      |
| Text Summarization                   | Automatic text summarization                             |
| Modular Architecture                 | Easily extendable and maintainable                       |

---

## Technology Stack

| Category                 | Technology            |
| ------------------------ | --------------------- |
| Programming Language     | Python                |
| Backend Framework        | FastAPI               |
| Local LLM                | Ollama                |
| Cloud LLM                | OpenAI                |
| Vector Database          | ChromaDB              |
| Embedding Models         | Sentence Transformers |
| Validation               | Pydantic              |
| Configuration Management | Pydantic Settings     |
| PDF Processing           | PyPDF                 |
| Word Processing          | Python-Docx           |

---

## System Architecture

| Directory | Responsibility                               |
| --------- | -------------------------------------------- |
| api/      | REST API endpoints and application startup   |
| cli/      | Interactive command-line interface           |
| config/   | Prompts and configuration management         |
| core/     | LLM abstractions and conversation management |
| llm/      | Schemas and structured output generation     |
| rag/      | Document processing and retrieval pipeline   |
| data/     | Knowledge base and vector database storage   |
| tests/    | Unit and integration tests                   |

---

## RAG Pipeline Workflow

```text
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Semantic Search
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Build Context
      │
      ▼
Send Context + Question to LLM
      │
      ▼
Generate Grounded Response
      │
      ▼
Return Answer
```

---

## Supported Operating Modes

| Mode | Description                           |
| ---- | ------------------------------------- |
| demo | Demonstrates all project capabilities |
| cli  | Interactive terminal chatbot          |
| rag  | Knowledge-base question answering     |
| api  | REST API server                       |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MohamedElzoka/AI-RAG-Assistant.git
cd AI-RAG-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root.

Example:

```env
LLM_PROVIDER=ollama

OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

OPENAI_API_KEY=your_api_key

EMBEDDING_MODEL=all-MiniLM-L6-v2

API_HOST=0.0.0.0
API_PORT=8000
```

---

## Running the Application

### CLI Mode

```bash
python main.py --mode cli
```

### Demo Mode

```bash
python main.py --mode demo
```

### RAG Mode

```bash
python main.py --mode rag
```

### API Mode

```bash
python main.py --mode api
```

---

## REST API Endpoints

| Method | Endpoint           | Description                  |
| ------ | ------------------ | ---------------------------- |
| POST   | /chat              | Standard LLM conversation    |
| POST   | /rag/query         | Query the knowledge base     |
| POST   | /rag/add-text      | Add text to the RAG database |
| POST   | /analyze/sentiment | Sentiment analysis           |
| POST   | /analyze/summarize | Text summarization           |
| GET    | /health            | Service health check         |
| GET    | /stats             | System statistics            |

---

## API Documentation

Once the API server is running:

```text
http://localhost:8000/docs
```

FastAPI automatically provides Swagger UI documentation for testing endpoints.

---

## Example Workflow

1. Load documents into the knowledge base.
2. Generate embeddings for document chunks.
3. Store embeddings inside ChromaDB.
4. Receive a user query.
5. Convert the query into an embedding.
6. Retrieve the most relevant document chunks.
7. Build contextual prompts.
8. Generate a grounded response using the selected LLM.
9. Return the answer through the CLI or API.

---

## Future Enhancements

| Planned Feature            |
| -------------------------- |
| Docker Deployment          |
| Kubernetes Support         |
| User Authentication        |
| Hybrid Search              |
| Advanced Memory Management |
| Monitoring and Logging     |
| Multi-Agent Workflows      |
| Web Interface              |

---

## Author

Mohamed Ayman Elzoka

Machine Learning Engineer

Master's Student in Artificial Intelligence

Benha University

---

## License

This project is intended for educational, research, and portfolio purposes.

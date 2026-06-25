# RAG Energy & Climate Chatbot

A lightweight Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **LangChain**, and **FAISS**. The application demonstrates document retrieval from a local vector store containing sample IPCC and IEA document excerpts and generates a synthesized response based on the retrieved context.

This project was developed as a demonstration backend for an AI-powered Energy & Climate Intelligence Platform.

---

## Features

- FastAPI REST API
- LangChain-powered Retrieval-Augmented Generation (RAG) workflow
- Local FAISS vector store
- Five sample IPCC/IEA document chunks
- Similarity search over document embeddings
- Mock local language model for answer synthesis
- JSON API responses
- Python test script demonstrating endpoint usage

---

## Project Structure

```
rag-energy-climate-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── rag.py
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── test_rag.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

- Python
- FastAPI
- LangChain
- FAISS
- Uvicorn

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd rag-energy-climate-chatbot
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

The API will be available at

```
http://127.0.0.1:8000
```

Interactive API documentation

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST `/chat/rag`

Request

```json
{
  "query": "How can renewable energy help reduce emissions?"
}
```

Example Response

```json
{
  "query": "How can renewable energy help reduce emissions?",
  "answer": "Based on the retrieved IPCC/IEA sample document chunks...",
  "documents_retrieved": 3,
  "retrieved_chunks": [
    {
      "source": "IPCC sample chunk 2",
      "content": "The IPCC identifies mitigation actions..."
    }
  ]
}
```

---

## Testing

Run the included test script

```bash
python test_rag.py
```

The script sends two example queries to the API and prints the synthesized responses along with the retrieved document chunks.

---

## RAG Workflow

1. User submits a natural language query.
2. The query is embedded using a lightweight local embedding implementation.
3. LangChain performs similarity search against the local FAISS vector store.
4. The top matching IPCC/IEA document chunks are retrieved.
5. A mock local language model synthesizes a response using the retrieved context.
6. The API returns the generated answer and retrieved sources as JSON.

---

## Notes

- This project uses a lightweight local embedding implementation for demonstration purposes.
- The FAISS index is stored locally and automatically created on first run if it does not already exist.
- The language model component is intentionally mocked to keep the project fully local without requiring external APIs or model downloads.

---

## License

This project is intended for educational and demonstration purposes.
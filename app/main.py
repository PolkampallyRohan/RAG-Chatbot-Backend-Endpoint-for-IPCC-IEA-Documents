from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer_query


app = FastAPI(
    title="RAG Energy & Climate Chatbot",
    description="A FastAPI backend using LangChain and FAISS for mock IPCC/IEA RAG retrieval.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the RAG chatbot")


@app.get("/")
def root():
    return {
        "message": "RAG Energy & Climate Chatbot API is running.",
        "endpoint": "/chat/rag",
    }


@app.post("/chat/rag")
def chat_rag(request: ChatRequest) -> dict:
    try:
        return answer_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
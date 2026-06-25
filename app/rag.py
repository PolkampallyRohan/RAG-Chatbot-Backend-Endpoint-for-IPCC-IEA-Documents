import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


FAISS_INDEX_DIR = "faiss_index"


class SimpleKeywordEmbeddings(Embeddings):
    keywords = [
        "climate", "warming", "emissions", "carbon", "energy",
        "renewable", "solar", "wind", "efficiency", "adaptation",
        "mitigation", "electricity", "fossil", "transition", "policy"
    ]

    def embed_query(self, text: str) -> List[float]:
        text = text.lower()
        return [float(text.count(keyword)) for keyword in self.keywords]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


embeddings = SimpleKeywordEmbeddings()


def create_sample_documents() -> List[Document]:
    return [
        Document(
            page_content=(
                "IPCC reports state that human-caused greenhouse gas emissions "
                "have led to widespread warming of the atmosphere, ocean, and land."
            ),
            metadata={"source": "IPCC sample chunk 1"},
        ),
        Document(
            page_content=(
                "The IPCC identifies mitigation actions such as reducing carbon emissions, "
                "expanding renewable energy, improving efficiency, and protecting forests."
            ),
            metadata={"source": "IPCC sample chunk 2"},
        ),
        Document(
            page_content=(
                "Climate adaptation includes preparing infrastructure, agriculture, water systems, "
                "and communities for climate risks such as heatwaves, floods, and droughts."
            ),
            metadata={"source": "IPCC sample chunk 3"},
        ),
        Document(
            page_content=(
                "IEA analysis highlights that clean energy transitions rely on solar power, "
                "wind power, energy storage, electric vehicles, and modern electricity grids."
            ),
            metadata={"source": "IEA sample chunk 1"},
        ),
        Document(
            page_content=(
                "The IEA emphasizes that reducing fossil fuel dependence and improving energy "
                "efficiency are important parts of achieving secure and sustainable energy systems."
            ),
            metadata={"source": "IEA sample chunk 2"},
        ),
    ]


def build_and_save_vector_store() -> FAISS:
    docs = create_sample_documents()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def load_or_create_vector_store() -> FAISS:
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return build_and_save_vector_store()


vector_store = load_or_create_vector_store()


def synthesize_answer(query: str, retrieved_docs: List[Document]) -> str:
    context = " ".join(doc.page_content for doc in retrieved_docs)

    return (
        f"Based on the retrieved IPCC/IEA sample document chunks, the answer to "
        f"'{query}' is: {context}"
    )


def answer_query(query: str) -> Dict[str, Any]:
    retrieved_docs = vector_store.similarity_search(query, k=3)
    answer = synthesize_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer,
        "documents_retrieved": len(retrieved_docs),
        "retrieved_chunks": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content,
            }
            for doc in retrieved_docs
        ],
    }
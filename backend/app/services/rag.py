import os
import uuid
import logging
import time

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient, models

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


class RAGService:
    def __init__(self):
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.qdrant_path = Path(os.getenv("QDRANT_PATH", "./data/qdrant"))
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.qdrant_path))
        self.collection = "legal_chunks"
        self.embedder = None
        self._collection_ready = False

    def _load_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            start = time.perf_counter()
            logger.warning("[RAG] Starting embedding model load")

            self.embedder = SentenceTransformer(
                self.embedding_model_name,
                device="cpu"
            )

            logger.warning(
                "[RAG] Embedding model loaded in %.2fs",
                time.perf_counter() - start
            )

        return self.embedder

    def _ensure_collection(self):
        if self._collection_ready:
            return

        dim = 384

        logger.warning("[RAG] Checking Qdrant collection")

        if not self.client.collection_exists(self.collection):
            logger.warning("[RAG] Creating Qdrant collection")

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )

        self._collection_ready = True
        logger.warning("[RAG] Collection ready")

    def index_document(self, document_id: str, filename: str, chunks: list[dict]):
        start = time.perf_counter()

        logger.warning("[RAG] Starting document indexing")

        self._ensure_collection()
        logger.warning("[RAG] Qdrant collection ready")

        texts = [c["text"] for c in chunks]
        logger.warning(
            "[RAG] Generating embeddings for %d chunks",
            len(texts)
        )

        vectors = self._load_embedder().encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        logger.warning(
            "[RAG] Embeddings generated in %.2fs",
            time.perf_counter() - start
        )

        points = []

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                models.PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{document_id}:{idx}"
                        )
                    ),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "page": int(chunk["page"]),
                        "text": chunk["text"],
                        "chunk_index": idx,
                    }
                )
            )

        logger.warning(
            "[RAG] Prepared %d Qdrant points",
            len(points)
        )

        if points:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True
            )

        logger.warning(
            "[RAG] Indexing completed in %.2fs",
            time.perf_counter() - start
        )

    def retrieve(self, document_id: str, query: str, limit: int = 5) -> list[str]:
        self._ensure_collection()
        vector = self._load_embedder().encode(query, normalize_embeddings=True).tolist()
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=limit,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id)
            )]), with_payload=True
        ).points
        return [f"[Page {h.payload['page']}] {h.payload['text']}" for h in hits if h.payload]

    def generate(self, prompt: str) -> str:
    	api_key = os.getenv("GROQ_API_KEY", "").strip()

    	if not api_key:
        	raise RuntimeError(
            	"GROQ_API_KEY is missing. Add it to your backend environment."
        	)

    	llm = ChatGroq(
        	model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        	api_key=api_key,
        	temperature=0.1,
    	)

    	messages = [
        	SystemMessage(
            	content=(
                	"You are LegalLens AI. Ground every factual claim in "
                	"supplied excerpts. Never invent citations. If evidence "
                	"is insufficient, say so. You provide educational "
                	"information, not legal advice."
            	)
        	),
        	HumanMessage(content=prompt),
    	]

    	response = llm.invoke(messages)
    	return str(response.content)

    def answer(self, document_id: str, question: str) -> dict:
        excerpts = self.retrieve(document_id, question, limit=5)
        if not excerpts:
            return {"answer": "I could not find relevant text in this document. Please try a more specific question.", "sources": []}
        prompt = (
            "Answer the user's question using only the excerpts below. Cite factual claims using exactly the page labels "
            "shown, e.g. [Page 2]. Never invent page numbers or facts. If the excerpts do not contain the answer, say "
            "the document does not specify it in the retrieved passages. Be concise.\n\n"
            f"QUESTION: {question}\n\nEXCERPTS:\n" + "\n\n".join(excerpts)
        )
        return {"answer": self.generate(prompt), "sources": excerpts}

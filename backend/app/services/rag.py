import os
import uuid
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
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
            self.embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self.embedder

    def _ensure_collection(self):
        if self._collection_ready:
            return
        dim = self._load_embedder().get_sentence_embedding_dimension()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
        self._collection_ready = True

    def index_document(self, document_id: str, filename: str, chunks: list[dict]):
        self._ensure_collection()
        texts = [c["text"] for c in chunks]
        vectors = self._load_embedder().encode(texts, normalize_embeddings=True).tolist()
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{idx}")),
                vector=vector,
                payload={"document_id": document_id, "filename": filename,
                         "page": int(chunk["page"]), "text": chunk["text"], "chunk_index": idx}
            ))
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

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

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any
import logging
import time

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, HTTPException, UploadFile
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.store import DocumentStore
from app.services.ingestion import extract_pdf_pages, chunk_pages
from app.services.rag import RAGService
from app.services.agent import LegalAgent

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LegalLens AI API", version="0.1.0",
              description="Educational legal-document RAG capstone API.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

store = DocumentStore(os.getenv("DATABASE_PATH", str(DATA_DIR / "legallens.db")))
rag = RAGService()
legal_agent = LegalAgent(rag)

class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=3, max_length=3000)


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=10, max_length=12000)

class AgentRequest(BaseModel):
    document_id: str
    request: str = Field(min_length=3, max_length=3000)


def get_document(document_id: str) -> dict[str, Any]:
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/health")
def health():
    return {"status": "ok", "service": "LegalLens AI API", "version": "0.1.0"}


@app.get("/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.post("/documents/upload")
async def upload_pdf(file: UploadFile = File(...)):
    start = time.perf_counter()

    def log_step(message: str):
        logger.warning(
            "[UPLOAD] %s | elapsed=%.2fs",
            message,
            time.perf_counter() - start,
        )

    log_step("Request received")

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    content = await file.read()
    log_step(f"File read: {len(content)} bytes")

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds 25 MB limit."
        )

    document_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name.replace("\x00", "")
    destination = UPLOAD_DIR / f"{document_id}.pdf"
    destination.write_bytes(content)

    log_step("PDF saved")

    try:
        pages = extract_pdf_pages(destination)
        log_step(f"PDF extracted: {len(pages)} pages")

        if not any(p["text"].strip() for p in pages):
            raise ValueError(
                "No extractable text found. Scanned PDFs need OCR."
            )

        chunks = chunk_pages(pages)
        log_step(f"Chunks created: {len(chunks)}")

        rag.index_document(document_id, safe_name, chunks)
        log_step("Qdrant indexing completed")

        store.add_document(
            document_id,
            safe_name,
            str(destination),
            len(pages),
            len(chunks),
        )
        log_step("Metadata saved")

    except Exception as exc:
        logger.exception("[UPLOAD] PDF processing failed")
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=f"Could not process PDF: {exc}",
        ) from exc

    return {
        "document_id": document_id,
        "filename": safe_name,
        "pages": len(pages),
        "chunks": len(chunks),
        "status": "processed",
    }


@app.post("/chat")
def chat(request: ChatRequest):
    doc = get_document(request.document_id)
    result = rag.answer(request.document_id, request.question)
    store.add_event(request.document_id, "question", request.question)
    return {"document_id": doc["id"], **result}


@app.post("/documents/{document_id}/summary")
def summary(document_id: str):
    doc = get_document(document_id)
    excerpts = rag.retrieve(document_id, "Summarize the main purpose, parties, obligations, dates, and termination terms.", limit=8)
    if not excerpts:
        raise HTTPException(status_code=404, detail="No indexed passages found.")
    prompt = (
        "Summarize this legal document for educational information. Use only the excerpts. "
        "Structure as: Purpose, Key obligations, Dates/notice periods, Missing or unclear terms. "
        "Cite every factual point with the supplied [Page N] label. If not stated, say not specified. "
        "Do not give definitive legal advice.\n\n" + "\n\n".join(excerpts)
    )
    return {"document_id": doc["id"], "summary": rag.generate(prompt)}


@app.post("/documents/{document_id}/risk-review")
def risk_review(document_id: str):
    doc = get_document(document_id)
    excerpts = rag.retrieve(document_id, "Identify potentially risky, unclear, one-sided, or missing contractual terms.", limit=10)
    if not excerpts:
        raise HTTPException(status_code=404, detail="No indexed passages found.")
    prompt = (
        "Review these contract excerpts for issues a reader may want to examine. Do not assert illegality. "
        "For each item provide: issue, why it may matter, question to ask, and evidence citation [Page N]. "
        "Distinguish missing information from actual risk. If evidence is insufficient, say so. "
        "Educational information only; not legal advice.\n\n" + "\n\n".join(excerpts)
    )
    return {"document_id": doc["id"], "risk_review": rag.generate(prompt)}

@app.post("/agent")
def run_agent(request: AgentRequest):
    doc = get_document(request.document_id)

    try:
        result = legal_agent.run(
            request.document_id,
            request.request,
        )
        store.add_event(
            request.document_id,
            "agent_request",
            request.request,
        )
        return {
            "document_id": doc["id"],
            **result,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow failed: {exc}",
        ) from exc

@app.post("/classify-clause")
def classify_clause(request: ClassifyRequest):
    """Uses optional trained classifier; never fabricates a prediction if weights are absent."""
    try:
        from app.services.classifier import classify
        result = classify(request.text)
        return result
    except FileNotFoundError:
        return {"status": "model_not_trained",
                "message": "Train the classifier using training/train_classifier.py and configure MODEL_DIR.",
                "prediction": None}


@app.get("/evaluation/sample")
def evaluation_sample():
    return {
        "note": "Starter test cases are illustrative; run them against your chosen sample contract and manually verify expected answers.",
        "cases": [
            {"question": "What is the notice period?", "expected": "Answer with a cited notice period if present; otherwise say not specified."},
            {"question": "What is the governing law?", "expected": "Cite the governing-law clause or say not specified."},
            {"question": "How long does confidentiality continue?", "expected": "Cite the duration if present; otherwise say not specified."},
        ],
    }

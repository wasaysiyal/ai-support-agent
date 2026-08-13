"""
main.py — FastAPI backend for the AI Support Agent.

Endpoints:
  POST /upload             -> upload a company PDF, ingest into RAG
  GET  /documents            -> list ingested document sources
  POST /generate-reply        -> run the agent, creates a pending ticket
  GET  /tickets                 -> list all tickets (most recent first)
  POST /tickets/{id}/approve    -> mark a ticket approved (simulates "sent")
  POST /tickets/{id}/reject       -> mark a ticket rejected
  GET  /health                     -> simple health check

Run with:
  uvicorn main:app --reload --port 8000
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import ingest_pdf, list_ingested_sources
from agent import generate_reply
from tools import create_support_ticket, list_tickets, update_ticket_status

app = FastAPI(title="AI Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a company PDF (FAQ, Returns policy, Shipping rules, etc.)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in the MVP.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = ingest_pdf(save_path, source_name=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    return {
        "filename": file.filename,
        "chunks_indexed": num_chunks,
        "message": f"'{file.filename}' indexed successfully.",
    }


@app.get("/documents")
def get_documents():
    """List all document sources currently in the knowledge base."""
    return {"sources": list_ingested_sources()}


class ReplyRequest(BaseModel):
    question: str


@app.post("/generate-reply")
def generate_reply_endpoint(payload: ReplyRequest):
    """Run the agent (may call tools) and persist the draft as a pending ticket."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = generate_reply(payload.question)

    ticket = create_support_ticket(
        customer_question=payload.question,
        ai_reply=result.get("reply", ""),
        confidence=result.get("confidence", 0),
        sources=result.get("sources", []),
        tools_used=result.get("tools_used", []),
    )

    result["ticket_id"] = ticket["id"]
    return result


@app.get("/tickets")
def get_tickets():
    """List all tickets, most recent first."""
    return {"tickets": list_tickets()}


@app.post("/tickets/{ticket_id}/approve")
def approve_ticket(ticket_id: str):
    ticket = update_ticket_status(ticket_id, "approved")
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket


@app.post("/tickets/{ticket_id}/reject")
def reject_ticket(ticket_id: str):
    ticket = update_ticket_status(ticket_id, "rejected")
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket

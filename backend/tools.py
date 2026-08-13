"""
tools.py — Concrete "actions" the agent can take, beyond just searching docs.

This is what turns the system from a RAG chatbot into an agent: the LLM is
given a menu of tools and decides for itself which one(s) to call based on
what the customer actually asked, rather than always running the same fixed
RAG-only pipeline.

Includes:
- get_order_status(order_id)   -> mocked order database lookup
- create_support_ticket(...)    -> persists a ticket to a local JSON file
- list/update ticket helpers used by the API layer for Approve/Reject
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

TICKETS_FILE = os.path.join(os.path.dirname(__file__), "tickets.json")

# --- Mock order database (stand-in for a real order/CRM system) ---
MOCK_ORDERS = {
    "4532": {"status": "Shipped", "eta": "tomorrow", "carrier": "DHL"},
    "1001": {"status": "Processing", "eta": "3-5 business days", "carrier": "-"},
    "2200": {"status": "Delivered", "eta": "already delivered", "carrier": "DHL"},
    "9999": {"status": "Cancelled", "eta": "-", "carrier": "-"},
}


def get_order_status(order_id: str) -> dict:
    """Look up the status of a customer order by ID."""
    order_id = str(order_id).strip().lstrip("#")
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"found": False, "message": f"No order found with ID {order_id}."}
    return {"found": True, "order_id": order_id, **order}


# --- Ticket persistence (simple JSON file — no DB server needed for MVP) ---

def _load_tickets() -> list:
    if not os.path.exists(TICKETS_FILE):
        return []
    with open(TICKETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tickets(tickets: list) -> None:
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)


def create_support_ticket(
    customer_question: str,
    ai_reply: str,
    confidence: int,
    sources: Optional[list] = None,
    tools_used: Optional[list] = None,
) -> dict:
    """Persist a new ticket with the AI's draft reply, status = pending."""
    tickets = _load_tickets()
    ticket = {
        "id": str(uuid.uuid4())[:8],
        "customer_question": customer_question,
        "ai_reply": ai_reply,
        "confidence": confidence,
        "sources": sources or [],
        "tools_used": tools_used or [],
        "status": "pending",  # pending -> approved | rejected
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tickets.append(ticket)
    _save_tickets(tickets)
    return ticket


def list_tickets() -> list:
    """Return all tickets, most recent first."""
    return list(reversed(_load_tickets()))


def update_ticket_status(ticket_id: str, status: str) -> Optional[dict]:
    """Set a ticket's status to 'approved' or 'rejected'. Returns the updated ticket, or None if not found."""
    tickets = _load_tickets()
    for t in tickets:
        if t["id"] == ticket_id:
            t["status"] = status
            t["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_tickets(tickets)
            return t
    return None


# --- Tool schema definitions, passed to the LLM so it knows what's available ---
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_company_docs",
            "description": "Search the company's uploaded knowledge base (FAQ, returns policy, shipping rules, etc.) for information relevant to the customer's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query, based on the customer's question."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status, ETA, and carrier for a specific customer order, given an order ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID mentioned by the customer, e.g. '4532'."}
                },
                "required": ["order_id"],
            },
        },
    },
]

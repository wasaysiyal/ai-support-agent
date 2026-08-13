# AI Customer Support Agent

A locally-run, tool-calling AI agent that reads customer support questions, decides for itself which internal tool to use (order lookup vs. knowledge-base search), retrieves grounded information, and drafts a reply for human review before it's sent. Built as a full-stack demo of agentic AI patterns — not a fixed RAG pipeline, but a genuine reasoning loop with tool selection, self-correction, and human-in-the-loop approval.

100% free and local: no OpenAI key, no cloud API cost. Runs entirely on your machine via Ollama.

---

## Table of Contents

- [What this is (and isn't)](#what-this-is-and-isnt)
- [Architecture](#architecture)
- [Tech stack, and why each piece was chosen](#tech-stack-and-why-each-piece-was-chosen)
- [How the agent loop actually works](#how-the-agent-loop-actually-works)
- [Engineering challenges found and fixed](#engineering-challenges-found-and-fixed)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running it](#running-it)
- [Testing it](#testing-it)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Pushing this to GitHub](#pushing-this-to-github)

---

## What this is (and isn't)

**This is an agent**, not a chatbot with search bolted on. The distinction matters:

| A fixed RAG pipeline (what this started as) | A tool-calling agent (what this became) |
|---|---|
| Every question runs the same hardcoded path: search docs → generate answer | The LLM is given a menu of tools and *decides* which one(s) fit the question |
| One shot, no branching | Multi-step loop: reason → act → observe → reason again |
| Can only do one thing (semantic search) | Can look up live order data, search documents, or both, based on what's actually being asked |

The agent has two tools available:
- `search_company_docs(query)` — semantic search over uploaded company PDFs (FAQ, policies, etc.)
- `get_order_status(order_id)` — looks up a specific order (mocked data for this demo, designed to be swapped for a real CRM/order system)

It decides which to call, executes it, evaluates the result, and either calls another tool or gives a final answer — with a confidence score attached so a human reviewer knows how much to trust it.

---

## Architecture

```
                    Customer question (pasted in dashboard)
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Deterministic pre-fetch  │   (regex-based order ID
                 │  (order number detection)  │   detection — see below)
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Agent loop (LLM)     │◄──┐
                 │   decides which tool(s)    │   │
                 │        to call             │   │ tool result fed
                 └────────────┬────────────┘   │ back into context
                              │                  │
                 ┌────────────┴────────────┐    │
                 ▼                          ▼    │
      search_company_docs()      get_order_status()
                 │                          │
                 ▼                          │
        ┌─────────────────┐                 │
        │  ChromaDB       │                 │
        │ (vector search) │                 │
        └────────┬────────┘                 │
                 │                          │
                 └────────────┬─────────────┘
                              │
                              ▼
                    Final JSON answer
              { reply, confidence, reasoning }
                              │
                              ▼
                  Ticket saved (pending)
                              │
                              ▼
              Human reviews in dashboard
                    Approve ──────► marked sent (ticket updated)
                    Reject  ──────► marked rejected (ticket updated)
```

---

## Tech stack, and why each piece was chosen

### Backend — Python + FastAPI
FastAPI was chosen over Flask/Django for three reasons: async support out of the box (useful once this talks to real external APIs like Gmail), automatic interactive API docs at `/docs` (genuinely useful mid-development), and Pydantic-based request validation with almost no boilerplate.

### LLM — Ollama running Llama 3.1 (8B), locally
No OpenAI/Anthropic API key required. Ollama exposes a local HTTP server (`localhost:11434`) with a chat API that supports **tool/function calling**, which is the feature this whole project depends on. Llama 3.1 (8B) was chosen over the smaller Llama 3.2 (3B) after **head-to-head testing** — see [Engineering challenges](#engineering-challenges-found-and-fixed) below — because it was measurably more reliable at both (a) not hallucinating and (b) correctly formatting tool calls.

Running locally also means customer data never leaves the machine — for a project ultimately pitched at compliance-conscious clients, that's a legitimate feature, not just a cost-saving shortcut.

### Embeddings — sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), local
Turns document chunks and questions into vectors for semantic search. Chosen for being free, fast enough on CPU, and multilingual (useful since this project was originally scoped for German-market customers, and multilingual robustness was tested along the way).

### Vector database — ChromaDB
Stores document embeddings persistently on disk (`backend/vectorstore/`). Chosen over Pinecone/Weaviate for being embeddable — no server to run, no account, no cost — which fits a demo/prototype stage. Configured explicitly with `hnsw:space: cosine` after discovering the default distance metric produced meaningless similarity scores (see below).

### PDF parsing — PyPDF
Extracts raw text from uploaded company documents (FAQs, policies, shipping rules) before chunking and embedding.

### Frontend — plain HTML/JS + Tailwind CSS (CLI build, not CDN)
No React/Next.js framework overhead for what is currently a single-page operational dashboard. Tailwind is compiled locally via the Tailwind CLI (`npm run build`) rather than loaded from a CDN, since the CDN build is explicitly not recommended for production use (larger bundle, no purging of unused classes). The compiled stylesheet is ~10KB.

### Ticket persistence — JSON file (`tickets.json`)
Every generated reply is saved as a ticket with a `pending` status; Approve/Reject actions update that status via real API calls (`POST /tickets/{id}/approve`, `/reject`). A JSON file was chosen deliberately over a full database for this stage — it's enough to prove the human-in-the-loop workflow is real (not just a UI mockup that shows a fake success message), without the setup overhead of PostgreSQL, which is intentionally deferred (see [Roadmap](#roadmap)).

---

## How the agent loop actually works

1. **Deterministic order-ID pre-fetch.** Before the LLM sees anything, a regex scans the customer's message for patterns like `order 4532`, `item #4532`, `#4532`. If found, `get_order_status()` is called immediately in code and the result is handed to the model as known context. This was added after testing revealed the LLM would sometimes only recognize *one* of two things it needed to do in a compound question (e.g. "where's my order AND can I return it if damaged" — it would answer the return question but silently drop the order lookup). Pre-fetching removes that failure mode for order lookups entirely, rather than just hoping better prompting fixes it.

2. **LLM reasoning turn.** The question (plus any pre-fetched context) and the two tool definitions are sent to Ollama's `/api/chat` endpoint with `tools` specified.

3. **Tool call or final answer.** The model either requests a tool call (or several) or returns a final JSON answer directly.

4. **Tool execution + de-duplication.** If a tool is requested, it's executed, and the exact `(tool, arguments)` pair is remembered — if the model tries to call the identical tool with identical arguments again, it's told to stop and answer instead of executing it again. This caps the loop at `MAX_TOOL_ITERATIONS = 3` as a hard safety limit.

5. **Malformed tool call recovery.** During testing, the smaller local model was observed to sometimes wrap its actual final answer *inside* a fake tool call (passing `reply`/`confidence`/`reasoning` as if they were tool arguments) instead of returning plain content. The loop detects this specific pattern — a tool call missing its real required argument but containing final-answer-shaped fields — and recovers the answer directly instead of executing the tool with garbage arguments and losing a correct result.

6. **Final answer parsing.** The model's final response is parsed as JSON (`{ reply, confidence, reasoning }`). If parsing fails, a safe fallback message is returned rather than showing raw/broken text to the customer.

7. **Ticket creation.** The result is persisted as a new ticket with `status: pending`, along with which tools were used and which document sources were cited — visible for a human reviewer before anything is "sent."

---

## Engineering challenges found and fixed

This section exists because *finding and fixing these was the actual work* — most of them wouldn't show up just from reading the final code.

**1. ChromaDB relevance scores were meaningless.**
The initial implementation assumed cosine similarity (`1 - distance`) but ChromaDB's default distance metric isn't cosine unless explicitly configured. This produced nonsensical negative "relevance" scores (e.g. `-17.46`) that were undebuggable. Fixed by explicitly setting `metadata={"hnsw:space": "cosine"}` on collection creation and correcting the score formula.

**2. Duplicate chunks silently accumulated.**
Re-uploading the same PDF during testing kept *adding* new embeddings instead of replacing old ones, so the vector database accumulated duplicate chunks that crowded out genuinely different content in search results. Fixed by deleting any existing chunks for a filename before re-ingesting it.

**3. Character-based chunking split answers from their own questions.**
The original chunking sliced text every 800 characters regardless of sentence or paragraph boundaries — occasionally cutting a chunk to start mid-word, disconnected from the FAQ question it was answering, which measurably hurt retrieval ranking. Fixed by switching to paragraph/question-boundary-aware chunking.

**4. Hallucination on out-of-scope questions.**
Asked about a topic the uploaded documents didn't cover (e.g. return policy, when the only uploaded document was an unrelated OHS safety FAQ), the smaller model (Llama 3.2) sometimes fabricated a plausible-sounding but entirely invented policy instead of admitting it didn't know. Mitigated with explicit strict-grounding rules in the system prompt, and ultimately resolved by switching the default model to Llama 3.1, which was empirically far more reliable at this in side-by-side testing with identical prompts.

**5. Malformed tool calls silently corrupted correct answers.**
The most subtle bug found: in one case, the model's *first* tool call correctly retrieved the right order status, but a second, malformed tool call overwrote that correct result with an incorrect "order not found" message — because the code executed the malformed call at face value instead of recognizing it as a disguised final answer. This was diagnosed by adding structured debug logging of every tool call's arguments and raw results, not by guessing.

**6. Multi-part questions only got half-answered.**
A compound question referencing both an order number and a separate policy question would sometimes only trigger one of the two needed tool calls. Fixed with the deterministic order-ID pre-fetch described above, verified with unit-style regex tests before deployment.

---

## Project structure

```
ai-support-agent/
├── backend/
│   ├── main.py           # FastAPI app: upload, generate-reply, tickets endpoints
│   ├── agent.py           # The agent loop: tool selection, execution, recovery
│   ├── rag.py              # PDF parsing, chunking, embeddings, ChromaDB search
│   ├── tools.py             # Tool definitions, mock order DB, ticket persistence
│   ├── requirements.txt
│   ├── uploads/               # Uploaded PDFs land here (gitignored)
│   ├── vectorstore/             # ChromaDB data (gitignored)
│   └── tickets.json              # Ticket history (gitignored)
├── frontend/
│   ├── index.html          # Single-page dashboard (upload, ask, approve/reject)
│   ├── package.json
│   ├── tailwind.config.js
│   └── src/
│       └── input.css         # Tailwind source
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js (for the Tailwind build step)
- [Ollama](https://ollama.com/download)

### 1. Clone and enter the project
```bash
git clone https://github.com/<your-username>/ai-support-agent.git
cd ai-support-agent
```

### 2. Pull the LLM
```bash
ollama pull llama3.1
```

### 3. Backend setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Frontend setup
```bash
cd ../frontend
npm install
npm run build
```

---

## Running it

You need three things running:

1. **Ollama** — usually runs automatically in the background after install. Verify at `http://localhost:11434`.
2. **Backend**:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn main:app --reload --port 8000
   ```
3. **Frontend** — open `frontend/index.html` directly in a browser.

---

## Testing it

1. Upload a company PDF (FAQ, policy document, etc.) via the dashboard.
2. Ask a question the PDF covers → expect a grounded answer with high confidence and the source PDF cited.
3. Ask about a specific order (e.g. `Where is my order 4532?`) → expect the agent to call `get_order_status` (mock data: orders `4532`, `1001`, `2200`, `9999` exist; anything else returns "not found").
4. Ask a compound question (e.g. `I ordered item 4532, can I return it if it arrives damaged?`) → expect both the order lookup (via deterministic pre-fetch) and a document search for the policy question.
5. Ask something the documents don't cover → expect an honest "I don't have that information" rather than an invented answer.
6. Click Approve or Reject → confirm the ticket status updates by checking `http://localhost:8000/tickets`.

---

## Known limitations

- Mock order database (`tools.py`) — not connected to a real CRM/order system
- No real email sending — Approve marks a ticket as sent, but doesn't dispatch anything
- No authentication on any API endpoint
- CORS is wide open (`allow_origins=["*"]`) — fine for local dev, not for public deployment
- Ticket storage is a JSON file, not a production database — fine for single-user testing, not for concurrent real traffic
- Local LLM inference is CPU-bound and noticeably slower than a cloud API call
- Occasional malformed tool calls from the LLM are caught and recovered, but represent an underlying reliability ceiling of local models compared to larger hosted ones

## Roadmap

- [ ] Real Gmail/Outlook API integration (replace the "paste an email" simulation)
- [ ] Real CRM integration (HubSpot/Salesforce/SAP) in place of the mock order database
- [ ] PostgreSQL for ticket storage, with multi-tenant company accounts
- [ ] Authentication and rate limiting on the API
- [ ] Automated test suite (currently validated via manual, logged test sessions)
- [ ] Optional cloud LLM fallback (OpenAI/Claude API) for lower-latency or higher-reliability scenarios

---


### 5. Make your repo look professional
- Add a short 1–2 sentence description and topics/tags (`ai-agent`, `rag`, `llm`, `fastapi`, `ollama`) in the GitHub repo settings sidebar
- Consider adding a screenshot or short GIF of the dashboard to the top of this README (drag-and-drop an image into the GitHub web editor to get a hosted URL, then reference it with `![dashboard](url)`)
- Add a `LICENSE` file (MIT is a common permissive default for portfolio projects) via GitHub's "Add file" button

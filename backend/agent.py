"""
agent.py — The AI support agent, now with real tool-calling (agentic loop).

Unlike a fixed RAG pipeline, this agent is given a menu of tools and DECIDES
for itself which one(s) to call based on the customer's question:

  - search_company_docs(query)   -> semantic search over uploaded PDFs
  - get_order_status(order_id)   -> look up a specific order

The loop:
  1. Send the question + available tools to the LLM
  2. If the LLM requests a tool call, execute it and feed the result back
  3. Repeat (up to a safety cap) until the LLM gives a final answer
  4. Parse the final answer into { reply, confidence, reasoning }

Requires Ollama running locally with a tool-calling-capable model, e.g.:
  ollama pull llama3.1
"""

import json
import requests
from rag import search_company_docs as _search_company_docs
from tools import get_order_status as _get_order_status, TOOL_DEFINITIONS

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"
MAX_TOOL_ITERATIONS = 3  # safety cap so the agent can't loop forever

SYSTEM_PROMPT = """You are a customer support agent for a company.

You have tools available:
- search_company_docs: search the company's FAQ/policy documents
- get_order_status: look up a specific order by ID

Decide which tool(s) you need based on the customer's question. If the
question is about a specific order (e.g. mentions an order number or asks
"where is my order"), use get_order_status. If it's about policy, pricing,
returns, shipping rules, or general questions, use search_company_docs. You
may use both if needed.

STRICT RULES:
1. Call each tool AT MOST ONCE per question. Never call the same tool with
   the same arguments twice — you already have the result from the first call.
2. As soon as a tool returns a result, use it to answer. Do not call more
   tools than necessary.
3. You may ONLY state facts, numbers, dates, policies, or statuses that
   appear explicitly in a tool's returned results. If the tool results do
   not explicitly mention the topic the customer asked about (e.g. no
   mention of "return" or "refund" anywhere in the text), you MUST say you
   don't have that information — do NOT describe any policy, timeframe, or
   procedure that isn't literally present in the tool output text.
4. If you are unsure whether the tool results actually answer the question,
   treat it as NOT answered and be honest about that, with a low confidence
   score.
5. If the customer's message contains MULTIPLE distinct questions or topics
   (e.g. asks about an order's status AND about a return/refund policy),
   address ALL of them in your reply. Order information may already be
   provided to you as a NOTE in the conversation — if so, still call
   search_company_docs for any separate policy question in the same message.
   Do not answer only part of a multi-part question.

Once you have everything you need, respond with ONLY a JSON object — no text,
reasoning, or explanation before or after it. Your entire response must start
with "{" and end with "}", in this exact shape:
{
  "reply": "<the full reply in English, professional and polite, signed 'Best regards,\\nSupport Team'>",
  "confidence": <integer 0-100>,
  "reasoning": "<one short sentence in English explaining the confidence score>"
}

Confidence guidance:
- 90-100: tool results directly and fully answer the question
- 60-89: partial information / some inference required
- 0-59: tool results are missing or irrelevant to the question
"""


def _run_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call requested by the LLM to the actual Python function."""
    if name == "search_company_docs":
        hits = _search_company_docs(args.get("query", ""))
        return {
            "results": [{"text": h["text"], "source": h["source"]} for h in hits]
        }
    elif name == "get_order_status":
        return _get_order_status(args.get("order_id", ""))
    else:
        return {"error": f"Unknown tool: {name}"}


def _extract_json(raw: str) -> dict:
    """Local models sometimes wrap JSON in extra text/markdown — extract it defensively."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _call_ollama(messages: list) -> dict:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


import re


def _extract_order_ids(text: str) -> list:
    """
    Deterministically find order numbers mentioned in the customer's message,
    e.g. "order 4532", "order #4532", "#4532". Using regex here instead of
    relying purely on the LLM to notice and call the right tool makes order
    lookups reliable even when the question also asks about something else
    (e.g. a policy question) and the model might otherwise only call one tool.
    """
    pattern = r"(?:order\s*#?|item\s*#?|#)\s*(\d{3,8})\b"
    return list(dict.fromkeys(re.findall(pattern, text, flags=re.IGNORECASE)))


def generate_reply(customer_question: str, company_id: str = "default") -> dict:
    """
    Main agent entry point — runs a real tool-calling loop.
    Returns: { reply, confidence, reasoning, sources, tools_used }
    """
    tools_used = []
    sources_seen = set()

    # --- Deterministic pre-fetch: if the customer mentions an order number,
    # look it up ourselves up front and hand the result to the model as
    # context, rather than hoping the model decides to call get_order_status.
    # This fixes the case where a question mixes an order reference with a
    # separate policy question — the model was found to sometimes only call
    # ONE of the two tools it actually needed. ---
    prefetch_note = ""
    order_ids = _extract_order_ids(customer_question)
    if order_ids:
        prefetched = {}
        for oid in order_ids:
            prefetched[oid] = _get_order_status(oid)
            tools_used.append("get_order_status")
        prefetch_note = (
            "\n\nNOTE: The system already looked up the following order(s) "
            "mentioned in the customer's message — use this data directly, "
            "you do not need to call get_order_status for these:\n"
            f"{json.dumps(prefetched, ensure_ascii=False)}"
        )
        print(f"    [prefetch] order lookup for {order_ids}: {prefetched}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Customer question:\n"""{customer_question}"""{prefetch_note}'},
    ]

    called_signatures = set()  # (tool_name, args_json) already executed — prevents loops

    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            result = _call_ollama(messages)
            message = result["message"]

            tool_calls = message.get("tool_calls")

            print(f"\n[agent] iteration {iteration + 1} | tool_calls: "
                  f"{[tc['function']['name'] for tc in tool_calls] if tool_calls else 'none (final answer)'}")

            if not tool_calls:
                # No more tools requested -> this is the final answer.
                raw = message.get("content", "")
                print(f"    RAW FINAL CONTENT: {raw}")  # DEBUG (full, untruncated)
                try:
                    parsed = _extract_json(raw)
                    if not parsed.get("reply"):
                        raise ValueError("Missing reply field")
                except (json.JSONDecodeError, ValueError, AttributeError):
                    parsed = {
                        "reply": "We're sorry, we couldn't generate a clear answer. "
                                 "A team member will follow up with you shortly.",
                        "confidence": 30,
                        "reasoning": "Model output could not be parsed as valid JSON.",
                    }
                parsed["sources"] = sorted(sources_seen)
                parsed["tools_used"] = tools_used
                return parsed

            # The model wants to call one or more tools — execute each,
            # but skip any exact repeat of a (tool, args) pair already run,
            # to prevent infinite loops on unreliable smaller models.
            messages.append(message)

            REQUIRED_ARG = {"search_company_docs": "query", "get_order_status": "order_id"}

            for call in tool_calls:
                fn_name = call["function"]["name"]
                fn_args = call["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                # --- Detect a "disguised final answer": smaller local models
                # sometimes wrap their actual final JSON answer inside a fake
                # tool call instead of returning plain content. If the args
                # are missing the tool's real required parameter but look
                # like a final-answer payload (reply/confidence present),
                # treat it AS the final answer instead of executing garbage. ---
                required_key = REQUIRED_ARG.get(fn_name)
                looks_like_final_answer = (
                    required_key and required_key not in fn_args
                    and "reply" in fn_args and "confidence" in fn_args
                )
                if looks_like_final_answer:
                    print(f"    !! Detected disguised final answer inside a "
                          f"'{fn_name}' tool call — recovering it directly "
                          f"instead of executing the tool with bad arguments.")
                    fn_args["sources"] = sorted(sources_seen)
                    fn_args["tools_used"] = tools_used
                    return fn_args
                # --- END detection ---

                signature = (fn_name, json.dumps(fn_args, sort_keys=True))

                if signature in called_signatures:
                    tool_result = {
                        "note": "You already called this tool with these exact "
                                "arguments. The result was already provided above "
                                "— use it. Do not call this again; give your final answer now."
                    }
                else:
                    tool_result = _run_tool(fn_name, fn_args)
                    called_signatures.add(signature)
                    tools_used.append(fn_name)
                    if fn_name == "search_company_docs":
                        for r in tool_result.get("results", []):
                            sources_seen.add(r["source"])

                # --- DEBUG: show exactly what args were sent and what came back ---
                print(f"    -> {fn_name}({fn_args})")
                print(f"    <- {json.dumps(tool_result, ensure_ascii=False)[:500]}")
                # --- END DEBUG ---

                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Safety cap hit without a clean final answer. Instead of giving up,
        # make one last call with tools disabled, forcing the model to
        # answer using whatever it already gathered.
        messages.append({
            "role": "user",
            "content": "Stop calling tools. Give your FINAL answer now, as the "
                       "JSON object described earlier, using only the "
                       "information already gathered above."
        })
        final = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=300,
        )
        final.raise_for_status()
        raw = final.json()["message"].get("content", "")
        try:
            parsed = _extract_json(raw)
            if not parsed.get("reply"):
                raise ValueError("Missing reply field")
        except (json.JSONDecodeError, ValueError, AttributeError):
            parsed = {
                "reply": "We're sorry, we couldn't fully process your request. "
                         "A team member will get back to you shortly.",
                "confidence": 0,
                "reasoning": "Agent exceeded max tool iterations without a parsable final answer.",
            }
        parsed["sources"] = sorted(sources_seen)
        parsed["tools_used"] = tools_used
        return parsed

    except requests.exceptions.ConnectionError:
        return {
            "reply": "(Error: could not reach Ollama at localhost:11434. "
                      "Make sure Ollama is running and you've pulled the model: `ollama pull llama3.1`.)",
            "confidence": 0,
            "reasoning": "Ollama connection failed.",
            "sources": [],
            "tools_used": [],
        }
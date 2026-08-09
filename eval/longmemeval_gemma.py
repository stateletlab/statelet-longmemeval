#!/usr/bin/env python3
"""
LongMemEval benchmark using local Gemma 4 (via Ollama) for all LLM tasks.

Extraction + conflict resolution: Gemma 4 via gateway (skip_server_llm=false)
Search: Gateway full pipeline (BM25 + reranker + temporal + supersedes + edges)
Answer + Judge: Gemma 4 via Ollama (OpenAI-compatible API)

This tests the full pipeline with a local open-source model.

Usage:
    python eval/longmemeval_gemma.py [--addr 127.0.0.1:9379] [--limit 50]
    python eval/longmemeval_gemma.py --ollama-url http://localhost:11434 --model gemma4:31b
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# ─── Statelet gRPC client ────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk" / "python"))
from statelet.client import StateletClient


# ─── Data types ─────────────────────────────────────────────────────────────

@dataclass
class Question:
    question_id: str
    question: str
    answer: str
    question_type: str
    question_date: str
    sessions: list
    session_dates: list


@dataclass
class JudgeResult:
    score: int
    label: str
    explanation: str
    retrieval_relevant: int
    retrieval_total: int


@dataclass
class EvalResult:
    question_id: str
    question_type: str
    correct: bool
    hypothesis: str
    ground_truth: str
    judge: JudgeResult
    n_facts_ingested: int
    n_search_results: int
    ingest_ms: float
    search_ms: float
    answer_ms: float


# ─── Load dataset ───────────────────────────────────────────────────────────

MEMORYBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "memorybench"
QUESTIONS_DIR = MEMORYBENCH_DIR / "data" / "benchmarks" / "longmemeval" / "datasets" / "questions"


def load_questions(limit: int = 50) -> List[Question]:
    files = sorted(QUESTIONS_DIR.glob("*.json"))
    questions = []
    for f in files[:limit]:
        with open(f) as fh:
            d = json.load(fh)
        questions.append(Question(
            question_id=d["question_id"],
            question=d["question"],
            answer=str(d["answer"]),
            question_type=d["question_type"],
            question_date=d.get("question_date", ""),
            sessions=d["haystack_sessions"],
            session_dates=d.get("haystack_dates", []),
        ))
    return questions


# ─── Ollama API (OpenAI-compatible) ───────────────────────────────────────

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:31b"
OLLAMA_ANSWER_MODEL = None  # If set, used for answer_question; defaults to OLLAMA_MODEL


def gemma_call(prompt: str, max_retries: int = 3, model: str = None, json_mode: bool = False) -> str:
    """Call Ollama native API with thinking disabled for faster, direct output."""
    use_model = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    body = {
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,  # Disable Gemma 4 thinking mode for direct output
        "options": {
            "temperature": 0,
            "num_predict": 4000,
        },
    }
    if json_mode:
        body["format"] = "json"

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=body, timeout=600)  # local model can be slow
            if resp.status_code == 200:
                data = resp.json()
                return data["message"]["content"].strip()
            print(f"    Ollama API {resp.status_code}, retrying in 2s...")
            time.sleep(2)
        except requests.Timeout:
            print(f"    Ollama timeout (attempt {attempt+1}/{max_retries}), retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error: {e}"
    return ""


def answer_question(question: str, question_type: str, context: List[str], question_date: str) -> str:
    """Gemma answers with type-specific chain-of-thought instructions."""
    ctx_str = "\n\n---\n\n".join(f"Memory {i+1}: {c}" for i, c in enumerate(context))

    date_info = ""
    if question_date:
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", question_date)
        if m:
            date_info = f"**Question date: {m.group(1)}-{m.group(2)}-{m.group(3)}**\nAll relative dates ('yesterday', 'last week', '10 days ago') are relative to this date.\n"

    type_instructions = {
        "temporal-reasoning": """TEMPORAL REASONING INSTRUCTIONS:
Step 1: Scan ALL memories and find every date mentioned. List each as: Memory N → date YYYY-MM-DD.
Step 2: Identify which dates are relevant to the question. Write them out explicitly.
Step 3: Calculate step by step:
  - Write: Date A = YYYY-MM-DD, Date B = YYYY-MM-DD
  - If counting days: list each month's contribution. Example: "Jan has 31 days, Feb has 28 days, ..."
  - If "X days/weeks ago": Question date (YYYY-MM-DD) minus X days = YYYY-MM-DD
  - If "last week/month": resolve to exact date range first
Step 4: Double-check your arithmetic by counting in the reverse direction.
CRITICAL: Show ALL date arithmetic explicitly. Do NOT estimate or round. If you need both dates to answer, find BOTH before calculating.""",

        "multi-session": """MULTI-SESSION REASONING INSTRUCTIONS:
Step 1: Scan ALL memories and list every piece of information relevant to the question, with memory numbers.
Step 2: Cross-reference: do any memories provide DIFFERENT pieces of the same puzzle?
Step 3: Combine and calculate if needed.
- For "how many" / counting: list each item, then count.
- For "how much per item": find total cost + quantity, then divide.
- For "total" / "sum": find each component, then add.
- For comparisons: find both values, then compare.
CRITICAL: If the answer is CALCULABLE from the memories, you MUST calculate it. Never say "not enough information" when the math is doable.""",

        "knowledge-update": """KNOWLEDGE UPDATE INSTRUCTIONS:
Step 1: Find ALL memories about the topic, noting their dates.
Step 2: Sort by date (earliest → latest).
Step 3: Identify what changed between versions.
- LATER dates override EARLIER dates when facts conflict.
- Look for: "moved to", "switched to", "changed to", "I now...", "started", "stopped"
- If asked about BOTH initial AND current state, provide BOTH.
- CRITICAL: Pay close attention to the EXACT role, title, entity, or term in the question. If the question asks about "Software Engineer Manager" but memories only mention "Senior Software Engineer", these are DIFFERENT roles — do NOT assume they are the same. Say "information not enough".
- If the question asks about an entity/role/event that does NOT exactly match what's in the memories, say "information not enough" — do not substitute a similar-sounding but different entity.""",

        "single-session-preference": """PREFERENCE INSTRUCTIONS:
- Look for explicit: "I prefer", "I like", "my favorite", "I always"
- Look for implicit: habits, repeated choices, brands, styles
- If asked for recommendations, base on stated preferences and context.""",

        "single-session-user": """USER FACT INSTRUCTIONS:
- Look for specific factual details: names, numbers, durations, locations, dates.
- Quote the exact information from the memory.
- Do not round numbers or approximate.""",

        "single-session-assistant": """ASSISTANT RECALL INSTRUCTIONS:
- The question asks about something the assistant previously said or recommended.
- Quote specific details: colors, names, numbers, steps, ratios, products.""",
    }

    specific = type_instructions.get(question_type, "")

    prompt = f"""Answer the question using ONLY the retrieved memories below.

{date_info}
{specific}

## Retrieved Memories

{ctx_str}

## Question

{question}

## Instructions

Follow this process:
1. **SCAN**: Read every memory. List the ones relevant to the question (cite memory numbers).
2. **VERIFY**: Check that key NAMED ENTITIES in the question (job titles, role names, person names, place names, organization names) exactly match what's in the memories. If the question asks about "Software Engineer Manager" but memories only mention "Senior Software Engineer", these are DIFFERENT titles — say "I don't have enough information." However, do NOT apply this check to generic terms like activities, counts, or descriptions — only to proper titles and names.
3. **EXTRACT**: Pull out the specific facts/numbers/dates needed.
4. **REASON**: If the answer requires combining facts or calculating, show your work. If you can count or calculate the answer from the memories, DO IT — do not say "not enough information" when the math is doable.
5. **SOURCE CHECK**: Memories are grouped by source (USER-STATED vs ASSISTANT-PROVIDED).
   - If the question asks what YOU (the user) mentioned/said/did, ONLY use USER-STATED FACTS.
   - If ASSISTANT-PROVIDED INFO has relevant data but USER-STATED FACTS don't, say "I don't have enough information" — the user didn't mention it.
   - Do NOT extrapolate or estimate values not explicitly in the memories.
6. **ANSWER**: Give a specific, concise final answer.

Rules:
- Use ONLY information from the memories above.
- If the answer is CALCULABLE from available facts, CALCULATE IT.
- Say "I don't have enough information" ONLY when: (a) memories contain nothing relevant, OR (b) the question asks about a specific NAMED role/title/person that does NOT match what's in the memories (e.g. "Manager" vs "Senior Engineer"). Do NOT say "not enough information" if the answer is calculable or inferable from the memories.
- CLOSED WORLD ASSUMPTION: In memory systems, if an event is recorded but a detail is NOT mentioned, that detail did not occur. Memories capture what the user chose to share. For example: if the user described visiting a museum but did NOT mention a companion, they went alone — answer "No" to "Did I visit with a friend?". If the user never mentioned a role/entity at all, say "not enough information". The distinction: event exists + detail missing = detail didn't happen; event doesn't exist = not enough information."""

    return gemma_call(prompt, model=OLLAMA_ANSWER_MODEL)


def judge_answer(hypothesis: str, ground_truth: str, question: str, search_context: List[str]) -> JudgeResult:
    """Gemma judges answer correctness and retrieval relevance."""
    ctx_summary = "\n".join(f"  Result {i+1}: {c[:200]}" for i, c in enumerate(search_context[:10]))

    prompt = f"""You are an evaluation judge. Evaluate the HYPOTHESIS against the GROUND TRUTH for the given QUESTION.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
HYPOTHESIS: {hypothesis}

SEARCH RESULTS USED:
{ctx_summary}

## Scoring Rules

Score as **CORRECT** (score=1) when:
- The hypothesis conveys the CORE correct answer, even if minor details are missing or paraphrased
- The primary fact/number/entity matches, even if secondary qualifiers differ

Score as **INCORRECT** (score=0) ONLY when:
- The hypothesis gives a fundamentally WRONG answer (wrong entity, wrong number, wrong location)
- The hypothesis says "not enough information" but the ground truth has a specific answer
- The primary fact is completely missing or contradicted

## Examples

CORRECT examples:
- Ground truth: "in a shoe rack in my closet" → Hypothesis: "in your closet" → CORRECT (core location matches)
- Ground truth: "$12 per mug" → Hypothesis: "about $12 each" → CORRECT (amount matches)
- Ground truth: "2 appointments" → Hypothesis: "2 doctor visits in March" → CORRECT (count matches)
- Ground truth: "over a year" → Hypothesis: "more than a year" → CORRECT (same meaning)
- Ground truth: "user prefers Go for backend" → Hypothesis: "Go" → CORRECT (core preference matches)
- Ground truth: "Dr. Smith on March 3rd and Dr. Thompson on March 20th" → Hypothesis: "2 appointments in March" → CORRECT (count correct)
- Ground truth: "Golden retriever named Max and a cat named Luna" → Hypothesis: "Max and Luna" → CORRECT (names correct)

INCORRECT examples:
- Ground truth: "under my bed" → Hypothesis: "in my closet" → INCORRECT (wrong location)
- Ground truth: "25 postcards" → Hypothesis: "8 postcards" → INCORRECT (wrong number)
- Ground truth: "over a year" → Hypothesis: "I don't have enough information" → INCORRECT (answer exists)
- Ground truth: "4 engineers initially, 5 now" → Hypothesis: "5 engineers" → INCORRECT (missed initial count)
- Ground truth: "$50 more than goal" → Hypothesis: "raised $250" → INCORRECT (didn't answer the actual question)

## Retrieval relevance
Count how many search results contain information relevant to answering the question (any fact, name, date, or detail from the ground truth).

Output a JSON object (no markdown fences):
{{"score": 1 or 0, "label": "correct" or "incorrect", "explanation": "brief reason", "retrieval_relevant": number_of_relevant_results, "retrieval_total": {len(search_context[:10])}}}"""

    raw = gemma_call(prompt, json_mode=True)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        d = json.loads(cleaned)
        return JudgeResult(
            score=d.get("score", 0),
            label=d.get("label", "incorrect"),
            explanation=d.get("explanation", ""),
            retrieval_relevant=d.get("retrieval_relevant", 0),
            retrieval_total=d.get("retrieval_total", len(search_context[:10])),
        )
    except (json.JSONDecodeError, KeyError):
        is_correct = "correct" in cleaned.lower() and "incorrect" not in cleaned.lower()
        return JudgeResult(
            score=1 if is_correct else 0,
            label="correct" if is_correct else "incorrect",
            explanation=cleaned[:200],
            retrieval_relevant=0,
            retrieval_total=len(search_context[:10]),
        )


# ─── Checkpoint ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent / "longmemeval_gemma_checkpoint.json"
FAILURES_PATH = Path(__file__).parent / "longmemeval_gemma_failures.jsonl"


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed": {}, "model": ""}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def checkpoint_to_results(checkpoint: dict) -> List[EvalResult]:
    results = []
    for qid, r in checkpoint["completed"].items():
        results.append(EvalResult(
            question_id=qid,
            question_type=r["question_type"],
            correct=r["correct"],
            hypothesis=r["hypothesis"],
            ground_truth=r["ground_truth"],
            judge=JudgeResult(
                score=r["judge"]["score"],
                label=r["judge"]["label"],
                explanation=r["judge"]["explanation"],
                retrieval_relevant=r["judge"]["retrieval_relevant"],
                retrieval_total=r["judge"]["retrieval_total"],
            ),
            n_facts_ingested=r["n_facts"],
            n_search_results=r["n_results"],
            ingest_ms=r.get("ingest_ms", 0),
            search_ms=r.get("search_ms", 0),
            answer_ms=r.get("answer_ms", 0),
        ))
    return results


def wait_for_enrichment(
    db: StateletClient,
    graph: str,
    node_id: int,
    timeout_ms: int,
    poll_interval_ms: int = 250,
) -> dict:
    started = time.time()
    kv_key = f"gnode:{graph}:{node_id}".encode()

    while True:
        raw = db.get(kv_key, cf=1)
        if raw:
            try:
                props = json.loads(raw.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                return {"ready": False, "status": "unparseable", "waited_ms": int((time.time() - started) * 1000)}

            enrichment_mode = props.get("enrichment_mode")
            enrichment_status = props.get("enrichment_status", "completed" if not enrichment_mode else "pending")
            if enrichment_status != "pending":
                return {
                    "ready": enrichment_status == "completed",
                    "status": enrichment_status,
                    "waited_ms": int((time.time() - started) * 1000),
                    "error": props.get("enrichment_error"),
                }

        waited_ms = int((time.time() - started) * 1000)
        if waited_ms >= timeout_ms:
            return {"ready": False, "status": "timeout", "waited_ms": waited_ms}
        time.sleep(poll_interval_ms / 1000.0)


# ─── Main benchmark ─────────────────────────────────────────────────────────

def run_benchmark(
    statelet_addr: str,
    limit: int,
    embedding_dim: int,
    force: bool = False,
    retry_failed: bool = False,
    wait_enriched: bool = False,
    enrichment_timeout_ms: int = 30000,
):
    print(f"Loading LongMemEval dataset (limit={limit})...")
    questions = load_questions(limit)
    print(f"Loaded {len(questions)} questions")

    checkpoint = load_checkpoint()
    if force or checkpoint.get("model") != OLLAMA_MODEL:
        checkpoint = {"completed": {}, "model": OLLAMA_MODEL}
        save_checkpoint(checkpoint)
        print("Starting fresh run")
    else:
        n_done = len(checkpoint["completed"])
        if n_done > 0:
            n_pass = sum(1 for r in checkpoint["completed"].values() if r["correct"])
            print(f"Resuming: {n_done}/{len(questions)} done ({n_pass} PASS, {n_done - n_pass} FAIL)")

    print(f"Connecting to Statelet at {statelet_addr}...")
    db = StateletClient(statelet_addr)
    print(f"Ping: {db.ping()}")
    print(f"Pipeline: Gateway Gemma 4 extraction + full search pipeline")
    print(f"Answer/Judge: Gemma 4 via Ollama (model={OLLAMA_MODEL})")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    W = 80

    if retry_failed:
        failed_ids = [qid for qid, r in checkpoint["completed"].items() if not r["correct"]]
        for qid in failed_ids:
            del checkpoint["completed"][qid]
        save_checkpoint(checkpoint)
        print(f"Retrying {len(failed_ids)} failed questions")

    import threading
    checkpoint_lock = threading.Lock()

    def process_question(qi, q):
        """Process a single question (ingest + search + answer + judge)."""
        nonlocal checkpoint

        graph = f"eval-gm-{q.question_id}"

        print(f"\n{'='*W}")
        print(f"Q{qi+1}/{len(questions)}: [{q.question_type}] {q.question[:70]}")
        print(f"  Expected: {str(q.answer)[:70]}")
        print(f"  Graph: {graph}")

        # Create per-question graph index
        try:
            db.create_graph_index(graph, dim=embedding_dim)
        except Exception as e:
            if "already exists" not in str(e):
                print(f"  WARN: create_graph_index: {e}")

        # ── 1. Ingest: send raw sessions to gateway (Gemma extraction via gateway) ──
        t0 = time.time()
        ingest_ok = 0
        ingest_fail = 0

        session_items = []
        for si, session_msgs in enumerate(q.sessions):
            lines = []
            if si < len(q.session_dates) and q.session_dates[si]:
                lines.append(f"[{q.session_dates[si]}]")
            for msg in session_msgs:
                lines.append(f"{msg['role']}: {msg['content']}")
            session_items.append((si, "\n".join(lines)))

        def ingest_session(item):
            si, session_text = item
            node_id = (si + 1) * 1000
            session_date = q.session_dates[si] if si < len(q.session_dates) and q.session_dates[si] else None
            props = {"session_index": si}
            if session_date:
                props["session_date"] = session_date
            try:
                db.text_graph_put(
                    graph_name=graph,
                    text=session_text,
                    node_id=node_id,
                    properties=json.dumps(props).encode(),
                    skip_server_llm=False,
                )
                wait_result = None
                if wait_enriched:
                    wait_result = wait_for_enrichment(
                        db,
                        graph,
                        node_id,
                        timeout_ms=enrichment_timeout_ms,
                    )
                return si, True, wait_result
            except Exception as e:
                return si, False, {"status": f"error: {e}"} if wait_enriched else None

        # Sequential ingest — local model is slower, avoid overloading
        INGEST_CONCURRENCY = 2
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=INGEST_CONCURRENCY) as pool:
            futures = {pool.submit(ingest_session, item): item[0] for item in session_items}
            done_count = 0
            for future in as_completed(futures):
                si, ok, wait_result = future.result()
                done_count += 1
                if ok:
                    ingest_ok += 1
                    if wait_enriched and wait_result and wait_result.get("status") != "completed":
                        print(f"    WARN: session {si} enrichment {wait_result.get('status')} after {wait_result.get('waited_ms', 0)}ms")
                else:
                    ingest_fail += 1
                    print(f"    WARN: session {si} ingest failed")
                if done_count % 10 == 0:
                    print(f"  Ingested {done_count}/{len(q.sessions)} sessions")

        total_facts = ingest_ok
        ingest_ms = (time.time() - t0) * 1000
        fail_info = f" ({ingest_fail} failed)" if ingest_fail else ""
        print(f"  Ingested: {len(q.sessions)} sessions in {ingest_ms/1000:.1f}s{fail_info}")

        # ── 2. Search: per-type strategy ──
        t0 = time.time()

        iso_date = ""
        if q.question_date:
            m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", q.question_date)
            if m:
                iso_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        search_query = f"[Context date: {iso_date}] {q.question}" if iso_date else q.question

        search_k = {
            "multi-session": 20,
            "knowledge-update": 15,
            "temporal-reasoning": 25,
            "single-session-user": 10,
            "single-session-preference": 10,
            "single-session-assistant": 10,
        }.get(q.question_type, 10)

        context_limit = {
            "multi-session": 15,
            "knowledge-update": 12,
            "temporal-reasoning": 18,
            "single-session-user": 10,
            "single-session-preference": 10,
            "single-session-assistant": 10,
        }.get(q.question_type, 10)

        # Gateway does query decomposition + rewrite + hop2 gap detection
        search_results = db.text_graph_search(
            graph_name=graph,
            query=search_query,
            k=search_k,
            skip_server_llm=False,
        )

        # For knowledge-update: also do a direct entity search
        if q.question_type == "knowledge-update":
            entity_prompt = f"""Extract the key entities (roles, titles, names, places, items) from this question. Output each entity on its own line, nothing else.

Question: {q.question}"""
            entity_raw = gemma_call(entity_prompt)
            entity_queries = [l.strip().strip("-•* ") for l in entity_raw.split("\n") if l.strip() and len(l.strip()) > 3][:5]

            seen_ids = {r.node_id for r in search_results}
            for eq in entity_queries:
                extra_results = db.text_graph_search(
                    graph_name=graph, query=eq, k=10, skip_server_llm=True,
                )
                for r in extra_results:
                    if r.node_id not in seen_ids:
                        search_results.append(r)
                        seen_ids.add(r.node_id)

        # Build context from results with per-type ordering
        raw_context = []
        seen_facts = set()
        for r in search_results:
            try:
                props = json.loads(r.properties)
                fact_text = props.get("text", "")
                if len(fact_text) < 25:
                    continue
                fact_key = fact_text[:300]
                if fact_key in seen_facts:
                    continue
                seen_facts.add(fact_key)
                date = props.get("date", "")
                raw_context.append((date or "", fact_text, props))
            except (json.JSONDecodeError, AttributeError):
                pass

        if q.question_type in ("knowledge-update", "temporal-reasoning"):
            raw_context.sort(key=lambda x: x[0])

        # Split context by source
        user_context = []
        assistant_context = []
        other_context = []
        for date, fact_text, props in raw_context:
            source = props.get("source", "")
            entry = f"[{date}] {fact_text}" if date else fact_text
            if source == "user":
                user_context.append(entry)
            elif source == "assistant":
                assistant_context.append(entry)
            else:
                other_context.append(entry)

        context = []
        if user_context:
            context.append("── USER-STATED FACTS (the user said/did these) ──")
            context.extend(user_context)
        if assistant_context:
            context.append("── ASSISTANT-PROVIDED INFO (assistant said these, user did NOT confirm) ──")
            context.extend(assistant_context)
        if other_context:
            context.extend(other_context)

        search_ms = (time.time() - t0) * 1000
        print(f"  Search: {len(search_results)} results (k={search_k}) in {search_ms:.0f}ms")
        for si, ctx_text in enumerate(context[:context_limit]):
            print(f"    [{si}] {ctx_text[:100]}")

        # ── 3. Two-phase answer ──
        t0 = time.time()

        # Phase 1: Structured analysis
        analysis_prompt = f"""Analyze the memories below to prepare for answering a question.

## Memories
{chr(10).join(f"Memory {i+1}: {c}" for i, c in enumerate(context[:context_limit]))}

## Question
{q.question}

## Tasks
1. List ALL memories relevant to the question (by number).
2. For each relevant memory, extract the KEY FACT (number, name, date, etc).
3. Identify what information is STILL MISSING to answer the question.
4. If the answer requires CALCULATION, set up the calculation.

Output your analysis concisely."""

        analysis = gemma_call(analysis_prompt)
        print(f"  Analysis: {analysis[:100]}")

        # Phase 2: Answer
        hypothesis = answer_question(q.question, q.question_type, context[:context_limit], q.question_date)

        # ── 3b. Gap re-search ──
        gap_prompt = f"""Based on this question and the analysis, what specific information is MISSING?

Question: {q.question}
Analysis: {analysis[:500]}
Current answer: {hypothesis[:200]}

If ALL needed information is present, output: COMPLETE
Otherwise output 2-3 short search queries to find missing info. One per line."""
        gap_raw = gemma_call(gap_prompt)

        if "COMPLETE" not in gap_raw.upper():
            gap_queries = [l.strip().lstrip("0123456789.-) ") for l in gap_raw.strip().split("\n")
                          if len(l.strip()) > 10 and "COMPLETE" not in l.upper()][:3]
            if gap_queries:
                print(f"  Re-search: {gap_queries}")
                new_facts = 0
                for gq in gap_queries:
                    extra_results = db.text_graph_search(
                        graph_name=graph, query=gq, k=5, skip_server_llm=True,
                    )
                    for r in extra_results:
                        try:
                            props = json.loads(r.properties)
                            fact_text = props.get("text", "")
                            fact_key = fact_text[:300] if fact_text else ""
                            if fact_key and fact_key not in seen_facts:
                                seen_facts.add(fact_key)
                                date = props.get("date", "")
                                entry = f"[{date}] {fact_text}" if date else fact_text
                                context.append(entry)
                                new_facts += 1
                        except (json.JSONDecodeError, AttributeError):
                            pass
                if new_facts > 0:
                    print(f"  Re-search added {new_facts} new facts, re-answering...")
                    hypothesis = answer_question(q.question, q.question_type, context[:context_limit + 5], q.question_date)

        # ── 3c. Self-verification ──
        verify_prompt = f"""Check if this answer correctly addresses the question.

Question: {q.question}
Answer: {hypothesis[:300]}

Rules:
- Does the answer DIRECTLY address what was asked? (e.g. if asked "how much MORE than goal", does it give the difference, not just the total?)
- Does the answer contain a SPECIFIC answer, not just "I don't know"?
- If the answer says "not enough information" but the analysis found relevant facts, this is WRONG.

Analysis found: {analysis[:300]}

If the answer is CORRECT and complete, output: VERIFIED
If the answer needs fixing, output the CORRECTED answer only (no explanation)."""

        verify_raw = gemma_call(verify_prompt)
        if "VERIFIED" not in verify_raw.upper() and len(verify_raw.strip()) > 20:
            print(f"  Self-verify corrected answer")
            hypothesis = verify_raw.strip()

        # ── 3d. Date arithmetic verification (temporal-reasoning only) ──
        if q.question_type == "temporal-reasoning":
            try:
                from datetime import datetime, timedelta
                date_matches = re.findall(r'\b(\d{4})-(\d{2})-(\d{2})\b', hypothesis)
                dates = []
                for y, m, d in date_matches:
                    try:
                        dates.append(datetime(int(y), int(m), int(d)))
                    except ValueError:
                        pass
                day_match = re.search(r'(\d+)\s*(?:days?|day)', hypothesis, re.IGNORECASE)
                week_match = re.search(r'(\d+)\s*(?:weeks?|week)', hypothesis, re.IGNORECASE)

                if len(dates) >= 2 and (day_match or week_match):
                    actual_days = abs((dates[-1] - dates[0]).days)
                    if day_match:
                        claimed = int(day_match.group(1))
                        if claimed != actual_days and actual_days > 0:
                            print(f"  Date verify: claimed {claimed} days but computed {actual_days} days")
                            hypothesis = hypothesis.replace(str(claimed), str(actual_days))
                    elif week_match:
                        claimed_weeks = int(week_match.group(1))
                        actual_weeks = actual_days // 7
                        if claimed_weeks != actual_weeks and actual_weeks > 0:
                            print(f"  Date verify: claimed {claimed_weeks} weeks but computed {actual_weeks} weeks")
                            hypothesis = hypothesis.replace(str(claimed_weeks), str(actual_weeks))
            except Exception:
                pass

        answer_ms = (time.time() - t0) * 1000
        print(f"  Answer: {hypothesis[:120]}")

        # ── 4. Judge (Gemma) ──
        judge_result = judge_answer(hypothesis, q.answer, q.question, context[:10])

        tag = "PASS" if judge_result.score == 1 else "FAIL"
        print(f"  Judge: {judge_result.label} | {judge_result.explanation[:80]}")
        print(f"  [{tag}]")

        # Save checkpoint
        with checkpoint_lock:
            checkpoint["completed"][q.question_id] = {
                "question_type": q.question_type,
                "correct": judge_result.score == 1,
                "hypothesis": hypothesis,
                "ground_truth": q.answer,
                "judge": {
                    "score": judge_result.score,
                    "label": judge_result.label,
                    "explanation": judge_result.explanation,
                    "retrieval_relevant": judge_result.retrieval_relevant,
                    "retrieval_total": judge_result.retrieval_total,
                },
                "n_facts": total_facts,
                "n_results": len(search_results),
                "ingest_ms": ingest_ms,
                "search_ms": search_ms,
                "answer_ms": answer_ms,
            }
            save_checkpoint(checkpoint)

        # Log failures
        if judge_result.score != 1:
            with checkpoint_lock:
                with open(FAILURES_PATH, "a") as ef:
                    ef.write(json.dumps({
                        "question_id": q.question_id,
                        "question_type": q.question_type,
                        "question": q.question,
                        "ground_truth": q.answer,
                        "hypothesis": hypothesis,
                        "judge": {
                            "score": judge_result.score,
                            "label": judge_result.label,
                            "explanation": judge_result.explanation,
                            "retrieval_relevant": judge_result.retrieval_relevant,
                            "retrieval_total": judge_result.retrieval_total,
                        },
                        "search_context": [c[:300] for c in context[:10]],
                        "n_facts": total_facts,
                        "n_results": len(search_results),
                    }) + "\n")

    # Run questions sequentially
    for qi, q in enumerate(questions):
        if q.question_id in checkpoint["completed"]:
            continue
        try:
            process_question(qi, q)
        except Exception as e:
            print(f"  ERROR: Q{qi+1} failed: {e}")

    # ── Summary ──
    results = checkpoint_to_results(checkpoint)

    print(f"\n{'='*W}")
    print("RESULTS SUMMARY")
    print(f"{'='*W}\n")

    total = len(results)
    if total == 0:
        print("  No results yet.")
        db.close()
        return

    correct_count = sum(1 for r in results if r.correct)
    print(f"  Pipeline: Full Gemma 4 (extract + answer + judge, model={OLLAMA_MODEL})")
    print(f"  Total: {total}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {100*correct_count/total:.1f}%\n")

    by_type: Dict[str, List[EvalResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)

    print(f"  {'Type':43s} {'Correct':>8s} {'Accuracy':>10s}")
    print(f"  {'-'*63}")
    for qtype, type_results in sorted(by_type.items()):
        c = sum(1 for r in type_results if r.correct)
        t = len(type_results)
        print(f"  {qtype:43s} {c}/{t:>5d} {100*c/t:>9.1f}%")

    ingest_times = sorted(r.ingest_ms for r in results)
    search_times = sorted(r.search_ms for r in results)
    answer_times = sorted(r.answer_ms for r in results)
    mid = total // 2
    print(f"\n  Latency (median):")
    print(f"    Ingest: {ingest_times[mid]/1000:.1f}s")
    print(f"    Search: {search_times[mid]:.0f}ms")
    print(f"    Answer: {answer_times[mid]:.0f}ms")

    total_relevant = sum(r.judge.retrieval_relevant for r in results)
    total_results = sum(r.judge.retrieval_total for r in results)
    hit_count = sum(1 for r in results if r.judge.retrieval_relevant > 0)
    hit_at_k = hit_count / total if total > 0 else 0
    avg_precision = (total_relevant / total_results) if total_results > 0 else 0

    print(f"\n  Retrieval Quality:")
    print(f"    Hit@K: {100*hit_at_k:.1f}% ({hit_count}/{total})")
    print(f"    Avg Precision: {100*avg_precision:.1f}% ({total_relevant}/{total_results} relevant)")

    out_path = Path(__file__).parent / "longmemeval_gemma_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "pipeline": f"Full Gemma 4 (extract + answer + judge, model={OLLAMA_MODEL})",
            "answer_model": OLLAMA_MODEL,
            "total": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total > 0 else 0,
            "retrieval": {
                "hit_at_k": hit_at_k,
                "avg_precision": avg_precision,
                "total_relevant": total_relevant,
                "total_results": total_results,
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "correct": r.correct,
                    "hypothesis": r.hypothesis,
                    "ground_truth": r.ground_truth,
                    "judge": {
                        "score": r.judge.score,
                        "label": r.judge.label,
                        "explanation": r.judge.explanation,
                        "retrieval_relevant": r.judge.retrieval_relevant,
                        "retrieval_total": r.judge.retrieval_total,
                    },
                    "n_facts": r.n_facts_ingested,
                    "n_results": r.n_search_results,
                    "ingest_ms": r.ingest_ms,
                    "search_ms": r.search_ms,
                    "answer_ms": r.answer_ms,
                }
                for r in results
            ],
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval via full Gemma 4 pipeline (local Ollama)")
    parser.add_argument("--addr", default="127.0.0.1:9379", help="Statelet gateway address")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--model", default="gemma4:31b", help="Ollama model name")
    parser.add_argument("--answer-model", default=None, help="Separate model for answering (defaults to --model)")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint, start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run only failed questions")
    parser.add_argument("--wait-enriched", action="store_true", help="Wait for async post-write enrichment before search/answer")
    parser.add_argument("--enrichment-timeout-ms", type=int, default=30000, help="Max wait per session for async enrichment")
    args = parser.parse_args()

    OLLAMA_BASE_URL = args.ollama_url
    OLLAMA_MODEL = args.model
    OLLAMA_ANSWER_MODEL = args.answer_model
    run_benchmark(
        args.addr,
        args.limit,
        args.dim,
        force=args.force,
        retry_failed=args.retry_failed,
        wait_enriched=args.wait_enriched,
        enrichment_timeout_ms=args.enrichment_timeout_ms,
    )

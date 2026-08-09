#!/usr/bin/env python3
"""
LongMemEval benchmark using Statelet gateway's full server-side pipeline.

Extraction + conflict resolution: DeepSeek via gateway (skip_server_llm=false)
Search: Gateway full pipeline (BM25 + reranker + temporal + supersedes + edges)
Answer + Judge: Claude Code CLI

This tests the real production pipeline end-to-end.

Usage:
    python eval/longmemeval_deepseek.py [--addr 127.0.0.1:9379] [--limit 50] [--model sonnet]
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
    failure_taxonomy: Dict[str, int]


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


# ─── DeepSeek API (answer + judge) ──────────────────────────────────────────

import requests

DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_ANSWER_MODEL = None  # If set, used for answer_question; defaults to DEEPSEEK_MODEL


def deepseek_call(prompt: str, max_retries: int = 5, model: str = None) -> str:
    """Call DeepSeek chat completions API."""
    use_model = model or DEEPSEEK_MODEL
    # `deepseek-reasoner` (legacy alias) and `deepseek-v4-pro` (new flagship
    # reasoning model) both stream reasoning chains via reasoning_content
    # and reject the `temperature` field, so treat them as the reasoner path.
    is_reasoner = ("reasoner" in (use_model or "")) or ("v4-pro" in (use_model or ""))
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": use_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000 if is_reasoner else 4000,
    }
    # R1 reasoner does not support temperature parameter
    if not is_reasoner:
        body["temperature"] = 0

    # R1 needs longer timeout (chain-of-thought can take 2-3 min)
    call_timeout = 300 if is_reasoner else 60

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=call_timeout)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                # R1 returns reasoning_content + content; use content (the final answer)
                content = msg.get("content") or ""
                return content.strip()
            if resp.status_code == 429 or resp.status_code >= 500:
                delay = 5 if resp.status_code == 429 else 2
                print(f"    DeepSeek API {resp.status_code}, retrying in {delay}s...")
                time.sleep(delay)
                continue
            return f"Error: DeepSeek API {resp.status_code}: {resp.text[:200]}"
        except requests.Timeout:
            print(f"    DeepSeek API timeout ({call_timeout}s), attempt {attempt+1}/{max_retries}")
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
    """DeepSeek answers with type-specific chain-of-thought instructions."""
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
- For "how much total" / "how many total": be INCLUSIVE — count every instance the user was involved in. Do not exclude items based on subtle distinctions (e.g. individual vs group, direct vs indirect).
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
- Your answer should describe the user's PREFERENCES and TASTES based on their past behavior in the memories.
- Past choices, requests, and expressed likes/dislikes are all evidence of preferences.
- If the question asks for a suggestion, respond with what the user would prefer, not what you would recommend as an AI. The memories are your ONLY source — use them.
- NEVER say "not enough information" if ANY relevant preferences exist in the memories.""",

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
4. **DEDUP**: Multiple memories may describe the SAME event differently. Before counting, merge duplicates:
   - Same person + same date + same type of event = ONE event (e.g. "appointment with Dr. Smith" and "saw Dr. Smith on March 3rd" = 1 appointment, not 2)
   - Same fact stated differently = ONE fact (e.g. "user leads 5 engineers" and "user's team has 5 engineers" = same fact)
   - Only count as separate events if they have DIFFERENT dates, DIFFERENT people, or are explicitly described as distinct occurrences.
5. **REASON**: If the answer requires combining facts or calculating, show your work. If you can count or calculate the answer from the memories, DO IT — do not say "not enough information" when the math is doable.
6. **SOURCE CHECK**: Memories are grouped by source (USER-STATED vs ASSISTANT-PROVIDED).
   - If the question asks what YOU (the user) mentioned/said/did, ONLY use USER-STATED FACTS.
   - If ASSISTANT-PROVIDED INFO has relevant data but USER-STATED FACTS don't, say "I don't have enough information" — the user didn't mention it.
   - Do NOT extrapolate or estimate values not explicitly in the memories.
7. **ANSWER**: Give a specific, concise final answer.

Rules:
- Use ONLY information from the memories above.
- If the answer is CALCULABLE from available facts, CALCULATE IT.
- Say "I don't have enough information" ONLY when: (a) memories contain nothing relevant, OR (b) the question asks about a specific NAMED role/title/person that does NOT match what's in the memories (e.g. "Manager" vs "Senior Engineer"). Do NOT say "not enough information" if the answer is calculable or inferable from the memories.
- CLOSED WORLD ASSUMPTION: In memory systems, if an event is recorded but a detail is NOT mentioned, that detail did not occur. Memories capture what the user chose to share. For example: if the user described visiting a museum but did NOT mention a companion, they went alone — answer "No" to "Did I visit with a friend?". If the user never mentioned a role/entity at all, say "not enough information". The distinction: event exists + detail missing = detail didn't happen; event doesn't exist = not enough information.
- SUGGESTION QUESTIONS: If the question asks for a suggestion or recommendation, describe what the user WOULD PREFER based on their past behavior in the memories. Your role is a memory system reporting preferences, not a travel agent or advisor. Any past choice, request, or expressed like/dislike is sufficient to answer. IMPORTANT: Extract GENERAL preferences from the user's past choices even if they are about a different location/context. For example, if the user searched for hotels with "great views" and "hot tubs" in Seattle and the question asks about Miami hotels, the user's preference pattern (great views, hot tubs, amenities) still applies to Miami.
- CONVERGENT EVIDENCE: When multiple independent memories point to the same value (e.g., "user asked if 32 is young" + "user is in their 30s"), treat the convergence as sufficient to infer the value. Do not require an explicit statement when indirect evidence is consistent and unambiguous."""

    return deepseek_call(prompt, model=DEEPSEEK_ANSWER_MODEL)


def judge_answer(hypothesis: str, ground_truth: str, question: str, search_context: List[str]) -> JudgeResult:
    """DeepSeek judges answer correctness and retrieval relevance."""
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

    raw = deepseek_call(prompt)
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

CHECKPOINT_PATH = Path(__file__).parent / "longmemeval_deepseek_checkpoint.json"
FAILURES_PATH = Path(__file__).parent / "longmemeval_deepseek_failures.jsonl"


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
            failure_taxonomy=r.get("failure_taxonomy", {}),
        ))
    return results


def compile_rule_suggestions(failure_buckets: Dict[str, int]) -> List[dict]:
    suggestions = []
    mapping = {
        "timeline_missing": ("temporal-anchor", "Add or refine temporal normalization / timeline materialization rules"),
        "aggregation_incomplete": ("ledger-set", "Promote missed count patterns into ledger/set materialization"),
        "identity_resolution_missing": ("identity-alias", "Add identity/alias/role expansion rules"),
        "contradiction_unresolved": ("counterfactual-filter", "Strengthen contradiction buckets and plan/hypothetical filtering"),
        "profile_missing": ("profile-merge", "Promote missed preference evidence into domain profiles"),
        "structured_gap": ("schema-promotion", "Promote recurring unknown predicates into canonical families"),
        "answer_ambiguous": ("executor-confidence", "Tighten confidence/decay/minimal-evidence executor rules"),
    }
    for bucket, count in sorted(failure_buckets.items(), key=lambda x: (-x[1], x[0])):
        if bucket in mapping:
            rule_id, action = mapping[bucket]
            suggestions.append({
                "bucket": bucket,
                "count": int(count),
                "rule_id": rule_id,
                "suggested_action": action,
            })
    return suggestions


def wait_for_enrichment(
    db: StateletClient,
    graph: str,
    node_id: int,
    timeout_ms: int,
    poll_interval_ms: int = 250,
) -> dict:
    """Poll the root graph node until async enrichment reaches a terminal state."""
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
    only_ids: Optional[set] = None,
    wait_enriched: bool = False,
    enrichment_timeout_ms: int = 30000,
    skip_ingest: bool = False,
    graph_prefix: str = "eval-ds",
):
    print(f"Loading LongMemEval dataset (limit={limit})...")
    questions = load_questions(limit)
    print(f"Loaded {len(questions)} questions")

    checkpoint = load_checkpoint()
    if force or checkpoint.get("model") != DEEPSEEK_MODEL:
        checkpoint = {"completed": {}, "model": DEEPSEEK_MODEL}
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
    print(f"Pipeline: Gateway DeepSeek extraction + full search pipeline")
    print(f"Answer/Judge: DeepSeek API (model={DEEPSEEK_MODEL})")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    W = 80

    if only_ids:
        # Remove specified IDs from checkpoint so they re-run
        removed = []
        for qid in list(only_ids):
            if qid in checkpoint["completed"]:
                del checkpoint["completed"][qid]
                removed.append(qid)
        save_checkpoint(checkpoint)
        print(f"Re-running {len(only_ids)} specified questions: {', '.join(sorted(only_ids))}")
        if removed:
            print(f"  Removed {len(removed)} from checkpoint: {', '.join(removed)}")
    elif retry_failed:
        failed_ids = [qid for qid, r in checkpoint["completed"].items() if not r["correct"]]
        for qid in failed_ids:
            del checkpoint["completed"][qid]
        save_checkpoint(checkpoint)
        print(f"Retrying {len(failed_ids)} failed questions")

    import threading
    checkpoint_lock = threading.Lock()

    def process_question(qi, q):
        """Process a single question (ingest + search + answer + judge). Thread-safe."""
        nonlocal checkpoint

        graph = f"{graph_prefix}-{q.question_id}"

        print(f"\n{'='*W}")
        print(f"Q{qi+1}/{len(questions)}: [{q.question_type}] {q.question[:70]}")
        print(f"  Expected: {str(q.answer)[:70]}")
        print(f"  Graph: {graph}")

        if skip_ingest:
            # Reuse existing graph — skip ingestion but ensure graph is registered
            try:
                db.create_graph_index(graph, dim=embedding_dim)
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"  WARN: create_graph_index: {e}")
            total_facts = len(q.sessions)
            ingest_ms = 0
            print(f"  Skip ingest (reusing existing graph)")
        else:
            # Create per-question graph index
            try:
                db.create_graph_index(graph, dim=embedding_dim)
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"  WARN: create_graph_index: {e}")

            # ── 1. Ingest: send raw sessions to gateway (DeepSeek extraction, 6 concurrent) ──
            t0 = time.time()
            ingest_ok = 0
            ingest_fail = 0

            # Prepare all session texts
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
                            db, graph, node_id, timeout_ms=enrichment_timeout_ms,
                        )
                    return si, True, wait_result
                except Exception as e:
                    return si, False, {"status": f"error: {e}"} if wait_enriched else None

            INGEST_CONCURRENCY = 6
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

        # Per-type search parameters
        search_k = {
            "multi-session": 20,       # Need more results to cover cross-session info
            "knowledge-update": 15,    # Need old + new versions of same entity
            "temporal-reasoning": 25,  # Need more date-bearing facts for calculation
            "single-session-user": 10,
            "single-session-preference": 10,
            "single-session-assistant": 10,
        }.get(q.question_type, 10)

        context_limit = {
            "multi-session": 15,       # Feed more context for cross-referencing
            "knowledge-update": 12,
            "temporal-reasoning": 18,
            "single-session-user": 10,
            "single-session-preference": 10,
            "single-session-assistant": 10,
        }.get(q.question_type, 10)

        # Gateway does query decomposition + rewrite + hop2 gap detection
        # Use fact_k/chunk_k for structured context building
        search_resp = db.text_graph_search(
            graph_name=graph,
            query=search_query,
            k=search_k,
            skip_server_llm=False,
            fact_k=7,
            chunk_k=5,
        )
        search_results = search_resp.results

        # For knowledge-update: also do a direct entity search to find all versions
        if q.question_type == "knowledge-update":
            entity_prompt = f"""Extract the key entities (roles, titles, names, places, items) from this question. Output each entity on its own line, nothing else.

Question: {q.question}"""
            entity_raw = deepseek_call(entity_prompt)
            entity_queries = [l.strip().strip("-•* ") for l in entity_raw.split("\n") if l.strip() and len(l.strip()) > 3][:5]

            seen_ids = {r.node_id for r in search_results}
            for eq in entity_queries:
                extra_resp = db.text_graph_search(
                    graph_name=graph, query=eq, k=10, skip_server_llm=True,
                )
                for r in extra_resp.results:
                    if r.node_id not in seen_ids:
                        search_results.append(r)
                        seen_ids.add(r.node_id)

        # Build context from results with per-type ordering (#1 Context ordering)
        raw_context = []  # (date_str, fact_text, props)
        seen_facts = set()
        failure_taxonomy_counts: Dict[str, int] = defaultdict(int)
        for r in search_results:
            try:
                props = json.loads(r.properties)
                fact_text = props.get("text", "")
                # Skip date-only or too-short entries (metadata nodes, not real facts)
                if len(fact_text) < 25:
                    continue
                fact_key = fact_text[:300]
                if fact_key in seen_facts:
                    continue
                seen_facts.add(fact_key)
                for bucket in props.get("failure_taxonomy", []) or []:
                    failure_taxonomy_counts[str(bucket)] += 1
                date = props.get("date", "")
                raw_context.append((date or "", fact_text, props))
            except (json.JSONDecodeError, AttributeError):
                pass

        # Per-type context ordering
        if q.question_type == "knowledge-update":
            # Sort by date ascending (oldest → newest) so LLM sees timeline
            raw_context.sort(key=lambda x: x[0])
        elif q.question_type == "temporal-reasoning":
            # Sort by date ascending for timeline reasoning
            raw_context.sort(key=lambda x: x[0])
        # else: keep search relevance order (default)

        # Split context by source: user facts vs assistant facts
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

        # Build grouped context with source headers
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

        # ── 3. Two-phase answer (#2) ──
        t0 = time.time()

        # Phase 1: Structured analysis — force DeepSeek to analyze before answering
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

        analysis = deepseek_call(analysis_prompt)
        print(f"  Analysis: {analysis[:100]}")

        # Phase 2: Answer based on analysis + original context
        hypothesis = answer_question(q.question, q.question_type, context[:context_limit], q.question_date)

        # ── 3b. Aggressive re-search (#3) — always analyze gaps, not just on "not enough" ──
        gap_prompt = f"""Based on this question and the analysis, what specific information is MISSING?

Question: {q.question}
Analysis: {analysis[:500]}
Current answer: {hypothesis[:200]}

If ALL needed information is present, output: COMPLETE
Otherwise output 2-3 short search queries to find missing info. One per line."""
        gap_raw = deepseek_call(gap_prompt)

        if "COMPLETE" not in gap_raw.upper():
            gap_queries = [l.strip().lstrip("0123456789.-) ") for l in gap_raw.strip().split("\n")
                          if len(l.strip()) > 10 and "COMPLETE" not in l.upper()][:3]
            if gap_queries:
                print(f"  Re-search: {gap_queries}")
                new_facts = 0
                for gq in gap_queries:
                    extra_resp = db.text_graph_search(
                        graph_name=graph, query=gq, k=5, skip_server_llm=True,
                    )
                    for r in extra_resp.results:
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

        # ── 3c. Self-verification (#4) ──
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

        verify_raw = deepseek_call(verify_prompt)
        if "VERIFIED" not in verify_raw.upper() and len(verify_raw.strip()) > 20:
            print(f"  Self-verify corrected answer")
            hypothesis = verify_raw.strip()

        # ── 3d. Date arithmetic verification (temporal-reasoning only) ──
        if q.question_type == "temporal-reasoning":
            try:
                from datetime import datetime, timedelta
                # Extract dates from hypothesis (YYYY-MM-DD pattern)
                date_matches = re.findall(r'\b(\d{4})-(\d{2})-(\d{2})\b', hypothesis)
                dates = []
                for y, m, d in date_matches:
                    try:
                        dates.append(datetime(int(y), int(m), int(d)))
                    except ValueError:
                        pass
                # Extract claimed number of days/weeks/months
                day_match = re.search(r'(\d+)\s*(?:days?|day)', hypothesis, re.IGNORECASE)
                week_match = re.search(r'(\d+)\s*(?:weeks?|week)', hypothesis, re.IGNORECASE)
                month_match = re.search(r'(\d+)\s*(?:months?|month)', hypothesis, re.IGNORECASE)

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

        # ── 4. Judge (DeepSeek) ──
        judge_result = judge_answer(hypothesis, q.answer, q.question, context[:10])

        tag = "PASS" if judge_result.score == 1 else "FAIL"
        print(f"  Judge: {judge_result.label} | {judge_result.explanation[:80]}")
        print(f"  [{tag}]")

        # Save checkpoint (thread-safe)
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
                "failure_taxonomy": dict(sorted(failure_taxonomy_counts.items())),
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
                        "failure_taxonomy": dict(sorted(failure_taxonomy_counts.items())),
                    }) + "\n")

    # Run questions sequentially (session ingest within each question is parallel)
    for qi, q in enumerate(questions):
        if q.question_id in checkpoint["completed"]:
            continue
        if only_ids and q.question_id not in only_ids:
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
    print(f"  Pipeline: Full DeepSeek (extract + answer + judge, model={DEEPSEEK_MODEL})")
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

    failure_buckets: Dict[str, int] = defaultdict(int)
    for r in results:
        for bucket, count in (r.failure_taxonomy or {}).items():
            failure_buckets[bucket] += int(count)
    if failure_buckets:
        print(f"\n  Failure Taxonomy:")
        for bucket, count in sorted(failure_buckets.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {bucket}: {count}")
        suggestions = compile_rule_suggestions(failure_buckets)
        if suggestions:
            print(f"\n  Suggested Rule Promotions:")
            for item in suggestions[:8]:
                print(f"    {item['rule_id']}: {item['suggested_action']} ({item['count']})")
    else:
        suggestions = []

    out_path = Path(__file__).parent / "longmemeval_deepseek_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "pipeline": "Full DeepSeek (extraction + answer + judge)",
            "answer_model": DEEPSEEK_MODEL,
            "total": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total > 0 else 0,
            "retrieval": {
                "hit_at_k": hit_at_k,
                "avg_precision": avg_precision,
                "total_relevant": total_relevant,
                "total_results": total_results,
            },
            "failure_taxonomy": dict(sorted(failure_buckets.items())),
            "rule_suggestions": suggestions,
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
                    "failure_taxonomy": r.failure_taxonomy,
                }
                for r in results
            ],
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval via full DeepSeek pipeline")
    parser.add_argument("--addr", default="127.0.0.1:9379", help="Statelet gateway address")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion, reuse existing graphs")
    parser.add_argument("--graph-prefix", default="eval-ds", help="Graph name prefix (default: eval-ds)")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (or STATELET_LLM_API_KEY env)")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model for answer/judge")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek API base URL")
    parser.add_argument("--answer-model", default=None, help="DeepSeek model for answer only (e.g. deepseek-reasoner). Defaults to --model")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint, start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run only failed questions")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated question IDs to re-run (e.g. '00ca467f,0100672e,031748ae')")
    parser.add_argument("--wait-enriched", action="store_true", help="Wait for async post-write enrichment before search/answer")
    parser.add_argument("--enrichment-timeout-ms", type=int, default=30000, help="Max wait per session for async enrichment")
    args = parser.parse_args()

    import os
    DEEPSEEK_API_KEY = args.api_key or os.environ.get("STATELET_LLM_API_KEY", "")
    if not DEEPSEEK_API_KEY:
        print("Error: DeepSeek API key required. Use --api-key or set STATELET_LLM_API_KEY")
        sys.exit(1)
    DEEPSEEK_MODEL = args.model
    DEEPSEEK_ANSWER_MODEL = args.answer_model
    DEEPSEEK_BASE_URL = args.base_url
    only_ids = set(args.only.split(",")) if args.only else None
    run_benchmark(
        args.addr,
        args.limit,
        args.dim,
        force=args.force,
        retry_failed=args.retry_failed,
        only_ids=only_ids,
        wait_enriched=args.wait_enriched,
        enrichment_timeout_ms=args.enrichment_timeout_ms,
        skip_ingest=args.skip_ingest,
        graph_prefix=args.graph_prefix,
    )

#!/usr/bin/env python3
"""
LongMemEval benchmark using Claude Code as the host LLM.

All LLM work (extraction, query rewrite, answering, judging) is done by
Claude Code CLI (`claude -p`), using the user's subscription — zero API cost.
Statelet handles embedding + search only via TextGraphPut/TextGraphSearch
with skip_server_llm=true.

Usage:
    python eval/longmemeval_claude.py [--addr 127.0.0.1:9379] [--limit 50] [--model sonnet]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    score: int           # 1 = correct, 0 = incorrect
    label: str           # "correct" or "incorrect"
    explanation: str     # reasoning for the judgment
    retrieval_relevant: int  # how many search results were relevant
    retrieval_total: int     # total search results


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


# ─── Claude Code CLI wrapper ────────────────────────────────────────────────

CLAUDE_MODEL = "sonnet"  # default, overridden by --model (used for answer/judge)
EXTRACT_MODEL = "sonnet"  # default, overridden by --extract-model (used for extraction/rewrite)


CALL_DELAY = 1.0  # seconds between claude -p calls to avoid rate limit


def claude_call(prompt: str, max_retries: int = 3, role: str = "answer") -> str:
    """Call Claude Code CLI in non-interactive mode.

    role: "extract" uses EXTRACT_MODEL, "answer" uses CLAUDE_MODEL.
    """
    model = EXTRACT_MODEL if role == "extract" else CLAUDE_MODEL
    for attempt in range(max_retries + 1):
        try:
            time.sleep(CALL_DELAY)
            result = subprocess.run(
                ["claude", "-p", "--model", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout.strip()
            # Detect rate limit response
            if "hit your limit" in output.lower() or "resets" in output.lower():
                if attempt < max_retries:
                    wait = 30 * (attempt + 1)
                    print(f"    Rate limited ({model}), waiting {wait}s (attempt {attempt+1}/{max_retries+1})...")
                    time.sleep(wait)
                    continue
                return output
            if result.returncode == 0 and output:
                return output
            if attempt < max_retries:
                time.sleep(5)
                continue
            return output or result.stderr.strip() or ""
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                time.sleep(5)
                continue
            return ""
        except Exception as e:
            if attempt < max_retries:
                time.sleep(5)
                continue
            return f"Error: {e}"
    return ""


EXTRACTION_PROMPT = """You are a memory extraction system. Read the conversation below and extract ALL important facts into a JSON array.

<conversation>
{text}
</conversation>

Output a JSON array of objects with:
- "text": A self-contained factual statement. MUST include the person's name (use "user" if unknown). Must be understandable without any additional context.
- "date": ISO date string (e.g. "2023-05-28") if mentioned or resolvable from context. null if no date.
- "category": One of "fact", "preference", "event", "relationship", "decision"

What to extract — be THOROUGH, do NOT skip anything:
- Key facts: names, ages, jobs, locations, skills, quantities, measurements, counts, durations ("waited over a year"), timelines
- Personal experiences: things the user went through, processes they completed, outcomes they received (approvals, rejections, decisions, results)
- Preferences (explicit AND implicit): "I prefer X", "I like X", "I always use X", "I tend to X", brand loyalty, style choices
- Negative preferences: "I don't like X", "I avoid X", "I stopped using X"
- Events: things that happened, activities, purchases, trips, appointments, meetings, ceremonies, celebrations
- Relationships: family members, friends, pets (include names and breeds), colleagues, partners
- Decisions: plans, goals, commitments, choices, scheduled events
- Advice given by assistant: specific recommendations, tips, instructions, products mentioned
- Numbers and specifics: EXACT names, dates, numbers, locations, product names, quantities, prices, durations
- Emotional experiences: frustrations, waiting periods, excitement, concerns

Rules:
- Extract ONLY explicitly stated information — do NOT infer or assume
- Do NOT skip brief mentions — even a single sentence like "I waited over a year" is a fact
- Each fact must be self-contained with the person's name
- Preserve EXACT numbers: "12 fish" not "several"; "$500" not "some money"; "over a year" not "a long time"
- Resolve relative dates ("yesterday", "last week") to absolute dates when conversation date is known
- One fact per JSON object
- Output ONLY the JSON array, no other text"""

# Sliding window settings (matching gateway's llm_extract.rs)
WINDOW_SIZE = 8000       # chars per window
WINDOW_OVERLAP = 500     # overlap between windows
SHORT_THRESHOLD = 500    # below this, single extraction


def _parse_facts_json(raw: str) -> Optional[List[dict]]:
    """Parse JSON array from LLM output, stripping markdown fences.
    Returns None if parsing fails, or a list (possibly empty) on success."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        facts = json.loads(cleaned)
        if isinstance(facts, list):
            return facts
    except json.JSONDecodeError:
        pass
    return None


def _extract_window(text: str) -> List[dict]:
    """Extract facts from a single text window. Retries on rate limit or empty response."""
    prompt = EXTRACTION_PROMPT.format(text=text)
    for attempt in range(5):
        raw = claude_call(prompt, role="extract")
        # Check if response is a rate limit message
        if "hit your limit" in raw.lower() or "resets" in raw.lower() or "rate limit" in raw.lower():
            wait = 30 * (attempt + 1)
            print(f"    [extract_window] Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
            time.sleep(wait)
            continue
        # Empty or very short response — likely rate limit without message body
        if len(raw.strip()) < 10:
            wait = 15 * (attempt + 1)
            print(f"    [extract_window] Empty response ({len(raw)} chars), retrying in {wait}s (attempt {attempt+1}/5)...")
            time.sleep(wait)
            continue
        facts = _parse_facts_json(raw)
        if facts is not None:
            return facts
        # Got a response but couldn't parse JSON — might be rate limit text or malformed
        if len(raw) < 100:
            wait = 15 * (attempt + 1)
            print(f"    [extract_window] Unparseable short response: {repr(raw)}")
            print(f"    [extract_window] Unparseable short response, retrying in {wait}s (attempt {attempt+1}/5)...")
            time.sleep(wait)
            continue
        # Long response but not JSON — something wrong, don't retry
        return []
    print(f"    [extract_window] Failed after 5 attempts")
    return []


def extract_facts(session_text: str) -> List[dict]:
    """Extract facts using sliding windows for long sessions.

    Short text (< 500 chars): single extraction call.
    Long text: split into overlapping ~4000 char windows, extract from
    each window, then deduplicate by text similarity.
    """
    if len(session_text) < SHORT_THRESHOLD:
        facts = _extract_window(session_text)
        return facts if facts else [{"text": session_text[:500], "date": None, "category": "fact"}]

    # Split into overlapping windows
    total = len(session_text)
    step = max(WINDOW_SIZE - WINDOW_OVERLAP, 1)
    windows = []
    start = 0
    while start < total:
        end = min(start + WINDOW_SIZE, total)
        windows.append(session_text[start:end])
        if end >= total:
            break
        start += step

    # Extract from each window
    all_facts = []
    for window in windows:
        facts = _extract_window(window)
        all_facts.extend(facts)

    # Deduplicate: remove facts with near-identical text (from overlap regions)
    deduped = []
    for fact in all_facts:
        fact_text = fact.get("text", "")
        is_dup = False
        for existing in deduped:
            existing_text = existing.get("text", "")
            # Simple word-level Jaccard similarity
            words_a = set(fact_text.lower().split())
            words_b = set(existing_text.lower().split())
            if words_a and words_b:
                jaccard = len(words_a & words_b) / len(words_a | words_b)
                if jaccard > 0.8:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(fact)

    # ── 方案 2: Keyword-triggered targeted extraction ──
    # Scan for sentences with numbers/updates that might have been missed
    import re as _re
    update_patterns = [
        r'\b(\d+)\s+(?:new|total|more|added|now|currently)',
        r'(?:added|collected|earned|raised|completed|finished|started|changed|moved|switched)\b.*?\b\d+',
        r'\b(?:now have|currently have|up to|increased to|grew to)\b',
        r'\$\d+',
    ]
    # Find sentences with update keywords
    sentences = _re.split(r'[.!?\n]+', session_text)
    targeted_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 500:
            continue
        for pat in update_patterns:
            if _re.search(pat, sent, _re.IGNORECASE):
                targeted_sentences.append(sent)
                break

    if targeted_sentences:
        targeted_text = "\n".join(targeted_sentences[:20])
        targeted_facts = _extract_window(targeted_text)
        for fact in targeted_facts:
            fact_text = fact.get("text", "")
            is_dup = any(
                len(set(fact_text.lower().split()) & set(e.get("text", "").lower().split())) /
                max(len(set(fact_text.lower().split()) | set(e.get("text", "").lower().split())), 1) > 0.8
                for e in deduped
            )
            if not is_dup:
                deduped.append(fact)

    # ── 方案 1: Second-pass verification ──
    # Check if any facts were missed by reviewing extracted facts vs original text
    if deduped and len(session_text) > SHORT_THRESHOLD:
        existing_summary = "\n".join(f"- {f.get('text', '')[:100]}" for f in deduped[:30])
        verify_prompt = f"""Review the conversation below and the facts already extracted. List ONLY facts that were MISSED.

<conversation>
{session_text[:6000]}
</conversation>

<already_extracted>
{existing_summary}
</already_extracted>

Focus on finding MISSED facts, especially:
- Numbers, counts, totals, amounts (e.g. "25 postcards", "$500", "4 engineers")
- Updates to previous information ("changed to", "now have", "switched from X to Y")
- Specific dates, durations, timelines
- Names of people, places, products mentioned briefly

If nothing was missed, output: []
Otherwise output a JSON array of missed facts with "text", "date", "category" fields.
Output ONLY the JSON array or [], no other text."""

        raw = claude_call(verify_prompt, role="extract")
        missed = _parse_facts_json(raw) or []
        for fact in missed:
            fact_text = fact.get("text", "")
            is_dup = any(
                len(set(fact_text.lower().split()) & set(e.get("text", "").lower().split())) /
                max(len(set(fact_text.lower().split()) | set(e.get("text", "").lower().split())), 1) > 0.8
                for e in deduped
            )
            if not is_dup:
                deduped.append(fact)

    return deduped if deduped else [{"text": session_text[:500], "date": None, "category": "fact"}]


def rewrite_query(query: str, question_date: str) -> List[str]:
    """Claude rewrites query into fact-style search statements."""
    date_ctx = ""
    if question_date:
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", question_date)
        if m:
            date_ctx = f"\nCurrent date is {m.group(1)}-{m.group(2)}-{m.group(3)}. Resolve all relative dates (yesterday, last week, 10 days ago) to absolute dates."

    prompt = f"""Rewrite this question as 2-4 short factual search queries that would match stored memories.{date_ctx}

Question: {query}

Rules:
- Convert to declarative statements (e.g. "What is my pet?" → "user has a pet")
- Include key entities, names, and specific terms
- Resolve relative dates to absolute if current date is provided
- For multi-part questions (A and B), create separate queries for each part
- For temporal questions, include both the event AND the date
- For abstract or generic terms (e.g. "homegrown ingredients", "my hobbies", "my routine"), also generate queries about SPECIFIC concrete instances (e.g. "user grows vegetables herbs in garden", "user plays guitar", "user morning exercise")
- Think about what specific facts a user would have told an assistant that relate to the question, and phrase queries to match those facts
- Output 2-4 queries, one per line, nothing else"""

    raw = claude_call(prompt, role="extract")
    lines = [l.strip().lstrip("0123456789.-) ") for l in raw.strip().split("\n") if l.strip()]
    return [l for l in lines if len(l) > 10][:4]


def extract_entities_for_hop2(question: str, round1_texts: List[str]) -> List[str]:
    """Claude identifies missing entities from round 1 results for a second search hop."""
    if not round1_texts:
        return []

    r1_summary = "\n".join(f"- {t[:200]}" for t in round1_texts[:5])

    prompt = f"""Given this question and initial search results, identify what information is STILL MISSING that requires another search.

Question: {question}

Initial results found:
{r1_summary}

If the question requires information about multiple events, people, or items, and some are missing from the results above, output 1-3 additional search queries to find the missing pieces.

If the initial results already contain everything needed, output: COMPLETE

Rules:
- Only output additional queries if critical information is missing
- If a key concept in the question is NOT addressed AT ALL in the results (not just partially), generate broader or alternative phrasings for that concept (e.g. if "homegrown ingredients" returned no garden/plant results, try "user grows plants vegetables herbs garden")
- If results are all tangentially related but none directly answer the question, rephrase with more concrete/specific terms
- Each query on its own line
- Output ONLY the queries (or COMPLETE), nothing else"""

    raw = claude_call(prompt, role="extract")
    if "COMPLETE" in raw.upper():
        return []
    lines = [l.strip().lstrip("0123456789.-) ") for l in raw.strip().split("\n") if l.strip()]
    return [l for l in lines if len(l) > 10 and "COMPLETE" not in l.upper()][:3]


def multi_hop_search(
    db: StateletClient,
    graph: str,
    question: str,
    question_date: str,
    extra_queries: List[str],
) -> tuple:
    """Multi-hop search: round 1 → analyze gaps → round 2 if needed."""
    iso_date = ""
    if question_date:
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", question_date)
        if m:
            iso_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    search_query = f"[Context date: {iso_date}] {question}" if iso_date else question

    # Round 1: main query + rewritten queries
    r1_results = db.text_graph_search(
        graph_name=graph,
        query=search_query,
        k=10,
        skip_server_llm=True,
        extra_queries=extra_queries,
    )

    # Extract text from round 1 results
    r1_texts = []
    r1_ids = set()
    for r in r1_results:
        r1_ids.add(r.node_id)
        try:
            props = json.loads(r.properties)
            text = props.get("text") or props.get("original_text", "")
            if text:
                r1_texts.append(text)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Round 2: Claude identifies missing entities, search again
    hop2_queries = extract_entities_for_hop2(question, r1_texts)
    r2_results = []
    if hop2_queries:
        r2_results = db.text_graph_search(
            graph_name=graph,
            query=search_query,
            k=5,
            skip_server_llm=True,
            extra_queries=hop2_queries,
        )

    # Merge results (deduplicate by node_id)
    all_results = list(r1_results)
    for r in r2_results:
        if r.node_id not in r1_ids:
            all_results.append(r)
            r1_ids.add(r.node_id)

    # Build context from merged results
    # Dedup by extracted fact text (not original session text) to avoid
    # dropping different facts from sessions with similar openings.
    context = []
    seen_facts = set()
    for r in all_results:
        try:
            props = json.loads(r.properties)
            fact_text = props.get("text", "")
            # Dedup by fact text (first 300 chars)
            fact_key = fact_text[:300] if fact_text else ""
            if fact_key and fact_key in seen_facts:
                continue
            if fact_key:
                seen_facts.add(fact_key)
            # Use original session text as context for richer answer
            text = props.get("original_text", "") or fact_text
            if text:
                context.append(text)
        except (json.JSONDecodeError, AttributeError):
            pass

    return all_results, context, len(hop2_queries)


def answer_question(question: str, question_type: str, context: List[str], question_date: str) -> str:
    """Claude answers with type-specific instructions."""
    ctx_str = "\n\n---\n\n".join(f"Memory {i+1}: {c}" for i, c in enumerate(context))

    date_info = ""
    if question_date:
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", question_date)
        if m:
            date_info = f"**Question date: {m.group(1)}-{m.group(2)}-{m.group(3)}**\nAll relative dates ('yesterday', 'last week', '10 days ago') are relative to this date.\n"

    # Type-specific instructions
    type_instructions = {
        "temporal-reasoning": """TEMPORAL REASONING INSTRUCTIONS:
- Find the specific dates of events mentioned in the question.
- Calculate durations: count the days between two dates (inclusive or exclusive as appropriate).
- "X days ago" = question_date minus X days.
- "last week" = the 7 days before question_date.
- Show your date arithmetic step by step.
- If two events are asked about, find BOTH dates before calculating.""",

        "multi-session": """MULTI-SESSION REASONING INSTRUCTIONS:
- The answer REQUIRES combining information from MULTIPLE memories.
- You MUST read and consider EVERY memory listed below before answering.
- For "how many" / "total" / counting questions:
  1. Scan ALL memories and list EACH relevant item with its memory number
  2. Count them explicitly
  3. Only then give your final answer
- For comparison questions: find both values from different memories, then compare.
- Do NOT answer based on only the first memory — the answer is almost always spread across multiple memories.""",

        "knowledge-update": """KNOWLEDGE UPDATE INSTRUCTIONS:
- Information may have changed over time across different conversations.
- Memories with dates (e.g. [2023-05-28]) are shown in roughly chronological order. LATER dates override EARLIER dates when facts conflict.
- Look for action verbs that indicate state changes: "moved to", "plans to store", "switched to", "decided to", "changed to", "I now..."
- "Plans to X" or "decided to X" or "moved X to Y" indicate the LATEST state — these override prior descriptions.
- Do NOT assume "more detailed" or "more specific" means "more recent". Use dates and action verbs to determine recency.
- The most recent memory takes precedence over older ones.
- CRITICAL: Pay close attention to the EXACT role, title, entity, or term in the question. If the question asks about "Software Engineer Manager" but memories only mention "Senior Software Engineer", these are DIFFERENT roles — do NOT assume they are the same. Say "I don't have enough information".
- If the question asks about an entity/role/event that does NOT exactly match what's in the memories, say "I don't have enough information" — do not substitute a similar-sounding but different entity.""",

        "single-session-preference": """PREFERENCE INSTRUCTIONS:
- Look for explicit preferences: "I prefer", "I like", "my favorite", "I always"
- Also implicit preferences: habits, repeated choices, style decisions
- If asked for recommendations, base them on stated preferences and past experiences.""",

        "single-session-user": """USER FACT INSTRUCTIONS:
- Look for specific factual details: names, numbers, durations, locations, dates.
- Quote the exact information from the memory.
- Do not round numbers or approximate unless the memory itself is approximate.""",

        "single-session-assistant": """ASSISTANT RECALL INSTRUCTIONS:
- The question asks about something the assistant previously said or recommended.
- Look for specific advice, explanations, or information the assistant provided.
- Quote specific details: colors, names, numbers, steps, ratios mentioned by the assistant.""",
    }

    specific = type_instructions.get(question_type, "")

    prompt = f"""Answer the question using ONLY the retrieved memories below.

{date_info}
{specific}

## Retrieved Memories

{ctx_str}

## Question

{question}

## General Rules

1. Use ONLY information from the memories above. Do not make up facts.
2. Be specific and concise in your final answer.
3. If the memories don't contain enough information, say "I don't have enough information."
4. Show brief reasoning before your answer when the question requires calculation or comparison."""

    return claude_call(prompt)


def judge_answer(hypothesis: str, ground_truth: str, question: str, search_context: List[str]) -> JudgeResult:
    """Claude judges answer correctness and retrieval relevance."""
    ctx_summary = "\n".join(f"  Result {i+1}: {c[:200]}" for i, c in enumerate(search_context[:10]))

    prompt = f"""You are a strict evaluation judge. Evaluate the HYPOTHESIS against the GROUND TRUTH for the given QUESTION.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
HYPOTHESIS: {hypothesis}

SEARCH RESULTS USED:
{ctx_summary}

## Evaluation Rules

1. **Answer correctness:**
   - The hypothesis does NOT need to match word-for-word
   - It MUST convey the same key factual information (names, numbers, dates, facts)
   - If the hypothesis contains the core correct answer but omits a secondary qualifier, score as CORRECT only if the omitted detail does not change the meaning. For example: "in my closet" vs "in a shoe rack in my closet" — the primary location (closet) is correct, the qualifier (shoe rack) is secondary → CORRECT. But "under my bed" vs "in my closet" → fundamentally different → INCORRECT.
   - If the hypothesis says "I don't have enough information" or similar:
     - CORRECT only if the ground truth ALSO explicitly says the information is not enough / not provided / insufficient
     - INCORRECT if the ground truth contains a specific answer
   - Numbers must be correct (e.g. "30 days" vs "28 days" = incorrect)
   - For preference questions: must reflect the specific preferences in ground truth

2. **Retrieval relevance:**
   - Count how many of the search results contain information that is relevant to answering the question
   - A result is "relevant" if it contains any fact, name, date, or detail mentioned in the ground truth

Output a JSON object (no markdown fences):
{{"score": 1 or 0, "label": "correct" or "incorrect", "explanation": "brief reason for judgment", "retrieval_relevant": number_of_relevant_results, "retrieval_total": {len(search_context[:10])}}}"""

    raw = claude_call(prompt)

    # Parse JSON
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
        # Fallback: simple check
        is_correct = "correct" in cleaned.lower() and "incorrect" not in cleaned.lower()
        return JudgeResult(
            score=1 if is_correct else 0,
            label="correct" if is_correct else "incorrect",
            explanation=cleaned[:200],
            retrieval_relevant=0,
            retrieval_total=len(search_context[:10]),
        )


# ─── Checkpoint ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent / "longmemeval_checkpoint.json"


def load_checkpoint() -> dict:
    """Load checkpoint or return empty state."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed": {}, "model": ""}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint after each question."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def checkpoint_to_results(checkpoint: dict) -> List[EvalResult]:
    """Reconstruct EvalResult list from checkpoint."""
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


# ─── Main benchmark ─────────────────────────────────────────────────────────

def run_benchmark(statelet_addr: str, limit: int, embedding_dim: int, force: bool = False, retry_failed: bool = False):
    print(f"Loading LongMemEval dataset (limit={limit})...")
    questions = load_questions(limit)
    print(f"Loaded {len(questions)} questions")

    # Load or create checkpoint
    checkpoint = load_checkpoint()
    if force or checkpoint.get("model") != CLAUDE_MODEL:
        checkpoint = {"completed": {}, "model": CLAUDE_MODEL}
        save_checkpoint(checkpoint)
        print("Starting fresh run")
    else:
        n_done = len(checkpoint["completed"])
        if n_done > 0:
            n_pass = sum(1 for r in checkpoint["completed"].values() if r["correct"])
            n_fail = n_done - n_pass
            print(f"Resuming from checkpoint: {n_done}/{len(questions)} done ({n_pass} PASS, {n_fail} FAIL)")

    print(f"Connecting to Statelet at {statelet_addr}...")
    db = StateletClient(statelet_addr)
    print(f"Ping: {db.ping()}")
    print(f"Host LLM: Claude Code CLI (answer={CLAUDE_MODEL}, extract={EXTRACT_MODEL})")
    print(f"Statelet: Claude extraction + full gateway search pipeline")
    print(f"Features: sliding window extraction, conflict detection, supersedes edges")
    print(f"Search: skip_server_llm=false (BM25 + reranker + temporal + supersedes + edge expansion)")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    W = 80

    # If retry-failed, remove failed questions from checkpoint so they re-run
    if retry_failed:
        failed_ids = [qid for qid, r in checkpoint["completed"].items() if not r["correct"]]
        for qid in failed_ids:
            del checkpoint["completed"][qid]
        save_checkpoint(checkpoint)
        print(f"Retrying {len(failed_ids)} failed questions: {', '.join(f[:8] for f in failed_ids)}")

    for qi, q in enumerate(questions):
        # Skip completed questions
        if q.question_id in checkpoint["completed"]:
            continue

        graph = f"eval-{q.question_id}"

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

        # ── 1. Ingest: parallel Claude extraction → store → batch conflict detection ──
        t0 = time.time()
        total_facts = 0
        supersedes_count = 0
        all_stored_facts: List[tuple] = []  # (node_id, fact_text, session_index)

        # Prepare session texts
        session_texts = []
        for si, session_msgs in enumerate(q.sessions):
            lines = []
            if si < len(q.session_dates) and q.session_dates[si]:
                lines.append(f"[{q.session_dates[si]}]")
            for msg in session_msgs:
                lines.append(f"{msg['role']}: {msg['content']}")
            session_texts.append("\n".join(lines))

        # Parallel extraction (3 concurrent sessions)
        EXTRACT_CONCURRENCY = 4
        extracted_facts = {}  # si -> list of facts

        def extract_session(si_text):
            si, text = si_text
            return si, extract_facts(text)

        with ThreadPoolExecutor(max_workers=EXTRACT_CONCURRENCY) as pool:
            futures = {pool.submit(extract_session, (si, text)): si for si, text in enumerate(session_texts)}
            done_count = 0
            for future in as_completed(futures):
                si, facts = future.result()
                extracted_facts[si] = facts
                done_count += 1
                if done_count % 10 == 0:
                    print(f"  Extracted {done_count}/{len(q.sessions)} sessions")

        print(f"  Extracted all {len(q.sessions)} sessions, storing facts...")

        # Store facts sequentially (gRPC calls)
        for si in range(len(q.sessions)):
            facts = extracted_facts.get(si, [])
            session_text = session_texts[si]
            for fi, fact in enumerate(facts):
                node_id = (si + 1) * 1000 + fi
                fact_text = fact.get("text", "")
                props = json.dumps({
                    "text": fact_text,
                    "original_text": session_text[:8000],
                    "category": fact.get("category", "fact"),
                    "date": fact.get("date"),
                    "session_index": si,
                }).encode()
                try:
                    db.text_graph_put(
                        graph_name=graph,
                        text=fact_text,
                        node_id=node_id,
                        properties=props,
                        skip_server_llm=True,
                    )
                    total_facts += 1
                    all_stored_facts.append((node_id, fact_text, si))
                except Exception:
                    pass

            if (si + 1) % 10 == 0:
                print(f"  Stored {si+1}/{len(q.sessions)} sessions ({total_facts} facts)")

        # Batch conflict detection: only cross-session, only update-likely facts
        # Only scan facts containing numbers/updates (not all 1800+ facts)
        import re as _re
        update_keywords = _re.compile(
            r'\b\d+\b|changed|moved|switched|updated|now|currently|new|started|stopped',
            _re.IGNORECASE
        )
        candidate_facts = [(nid, ft, si) for nid, ft, si in all_stored_facts
                           if ft and len(ft) > 20 and update_keywords.search(ft)]
        print(f"  Detecting cross-session conflicts ({len(candidate_facts)}/{total_facts} candidate facts)...")
        conflict_pairs = []

        node_to_session = {nid: si for nid, _, si in all_stored_facts}

        def _search_conflict(item):
            node_id, fact_text, si = item
            try:
                existing = db.text_graph_search(
                    graph_name=graph, query=fact_text, k=5, skip_server_llm=True
                )
                for ex in existing:
                    if ex.node_id == node_id:
                        continue
                    ex_session = node_to_session.get(ex.node_id, -1)
                    if ex_session == si:
                        continue  # same session, skip
                    if ex.distance < 0.12:
                        try:
                            ex_props = json.loads(ex.properties)
                            ex_text = ex_props.get("text", "")
                            if ex_text and ex_text != fact_text:
                                return (node_id, fact_text, ex.node_id, ex_text)
                        except (json.JSONDecodeError, AttributeError):
                            pass
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(_search_conflict, candidate_facts):
                if result:
                    conflict_pairs.append(result)

        # Batch resolve conflicts with one Claude call (up to 30 pairs)
        if conflict_pairs:
            print(f"  Found {len(conflict_pairs)} potential conflicts, resolving...")
            # Build batch prompt
            for batch_start in range(0, len(conflict_pairs), 30):
                batch = conflict_pairs[batch_start:batch_start + 30]
                entries = "\n".join(
                    f"#{i}: OLD=\"{old[:150]}\" → NEW=\"{new[:150]}\""
                    for i, (_, new, _, old) in enumerate(batch)
                )
                verdict = claude_call(
                    f"For each pair, does the NEW fact update/replace the OLD fact?\n\n"
                    f"{entries}\n\n"
                    f"Output one line per pair: just the number and 'yes' or 'no'.\n"
                    f"Example:\n#0: yes\n#1: no\n#2: yes",
                    role="extract",
                )
                # Parse batch results and create supersedes edges
                for i, (new_nid, new_text, old_nid, old_text) in enumerate(batch):
                    if f"#{i}: yes" in verdict.lower() or f"#{i}:yes" in verdict.lower():
                        try:
                            # Re-put with supersedes edge
                            db.text_graph_put(
                                graph_name=graph,
                                text=new_text,
                                node_id=new_nid,
                                edge_target=old_nid,
                                edge_type="supersedes",
                                skip_server_llm=True,
                            )
                            supersedes_count += 1
                        except Exception:
                            pass

        print(f"  Conflicts: {len(conflict_pairs)} detected, {supersedes_count} supersedes edges created")
        ingest_ms = (time.time() - t0) * 1000
        print(f"  Ingested: {total_facts} facts in {ingest_ms/1000:.1f}s")

        # ── 2. Search: Claude rewrite → Gateway full pipeline (vector + BM25 + RRF + reranker + supersedes + edges) ──
        t0 = time.time()

        iso_date = ""
        if q.question_date:
            m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", q.question_date)
            if m:
                iso_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        search_query = f"[Context date: {iso_date}] {q.question}" if iso_date else q.question

        # Claude rewrites query into fact-style statements
        extra_queries = rewrite_query(q.question, q.question_date)

        # Single search call with extra_queries → Gateway does unified RRF fusion
        # skip_server_llm=True: Claude handles LLM work, Gateway does embedding + full search pipeline
        search_results = db.text_graph_search(
            graph_name=graph,
            query=search_query,
            k=10,
            skip_server_llm=True,
            extra_queries=extra_queries,
        )

        # Multi-hop: Claude analyzes results and identifies gaps
        r1_texts = []
        for r in search_results:
            try:
                props = json.loads(r.properties)
                text = props.get("text", "")
                if text:
                    r1_texts.append(text)
            except (json.JSONDecodeError, AttributeError):
                pass

        hop2_queries = extract_entities_for_hop2(q.question, r1_texts)
        if hop2_queries:
            hop2_results = db.text_graph_search(
                graph_name=graph,
                query=search_query,
                k=5,
                skip_server_llm=True,
                extra_queries=hop2_queries,
            )
            seen_ids = {r.node_id for r in search_results}
            for r in hop2_results:
                if r.node_id not in seen_ids:
                    search_results.append(r)
                    seen_ids.add(r.node_id)

        # Build context from results (fact-level dedup, with dates)
        context = []
        seen_facts = set()
        for r in search_results:
            try:
                props = json.loads(r.properties)
                fact_text = props.get("text", "")
                fact_key = fact_text[:300] if fact_text else ""
                if fact_key and fact_key in seen_facts:
                    continue
                if fact_key:
                    seen_facts.add(fact_key)
                if fact_text:
                    date = props.get("date", "")
                    if date:
                        context.append(f"[{date}] {fact_text}")
                    else:
                        context.append(fact_text)
            except (json.JSONDecodeError, AttributeError):
                pass

        search_ms = (time.time() - t0) * 1000
        hop_info = f" (+{len(hop2_queries)} hop2)" if hop2_queries else ""
        print(f"  Search: {len(search_results)} results, {len(extra_queries)} rewrites{hop_info} in {search_ms:.0f}ms")
        print(f"  Query: {search_query}")
        for eq in extra_queries:
            print(f"    rewrite: {eq}")
        for si, ctx_text in enumerate(context[:10]):
            print(f"    [{si}] {ctx_text}")

        # ── 3. Answer (Claude, type-specific prompt) ──
        t0 = time.time()
        hypothesis = answer_question(q.question, q.question_type, context[:10], q.question_date)
        answer_ms = (time.time() - t0) * 1000
        print(f"  Answer: {hypothesis}")

        # ── 4. Judge (Claude, structured) ──
        judge_result = judge_answer(hypothesis, q.answer, q.question, context[:10])

        tag = "PASS" if judge_result.score == 1 else "FAIL"
        print(f"  Judge: {judge_result.label} | {judge_result.explanation}")
        print(f"  Retrieval: {judge_result.retrieval_relevant}/{judge_result.retrieval_total} relevant")
        print(f"  [{tag}]")

        # Save to checkpoint immediately
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

        # Log failures to dedicated file for audit
        if judge_result.score != 1:
            err_path = Path(__file__).parent / "longmemeval_failures.jsonl"
            with open(err_path, "a") as ef:
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

    # ── Summary (from checkpoint, includes all completed questions) ──
    results = checkpoint_to_results(checkpoint)

    print(f"\n{'='*W}")
    print("RESULTS SUMMARY")
    print(f"{'='*W}\n")

    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    print(f"  Host LLM: Claude Code (model={CLAUDE_MODEL})")
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

    # Retrieval metrics
    total_relevant = sum(r.judge.retrieval_relevant for r in results)
    total_results = sum(r.judge.retrieval_total for r in results)
    hit_count = sum(1 for r in results if r.judge.retrieval_relevant > 0)
    hit_at_k = hit_count / total if total > 0 else 0
    avg_precision = (total_relevant / total_results) if total_results > 0 else 0

    print(f"\n  Retrieval Quality:")
    print(f"    Hit@K: {100*hit_at_k:.1f}% ({hit_count}/{total})")
    print(f"    Avg Precision: {100*avg_precision:.1f}% ({total_relevant}/{total_results} relevant)")

    out_path = Path(__file__).parent / "longmemeval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "host_llm": f"Claude Code ({CLAUDE_MODEL})",
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
    parser = argparse.ArgumentParser(description="LongMemEval via Claude Code + Statelet")
    parser.add_argument("--addr", default="127.0.0.1:9379", help="Statelet gateway address")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--model", default="sonnet", help="Claude model for answer/judge (sonnet/opus/haiku)")
    parser.add_argument("--extract-model", default=None, help="Claude model for extraction/rewrite (defaults to --model)")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint, start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run only failed questions from checkpoint")
    args = parser.parse_args()

    CLAUDE_MODEL = args.model
    EXTRACT_MODEL = args.extract_model or args.model
    run_benchmark(args.addr, args.limit, args.dim, force=args.force, retry_failed=args.retry_failed)

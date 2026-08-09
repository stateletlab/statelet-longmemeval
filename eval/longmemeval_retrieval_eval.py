#!/usr/bin/env python3
"""
LongMemEval benchmark — pure retrieval quality evaluation.

No answer extraction, no LLM. Measures whether the search pipeline
retrieves sessions that contain the ground-truth answer.

Metrics per question:
  - hit@k: Did ANY of the top-k results come from an answer session?
  - recall@k: What fraction of answer sessions appear in top-k?
  - MRR: Reciprocal rank of the first hit.
  - relevance: Token F1 between each search result text and ground-truth answer.

Usage:
    python eval/longmemeval_retrieval_eval.py --limit 100
    python eval/longmemeval_retrieval_eval.py --limit 100 --run-id my-run --k 20
    python eval/longmemeval_retrieval_eval.py --limit 100 --run-id my-run --retry-failed

#763 LongMemEval Phase 1 — sentence-unit recall (multi-vector MAX aggregation):
    The unit segmenter and session-MAX collapse are SERVER-SIDE env toggles on
    the gateway (this harness drives the existing TextGraphPut / TextGraphSearch
    RPCs unchanged). Run the A/B by flipping the gateway env between two runs:

      # (a) baseline: mean-pooled multi-sentence-window chunks
      STATELET_RETURN_SESSIONS=1 \
          python eval/longmemeval_retrieval_eval.py --run-id baseline --k 20

      # (b) Phase 1: rule-based ~16-48-tok units, group-by-session MAX
      STATELET_SENTENCE_UNITS=1 STATELET_RETURN_SESSIONS=1 \
      STATELET_SESSION_AGG=max \
          python eval/longmemeval_retrieval_eval.py --run-id units-max --k 20

    (set the same STATELET_SENTENCE_UNITS / STATELET_RETURN_SESSIONS on the gateway
    process — see scripts/cluster-start.sh). STATELET_SESSION_AGG=topp (with
    optional STATELET_SESSION_AGG_P=<n>, default 2) selects the top-p-mean hedge
    instead of pure MAX. Both fold units to sessions by MAX/top-p, never SUM, so
    long sessions are not biased. Acceptance: recall@k / per-category recall on
    single- and multi-session needle questions beat the baseline run by a clear
    margin; commit both retrieval_eval_results_*.json as the artifact. LLM-free.
"""

import argparse
import atexit
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, TextIO, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# ─── Setup paths ──────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "sdk" / "python"))
from statelet.client import StateletClient

MEMORYBENCH_DIR = PROJECT_DIR.parent / "memorybench"
QUESTIONS_DIR = MEMORYBENCH_DIR / "data" / "benchmarks" / "longmemeval" / "datasets" / "questions"

# ─── Logging ──────────────────────────────────────────────────────────────

RUN_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
LOG_PATH = EVAL_DIR / f"retrieval_eval_run_{RUN_ID}.log"
DETAIL_LOG_PATH = EVAL_DIR / f"retrieval_eval_detail_{RUN_ID}.jsonl"
CHECKPOINT_PATH = EVAL_DIR / f"retrieval_eval_checkpoint_{RUN_ID}.json"
INGEST_OUTBOX_PATH = EVAL_DIR / f"retrieval_eval_outbox_{RUN_ID}.jsonl"
LOCK_PATH = EVAL_DIR / "longmemeval_retrieval_eval.lock"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger("retrieval_eval")


def _sanitize_run_id(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id.strip())
    cleaned = cleaned.strip("-._")
    if not cleaned:
        raise ValueError("run_id is empty after sanitization")
    return cleaned


def configure_run_context(run_id: str = "", checkpoint_path: str = "") -> None:
    global RUN_ID, LOG_PATH, DETAIL_LOG_PATH, CHECKPOINT_PATH, INGEST_OUTBOX_PATH
    if run_id:
        RUN_ID = _sanitize_run_id(run_id)
    LOG_PATH = EVAL_DIR / f"retrieval_eval_run_{RUN_ID}.log"
    DETAIL_LOG_PATH = EVAL_DIR / f"retrieval_eval_detail_{RUN_ID}.jsonl"
    if checkpoint_path:
        CHECKPOINT_PATH = Path(checkpoint_path).expanduser().resolve()
    else:
        CHECKPOINT_PATH = EVAL_DIR / f"retrieval_eval_checkpoint_{RUN_ID}.json"
    INGEST_OUTBOX_PATH = EVAL_DIR / f"retrieval_eval_outbox_{RUN_ID}.jsonl"
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH),
        ],
    )


def acquire_single_instance_lock() -> TextIO:
    lock_fp = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fp.seek(0)
        owner = lock_fp.read().strip() or "unknown owner"
        lock_fp.close()
        raise RuntimeError(
            f"another retrieval_eval run is active ({owner}); "
            f"use --allow-concurrent only if you fully isolate graph/checkpoint paths"
        )
    lock_fp.seek(0)
    lock_fp.truncate()
    lock_fp.write(f"pid={os.getpid()} run_id={RUN_ID} started_at={datetime.now().isoformat()}\n")
    lock_fp.flush()
    return lock_fp


def release_single_instance_lock(lock_fp: Optional[TextIO]) -> None:
    if lock_fp is None:
        return
    try:
        lock_fp.seek(0); lock_fp.truncate(); lock_fp.flush()
    except Exception:
        pass
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_fp.close()
    except Exception:
        pass


# ─── Timeouts ─────────────────────────────────────────────────────────────

PING_TIMEOUT_S = 5.0
GRAPH_CREATE_TIMEOUT_S = 45.0
GRAPH_READY_TIMEOUT_S = 30.0
TEXT_GRAPH_PUT_TIMEOUT_S = 600.0
TEXT_GRAPH_SEARCH_TIMEOUT_S = 120.0
OVERALL_INGEST_TIMEOUT_S = 1800.0
GRAPH_INIT_MAX_ATTEMPTS = 3
DEFAULT_MAX_WORKERS = 2
INGEST_SESSION_MAX_ATTEMPTS = 3
INGEST_RETRY_BASE_DELAY_S = 1.0
INGEST_RECOVERY_MAX_ATTEMPTS = 8
INGEST_RECOVERY_BASE_DELAY_S = 2.0
INGEST_RECOVERY_MAX_DELAY_S = 15.0
DEFAULT_K = 20

# (#827 LongMemEval Phase 5a) Multi-granularity ingest + RRF fusion. When
# non-empty, the search request asks the gateway to run one ANN ranking per
# granularity (sentence/round/session/fact), fold each to sessions, and fuse the
# per-granularity session rankings via Reciprocal Rank Fusion. Empty ⇒ legacy
# single-granularity (sentence) behavior. Set from --granularities / --rrf-k.
# The gateway must also be started with STATELET_MULTI_GRANULARITY=1 so ingest
# writes the round/session granularity indexes these rankings read.
SEARCH_GRANULARITIES: List[str] = []
SEARCH_RRF_K = 0

# (#830 LongMemEval Phase 5d) Read-time entity consolidation. When True, the
# search request asks the gateway to dereference each query entity term through
# its persisted `gcanon:` identity cluster (Phase 5c) and OR-expand retrieval over
# the cluster's surface forms, so cross-session evidence written under any surface
# is recalled. Set from --entity-consolidate. The gateway must be started with
# STATELET_ENTITY_RESOLVE=1 so the `gcanon:` clusters this dereferences exist.
SEARCH_ENTITY_CONSOLIDATE = False

# Include assistant turns by default — required for "previous conversation" recall questions.
# Set LMJ_INCLUDE_ASSISTANT=0 to revert to user-only mode.
INGEST_INCLUDE_ASSISTANT = os.environ.get("LMJ_INCLUDE_ASSISTANT", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
EXTRA_QUERY_MAX = max(0, min(int(os.environ.get("LMJ_EXTRA_QUERY_MAX", "6")), 12))


def is_transient_graph_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = [
        "h2 protocol error", "http2 error", "rst_stream", "error code 8",
        "statuscode.cancelled", "statuscode.unavailable", "transport error",
        "connection reset", "broken pipe", "deadline exceeded",
        "no shard for cf", "invalid argument: no shard",
    ]
    return any(marker in text for marker in transient_markers)


def infer_embedding_dim(cli_dim: Optional[int]) -> int:
    if cli_dim and cli_dim > 0:
        return cli_dim
    model_path = os.environ.get("STATELET_EMBEDDING_MODEL", "").lower()
    if model_path:
        if "bge-m3" in model_path:
            return 1024
        if "multilingual-e5-base" in model_path:
            return 768
        if "multilingual-e5-small" in model_path:
            return 384
        if "multilingual-e5-large" in model_path:
            return 1024
        if "bge-base" in model_path:
            return 768
        if "bge-large" in model_path:
            return 1024
    # Default: multilingual-e5-small (cluster-start.sh default)
    return 384


# ─── Preference Query Expansion ───────────────────────────────────────────

_PREFERENCE_PATTERNS = [
    r"(?i)^can you (suggest|recommend)",
    r"(?i)^any (tips|suggestions|recommendations|advice)\b",
    r"(?i)^what should (i|we)\b",
    r"(?i)^do you have any (helpful )?(tips|suggestions|recommendations)",
    r"(?i)^(could you|please) (suggest|recommend)",
    r"(?i)^i'?m (a bit )?(anxious|worried|looking for|need(ing)?)\b",
    r"(?i)^i'?ve been having trouble\b",
]

_PREFERENCE_PREAMBLES = [
    r"^can you (?:suggest|recommend)\s+(?:some|a|an)?\s*",
    r"^any (?:tips|suggestions|recommendations|advice) (?:for|on|about|with)\s*",
    r"^what should (?:i|we) (?:serve|use|do|try|get|buy|watch|read|consider|pick)\s*",
    r"^do you have any (?:helpful )?(?:tips|suggestions|recommendations)(?: for| on| about)?\s*",
    r"^i'?m (?:a bit )?(?:anxious|worried) about\s*",
    r"^i'?ve been having trouble with\s*",
    r"^i'?m looking for\s*",
]

_PREF_STOPWORDS = {
    "my", "me", "i", "a", "an", "the", "for", "of", "to", "in", "on",
    "with", "some", "any", "this", "that", "these", "those", "current",
    "upcoming", "recent", "new", "good", "great", "useful", "helpful",
}


def _preference_query_score(question: str) -> float:
    normalized = f" {normalize_text(question)} "
    score = 0.0
    score += sum(
        1
        for token in (
            " suggest ",
            " recommend ",
            " advice ",
            " tips ",
            " looking for ",
            " what should ",
            " useful ",
            " helpful ",
            " compatible ",
            " accessor ",
            " amenity ",
            " during my ",
            " battery life ",
            " homegrown ",
            " ingredients ",
            " commute ",
            " weekend ",
        )
        if token in normalized
    ) * 0.18
    if any(token in normalized for token in (" my ", " me ", " i ")):
        score += 0.12
    if any(token in normalized for token in (
        " how many ", " how much ", " difference ", " than ", " instead of ",
    )):
        score -= 0.35
    if any(token in normalized for token in (
        " previous chat ", " previous conversation ", " remind me ", " what did you ",
    )):
        score -= 0.45
    return max(0.0, min(1.0, score))


def is_preference_query(question: str) -> bool:
    return _preference_query_score(question) >= 0.52


def _push_pref_terms(target: List[str], values: List[str], max_terms: int = 4) -> None:
    for value in values:
        norm = normalize_text(value)
        if len(norm) < 3 or norm in _PREF_STOPWORDS:
            continue
        if norm not in target:
            target.append(norm)
        if len(target) >= max_terms:
            return


def _pref_terms_after_markers(question: str, markers: Tuple[str, ...], max_terms: int = 4) -> List[str]:
    normalized = normalize_text(question)
    out: List[str] = []
    for marker in markers:
        if marker in normalized:
            rhs = normalized.split(marker, 1)[1]
            _push_pref_terms(out, rhs.split()[: max_terms * 2], max_terms)
        if len(out) >= max_terms:
            break
    return out


def _preference_setup_terms(question: str) -> List[str]:
    out: List[str] = []
    _push_pref_terms(out, _pref_terms_after_markers(question, (" for my ", " with my ", " using my ", " on my ")))
    _push_pref_terms(out, _preference_topic_keywords(question))
    return out[:4]


def _preference_context_terms(question: str) -> List[str]:
    out: List[str] = []
    _push_pref_terms(
        out,
        _pref_terms_after_markers(
            question,
            (" during my ", " this ", " lately ", " with ", " worried about ", " trouble with "),
        ),
    )
    for phrase in ("battery life", "homegrown ingredients", "this weekend", "my commute"):
        if phrase in normalize_text(question):
            _push_pref_terms(out, [phrase])
    if not out:
        _push_pref_terms(out, _preference_topic_keywords(question))
    return out[:4]


def _preference_task_terms(question: str) -> List[str]:
    normalized = normalize_text(question)
    out: List[str] = []
    for term in (
        "accessories", "accessory", "activities", "activity", "dinner", "lunch",
        "breakfast", "recipe", "recipes", "meal", "ideas", "tips", "phone",
        "camera", "battery", "commute",
    ):
        if term in normalized:
            _push_pref_terms(out, [term])
    if not out:
        _push_pref_terms(out, _preference_topic_keywords(question))
    return out[:4]


def _preference_positive_terms(question: str) -> List[str]:
    normalized = normalize_text(question)
    out: List[str] = []
    for term in ("compatible", "useful", "helpful", "durable", "protective", "premium", "quality", "showcase", "fresh"):
        if term in normalized:
            _push_pref_terms(out, [term])
    return out[:4]


def _preference_negative_terms(question: str) -> List[str]:
    return _pref_terms_after_markers(question, (" instead of ", " rather than ", " avoid ", " but not "))


def _preference_topic_keywords(question: str) -> List[str]:
    """Strip preamble and return content keywords for entity lookup."""
    text = question.strip()
    for p in _PREFERENCE_PREAMBLES:
        text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
    words = [w for w in re.findall(r"\b[a-zA-Z]\w+\b", text)
             if w.lower() not in _PREF_STOPWORDS and len(w) >= 3]
    return words[:5]


def expand_preference_extra_queries(
    db: StateletClient,
    graph: str,
    question: str,
    prelim_k: int = 10,
    max_entities: int = 3,
) -> List[str]:
    """
    For a detected preference query, do a preliminary search with topic keywords
    to find the user's specific entities (e.g. "iPhone 13 Pro" for "phone").
    Returns a list of entity strings to append to extra_queries.
    """
    topic_kw = _preference_topic_keywords(question)
    if not topic_kw:
        return []
    topic_query = " ".join(topic_kw)
    try:
        prelim = db.text_graph_search(
            graph_name=graph,
            query=topic_query,
            k=prelim_k,
            skip_server_llm=True,
        )
    except Exception:
        return []

    entities: List[str] = []
    seen: set = set()
    prelim_list = prelim.results if hasattr(prelim, 'results') else prelim
    for r in prelim_list:
        try:
            props = json.loads(r.properties)
            text = props.get("text", "").strip()
        except Exception:
            continue
        # Entity nodes are short and specific; skip long passages
        if not (4 <= len(text) <= 80):
            continue
        # Must contain at least one proper noun / capitalised token
        if not re.search(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z0-9][a-zA-Z0-9]*)*\b", text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(text)
        if len(entities) >= max_entities:
            break

    return entities


def build_extra_queries(question: str, max_queries: int = EXTRA_QUERY_MAX) -> List[str]:
    if max_queries <= 0:
        return []
    # Preserve quoted phrases and Title Case multi-word phrases before lowercasing.
    quoted_phrases = re.findall(r'"([^"]{3,})"', question)
    # Basic stopwords — deliberately keep relation/comparison words:
    #   older/younger/long/years/ago/before/after/difference/since
    # so that temporal and comparison questions retain their key signals.
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
        "had", "has", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
        "the", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
        "now", "current", "currently", "many", "much", "count", "number", "amount",
    }
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    tokens = [t for t in normalized.split() if len(t) >= 3 and t not in stopwords]
    if not tokens:
        return []
    out: List[str] = []
    seen = set()

    def push(q: str) -> None:
        q = q.strip()
        if len(q) < 3:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(q)

    # Quoted and Title-Case phrases get priority — exact proper nouns matter most.
    for phrase in quoted_phrases:
        push(phrase)
        if len(out) >= max_queries:
            return out[:max_queries]

    for n in (3, 2):
        if len(tokens) < n:
            continue
        for i in range(0, len(tokens) - n + 1):
            push(" ".join(tokens[i:i + n]))
            if len(out) >= max_queries:
                return out[:max_queries]
    return out[:max_queries]


# ─── Types ────────────────────────────────────────────────────────────────

@dataclass
class Question:
    question_id: str
    question: str
    answer: str
    question_type: str
    question_date: str
    sessions: list
    session_dates: list
    answer_session_ids: list
    haystack_session_ids: list


@dataclass
class SearchHit:
    rank: int
    node_id: int
    session_index: int
    distance: float
    text: str
    is_answer_session: bool
    is_hit_proxy: bool
    answer_span_match: bool
    token_f1: float


@dataclass
class RetrievalResult:
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    n_sessions: int
    n_answer_sessions: int
    answer_session_indices: list
    n_search_results: int
    ingest_ms: float
    search_ms: float
    # Retrieval metrics
    hit_at_k: bool
    recall_at_k: float
    mrr: float
    answer_span_hit_at_k: bool
    answer_span_recall_at_k: float
    derived_answer_hit_at_k: bool
    answer_result_hit_at_k: bool
    required_slot_coverage_at_k: float
    derivable_at_k: bool
    operand_complete_but_not_derived_at_k: bool
    best_token_f1: float
    avg_token_f1: float
    # Per-result details
    hits: list  # List[SearchHit] as dicts
    # Funnel diagnostics (Phase 1b)
    n_raw_total: int = 0          # results returned by gateway before eval-side dedup
    n_dedup_dropped: int = 0      # how many were dropped by source_text dedup


@dataclass
class DerivabilitySlotPlan:
    slot_id: str
    anchors: List[str]
    source_hint: str = ""


@dataclass
class DerivabilityPlan:
    kind: str
    required_slots: List[DerivabilitySlotPlan]


# ─── Text similarity ─────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(text: str, reference: str) -> float:
    """Token-level F1 between a search result text and the ground truth answer."""
    hyp_tokens = normalize_text(text).split()
    ref_tokens = normalize_text(reference).split()
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    common = Counter(hyp_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


DERIVED_ANSWER_PASS_THRESHOLD = float(
    os.environ.get("LONGMEMEVAL_DERIVED_PASS_THRESHOLD", "0.65")
)
STRUCTURED_EXEC_MIN_CONFIDENCE = float(
    os.environ.get("LONGMEMEVAL_STRUCTURED_MIN_CONF", "0.60")
)


def normalize_answer(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50",
    }
    for word, digit in number_words.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    text = re.sub(r"\$\s*(\d)", r"\1 dollars", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokenize_answer(text: str) -> List[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match_score(hypothesis: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(hypothesis) == normalize_answer(ground_truth) else 0.0


def token_f1_score(hypothesis: str, ground_truth: str) -> float:
    hyp_tokens = tokenize_answer(hypothesis)
    ref_tokens = tokenize_answer(ground_truth)
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    common = Counter(hyp_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def ngram_counts(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4_score(hypothesis: str, ground_truth: str) -> float:
    hyp_tokens = tokenize_answer(hypothesis)
    ref_tokens = tokenize_answer(ground_truth)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions: List[float] = []
    for n in (1, 2, 3, 4):
        hyp_ng = ngram_counts(hyp_tokens, n)
        ref_ng = ngram_counts(ref_tokens, n)
        if not hyp_ng:
            precisions.append(1e-9)
            continue
        clipped = sum(min(count, ref_ng[gram]) for gram, count in hyp_ng.items())
        total = sum(hyp_ng.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    geo_mean = math.exp(sum(math.log(max(p, 1e-12)) for p in precisions) / 4.0)
    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / max(hyp_len, 1)))
    return bp * geo_mean


def rouge_n_f1(hypothesis: str, ground_truth: str, n: int = 1) -> float:
    hyp_tokens = tokenize_answer(hypothesis)
    ref_tokens = tokenize_answer(ground_truth)
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0

    hyp_ng = ngram_counts(hyp_tokens, n)
    ref_ng = ngram_counts(ref_tokens, n)
    if not hyp_ng or not ref_ng:
        return 0.0
    overlap = sum(min(count, ref_ng[gram]) for gram, count in hyp_ng.items())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp_ng.values())
    recall = overlap / sum(ref_ng.values())
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] > prev[j] else prev[j]
        prev = curr
    return prev[-1]


def rouge_l_f1(hypothesis: str, ground_truth: str) -> float:
    hyp_tokens = tokenize_answer(hypothesis)
    ref_tokens = tokenize_answer(ground_truth)
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_judge(hypothesis: str, ground_truth: str, pass_threshold: float) -> Dict[str, float]:
    em = exact_match_score(hypothesis, ground_truth)
    f1 = token_f1_score(hypothesis, ground_truth)
    bleu4 = bleu4_score(hypothesis, ground_truth)
    rouge1 = rouge_n_f1(hypothesis, ground_truth, n=1)
    rouge_l = rouge_l_f1(hypothesis, ground_truth)
    composite = max(em, f1, bleu4, rouge1, rouge_l)
    score = 1 if composite >= pass_threshold else 0
    return {
        "score": score,
        "em": em,
        "f1": f1,
        "bleu4": bleu4,
        "rouge1_f1": rouge1,
        "rougeL_f1": rouge_l,
        "composite_score": composite,
    }


def _extract_span_from_fact(question_lower: str, fact: str) -> str:
    f_lower = fact.lower()

    if "how many" in question_lower or "how much" in question_lower:
        numbers = re.findall(r"\$?\d+(?:\.\d+)?(?:\s*(?:dollars|percent|%|years?|months?|weeks?|days?|hours?))?", fact)
        if numbers:
            return numbers[-1]

    if "how long" in question_lower:
        durations = re.findall(r"(?:over |about |approximately |nearly )?\d+\s*(?:years?|months?|weeks?|days?|hours?|minutes?)", f_lower)
        if durations:
            return durations[0]
        durations = re.findall(r"(?:over |about |nearly )?(?:a |an )?(?:few |couple )?(?:years?|months?|weeks?|days?)", f_lower)
        if durations:
            return durations[0]

    if "where" in question_lower:
        locs = re.findall(r"(?:in|at|on|under|inside|behind|near|next to|beside)\s+(?:(?:my|the|a|an)\s+)?(?:\w+\s*){1,4}", f_lower)
        if locs:
            q_words_set = set(re.findall(r"\w{3,}", question_lower))
            best = max(locs, key=lambda l: len(set(re.findall(r"\w{3,}", l)) - q_words_set))
            return best.strip().rstrip(".,;!?")

    if "spend" in question_lower or "pay" in question_lower or "cost" in question_lower:
        prices = re.findall(r"\$\d+(?:\.\d+)?(?:\s*(?:each|per|a piece))?", fact)
        if prices:
            return prices[0]

    return ""


def extract_precise_answer_span(question: str, facts: List[str]) -> str:
    if not facts:
        return ""

    q_lower = question.lower()
    q_words = set(re.findall(r"\w{3,}", q_lower))
    q_words -= {
        "what", "where", "when", "which", "how", "many", "much", "does",
        "did", "was", "were", "are", "the", "you", "your", "about", "have",
        "has", "been", "would", "should", "could", "can", "will",
    }

    scored = []
    for fact in facts:
        f_lower = fact.lower()
        f_words = set(re.findall(r"\w{3,}", f_lower))
        overlap = len(q_words & f_words)
        length_bonus = 1.0 / (1.0 + len(fact) / 100.0)
        scored.append((overlap + length_bonus, fact))
    scored.sort(key=lambda item: -item[0])

    best_span = ""
    for _, fact in scored[:5]:
        span = _extract_span_from_fact(q_lower, fact)
        if span and (not best_span or len(span) < len(best_span)):
            best_span = span
    return best_span


def extract_triple_answer(question: str, results: list) -> str:
    q_lower = question.lower()
    q_words = set(re.findall(r"\w{3,}", q_lower))
    q_words -= {
        "what", "where", "when", "which", "how", "many", "much", "does",
        "did", "was", "were", "are", "the", "you", "your", "about", "have",
        "has", "been", "would", "should", "could", "can", "will", "long",
    }

    best_obj = ""
    best_score = 0.0
    for r in results:
        props = r.get("props") or {}
        triples = props.get("triples", [])
        if isinstance(triples, str):
            try:
                triples = json.loads(triples)
            except (json.JSONDecodeError, TypeError):
                triples = []
        if not isinstance(triples, list):
            continue

        for triple in triples:
            if not isinstance(triple, dict):
                continue
            subject = str(triple.get("subject", "")).lower().replace("_", " ")
            predicate = str(triple.get("predicate", "")).lower().replace("_", " ")
            obj = str(triple.get("object", ""))
            if not obj or len(obj) > 200:
                continue

            triple_words = set(re.findall(r"\w{3,}", f"{subject} {predicate}"))
            overlap = len(q_words & triple_words)
            brevity = 1.0 / (1.0 + len(obj) / 20.0)
            score = overlap + brevity
            if "how many" in q_lower and predicate in ("count", "count_delta", "quantity"):
                score += 2.0
            if "how long" in q_lower and predicate in ("duration", "waited"):
                score += 2.0
            if "how much" in q_lower and predicate in ("price", "cost", "income"):
                score += 2.0

            if score > best_score:
                best_score = score
                best_obj = obj

    return best_obj if best_score >= 1.5 and best_obj else ""


def _coerce_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def extract_deterministic_answer_candidate(question: str, results: list) -> Dict[str, object]:
    best_answer = ""
    best_source = "none"
    best_confidence = 0.0

    for r in results:
        props = r.get("props") or {}
        exec_answer = str(props.get("computed_answer", "") or "").strip()
        exec_conf = _coerce_float(
            props.get("computed_answer_confidence", 0.0) or props.get("structured_confidence", 0.0)
        )
        if exec_answer and exec_conf >= STRUCTURED_EXEC_MIN_CONFIDENCE and exec_conf >= best_confidence:
            best_answer = exec_answer
            best_source = "structured_exec"
            best_confidence = exec_conf

    if best_answer:
        return {
            "answer": best_answer,
            "source": best_source,
            "confidence": best_confidence,
        }

    triple_answer = extract_triple_answer(question, results)
    if triple_answer:
        return {
            "answer": triple_answer,
            "source": "triple_lookup",
            "confidence": 0.6,
        }

    precise_span = extract_precise_answer_span(
        question,
        [r.get("text", "") for r in results if isinstance(r.get("text", ""), str) and len(r.get("text", "")) >= 10],
    )
    if precise_span:
        return {
            "answer": precise_span,
            "source": "precise_span",
            "confidence": 0.45,
        }

    return {
        "answer": "",
        "source": "none",
        "confidence": 0.0,
    }


DERIVABILITY_STOPWORDS = {
    "about", "after", "before", "best", "chat", "classes", "conversation", "count", "current",
    "days", "difference", "dinner", "first", "from", "good", "have", "helpful", "hotel",
    "how", "idea", "ideas", "instead", "many", "more", "move", "much", "number", "previous",
    "prior", "question", "remind", "same", "serve", "should", "since", "some", "suggest",
    "than", "what", "when", "where", "which", "with", "work", "would",
}


def salient_query_terms(text: str, max_terms: int = 6) -> List[str]:
    out: List[str] = []
    for token in normalize_text(text).split():
        if len(token) < 3 or token in DERIVABILITY_STOPWORDS:
            continue
        if token not in out:
            out.append(token)
        if len(out) >= max_terms:
            break
    return out


def phrase_after_marker(text: str, markers: List[str], max_terms: int = 4) -> List[str]:
    normalized = normalize_text(text)
    for marker in markers:
        if marker in normalized:
            rhs = normalized.split(marker, 1)[1]
            terms = salient_query_terms(rhs, max_terms)
            if terms:
                return terms
    return []


def exact_phrase_candidates(text: str) -> List[str]:
    out: List[str] = []
    for match in re.findall(r"['\"]([^'\"]+)['\"]", text):
        phrase = normalize_text(match)
        if len(phrase) >= 3 and phrase not in out:
            out.append(phrase)
    normalized = normalize_text(text)
    for marker in ("about ", "after ", "called ", "named "):
        if marker in normalized:
            phrase = " ".join(salient_query_terms(normalized.split(marker, 1)[1], 5))
            if len(phrase) >= 3 and phrase not in out:
                out.append(phrase)
    return out[:4]


def build_derivability_plan(question: str) -> Optional[DerivabilityPlan]:
    normalized = f" {normalize_text(question)} "

    if any(marker in normalized for marker in (
        " previous chat ", " previous conversation ", " previous discussion ",
        " remind me ", " what was the move ", " what did you ", " you said ", " you mentioned ",
    )):
        context_terms = exact_phrase_candidates(question) or salient_query_terms(question, 6)
        focus_terms = salient_query_terms(question, 4)
        return DerivabilityPlan(
            kind="exact_turn",
            required_slots=[
                DerivabilitySlotPlan("context_anchor", context_terms),
                DerivabilitySlotPlan("assistant_turn", focus_terms, source_hint="assistant"),
            ],
        )

    if is_preference_query(question):
        setup = _preference_setup_terms(question)
        context = _preference_context_terms(question)
        task = _preference_task_terms(question)
        positive = _preference_positive_terms(question)
        negative = _preference_negative_terms(question)
        if setup and context and task:
            slots = [
                DerivabilitySlotPlan("owned_setup_or_entity", setup),
                DerivabilitySlotPlan("current_context_or_pain", context),
                DerivabilitySlotPlan("task_intent", task),
            ]
            if positive:
                slots.append(DerivabilitySlotPlan("positive_preference", positive))
            if negative:
                slots.append(DerivabilitySlotPlan("negative_or_contrast_preference", negative))
            return DerivabilityPlan(
                kind="preference_grounding",
                required_slots=slots,
            )

    if " percent " in normalized or " percentage " in normalized:
        anchors = salient_query_terms(question, 6)
        if anchors:
            return DerivabilityPlan(
                kind="numeric_ratio",
                required_slots=[
                    DerivabilitySlotPlan("numerator_context", anchors[:3]),
                    DerivabilitySlotPlan("denominator_context", anchors[1:4] or anchors[:3]),
                    DerivabilitySlotPlan("unit", ["percent", "percentage", "%"]),
                ],
            )

    if any(marker in normalized for marker in (" instead of ", " than ", " difference ", " save ")):
        left = []
        right = []
        if " instead of " in normalized:
            lhs, rhs = normalized.split(" instead of ", 1)
            left = salient_query_terms(lhs, 3)[-3:]
            right = salient_query_terms(rhs, 3)
        elif " than " in normalized:
            lhs, rhs = normalized.split(" than ", 1)
            left = salient_query_terms(lhs, 3)[-3:]
            right = salient_query_terms(rhs, 4)
        if left and right:
            slots = [
                DerivabilitySlotPlan("left_operand", left),
                DerivabilitySlotPlan("right_operand", right),
            ]
            if any(marker in normalized for marker in (" dollar ", " year ", " years ", " percent ")):
                unit_terms = ["dollar", "$"] if " dollar " in normalized else (
                    ["year", "years"] if " year " in normalized or " years " in normalized else ["percent", "%"]
                )
                slots.append(DerivabilitySlotPlan("unit", unit_terms))
            return DerivabilityPlan(kind="numeric_delta", required_slots=slots)

    if " how many " in normalized and any(marker in normalized for marker in (" since ", " before ", " after ", " first ", " last ")):
        target = salient_query_terms(question, 6)
        boundary = phrase_after_marker(question, [" since ", " before ", " after ", " during ", " within "])
        slots = [DerivabilitySlotPlan("target", target)]
        if boundary:
            slots.append(DerivabilitySlotPlan("boundary", boundary))
        return DerivabilityPlan(kind="count_window", required_slots=slots)

    if any(marker in normalized for marker in (
        " before getting ", " before having ", " after getting ", " after having ",
        " prior to ", " right before ", " right after ", " initially ", " current ", " currently ", " now ",
    )):
        boundary = phrase_after_marker(question, [
            " before getting ", " before having ", " after getting ", " after having ",
            " prior to ", " right before ", " right after ",
        ])
        target = [t for t in salient_query_terms(question, 6) if t not in boundary]
        if boundary and target:
            return DerivabilityPlan(
                kind="state_contrast",
                required_slots=[
                    DerivabilitySlotPlan("anchor_event", boundary),
                    DerivabilitySlotPlan("target_state", target),
                ],
            )

    if any(marker in normalized for marker in (" with a friend ", " with friend ", " or not ")):
        anchor = phrase_after_marker(question, [" visiting ", " visit ", " about ", " after "])
        slots = []
        if anchor:
            slots.append(DerivabilitySlotPlan("anchor_event", anchor))
        slots.append(DerivabilitySlotPlan("relation", ["friend"]))
        return DerivabilityPlan(kind="boolean_relation", required_slots=slots)

    factoid = any(marker in normalized for marker in (
        " how much ", " how many ", " how old ", " what is ", " what was ", " who is ", " where is ",
        " birth year ", " amount ", " total ", " cost ", " name ",
    ))
    anchors = salient_query_terms(question, 6)
    if factoid and anchors:
        return DerivabilityPlan(
            kind="direct_attribute",
            required_slots=[DerivabilitySlotPlan("answer_attribute", anchors)],
        )
    return None


def result_signal_text(result: dict) -> str:
    props = result.get("props") or {}
    parts = [result.get("text", "")]
    for key in ("original_text", "domain", "subject", "slot", "predicate", "structured_predicate", "object", "category", "structured_view"):
        value = props.get(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(part for part in parts if part)


def slot_match_score(slot: DerivabilitySlotPlan, result: dict) -> float:
    signal_text = normalize_text(result_signal_text(result))
    if not signal_text:
        return 0.0
    score = 0.0
    for anchor in slot.anchors:
        anchor_norm = normalize_text(anchor)
        if not anchor_norm:
            continue
        if anchor_norm in signal_text:
            score += 1.0
            continue
        tokens = anchor_norm.split()
        overlap = sum(1 for token in tokens if token in signal_text)
        if tokens and overlap:
            score += 0.35 * (overlap / len(tokens))
    props = result.get("props") or {}
    if slot.source_hint:
        source = str(props.get("source", ""))
        if source.lower() == slot.source_hint.lower():
            score += 0.75
    if slot.slot_id in {"left_operand", "right_operand", "answer_attribute", "unit"} and any(ch.isdigit() for ch in signal_text):
        score += 0.2
    if props.get("raw_chunk") is True or props.get("category") in {"raw_chunk", "raw_text"}:
        score += 0.1
    if props.get("high_value_span") is True:
        score += 0.1
    return score


# ─── Load dataset ─────────────────────────────────────────────────────────

def load_questions(limit: Optional[int]) -> List[Question]:
    files = sorted(QUESTIONS_DIR.glob("*.json"))
    if not files:
        log.error(f"No questions found in {QUESTIONS_DIR}")
        sys.exit(1)
    questions = []
    selected_files = files if limit is None or limit <= 0 else files[:limit]
    for f in selected_files:
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
            answer_session_ids=d.get("answer_session_ids", []),
            haystack_session_ids=d.get("haystack_session_ids", []),
        ))
    return questions


# ─── Checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                cp = json.load(f)
        except json.JSONDecodeError as e:
            bad_path = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".corrupt")
            CHECKPOINT_PATH.rename(bad_path)
            log.warning("checkpoint corrupt, moved to %s", bad_path)
            cp = {}
        cp.setdefault("completed", {})
        cp.setdefault("inflight", {})
        cp.setdefault("run_id", RUN_ID)
        return cp
    return {"completed": {}, "inflight": {}, "run_id": RUN_ID}


# Process-level lock so the atomic-rename in save_checkpoint is safe
# across concurrent worker threads when --parallel-questions > 1.
_checkpoint_lock = threading.Lock()
_detail_log_lock = threading.Lock()


def save_checkpoint(cp: dict):
    cp["run_id"] = RUN_ID
    tmp_path = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".tmp")
    with _checkpoint_lock:
        with open(tmp_path, "w") as f:
            json.dump(cp, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, CHECKPOINT_PATH)


def mark_inflight(checkpoint: dict, q: Question, *, stage: str,
                  sessions_ok: int = 0, sessions_failed: int = 0,
                  sessions_total: Optional[int] = None, note: str = "") -> None:
    inflight = checkpoint.setdefault("inflight", {})
    existing = inflight.get(q.question_id, {})
    inflight[q.question_id] = {
        "question_type": q.question_type,
        "started_at": existing.get("started_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "stage": stage,
        "sessions_ok": sessions_ok,
        "sessions_failed": sessions_failed,
        "sessions_total": sessions_total if sessions_total is not None else len(q.sessions),
        "note": note,
    }
    save_checkpoint(checkpoint)


def clear_inflight(checkpoint: dict, question_id: str) -> None:
    inflight = checkpoint.get("inflight")
    if not inflight:
        return
    inflight.pop(question_id, None)
    save_checkpoint(checkpoint)


# ─── Graph / Ingest (reused from search_judge) ───────────────────────────

def build_graph_prefix(run_id: str) -> str:
    token = _sanitize_run_id(run_id).replace("_", "-")
    return f"reteval-{token}"


def build_session_idempotency_key(question_id, session_index, session_date_iso, payload_checksum):
    qid = re.sub(r"[^a-z0-9_.-]+", "-", question_id.lower()).strip("-._") or "unknown"
    date_part = session_date_iso or "nodate"
    return f"reteval:{qid}:s{session_index}:{date_part}:{payload_checksum}"


def wait_for_graph_ready(db: StateletClient, graph: str) -> None:
    deadline = time.time() + GRAPH_READY_TIMEOUT_S
    attempt = 0
    last_error = None
    while time.time() < deadline:
        attempt += 1
        try:
            db.graph_get_node(graph, 0, timeout_s=2.0)
            log.info("graph ready: %s (attempt %d)", graph, attempt)
            return
        except Exception as e:
            last_error = e
            time.sleep(min(0.5 * attempt, 2.0))
    raise RuntimeError(f"graph readiness failed for {graph}: {last_error}")


def ensure_graph_ready(db, q, graph, dim, checkpoint=None):
    last_error = None
    for attempt in range(1, GRAPH_INIT_MAX_ATTEMPTS + 1):
        if checkpoint is not None:
            mark_inflight(checkpoint, q, stage="creating_graph",
                          note=f"attempt {attempt}/{GRAPH_INIT_MAX_ATTEMPTS}")
        try:
            db.create_graph_index(graph, dim=dim, timeout_s=GRAPH_CREATE_TIMEOUT_S)
            wait_for_graph_ready(db, graph)
            return
        except Exception as e:
            last_error = e
            if is_transient_graph_error(e) and attempt < GRAPH_INIT_MAX_ATTEMPTS:
                log.warning("graph init attempt %d failed: %s", attempt, e)
                time.sleep(2 * attempt)
                continue
            if attempt < GRAPH_INIT_MAX_ATTEMPTS:
                log.warning("graph init attempt %d failed: %s", attempt, e)
                time.sleep(2 * attempt)
                continue
    raise RuntimeError(f"graph init failed for {graph}: {last_error}")


def append_ingest_outbox_records(graph, question_id, failed_payloads):
    if not failed_payloads:
        return
    try:
        INGEST_OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INGEST_OUTBOX_PATH, "a", encoding="utf-8") as fh:
            now_iso = datetime.now().isoformat()
            for p in failed_payloads:
                rec = {
                    "ts": now_iso, "run_id": RUN_ID, "graph": graph,
                    "question_id": question_id, "session_index": p["session_index"],
                    "node_id": p["node_id"], "text": p["text"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("failed to write outbox: %s", e)


def ingest_question(db, q, graph, dim, max_workers, checkpoint=None):
    ensure_graph_ready(db, q, graph, dim, checkpoint=checkpoint)
    if checkpoint is not None:
        mark_inflight(checkpoint, q, stage="ingesting", sessions_ok=0,
                      sessions_total=len(q.sessions), note="graph ready")

    t0 = time.time()
    ok = 0
    fail = 0

    def build_ingest_payload(item):
        si, session_msgs, session_date = item
        lines = []
        if session_date:
            lines.append(f"[{session_date}]")
        filtered_msgs = []
        for msg in session_msgs:
            role = (msg.get("role", "") or "").strip().lower()
            if role in {"user", "human"} or INGEST_INCLUDE_ASSISTANT:
                filtered_msgs.append(msg)
        if not filtered_msgs:
            filtered_msgs = session_msgs
        for msg in filtered_msgs:
            role = (msg.get("role", "") or "user").strip().lower()
            content = (msg.get("content", "") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        node_id = (si + 1) * 1000
        session_date_iso = session_date or ""
        if session_date_iso:
            dm = re.match(r"(\d{4})/(\d{2})/(\d{2})", session_date_iso)
            if dm:
                session_date_iso = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
        payload_checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        idempotency_key = build_session_idempotency_key(
            q.question_id, si, session_date_iso, payload_checksum)
        properties_obj = {
            "session_index": si,
            "session_date": session_date_iso,
            "ingest_run_id": RUN_ID,
            "idempotency_key": idempotency_key,
            "payload_checksum": payload_checksum,
        }
        return {
            "session_index": si, "session_date": session_date_iso,
            "node_id": node_id, "text": text,
            "idempotency_key": idempotency_key, "payload_checksum": payload_checksum,
            "properties_obj": properties_obj,
            "properties_bytes": json.dumps(properties_obj).encode(),
        }

    def do_ingest(item, *, max_attempts, base_delay_s, max_delay_s, phase):
        payload = build_ingest_payload(item)
        si = payload["session_index"]
        for attempt in range(1, max_attempts + 1):
            try:
                db.text_graph_put(
                    graph_name=graph, text=payload["text"],
                    node_id=payload["node_id"], properties=payload["properties_bytes"],
                    skip_server_llm=True, timeout_s=TEXT_GRAPH_PUT_TIMEOUT_S,
                )
                return True
            except Exception as e:
                transient = is_transient_graph_error(e)
                if transient and attempt < max_attempts:
                    jitter = 1.0 + random.uniform(0.0, 0.3)
                    backoff_s = min(max_delay_s, base_delay_s * attempt * jitter)
                    log.warning("  ingest session %d transient failure (%d/%d, %s): %s; retry in %.1fs",
                                si, attempt, max_attempts, phase, e, backoff_s)
                    time.sleep(backoff_s)
                    continue
                log.warning("  ingest session %d failed (%s): %s", si, phase, e)
                return False

    items = []
    for si, session_msgs in enumerate(q.sessions):
        sd = q.session_dates[si] if si < len(q.session_dates) else ""
        items.append((si, session_msgs, sd))

    worker_count = max(1, min(max_workers, len(items)))
    pool = ThreadPoolExecutor(max_workers=worker_count)
    future_map = {
        pool.submit(do_ingest, item,
                    max_attempts=INGEST_SESSION_MAX_ATTEMPTS,
                    base_delay_s=INGEST_RETRY_BASE_DELAY_S,
                    max_delay_s=5.0, phase="parallel"): item
        for item in items
    }
    futures = list(future_map.keys())
    failed_items = []
    processed = set()
    try:
        for f in as_completed(futures, timeout=OVERALL_INGEST_TIMEOUT_S):
            processed.add(f)
            item = future_map[f]
            try:
                success = f.result()
            except Exception as e:
                success = False
                log.warning("  ingest worker crashed session %d: %s", item[0], e)
            if success:
                ok += 1
            else:
                fail += 1
                failed_items.append(item)
            if checkpoint is not None:
                mark_inflight(checkpoint, q, stage="ingesting",
                              sessions_ok=ok, sessions_failed=fail,
                              sessions_total=len(items))
    except FuturesTimeoutError:
        log.warning("  ingest timed out after %.0fs", OVERALL_INGEST_TIMEOUT_S)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for f in futures:
            if f in processed:
                continue
            item = future_map[f]
            if f.done():
                try:
                    success = f.result()
                except Exception:
                    success = False
                if success:
                    ok += 1
                else:
                    fail += 1
                    failed_items.append(item)
            else:
                fail += 1
                failed_items.append(item)

    # Recovery phase
    if failed_items:
        unique_failed = {item[0]: item for item in failed_items}
        failed_items = [unique_failed[k] for k in sorted(unique_failed.keys())]
        log.warning("  ingest recovery: retrying %d failed sessions serially", len(failed_items))
        recovered = 0
        still_failed = []
        for item in failed_items:
            success = do_ingest(item,
                                max_attempts=INGEST_RECOVERY_MAX_ATTEMPTS,
                                base_delay_s=INGEST_RECOVERY_BASE_DELAY_S,
                                max_delay_s=INGEST_RECOVERY_MAX_DELAY_S,
                                phase="serial_recovery")
            if success:
                recovered += 1
            else:
                still_failed.append(item)
        if recovered:
            ok += recovered
            fail -= recovered
        failed_items = still_failed

    if failed_items:
        failed_ids = [si for si, _, _ in failed_items]
        failed_payloads = [build_ingest_payload(item) for item in failed_items]
        append_ingest_outbox_records(graph, q.question_id, failed_payloads)
        raise RuntimeError(
            f"ingest incomplete: {len(failed_items)}/{len(items)} sessions failed "
            f"(session_ids={failed_ids[:20]})")

    ms = (time.time() - t0) * 1000
    return ok, ms


# ─── Search ───────────────────────────────────────────────────────────────

SEARCH_MAX_RETRIES = 3
SEARCH_RETRY_BASE_DELAY_S = 2.0


def search_question(db: StateletClient, q: Question, graph: str, k: int):
    """Search and return raw results with node_id preserved. Returns (results, search_ms)."""
    t0 = time.time()

    iso_date = ""
    if q.question_date:
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", q.question_date)
        if m:
            iso_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    search_query = f"[Context date: {iso_date}] {q.question}" if iso_date else q.question

    extra_queries = build_extra_queries(q.question)
    # NOTE: preference augmentation block removed — referenced
    # build_preference_augmented_extra_queries / merge_query_lists which
    # were never defined, causing every single-session-preference
    # question to error out (recorded as miss). Same effect as the
    # function returning empty.
    if extra_queries:
        log.info("  search extra_queries=%s", extra_queries)

    raw_results = None
    last_error = None
    for attempt in range(1, SEARCH_MAX_RETRIES + 1):
        try:
            raw_results = db.text_graph_search(
                graph_name=graph,
                query=search_query,
                k=k,
                skip_server_llm=True,
                extra_queries=extra_queries,
                granularities=SEARCH_GRANULARITIES or None,
                rrf_k=SEARCH_RRF_K,
                entity_consolidate=SEARCH_ENTITY_CONSOLIDATE,
                timeout_s=TEXT_GRAPH_SEARCH_TIMEOUT_S,
            )
            break
        except Exception as e:
            last_error = e
            if is_transient_graph_error(e) and attempt < SEARCH_MAX_RETRIES:
                delay = SEARCH_RETRY_BASE_DELAY_S * attempt
                log.warning("  search attempt %d/%d failed (transient): %s; retrying in %.1fs",
                            attempt, SEARCH_MAX_RETRIES, e, delay)
                time.sleep(delay)
                continue
            raise

    # Sort by distance ascending before dedup so we keep the best-scoring copy
    # of each unique text when multiple extra_queries produce duplicate results.
    # text_graph_search returns a TextGraphSearchResponse; use .results for
    # backward-compatible access to the combined result list.
    raw_list = raw_results.results if hasattr(raw_results, 'results') else raw_results
    raw_sorted = sorted(raw_list, key=lambda r: r.distance)
    n_raw_total = len(raw_sorted)
    results = []
    seen = set()
    n_dedup_dropped = 0
    n_short_dropped = 0
    n_parse_dropped = 0
    for r in raw_sorted:
        try:
            props = json.loads(r.properties)
            # (A) Prefer source_text (verbatim sentence) over text (which can
            # be a synthesized "subject predicate object" string for
            # frame_extract / event_state facts and unsuitable for LLM context).
            source_text = props.get("source_text") or ""
            display_text = source_text if source_text else props.get("text", "")
            text = display_text
            if len(text) < 20:
                n_short_dropped += 1
                continue
            if text[:200] in seen:
                n_dedup_dropped += 1
                continue
            seen.add(text[:200])
            results.append({
                "node_id": r.node_id,
                "distance": r.distance,
                "text": text,
                "props": props,
            })
        except (json.JSONDecodeError, AttributeError):
            n_parse_dropped += 1

    ms = (time.time() - t0) * 1000
    funnel = {
        "n_raw_total": n_raw_total,
        "n_after_dedup": len(results),
        "n_dedup_dropped": n_dedup_dropped,
        "n_short_dropped": n_short_dropped,
        "n_parse_dropped": n_parse_dropped,
    }
    return results, ms, funnel


# ─── Evaluation ───────────────────────────────────────────────────────────

def evaluate_retrieval(q: Question, results: list, k: int) -> RetrievalResult:
    """Evaluate retrieval quality for a single question."""
    # Build answer session index set
    answer_id_set = set(q.answer_session_ids)
    answer_indices = set()
    for i, sid in enumerate(q.haystack_session_ids):
        if sid in answer_id_set:
            answer_indices.add(i)

    # Evaluate each result
    hits = []
    first_hit_rank = None
    hit_count = 0
    answer_sessions_hit = set()
    answer_span_sessions_hit = set()
    derivability_plan = build_derivability_plan(q.question)
    slot_best_scores: Dict[str, float] = {}

    def derive_session_index(result: dict) -> int:
        props = result.get("props") or {}
        if isinstance(props, dict):
            si = props.get("session_index")
            if isinstance(si, int):
                return si
            parent_node_id = props.get("parent_node_id")
            if isinstance(parent_node_id, int) and parent_node_id >= 1000:
                return (parent_node_id // 1000) - 1

        node_id = result["node_id"]
        is_raw_chunk = False
        if isinstance(props, dict):
            is_raw_chunk = (
                props.get("raw_chunk") is True
                or props.get("category") in {"raw_chunk", "raw_text"}
                or props.get("source") == "session_raw"
            )
        if is_raw_chunk:
            return -1

        # Recover session_index from fact/session node_id.
        # Ingest sets: session_node_id = (session_index + 1) * 1000
        # Gateway sets: fact_id = session_node_id * 1000 + fact_offset
        # So: fact_id // 1000 = session_node_id, session_node_id // 1000 - 1 = session_index
        if node_id >= 1000000:
            # Fact node: recover via two divisions
            session_node_id = node_id // 1000
            return (session_node_id // 1000) - 1
        if node_id >= 1000:
            # Session node
            return (node_id // 1000) - 1
        return -1

    for rank, r in enumerate(results[:k]):
        node_id = r["node_id"]
        session_index = derive_session_index(r)
        is_answer_session = session_index in answer_indices
        f1 = token_f1(r["text"], q.answer)
        answer_span_match = is_answer_session and f1 > 0.0
        # Text-based fallback: if session_index mapping missed, check text overlap.
        is_hit_proxy = not is_answer_session and f1 > 0.3
        is_hit = is_answer_session or is_hit_proxy

        if is_hit:
            hit_count += 1
            if is_answer_session:
                answer_sessions_hit.add(session_index)
            if first_hit_rank is None:
                first_hit_rank = rank + 1  # 1-indexed
        if answer_span_match:
            answer_span_sessions_hit.add(session_index)

        if derivability_plan is not None:
            for slot in derivability_plan.required_slots:
                score = slot_match_score(slot, r)
                if score > slot_best_scores.get(slot.slot_id, 0.0):
                    slot_best_scores[slot.slot_id] = score

        hits.append(SearchHit(
            rank=rank + 1,
            node_id=node_id,
            session_index=session_index,
            distance=r["distance"],
            text=r["text"][:300],
            is_answer_session=is_answer_session,
            is_hit_proxy=is_hit_proxy,
            answer_span_match=answer_span_match,
            token_f1=f1,
        ))

    n_answer_sessions = len(answer_indices)
    hit_at_k = hit_count > 0
    recall_at_k = len(answer_sessions_hit) / n_answer_sessions if n_answer_sessions > 0 else 0.0
    mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0
    answer_span_hit_at_k = len(answer_span_sessions_hit) > 0
    answer_span_recall_at_k = (
        len(answer_span_sessions_hit) / n_answer_sessions if n_answer_sessions > 0 else 0.0
    )
    derived_candidate = extract_deterministic_answer_candidate(q.question, results[:k])
    derived_judged = (
        metric_judge(str(derived_candidate["answer"]), q.answer, DERIVED_ANSWER_PASS_THRESHOLD)
        if derived_candidate.get("answer")
        else None
    )
    derived_answer_hit_at_k = (
        not answer_span_hit_at_k
        and derived_judged is not None
        and derived_judged.get("score", 0) == 1
    )
    answer_result_hit_at_k = answer_span_hit_at_k or derived_answer_hit_at_k
    f1_scores = [h.token_f1 for h in hits]
    best_f1 = max(f1_scores) if f1_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    if derivability_plan and derivability_plan.required_slots:
        # Tightened slot-coverage threshold (was > 0.0):
        # require a full-anchor match (score >= 1.0) before counting
        # the slot as covered. Partial-overlap matches (e.g. just one
        # token of "Sunday mass at St. Mary's Church" appearing in a
        # chunk) added 0.35 of score and trivially crossed > 0.0,
        # producing SlotCov=1.0 false positives on temporal-reasoning
        # questions where the actual operand date never made it into
        # top-K. This change makes the operand-complete signal honest:
        # SlotCov=1.0 now means EVERY operand had a chunk that matched
        # the anchor text in full, not just shared a token.
        SLOT_COVERAGE_THRESHOLD = 1.0
        covered_slots = sum(
            1 for slot in derivability_plan.required_slots
            if slot_best_scores.get(slot.slot_id, 0.0) >= SLOT_COVERAGE_THRESHOLD
        )
        required_slot_coverage_at_k = covered_slots / len(derivability_plan.required_slots)
        if derivability_plan.kind in {"numeric_delta", "numeric_ratio", "count_window", "state_contrast", "boolean_relation"}:
            derivable_at_k = required_slot_coverage_at_k >= 0.999
        else:
            derivable_at_k = required_slot_coverage_at_k >= 0.999 and (hit_at_k or best_f1 > 0.10)
    else:
        required_slot_coverage_at_k = 0.0
        derivable_at_k = False
    operand_complete_but_not_derived_at_k = (
        derivability_plan is not None
        and bool(derivability_plan.required_slots)
        and required_slot_coverage_at_k >= 0.999
        and not answer_span_hit_at_k
    )

    return RetrievalResult(
        question_id=q.question_id,
        question_type=q.question_type,
        question=q.question,
        ground_truth=q.answer,
        n_sessions=len(q.sessions),
        n_answer_sessions=n_answer_sessions,
        answer_session_indices=sorted(answer_indices),
        n_search_results=len(results),
        ingest_ms=0.0,  # filled later
        search_ms=0.0,  # filled later
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        answer_span_hit_at_k=answer_span_hit_at_k,
        answer_span_recall_at_k=answer_span_recall_at_k,
        derived_answer_hit_at_k=derived_answer_hit_at_k,
        answer_result_hit_at_k=answer_result_hit_at_k,
        required_slot_coverage_at_k=required_slot_coverage_at_k,
        derivable_at_k=derivable_at_k,
        operand_complete_but_not_derived_at_k=operand_complete_but_not_derived_at_k,
        best_token_f1=best_f1,
        avg_token_f1=avg_f1,
        hits=[asdict(h) for h in hits],
    )


# ─── Top-K dump (Phase 0 diagnostic) ─────────────────────────────────────

def classify_chunk_type(props: dict) -> str:
    """Map a result's properties JSON to a coarse chunk-type label for dump tables."""
    if not isinstance(props, dict):
        return "unknown"
    if props.get("pseudo_question"):
        return "pseudo_q"
    cat = (props.get("category") or "").lower()
    src = (props.get("source") or "").lower()
    if cat == "session_summary":
        return "summary"
    if cat in {"raw_chunk", "raw_text"} or props.get("raw_chunk") is True or src == "session_raw":
        return "raw_chunk"
    if cat == "fact":
        st = (props.get("source_text") or props.get("text") or "").strip()
        if 15 <= len(st) <= 80 and st[:2].lower() in {"i ", "i'", "my", "i’"}:
            return "atomic_i"
        return "fact"
    return cat or "other"


def dump_topk(q, results, result, k, funnel=None):
    """Print a detailed top-K composition table for one question (Phase 0 diagnostic)."""
    n_returned = len(results)
    fill_pct = (n_returned / k * 100) if k > 0 else 0.0

    type_counts = defaultdict(int)
    for r in results[:k]:
        type_counts[classify_chunk_type(r.get("props") or {})] += 1
    composition = ", ".join(f"{t}:{c}" for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]))

    log.info("─" * 100)
    log.info(f"  DUMP qid={q.question_id} type={q.question_type}")
    log.info(f"  Q : {q.question}")
    log.info(f"  GT: {(q.answer or '')[:240]}")
    if funnel:
        log.info(
            f"  Funnel: gateway_raw={funnel['n_raw_total']} → after_dedup={funnel['n_after_dedup']} "
            f"(dedup_dropped={funnel['n_dedup_dropped']}, short_dropped={funnel['n_short_dropped']}) "
            f"| {n_returned}/{k} ({fill_pct:.0f}% filled)"
        )
    else:
        log.info(f"  Funnel: {n_returned}/{k} returned ({fill_pct:.0f}% filled)")
    log.info(
        f"  Eval: hit={result.hit_at_k} span={result.answer_span_hit_at_k} "
        f"deriv={result.derivable_at_k} slot_cov={result.required_slot_coverage_at_k:.2f}"
    )
    log.info(f"  Top-K composition: {composition or '(empty)'}")
    log.info(f"  {'#':>2} {'type':<11} {'sess':>4} {'dist':>6} {'span':<5} {'f1':>5}  text")
    hits_by_rank = {h["rank"]: h for h in result.hits}
    for i, r in enumerate(results[:k]):
        rank = i + 1
        h = hits_by_rank.get(rank, {})
        ctype = classify_chunk_type(r.get("props") or {})
        sess = h.get("session_index", -1)
        dist = r.get("distance", 0.0)
        span = "✓" if h.get("answer_span_match") else " "
        ans  = "A" if h.get("is_answer_session") else (" " if not h.get("is_hit_proxy") else "p")
        f1 = h.get("token_f1", 0.0)
        text = (r.get("text") or "").replace("\n", " ")[:120]
        log.info(f"  {rank:>2} {ctype:<11} {sess:>4} {dist:>6.3f} {ans}{span:<4} {f1:>5.2f}  {text}")
    log.info("─" * 100)


def load_question_ids(path: str):
    """Load a question-id list from JSON ({question_ids:[...]}) or plain text (one per line)."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.error(f"--question-ids file not found: {path}")
        sys.exit(1)
    raw = p.read_text()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "question_ids" in data:
            return [str(x) for x in data["question_ids"]]
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]


# ─── Process question ────────────────────────────────────────────────────

def process_question(db, qi, q, total, dim, k, run_graph_prefix, max_workers,
                     checkpoint=None, skip_ingest=False, dump_topk_flag=False):
    graph = f"{run_graph_prefix}-{q.question_id}"

    log.info(f"Q{qi+1}/{total} [{q.question_type}] {q.question[:70]}")
    log.info(f"  Expected: {q.answer[:70]}")
    log.info(f"  Answer sessions: {q.answer_session_ids}")
    log.info(f"  Graph: {graph}")

    # 1. Ingest (skip if reusing existing graphs)
    if skip_ingest:
        n_facts = len(q.sessions)
        ingest_ms = 0.0
        log.info(f"  Ingest: SKIPPED (reusing existing graph)")
    else:
        try:
            n_facts, ingest_ms = ingest_question(db, q, graph, dim, max_workers, checkpoint)
        except Exception as e:
            if checkpoint is not None:
                mark_inflight(checkpoint, q, stage="ingest_failed", note=str(e)[:200])
            raise
        log.info(f"  Ingest: {n_facts} sessions in {ingest_ms:.0f}ms")
    if checkpoint is not None:
        mark_inflight(checkpoint, q, stage="searching",
                      sessions_ok=n_facts, sessions_total=len(q.sessions))

    # 2. Search
    results, search_ms, funnel = search_question(db, q, graph, k)
    log.info(
        f"  Search: {len(results)} results in {search_ms:.0f}ms "
        f"(raw={funnel['n_raw_total']}, dedup_dropped={funnel['n_dedup_dropped']}, "
        f"short_dropped={funnel['n_short_dropped']})"
    )

    # 3. Evaluate retrieval
    result = evaluate_retrieval(q, results, k)
    result.ingest_ms = ingest_ms
    result.search_ms = search_ms
    result.n_raw_total = funnel["n_raw_total"]
    result.n_dedup_dropped = funnel["n_dedup_dropped"]

    # Log hits
    for h in result.hits:
        marker = ""
        if h["is_answer_session"]:
            marker = " <<< ANSWER"
        elif h.get("is_hit_proxy"):
            marker = " <<< PROXY"
        if h.get("answer_span_match"):
            marker += " SPAN"
        log.info(f"    [{h['rank']:2d}] s={h['session_index']:3d} d={h['distance']:.3f} f1={h['token_f1']:.3f}{marker} | {h['text'][:80]}")

    hit_label = "HIT" if result.hit_at_k else "MISS"
    log.info(
        f"  Result: {hit_label} | "
        f"Hit@{k}={result.hit_at_k} "
        f"Avg Recall@{k}={result.recall_at_k:.3f} "
        f"Avg MRR={result.mrr:.3f} "
        f"Avg Best Token-F1={result.best_token_f1:.3f} "
        f"Avg Token-F1={result.avg_token_f1:.3f} "
        f"Answer-Span Hit@{k}={result.answer_span_hit_at_k} "
        f"Avg Answer-Span Recall@{k}={result.answer_span_recall_at_k:.3f} "
        f"Derived-Answer Hit@{k}={result.derived_answer_hit_at_k} "
        f"Answer-Result Hit@{k}={result.answer_result_hit_at_k} "
        f"Avg Required Slot Coverage@{k}={result.required_slot_coverage_at_k:.3f} "
        f"Avg Derivable@{k}={result.derivable_at_k} "
        f"Operand-Complete-But-Not-Derived@{k}={result.operand_complete_but_not_derived_at_k} "
        f"n_sessions={result.n_sessions} "
        f"n_answer_sessions={result.n_answer_sessions} "
        f"n_search_results={result.n_search_results} "
        f"ingest_ms={result.ingest_ms:.0f} "
        f"search_ms={result.search_ms:.0f}"
    )

    if dump_topk_flag:
        dump_topk(q, results, result, k, funnel=funnel)

    return result


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LongMemEval retrieval quality evaluation")
    parser.add_argument("--addr", default="127.0.0.1:9379", help="Gateway gRPC address")
    parser.add_argument("--mgmt-addr", default="127.0.0.1:9380", help="Gateway management HTTP address")
    parser.add_argument("--user", default="admin", help="Auth username")
    parser.add_argument("--password", default="admin", help="Auth password")
    parser.add_argument("--limit", type=int, default=100, help="Number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions")
    parser.add_argument("--dim", type=int, default=0, help="Embedding dimension (0=auto)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k results to evaluate (default: 20)")
    parser.add_argument("--force", action="store_true", help="Start fresh")
    parser.add_argument("--retry-failed", action="store_true", help="Retry missed only")
    parser.add_argument("--only", type=str, default="", help="Comma-separated question IDs")
    parser.add_argument("--question-ids", type=str, default="",
                        help="Path to JSON ({question_ids:[...]}) or text file with question IDs (one per line)")
    parser.add_argument("--dump-topk", action="store_true",
                        help="Print detailed Top-K composition table per question (Phase 0 diagnostic)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Concurrent ingest workers (within-question)")
    parser.add_argument("--parallel-questions", type=int, default=1,
                        help="Run this many questions in parallel via outer ThreadPoolExecutor. "
                             "When > 1, sets --max-workers=1 inside each question (sessions stay serial), "
                             "so total concurrency = parallel_questions × 1.")
    parser.add_argument("--run-id", type=str, default="", help="Stable run ID for namespace")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Override checkpoint path")
    parser.add_argument("--allow-concurrent", action="store_true", help="Allow concurrent runs")
    parser.add_argument(
        "--reuse-graph-prefix", type=str, default="",
        help="Reuse already-ingested graphs from a prior run (e.g. 'nollm-20260409-055519-44220'). "
             "Skips ingest entirely, only does search + eval.",
    )
    # (#827 LongMemEval Phase 5a) Multi-granularity ingest + RRF fusion. Phase 0
    # diagnostic: compare sentence-only vs +multi-granularity RRF recall@k.
    parser.add_argument(
        "--granularities", type=str, default="",
        help="Comma-separated granularity set for multi-granularity RRF fusion "
             "(subset of sentence,round,session,fact). Empty ⇒ legacy sentence-only. "
             "Example: --granularities sentence,round,session,fact . The gateway must "
             "also run with STATELET_MULTI_GRANULARITY=1 so the round/session indexes exist.",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=0,
        help="RRF constant k in score=Σ_g 1/(k+rank_g) for multi-granularity fusion "
             "(0 ⇒ server default 60).",
    )
    # (#830 LongMemEval Phase 5d) Read-time entity consolidation on/off flag so
    # multi-session recall@k is measured both ways (acceptance metric). The gateway
    # must run with STATELET_ENTITY_RESOLVE=1 so the `gcanon:` clusters this
    # dereferences were written at ingest / by the 5c ResolveEntities batch.
    parser.add_argument(
        "--entity-consolidate", action="store_true",
        help="(#830 Phase 5d) Dereference each query entity term through its "
             "persisted gcanon: identity cluster and OR-expand retrieval over the "
             "cluster's surfaces, so cross-session evidence under any surface form "
             "is recalled. Requires the gateway to run with STATELET_ENTITY_RESOLVE=1.",
    )
    args = parser.parse_args()

    global SEARCH_GRANULARITIES, SEARCH_RRF_K, SEARCH_ENTITY_CONSOLIDATE
    SEARCH_GRANULARITIES = [
        g.strip() for g in args.granularities.split(",") if g.strip()
    ]
    SEARCH_RRF_K = int(args.rrf_k)
    SEARCH_ENTITY_CONSOLIDATE = bool(args.entity_consolidate)
    if SEARCH_GRANULARITIES:
        log.info(
            "  Multi-granularity RRF: granularities=%s rrf_k=%s",
            ",".join(SEARCH_GRANULARITIES), SEARCH_RRF_K or "default(60)",
        )
    if SEARCH_ENTITY_CONSOLIDATE:
        log.info(
            "  Read-time entity consolidation: ON "
            "(gateway must run with STATELET_ENTITY_RESOLVE=1)",
        )

    configure_run_context(args.run_id, args.checkpoint_path)
    skip_ingest = bool(args.reuse_graph_prefix)
    if skip_ingest:
        run_graph_prefix = args.reuse_graph_prefix
    else:
        run_graph_prefix = build_graph_prefix(RUN_ID)
    args.max_workers = max(1, args.max_workers)

    if not args.allow_concurrent:
        try:
            lock_fp = acquire_single_instance_lock()
            atexit.register(release_single_instance_lock, lock_fp)
        except RuntimeError as e:
            log.error(str(e))
            sys.exit(2)

    log.info("=" * 80)
    log.info(f"LongMemEval Retrieval Eval — Run {RUN_ID}")
    log.info(f"  Gateway: {args.addr} | k={args.k}")
    log.info(f"  Limit: {args.limit} | Workers: {args.max_workers}")
    log.info(f"  Graph prefix: {run_graph_prefix}")
    if skip_ingest:
        log.info(f"  Mode: SEARCH-ONLY (reusing graphs from '{args.reuse_graph_prefix}')")
    else:
        log.info(f"  Mode: FULL (ingest + search)")
    log.info(f"  Log: {LOG_PATH}")
    log.info(f"  Detail: {DETAIL_LOG_PATH}")
    log.info(f"  Checkpoint: {CHECKPOINT_PATH}")
    log.info("=" * 80)

    only_ids = {qid.strip() for qid in args.only.split(",") if qid.strip()} or None
    file_ids = load_question_ids(args.question_ids)
    if file_ids:
        only_ids = (only_ids or set()) | set(file_ids)
        log.info(f"Loaded {len(file_ids)} question IDs from {args.question_ids}")
    questions = load_questions(None if only_ids else (args.offset + args.limit if args.offset > 0 else args.limit))
    if args.offset > 0 and not only_ids:
        questions = questions[args.offset:]
    if only_ids:
        questions = [q for q in questions if q.question_id in only_ids]
        found_ids = {q.question_id for q in questions}
        missing_ids = sorted(only_ids - found_ids)
        if missing_ids:
            log.warning(f"--only IDs not found in dataset: {missing_ids}")
        if not questions:
            log.error("No questions selected after applying --only")
            sys.exit(1)
    log.info(f"Loaded {len(questions)} questions")

    type_counts = defaultdict(int)
    for q in questions:
        type_counts[q.question_type] += 1
    for t, c in sorted(type_counts.items()):
        log.info(f"  {t}: {c}")

    checkpoint = load_checkpoint()
    cp_run_id = checkpoint.get("run_id")
    if cp_run_id and cp_run_id != RUN_ID and not args.force:
        log.warning("Checkpoint run_id=%s differs from current %s", cp_run_id, RUN_ID)
    if args.force:
        checkpoint = {"completed": {}, "inflight": {}, "run_id": RUN_ID}
        save_checkpoint(checkpoint)
        log.info("Fresh start (--force)")

    if only_ids:
        for qid in list(only_ids):
            checkpoint["completed"].pop(qid, None)
        save_checkpoint(checkpoint)
        log.info(f"Re-running: {only_ids}")
    elif args.retry_failed:
        missed = [qid for qid, r in checkpoint["completed"].items() if not r.get("hit_at_k")]
        for qid in missed:
            del checkpoint["completed"][qid]
        save_checkpoint(checkpoint)
        log.info(f"Retrying {len(missed)} missed")

    n_done = len(checkpoint["completed"])
    if n_done > 0:
        n_hit = sum(1 for r in checkpoint["completed"].values() if r.get("hit_at_k"))
        log.info(f"Checkpoint: {n_done} done ({n_hit} hit, {n_done - n_hit} miss)")
    if checkpoint.get("inflight"):
        checkpoint["inflight"] = {}
        save_checkpoint(checkpoint)

    # Login to get JWT token (gateway has auth enabled).
    # Retry login since gateway may still be loading models.
    mgmt_addr = args.addr.replace(":9379", ":9380")
    jwt_token = None
    for attempt in range(1, 31):
        try:
            jwt_token = StateletClient.login(mgmt_addr, "admin", "admin")
            log.info(f"Authenticated as admin via login (attempt {attempt})")
            break
        except Exception:
            if attempt == 30:
                jwt_token = "dummy-token-auth-disabled"
                log.info(f"Login failed after 30 attempts, using dummy token")
            else:
                time.sleep(2)
    db = StateletClient(
        args.addr, token=jwt_token,
        ping_timeout_s=PING_TIMEOUT_S,
        graph_admin_timeout_s=GRAPH_CREATE_TIMEOUT_S,
        text_graph_put_timeout_s=TEXT_GRAPH_PUT_TIMEOUT_S,
        text_graph_search_timeout_s=TEXT_GRAPH_SEARCH_TIMEOUT_S,
    )
    for attempt in range(1, 16):
        try:
            pong = db.ping()
            log.info(f"Connected: {pong} (attempt {attempt})")
            break
        except Exception as e:
            if attempt == 15:
                raise
            log.warning(f"Gateway not ready ({attempt}/15): {e}")
            time.sleep(2)
    time.sleep(3)
    graph_dim = infer_embedding_dim(args.dim)
    log.info(f"Embedding dim={graph_dim}")

    # Run
    start_time = time.time()

    # Mode selection: if parallel_questions > 1, run questions concurrently
    # via ThreadPoolExecutor and force max_workers=1 inside each question
    # (sessions stay serial). Total concurrency = parallel_questions × 1
    # ingest stream per question = parallel_questions concurrent streams.
    parallel_q = max(1, args.parallel_questions)
    inner_workers = 1 if parallel_q > 1 else args.max_workers
    if parallel_q > 1:
        log.info(
            f"[parallel-questions] running {parallel_q} questions concurrently, "
            f"inner max_workers={inner_workers} (sessions serial within each question)"
        )

    # Lock to serialize checkpoint/detail writes across worker threads.
    write_lock = threading.Lock()

    def _record_success(q: Question, qi: int, result: RetrievalResult) -> None:
        with write_lock:
            checkpoint["completed"][q.question_id] = {
                "question_type": result.question_type,
                "question": result.question[:200],
                "ground_truth": result.ground_truth[:200],
                "n_sessions": result.n_sessions,
                "n_answer_sessions": result.n_answer_sessions,
                "answer_session_indices": result.answer_session_indices,
                "n_search_results": result.n_search_results,
                "n_raw_total": result.n_raw_total,
                "n_dedup_dropped": result.n_dedup_dropped,
                "ingest_ms": result.ingest_ms,
                "search_ms": result.search_ms,
                "hit_at_k": result.hit_at_k,
                "recall_at_k": result.recall_at_k,
                "mrr": result.mrr,
                "answer_span_hit_at_k": result.answer_span_hit_at_k,
                "answer_span_recall_at_k": result.answer_span_recall_at_k,
                "derived_answer_hit_at_k": result.derived_answer_hit_at_k,
                "answer_result_hit_at_k": result.answer_result_hit_at_k,
                "required_slot_coverage_at_k": result.required_slot_coverage_at_k,
                "derivable_at_k": result.derivable_at_k,
                "operand_complete_but_not_derived_at_k": result.operand_complete_but_not_derived_at_k,
                "best_token_f1": result.best_token_f1,
                "avg_token_f1": result.avg_token_f1,
            }
            clear_inflight(checkpoint, q.question_id)
            save_checkpoint(checkpoint)
            with _detail_log_lock:
                with open(DETAIL_LOG_PATH, "a") as f:
                    f.write(json.dumps(asdict(result), default=str) + "\n")

    def _record_failure(q: Question, qi: int, e: Exception) -> None:
        log.error(f"Q{qi+1} ERROR: {e}")
        log.error(traceback.format_exc())
        with write_lock:
            checkpoint["completed"][q.question_id] = {
                "question_type": q.question_type,
                "question": q.question[:200],
                "ground_truth": q.answer[:200],
                "n_sessions": len(q.sessions),
                "n_answer_sessions": len(q.answer_session_ids),
                "answer_session_indices": [],
                "n_search_results": 0,
                "ingest_ms": 0, "search_ms": 0,
                "hit_at_k": False, "recall_at_k": 0.0,
                "mrr": 0.0,
                "answer_span_hit_at_k": False,
                "answer_span_recall_at_k": 0.0,
                "derived_answer_hit_at_k": False,
                "answer_result_hit_at_k": False,
                "required_slot_coverage_at_k": 0.0,
                "derivable_at_k": False,
                "operand_complete_but_not_derived_at_k": False,
                "best_token_f1": 0.0,
                "avg_token_f1": 0.0,
                "error": str(e)[:200],
            }
            clear_inflight(checkpoint, q.question_id)
            save_checkpoint(checkpoint)

    def _run_question(qi: int, q: Question) -> None:
        if q.question_id in checkpoint["completed"]:
            if "error" not in checkpoint["completed"][q.question_id]:
                return
        try:
            with write_lock:
                mark_inflight(checkpoint, q, stage="starting")
            result = process_question(
                db, qi, q, len(questions), graph_dim, args.k,
                run_graph_prefix, inner_workers, checkpoint,
                skip_ingest=skip_ingest, dump_topk_flag=args.dump_topk)
            _record_success(q, qi, result)
        except Exception as e:
            _record_failure(q, qi, e)

    if parallel_q > 1:
        with ThreadPoolExecutor(max_workers=parallel_q) as outer_pool:
            futs = [outer_pool.submit(_run_question, qi, q) for qi, q in enumerate(questions)]
            for f in as_completed(futs):
                # propagate any uncaught exception (record_failure swallows
                # the per-question kind, so this should be rare).
                try:
                    f.result()
                except Exception as outer_e:
                    log.error(f"outer worker exception: {outer_e}")
    else:
        for qi, q in enumerate(questions):
            _run_question(qi, q)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    all_results = checkpoint["completed"]
    total = len(all_results)

    n_hit = sum(1 for r in all_results.values() if r.get("hit_at_k"))
    n_answer_span_hit = sum(1 for r in all_results.values() if r.get("answer_span_hit_at_k"))
    n_derived_answer_hit = sum(1 for r in all_results.values() if r.get("derived_answer_hit_at_k"))
    n_answer_result_hit = sum(1 for r in all_results.values() if r.get("answer_result_hit_at_k"))
    avg_recall = sum(r.get("recall_at_k", 0) for r in all_results.values()) / max(total, 1)
    avg_mrr = sum(r.get("mrr", 0) for r in all_results.values()) / max(total, 1)
    avg_answer_span_recall = sum(r.get("answer_span_recall_at_k", 0) for r in all_results.values()) / max(total, 1)
    avg_slot_coverage = sum(r.get("required_slot_coverage_at_k", 0) for r in all_results.values()) / max(total, 1)
    avg_derivable = sum(1 for r in all_results.values() if r.get("derivable_at_k")) / max(total, 1)
    avg_operand_gap = sum(
        1 for r in all_results.values()
        if r.get("operand_complete_but_not_derived_at_k")
    ) / max(total, 1)
    avg_best_f1 = sum(r.get("best_token_f1", 0) for r in all_results.values()) / max(total, 1)
    avg_avg_f1 = sum(r.get("avg_token_f1", 0) for r in all_results.values()) / max(total, 1)
    avg_ingest = sum(r.get("ingest_ms", 0) for r in all_results.values()) / max(total, 1)
    avg_search = sum(r.get("search_ms", 0) for r in all_results.values()) / max(total, 1)

    log.info(f"\n{'=' * 80}")
    log.info(f"RESULTS — Run {RUN_ID} (k={args.k})")
    log.info(f"{'=' * 80}")
    log.info(f"Total: {total} | Hit@{args.k}: {n_hit}/{total} ({n_hit/max(total,1)*100:.1f}%)")
    log.info(f"Avg Recall@{args.k}: {avg_recall:.3f} | Avg MRR: {avg_mrr:.3f}")
    log.info(
        f"Answer-Span Hit@{args.k}: {n_answer_span_hit}/{total} ({n_answer_span_hit/max(total,1)*100:.1f}%) | "
        f"Avg Answer-Span Recall@{args.k}: {avg_answer_span_recall:.3f}"
    )
    log.info(
        f"Derived-Answer Hit@{args.k}: {n_derived_answer_hit}/{total} ({n_derived_answer_hit/max(total,1)*100:.1f}%) | "
        f"Answer-Result Hit@{args.k}: {n_answer_result_hit}/{total} ({n_answer_result_hit/max(total,1)*100:.1f}%)"
    )
    log.info(f"Avg Required Slot Coverage@{args.k}: {avg_slot_coverage:.3f} | Derivable@{args.k}: {avg_derivable:.3f}")
    log.info(f"Operand-Complete-But-Not-Derived@{args.k}: {avg_operand_gap:.3f}")
    log.info(f"Avg Best Token-F1: {avg_best_f1:.3f} | Avg Token-F1: {avg_avg_f1:.3f}")

    # Per-type breakdown
    type_stats = defaultdict(lambda: {
        "total": 0,
        "hit": 0,
        "answer_span_hit": 0,
        "derived_answer_hit": 0,
        "answer_result_hit": 0,
        "recall": [],
        "answer_span_recall": [],
        "mrr": [],
        "slot_cov": [],
        "derivable": 0,
        "operand_gap": 0,
        "best_f1": [],
    })
    for r in all_results.values():
        t = r.get("question_type", "unknown")
        type_stats[t]["total"] += 1
        if r.get("hit_at_k"):
            type_stats[t]["hit"] += 1
        if r.get("answer_span_hit_at_k"):
            type_stats[t]["answer_span_hit"] += 1
        if r.get("derived_answer_hit_at_k"):
            type_stats[t]["derived_answer_hit"] += 1
        if r.get("answer_result_hit_at_k"):
            type_stats[t]["answer_result_hit"] += 1
        type_stats[t]["recall"].append(r.get("recall_at_k", 0))
        type_stats[t]["answer_span_recall"].append(r.get("answer_span_recall_at_k", 0))
        type_stats[t]["mrr"].append(r.get("mrr", 0))
        type_stats[t]["slot_cov"].append(r.get("required_slot_coverage_at_k", 0))
        if r.get("derivable_at_k"):
            type_stats[t]["derivable"] += 1
        if r.get("operand_complete_but_not_derived_at_k"):
            type_stats[t]["operand_gap"] += 1
        type_stats[t]["best_f1"].append(r.get("best_token_f1", 0))

    log.info(f"\nPer-type breakdown:")
    log.info(
        f"  {'Type':35s} | {'Hit@k':10s} | {'SpanHit':10s} | {'DerHit':10s} | {'AnsHit':10s} | "
        f"{'Recall':8s} | {'SpanRec':8s} | {'MRR':8s} | {'SlotCov':8s} | {'Deriv':8s} | {'OpGap':8s} | {'BestF1':8s}"
    )
    log.info(f"  {'-'*158}")
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        hit_pct = s["hit"] / s["total"] * 100 if s["total"] > 0 else 0
        span_hit_pct = s["answer_span_hit"] / s["total"] * 100 if s["total"] > 0 else 0
        derived_hit_pct = s["derived_answer_hit"] / s["total"] * 100 if s["total"] > 0 else 0
        answer_result_hit_pct = s["answer_result_hit"] / s["total"] * 100 if s["total"] > 0 else 0
        avg_r = sum(s["recall"]) / len(s["recall"])
        avg_span_r = sum(s["answer_span_recall"]) / len(s["answer_span_recall"])
        avg_m = sum(s["mrr"]) / len(s["mrr"])
        avg_slot = sum(s["slot_cov"]) / len(s["slot_cov"])
        deriv_rate = s["derivable"] / s["total"] if s["total"] > 0 else 0
        operand_gap_rate = s["operand_gap"] / s["total"] if s["total"] > 0 else 0
        avg_bf = sum(s["best_f1"]) / len(s["best_f1"])
        log.info(
            f"  {t:35s} | {s['hit']:3d}/{s['total']:3d} {hit_pct:5.1f}% | "
            f"{s['answer_span_hit']:3d}/{s['total']:3d} {span_hit_pct:5.1f}% | "
            f"{s['derived_answer_hit']:3d}/{s['total']:3d} {derived_hit_pct:5.1f}% | "
            f"{s['answer_result_hit']:3d}/{s['total']:3d} {answer_result_hit_pct:5.1f}% | "
            f"{avg_r:.3f}    | {avg_span_r:.3f}    | {avg_m:.3f}    | {avg_slot:.3f}    | {deriv_rate:.3f}    | "
            f"{operand_gap_rate:.3f}    | {avg_bf:.3f}"
        )

    log.info(f"\nTiming:")
    log.info(f"  Total elapsed: {elapsed:.1f}s")
    log.info(f"  Avg ingest: {avg_ingest:.0f}ms | Avg search: {avg_search:.0f}ms")

    # Save final results
    results_path = EVAL_DIR / f"retrieval_eval_results_{RUN_ID}.json"
    with open(results_path, "w") as f:
        json.dump({
            "benchmark": "LongMemEval",
            "mode": "retrieval-only",
            "run_id": RUN_ID,
            "k": args.k,
            "total": total,
            "hit_at_k": n_hit,
            "hit_rate": n_hit / max(total, 1),
            "answer_span_hit_at_k": n_answer_span_hit,
            "answer_span_hit_rate": n_answer_span_hit / max(total, 1),
            "derived_answer_hit_at_k": n_derived_answer_hit,
            "derived_answer_hit_rate": n_derived_answer_hit / max(total, 1),
            "answer_result_hit_at_k": n_answer_result_hit,
            "answer_result_hit_rate": n_answer_result_hit / max(total, 1),
            "avg_recall_at_k": avg_recall,
            "avg_mrr": avg_mrr,
            "avg_answer_span_recall_at_k": avg_answer_span_recall,
            "avg_required_slot_coverage_at_k": avg_slot_coverage,
            "avg_derivable_at_k": avg_derivable,
            "avg_operand_complete_but_not_derived_at_k": avg_operand_gap,
            "avg_best_token_f1": avg_best_f1,
            "avg_token_f1": avg_avg_f1,
            "per_type": {t: {
                "total": s["total"], "hit": s["hit"],
                "hit_rate": s["hit"] / s["total"] if s["total"] > 0 else 0,
                "answer_span_hit": s["answer_span_hit"],
                "answer_span_hit_rate": s["answer_span_hit"] / s["total"] if s["total"] > 0 else 0,
                "derived_answer_hit": s["derived_answer_hit"],
                "derived_answer_hit_rate": s["derived_answer_hit"] / s["total"] if s["total"] > 0 else 0,
                "answer_result_hit": s["answer_result_hit"],
                "answer_result_hit_rate": s["answer_result_hit"] / s["total"] if s["total"] > 0 else 0,
                "avg_recall": sum(s["recall"]) / len(s["recall"]),
                "avg_answer_span_recall": sum(s["answer_span_recall"]) / len(s["answer_span_recall"]),
                "avg_mrr": sum(s["mrr"]) / len(s["mrr"]),
                "avg_required_slot_coverage": sum(s["slot_cov"]) / len(s["slot_cov"]),
                "avg_derivable": s["derivable"] / s["total"] if s["total"] > 0 else 0,
                "avg_operand_complete_but_not_derived": s["operand_gap"] / s["total"] if s["total"] > 0 else 0,
                "avg_best_f1": sum(s["best_f1"]) / len(s["best_f1"]),
            } for t, s in sorted(type_stats.items())},
            "timing": {
                "total_seconds": elapsed,
                "avg_ingest_ms": avg_ingest,
                "avg_search_ms": avg_search,
            },
        }, f, indent=2)
    log.info(f"\nResults saved: {results_path}")
    log.info(f"Detail log: {DETAIL_LOG_PATH}")
    log.info(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()

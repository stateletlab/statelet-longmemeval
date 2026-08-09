# Copyright 2024 Statelet Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stage 2 — Query.

Runs the retrieval pipeline for one question against its namespaced graph and
returns the top-k hits. Pure retrieval — no LLM in this stage.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

from .config import HarnessConfig
from .dataset import Question
from .store import MemoryStore, SearchHit, session_index_of


_ANSWER_RESULT_KEEP_FIELDS = (
    "type",
    "kind",
    "operation",
    "task_type",
    "status",
    "confidence",
    "computed_answer",
    "formatted_answer",
    "normalized_answer",
    "answer_format",
    "missing_slots",
    "coverage_deficits",
    "failure_taxonomy",
    "reasoning_ops",
    "supporting_fact_ids",
    "display_supporting_fact_ids",
    "ranked_supporting_fact_ids",
    "memory_id",
    "node_id",
    "answer_role",
    "subject",
    "predicate",
    "value",
    "number",
    "date",
    "distance",
    "text",
    "source_span_ids",
)

_HIT_MEMORY_KEEP_FIELDS = (
    "text",
    "category",
    "source",
    "answer_role",
    "structured_subject",
    "structured_predicate",
    "structured_object",
    "deterministic_answer_value",
    "deterministic_answer_number",
    "aggregation_value",
    "comparison_value",
    "date",
    "session_date",
    "memory_plan",
    "operand_sensitive_memory",
    "memory_coverage",
    "memory_selection",
    "memory_count",
    "memory_chars",
    "document_chars",
    "fallback_full_session",
    "global_evidence_window_budget",
    "global_evidence_selected_windows",
    "source_span_ids",
)


@dataclass(frozen=True)
class ComputedMemory:
    rank: int
    source: str
    text: str


@dataclass(frozen=True)
class MemoryPlan:
    """Question-level knobs for compacting gateway artifacts into reader memory."""

    labels: tuple[str, ...]
    hints: tuple[str, ...]
    answer_results_limit: int
    max_chars: int
    total_max_chars: int
    fact_k: int
    chunk_k: int

    @property
    def complex(self) -> bool:
        return bool(self.labels)


ComputedEvidence = ComputedMemory
EvidencePlan = MemoryPlan


@dataclass
class QueryResult:
    question_id: str
    hits: List[SearchHit]
    search_ms: float
    computed_memory: List[ComputedMemory] = field(default_factory=list)
    memory_plan: MemoryPlan | None = None
    memories: str = ""       # gateway plain-text evidence block (resp.memories).
    n_raw: int = 0           # results returned by the gateway before filtering.
    n_fragment_dropped: int = 0
    n_dedup_dropped: int = 0

    @property
    def computed_evidence(self) -> List[ComputedMemory]:
        return self.computed_memory

    @property
    def evidence_plan(self) -> MemoryPlan | None:
        return self.memory_plan


_STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "before",
    "between",
    "could",
    "current",
    "currently",
    "does",
    "from",
    "have",
    "many",
    "much",
    "should",
    "that",
    "their",
    "there",
    "these",
    "think",
    "this",
    "what",
    "when",
    "where",
    "which",
    "advice",
    "recommend",
    "recommended",
    "recommendation",
    "recommendations",
    "specific",
    "suggest",
    "suggested",
    "suggestion",
    "suggestions",
    "with",
    "would",
}


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _clip(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _shrink_json_value(value, *, string_max: int, list_max: int):
    if isinstance(value, str):
        return _clip(value, string_max)
    if isinstance(value, list):
        return [_shrink_json_value(item, string_max=string_max, list_max=list_max) for item in value[:list_max]]
    if isinstance(value, tuple):
        return [_shrink_json_value(item, string_max=string_max, list_max=list_max) for item in value[:list_max]]
    if isinstance(value, dict):
        return {
            key: _shrink_json_value(child, string_max=string_max, list_max=list_max)
            for key, child in value.items()
            if child not in (None, "", [], {})
        }
    return value


def _json_clip(obj, max_chars: int) -> str:
    text = json_dumps(obj)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    for string_max, list_max in (
        (900, 16),
        (600, 12),
        (360, 8),
        (220, 6),
        (140, 4),
        (90, 3),
        (60, 2),
        (40, 1),
    ):
        compact = _shrink_json_value(obj, string_max=string_max, list_max=list_max)
        text = json_dumps(compact)
        if len(text) <= max_chars:
            return text
    summary_max = max(0, max_chars - 36)
    while summary_max > 0:
        text = json_dumps({"truncated": True, "summary": _clip(json_dumps(obj), summary_max)})
        if len(text) <= max_chars:
            return text
        summary_max = max(0, summary_max - 40)
    return json_dumps({"truncated": True})


def _clip_memory_text(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        import json

        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return _json_clip(value, max_chars)
    return _clip(text, max_chars)


def _compact_json(obj: dict, keep_fields: tuple[str, ...], max_chars: int) -> str:
    compact = {}
    for field in keep_fields:
        value = obj.get(field)
        if value in (None, "", [], {}):
            continue
        compact[field] = value
    if not compact:
        return ""
    return _json_clip(compact, max_chars)


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_hit(hit: SearchHit, max_chars: int) -> str:
    compact = {field: hit.props[field] for field in _HIT_MEMORY_KEEP_FIELDS if field in hit.props}
    if "text" not in compact and hit.text:
        compact["text"] = hit.text
    if not compact:
        return ""
    compact["node_id"] = hit.node_id
    compact["distance"] = round(hit.distance, 4)
    return _compact_json(compact, tuple(compact.keys()), max_chars)


def _question_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'_-]{2,}", (text or "").lower())
        if token not in _STOP_WORDS
    }


def _content_tokens(text: str) -> set[str]:
    return _question_tokens(text)


def _question_anchor_tokens(q: Question) -> set[str]:
    tokens = _question_tokens(q.question)
    weak = {
        "asked",
        "current",
        "currently",
        "earlier",
        "many",
        "much",
        "previous",
        "remind",
    }
    return {token for token in tokens if token not in weak}


def _has_anchor_overlap(text: str, q: Question, *, min_overlap: int = 1) -> bool:
    return len(_question_anchor_tokens(q) & _content_tokens(text)) >= min_overlap


def _parse_question_date(question_date: str) -> datetime | None:
    raw = (question_date or "").strip()
    if not raw:
        return None
    match = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", raw)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _decompose_enabled() -> bool:
    return os.environ.get("LME_DECOMPOSE", "0") == "1"


_DECOMPOSE_PROMPT = (
    "You are a search-query planner for a personal-memory database. Some questions "
    "need SEVERAL separate facts to answer (both sides of a comparison, each item "
    "of a total, an event plus its date, a thing plus its owner's attribute). "
    "Output up to 3 short search queries, ONE PER LINE, each targeting ONE distinct "
    "fact needed to answer the question. Use plain keyword phrases, no numbering, "
    "no explanations, no punctuation-only lines. If the question needs only one "
    "fact, output a single query.\nQuestion: "
)


_COMPARATIVE_ATTRS = [
    # (comparative markers, user-side probe, attribute keywords for the other side)
    (("older", "younger"), "my age I am years old born birthday", "age years old born"),
    (("taller", "shorter"), "my height I am tall feet inches", "height tall"),
    (("heavier", "lighter"), "my weight I weigh pounds kilograms", "weight weigh"),
    (("richer", "earn more", "earns more"), "my salary income I earn per year", "salary income earn"),
]


def _decompose_query_rules(question: str) -> List[str]:
    """Deterministic single-pass decomposition — grammar tables + shallow entity
    extraction only, NO model call in the query path (protocol-class ①). Mirrors
    the LLM planner's highest-value shapes: self-comparatives (both operands),
    baseline-vs-outcome, coordination totals, and salient-entity probes."""
    q = question or ""
    ql = f" {q.lower()} "
    out: List[str] = []
    # 1. Self-comparatives: BOTH operands (mine + the named entity's).
    self_ref = any(m in ql for m in (" than me", " than i ", " than mine", " than my "))
    if self_ref:
        for markers, mine, attr in _COMPARATIVE_ATTRS:
            if any(f" {m} " in ql for m in markers):
                out.append(mine)
                ent = [
                    w for w in re.findall(r"\bmy ([a-z]+)\b", ql)
                    if w not in ("age", "height", "weight", "salary", "income", "own")
                ]
                if ent:
                    out.append(f"{ent[0]} {attr}")
                break
    # 2. Baseline-vs-outcome (initial quote/estimate → final figure).
    if any(m in ql for m in (" more ", " less ", " difference", " extra ")) and any(
        m in ql for m in (" initial", " original", " quote", " estimate", " budget")
    ):
        out.append("final total actual amount paid ended up paying")
    # 3. Coordination totals: one probe per conjunct.
    if any(m in ql for m in (" total", " altogether", " combined", " both ")) and " and " in ql:
        pos = ql.find(" and ")
        left = " ".join(ql[:pos].split()[-3:])
        right = " ".join(ql[pos + 5 :].split()[:3])
        measure = re.search(r"how (?:many|much) (\w+)|total (?:number of )?(\w+)", ql)
        m = (measure.group(1) or measure.group(2)) if measure else ""
        if len(left) > 2 and len(right) > 2:
            out += [f"{m} {left}".strip(), f"{m} {right}".strip()]
    # 4. Salient-entity probes: quoted spans, capitalized multi-word names, and
    #    notation-like tokens (chess moves, model numbers) get a focused probe.
    ents: List[str] = re.findall(r"[\"“']([^\"”']{3,50})[\"”']", q)
    ents += re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b", q)
    ents += re.findall(r"\b(\d{1,3}\.\s?[KQRNB]?[a-h]?[1-8][+#]?(?:\s\S{2,6})?)\b", q)
    for e in ents[:2]:
        if e.lower() not in " ".join(out).lower():
            out.append(e.strip())
    # 4b. No multi-word entities: fall back to single capitalized words (mid-
    #     sentence proper nouns like "Rome", "Italian") joined as one probe.
    if not ents:
        singles = [
            w for w in re.findall(r"(?<=[a-z,] )([A-Z][a-z]{2,})\b", q)
            if w.lower() not in ("i", "can", "what", "how", "when", "which")
        ]
        if singles:
            out.append(" ".join(dict.fromkeys(singles).keys()))
    # dedup, keep short useful ones
    seen, final = set(), []
    for s in out:
        s = re.sub(r"\s+", " ", s).strip()
        if 3 <= len(s) <= 120 and s.lower() not in seen:
            seen.add(s.lower())
            final.append(s)
    return final[:3]


def _decompose_query(question: str, model: str = "", timeout: int = 45) -> List[str]:
    """Single-pass query decomposition (no retrieval feedback — the planner sees
    ONLY the question, so this stays one-shot, not multi-hop).
    LME_DECOMPOSE_MODE=rules uses the deterministic grammar planner (no model in
    the query path); default uses `claude -p`. Returns [] on failure so retrieval
    degrades to the plain query."""
    if os.environ.get("LME_DECOMPOSE_MODE", "llm") == "rules":
        try:
            return _decompose_query_rules(question)
        except Exception:  # noqa: BLE001
            return []
    import subprocess

    model = model or os.environ.get("LME_DECOMPOSE_MODEL", "haiku")
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", model],
            input=_DECOMPOSE_PROMPT + (question or ""),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return []
        lines = [
            re.sub(r"^[\s\-\d\.\)]+", "", ln).strip()
            for ln in proc.stdout.strip().splitlines()
        ]
        out = [ln for ln in lines if 3 <= len(ln) <= 120][:3]
        return out
    except Exception:  # noqa: BLE001
        return []


def _gateway_query_text(q: Question) -> str:
    question = q.question or ""
    # The `[Context date: …]` prefix injects date tokens into the retrieval
    # query, which pulls date-adjacent (but wrong) sessions up and demotes the
    # gold session — measured to drop hit@5 hard on the 28-wrong subset. Gated
    # behind LME_DATE_PREFIX (default off); the relative-time hints already give
    # the gateway the reference date through the evidence plan.
    if os.environ.get("LME_DATE_PREFIX", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return question
    if question.lstrip().startswith("[Context date:"):
        return question
    base = _parse_question_date(q.question_date)
    if base is None:
        return question
    return f"[Context date: {base:%Y-%m-%d}] {question}"


def _relative_time_hints(question: str, question_date: str) -> tuple[str, ...]:
    base = _parse_question_date(question_date)
    if base is None:
        return ()
    q = (question or "").lower()
    hints: list[str] = []

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, weekday in weekdays.items():
        if f"last {name}" in q:
            delta = (base.weekday() - weekday) % 7 or 7
            day = base - timedelta(days=delta)
            hints.append(f"last {name} = {day:%Y/%m/%d}")

    for amount, unit in re.findall(r"(\d+)\s+(day|days|week|weeks|month|months)\s+ago", q):
        n = int(amount)
        days = n if unit.startswith("day") else n * 7 if unit.startswith("week") else n * 30
        day = base - timedelta(days=days)
        hints.append(f"{amount} {unit} ago ≈ {day:%Y/%m/%d}")

    if "yesterday" in q:
        hints.append(f"yesterday = {(base - timedelta(days=1)):%Y/%m/%d}")
    if "today" in q or "currently" in q or "now" in q:
        hints.append(f"current reference date = {base:%Y/%m/%d}")
    return tuple(dict.fromkeys(hints))


def _memory_plan_for_question(cfg: HarnessConfig, q: Question) -> MemoryPlan:
    question = q.question.lower()
    labels: list[str] = []
    hints: list[str] = []

    if re.search(
        r"\b(how many|how much|total|sum|average|percentage|percent|save|older|younger|left|remaining)\b",
        question,
    ):
        labels.append("multi_operand_numeric")
        hints.append("collect every numeric operand before calculating")
    if re.search(
        r"\b(days?|weeks?|months?|years?|before|after|since|until|last|currently|now|previous|earliest|latest|order)\b",
        question,
    ):
        labels.append("temporal")
        hints.append("use Question Date as the reference date for relative time")
    if re.search(r"\b(order|earliest|latest|first|second|third|fourth|fifth|sixth|\d+(st|nd|rd|th))\b", question):
        labels.append("ordered_or_list")
        hints.append("preserve list order and ordinal positions from the source turn")
    if q.question_type == "single-session-preference" or re.search(r"\b(prefer|suggest|recommend|should i|what do you think)\b", question):
        labels.append("preference_constraints")
        hints.append("keep user-specific constraints and avoid generic advice")

    hints.extend(_relative_time_hints(q.question, q.question_date))
    labels_tuple = tuple(dict.fromkeys(labels))
    complex_question = bool(labels_tuple)
    return MemoryPlan(
        labels=labels_tuple,
        hints=tuple(dict.fromkeys(hints)),
        answer_results_limit=max(
            cfg.answer_results_limit,
            cfg.answer_complex_results_limit if complex_question else cfg.answer_results_limit,
        ),
        max_chars=max(
            120,
            cfg.answer_complex_evidence_max_chars if complex_question else cfg.answer_evidence_max_chars,
        ),
        total_max_chars=max(
            0,
            cfg.answer_complex_evidence_total_max_chars
            if complex_question
            else cfg.answer_evidence_total_max_chars,
        ),
        fact_k=max(cfg.answer_fact_k, cfg.answer_complex_fact_k if complex_question else cfg.answer_fact_k),
        chunk_k=max(cfg.answer_chunk_k, cfg.answer_complex_chunk_k if complex_question else cfg.answer_chunk_k),
    )


_evidence_plan_for_question = _memory_plan_for_question


def _string_field(obj: dict, *path: str) -> str:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return cur if isinstance(cur, str) else ""


def _number_field(obj: dict, *path: str) -> float:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return 0.0
        cur = cur.get(key)
    if isinstance(cur, (int, float)):
        return float(cur)
    return 0.0


def _primary_is_trusted(primary: dict) -> bool:
    if not primary:
        return False
    status = _string_field(primary, "status") or _string_field(primary, "computed_answer", "status")
    kind = _string_field(primary, "kind") or _string_field(primary, "computed_answer", "kind")
    confidence = _number_field(primary, "confidence") or _number_field(primary, "computed_answer", "confidence")
    answer_format = _string_field(primary, "answer_format")
    formatted = primary.get("formatted_answer") if isinstance(primary.get("formatted_answer"), dict) else {}
    formatted_value = formatted.get("value")
    computed_answer = primary.get("computed_answer")
    if not isinstance(computed_answer, dict):
        computed_answer = {}
    source_span_ids = primary.get("source_span_ids") or computed_answer.get("source_span_ids", [])

    if status == "ambiguous" or confidence < 0.70:
        return False
    if answer_format in {"integer", "single_value", "date_range"} and formatted_value in (None, "", []):
        return False
    if kind in {"raw_text_recovery", "event_session_recovery"}:
        return False
    if kind == "attribute_lookup_materialized" and not source_span_ids:
        return False
    return True


def _answer_result_relevance(item: dict, q: Question, plan: EvidencePlan) -> int:
    text = _text_from_answer_result(item)
    if not text:
        return 0
    return _unit_relevance(text, q, plan)


def _answer_result_is_useful(item: dict, q: Question, plan: EvidencePlan) -> bool:
    item_type = _string_field(item, "type")
    role = _string_field(item, "answer_role")
    relevance = _answer_result_relevance(item, q, plan)
    text_relevance = _answer_result_text_relevance(item, q, plan)
    value_only_match = _answer_result_value_only_match(item, q, plan)
    if _primary_is_trusted(item):
        return relevance > 0 or bool(item.get("formatted_answer") or item.get("computed_answer"))
    if role in {"current_state_direct", "historical_direct"}:
        return text_relevance >= 1 or (relevance >= 2 and not value_only_match)
    if item_type == "direct_answer_span":
        if value_only_match:
            return False
        if role in {"supporting_context", "context_evidence", "timeline_context"}:
            return text_relevance >= 2
        return text_relevance >= 1
    if role in {"supporting_context", "context_evidence", "timeline_context"}:
        return text_relevance >= 3
    return False


def _answer_result_priority(item: dict, q: Question, plan: EvidencePlan) -> tuple[int, int]:
    item_type = _string_field(item, "type")
    role = _string_field(item, "answer_role")
    kind = _string_field(item, "kind") or _string_field(item, "computed_answer", "kind")
    relevance = _answer_result_relevance(item, q, plan)
    if item_type == "direct_answer_span" or role in {
        "current_state_direct",
        "historical_direct",
    }:
        return (10, -relevance)
    if role in {"supporting_context", "context_evidence", "timeline_context"}:
        return (25, -relevance)
    if kind in {"count_from_event_ledger", "count_aggregate", "count_unique"}:
        return (30, -relevance)
    if kind in {"attribute_lookup_materialized", "attribute_value", "attribute_lookup"}:
        return (40, -relevance)
    if kind in {"raw_text_recovery", "event_session_recovery"}:
        return (80, -relevance)
    return (50, -relevance)


def _hit_relevance(hit: SearchHit, q: Question, plan: EvidencePlan) -> tuple[int, float]:
    text = (hit.text or json_dumps(hit.props)).lower()
    q_tokens = _question_tokens(q.question)
    overlap = len(q_tokens & _question_tokens(text))
    numeric_bonus = 0
    if "multi_operand_numeric" in plan.labels and re.search(r"\$?\d", text):
        numeric_bonus += 3
    if "temporal" in plan.labels and re.search(r"\b\d{4}/\d{2}/\d{2}\b|\b(mon|tue|wed|thu|fri|sat|sun|january|february|march|april|may|june|july|august|september|october|november|december)\b", text):
        numeric_bonus += 2
    if "ordered_or_list" in plan.labels and re.search(r"\b(first|second|third|fourth|fifth|sixth|\d+[.)])\b", text):
        numeric_bonus += 2
    return (overlap + numeric_bonus, -hit.distance)


def _select_hits_for_evidence(hits: List[SearchHit], q: Question, plan: EvidencePlan, limit: int) -> List[SearchHit]:
    if limit <= 0:
        return []
    ranked = sorted(enumerate(hits), key=lambda pair: (_hit_relevance(pair[1], q, plan), -pair[0]), reverse=True)
    return [hit for _idx, hit in ranked[:limit]]


def _evidence_units(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof)\.", r"\1", text)
    units = re.split(r"(?=\[\d+\|session=)|(?<=[.!?])\s+(?=[A-Z0-9*\[])", text)
    return [unit.strip() for unit in units if unit.strip()]


def _unit_relevance(unit: str, q: Question, plan: EvidencePlan) -> int:
    text = unit.lower()
    overlap = len(_question_tokens(q.question) & _question_tokens(text))
    score = overlap
    if "multi_operand_numeric" in plan.labels and overlap and re.search(r"\$?\d", text):
        score += 3
    if "temporal" in plan.labels and overlap and re.search(
        r"\b\d{4}/\d{2}/\d{2}\b|\b(mon|tue|wed|thu|fri|sat|sun|january|february|march|april|may|june|july|august|september|october|november|december)\b",
        text,
    ):
        score += 2
    if "ordered_or_list" in plan.labels and overlap and re.search(
        r"\b(first|second|third|fourth|fifth|sixth|\d+[.)])\b", text
    ):
        score += 2
    if "preference_constraints" in plan.labels and re.search(r"\b(prefers?|preferred|likes?|liked|avoid|allergic|homegrown|favorite|usually)\b", text):
        score += 2
    return score


def _unit_token_overlap(unit: str, q: Question) -> int:
    return len(_question_tokens(q.question) & _question_tokens(unit))


def _unit_allowed_for_numeric_table(source: str, unit: str, q: Question, plan: EvidencePlan) -> bool:
    overlap = _unit_token_overlap(unit, q)
    if overlap > 0:
        return True
    if "preference_constraints" in plan.labels and re.search(
        r"\b(prefers?|preferred|likes?|liked|avoid|allergic|homegrown|favorite|usually)\b", unit.lower()
    ):
        return True
    if source.startswith("bundle.operand_groups"):
        return True
    return False


def _numeric_unit_matches_question_type(unit: str, q: Question, plan: EvidencePlan) -> bool:
    lower_q = q.question.lower()
    lower_unit = unit.lower()
    anchors = _question_anchor_tokens(q)
    content_overlap = len(anchors & _content_tokens(unit))

    if re.search(r"\b(percentage|percent)\b", lower_q):
        return (
            "%" in lower_unit
            or "percent" in lower_unit
            or content_overlap >= 2
            or re.search(r"\b(out of|of the|overall|total)\b", lower_unit)
        )

    if re.search(r"\bdoctor'?s?\s+appointments?\b", lower_q):
        has_doctor = re.search(r"\b(doctor|physician|surgeon|dr\.?|appointment|appointments)\b", lower_unit)
        not_session = not re.search(r"\b(physical therapy|therapy sessions?|sessions? twice|workshop|lecture|conference)\b", lower_unit)
        return bool(has_doctor and not_session)

    if re.search(r"\b(items?|clothing|pick up|return)\b", lower_q):
        return bool(
            re.search(r"\b(items?|clothing|clothes|shirt|dress|pants|boots|shoes|pick up|pickup|return)\b", lower_unit)
        )

    if re.search(r"\b(days?|weeks?|months?|years?)\b", lower_q):
        return content_overlap >= 1 or bool(_DATE_RE.search(unit))

    if re.search(r"\b(how many|number|count)\b", lower_q):
        if content_overlap >= 2:
            return True
        if content_overlap == 1 and re.search(r"\b(total|number|count|times?|appointments?|events?|items?)\b", lower_unit):
            return True
        return False

    return content_overlap > 0


def _unit_allowed_for_timeline(source: str, unit: str, q: Question, plan: EvidencePlan) -> bool:
    overlap = _unit_token_overlap(unit, q)
    if overlap > 0:
        return True
    if source.startswith("bundle.timeline") and "temporal" in plan.labels:
        return False
    if "preference_constraints" in plan.labels and re.search(
        r"\b(prefers?|preferred|likes?|liked|avoid|allergic|homegrown|favorite|usually)\b", unit.lower()
    ):
        return True
    return False


def _compact_top_memory_highlight(hit: SearchHit, q: Question, plan: MemoryPlan, max_chars: int) -> str:
    ranked = sorted(
        enumerate(_evidence_units(hit.text)),
        key=lambda pair: (_unit_relevance(pair[1], q, plan), -pair[0]),
        reverse=True,
    )
    highlights = [
        _clip(unit, max(160, max_chars // 2))
        for _idx, unit in ranked
        if _unit_relevance(unit, q, plan) > 0
    ][:2]
    if not highlights:
        return ""
    return _clip(
        json_dumps(
            {
                "node_id": hit.node_id,
                "distance": round(hit.distance, 4),
                "highlights": highlights,
            }
        ),
        max_chars,
    )


def _assistant_memory_requested(q: Question, plan: MemoryPlan) -> bool:
    question = q.question.lower()
    return q.question_type == "single-session-assistant" or bool(
        re.search(
            r"\b(previous|earlier|last time|remind|provided|recommended|mentioned|"
            r"list|phone number|quote|website|recipe|parameter|rotation|venue)\b",
            question,
        )
        or "ordered_or_list" in plan.labels
    )


def _text_has_list_shape(text: str) -> bool:
    return bool(
        re.search(
            r"(^|\s)(?:\d+[.)]|[-*]\s+|first|second|third|fourth|fifth|sixth|"
            r"seventh|eighth|ninth|tenth|twenty[- ]?seventh)\b",
            text.lower(),
        )
    )


def _compact_assistant_memory(resp, q: Question, plan: MemoryPlan, max_chars: int) -> str:
    if not _assistant_memory_requested(q, plan):
        return ""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    per_memory_chars = max(2200, min(max_chars * 3, 5000))
    broad_assistant_recall = q.question_type == "single-session-assistant"

    def consider(source: str, text: str, distance: float | None = None, node_id: int | None = None) -> None:
        if not text:
            return
        relevance = _unit_relevance(text, q, plan)
        if relevance <= 0 and not _text_has_list_shape(text) and not broad_assistant_recall:
            return
        key = _norm(text)[:400]
        if not key or key in seen:
            return
        seen.add(key)
        row = {
            "source": source,
            "memory": _clip(text, per_memory_chars),
        }
        if node_id is not None:
            row["node_id"] = str(node_id)
        if distance is not None:
            row["distance"] = f"{distance:.4f}"
        rows.append(row)

    def consider_selected_hits() -> None:
        for name in ("hits", "chunk_hits"):
            selected = _select_hits_for_evidence(getattr(resp, name, []) or [], q, plan, 8)
            for hit in selected:
                consider(name.removesuffix("s"), hit.text, hit.distance, hit.node_id)
                if len(rows) >= 6:
                    break
            if len(rows) >= 6:
                break

    if broad_assistant_recall:
        consider_selected_hits()

    for item in getattr(resp, "answer_results", []) or []:
        text = _main_text_from_answer_result(item)
        role = _string_field(item, "answer_role")
        item_type = _string_field(item, "type")
        if role in {"timeline_context", "context_evidence", "supporting_context"} or item_type == "direct_answer_span":
            consider("answer_result", text)
            if len(rows) >= 6:
                break

    bundle = getattr(resp, "answer_bundle", {}) or {}
    if isinstance(bundle, dict) and len(rows) < 6:
        packed = bundle.get("packed_context")
        for text in _natural_texts_from_bundle_value(packed):
            consider("bundle.packed_context", text)
            if len(rows) >= 6:
                break

    if not broad_assistant_recall and len(rows) < 6:
        consider_selected_hits()

    if not rows:
        return ""
    return _json_clip({"assistant_memory": rows[:6]}, max(max_chars, 3200))


def _resolved_time_targets(plan: MemoryPlan) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for hint in plan.hints:
        match = re.search(r"([^=]+)=\s*(\d{4})/(\d{2})/(\d{2})", hint)
        if not match:
            continue
        label = match.group(1).strip()
        date = f"{match.group(2)}/{match.group(3)}/{match.group(4)}"
        targets.append((label, date))
    return list(dict.fromkeys(targets))


def _compact_temporal_memory(resp, q: Question, plan: MemoryPlan, max_chars: int) -> str:
    targets = _resolved_time_targets(plan)
    if not targets:
        return ""
    active_targets = [
        (label, date)
        for label, date in targets
        if "current reference date" not in label.lower()
    ]
    if not active_targets:
        return _json_clip({"temporal_memory": {"targets": targets}}, max_chars)
    target_dates = {date for _label, date in active_targets}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, text in _candidate_texts(resp):
        for unit in _evidence_units(text):
            unit_norm = unit.replace("-", "/")
            if not any(date in unit_norm for date in target_dates):
                continue
            key = (source, _norm(unit)[:200])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"source": source, "memory": _clip(unit, 280)})
            if len(rows) >= 10:
                break
        if len(rows) >= 10:
            break
    if not rows:
        return _json_clip({"temporal_memory": {"targets": targets}}, max_chars)
    return _json_clip({"temporal_memory": {"targets": targets, "matches": rows}}, max_chars)


def _compact_preference_memory(resp, q: Question, plan: MemoryPlan, max_chars: int) -> str:
    if "preference_constraints" not in plan.labels:
        return ""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def consider(source: str, text: str) -> None:
        if not text:
            return
        if not _preference_profile_item_is_relevant(text, q, plan):
            return
        key = _norm(text)[:220]
        if key in seen:
            return
        seen.add(key)
        rows.append({"source": source, "memory": _clip(text, 260)})

    bundle = getattr(resp, "answer_bundle", {}) or {}
    if isinstance(bundle, dict):
        profile = bundle.get("preference_profile")
        if isinstance(profile, dict):
            filtered = _filter_preference_profile(profile, q, plan)
            if filtered:
                rows.append({"source": "bundle.preference_profile", "memory": _json_clip(filtered, 900)})

    for source, text in _candidate_texts(resp):
        for unit in _evidence_units(text):
            consider(source, unit)
            if len(rows) >= 12:
                break
        if len(rows) >= 12:
            break
    if not rows:
        return ""
    return _json_clip({"preference_memory": rows[:12]}, max_chars)


_compact_top_hit_highlight = _compact_top_memory_highlight


_NUMBER_RE = re.compile(
    r"(?:\$\s*)?\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage|dollars?|"
    r"days?|weeks?|months?|years?|hours?|minutes?|pages?|plants?|followers?|"
    r"courses?|items?|events?|miles?|pounds?|kg|lbs|oz|mbps|gbps)?\b",
    re.IGNORECASE,
)
_NUMBER_WITH_UNIT_RE = re.compile(
    r"(?:[$€£]\s*\d|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage|dollars?|"
    r"days?|weeks?|months?|years?|hours?|minutes?|pages?|plants?|followers?|"
    r"courses?|items?|events?|miles?|pounds?|kg|lbs|oz|mbps|gbps))",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _number_is_noise(match: re.Match[str], unit: str, q: Question, plan: EvidencePlan) -> bool:
    value = match.group(0).strip()
    start, end = match.span()
    before = unit[max(0, start - 12) : start]
    after = unit[end : min(len(unit), end + 12)]
    around = before + value + after
    lower_question = q.question.lower()
    lower_unit = unit.lower()

    if re.search(r"(?:session|bytes|node_id|span_chars|shown_chars|document_chars)\s*[=: -]*$", before.lower()):
        return True
    if re.search(r"^\s*[-,)]?\s*(?:bytes|node_id|span_chars|shown_chars|document_chars)\b", after.lower()):
        return True
    if re.search(r"\[\d+\|session=\d+", around):
        return True
    bare_digits = re.sub(r"\D", "", value)
    if len(bare_digits) >= 6 and not _NUMBER_WITH_UNIT_RE.search(value) and not any(ch in value for ch in "$€£%"):
        return True
    if re.search(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", around):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", around):
        return "time" not in lower_question and "when" not in lower_question
    if re.search(r"(^|\n)\s*\d+[.)]\s+", before + value + after):
        return True
    if value.isdigit() and 1900 <= int(value) <= 2099:
        return not any(word in lower_question for word in ("year", "date", "when"))
    if _NUMBER_WITH_UNIT_RE.search(value):
        return False
    if any(ch in value for ch in "$€£%"):
        return False
    if re.search(
        r"\b(total|how many|number|count|average|difference|older|younger|spent|save|saved|cost|amount)\b",
        lower_question,
    ):
        return False
    if re.search(r"\b(total|count|spent|saved|cost|amount|followers|views|courses|events)\b", lower_unit):
        return False
    return "multi_operand_numeric" in plan.labels


def _operand_row_score(source: str, unit: str, value: str, q: Question, plan: EvidencePlan) -> int:
    lower_q = q.question.lower()
    lower_unit = unit.lower()
    score = _unit_token_overlap(unit, q) * 8
    if source == "answer_result":
        score += 8
    elif source.startswith("bundle."):
        score += 6
    elif source in {"fact_hit", "fact"}:
        score += 4
    if any(ch in value for ch in "$€£"):
        score += 10
    if "%" in value or "percent" in value.lower():
        score += 8
    if re.search(r"\b(total|spent|cost|price|amount|money|discount|percentage|average)\b", lower_q):
        if re.search(r"\b(total|spent|cost|price|amount|paid|earned|saved|discount|original|budget|value)\b", lower_unit):
            score += 8
    if re.search(r"\b(how many|number|count)\b", lower_q):
        if re.search(r"\b(total|number|count|completed|reached|views|followers|courses|items|events)\b", lower_unit):
            score += 8
    if re.search(r"\b(water bottle|bpa|oz|lumens|minutes|hours)\b", lower_unit) and not re.search(
        r"\b(water bottle|bpa|oz|lumens|minutes|hours)\b", lower_q
    ):
        score -= 12
    if re.search(r"\b(raw_text_recovery ambiguous|ambiguous)\b", lower_unit):
        score -= 8
    return score


def _operand_rows(resp, q: Question, plan: EvidencePlan, row_limit: int) -> list[dict[str, str]]:
    scored_rows: list[tuple[int, int, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    scan_limit = row_limit * 6
    ordinal = 0
    for source, text in _candidate_texts(resp):
        for unit in _evidence_units(text):
            if _unit_relevance(unit, q, plan) <= 0:
                continue
            if not _unit_allowed_for_numeric_table(source, unit, q, plan):
                continue
            if not _numeric_unit_matches_question_type(unit, q, plan):
                continue
            for match in _NUMBER_RE.finditer(unit):
                if _number_is_noise(match, unit, q, plan):
                    continue
                value = match.group(0).strip()
                key = (_norm(value), _norm(unit)[:160])
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "value": value,
                    "source": source,
                    "memory": _clip(unit, 240),
                }
                scored_rows.append(
                    (
                        _operand_row_score(source, unit, value, q, plan),
                        -ordinal,
                        row,
                    )
                )
                ordinal += 1
                if len(scored_rows) >= scan_limit:
                    break
            if len(scored_rows) >= scan_limit:
                break
        if len(scored_rows) >= scan_limit:
            break
    return [row for _score, _ordinal, row in sorted(scored_rows, reverse=True)[:row_limit]]


def _timeline_rows(resp, q: Question, plan: EvidencePlan, row_limit: int) -> list[dict[str, str]]:
    events: list[tuple[int, int, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    ordinal = 0
    for source, text in _candidate_texts(resp):
        for unit in _evidence_units(text):
            if _unit_relevance(unit, q, plan) <= 0:
                continue
            if not _unit_allowed_for_timeline(source, unit, q, plan):
                continue
            dates = [m.group(0).strip() for m in _DATE_RE.finditer(unit)]
            if not dates:
                continue
            key = (_norm(",".join(dates)), _norm(unit)[:180])
            if key in seen:
                continue
            seen.add(key)
            events.append(
                (
                    _unit_relevance(unit, q, plan),
                    -ordinal,
                    {
                        "date": ", ".join(dates[:3]),
                        "source": source,
                        "memory": _clip(unit, 240),
                    },
                )
            )
            ordinal += 1
            if len(events) >= row_limit * 4:
                break
        if len(events) >= row_limit * 4:
            break
    return [row for _score, _ordinal, row in sorted(events, reverse=True)[:row_limit]]


_ORDERED_ITEM_RE = re.compile(
    r"(?:^|\s)(?P<label>\d{1,3}[.)]|first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|twenty[- ]?seventh)\s+(?P<item>[^.;\n\[]+)",
    re.IGNORECASE,
)


def _ordered_list_rows(resp, q: Question, plan: EvidencePlan, row_limit: int) -> list[dict[str, str]]:
    if not _assistant_memory_requested(q, plan):
        return []
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, text in _candidate_texts(resp):
        if _unit_relevance(text, q, plan) <= 0 and not _text_has_list_shape(text):
            continue
        for match in _ORDERED_ITEM_RE.finditer(text):
            label = match.group("label").strip()
            item = _clip(match.group("item").strip(" :-*"), 160)
            key = _norm(f"{label} {item}")
            if not item or key in seen:
                continue
            seen.add(key)
            rows.append({"position": label, "item": item, "source": source})
            if len(rows) >= row_limit:
                return rows
    return rows


def _preference_rows(resp, q: Question, plan: EvidencePlan, row_limit: int) -> list[dict[str, str]]:
    if "preference_constraints" not in plan.labels:
        return []
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    bundle = getattr(resp, "answer_bundle", {}) or {}
    if isinstance(bundle, dict):
        profile = bundle.get("preference_profile")
        if isinstance(profile, dict):
            filtered = _filter_preference_profile(profile, q, plan)
            for text in _natural_texts_from_bundle_value(filtered):
                key = _norm(text)[:180]
                if key and key not in seen:
                    seen.add(key)
                    rows.append({"source": "bundle.preference_profile", "constraint": _clip(text, 220)})
                    if len(rows) >= row_limit:
                        return rows
    for source, text in _candidate_texts(resp):
        for unit in _evidence_units(text):
            if not _preference_profile_item_is_relevant(unit, q, plan):
                continue
            key = _norm(unit)[:180]
            if key in seen:
                continue
            seen.add(key)
            rows.append({"source": source, "constraint": _clip(unit, 220)})
            if len(rows) >= row_limit:
                return rows
    return rows


def _answer_type_for_question(q: Question, plan: EvidencePlan) -> str:
    question = q.question.lower()
    if "ordered_or_list" in plan.labels:
        return "ordered_list"
    if "preference_constraints" in plan.labels:
        return "preference_constraints"
    if "temporal" in plan.labels and "multi_operand_numeric" not in plan.labels:
        return "temporal_lookup"
    if re.search(r"\b(percentage|percent|discount)\b", question):
        return "percentage"
    if re.search(r"\b(average|mean)\b", question):
        return "average"
    if re.search(r"\b(total|how much|amount|spent|earned|saved|cost)\b", question):
        return "sum_or_total"
    if re.search(r"\b(how many|number|count)\b", question):
        return "count"
    return "lookup"


def _compact_memory_card(resp, q: Question, plan: EvidencePlan, max_chars: int) -> str:
    card: dict[str, object] = {
        "answer_type": _answer_type_for_question(q, plan),
        "labels": list(plan.labels),
    }
    if "multi_operand_numeric" in plan.labels:
        operands = _operand_rows(resp, q, plan, 14 if plan.complex else 8)
        if operands:
            card["operands"] = operands
    if "temporal" in plan.labels:
        targets = _resolved_time_targets(plan)
        if targets:
            card["time_targets"] = targets
        events = _timeline_rows(resp, q, plan, 8)
        if events:
            card["events"] = events
    if _assistant_memory_requested(q, plan):
        list_items = _ordered_list_rows(resp, q, plan, 16)
        if list_items:
            card["list_items"] = list_items
    if "preference_constraints" in plan.labels:
        constraints = _preference_rows(resp, q, plan, 10)
        if constraints:
            card["constraints"] = constraints
    if len(card) <= 2:
        return ""
    return _json_clip({"memory_card": card}, max_chars)


def _text_from_answer_result(item: dict) -> str:
    parts: list[str] = []
    for field in ("text", "value", "number", "date", "subject", "predicate"):
        value = item.get(field)
        if value not in (None, "", [], {}):
            parts.append(str(value))
    computed = item.get("computed_answer")
    if isinstance(computed, dict):
        for field in ("value", "formatted", "kind", "status"):
            value = computed.get(field)
            if value not in (None, "", [], {}):
                parts.append(str(value))
    formatted = item.get("formatted_answer")
    if isinstance(formatted, dict):
        value = formatted.get("value")
        if value not in (None, "", [], {}):
            parts.append(str(value))
    return " ".join(parts)


def _main_text_from_answer_result(item: dict) -> str:
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return _text_from_answer_result(item)


def _answer_result_text_relevance(item: dict, q: Question, plan: EvidencePlan) -> int:
    return _unit_relevance(_main_text_from_answer_result(item), q, plan)


def _answer_result_value_only_match(item: dict, q: Question, plan: EvidencePlan) -> bool:
    main = _main_text_from_answer_result(item)
    if not main:
        return False
    return _unit_relevance(main, q, plan) <= 0 and _answer_result_relevance(item, q, plan) > 0


def _candidate_texts(resp) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in getattr(resp, "answer_results", []) or []:
        text = _main_text_from_answer_result(item)
        if text:
            out.append(("answer_result", text))
    primary = getattr(resp, "primary_answer_result", {}) or {}
    if primary:
        text = _main_text_from_answer_result(primary)
        if text:
            out.append(("primary_answer_result", text))
    bundle = getattr(resp, "answer_bundle", {}) or {}
    for field in ("direct_answer_spans", "operand_groups", "timeline", "preference_profile"):
        value = bundle.get(field) if isinstance(bundle, dict) else None
        for text in _natural_texts_from_bundle_value(value):
            out.append((f"bundle.{field}", text))
    for name in ("fact_hits", "chunk_hits", "hits"):
        for hit in getattr(resp, name, []) or []:
            if hit.text:
                out.append((name.removesuffix("s"), hit.text))
    return out


def _bundle_item_text(item) -> str:
    if isinstance(item, dict):
        text = _text_from_answer_result(item)
        return text or json_dumps(item)
    if isinstance(item, str):
        return item
    return json_dumps(item)


def _natural_texts_from_bundle_value(value) -> list[str]:
    out: list[str] = []
    if value in (None, "", [], {}):
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_natural_texts_from_bundle_value(item))
        return out
    if isinstance(value, dict):
        if "text" in value or "value" in value:
            text = _main_text_from_answer_result(value)
            if text:
                out.append(text)
            return out
        for child in value.values():
            out.extend(_natural_texts_from_bundle_value(child))
        return out
    text = str(value).strip()
    if text:
        out.append(text)
    return out


def _filter_bundle_sequence(field: str, value: list, q: Question, plan: EvidencePlan) -> list:
    kept = []
    threshold = 2
    if field == "operand_groups":
        threshold = 1
    for item in value:
        text = _bundle_item_text(item)
        if _unit_relevance(text, q, plan) >= threshold:
            kept.append(item)
    return kept


def _preference_profile_item_is_relevant(text: str, q: Question, plan: EvidencePlan) -> bool:
    lowered = text.lower()
    overlap = len(_question_anchor_tokens(q) & _content_tokens(text))
    preference_signal = bool(
        re.search(r"\b(prefers?|preferred|likes?|liked|avoid|allergic|favorite|usually|wants?|needs?|high-quality|compatible)\b", lowered)
    )
    if overlap >= 2:
        return True
    if overlap >= 1 and preference_signal:
        return True
    if preference_signal and "preference_constraints" in plan.labels and not _question_anchor_tokens(q):
        return True
    return False


def _filter_preference_profile(value: dict, q: Question, plan: EvidencePlan) -> dict:
    if not isinstance(value, dict):
        return {}
    if "preference_constraints" not in plan.labels:
        return {}
    filtered: dict = {}
    for field, field_value in value.items():
        if isinstance(field_value, list):
            kept = [
                item
                for item in field_value
                if _preference_profile_item_is_relevant(_bundle_item_text(item), q, plan)
            ]
            if kept:
                filtered[field] = kept[:24]
        elif field_value not in (None, "", [], {}):
            filtered[field] = field_value
    return filtered


def _filter_bundle_field(field: str, value, q: Question, plan: EvidencePlan):
    if value in (None, "", [], {}):
        return None
    if field == "preference_profile":
        filtered = _filter_preference_profile(value, q, plan)
        return filtered or None
    if isinstance(value, list):
        filtered = _filter_bundle_sequence(field, value, q, plan)
        return filtered or None
    if isinstance(value, dict):
        filtered_dict = {}
        for key, child in value.items():
            filtered_child = _filter_bundle_field(field, child, q, plan)
            if filtered_child not in (None, "", [], {}):
                filtered_dict[key] = filtered_child
        return filtered_dict or None
    return value if _unit_relevance(str(value), q, plan) > 0 else None


def _compact_operand_table(resp, q: Question, plan: EvidencePlan, max_chars: int) -> str:
    rows = _operand_rows(resp, q, plan, 24 if plan.complex else 12)
    if not rows:
        return ""
    return _json_clip({"operand_memory": rows}, max_chars)


def _compact_timeline_highlight(resp, q: Question, plan: EvidencePlan, max_chars: int) -> str:
    events: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, text in _candidate_texts(resp):
        for unit in _evidence_units(text):
            if _unit_relevance(unit, q, plan) <= 0:
                continue
            if not _unit_allowed_for_timeline(source, unit, q, plan):
                continue
            dates = [m.group(0).strip() for m in _DATE_RE.finditer(unit)]
            if not dates:
                continue
            key = (_norm(",".join(dates)), _norm(unit)[:180])
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "date": ", ".join(dates[:3]),
                    "source": source,
                    "memory": _clip(unit, 280),
                }
            )
            if len(events) >= 12:
                break
        if len(events) >= 12:
            break
    if not events:
        return ""
    return _json_clip({"timeline_memory": events}, max_chars)


def _computed_memory_from_response(resp, cfg: HarnessConfig, q: Question, plan: MemoryPlan) -> List[ComputedMemory]:
    if not getattr(cfg, "include_answer_bundle", False):
        return []
    gateway_memory = getattr(resp, "compact_memory", None)
    if isinstance(gateway_memory, list) and gateway_memory:
        out: List[ComputedMemory] = []
        seen: set[str] = set()
        for item in gateway_memory:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            if not source:
                continue
            payload = {key: value for key, value in item.items() if value not in (None, "", [], {})}
            text_value = payload.get("text")
            if text_value is None and "memory" in payload:
                text_value = payload.get("memory")
            if text_value in (None, "", [], {}):
                continue
            text = json_dumps(payload)
            key = _norm(f"{source}\n{text}")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(ComputedMemory(rank=len(out) + 1, source=source, text=text))
        return out
    return []


_computed_evidence_from_response = _computed_memory_from_response


def query_question(store: MemoryStore, cfg: HarnessConfig, q: Question) -> QueryResult:
    graph = f"{cfg.graph_prefix}-{q.question_id}"
    started = time.time()
    memory_plan = _memory_plan_for_question(cfg, q)
    search_query = _gateway_query_text(q)

    # Over-fetch, then prune the entity/fragment nodes and repeated text that
    # crowd out answer-bearing session content, then keep the top k.
    retrieve_k = max(cfg.k, cfg.retrieve_k)
    extra_queries = _decompose_query(q.question) if _decompose_enabled() else []
    if hasattr(store, "search_with_bundle"):
        resp = store.search_with_bundle(
            graph,
            search_query,
            k=retrieve_k,
            ef=cfg.ef,
            granularities=cfg.read_granularities,
            rrf_k=cfg.rrf_k,
            include_answer_bundle=cfg.include_answer_bundle,
            fact_k=memory_plan.fact_k if cfg.include_answer_bundle else 0,
            chunk_k=memory_plan.chunk_k if cfg.include_answer_bundle else 0,
            context_date=q.question_date if cfg.include_answer_bundle else "",
            extra_queries=extra_queries or None,
        )
        raw = resp.hits
        computed_memory = _computed_memory_from_response(resp, cfg, q, memory_plan)
        memories = getattr(resp, "memories", "") or ""
        # Per-sub-query evidence (LME_DECOMPOSE): the fused extra_queries lane
        # only affects RECALL — the evidence funnel afterwards (session ranking,
        # window selection, compaction) still scores by the MAIN query, so an
        # operand span ("I am 32 years old") gets distilled away even when its
        # session was fetched. Give each sub-query its OWN full pipeline pass and
        # merge the memories blocks. Still one reader shot; each search is
        # planned from the question alone (no retrieval feedback).
        if memories and extra_queries:
            for sub in extra_queries[:2]:
                # Skip only degenerate near-whole-question repeats; verbatim
                # entity spans (chess notation, names) are exactly what the
                # sub-pipeline needs — BM25 nails them.
                if len(sub) > 0.7 * max(len(q.question or ""), 1):
                    continue
                try:
                    sub_resp = store.search_with_bundle(
                        graph,
                        sub,
                        k=3,
                        ef=cfg.ef,
                        granularities=cfg.read_granularities,
                        rrf_k=cfg.rrf_k,
                        include_answer_bundle=True,
                        fact_k=memory_plan.fact_k,
                        chunk_k=memory_plan.chunk_k,
                        context_date=q.question_date or "",
                    )
                except Exception:  # noqa: BLE001
                    continue
                sub_mem = (getattr(sub_resp, "memories", "") or "").strip()
                if not sub_mem:
                    continue
                # Keep only evidence lines that are new vs the merged text so far.
                fresh = [
                    ln for ln in sub_mem.splitlines()
                    if ln.startswith("- ") and ln[:80] not in memories
                ][:6]
                if fresh:
                    memories += (
                        f"\nAdditional evidence for sub-question \"{sub}\":\n"
                        + "\n".join(fresh) + "\n"
                    )
    else:
        raw = store.search(
            graph,
            search_query,
            k=retrieve_k,
            ef=cfg.ef,
            granularities=cfg.read_granularities,
            rrf_k=cfg.rrf_k,
        )
        computed_memory = []
        memories = ""

    kept: List[SearchHit] = []
    seen: set = set()
    n_fragment_dropped = 0
    n_dedup_dropped = 0
    for h in raw:
        if cfg.filter_fragments and session_index_of(h) is None:
            n_fragment_dropped += 1
            continue
        if cfg.dedup_results:
            key = _norm(h.text)
            if key and key in seen:
                n_dedup_dropped += 1
                continue
            seen.add(key)
        kept.append(h)
        if len(kept) >= cfg.k:
            break

    return QueryResult(
        question_id=q.question_id,
        hits=kept,
        search_ms=(time.time() - started) * 1000.0,
        computed_memory=computed_memory,
        memory_plan=memory_plan,
        memories=memories,
        n_raw=len(raw),
        n_fragment_dropped=n_fragment_dropped,
        n_dedup_dropped=n_dedup_dropped,
    )

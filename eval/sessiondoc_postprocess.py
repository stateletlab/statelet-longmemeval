"""Post-process Step 1 validation for the chunk-index + session-return idea.

Takes the chunk-level detail jsonl from a v2-final-100 run and re-scores each
question two ways:

  chunk@N    — keep the existing chunk-mode but truncate to top-N (apples-to-apples).
  session@N  — aggregate top-K chunks → top-N distinct sessions by best chunk
               score, replace each "result" with FULL session text, then run
               the same evaluator. This is what server-side session-return mode
               would produce.

Reuses the production evaluator (`evaluate_retrieval`) so the metrics are
directly comparable with the baseline run output.

Usage:
    python eval/sessiondoc_postprocess.py \
        --detail eval/retrieval_eval_detail_v2-final-100.jsonl \
        --N 5

By default runs N ∈ {5, 10} with both text modes (all turns / user-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

# Stub the gRPC client before importing the eval module — we don't need it for
# pure metric recomputation, and the user's environment has a broken protobuf
# pyext install that crashes at `import statelet.client`.
import types

_statelet_stub = types.ModuleType("statelet")
_client_stub = types.ModuleType("statelet.client")


class _StubClient:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("StateletClient stubbed in post-process; not callable")


_client_stub.StateletClient = _StubClient
_statelet_stub.client = _client_stub
sys.modules.setdefault("statelet", _statelet_stub)
sys.modules.setdefault("statelet.client", _client_stub)

import longmemeval_retrieval_eval as lmev  # noqa: E402

QUESTIONS_DIR = lmev.QUESTIONS_DIR


def load_question(question_id: str) -> lmev.Question:
    path = QUESTIONS_DIR / f"{question_id}.json"
    if not path.exists():
        for p in QUESTIONS_DIR.glob("*.json"):
            with open(p) as f:
                d = json.load(f)
            if d.get("question_id") == question_id:
                path = p
                break
    with open(path) as f:
        d = json.load(f)
    return lmev.Question(
        question_id=d["question_id"],
        question=d["question"],
        answer=str(d["answer"]),
        question_type=d["question_type"],
        question_date=d.get("question_date", ""),
        sessions=d["haystack_sessions"],
        session_dates=d.get("haystack_dates", []),
        answer_session_ids=d.get("answer_session_ids", []),
        haystack_session_ids=d.get("haystack_session_ids", []),
    )


def build_session_text(session_turns: list, mode: str = "all") -> str:
    """mode='all' → every turn; mode='user' → only user turns (MemPalace style)."""
    if not isinstance(session_turns, list):
        return str(session_turns)
    parts: List[str] = []
    for turn in session_turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        if mode == "user" and role != "user":
            continue
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content}" if role else content)
    return "\n".join(parts)


def aggregate_chunks_to_sessions(
    hits: List[dict],
    *,
    pool_k: int,
    top_n: int,
    method: str = "best",  # "best" | "rrf" | "best+density"
) -> List[dict]:
    """Take chunk-level hits (already ordered best→worst by distance), keep
    only the first `pool_k` to define the candidate pool, then collapse to
    distinct sessions and rank.

    method:
      best         — best (lowest) distance chunk per session.
      rrf          — sum of 1/(60+rank) per session, ranked by sum desc.
      best+density — best distance, but boost sessions that appear ≥2 times
                     in the pool (multi-chunk hit = stronger signal).
    """
    pool = hits[:pool_k]
    best_by_session: Dict[int, dict] = {}
    counts: Dict[int, int] = {}
    rrf_sum: Dict[int, float] = {}
    rank_min: Dict[int, int] = {}
    for rank, h in enumerate(pool):
        sid = h.get("session_index", -1)
        if sid < 0:
            continue
        counts[sid] = counts.get(sid, 0) + 1
        rrf_sum[sid] = rrf_sum.get(sid, 0.0) + 1.0 / (60.0 + rank)
        if sid not in rank_min or rank < rank_min[sid]:
            rank_min[sid] = rank
        cur = best_by_session.get(sid)
        if cur is None or h["distance"] < cur["distance"]:
            best_by_session[sid] = h

    if method == "best":
        ranked = sorted(best_by_session.values(), key=lambda h: h["distance"])
    elif method == "rrf":
        ranked_sids = sorted(rrf_sum.keys(), key=lambda s: -rrf_sum[s])
        ranked = [best_by_session[s] for s in ranked_sids]
    elif method == "best+density":
        # score = distance - 0.02 * (count - 1)  (lower better; count=1 unchanged)
        def key(h):
            sid = h["session_index"]
            return h["distance"] - 0.02 * (counts.get(sid, 1) - 1)
        ranked = sorted(best_by_session.values(), key=key)
    else:
        raise ValueError(f"unknown method: {method}")
    return ranked[:top_n]


def build_session_result(
    session_index: int,
    distance: float,
    full_text: str,
) -> dict:
    """Synthetic result that the evaluator can consume. node_id is derived from
    session_index using the convention: session_node_id = (idx + 1) * 1000."""
    node_id = (session_index + 1) * 1000
    return {
        "node_id": node_id,
        "distance": distance,
        "text": full_text,
        "props": {
            "session_index": session_index,
            "raw_chunk": True,
            "category": "raw_chunk",
        },
    }


def build_chunk_result(hit: dict, full_chunk_text: Optional[str] = None) -> dict:
    """Convert a saved hit back into the dict shape `evaluate_retrieval`
    expects. The detail jsonl truncates `text` to 300 chars; that's what the
    evaluator already saw, so reuse it for the chunk-mode baseline."""
    return {
        "node_id": hit["node_id"],
        "distance": hit["distance"],
        "text": hit.get("text", ""),
        "props": {"session_index": hit.get("session_index", -1)},
    }


def run_mode(
    detail_lines: List[dict],
    *,
    mode: str,         # "chunk" | "session"
    n: int,            # top-N
    pool_k: int,       # candidate pool size for session aggregation
    text_mode: str,    # "all" | "user", only used for session mode
    agg_method: str = "best",
) -> dict:
    agg = {
        "n_questions": 0,
        "hit": 0,
        "span_hit": 0,
        "ans_hit": 0,
        "der_hit": 0,
        "recall_sum": 0.0,
        "span_recall_sum": 0.0,
        "mrr_sum": 0.0,
        "slot_cov_sum": 0.0,
        "derivable_sum": 0.0,
        "best_f1_sum": 0.0,
        "by_type": defaultdict(lambda: {"n": 0, "hit": 0, "span_hit": 0, "ans_hit": 0,
                                         "slot_cov": 0.0, "derivable": 0.0, "span_rec": 0.0}),
    }
    for rec in detail_lines:
        qid = rec["question_id"]
        try:
            q = load_question(qid)
        except FileNotFoundError:
            continue
        hits = rec.get("hits", [])
        if not hits:
            continue

        if mode == "chunk":
            results = [build_chunk_result(h) for h in hits[:n]]
        else:
            top_sessions = aggregate_chunks_to_sessions(hits, pool_k=pool_k, top_n=n, method=agg_method)
            results = []
            for h in top_sessions:
                sid = h["session_index"]
                if sid < 0 or sid >= len(q.sessions):
                    continue
                full_text = build_session_text(q.sessions[sid], mode=text_mode)
                results.append(build_session_result(sid, h["distance"], full_text))

        if not results:
            continue
        ev = lmev.evaluate_retrieval(q, results, k=n)

        agg["n_questions"] += 1
        agg["hit"] += int(ev.hit_at_k)
        agg["span_hit"] += int(ev.answer_span_hit_at_k)
        agg["ans_hit"] += int(ev.answer_result_hit_at_k)
        agg["der_hit"] += int(ev.derived_answer_hit_at_k)
        agg["recall_sum"] += ev.recall_at_k
        agg["span_recall_sum"] += ev.answer_span_recall_at_k
        agg["mrr_sum"] += ev.mrr
        agg["slot_cov_sum"] += ev.required_slot_coverage_at_k
        agg["derivable_sum"] += float(ev.derivable_at_k)
        agg["best_f1_sum"] += ev.best_token_f1
        bt = agg["by_type"][q.question_type]
        bt["n"] += 1
        bt["hit"] += int(ev.hit_at_k)
        bt["span_hit"] += int(ev.answer_span_hit_at_k)
        bt["ans_hit"] += int(ev.answer_result_hit_at_k)
        bt["slot_cov"] += ev.required_slot_coverage_at_k
        bt["derivable"] += float(ev.derivable_at_k)
        bt["span_rec"] += ev.answer_span_recall_at_k

    return agg


def fmt_summary(label: str, agg: dict) -> str:
    n = max(agg["n_questions"], 1)
    return (
        f"{label:32s} n={agg['n_questions']:3d} | "
        f"Hit={agg['hit']/n:.3f}  Span={agg['span_hit']/n:.3f}  "
        f"Ans={agg['ans_hit']/n:.3f}  Der={agg['der_hit']/n:.3f} | "
        f"Recall={agg['recall_sum']/n:.3f}  SpanRec={agg['span_recall_sum']/n:.3f}  "
        f"MRR={agg['mrr_sum']/n:.3f} | "
        f"SlotCov={agg['slot_cov_sum']/n:.3f}  Deriv={agg['derivable_sum']/n:.3f}  "
        f"BestF1={agg['best_f1_sum']/n:.3f}"
    )


def fmt_by_type(agg: dict) -> str:
    lines = ["    Per-type:"]
    for t in sorted(agg["by_type"]):
        bt = agg["by_type"][t]
        n = max(bt["n"], 1)
        lines.append(
            f"      {t:32s} n={bt['n']:3d} "
            f"Hit={bt['hit']/n:.2f} Span={bt['span_hit']/n:.2f} "
            f"Ans={bt['ans_hit']/n:.2f} SlotCov={bt['slot_cov']/n:.2f} "
            f"Deriv={bt['derivable']/n:.2f} SpanRec={bt['span_rec']/n:.2f}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detail", required=True)
    p.add_argument("--pool-k", type=int, default=20,
                   help="how many chunks to consider when aggregating to sessions")
    p.add_argument("--ns", default="5,10",
                   help="comma list of N values to evaluate")
    p.add_argument("--text-modes", default="all,user",
                   help="comma list: all | user")
    p.add_argument("--agg-methods", default="best",
                   help="comma list: best | rrf | best+density")
    args = p.parse_args()

    detail = []
    with open(args.detail) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            detail.append(json.loads(line))
    print(f"loaded {len(detail)} detail records from {args.detail}", flush=True)

    ns = [int(x) for x in args.ns.split(",")]
    text_modes = [x.strip() for x in args.text_modes.split(",")]
    agg_methods = [x.strip() for x in args.agg_methods.split(",")]

    print()
    print("=" * 120)
    print(f"sessiondoc post-process — pool-k={args.pool_k}")
    print("=" * 120)

    # Baseline at @20 (sanity — should match the published v2-final-100 numbers).
    base20 = run_mode(detail, mode="chunk", n=20, pool_k=args.pool_k, text_mode="all")
    print(fmt_summary("BASELINE chunk@20 (sanity)", base20))
    print()

    # Apples-to-apples: chunk@N vs session@N for each N.
    for n in ns:
        print(f"--- N={n} ---")
        chunk_n = run_mode(detail, mode="chunk", n=n, pool_k=args.pool_k, text_mode="all")
        print(fmt_summary(f"chunk@{n}", chunk_n))
        print(fmt_by_type(chunk_n))
        for tm in text_modes:
            for am in agg_methods:
                sess_n = run_mode(detail, mode="session", n=n, pool_k=args.pool_k,
                                  text_mode=tm, agg_method=am)
                print(fmt_summary(f"session@{n}  text={tm}  agg={am}", sess_n))
                print(fmt_by_type(sess_n))
        print()


if __name__ == "__main__":
    main()

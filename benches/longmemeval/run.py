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
"""ONE command — end-to-end LongMemEval Phase 0 harness.

Stages, in order:
  1. Inject all questions — segment + ingest the multi-session history (LLM-free).
  2. Query all questions  — top-k retrieval (LLM-free).
  3. hit@k  — score retrieval vs gold sessions (LLM-free GATE).
  4. Answer  — DeepSeek reads retrieved context -> answer.        (needs API key)
  5. Judge   — DeepSeek compares answer vs gold -> correct/incorrect.
  6. Report  — per-category + overall hit@k / recall@k / accuracy / mean tokens,
               plus a row vs mem0's published numbers.

Run it:
    # LLM-free retrieval gate (no API key) — all 500 questions:
    python -m benches.longmemeval

    # full pipeline incl. accuracy:
    DEEPSEEK_API_KEY=sk-... python -m benches.longmemeval

    # smoke (skip ingest if already injected, 20 questions):
    python -m benches.longmemeval --limit 20

Reproducible: seeded, config-pinned (see config.py), one namespaced graph per
question. Retrieval metrics are computable WITHOUT DeepSeek.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .abstention import (
    ABSTAIN_ANSWER,
    Confidence,
    calibrate,
    is_abstention,
    split_dev_test,
)
from .config import HarnessConfig
from .dataset import Question, category_counts, load_questions
from .deepseek import DeepSeekClient, estimate_tokens
from .inject import inject_question
from .metrics import MetricsAccumulator, score_retrieval, token_f1
from .query import QueryResult, query_question
from .report import build_report, render_table, write_report
from .store import MemoryStore, session_index_of


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="benches.longmemeval",
        description="LongMemEval Phase 0 harness: inject -> query -> hit@k -> DeepSeek answer+judge -> accuracy.",
    )
    p.add_argument("--addr", default=None, help="gateway address (default: config / STATELET_ADDR)")
    p.add_argument("--dataset-dir", default=None, help="LongMemEval questions dir")
    p.add_argument("--limit", type=int, default=0, help="number of questions (0 = all 500)")
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="resume from the Nth question (1-based, in the deterministic sorted "
        "order shown as [i/N]); skips the first N-1. Default 0 = from the beginning.",
    )
    p.add_argument("--only", default="", help="comma-separated question ids")
    p.add_argument("--k", type=int, default=None, help="top-k for retrieval (default: 5)")
    p.add_argument(
        "--granularity",
        choices=["session", "turn", "unit"],
        default=None,
        help="ingest node granularity (default: session). 'unit' = legacy 8-64 token chunks.",
    )
    p.add_argument(
        "--retrieve-k",
        type=int,
        default=None,
        help="over-fetch this many before filtering/dedup down to k (default 50)",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="keep entity/fragment hits that don't map to a session (default: drop them)",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="keep repeated retrieved text (default: collapse duplicates)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="concurrent put_unit RPCs per question during ingest "
        "(default: config / STATELET_INGEST_WORKERS=8; 1 = sequential)",
    )
    p.add_argument(
        "--refresh-wait-s",
        type=float,
        default=None,
        help="wait up to this many seconds after all injection for NRT refresh visibility "
        "(default: config / STATELET_LME_REFRESH_WAIT_S; auto 600s when STATELET_TEXT_GRAPH_NRT=1)",
    )
    p.add_argument(
        "--per-question-wait-s",
        type=float,
        default=None,
        help="sleep this many seconds after each question's ingest before starting the next "
        "(default: config / STATELET_LME_PER_QUESTION_WAIT_S=0)",
    )
    p.add_argument("--graph-prefix", default=None, help="namespace prefix for per-question graphs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help="skip stage 1 (graphs already injected from a previous run)",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="drop each per-question graph before ingest so a rerun starts "
        "clean (default: reuse/accumulate). Deletes graph data — opt-in only.",
    )
    p.add_argument(
        "--inject-only",
        action="store_true",
        help="run only injection (honors --reset) and exit before query/metrics",
    )
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="force LLM-free mode (skip DeepSeek answer+judge even if key set)",
    )
    p.add_argument(
        "--abstention",
        action="store_true",
        help="Phase 6: calibrate a confidence threshold on a dev split and "
        "abstain (LLM-free) on low-confidence test questions",
    )
    p.add_argument(
        "--abstention-dev-fraction",
        type=float,
        default=0.5,
        help="fraction of questions used to calibrate the abstention threshold "
        "(rest are the held-out test split; default 0.5)",
    )
    p.add_argument(
        "--abstention-min-retention",
        type=float,
        default=0.95,
        help="guardrail: minimum answerable retention the calibrated threshold "
        "must keep on dev (default 0.95 — 'no material drop')",
    )
    p.add_argument("--out", default=None, help="report JSON path (default: benches/longmemeval/last_report.json)")
    p.add_argument(
        "--detail-log",
        default=None,
        help="file for per-question Q/A + top-k hits (default: "
        "benches/longmemeval/last_run_detail_YYYYMMDD_HHMMSS_ffffff.log). "
        "Use '-' to print to stdout instead.",
    )
    p.add_argument(
        "--detail-max-chars",
        type=int,
        default=None,
        help="max characters of each retrieved hit written to the detail log "
        "(default: config / 0, no truncation)",
    )
    p.add_argument(
        "--no-answer-bundle",
        action="store_true",
        help="do not request gateway answer_bundle artifacts for reader memory",
    )
    p.add_argument(
        "--answer-results-limit",
        type=int,
        default=None,
        help="max answer_result JSON objects to compact into each detail-log question",
    )
    p.add_argument(
        "--answer-complex-results-limit",
        type=int,
        default=None,
        help="max answer_result objects for operand/temporal/list/preference questions",
    )
    p.add_argument(
        "--answer-evidence-max-chars",
        type=int,
        default=None,
        help="max characters per compacted computed/direct memory item",
    )
    p.add_argument(
        "--answer-complex-evidence-max-chars",
        type=int,
        default=None,
        help="max characters per computed/direct memory item for complex questions",
    )
    p.add_argument(
        "--answer-evidence-total-max-chars",
        type=int,
        default=None,
        help="max total compacted computed/direct memory characters per question",
    )
    p.add_argument(
        "--answer-complex-evidence-total-max-chars",
        type=int,
        default=None,
        help="max total computed/direct memory characters for complex questions",
    )
    p.add_argument(
        "--answer-fact-k",
        type=int,
        default=None,
        help="number of gateway fact_results to compact as direct memory",
    )
    p.add_argument(
        "--answer-complex-fact-k",
        type=int,
        default=None,
        help="number of gateway fact_results for complex questions",
    )
    p.add_argument(
        "--answer-chunk-k",
        type=int,
        default=None,
        help="number of gateway chunk_results to compact as raw-span memory",
    )
    p.add_argument(
        "--answer-complex-chunk-k",
        type=int,
        default=None,
        help="number of gateway chunk_results for complex questions",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> HarnessConfig:
    cfg = HarnessConfig()
    if args.addr:
        cfg.gateway_addr = args.addr
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.limit:
        cfg.limit = args.limit
    if args.start:
        cfg.start = max(0, args.start)
    if args.only:
        cfg.only_ids = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.k is not None:
        cfg.k = max(args.k, 1)
    if args.workers is not None:
        cfg.ingest_workers = max(1, args.workers)
    if args.refresh_wait_s is not None:
        cfg.refresh_wait_s = max(0.0, args.refresh_wait_s)
    if args.per_question_wait_s is not None:
        cfg.per_question_wait_s = max(0.0, args.per_question_wait_s)
    if args.granularity is not None:
        cfg.ingest_granularity = args.granularity
    if args.retrieve_k is not None:
        cfg.retrieve_k = args.retrieve_k
    if args.no_filter:
        cfg.filter_fragments = False
    if args.no_dedup:
        cfg.dedup_results = False
    if args.detail_log is not None:
        cfg.detail_log = args.detail_log
    if args.detail_max_chars is not None:
        cfg.detail_max_chars = max(0, args.detail_max_chars)
    if args.no_answer_bundle:
        cfg.include_answer_bundle = False
    if args.answer_results_limit is not None:
        cfg.answer_results_limit = max(0, args.answer_results_limit)
    if args.answer_complex_results_limit is not None:
        cfg.answer_complex_results_limit = max(0, args.answer_complex_results_limit)
    if args.answer_evidence_max_chars is not None:
        cfg.answer_evidence_max_chars = max(120, args.answer_evidence_max_chars)
    if args.answer_complex_evidence_max_chars is not None:
        cfg.answer_complex_evidence_max_chars = max(120, args.answer_complex_evidence_max_chars)
    if args.answer_evidence_total_max_chars is not None:
        cfg.answer_evidence_total_max_chars = max(0, args.answer_evidence_total_max_chars)
    if args.answer_complex_evidence_total_max_chars is not None:
        cfg.answer_complex_evidence_total_max_chars = max(0, args.answer_complex_evidence_total_max_chars)
    if args.answer_fact_k is not None:
        cfg.answer_fact_k = max(0, args.answer_fact_k)
    if args.answer_complex_fact_k is not None:
        cfg.answer_complex_fact_k = max(0, args.answer_complex_fact_k)
    if args.answer_chunk_k is not None:
        cfg.answer_chunk_k = max(0, args.answer_chunk_k)
    if args.answer_complex_chunk_k is not None:
        cfg.answer_complex_chunk_k = max(0, args.answer_complex_chunk_k)
    if args.reset:
        cfg.reset = True
    if args.graph_prefix:
        cfg.graph_prefix = args.graph_prefix
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def _login(cfg: HarnessConfig) -> Optional[str]:
    """Obtain a JWT for the gateway.

    The cluster scripts start the gateway with `GATEWAY_JWT_SECRET` set, so
    auth is on and every TextGraphPut/Search needs a bearer token (only `ping`
    is exempt — which is why an unauthenticated run looks connected until the
    first real RPC fails with UNAUTHENTICATED). `STATELET_TOKEN` short-circuits
    the login; otherwise we log in as username/password (default admin/admin)
    against the management API. On a `--no-auth` gateway login fails harmlessly
    and we return None — the data RPCs run unauthenticated there.
    """
    import os

    env_token = os.environ.get("STATELET_TOKEN")
    if env_token:
        print("Auth: using STATELET_TOKEN")
        return env_token

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk" / "python"))
    from statelet.client import StateletClient  # noqa: E402

    mgmt_addr = cfg.resolved_mgmt_addr()
    try:
        token = StateletClient.login(mgmt_addr, cfg.username, cfg.password)
        print(f"Auth: logged in as {cfg.username} via {mgmt_addr}")
        return token
    except Exception as exc:  # noqa: BLE001 — no-auth gateway or mgmt down
        print(f"Auth: login at {mgmt_addr} failed ({exc}); proceeding unauthenticated")
        return None


def _wait_for_gateway(store: MemoryStore, attempts: int = 20, delay_s: float = 3.0) -> str:
    """Ping the gateway, retrying with backoff before giving up.

    A single heavy ingest run (or a concurrent bench instance) can peg the
    gateway's CPU so a fresh client's first Ping exceeds its deadline even
    though the gateway is alive. Rather than hard-fail, wait it out — matches
    the login-retry the sibling eval already does.
    """
    import time as _time

    last = ""
    for attempt in range(1, attempts + 1):
        try:
            return store.ping()
        except Exception as exc:  # noqa: BLE001 — gateway busy/loading, not dead
            last = str(exc).splitlines()[0][:80]
            if attempt < attempts:
                print(f"  ping attempt {attempt}/{attempts} failed ({last}); "
                      f"gateway busy — retrying in {delay_s:.0f}s")
                _time.sleep(delay_s)
    raise SystemExit(
        f"Gateway at {store._addr} unreachable after {attempts} pings ({last}). "
        f"Is it CPU-saturated by another bench run? (ps -o %cpu -p <gateway pid>)"
    )


def _default_detail_log_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(__file__).resolve().parent / f"last_run_detail_{ts}.log"


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _query_latency_summary(values: List[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "avg_s": None,
            "p50_s": None,
            "p75_s": None,
            "p90_s": None,
            "p95_s": None,
            "p99_s": None,
        }
    return {
        "count": len(values),
        "avg_s": sum(values) / len(values),
        "p50_s": _percentile(values, 50),
        "p75_s": _percentile(values, 75),
        "p90_s": _percentile(values, 90),
        "p95_s": _percentile(values, 95),
        "p99_s": _percentile(values, 99),
    }


def _emit_detail(fh, lines: List[str]) -> None:
    """Write a question's detail block to the log file (flushed so it's
    tailable mid-run), or to stdout when no file is open (--detail-log -)."""
    if fh is None:
        for line in lines:
            print(line)
        return
    fh.write("\n".join(lines) + "\n")
    fh.flush()


def _int_prop(props: dict, key: str) -> Optional[int]:
    value = props.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _hit_length_note(hit, shown_text: str) -> str:
    shown_chars = len(shown_text)
    span_chars = _int_prop(hit.props, "span_chars")
    document_chars = _int_prop(hit.props, "document_chars")
    parts = []
    if span_chars is not None:
        parts.append(f"span_chars={span_chars}")
        if shown_chars != span_chars:
            parts.append(f"shown_chars={shown_chars}")
    elif shown_chars:
        parts.append(f"text_chars={shown_chars}")
    if document_chars is not None:
        parts.append(f"document_chars={document_chars}")
    return f" [{' '.join(parts)}]" if parts else ""


def _preview(text: str, max_chars: int = 180) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _refresh_complete(value: dict) -> bool:
    if not value:
        return False
    status = value.get("enrichment_status") or value.get("refresh_status")
    if status in {"visible", "completed", "committed"}:
        return True
    # Synchronous ingest writes graph-node KV without enrichment_status; any
    # materialized gnode is already query-visible in that mode.
    return status is None


def _wait_for_refresh_barrier(store: MemoryStore, stats_list: List[object], timeout_s: float) -> None:
    if timeout_s <= 0:
        return
    pending = {
        (stats.graph, node_id)
        for stats in stats_list
        for node_id in getattr(stats, "node_ids", [])
    }
    if not pending:
        return

    deadline = time.time() + timeout_s
    last_print = 0.0
    total = len(pending)
    print(f"Refresh barrier: waiting for {total} injected nodes (timeout={timeout_s:.0f}s)")
    while pending:
        for graph, node_id in list(pending):
            value = store.get_json(f"gnode:{graph}:{node_id}".encode(), cf=1)
            if _refresh_complete(value):
                pending.remove((graph, node_id))

        if not pending:
            print(f"Refresh barrier: all {total} nodes visible")
            return

        now = time.time()
        if now >= deadline:
            print(
                f"Refresh barrier: timed out with {len(pending)}/{total} nodes still pending; "
                "continuing to query"
            )
            return
        if now - last_print >= 5.0:
            print(f"Refresh barrier: {total - len(pending)}/{total} visible")
            last_print = now
        time.sleep(1.0)


def _sleep_between_questions(seconds: float, question_index: int, total_questions: int) -> None:
    if seconds <= 0 or question_index >= total_questions:
        return
    deadline = time.time() + seconds
    print(
        f"Per-question wait: sleeping {seconds:.0f}s before question "
        f"{question_index + 1}/{total_questions}",
        flush=True,
    )
    last_print = time.time()
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        if time.time() - last_print >= 30.0:
            print(f"Per-question wait: {remaining:.0f}s remaining", flush=True)
            last_print = time.time()
        time.sleep(min(1.0, remaining))


def run(cfg: HarnessConfig, args: argparse.Namespace) -> dict:
    random.seed(cfg.seed)
    inject_only = getattr(args, "inject_only", False)
    if inject_only and args.no_ingest:
        raise SystemExit("--inject-only cannot be combined with --no-ingest")

    questions, q_total = load_questions(
        cfg.dataset_dir,
        limit=cfg.limit,
        only_ids=cfg.only_ids,
        start=cfg.start,
        with_total=True,
    )
    if not questions:
        raise SystemExit(f"No questions loaded from {cfg.dataset_dir}")

    q_offset = (cfg.start - 1) if (cfg.start and cfg.start > 0) else 0
    if cfg.start and cfg.start > 0:
        print(
            f"Resuming from question #{cfg.start}/{q_total} (sorted order); "
            f"{len(questions)} remaining."
        )
    print(f"Loaded {len(questions)} questions from {cfg.dataset_dir}")
    print(f"Categories: {category_counts(questions)}")
    print(
        f"Gateway: {cfg.gateway_addr}  k={cfg.k}  seed={cfg.seed}  "
        f"ingest_workers={cfg.ingest_workers}"
    )
    print(
        f"Ingest: granularity={cfg.ingest_granularity}  "
        f"retrieve_k={cfg.retrieve_k}  filter_fragments={cfg.filter_fragments}  "
        f"dedup={cfg.dedup_results}"
    )
    print(
        f"Answer bundle: enabled={cfg.include_answer_bundle}  "
        f"answer_results_limit={cfg.answer_results_limit}  "
        f"fact_k={cfg.answer_fact_k}  chunk_k={cfg.answer_chunk_k}  "
        f"max_chars={cfg.answer_evidence_max_chars}  "
        f"total_max_chars={cfg.answer_evidence_total_max_chars}"
    )
    print(
        "Complex memory: "
        f"answer_results_limit={cfg.answer_complex_results_limit}  "
        f"fact_k={cfg.answer_complex_fact_k}  chunk_k={cfg.answer_complex_chunk_k}  "
        f"max_chars={cfg.answer_complex_evidence_max_chars}  "
        f"total_max_chars={cfg.answer_complex_evidence_total_max_chars}"
    )
    judge_enabled = cfg.deepseek.enabled and not args.retrieval_only
    if judge_enabled:
        print(f"DeepSeek: ENABLED (answer={cfg.deepseek.model_for('answer')}, judge={cfg.deepseek.model_for('judge')})")
    else:
        reason = "retrieval-only flag" if args.retrieval_only else "no DEEPSEEK_API_KEY"
        print(f"DeepSeek: DISABLED ({reason}) — hit@{cfg.k} / recall@{cfg.k} only")

    token = _login(cfg)
    store = MemoryStore(cfg.gateway_addr, token=token, relogin=lambda: _login(cfg))
    print(f"Ping: {_wait_for_gateway(store)}")

    ds: Optional[DeepSeekClient] = DeepSeekClient(cfg.deepseek) if judge_enabled else None

    acc = MetricsAccumulator()
    started = time.time()

    # Per-question Q/A + top-k detail goes to a log file (keeps the terminal to
    # one summary line per question). '-' routes it back to stdout.
    detail_fh = None
    if cfg.detail_log != "-":
        detail_path = cfg.detail_log or str(_default_detail_log_path())
        detail_fh = open(detail_path, "w")
        print(f"Per-question detail → {detail_path}")

    # ── Pass 1: inject every question first ──
    # NRT ingest acknowledges before the refresh/indexing work is visible. Running
    # all injection up front gives the background refresh worker time to drain, and
    # the optional barrier below prevents retrieval metrics from racing the queue.
    ingest_notes = {}
    ingest_stats = []
    if not args.no_ingest:
        for i, q in enumerate(questions, start=1):
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"[{q_offset + i}/{q_total}] {q.question_id[:24]:<24} "
                f"[{q.question_type:<26}] ingesting {len(q.sessions)} sessions "
                f"(workers={cfg.ingest_workers})...",
                end="\r" if sys.stdout.isatty() else "\n",
                flush=True,
            )
            stats = inject_question(store, cfg, q)
            ingest_stats.append(stats)
            ingest_note = f"ingest {stats.n_units}u/{stats.n_sessions}s {stats.elapsed_s:.1f}s"
            if stats.n_failed:
                ingest_note += f" ({stats.n_failed} failed)"
            ingest_notes[q.question_id] = ingest_note
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"[{q_offset + i}/{q_total}] {q.question_id[:24]:<24} "
                f"[{q.question_type:<26}] {ingest_note}",
                flush=True,
            )
            _sleep_between_questions(cfg.per_question_wait_s, q_offset + i, q_total)
        _wait_for_refresh_barrier(store, ingest_stats, cfg.refresh_wait_s)
    else:
        for q in questions:
            ingest_notes[q.question_id] = "ingest skipped"

    if inject_only:
        if detail_fh is not None:
            detail_fh.close()
        elapsed = time.time() - started
        report = {
            "mode": "inject_only",
            "config": cfg.to_dict(),
            "elapsed_s": elapsed,
            "n": len(questions),
            "questions": [
                {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "ingest": ingest_notes.get(q.question_id, ""),
                }
                for q in questions
            ],
        }
        out_path = args.out or str(Path(__file__).resolve().parent / "last_report.json")
        write_report(out_path, report)
        print()
        print(f"Inject-only complete: {len(questions)} question(s) in {elapsed:.1f}s")
        print(f"Report written to {out_path}")
        return report

    # ── Pass 2: query + LLM-free retrieval score ──
    # We materialise (question, result, confidence) so Phase 6 can split a dev
    # split, calibrate a threshold on it, and apply it to the held-out test
    # split — all BEFORE the (expensive, LLM) answer/judge pass.
    records: List[tuple] = []  # (Question, QueryResult, Confidence)
    query_latencies_s: List[float] = []
    for i, q in enumerate(questions, start=1):
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"[{q_offset + i}/{q_total}] {q.question_id[:24]:<24} "
            f"[{q.question_type:<26}] querying...",
            end="\r" if sys.stdout.isatty() else "\n",
            flush=True,
        )
        try:
            qr = query_question(store, cfg, q)
        except Exception as exc:  # noqa: BLE001
            # Per-question fault tolerance: one broken/missing graph must not
            # kill the whole run (four incidents of a NOT_FOUND graph
            # truncating a 86/107/500-question sweep). Log and move on; the
            # question scores as a retrieval miss.
            print(
                f"      QUERY-ERROR {q.question_id}: {str(exc).splitlines()[0][:140]} — skipped",
                flush=True,
            )
            continue
        query_latencies_s.append(qr.search_ms / 1000.0)
        rscore = score_retrieval(q, qr, cfg.k)
        acc.add_retrieval(rscore)
        conf = Confidence.from_result(qr)
        records.append((q, qr, conf))
        ingest_note = ingest_notes.get(q.question_id, "ingest skipped")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = (
            f"{ts} [{q_offset + i}/{q_total}] {q.question_id[:24]:<24} "
            f"[{q.question_type:<26}] hit@{cfg.k}={'Y' if rscore.hit_at_k else 'n'} "
            f"recall={rscore.recall_at_k:.2f} span={rscore.answer_span_recall_at_k:.2f} "
            f"f1={rscore.best_token_f1:.2f} rL={rscore.best_rouge_l:.2f} "
            f"top={conf.top_score:.3f} gap={conf.score_gap:.3f} "
            f"query={qr.search_ms / 1000:.1f}s {ingest_note}"
        )
        # Concise summary stays on stdout; the verbose Q/A + top-k dump goes to
        # the detail log (or stdout when --detail-log -).
        print(summary, flush=True)
        gold = set(q.gold_session_indices)
        detail_lines = [
            summary,
            f"      Q: {q.question}",
            f"      D: {q.question_date}",
            f"      A: {q.answer}",
        ]
        has_memories = bool(getattr(qr, "memories", ""))
        if has_memories:
            # Gateway plain-text evidence block, newline-escaped onto one line so
            # the line-oriented detail.log parser (reader_judge) can recover it.
            # When present it supersedes the reader prompt, so the per-source
            # `@N` computed lines below are redundant — skip them (the `#N` hits
            # stay for retrieval metrics / session diagnostics).
            escaped = qr.memories.replace("\\", "\\\\").replace("\n", "\\n")
            detail_lines.append(f"      MEMORIES: {escaped}")
        else:
            for item in qr.computed_evidence or []:
                detail_lines.append(f"      @{item.rank:<2} {item.source} | {item.text}")
        for rank, hit in enumerate(qr.hits[: cfg.k], start=1):
            si = session_index_of(hit)
            mark = "*" if si is not None and si in gold else " "  # * == gold session
            text = " ".join(hit.text.split())
            preview = _preview(text, cfg.detail_max_chars)
            length_note = _hit_length_note(hit, preview)
            detail_lines.append(
                f"      {mark}#{rank:<2} sess={si!s:>4} dist={hit.distance:.3f} "
                f"f1={token_f1(hit.text, q.answer):.2f}{length_note} | {preview}"
            )
        _emit_detail(detail_fh, detail_lines)

    if detail_fh is not None:
        detail_fh.close()

    # ── Phase 6: calibrate abstention threshold on a dev split ──
    abst_report: Optional[dict] = None
    abstain_ids: set = set()
    if getattr(args, "abstention", False):
        dev_fraction = getattr(args, "abstention_dev_fraction", 0.5)
        min_retention = getattr(args, "abstention_min_retention", 0.95)
        # Split on the records themselves (carrying the question) so the policy
        # is calibrated on the dev split and evaluated on the DISJOINT held-out
        # test split — reporting on the dev examples the threshold was fit on
        # would leak and over-state the metric.
        dev_records, test_records = split_dev_test(
            [(is_abstention(q), (q, conf)) for (q, _qr, conf) in records],
            dev_fraction=dev_fraction,
            seed=cfg.seed,
        )
        dev = [(lbl, payload[1]) for lbl, payload in dev_records]
        calib = calibrate(dev, min_answerable_retention=min_retention)
        policy = calib.policy
        print()
        print(
            f"Abstention calibrated on dev (n={calib.stats.n}, "
            f"abst={calib.stats.n_abstention}, ans={calib.stats.n_answerable}): "
            f"{policy.to_dict()}  objective={calib.objective:.3f}"
        )

        # Apply the calibrated policy to the held-out test split only; record
        # per-question abstention metrics so the report shows the
        # precision/recall tradeoff on data the threshold never saw.
        n_test_abstain = 0
        for is_abs, (q, conf) in test_records:
            decided = policy.should_abstain(conf)
            acc.add_abstention(q.question_type, is_abs, decided)
            if decided:
                abstain_ids.add(q.question_id)
                n_test_abstain += 1
        print(
            f"Applied policy: abstained on {n_test_abstain}/{len(test_records)} "
            f"test questions "
            f"(abstention-acc={_pct(acc.overall.abstention_accuracy)}, "
            f"answerable-retention={_pct(acc.overall.answerable_retention)})"
        )
        abst_report = {
            "calibration": calib.to_dict(),
            "dev_fraction": dev_fraction,
            "min_answerable_retention": min_retention,
            "n_dev": len(dev_records),
            "n_test": len(test_records),
            "n_abstained": n_test_abstain,
        }

    # ── Pass 2: DeepSeek answer + judge (if enabled) ──
    if ds is not None:
        for q, qr, _conf in records:
            if q.question_id in abstain_ids:
                # Phase 6: low confidence -> abstain instead of forcing an
                # answer. We still submit the canonical abstention reply to the
                # judge so the metric reflects the decision (LLM-free choice).
                ans = ABSTAIN_ANSWER
                context: List[str] = []
            else:
                context = [h.text for h in qr.hits[: cfg.k] if h.text]
                ans = ds.answer(q.question, q.question_date, context)
            judgement = ds.judge(q.question, q.answer, ans)
            tokens = estimate_tokens(q.question, *context, ans)
            acc.add_judgement(q.question_type, judgement.correct, tokens)

    elapsed = time.time() - started

    print()
    print(render_table(acc, cfg.k, judged=judge_enabled))
    print()

    report = build_report(
        cfg, acc, judged=judge_enabled, elapsed_s=elapsed, abstention=abst_report
    )
    report["query_latency_s"] = _query_latency_summary(query_latencies_s)
    out_path = args.out or str(Path(__file__).resolve().parent / "last_report.json")
    write_report(out_path, report)
    print(f"Report written to {out_path}")
    return report


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    run(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

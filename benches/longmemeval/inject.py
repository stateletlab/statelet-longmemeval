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
"""Stage 1 — Inject.

For one question's multi-session chat history:
  * sentence-segment every message (merge-short / split-long), then
  * embed + ingest each unit through the normal gateway write path with
    bitemporal timestamps derived from session/message order, and
  * stamp `session_index` into props so hit@k can map results back to gold.

One namespaced graph per question (`<prefix>-<question_id>`), so questions are
fully isolated and a run is restartable per-question.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .config import HarnessConfig
from .dataset import Question
from .segment import segment
from .store import MemoryStore, _SESSION_STRIDE


# Bitemporal base: sessions are spaced one "tick" apart; messages within a
# session and units within a message get monotonically increasing valid_from so
# later facts sort after earlier ones even without real dates.
_SESSION_TICK = 1_000_000  # microsecond-ish spacing between sessions.
_MESSAGE_TICK = 1_000      # spacing between messages within a session.

# Node-id encoding so retrieval results map back to their haystack session.
# The gateway DISCARDS caller props on ingest (it owns its own schema), so the
# `session_index` we stamp never comes back on search. Instead we make the
# session recoverable from the node id, mirroring the working sibling eval:
# we put each unit under parent id `(session_index + 1) * _SESSION_STRIDE + seq`
# (seq < _SESSION_STRIDE, unique per unit within the session). The gateway then
# derives fact node ids as `parent * 1000 + offset` and stamps `parent_node_id`
# on session_raw/session_view nodes — both recover the session via
# `parent // _SESSION_STRIDE - 1` (store.session_index_of). The stride is
# imported from store (the decoder) so the two never drift.


def _parse_session_date(date_str: str) -> Optional[int]:
    """Best-effort parse of a LongMemEval session date to epoch seconds."""
    if not date_str:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def _valid_from(session_idx: int, msg_idx: int, unit_idx: int, session_epoch: Optional[int]) -> int:
    """Derive a bitemporal valid_from from order (+ real date when available)."""
    base = session_epoch if session_epoch is not None else session_idx * _SESSION_TICK
    return base + msg_idx * _MESSAGE_TICK + unit_idx


@dataclass
class InjectStats:
    question_id: str
    graph: str
    n_sessions: int
    n_messages: int
    n_units: int
    n_failed: int
    elapsed_s: float
    node_ids: List[int]


@dataclass
class _Unit:
    """One segmented retrieval unit, ready to put."""

    text: str
    props: dict
    valid_from: int
    node_id: int


def _session_id(q: Question, si: int) -> str:
    return q.haystack_session_ids[si] if si < len(q.haystack_session_ids) else f"sess_{si}"


def _build_units(cfg: HarnessConfig, q: Question) -> tuple[List[_Unit], int]:
    """Turn a question's full history into put-ready nodes at cfg.ingest_granularity.

    Returns (units, n_messages). Pure/CPU-only — no network — so the (slow,
    network-bound) puts can be fanned out afterwards. Every node is put under a
    parent id `(si+1)*_SESSION_STRIDE + seq` so the session is recoverable from
    the node id (the gateway discards caller props — see store.session_index_of).
    """
    gran = cfg.ingest_granularity
    units: List[_Unit] = []
    n_messages = 0

    for si, session in enumerate(q.sessions):
        session_date = q.session_dates[si] if si < len(q.session_dates) else ""
        session_epoch = _parse_session_date(session_date)
        session_base = (si + 1) * _SESSION_STRIDE
        base_props = {
            "session_index": si,
            "session_id": _session_id(q, si),
            "session_date": session_date,
        }

        if gran == "session":
            # One node = the whole conversation episode (matches the eval).
            lines: List[str] = []
            if session_date:
                lines.append(f"[{session_date}]")
            for msg in session:
                n_messages += 1
                content = (msg.get("content", "") or "").strip()
                if content:
                    lines.append(f"{msg.get('role', 'user')}: {content}")
            text = "\n".join(lines)
            if text:
                props = {**base_props, "role": "session", "text": text}
                units.append(
                    _Unit(text, props, _valid_from(si, 0, 0, session_epoch), session_base)
                )
            continue

        if gran == "turn":
            # One node per message.
            for mi, msg in enumerate(session):
                n_messages += 1
                content = (msg.get("content", "") or "").strip()
                if not content:
                    continue
                props = {**base_props, "message_index": mi,
                         "role": msg.get("role", "user"), "text": content}
                units.append(
                    _Unit(content, props, _valid_from(si, mi, 0, session_epoch), session_base + mi)
                )
            continue

        # gran == "unit": legacy sub-sentence segmentation.
        unit_seq = 0
        for mi, msg in enumerate(session):
            n_messages += 1
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            for ui, unit in enumerate(segment(content, cfg.min_unit_tokens, cfg.max_unit_tokens)):
                props = {**base_props, "message_index": mi, "role": role,
                         "unit_index": ui, "text": unit}
                units.append(
                    _Unit(unit, props, _valid_from(si, mi, ui, session_epoch), session_base + unit_seq)
                )
                unit_seq += 1

    return units, n_messages


def inject_question(store: MemoryStore, cfg: HarnessConfig, q: Question) -> InjectStats:
    """Segment + ingest one question's full history. Returns ingest stats.

    Each unit is an independent, synchronous gRPC put that embeds server-side,
    so a single question's hundreds of units dominate wall-clock. The puts carry
    their own bitemporal `valid_from`, so order is irrelevant and we fan them out
    across `cfg.ingest_workers` threads (gRPC Python channels are thread-safe).
    """
    graph = f"{cfg.graph_prefix}-{q.question_id}"
    started = time.time()

    # `--reset`: drop any prior graph so the re-ingest starts clean. Without it
    # ensure_graph is idempotent and a rerun ACCUMULATES duplicate units into
    # the existing graph (which pollutes recall — see the polluted lme-p0 run).
    if cfg.reset:
        store.drop_graph(graph)

    # The graph index must exist before the first put (the derived `__fact`
    # index is not auto-created), else put_unit fails with NOT_FOUND.
    # dim=0 ⇒ the gateway auto-fills the dimension from its loaded embedding
    # model (the single source of truth), so swapping the embedder never leaves
    # graphs at a stale dim. Fall back to the name-inferred dim only if the
    # gateway is too old to auto-fill (rejects dim 0).
    try:
        store.ensure_graph(graph, dim=0)
    except Exception:  # noqa: BLE001 — older gateway: pass the inferred dim
        store.ensure_graph(graph, dim=cfg.resolved_embed_dim())

    units, n_messages = _build_units(cfg, q)

    total = len(units)

    def _put(u: _Unit):
        """Put one unit; return (duration_s, error_or_None). Catches internally
        so a failure still reports its latency (and the reason) instead of being
        swallowed — each unit is one session under --granularity session."""
        t0 = time.time()
        try:
            store.put_unit(
                graph, u.text, u.props, valid_from=u.valid_from, valid_to=0, node_id=u.node_id
            )
            return (time.time() - t0, None)
        except Exception as exc:  # noqa: BLE001 — report + count, keep going
            # gRPC errors stringify across multiple lines (the StatusCode and
            # details are NOT on the first line); pull them out so the reason is
            # actually visible. Fall back to a flattened repr otherwise.
            code = getattr(exc, "code", None)
            details = getattr(exc, "details", None)
            if callable(code) and callable(details):
                return (time.time() - t0, f"{code()} {details()}")
            return (time.time() - t0, " ".join(str(exc).split()))

    n_failed = 0
    done = 0
    durations: List[float] = []
    lock = threading.Lock()

    def _report(u: _Unit, dur: float, err: Optional[str]) -> None:
        nonlocal done, n_failed
        with lock:
            done += 1
            durations.append(dur)
            if err is not None:
                n_failed += 1
            status = "ok" if err is None else f"FAILED: {err.splitlines()[0][:140]}"
            print(
                f"      inject {done}/{total} node={u.node_id} {dur:5.1f}s {status}",
                file=sys.stderr,
                flush=True,
            )

    workers = max(1, cfg.ingest_workers)
    if workers == 1:
        for u in units:
            dur, err = _put(u)
            _report(u, dur, err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_unit = {pool.submit(_put, u): u for u in units}
            for fut in as_completed(fut_to_unit):
                u = fut_to_unit[fut]
                dur, err = fut.result()
                _report(u, dur, err)

    if durations:
        durations.sort()
        slowest = durations[-1]
        median = durations[len(durations) // 2]
        print(
            f"      inject done: {total - n_failed}/{total} ok, "
            f"per-session median={median:.1f}s slowest={slowest:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return InjectStats(
        question_id=q.question_id,
        graph=graph,
        n_sessions=len(q.sessions),
        n_messages=n_messages,
        n_units=len(units),
        n_failed=n_failed,
        elapsed_s=time.time() - started,
        node_ids=[u.node_id for u in units],
    )

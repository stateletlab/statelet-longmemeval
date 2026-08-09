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
"""Pinned configuration for the LongMemEval Phase 0 harness.

Everything that affects reproducibility lives here: dataset path, gateway
address, embedder/k knobs (server-side toggles are env-driven on the gateway,
so we only record them here for the report), DeepSeek model + sampling, and the
mem0 published baseline we compare against.

A run is reproducible from (config + seed + dataset + gateway env). The CLI
(`run.py`) may override individual fields, but the defaults below are the pinned
parity configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# The six LongMemEval question categories, in canonical report order.
CATEGORIES: List[str] = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
]


# mem0's published LongMemEval numbers (the ~94% run we mirror for parity).
# Source: https://mem0.ai/research and https://github.com/mem0ai/memory-benchmarks
# These are accuracy (%), the LLM-judged metric — the side-by-side comparison row
# in the report. Categories without a published per-type number are omitted.
MEM0_ACCURACY: Dict[str, float] = {
    "overall": 94.4,
    "multi-session": 88.0,
    "temporal-reasoning": 76.7,
}


def _default_dataset_dir() -> str:
    """Locate the LongMemEval questions directory.

    Precedence:
      1. LONGMEMEVAL_QUESTIONS_DIR env var.
      2. The memorybench sibling checkout used by the legacy eval/ harness.
      3. A bundled `data/questions/` under this package (small smoke set).
    """
    env = os.environ.get("LONGMEMEVAL_QUESTIONS_DIR")
    if env:
        return env
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    sibling = (
        repo_root.parent
        / "memorybench"
        / "data"
        / "benchmarks"
        / "longmemeval"
        / "datasets"
        / "questions"
    )
    if sibling.is_dir():
        return str(sibling)
    bundled = here.parent / "data" / "questions"
    return str(bundled)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _default_refresh_wait_s() -> float:
    raw = os.environ.get("STATELET_LME_REFRESH_WAIT_S")
    if raw is not None:
        return float(raw)
    ingest_mode = os.environ.get("STATELET_INGEST_MODE", "").strip().lower()
    if _env_flag("STATELET_TEXT_GRAPH_NRT") or ingest_mode in {
        "nrt",
        "near-real-time",
        "near_real_time",
    }:
        return 600.0
    return 0.0


@dataclass
class DeepSeekConfig:
    """Reader + judge LLM. The ONLY LLM in the loop (the memory layer is
    LLM-free per #741). Accuracy requires `DEEPSEEK_API_KEY`; hit@k / recall@k
    do not."""

    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    # deepseek-chat for the answerer + judge (parity with mem0's reader). A
    # reasoner model may be selected for the answerer via DEEPSEEK_ANSWER_MODEL.
    model: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    answer_model: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_ANSWER_MODEL", "")
    )
    judge_model: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_JUDGE_MODEL", "")
    )
    # Pinned to 0 for deterministic, reproducible judging (matches mem0/LME).
    temperature: float = 0.0
    max_retries: int = 5
    timeout_s: int = 90

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def model_for(self, role: str) -> str:
        if role == "answer" and self.answer_model:
            return self.answer_model
        if role == "judge" and self.judge_model:
            return self.judge_model
        return self.model


@dataclass
class HarnessConfig:
    """Top-level pinned config for one harness run."""

    # ── Dataset ──
    dataset_dir: str = field(default_factory=_default_dataset_dir)
    limit: int = 0  # 0 == all 500 questions.
    start: int = 0  # 1-based resume index into the sorted set (0 == from the start).
    only_ids: Optional[List[str]] = None

    # ── Gateway / store ──
    gateway_addr: str = field(
        default_factory=lambda: os.environ.get("STATELET_ADDR", "127.0.0.1:9379")
    )
    # Management HTTP address for JWT login. Empty == derive from gateway_addr
    # (swap the gRPC :9379 for the mgmt :9380). The cluster scripts start the
    # gateway with GATEWAY_JWT_SECRET set, so auth is ON; ping is exempt but
    # TextGraphPut/Search require an `Authorization: Bearer <token>` header.
    mgmt_addr: str = field(default_factory=lambda: os.environ.get("STATELET_MGMT_ADDR", ""))
    username: str = field(default_factory=lambda: os.environ.get("STATELET_USER", "admin"))
    password: str = field(default_factory=lambda: os.environ.get("STATELET_PASSWORD", "admin"))
    graph_prefix: str = "lme-p0"  # one namespaced graph per question.
    # Drop each per-question graph before ingest so a rerun starts clean.
    # Off by default (never deletes data unless asked); set via --reset.
    reset: bool = False

    def resolved_mgmt_addr(self) -> str:
        if self.mgmt_addr:
            return self.mgmt_addr
        return self.gateway_addr.replace(":9379", ":9380")

    # ── Ingest granularity ──
    # "session" (default): one node per session — the whole conversation episode
    # as one put, matching eval/longmemeval_retrieval_eval.py. Far fewer, more
    # coherent nodes than sub-sentence units, which were drowning the top-k in
    # tiny fragments. "turn": one node per message. "unit": legacy 8-64 token
    # sub-sentence segmentation (over-chunked; kept for A/B only).
    ingest_granularity: str = field(
        default_factory=lambda: os.environ.get("STATELET_INGEST_GRANULARITY", "session")
    )

    # ── Retrieval ──
    k: int = 5  # hit@k / recall@k. Default reader context returns top-5 sessions.
    ef: int = 0  # 0 == server default HNSW ef.
    # Over-fetch then filter/dedup down to k: the gateway also surfaces entity/
    # fragment nodes (unmappable to a session) that crowd out answer-bearing
    # session text. We retrieve `retrieve_k`, drop fragment nodes, dedup repeated
    # text, then keep the top k. retrieve_k <= k disables over-fetch.
    retrieve_k: int = 50
    filter_fragments: bool = True  # drop hits with no recoverable session.
    dedup_results: bool = True     # collapse identical retrieved text.
    # ── Multi-granularity read funnel (#741 ②/⑤, #827) ──
    # Granularities the gateway runs one ANN ranking over and RRF-fuses at query
    # time (sentence→__chunk, round, session, fact→__fact). Pinned in the request
    # here rather than left to the gateway's STATELET_GRANULARITIES env, so a run's
    # read funnel is reproducible from config alone (and doesn't silently degrade
    # to session-only if the server env drifts). Empty list ⇒ defer to the server.
    # Read-time multi-granularity request. `sentence` is omitted by default because
    # it floods the funnel with sub-fragment nodes that can crowd out session-level
    # memory. Override with STATELET_GRANULARITIES.
    read_granularities: List[str] = field(
        default_factory=lambda: [
            g.strip()
            for g in os.environ.get(
                "STATELET_GRANULARITIES", "session,round,fact"
            ).split(",")
            if g.strip()
        ]
    )
    # RRF constant k for the multi-granularity fusion (0 ⇒ server default, 60).
    # 10 (top-heavy fusion) lets a channel's rank-1 hit dominate, which helps
    # answer-bearing sessions surface for count/aggregation queries. Override with
    # STATELET_RRF_K.
    rrf_k: int = field(default_factory=lambda: int(os.environ.get("STATELET_RRF_K", "10")))
    # The memory layer (extraction + retrieval) stays LLM-free per #741, so the
    # Concurrent put_unit RPCs per question during ingest. Each put embeds
    # server-side, so ingest is network-bound — fanning out across threads is
    # the difference between a usable and an unusable full 500-question run.
    # 1 == sequential. Override via --workers / STATELET_INGEST_WORKERS.
    ingest_workers: int = field(
        default_factory=lambda: int(os.environ.get("STATELET_INGEST_WORKERS", "8"))
    )
    # When the gateway runs near-real-time TextGraphPut, puts return after source
    # durability + refresh enqueue. The harness waits for refresh visibility before
    # querying so retrieval metrics do not race the background indexer.
    refresh_wait_s: float = field(default_factory=_default_refresh_wait_s)
    # Optional throttle between per-question ingest batches. This is useful with
    # NRT ingest, where TextGraphPut can enqueue much faster than the background
    # refresh worker can build facts/chunks/atoms.
    per_question_wait_s: float = field(
        default_factory=lambda: float(
            os.environ.get("STATELET_LME_PER_QUESTION_WAIT_S", "0")
        )
    )

    # ── Injector (sentence segmentation) ──
    # Merge units shorter than this many whitespace tokens into the next; split
    # units longer than max_unit_tokens at sentence boundaries. Mirrors the
    # server-side STATELET_SENTENCE_UNITS rule-based segmenter (~16-48 tok).
    min_unit_tokens: int = 8
    max_unit_tokens: int = 64

    # ── Output ──
    # Path for the per-question Q/A + top-k detail dump. None → a default file
    # (last_run_detail_YYYYMMDD_HHMMSS_ffffff.log); "-" → print to stdout.
    detail_log: Optional[str] = None
    # Max characters of each retrieved hit written to the detail log.
    # 0 means write the full retrieved text. Reader-judge consumes this file as
    # its memory prompt, so the default must not hide list items, operands, or
    # long assistant answers behind a preview cap.
    detail_max_chars: int = 0
    # Ask the gateway for answer-oriented JSON artifacts and write only a
    # compressed subset to the reader detail log. This gives multi-hop/count/
    # temporal questions their operands without dumping the full bundle into the
    # LLM prompt.
    include_answer_bundle: bool = field(
        default_factory=lambda: not _env_flag("LONGMEMEVAL_NO_ANSWER_BUNDLE")
    )
    answer_results_limit: int = field(
        default_factory=lambda: int(os.environ.get("LONGMEMEVAL_ANSWER_RESULTS_LIMIT", "2"))
    )
    # Complex questions (multi-hop counts, temporal arithmetic, ordered lists,
    # preference questions with constraints) need more operands than simple
    # lookup questions. These caps are applied per-question by query.py only
    # when the question classifier says the extra context is likely useful.
    answer_complex_results_limit: int = field(
        default_factory=lambda: int(
            os.environ.get("LONGMEMEVAL_ANSWER_COMPLEX_RESULTS_LIMIT", "6")
        )
    )
    answer_evidence_max_chars: int = field(
        default_factory=lambda: int(os.environ.get("LONGMEMEVAL_ANSWER_EVIDENCE_MAX_CHARS", "800"))
    )
    answer_complex_evidence_max_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("LONGMEMEVAL_ANSWER_COMPLEX_EVIDENCE_MAX_CHARS", "1000")
        )
    )
    answer_evidence_total_max_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("LONGMEMEVAL_ANSWER_EVIDENCE_TOTAL_MAX_CHARS", "3000")
        )
    )
    answer_complex_evidence_total_max_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("LONGMEMEVAL_ANSWER_COMPLEX_EVIDENCE_TOTAL_MAX_CHARS", "6000")
        )
    )
    answer_fact_k: int = field(
        default_factory=lambda: int(os.environ.get("LONGMEMEVAL_ANSWER_FACT_K", "2"))
    )
    answer_complex_fact_k: int = field(
        default_factory=lambda: int(os.environ.get("LONGMEMEVAL_ANSWER_COMPLEX_FACT_K", "4"))
    )
    answer_chunk_k: int = field(
        default_factory=lambda: int(os.environ.get("LONGMEMEVAL_ANSWER_CHUNK_K", "2"))
    )
    answer_complex_chunk_k: int = field(
        default_factory=lambda: int(os.environ.get("LONGMEMEVAL_ANSWER_COMPLEX_CHUNK_K", "4"))
    )

    # ── Reproducibility ──
    seed: int = 1337

    # ── Embedder (recorded for the report; the actual embed happens
    # server-side on the gateway via STATELET_*_MODEL env). ──
    embedder: str = field(
        default_factory=lambda: os.environ.get("STATELET_EMBED_MODEL_NAME", "multilingual-e5-small")
    )
    # HNSW index dimension for graph creation. 0 == infer from the embedder
    # name (the gateway rejects dim<=0). Override via STATELET_EMBED_DIM.
    embed_dim: int = field(
        default_factory=lambda: int(os.environ.get("STATELET_EMBED_DIM", "0"))
    )

    def resolved_embed_dim(self) -> int:
        if self.embed_dim > 0:
            return self.embed_dim
        name = self.embedder.lower()
        if "bge-m3" in name or "e5-large" in name or "bge-large" in name:
            return 1024
        if "e5-base" in name or "bge-base" in name:
            return 768
        # Default: multilingual-e5-small (cluster-start.sh default).
        return 384

    # ── DeepSeek reader + judge ──
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't leak the API key into pinned report artifacts.
        if "deepseek" in d and isinstance(d["deepseek"], dict):
            d["deepseek"]["api_key"] = "***" if self.deepseek.enabled else ""
        return d

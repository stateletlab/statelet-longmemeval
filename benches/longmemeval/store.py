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
"""Thin wrapper over the Statelet Python SDK gateway client.

Isolates the harness from SDK details: opens the connection, exposes the two
RPCs the harness needs (TextGraphPut / TextGraphSearch), and parses the
JSON `properties` blob carried on each search result back into a dict.

The harness writes through the NORMAL gateway write path. Per #741 the memory
layer's LLM usage is controlled SERVER-SIDE by STATELET_LLM_EXTRACT: the
per-request skip_server_llm flag was request-logged but overridden by the
gateway env, so the harness no longer sends it.

That means ingestion is LLM-free only when the gateway says so. With
STATELET_LLM_EXTRACT unset the gateway uses its rule-based extractor +
embedder; with it set to 1 an LLM (STATELET_LLM_MODEL, DeepSeek by default,
and `cluster-741.sh` turns this ON) distills facts and resolves conflicts
during ingest. Retrieval stays local either way.

Nothing on this side of the wire can observe or record which mode was in
effect, so a run's ingest configuration is NOT recoverable from harness
artifacts — encode it in the run directory name. See the README's phase table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Canonical session<->node-id encoding stride (the injector imports this).
# Each unit is put under parent id `(session_index + 1) * _SESSION_STRIDE + seq`
# with `seq < _SESSION_STRIDE`, so `parent // _SESSION_STRIDE - 1` recovers the
# session. The gateway then derives fact node ids as `parent * 1000 + offset`.
_SESSION_STRIDE = 1_000_000
_FACT_ID_MULTIPLIER = 1_000  # gateway: fact_node_id = parent_id * 1000 + offset
_DERIVED_GRAPH_SUFFIXES = ("__fact", "__chunk", "__round", "__session", "__entity")


def _sdk_path() -> str:
    # benches/longmemeval/store.py -> repo_root/sdk/python
    return str(Path(__file__).resolve().parents[2] / "sdk" / "python")


@dataclass
class SearchHit:
    node_id: int
    distance: float
    props: dict
    text: str = ""


@dataclass
class SearchResponse:
    hits: List[SearchHit]
    fact_hits: List[SearchHit]
    chunk_hits: List[SearchHit]
    answer_bundle: dict
    primary_answer_result: dict
    answer_results: List[dict]
    compact_memory: List[dict] = field(default_factory=list)
    # Gateway-built plain-text evidence block (resp.memories), when
    # STATELET_READER_BLOCK=1. Empty otherwise. Ready to feed an LLM reader.
    memories: str = ""


def _rpc_code_name(exc: Exception) -> str:
    code_fn = getattr(exc, "code", None)
    if not callable(code_fn):
        return ""
    try:
        code = code_fn()
    except Exception:  # noqa: BLE001
        return ""
    return getattr(code, "name", str(code).split(".")[-1]).upper()


def _rpc_details(exc: Exception) -> str:
    details_fn = getattr(exc, "details", None)
    if callable(details_fn):
        try:
            return str(details_fn())
        except Exception:  # noqa: BLE001
            pass
    return " ".join(str(exc).split())


def _rpc_metadata_value(exc: Exception, key: str) -> Optional[str]:
    trailers_fn = getattr(exc, "trailing_metadata", None)
    if not callable(trailers_fn):
        return None
    try:
        trailers = trailers_fn() or []
    except Exception:  # noqa: BLE001
        return None
    key_l = key.lower()
    for item in trailers:
        try:
            k, v = item
        except Exception:  # noqa: BLE001
            continue
        if str(k).lower() == key_l:
            if isinstance(v, bytes):
                return v.decode("utf-8", "replace")
            return str(v)
    return None


class MemoryStore:
    """One gateway connection, namespaced graphs per question."""

    def __init__(
        self,
        addr: str,
        token: Optional[str] = None,
        relogin=None,
    ):
        sys.path.insert(0, _sdk_path())
        from statelet.client import StateletClient  # noqa: E402

        self._addr = addr
        # Zero-arg callable returning a fresh JWT (or None). The gateway's
        # tokens expire after 24h — shorter than a full 500-question ingest —
        # so a long run must be able to re-login mid-flight instead of dying
        # with UNAUTHENTICATED: ExpiredSignature.
        self._relogin = relogin
        # The SDK defaults (put 90s / search 60s) are too short for this harness:
        # a single put runs the full no-LLM extraction pipeline server-side
        # (gliner + t5 + svo + sparse + reranker), and the FIRST put also pays
        # cold lazy-loading of all those ONNX models — a large session easily
        # exceeds 60-90s and the RPC dies with _InactiveRpcError before the
        # gateway finishes (which then looks like a spurious "failed"). Raise the
        # client deadlines (env-overridable) so slow-but-succeeding ops aren't cut.
        put_timeout = float(os.environ.get("STATELET_PUT_TIMEOUT_S", "300"))
        search_timeout = float(os.environ.get("STATELET_SEARCH_TIMEOUT_S", "180"))
        self._put_max_retries = int(os.environ.get("STATELET_LME_PUT_RETRIES", "20"))
        self._put_retry_base_s = float(os.environ.get("STATELET_LME_PUT_RETRY_BASE_S", "5"))
        self._put_retry_max_s = float(os.environ.get("STATELET_LME_PUT_RETRY_MAX_S", "300"))
        # `token` authenticates against a JWT-enabled gateway (the cluster
        # scripts enable auth). None is fine for a --no-auth gateway: ping and
        # the data RPCs both run unauthenticated there.
        self._client_cls = StateletClient
        self._put_timeout = put_timeout
        self._search_timeout = search_timeout
        self._connect(token)

    def _connect(self, token: Optional[str]) -> None:
        # Graph admin (drop/create) fans a delete out over ~20 KV prefixes ×
        # shards; on a 50-session graph that regularly exceeds the SDK's 30s
        # default right after a gateway restart. 300s keeps --reset alive.
        admin_timeout = float(os.environ.get("STATELET_GRAPH_ADMIN_TIMEOUT_S", "300"))
        self._db = self._client_cls(
            self._addr,
            token=token,
            text_graph_put_timeout_s=self._put_timeout,
            text_graph_search_timeout_s=self._search_timeout,
            graph_admin_timeout_s=admin_timeout,
        )

    def _refresh_token(self) -> bool:
        """Re-login and rebuild the client. Returns True if a token was obtained."""
        if self._relogin is None:
            return False
        try:
            token = self._relogin()
        except Exception as exc:  # noqa: BLE001
            print(f"      re-login failed: {exc}", file=sys.stderr, flush=True)
            return False
        if not token:
            return False
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass
        self._connect(token)
        print("      re-login OK (token refreshed)", file=sys.stderr, flush=True)
        return True

    def ping(self) -> str:
        return self._db.ping()

    def drop_graph(self, graph: str) -> None:
        """Drop a logical graph and its derived indexes if they exist.

        Newer gateways cascade `drop_graph_index(graph)` to derived indexes, but
        older gateways only drop the logical graph. Explicitly trying the known
        derived names keeps `--reset` usable across both versions.
        """
        for name in (graph, *(f"{graph}{suffix}" for suffix in _DERIVED_GRAPH_SUFFIXES)):
            # Slow-path retry: a drop right after gateway startup (cold model
            # load + a large fan-out delete) can exceed the RPC deadline once
            # or twice before succeeding; the operation is idempotent.
            for attempt in range(4):
                try:
                    self._db.drop_graph_index(name)
                    break
                except Exception as exc:  # noqa: BLE001
                    code = _rpc_code_name(exc)
                    if code == "UNAUTHENTICATED" and self._refresh_token():
                        continue
                    msg = str(exc).lower()
                    if "not found" in msg or "not_found" in msg or "does not exist" in msg:
                        break
                    if code in ("DEADLINE_EXCEEDED", "UNAVAILABLE") and attempt < 3:
                        print(
                            f"      drop retry graph={name} code={code} attempt={attempt + 1}/4",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(15)
                        continue
                    raise

    def ensure_graph(self, graph: str, dim: int, timeout_s: Optional[float] = None) -> None:
        """Create the graph index if it doesn't exist yet.

        text_graph_put writes into a derived `<graph>__fact` index that the
        gateway will not auto-create, so an un-created graph fails the first
        put with NOT_FOUND. `dim` must match the gateway's embedder dimension
        (the gateway rejects dim<=0). Idempotent: an already-existing graph is
        treated as success.
        """
        try:
            self._db.create_graph_index(graph, dim=dim, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            if _rpc_code_name(exc) == "UNAUTHENTICATED" and self._refresh_token():
                self.ensure_graph(graph, dim, timeout_s=timeout_s)
                return
            msg = str(exc).lower()
            if "already" in msg or "exists" in msg:
                return
            raise

    def put_unit(
        self,
        graph: str,
        text: str,
        props: dict,
        valid_from: int = 0,
        valid_to: int = 0,
        node_id: int = 0,
        timeout_s: Optional[float] = None,
    ) -> int:
        """Ingest one retrieval unit with bitemporal validity + session props.

        `node_id` encodes the haystack session (see inject._SESSION_STRIDE): the
        gateway discards `props` on ingest, so the session is recovered from the
        node id at search time, not from the properties blob.
        """
        retryable = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED"}
        attempt = 0
        while True:
            try:
                return self._db.text_graph_put(
                    graph,
                    text,
                    node_id=node_id,
                    properties=json.dumps(props).encode(),
                    edge_valid_from=valid_from,
                    edge_valid_to=valid_to,
                    timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                code = _rpc_code_name(exc)
                if code == "UNAUTHENTICATED" and attempt < self._put_max_retries:
                    # 24h JWT expiry mid-run: refresh and go again.
                    if self._refresh_token():
                        attempt += 1
                        continue
                if code not in retryable or attempt >= self._put_max_retries:
                    raise
                delay_s = self._retry_delay_s(exc, attempt)
                print(
                    "      put retry "
                    f"node={node_id} graph={graph} code={code} "
                    f"attempt={attempt + 1}/{self._put_max_retries} "
                    f"sleep={delay_s:.1f}s reason={_rpc_details(exc)[:160]}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_s)
                attempt += 1

    def _retry_delay_s(self, exc: Exception, attempt: int) -> float:
        retry_after_ms = _rpc_metadata_value(exc, "retry-after-ms")
        if retry_after_ms:
            try:
                return max(0.1, min(self._put_retry_max_s, float(retry_after_ms) / 1000.0))
            except ValueError:
                pass
        retry_after_s = _rpc_metadata_value(exc, "retry-after")
        if retry_after_s:
            try:
                return max(0.1, min(self._put_retry_max_s, float(retry_after_s)))
            except ValueError:
                pass
        return min(self._put_retry_max_s, self._put_retry_base_s * (2 ** min(attempt, 6)))

    def search(
        self,
        graph: str,
        query: str,
        k: int,
        ef: int = 0,
        granularities: Optional[List[str]] = None,
        rrf_k: int = 0,
        timeout_s: Optional[float] = None,
    ) -> List[SearchHit]:
        """Top-k retrieval; results carry the JSON props we wrote at ingest.

        `granularities` pins the multi-granularity read funnel in the request
        (empty/None ⇒ the gateway falls back to its STATELET_GRANULARITIES env)."""
        return self.search_with_bundle(
            graph,
            query,
            k,
            ef=ef,
            granularities=granularities,
            rrf_k=rrf_k,
            timeout_s=timeout_s,
        ).hits

    def search_with_bundle(
        self,
        graph: str,
        query: str,
        k: int,
        ef: int = 0,
        granularities: Optional[List[str]] = None,
        rrf_k: int = 0,
        include_answer_bundle: bool = False,
        fact_k: int = 0,
        chunk_k: int = 0,
        context_date: str = "",
        timeout_s: Optional[float] = None,
        extra_queries: Optional[List[str]] = None,
    ) -> SearchResponse:
        """Top-k retrieval plus optional answer-oriented gateway artifacts."""
        resp = self._db.text_graph_search(
            graph,
            query,
            k=k,
            ef=ef,
            include_answer_bundle=include_answer_bundle,
            context_date=context_date,
            fact_k=fact_k,
            chunk_k=chunk_k,
            granularities=granularities or None,
            rrf_k=rrf_k,
            timeout_s=timeout_s,
            extra_queries=extra_queries,
        )
        def convert(results) -> List[SearchHit]:
            hits: List[SearchHit] = []
            for r in results:
                props = _parse_props(getattr(r, "properties", b""))
                hits.append(
                    SearchHit(
                        node_id=r.node_id,
                        distance=r.distance,
                        props=props,
                        text=props.get("text", ""),
                    )
                )
            return hits

        answer_results = []
        for raw in getattr(resp, "answer_results_json", []) or []:
            parsed = _parse_props(raw)
            if parsed:
                answer_results.append(parsed)
        answer_bundle = _parse_props(getattr(resp, "answer_bundle_json", b""))
        compact_memory = answer_bundle.get("compact_memory", [])
        if not isinstance(compact_memory, list):
            compact_memory = []
        return SearchResponse(
            hits=convert(resp.results),
            fact_hits=convert(getattr(resp, "fact_results", []) or []),
            chunk_hits=convert(getattr(resp, "chunk_results", []) or []),
            answer_bundle=answer_bundle,
            primary_answer_result=_parse_props(
                getattr(resp, "primary_answer_result_json", b"")
            ),
            answer_results=answer_results,
            compact_memory=compact_memory,
            memories=getattr(resp, "memories", "") or "",
        )

    def get_json(self, key: bytes, *, cf: int = 1) -> dict:
        raw = self._db.get(key, cf=cf)
        return _parse_props(raw)


def _parse_props(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = _strip_json_prefix(bytes(raw))
        try:
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _strip_json_prefix(raw: bytes) -> bytes:
    if not raw:
        return raw
    if raw[:1] in {b"{", b"[", b'"'}:
        return raw
    starts = [idx for marker in (b"{", b"[") if (idx := raw.find(marker)) >= 0]
    if starts:
        return raw[min(starts):]
    return raw


def _session_from_parent(parent: int) -> Optional[int]:
    """Decode a parent id `(si+1)*_SESSION_STRIDE + seq` back to its session."""
    if parent >= _SESSION_STRIDE:
        return parent // _SESSION_STRIDE - 1
    return None


def session_index_of(hit: SearchHit) -> Optional[int]:
    """Recover the haystack session index a hit belongs to.

    The gateway discards caller props on ingest, so `session_index` rarely
    survives — recovery is via the node-id encoding the injector chose
    (see _SESSION_STRIDE). Three node shapes come back from a search:

      * extracted fact nodes — id = parent * _FACT_ID_MULTIPLIER + offset, so
        the parent (which encodes the session) is `node_id // _FACT_ID_MULTIPLIER`;
      * session_raw / session_view nodes — carry `parent_node_id` in props;
      * entity nodes (e.g. "Chicago") — no parent, unmappable -> None.

    `session_index` in props is still honored first, for forward-compat in case
    a future gateway preserves it.
    """
    si = hit.props.get("session_index")
    if isinstance(si, int):
        return si

    parent = hit.props.get("parent_node_id")
    if isinstance(parent, int):
        decoded = _session_from_parent(parent)
        if decoded is not None:
            return decoded

    # Extracted fact node: node_id = parent * _FACT_ID_MULTIPLIER + offset.
    # Bounded below by the smallest fact id (parent >= _SESSION_STRIDE) and above
    # to exclude the gateway's snowflake ids (~1e18) for entity/raw nodes.
    nid = hit.node_id
    if _SESSION_STRIDE * _FACT_ID_MULTIPLIER <= nid < _SESSION_STRIDE * _SESSION_STRIDE:
        return _session_from_parent(nid // _FACT_ID_MULTIPLIER)
    return None

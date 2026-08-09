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
"""End-to-end orchestration test driving run.run() against an in-memory fake
store (no gateway, no DeepSeek). Exercises inject -> query -> hit@k -> report.

    python benches/longmemeval/tests/test_run_e2e.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benches.longmemeval import run as run_mod  # noqa: E402
from benches.longmemeval.config import HarnessConfig  # noqa: E402
from benches.longmemeval.store import SearchHit  # noqa: E402

_DATA = str(_REPO_ROOT / "benches" / "longmemeval" / "data" / "questions")


class FakeStore:
    """In-memory stand-in for MemoryStore. Stores units per graph and does a
    trivial keyword overlap ranker so retrieval is deterministic and testable.
    """

    def __init__(self, *_args, **_kwargs):
        self.graphs = defaultdict(list)  # graph -> [(text, props)]
        self._next_id = 1

    def ping(self):
        return "ok"

    def ensure_graph(self, graph, dim, timeout_s=None):
        self.graphs.setdefault(graph, [])

    def drop_graph(self, graph):
        self.graphs.pop(graph, None)

    def put_unit(
        self,
        graph,
        text,
        props,
        valid_from=0,
        valid_to=0,
        node_id=0,
        timeout_s=None,
    ):
        props = dict(props)
        props.setdefault("text", text)
        nid = node_id or self._next_id
        self.graphs[graph].append((nid, text, props))
        self._next_id += 1
        return nid

    def search(
        self,
        graph,
        query,
        k,
        ef=0,
        granularities=None,
        rrf_k=0,
        timeout_s=None,
    ):
        qtok = set(query.lower().split())
        scored = []
        for nid, text, props in self.graphs[graph]:
            overlap = len(qtok & set(text.lower().split()))
            scored.append((overlap, nid, text, props))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            SearchHit(node_id=nid, distance=1.0 / (1 + ov), props=props, text=props.get("text", text))
            for ov, nid, text, props in scored[:k]
        ]

    def get_json(self, key, *, cf=1):
        if key.startswith(b"gnode:"):
            return {"category": "raw_text"}
        return {}


class TestRunE2E(unittest.TestCase):
    def test_retrieval_only_full_run(self):
        cfg = HarnessConfig(dataset_dir=_DATA, k=10)
        # No API key -> retrieval-only regardless.
        cfg.deepseek.api_key = ""
        args = Namespace(no_ingest=False, retrieval_only=True, out=None)

        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "report.json")
            args.out = out

            # Inject the fake store via monkeypatch.
            orig = run_mod.MemoryStore
            run_mod.MemoryStore = FakeStore
            try:
                report = run_mod.run(cfg, args)
            finally:
                run_mod.MemoryStore = orig

            self.assertFalse(report["judged"])
            self.assertEqual(report["overall"]["n"], 2)
            # The keyword ranker should surface the gold sessions for both
            # smoke questions ("dog/border collie", "Globex/work").
            self.assertGreaterEqual(report["overall"]["hit_at_k"], 0.5)
            # Artifact written and reloadable.
            with open(out) as f:
                disk = json.load(f)
            self.assertEqual(disk["k"], 10)
            self.assertIn("by_category", disk)

    def test_abstention_run(self):
        # Synthetic dataset: answerable questions whose query keywords match a
        # gold unit (high confidence) interleaved with `_abs` abstention
        # questions whose keywords match nothing (low confidence). Phase 6
        # calibration on the dev split should learn to abstain on the latter.
        questions = []
        for i in range(6):
            sessions = [
                [
                    {"role": "user", "content": f"My favorite color number {i} is teal{i}."},
                    {"role": "assistant", "content": "Got it."},
                ]
            ]
            # Answerable: query overlaps the stored content.
            questions.append({
                "question_id": f"ans_{i:04d}",
                "question_type": "single-session-user",
                "question": f"favorite color number {i} teal{i}",
                "answer": f"teal{i}",
                "question_date": "2023-01-01",
                "haystack_session_ids": ["s0"],
                "answer_session_ids": ["s0"],
                "haystack_dates": ["2023/01/01 (Sun) 09:00"],
                "haystack_sessions": sessions,
            })
            # Abstention: query keywords appear nowhere in the haystack.
            questions.append({
                "question_id": f"abs_{i:04d}_abs",
                "question_type": "single-session-user",
                "question": f"zzqqxx unrelated nonsense token {i}",
                "answer": "No information available.",
                "question_date": "2023-01-01",
                "haystack_session_ids": ["s0"],
                "answer_session_ids": [],
                "haystack_dates": ["2023/01/01 (Sun) 09:00"],
                "haystack_sessions": sessions,
            })

        with tempfile.TemporaryDirectory() as td:
            ds_dir = Path(td) / "questions"
            ds_dir.mkdir()
            for q in questions:
                with open(ds_dir / f"{q['question_id']}.json", "w") as f:
                    json.dump(q, f)

            cfg = HarnessConfig(dataset_dir=str(ds_dir), k=10)
            cfg.deepseek.api_key = ""
            out = str(Path(td) / "report.json")
            args = Namespace(
                no_ingest=False,
                retrieval_only=True,
                out=out,
                abstention=True,
                abstention_dev_fraction=0.5,
                abstention_min_retention=0.95,
            )

            orig = run_mod.MemoryStore
            run_mod.MemoryStore = FakeStore
            try:
                report = run_mod.run(cfg, args)
            finally:
                run_mod.MemoryStore = orig

            self.assertIn("abstention", report)
            ab = report["overall"]["abstention"]
            # Metrics are reported on the held-out test split only (the policy
            # is calibrated on the disjoint dev split). With 6 abstention + 6
            # answerable and dev_fraction=0.5, each split is stratified 3/3.
            self.assertEqual(ab["n_abstention"], 3)
            self.assertEqual(ab["n_answerable"], 3)
            block = report["abstention"]
            self.assertEqual(block["n_test"], 6)
            self.assertEqual(block["n_dev"] + block["n_test"], 12)
            # The separable synthetic set should let calibration recover the
            # abstention category without dropping answerable questions.
            self.assertGreaterEqual(ab["abstention_accuracy"], 0.99)
            self.assertGreaterEqual(ab["answerable_retention"], 0.99)
            # A non-trivial threshold was learned (not the never-abstain default).
            pol = report["abstention"]["calibration"]["policy"]
            self.assertTrue(pol["tau_top"] is not None or pol["tau_gap"] is not None)


if __name__ == "__main__":
    unittest.main(verbosity=2)

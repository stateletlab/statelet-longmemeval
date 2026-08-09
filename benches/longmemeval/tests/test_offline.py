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
"""Offline tests: segmentation, dataset gold mapping, metrics, judge parsing,
and reporting. No gateway, no DeepSeek — runnable as

    python benches/longmemeval/tests/test_offline.py

or under pytest. Covers the LLM-free retrieval gate end to end against the
bundled smoke dataset.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

# Make `benches` importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benches.longmemeval import deepseek, metrics, reader_judge, report, segment  # noqa: E402
from benches.longmemeval.config import CATEGORIES, MEM0_ACCURACY, HarnessConfig  # noqa: E402
from benches.longmemeval.dataset import load_questions  # noqa: E402
from benches.longmemeval.query import (  # noqa: E402
    QueryResult,
    _computed_evidence_from_response,
    _evidence_plan_for_question,
    _gateway_query_text,
    _json_clip,
)
from benches.longmemeval.dataset import Question  # noqa: E402
from benches.longmemeval.store import SearchHit, session_index_of  # noqa: E402


_DATA = str(_REPO_ROOT / "benches" / "longmemeval" / "data" / "questions")


class TestSegment(unittest.TestCase):
    def test_merges_short_units(self):
        units = segment.segment("Hi. Ok. I really love hiking on the weekends a lot.", min_unit_tokens=8)
        # The two short sentences must not survive as standalone tiny units.
        self.assertTrue(all(len(u.split()) >= 4 for u in units), units)
        self.assertEqual("".join(units).count("hiking"), 1)

    def test_splits_long_units(self):
        long_text = "a, " * 80 + "end."
        units = segment.segment(long_text, max_unit_tokens=20)
        self.assertTrue(all(len(u.split()) <= 20 for u in units), [len(u.split()) for u in units])

    def test_empty(self):
        self.assertEqual(segment.segment("   "), [])

    def test_deterministic(self):
        text = "The user adopted a border collie named Rufus. He is tireless on the trail and loves long hikes."
        self.assertEqual(segment.segment(text), segment.segment(text))


class TestDataset(unittest.TestCase):
    def test_load_smoke(self):
        qs = load_questions(_DATA)
        self.assertEqual(len(qs), 2)
        ids = {q.question_id for q in qs}
        self.assertEqual(ids, {"smoke_0001", "smoke_0002"})

    def test_gold_index_mapping(self):
        qs = {q.question_id: q for q in load_questions(_DATA)}
        # s1 is the gold session id at index 1.
        self.assertEqual(qs["smoke_0001"].gold_session_indices, [1])
        # k2 is the gold session id at index 2.
        self.assertEqual(qs["smoke_0002"].gold_session_indices, [2])

    def test_limit_and_only(self):
        self.assertEqual(len(load_questions(_DATA, limit=1)), 1)
        only = load_questions(_DATA, only_ids=["smoke_0002"])
        self.assertEqual([q.question_id for q in only], ["smoke_0002"])

    def test_start_limit_resume_window(self):
        qs, total = load_questions(_DATA, start=2, limit=1, with_total=True)
        self.assertEqual(total, 2)
        self.assertEqual([q.question_id for q in qs], ["smoke_0002"])


def _hit(session_index, node_id=1, dist=0.1):
    return SearchHit(node_id=node_id, distance=dist, props={"session_index": session_index}, text="x")


class TestMetrics(unittest.TestCase):
    def test_session_index_recovery(self):
        self.assertEqual(session_index_of(_hit(3)), 3)
        # parent_node_id convention: (idx+1)*1_000_000 + n
        h = SearchHit(node_id=2_000_005, distance=0.0, props={"parent_node_id": 2_000_005})
        self.assertEqual(session_index_of(h), 1)
        self.assertIsNone(session_index_of(SearchHit(node_id=5, distance=0.0, props={})))

    def test_hit_and_recall(self):
        q = load_questions(_DATA, only_ids=["smoke_0001"])[0]  # gold index 1
        # gold at rank 2, plus a non-gold; recall should be 1/1 = 1.0.
        qr = QueryResult("smoke_0001", [_hit(0), _hit(1), _hit(2)], 1.0)
        s = metrics.score_retrieval(q, qr, k=10)
        self.assertTrue(s.hit_at_k)
        self.assertEqual(s.first_hit_rank, 2)
        self.assertAlmostEqual(s.recall_at_k, 1.0)
        self.assertAlmostEqual(s.mrr, 0.5)

    def test_miss(self):
        q = load_questions(_DATA, only_ids=["smoke_0001"])[0]  # gold index 1
        qr = QueryResult("smoke_0001", [_hit(0), _hit(2)], 1.0)
        s = metrics.score_retrieval(q, qr, k=10)
        self.assertFalse(s.hit_at_k)
        self.assertEqual(s.recall_at_k, 0.0)
        self.assertEqual(s.mrr, 0.0)

    def test_k_truncation(self):
        q = load_questions(_DATA, only_ids=["smoke_0001"])[0]  # gold index 1
        # gold only appears at rank 3 but k=2 -> miss.
        qr = QueryResult("smoke_0001", [_hit(0), _hit(2), _hit(1)], 1.0)
        s = metrics.score_retrieval(q, qr, k=2)
        self.assertFalse(s.hit_at_k)

    def test_accumulator_per_category(self):
        acc = metrics.MetricsAccumulator()
        for qid in ("smoke_0001", "smoke_0002"):
            q = load_questions(_DATA, only_ids=[qid])[0]
            qr = QueryResult(qid, [_hit(i) for i in q.gold_session_indices], 1.0)
            acc.add_retrieval(metrics.score_retrieval(q, qr, k=10))
        self.assertEqual(acc.overall.n, 2)
        self.assertAlmostEqual(acc.overall.hit_at_k, 1.0)
        self.assertIn("single-session-user", acc.by_category)
        self.assertIn("knowledge-update", acc.by_category)
        # No judgements added -> accuracy stays None (retrieval-only).
        self.assertIsNone(acc.overall.accuracy)


class TestJudgeParsing(unittest.TestCase):
    def test_reader_judge_default_batch_and_concurrency(self):
        with patch.dict("os.environ", {}, clear=True):
            args = reader_judge.parse_args(["detail.log"])
        self.assertEqual(args.batch_size, 25)
        self.assertEqual(args.concurrency, 5)
        self.assertEqual(args.max_record_chars, 2600)

    def test_reader_judge_select_records_by_qid(self):
        records = [
            reader_judge.Record(i, 3, f"q{i}", "multi-session", "Q", "", "A", (), ())
            for i in range(1, 4)
        ]
        selected = reader_judge.select_records(records, start_index=1, limit=0, only_ids=["q2"])
        self.assertEqual([record.qid for record in selected], ["q2"])

    def test_plain_json(self):
        j = deepseek._parse_judgement('{"correct": true, "explanation": "matches"}')
        self.assertTrue(j.correct)
        self.assertEqual(j.label, "CORRECT")

    def test_fenced_json(self):
        j = deepseek._parse_judgement('```json\n{"correct": false, "explanation": "wrong"}\n```')
        self.assertFalse(j.correct)

    def test_embedded_json(self):
        j = deepseek._parse_judgement('Here is my verdict: {"correct": true} done.')
        self.assertTrue(j.correct)

    def test_freetext_fallback(self):
        self.assertTrue(deepseek._parse_judgement("The answer is correct.").correct)
        self.assertFalse(deepseek._parse_judgement("This is incorrect.").correct)

    def test_reader_judge_parses_computed_memory(self):
        detail = "\n".join(
            [
                "2026-06-19 00:00:00 [1/1] q1 [multi-session] hit@10=Y recall=1.00",
                "      Q: How many appointments?",
                "      D: 2026-06-19",
                "      A: 2",
                "      @1  primary_answer_result | {\"computed_answer\":2}",
                "      @2  fact_result | {\"text\":\"March 3 appointment\"}",
                "      *#1  sess=   1 dist=0.100 f1=0.50 | Memory pack one",
                "       #2  sess=   2 dist=0.200 f1=0.00 | Memory pack two",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "detail.log"
            path.write_text(detail, encoding="utf-8")
            records = reader_judge.parse_detail_log(path, top_k=1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].question_date, "2026-06-19")
            self.assertEqual(len(records[0].computed_memory), 2)
            self.assertEqual(records[0].computed_memory[0].source, "primary_answer_result")
            self.assertEqual(len(records[0].hits), 1)
            out_dir = Path(td) / "reader"
            out_dir.mkdir()
            chunk = reader_judge.write_chunk(records, out_dir)
            text = chunk.read_text(encoding="utf-8")
            self.assertIn("Computed Memory:", text)
            self.assertIn("Question Date: 2026-06-19", text)
            self.assertIn("Top-k Memory:", text)
            self.assertIn("@1 primary_answer_result", text)
            self.assertNotIn("A: 2", text)
            self.assertNotIn("*#1", text)

    def test_reader_judge_prioritizes_direct_memory_over_raw_json(self):
        raw = (
            '{"type":"raw_text_recovery","computed_answer":'
            '{"kind":"raw_text_recovery","value":"game_of_throne is_a tv show; '
            'amazon_prime is_a streaming_service"},"formatted_answer":'
            '{"value":"game_of_throne is_a tv show; amazon_prime is_a streaming_service"}}'
        )
        direct = (
            '{"type":"direct_answer_span","answer_role":"supporting_context",'
            '"date":"2023/05/28","text":"User waited over a year for their '
            'asylum application to be approved"}'
        )
        record = reader_judge.Record(
            1,
            1,
            "q1",
            "single-session-user",
            "How long did I wait for the decision on my asylum application?",
            "2023/05/30",
            "over a year",
            (
                reader_judge.ComputedMemory(1, "answer_result", raw),
                reader_judge.ComputedMemory(2, "answer_result", direct),
            ),
            (
                reader_judge.Hit(
                    1,
                    True,
                    "Memory pack: early unrelated housing advice. "
                    "[5|session=43] speaking of waiting, Over a year of uncertainty "
                    "about my asylum application was really tough.",
                ),
            ),
        )

        text = "\n".join(reader_judge._format_record(record, max_record_chars=1400))
        self.assertIn("over a year", text.lower())
        self.assertIn("asylum application", text.lower())
        self.assertIn("@2 answer_result", text)
        if "@1 answer_result" in text:
            self.assertLess(text.index("@2 answer_result"), text.index("@1 answer_result"))


class TestEvidencePacking(unittest.TestCase):
    def _question(self, question: str, question_type: str = "multi-session") -> Question:
        return Question(
            question_id="q",
            question=question,
            answer="2",
            question_type=question_type,
            question_date="2023/03/09 (Thu) 15:47",
            sessions=[],
            session_dates=[],
            haystack_session_ids=[],
            answer_session_ids=[],
        )

    def test_json_clip_keeps_compact_memory_parseable(self):
        payload = {
            "operand_table": {
                "included_count": 24,
                "rows": [
                    {
                        "value": str(idx),
                        "source": "answer_result",
                        "memory": "March appointment with Dr Smith " * 20,
                    }
                    for idx in range(20)
                ],
            }
        }
        compact = _json_clip(payload, 420)
        parsed = json.loads(compact)
        self.assertIsInstance(parsed, dict)
        self.assertLessEqual(len(compact), 420)

    def test_complex_question_gets_larger_budget_and_time_hints(self):
        cfg = HarnessConfig()
        q = self._question("How many days ago was last Saturday and how much did I save?")
        plan = _evidence_plan_for_question(cfg, q)
        self.assertIn("multi_operand_numeric", plan.labels)
        self.assertIn("temporal", plan.labels)
        self.assertGreaterEqual(plan.answer_results_limit, cfg.answer_complex_results_limit)
        self.assertGreaterEqual(plan.fact_k, cfg.answer_complex_fact_k)
        self.assertTrue(any("last saturday = 2023/03/04" in h.lower() for h in plan.hints))

    def test_gateway_query_text_carries_question_date(self):
        q = self._question("How many days ago was last Saturday?")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_gateway_query_text(q), "How many days ago was last Saturday?")
        with patch.dict("os.environ", {"LME_DATE_PREFIX": "1"}):
            self.assertEqual(
                _gateway_query_text(q),
                "[Context date: 2023-03-09] How many days ago was last Saturday?",
            )

    def test_direct_answer_span_precedes_untrusted_primary(self):
        cfg = HarnessConfig()
        q = self._question("How many doctor's appointments did I go to in March?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={
                "kind": "raw_text_recovery",
                "status": "ambiguous",
                "confidence": 0.3,
                "computed_answer": {"value": "wrong unrelated recovery"},
            },
            answer_results=[
                {
                    "kind": "raw_text_recovery",
                    "status": "ambiguous",
                    "confidence": 0.3,
                    "computed_answer": {"value": "wrong unrelated recovery"},
                },
                {
                    "type": "direct_answer_span",
                    "answer_role": "supporting_context",
                    "text": "March 3 appointment and March 20 appointment.",
                    "value": "2",
                },
            ],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        sources = [item.source for item in evidence]
        self.assertGreaterEqual(len(evidence), 2)
        self.assertEqual(evidence[0].source, "memory_plan")
        self.assertIn("answer_result", sources)
        answer_result = next(item for item in evidence if item.source == "answer_result")
        self.assertIn("March 3 appointment", answer_result.text)
        self.assertTrue(all(item.source != "primary_answer_result" for item in evidence))

    def test_irrelevant_answer_result_is_filtered(self):
        cfg = HarnessConfig()
        q = self._question("What percentage of leadership positions do women hold in my company?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[
                {
                    "type": "direct_answer_span",
                    "answer_role": "context_evidence",
                    "text": "Abstract Expressionism includes Jackson Pollock and Mark Rothko.",
                    "value": "unrelated",
                },
                {
                    "type": "direct_answer_span",
                    "answer_role": "context_evidence",
                    "text": "Women occupy 20 of the leadership positions in my company.",
                    "value": "20%",
                },
            ],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        answer_text = "\n".join(item.text for item in evidence if item.source == "answer_result")
        self.assertIn("leadership positions", answer_text)
        self.assertNotIn("Abstract Expressionism", answer_text)

    def test_value_only_answer_result_match_is_filtered(self):
        cfg = HarnessConfig()
        q = self._question("Which rewards program did I sign up for?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[
                {
                    "type": "direct_answer_span",
                    "answer_role": "supporting_context",
                    "text": "User is tracking TV shows across Netflix and Prime Video.",
                    "value": "user signed up for the ShopRite rewards program",
                },
                {
                    "type": "direct_answer_span",
                    "answer_role": "supporting_context",
                    "text": "User signed up for the ShopRite rewards program today.",
                },
            ],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        answer_text = "\n".join(item.text for item in evidence if item.source == "answer_result")
        self.assertIn("ShopRite rewards program today", answer_text)
        self.assertNotIn("tracking TV shows", answer_text)

    def test_irrelevant_bundle_direct_spans_are_filtered(self):
        cfg = HarnessConfig()
        q = self._question("What percentage of leadership positions do women hold in my company?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "direct_answer_spans": [
                    {"text": "Abstract Expressionism includes Jackson Pollock and Mark Rothko."},
                    {"text": "Women occupy 20 leadership roles out of 100 total positions."},
                ],
                "timeline": [
                    {"date": "2023/03/01", "text": "Abstract Expressionism museum tickets."},
                    {"date": "2023/03/02", "text": "Leadership roles were reviewed."},
                ],
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        bundle_text = "\n".join(item.text for item in evidence if item.source.startswith("bundle."))
        self.assertIn("leadership", bundle_text)
        self.assertNotIn("Abstract Expressionism", bundle_text)

    def test_operand_table_requires_query_overlap_for_numeric_hits(self):
        cfg = HarnessConfig()
        q = self._question("What percentage of leadership positions do women hold in my company?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[
                SearchHit(node_id=1, distance=0.1, props={}, text="I bought a 32 oz water bottle."),
                SearchHit(
                    node_id=2,
                    distance=0.2,
                    props={},
                    text="Women hold 20 leadership positions out of 100 total leadership positions.",
                ),
            ],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        operand_text = "\n".join(item.text for item in evidence if item.source == "operand_memory")
        self.assertIn("leadership positions", operand_text)
        self.assertNotIn("32 oz", operand_text)

    def test_operand_table_filters_wrong_event_type(self):
        cfg = HarnessConfig()
        q = self._question("How many doctor's appointments did I go to in March?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[
                SearchHit(
                    node_id=1,
                    distance=0.1,
                    props={},
                    text="I started physical therapy sessions twice a week since March 25.",
                ),
                SearchHit(
                    node_id=2,
                    distance=0.2,
                    props={},
                    text="I had a follow-up appointment with orthopedic surgeon Dr. Thompson on March 20.",
                ),
                SearchHit(
                    node_id=3,
                    distance=0.3,
                    props={},
                    text="I saw my primary care physician Dr. Smith on March 3 for bronchitis.",
                ),
            ],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        operand_text = "\n".join(item.text for item in evidence if item.source == "operand_memory")
        self.assertIn("March 20", operand_text)
        self.assertIn("March 3", operand_text)
        self.assertNotIn("physical therapy", operand_text)

    def test_memory_card_precedes_operand_memory(self):
        cfg = HarnessConfig()
        q = self._question("What percentage of leadership positions do women hold in my company?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[
                SearchHit(
                    node_id=2,
                    distance=0.2,
                    props={},
                    text="Women hold 20 leadership positions out of 100 total leadership positions in 2023.",
                ),
            ],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        sources = [item.source for item in evidence]
        self.assertEqual(sources[:3], ["memory_plan", "memory_card", "operand_memory"])
        memory_card = next(item.text for item in evidence if item.source == "memory_card")
        self.assertIn('"answer_type":"percentage"', memory_card)
        self.assertIn('"operands"', memory_card)
        self.assertIn('"value":"20"', memory_card)
        self.assertNotIn('"value":"2023"', memory_card)

    def test_operand_table_ignores_bundle_metadata_numbers(self):
        cfg = HarnessConfig()
        q = self._question("How much did I pay for the writing workshop?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "direct_answer_spans": [
                    {
                        "text": "It was a two-day writing workshop, and I paid $200 to attend.",
                        "distance": 0.07148677855730057,
                        "node_id": 9223596512615923725,
                        "source_span_ids": [9223596512615923725],
                    }
                ]
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        operand_text = "\n".join(item.text for item in evidence if item.source == "operand_memory")
        self.assertIn("$200", operand_text)
        self.assertNotIn("0.071486", operand_text)
        self.assertNotIn("9223596512615923725", operand_text)

    def test_preference_question_prioritizes_preference_profile(self):
        cfg = HarnessConfig()
        q = self._question("What should I serve for dinner?", "single-session-preference")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "preference_profile": {
                    "contexts": [
                        "User prefers dinner suggestions that use homegrown cherry tomatoes",
                        "User likes Voluspa scented candles",
                    ],
                    "entities": ["dinner", "homegrown ingredients", "candle wax"],
                },
                "event_clusters": [{"top_texts": ["generic garden"]}],
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        sources = [e.source for e in evidence]
        self.assertIn("bundle.preference_profile", sources)
        if "bundle.event_clusters" in sources:
            self.assertLess(sources.index("bundle.preference_profile"), sources.index("bundle.event_clusters"))
        profile_text = next(e.text for e in evidence if e.source == "bundle.preference_profile")
        self.assertIn("homegrown cherry tomatoes", profile_text)
        self.assertNotIn("Voluspa", profile_text)

    def test_preference_profile_filters_to_question_overlap(self):
        cfg = HarnessConfig()
        q = self._question(
            "Can you suggest some accessories that would complement my current photography setup?",
            "single-session-preference",
        )
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "preference_profile": {
                    "contexts": [
                        "User asked for tourist destinations that prioritize sustainable tourism",
                        "User prefers a compatible tripod for their current photography setup",
                        "User is thinking of getting a personalized baby blanket as a gift",
                    ],
                    "entities": ["tripod", "tourism", "blanket"],
                    "domains": ["photography_setup", "general"],
                }
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        profile_text = "\n".join(item.text for item in evidence if item.source == "bundle.preference_profile")
        self.assertIn("tripod", profile_text)
        self.assertNotIn("tourist destinations", profile_text)
        self.assertNotIn("baby blanket", profile_text)

    def test_non_preference_question_does_not_emit_preference_profile(self):
        cfg = HarnessConfig()
        q = self._question("How many doctor's appointments did I go to in March?", "multi-session")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "preference_profile": {
                    "contexts": [
                        "User prefers physical therapy sessions twice a week in March",
                    ],
                    "entities": ["physical therapy"],
                },
                "event_clusters": [{"top_texts": ["March doctor appointment"]}],
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        sources = [e.source for e in evidence]
        self.assertNotIn("bundle.preference_profile", sources)
        self.assertNotIn("preference_memory", sources)

    def test_fact_and_chunk_results_are_not_starved_by_metadata_bundle(self):
        cfg = HarnessConfig()
        cfg.answer_evidence_total_max_chars = 700
        cfg.answer_complex_evidence_total_max_chars = 700
        q = self._question("How many dollars did I save?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "direct_answer_spans": [{"text": "saved $12"}],
                "packed_context": {"text": "metadata " * 200},
                "answerability": {"debug": "coverage " * 200},
                "coverage": {"debug": "coverage " * 200},
            },
            fact_hits=[
                SearchHit(
                    node_id=1,
                    distance=0.1,
                    props={"text": "The raw fact says I saved $12 on the ticket."},
                    text="The raw fact says I saved $12 on the ticket.",
                )
            ],
            chunk_hits=[
                SearchHit(
                    node_id=2,
                    distance=0.2,
                    props={"text": "The taxi cost was $22 and the train was $10."},
                    text="The taxi cost was $22 and the train was $10.",
                )
            ],
            hits=[
                SearchHit(
                    node_id=3,
                    distance=0.3,
                    props={"text": "I saved $12 by taking the train."},
                    text="I saved $12 by taking the train.",
                )
            ],
        )
        sources = [e.source for e in _computed_evidence_from_response(resp, cfg, q, plan)]
        self.assertIn("fact_result", sources)
        self.assertIn("chunk_result", sources)
        if "bundle.answerability" in sources:
            self.assertLess(sources.index("fact_result"), sources.index("bundle.answerability"))

    def test_top_memory_highlight_surfaces_relevant_raw_hit_before_metadata(self):
        cfg = HarnessConfig()
        cfg.answer_evidence_total_max_chars = 900
        cfg.answer_complex_evidence_total_max_chars = 900
        q = self._question("What is the total amount I spent on gifts for my coworker and brother?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={
                "direct_answer_spans": [{"text": "brother gift $100"}],
                "answerability": {"debug": "metadata " * 200},
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[
                SearchHit(
                    node_id=7,
                    distance=0.2,
                    props={},
                    text=(
                        "Memory pack: [1|session=1] user: I spent $100 on my brother's gift. "
                        "[2|session=2] user: I also spent $100 on a coworker gift."
                    ),
                )
            ],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        sources = [item.source for item in evidence]
        self.assertIn("top_memory_highlight", sources)
        self.assertIn("coworker gift", next(item.text for item in evidence if item.source == "top_memory_highlight"))
        if "bundle.answerability" in sources:
            self.assertLess(sources.index("top_memory_highlight"), sources.index("bundle.answerability"))

    def test_assistant_memory_preserves_ordered_list_context(self):
        cfg = HarnessConfig()
        cfg.answer_complex_evidence_total_max_chars = 5000
        q = self._question(
            "What was the 7th work from home job for seniors you recommended?",
            "single-session-assistant",
        )
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        long_list = (
            "assistant: Here are work from home jobs for seniors: "
            "1. Virtual assistant. 2. Bookkeeper. 3. Online tutor. "
            "4. Customer support representative. 5. Proofreader. "
            "6. Data entry clerk. 7. Transcriptionist. 8. Pet sitter coordinator."
        )
        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[SearchHit(node_id=9, distance=0.1, props={}, text=long_list)],
        )
        memory = _computed_evidence_from_response(resp, cfg, q, plan)
        assistant_text = "\n".join(item.text for item in memory if item.source == "assistant_memory")
        self.assertIn("7. Transcriptionist", assistant_text)
        self.assertIn("assistant_memory", [item.source for item in memory])

    def test_assistant_memory_prefers_top_hits_over_weak_bundle(self):
        cfg = HarnessConfig()
        q = self._question(
            "Can you remind me what was the 7th job in the list you provided?",
            "single-session-assistant",
        )
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[
                {
                    "type": "direct_answer_span",
                    "answer_role": "supporting_context",
                    "text": "User asked for work-from-home job ideas for seniors",
                },
                {
                    "type": "direct_answer_span",
                    "answer_role": "timeline_context",
                    "text": "Unrelated smoothie calorie estimate.",
                },
            ],
            answer_bundle={
                "packed_context": [
                    "Weak summary: user asked for job ideas.",
                    "Unrelated Zumba practice context.",
                ]
            },
            fact_hits=[],
            chunk_hits=[],
            hits=[
                SearchHit(
                    node_id=7,
                    distance=0.01,
                    props={},
                    text=(
                        "Memory pack: user asked for work from home jobs. "
                        "1. Greeter 2. Tutor 3. Bookkeeper 4. Scheduler "
                        "5. Support rep 6. Proofreader 7. Transcriptionist"
                    ),
                )
            ],
        )
        memory = _computed_evidence_from_response(resp, cfg, q, plan)
        packed = next(item for item in memory if item.source == "assistant_memory")
        self.assertIn('"source":"hit"', packed.text)
        self.assertIn("7. Transcriptionist", packed.text)
        self.assertLess(packed.text.index('"source":"hit"'), packed.text.index("Weak summary"))

    def test_temporal_memory_emits_resolved_target_matches(self):
        cfg = HarnessConfig()
        q = self._question("What did I buy last Saturday?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[SearchHit(node_id=10, distance=0.1, props={}, text="[2023/03/04] user: I bought a smoker.")],
        )
        memory = _computed_evidence_from_response(resp, cfg, q, plan)
        temporal_text = "\n".join(item.text for item in memory if item.source == "temporal_memory")
        self.assertIn("last saturday", temporal_text.lower())
        self.assertIn("2023/03/04", temporal_text)
        self.assertIn("bought a smoker", temporal_text)

    def test_operand_table_and_timeline_highlight_are_emitted(self):
        cfg = HarnessConfig()
        q = self._question("How many days after March 3 did I spend $100?")
        plan = _evidence_plan_for_question(cfg, q)

        @dataclass
        class Resp:
            primary_answer_result: dict
            answer_results: list
            answer_bundle: dict
            fact_hits: list
            chunk_hits: list
            hits: list

        resp = Resp(
            primary_answer_result={},
            answer_results=[],
            answer_bundle={},
            fact_hits=[],
            chunk_hits=[],
            hits=[
                SearchHit(
                    node_id=8,
                    distance=0.2,
                    props={},
                    text="[2023/03/03] user: I spent $100 on a coworker gift after March 3.",
                )
            ],
        )
        evidence = _computed_evidence_from_response(resp, cfg, q, plan)
        by_source = {item.source: item.text for item in evidence}
        self.assertIn("operand_memory", by_source)
        self.assertIn("$100", by_source["operand_memory"])
        self.assertNotIn('"value":"2023"', by_source["operand_memory"])
        self.assertIn("timeline_memory", by_source)
        self.assertIn("2023/03/03", by_source["timeline_memory"])


class TestReport(unittest.TestCase):
    def test_render_and_build(self):
        cfg = HarnessConfig(dataset_dir=_DATA, k=10)
        acc = metrics.MetricsAccumulator()
        for qid in ("smoke_0001", "smoke_0002"):
            q = load_questions(_DATA, only_ids=[qid])[0]
            qr = QueryResult(qid, [_hit(i) for i in q.gold_session_indices], 1.0)
            acc.add_retrieval(metrics.score_retrieval(q, qr, k=10))
        table = report.render_table(acc, k=10, judged=False)
        self.assertIn("OVERALL", table)
        self.assertIn("hit@10", table)
        self.assertIn("retrieval-only", table)
        rep = report.build_report(cfg, acc, judged=False, elapsed_s=1.0)
        self.assertEqual(rep["k"], 10)
        self.assertFalse(rep["judged"])
        self.assertEqual(rep["mem0_baseline"]["overall"], 94.4)
        # API key never leaks into the artifact.
        self.assertEqual(rep["config"]["deepseek"]["api_key"], "")


class TestConfig(unittest.TestCase):
    def test_categories_and_mem0(self):
        self.assertEqual(len(CATEGORIES), 6)
        self.assertIn("temporal-reasoning", CATEGORIES)
        self.assertEqual(MEM0_ACCURACY["overall"], 94.4)
        self.assertEqual(MEM0_ACCURACY["temporal-reasoning"], 76.7)
        self.assertEqual(MEM0_ACCURACY["multi-session"], 88.0)

    def test_detail_log_is_untruncated_by_default(self):
        self.assertEqual(HarnessConfig().detail_max_chars, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

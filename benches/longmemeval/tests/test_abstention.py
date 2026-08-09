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
"""Phase 6 offline tests — calibrated abstention. No gateway, no LLM.

Runnable as

    python benches/longmemeval/tests/test_abstention.py

or under pytest. Exercises the confidence signals, the policy, dev-split
calibration (no hardcoded threshold), the answerable-retention guardrail, and
the metrics/report wiring.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benches.longmemeval import metrics, report  # noqa: E402
from benches.longmemeval.abstention import (  # noqa: E402
    DISABLED,
    AbstentionPolicy,
    Confidence,
    _candidate_thresholds,
    calibrate,
    evaluate_policy,
    is_abstention,
    similarity_of,
    split_dev_test,
)
from benches.longmemeval.dataset import Question  # noqa: E402
from benches.longmemeval.store import SearchHit  # noqa: E402


def _hit(dist: float) -> SearchHit:
    return SearchHit(node_id=1, distance=dist, props={}, text="x")


def _q(qid: str, answer: str = "something", qtype: str = "single-session-user") -> Question:
    return Question(
        question_id=qid,
        question="q?",
        answer=answer,
        question_type=qtype,
        question_date="2023-01-01",
        sessions=[],
        session_dates=[],
        haystack_session_ids=[],
        answer_session_ids=[],
    )


class TestIsAbstention(unittest.TestCase):
    def test_abs_suffix(self):
        self.assertTrue(is_abstention(_q("foo_abs")))
        self.assertTrue(is_abstention(_q("bar_abstention")))
        self.assertFalse(is_abstention(_q("baz")))

    def test_no_info_answer(self):
        self.assertTrue(is_abstention(_q("x", answer="No information available.")))
        self.assertTrue(is_abstention(_q("x", answer="This is unanswerable")))
        self.assertFalse(is_abstention(_q("x", answer="A border collie")))

    def test_empty_answer_is_not_abstention(self):
        # Empty answer alone is ambiguous; only the _abs suffix / explicit
        # phrasing flags abstention.
        self.assertFalse(is_abstention(_q("x", answer="")))


class TestSimilarity(unittest.TestCase):
    def test_monotone_decreasing(self):
        self.assertAlmostEqual(similarity_of(_hit(0.0)), 1.0)
        self.assertGreater(similarity_of(_hit(0.1)), similarity_of(_hit(0.5)))
        self.assertGreater(similarity_of(_hit(0.5)), similarity_of(_hit(5.0)))

    def test_bounded(self):
        for d in (0.0, 0.5, 10.0, 1e6):
            s = similarity_of(_hit(d))
            self.assertTrue(0.0 < s <= 1.0, s)

    def test_negative_and_nan(self):
        self.assertAlmostEqual(similarity_of(_hit(-1.0)), 1.0)  # clamped at 0
        self.assertEqual(similarity_of(_hit(float("nan"))), 0.0)


class TestConfidence(unittest.TestCase):
    def test_empty(self):
        c = Confidence.from_hits([])
        self.assertEqual((c.top_score, c.score_gap, c.n_hits), (0.0, 0.0, 0))

    def test_single_hit_gap_equals_top(self):
        c = Confidence.from_hits([_hit(0.0)])
        self.assertAlmostEqual(c.top_score, 1.0)
        self.assertAlmostEqual(c.score_gap, 1.0)

    def test_gap(self):
        # closer rank-1 (0.0 -> sim 1.0), farther rank-2 (1.0 -> sim 0.5).
        c = Confidence.from_hits([_hit(0.0), _hit(1.0)])
        self.assertAlmostEqual(c.top_score, 1.0)
        self.assertAlmostEqual(c.score_gap, 0.5)

    def test_sorts_defensively(self):
        # Out-of-order distances still yield rank-1 == best similarity.
        c = Confidence.from_hits([_hit(2.0), _hit(0.0)])
        self.assertAlmostEqual(c.top_score, 1.0)


class TestPolicy(unittest.TestCase):
    def test_default_never_abstains(self):
        p = AbstentionPolicy()
        self.assertFalse(p.should_abstain(Confidence(0.0, 0.0, 0)))
        self.assertEqual(p.to_dict(), {"tau_top": None, "tau_gap": None})

    def test_top_threshold(self):
        p = AbstentionPolicy(tau_top=0.5)
        self.assertTrue(p.should_abstain(Confidence(0.4, 1.0, 5)))
        self.assertFalse(p.should_abstain(Confidence(0.6, 0.0, 5)))

    def test_gap_threshold(self):
        p = AbstentionPolicy(tau_gap=0.2)
        self.assertTrue(p.should_abstain(Confidence(0.9, 0.1, 5)))
        self.assertFalse(p.should_abstain(Confidence(0.9, 0.3, 5)))

    def test_or_semantics(self):
        p = AbstentionPolicy(tau_top=0.5, tau_gap=0.2)
        # fails top -> abstain even though gap is fine
        self.assertTrue(p.should_abstain(Confidence(0.4, 0.9, 5)))
        # fails gap -> abstain even though top is fine
        self.assertTrue(p.should_abstain(Confidence(0.9, 0.1, 5)))
        # passes both
        self.assertFalse(p.should_abstain(Confidence(0.9, 0.9, 5)))


class TestCandidates(unittest.TestCase):
    def test_includes_disabled_and_extremes(self):
        cands = _candidate_thresholds([0.2, 0.8])
        self.assertIn(DISABLED, cands)
        # a midpoint exists between the two values
        self.assertTrue(any(abs(c - 0.5) < 1e-9 for c in cands))
        # never-fires and always-fires bounds present
        self.assertTrue(any(c < 0.2 for c in cands if c != DISABLED))
        self.assertTrue(any(c > 0.8 for c in cands))

    def test_empty(self):
        self.assertEqual(_candidate_thresholds([]), [DISABLED])


class TestCalibrate(unittest.TestCase):
    def _separable(self):
        # Answerable Qs have high top score; abstention Qs have low top score.
        labelled = []
        for _ in range(10):
            labelled.append((False, Confidence(0.9, 0.4, 10)))  # answerable
        for _ in range(10):
            labelled.append((True, Confidence(0.2, 0.05, 10)))  # abstention
        return labelled

    def test_perfectly_separable(self):
        res = calibrate(self._separable())
        self.assertGreater(res.stats.abstention_accuracy, 0.99)
        self.assertGreater(res.stats.answerable_retention, 0.99)
        self.assertAlmostEqual(res.objective, 1.0, places=6)
        # A real threshold was learned, not the never-abstain default
        # (at least one of the two signals is enabled).
        self.assertTrue(
            res.policy.tau_top != DISABLED or res.policy.tau_gap != DISABLED
        )

    def test_no_abstention_returns_never_policy(self):
        labelled = [(False, Confidence(0.5, 0.1, 5)) for _ in range(5)]
        res = calibrate(labelled)
        self.assertEqual(res.policy.tau_top, DISABLED)
        self.assertEqual(res.policy.tau_gap, DISABLED)

    def test_retention_guardrail(self):
        # Overlapping distributions: the only way to catch abstention is to
        # also wrongly abstain on some answerable Qs. A strict retention
        # guardrail must keep answerable retention high.
        labelled = []
        for s in (0.55, 0.6, 0.65, 0.7, 0.75):
            labelled.append((False, Confidence(s, 0.1, 5)))
        for s in (0.5, 0.55, 0.6):
            labelled.append((True, Confidence(s, 0.1, 5)))
        strict = calibrate(labelled, min_answerable_retention=1.0)
        self.assertGreaterEqual(strict.stats.answerable_retention, 1.0)

    def test_not_hardcoded_adapts_to_scale(self):
        # Only the TOP score discriminates (gap held identical), and the scale
        # is compressed near zero -> calibration must pick a tau_top on the
        # observed scale, proving the threshold is data-driven, not hardcoded.
        lo = [(True, Confidence(0.02, 0.01, 5)) for _ in range(8)]
        hi = [(False, Confidence(0.09, 0.01, 5)) for _ in range(8)]
        res = calibrate(lo + hi)
        self.assertNotEqual(res.policy.tau_top, DISABLED)
        self.assertTrue(0.02 < res.policy.tau_top < 0.09, res.policy.tau_top)


class TestSplit(unittest.TestCase):
    def test_stratified_deterministic(self):
        labelled = [(i % 3 == 0, Confidence(0.5, 0.1, 5)) for i in range(30)]
        d1, t1 = split_dev_test(labelled, dev_fraction=0.5, seed=7)
        d2, t2 = split_dev_test(labelled, dev_fraction=0.5, seed=7)
        self.assertEqual(d1, d2)
        self.assertEqual(t1, t2)
        # both splits carry some abstention examples (stratified)
        self.assertTrue(any(a for a, _ in d1))
        self.assertTrue(any(a for a, _ in t1))

    def test_partition_is_complete(self):
        labelled = [(i % 2 == 0, Confidence(float(i), 0.0, 1)) for i in range(20)]
        dev, test = split_dev_test(labelled, dev_fraction=0.3, seed=1)
        self.assertEqual(len(dev) + len(test), len(labelled))


class TestMetricsWiring(unittest.TestCase):
    def test_aggregate_abstention(self):
        acc = metrics.MetricsAccumulator()
        # abstention Q, correctly abstained
        acc.add_abstention("temporal-reasoning", is_abstention=True, decided_abstain=True)
        # abstention Q, wrongly answered
        acc.add_abstention("temporal-reasoning", is_abstention=True, decided_abstain=False)
        # answerable Q, correctly answered
        acc.add_abstention("multi-session", is_abstention=False, decided_abstain=False)
        # answerable Q, wrongly abstained
        acc.add_abstention("multi-session", is_abstention=False, decided_abstain=True)

        self.assertAlmostEqual(acc.overall.abstention_accuracy, 0.5)
        self.assertAlmostEqual(acc.overall.answerable_retention, 0.5)
        self.assertAlmostEqual(
            acc.by_category["temporal-reasoning"].abstention_accuracy, 0.5
        )
        self.assertIsNone(
            acc.by_category["temporal-reasoning"].answerable_retention
        )

    def test_report_includes_abstention_block(self):
        acc = metrics.MetricsAccumulator()
        acc.add_abstention("multi-session", is_abstention=True, decided_abstain=True)
        acc.add_abstention("multi-session", is_abstention=False, decided_abstain=False)
        table = report.render_table(acc, k=10, judged=False)
        self.assertIn("abstention (Phase 6)", table)

        rep = report.build_report(
            _cfg(),
            acc,
            judged=False,
            elapsed_s=1.0,
            abstention={"calibration": {"policy": {"tau_top": 0.5, "tau_gap": None}}},
        )
        self.assertIn("abstention", rep)
        self.assertIn("abstention", rep["overall"])
        self.assertAlmostEqual(rep["overall"]["abstention"]["abstention_accuracy"], 1.0)

    def test_report_omits_abstention_when_unused(self):
        acc = metrics.MetricsAccumulator()
        rep = report.build_report(_cfg(), acc, judged=False, elapsed_s=1.0)
        self.assertNotIn("abstention", rep)
        self.assertNotIn("abstention", rep["overall"])


def _cfg():
    from benches.longmemeval.config import HarnessConfig

    return HarnessConfig()


if __name__ == "__main__":
    unittest.main(verbosity=2)

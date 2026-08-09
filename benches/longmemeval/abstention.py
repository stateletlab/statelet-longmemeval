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
"""Phase 6 — calibrated abstention (LLM-free, pure thresholding).

LongMemEval has an *abstention* category: some questions are unanswerable from
the stored memory (the gold answer is "no information"). A pure top-k retriever
always returns something, so it answers — and fails — every abstention question.
A calibrated confidence threshold recovers this category by *abstaining* when
retrieval confidence is too low.

This module implements the #741 Phase 6 design:

  * Two confidence signals computed from the retrieval result alone (no LLM):
      - `top_score`  — the top fused score (rank-1 confidence).
      - `score_gap`  — the gap between rank 1 and rank 2 (margin / decisiveness).
    The gateway returns a *distance* per hit (smaller == closer); we convert it
    to a bounded similarity in [0, 1] so the thresholds are length/scale stable
    and comparable across questions.

  * A `AbstentionPolicy` abstains iff `top_score < tau_top` OR
    `score_gap < tau_gap`. With either threshold set to `-inf` that signal is
    disabled, so the policy degrades to top-only, gap-only, or never-abstain.

  * The thresholds are **calibrated on a dev split** (`calibrate`), never
    hardcoded: we sweep candidate (tau_top, tau_gap) drawn from the observed
    score distribution and pick the pair that maximises a balanced objective —
    abstention-category accuracy *and* answerable accuracy — so the
    precision/recall tradeoff is tuned, not biased toward abstaining on
    everything.

LongMemEval marks abstention questions with the canonical `_abs` suffix on the
`question_id` (and an empty / "no information" gold answer); `is_abstention`
encodes that convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypeVar

from .dataset import Question
from .query import QueryResult
from .store import SearchHit

# Payload type carried alongside the abstention label through `split_dev_test`.
T = TypeVar("T")

# Sentinel meaning "this signal is disabled" — a threshold of -inf never fires.
DISABLED = float("-inf")

# Canonical answer submitted to the reader/judge when the policy abstains. The
# LongMemEval judge scores this as CORRECT for an abstention question and
# INCORRECT for an answerable one — exactly the precision/recall tradeoff we
# calibrate.
ABSTAIN_ANSWER = "I don't have enough information in memory to answer that."


def is_abstention(q: Question) -> bool:
    """True iff this is a LongMemEval abstention (unanswerable) question.

    LongMemEval flags abstention questions with the canonical `_abs` suffix on
    the question id. We also treat an explicit "no information" style gold
    answer as abstention, so the signal works on datasets that drop the suffix.
    """
    qid = (q.question_id or "").lower()
    if qid.endswith("_abs") or qid.endswith("_abstention"):
        return True
    ans = (q.answer or "").strip().lower()
    if not ans:
        return False
    abstain_phrases = (
        "no information",
        "not mentioned",
        "cannot be answered",
        "can't be answered",
        "no relevant information",
        "not enough information",
        "unanswerable",
    )
    return any(p in ans for p in abstain_phrases)


def similarity_of(hit: SearchHit) -> float:
    """Map a gateway *distance* to a bounded similarity in (0, 1].

    The gateway returns a distance (smaller == more similar). We use the
    monotone, scale-tolerant transform `1 / (1 + max(distance, 0))`:
      * cosine/L2 distance 0 -> similarity 1.0 (identical),
      * large distance -> similarity -> 0.0,
      * monotonically decreasing, so ranking by similarity == ranking by
        ascending distance (the gateway's own order).
    This keeps the calibrated thresholds comparable across questions regardless
    of the underlying distance metric's absolute scale.
    """
    d = hit.distance
    if d is None or math.isnan(d):
        return 0.0
    return 1.0 / (1.0 + max(d, 0.0))


@dataclass
class Confidence:
    """LLM-free confidence signals derived from one retrieval result."""

    top_score: float  # similarity of rank-1 hit (0.0 if no hits).
    score_gap: float  # rank1 - rank2 similarity (top_score if <2 hits).
    n_hits: int

    @classmethod
    def from_hits(cls, hits: Sequence[SearchHit]) -> "Confidence":
        if not hits:
            return cls(top_score=0.0, score_gap=0.0, n_hits=0)
        sims = [similarity_of(h) for h in hits]
        # The gateway returns hits ascending by distance, i.e. descending by
        # similarity; sort defensively so rank-1/rank-2 are unambiguous.
        sims.sort(reverse=True)
        top = sims[0]
        gap = top - sims[1] if len(sims) > 1 else top
        return cls(top_score=top, score_gap=gap, n_hits=len(hits))

    @classmethod
    def from_result(cls, qr: QueryResult) -> "Confidence":
        return cls.from_hits(qr.hits)


@dataclass
class AbstentionPolicy:
    """Abstain iff top_score < tau_top OR score_gap < tau_gap.

    A threshold of `DISABLED` (-inf) turns that signal off. The default policy
    never abstains (both disabled), so callers must `calibrate` before it does
    anything — abstention thresholds are never hardcoded.
    """

    tau_top: float = DISABLED
    tau_gap: float = DISABLED

    def should_abstain(self, conf: Confidence) -> bool:
        if self.tau_top != DISABLED and conf.top_score < self.tau_top:
            return True
        if self.tau_gap != DISABLED and conf.score_gap < self.tau_gap:
            return True
        return False

    def to_dict(self) -> dict:
        def _j(v: float) -> Optional[float]:
            return None if v == DISABLED else v

        return {"tau_top": _j(self.tau_top), "tau_gap": _j(self.tau_gap)}


@dataclass
class CalibrationStats:
    """Outcome of evaluating one policy on a labelled split."""

    n: int = 0
    n_abstention: int = 0
    n_answerable: int = 0
    # Correct decisions: abstain on abstention Qs, answer on answerable Qs.
    abstention_correct: int = 0
    answerable_kept: int = 0  # answerable Qs we (correctly) did NOT abstain on.

    @property
    def abstention_accuracy(self) -> float:
        return self.abstention_correct / self.n_abstention if self.n_abstention else 0.0

    @property
    def answerable_retention(self) -> float:
        """Fraction of answerable questions we did NOT wrongly abstain on."""
        return self.answerable_kept / self.n_answerable if self.n_answerable else 1.0

    @property
    def balanced_score(self) -> float:
        """Balanced objective: harmonic-style mean of the two rates.

        Maximising this tunes the precision/recall tradeoff — it rewards
        catching abstention questions *only* while still answering the
        answerable ones, so it cannot be gamed by abstaining on everything.
        """
        a = self.abstention_accuracy
        r = self.answerable_retention
        if a + r == 0:
            return 0.0
        return 2.0 * a * r / (a + r)


def evaluate_policy(
    policy: AbstentionPolicy,
    labelled: Sequence[Tuple[bool, Confidence]],
) -> CalibrationStats:
    """Score `policy` over (is_abstention, confidence) pairs."""
    st = CalibrationStats()
    for abstain_gold, conf in labelled:
        st.n += 1
        decided_abstain = policy.should_abstain(conf)
        if abstain_gold:
            st.n_abstention += 1
            if decided_abstain:
                st.abstention_correct += 1
        else:
            st.n_answerable += 1
            if not decided_abstain:
                st.answerable_kept += 1
    return st


def _candidate_thresholds(values: Sequence[float]) -> List[float]:
    """Midpoints between sorted unique observed values, plus DISABLED.

    Sweeping midpoints (rather than the raw values) means every distinct
    split of the observed distribution is reachable; DISABLED lets the sweep
    drop the signal entirely when it does not help.
    """
    uniq = sorted({round(v, 6) for v in values})
    cands: List[float] = [DISABLED]
    if not uniq:
        return cands
    cands.append(uniq[0] - 1e-6)  # below everything == never fires.
    for a, b in zip(uniq, uniq[1:]):
        cands.append((a + b) / 2.0)
    cands.append(uniq[-1] + 1e-6)  # above everything == always fires.
    return cands


@dataclass
class CalibrationResult:
    policy: AbstentionPolicy
    stats: CalibrationStats
    n_top_candidates: int = 0
    n_gap_candidates: int = 0
    objective: float = 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.to_dict(),
            "objective": self.objective,
            "dev": {
                "n": self.stats.n,
                "n_abstention": self.stats.n_abstention,
                "n_answerable": self.stats.n_answerable,
                "abstention_accuracy": self.stats.abstention_accuracy,
                "answerable_retention": self.stats.answerable_retention,
                "balanced_score": self.stats.balanced_score,
            },
            "grid": {
                "n_top_candidates": self.n_top_candidates,
                "n_gap_candidates": self.n_gap_candidates,
            },
        }


def calibrate(
    labelled: Sequence[Tuple[bool, Confidence]],
    min_answerable_retention: float = 0.95,
) -> CalibrationResult:
    """Fit (tau_top, tau_gap) on a labelled dev split.

    Grid-searches every candidate threshold pair drawn from the observed score
    distribution and returns the policy maximising `balanced_score`, subject to
    a guardrail: never drop answerable retention below
    `min_answerable_retention` *unless no feasible policy meets it* (then we
    fall back to the best balanced score). This encodes the acceptance bar —
    "no material drop on answerable questions".

    Returns a never-abstain policy if the dev split has no abstention examples
    (nothing to calibrate against).
    """
    confs = [c for _, c in labelled]
    has_abstention = any(a for a, _ in labelled)

    never = AbstentionPolicy()
    if not labelled or not has_abstention:
        return CalibrationResult(policy=never, stats=evaluate_policy(never, labelled))

    top_cands = _candidate_thresholds([c.top_score for c in confs])
    gap_cands = _candidate_thresholds([c.score_gap for c in confs])

    best: Optional[CalibrationResult] = None
    best_feasible: Optional[CalibrationResult] = None

    for tt in top_cands:
        for tg in gap_cands:
            policy = AbstentionPolicy(tau_top=tt, tau_gap=tg)
            stats = evaluate_policy(policy, labelled)
            obj = stats.balanced_score
            cand = CalibrationResult(
                policy=policy,
                stats=stats,
                n_top_candidates=len(top_cands),
                n_gap_candidates=len(gap_cands),
                objective=obj,
            )
            if best is None or _better(cand, best):
                best = cand
            if stats.answerable_retention >= min_answerable_retention:
                if best_feasible is None or _better(cand, best_feasible):
                    best_feasible = cand

    chosen = best_feasible or best
    assert chosen is not None
    return chosen


def _better(a: CalibrationResult, b: CalibrationResult) -> bool:
    """Tie-break: higher objective, then higher abstention accuracy, then
    higher answerable retention, then the *simpler* (fewer enabled) policy."""
    if a.objective != b.objective:
        return a.objective > b.objective
    if a.stats.abstention_accuracy != b.stats.abstention_accuracy:
        return a.stats.abstention_accuracy > b.stats.abstention_accuracy
    if a.stats.answerable_retention != b.stats.answerable_retention:
        return a.stats.answerable_retention > b.stats.answerable_retention
    return _n_enabled(a.policy) < _n_enabled(b.policy)


def _n_enabled(p: AbstentionPolicy) -> int:
    return int(p.tau_top != DISABLED) + int(p.tau_gap != DISABLED)


def split_dev_test(
    labelled: Sequence[Tuple[bool, T]],
    dev_fraction: float = 0.5,
    seed: int = 1337,
) -> Tuple[List[Tuple[bool, T]], List[Tuple[bool, T]]]:
    """Deterministic, stratified dev/test split of labelled items.

    Each item is a `(is_abstention, payload)` pair; only the boolean label is
    used for stratification, so the payload can be a `Confidence` (calibration)
    or any record (e.g. the full question) the caller needs to recover the
    held-out test split. Stratified by the abstention label so both splits
    contain abstention and answerable examples even on small sets; deterministic
    under `seed` for reproducibility (Phase 0 contract).
    """
    import random

    rng = random.Random(seed)
    pos = [x for x in labelled if x[0]]
    neg = [x for x in labelled if not x[0]]
    rng.shuffle(pos)
    rng.shuffle(neg)

    dev: List[Tuple[bool, T]] = []
    test: List[Tuple[bool, T]] = []
    for group in (pos, neg):
        cut = int(round(len(group) * dev_fraction))
        dev.extend(group[:cut])
        test.extend(group[cut:])
    return dev, test

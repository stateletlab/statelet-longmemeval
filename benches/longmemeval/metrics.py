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
"""Stage 3 — hit@k / recall@k (the LLM-FREE retrieval gate).

Each LongMemEval question carries labeled gold-evidence sessions. We map each
retrieved unit back to its haystack session index (stamped at ingest), then:

  * hit@k     = 1 if ANY of the top-k results comes from a gold session.
  * recall@k  = fraction of gold sessions covered by the top-k results.
  * mrr       = reciprocal rank of the first gold hit.

These need NO LLM and NO API key — they are the gate every Phase of #741 is
judged on. Aggregation is per-category and overall.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .dataset import Question
from .query import QueryResult
from .store import session_index_of


# ── Content-level similarity (ported from eval/longmemeval_retrieval_eval.py) ──
# These score retrieved TEXT against the gold answer, independent of the
# session→hit mapping. They predict downstream LLM answer+judge accuracy far
# better than session-level recall: the LLM can only answer if the answer
# content actually landed in the retrieved context.

def _normalize_text(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
}


def _normalize_answer(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    text = re.sub(r"\$\s*(\d)", r"\1 dollars", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(hyp_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(text: str, reference: str) -> float:
    """Token-level F1 between a retrieved unit and the gold answer."""
    return _f1(_normalize_text(text).split(), _normalize_text(reference).split())


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            curr[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(curr[j - 1], prev[j])
        prev = curr
    return prev[-1]


def rouge_l_f1(text: str, reference: str) -> float:
    """ROUGE-L F1 (LCS overlap) between a retrieved unit and the gold answer."""
    hyp = _normalize_answer(text).split()
    ref = _normalize_answer(reference).split()
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_len(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


@dataclass
class RetrievalScore:
    question_id: str
    question_type: str
    k: int
    hit_at_k: bool
    recall_at_k: float
    mrr: float
    n_gold_sessions: int
    n_gold_hit: int
    first_hit_rank: Optional[int]
    # Content-level metrics vs the gold answer (predict answer+judge accuracy).
    answer_span_hit_at_k: bool      # any gold-session hit shares a token w/ answer.
    answer_span_recall_at_k: float  # fraction of gold sessions with such a hit.
    best_token_f1: float            # max over top-k of token-F1(hit, answer).
    avg_token_f1: float             # mean over top-k.
    best_rouge_l: float             # max over top-k of ROUGE-L(hit, answer).


def score_retrieval(q: Question, qr: QueryResult, k: int) -> RetrievalScore:
    gold = set(q.gold_session_indices)
    top = qr.hits[:k]
    answer = q.answer

    gold_hit: set = set()
    span_hit: set = set()
    first_hit_rank: Optional[int] = None
    f1s: List[float] = []
    rouges: List[float] = []
    for rank, hit in enumerate(top, start=1):
        si = session_index_of(hit)
        if si is not None and si in gold:
            gold_hit.add(si)
            if first_hit_rank is None:
                first_hit_rank = rank
        f1 = token_f1(hit.text, answer)
        f1s.append(f1)
        rouges.append(rouge_l_f1(hit.text, answer))
        # answer-span match: a gold-session hit that shares answer content.
        if si is not None and si in gold and f1 > 0.0:
            span_hit.add(si)

    n_gold = len(gold)
    recall = (len(gold_hit) / n_gold) if n_gold > 0 else 0.0
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
    return RetrievalScore(
        question_id=q.question_id,
        question_type=q.question_type,
        k=k,
        hit_at_k=first_hit_rank is not None,
        recall_at_k=recall,
        mrr=mrr,
        n_gold_sessions=n_gold,
        n_gold_hit=len(gold_hit),
        first_hit_rank=first_hit_rank,
        answer_span_hit_at_k=len(span_hit) > 0,
        answer_span_recall_at_k=(len(span_hit) / n_gold) if n_gold > 0 else 0.0,
        best_token_f1=max(f1s) if f1s else 0.0,
        avg_token_f1=(sum(f1s) / len(f1s)) if f1s else 0.0,
        best_rouge_l=max(rouges) if rouges else 0.0,
    )


@dataclass
class Aggregate:
    n: int = 0
    hit_at_k_sum: int = 0
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    # Content-level metrics (vs gold answer).
    span_hit_sum: int = 0
    span_recall_sum: float = 0.0
    best_f1_sum: float = 0.0
    avg_f1_sum: float = 0.0
    best_rouge_l_sum: float = 0.0
    # Accuracy (filled in only when the LLM judge stage ran).
    judged: int = 0
    correct: int = 0
    tokens_sum: int = 0
    # Abstention (Phase 6) — filled in only when an abstention policy ran.
    # Each question is either an abstention (unanswerable) or answerable Q; the
    # policy "decides to abstain" per question. A decision is *correct* when it
    # abstains on an abstention Q or answers an answerable Q.
    abst_n_abstention: int = 0
    abst_n_answerable: int = 0
    abst_correct_abstain: int = 0  # abstention Qs we correctly abstained on.
    abst_kept_answerable: int = 0  # answerable Qs we correctly did NOT abstain on.

    def add_retrieval(self, s: RetrievalScore) -> None:
        self.n += 1
        self.hit_at_k_sum += 1 if s.hit_at_k else 0
        self.recall_sum += s.recall_at_k
        self.mrr_sum += s.mrr
        self.span_hit_sum += 1 if s.answer_span_hit_at_k else 0
        self.span_recall_sum += s.answer_span_recall_at_k
        self.best_f1_sum += s.best_token_f1
        self.avg_f1_sum += s.avg_token_f1
        self.best_rouge_l_sum += s.best_rouge_l

    def add_judgement(self, correct: bool, tokens: int) -> None:
        self.judged += 1
        self.correct += 1 if correct else 0
        self.tokens_sum += tokens

    def add_abstention(self, is_abstention: bool, decided_abstain: bool) -> None:
        if is_abstention:
            self.abst_n_abstention += 1
            if decided_abstain:
                self.abst_correct_abstain += 1
        else:
            self.abst_n_answerable += 1
            if not decided_abstain:
                self.abst_kept_answerable += 1

    @property
    def hit_at_k(self) -> float:
        return self.hit_at_k_sum / self.n if self.n else 0.0

    @property
    def recall_at_k(self) -> float:
        return self.recall_sum / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.mrr_sum / self.n if self.n else 0.0

    @property
    def answer_span_hit_at_k(self) -> float:
        return self.span_hit_sum / self.n if self.n else 0.0

    @property
    def answer_span_recall_at_k(self) -> float:
        return self.span_recall_sum / self.n if self.n else 0.0

    @property
    def best_token_f1(self) -> float:
        return self.best_f1_sum / self.n if self.n else 0.0

    @property
    def avg_token_f1(self) -> float:
        return self.avg_f1_sum / self.n if self.n else 0.0

    @property
    def best_rouge_l(self) -> float:
        return self.best_rouge_l_sum / self.n if self.n else 0.0

    @property
    def accuracy(self) -> Optional[float]:
        return (self.correct / self.judged) if self.judged else None

    @property
    def mean_tokens(self) -> Optional[float]:
        return (self.tokens_sum / self.judged) if self.judged else None

    @property
    def abstention_accuracy(self) -> Optional[float]:
        """Fraction of abstention (unanswerable) Qs we correctly abstained on."""
        if not self.abst_n_abstention:
            return None
        return self.abst_correct_abstain / self.abst_n_abstention

    @property
    def answerable_retention(self) -> Optional[float]:
        """Fraction of answerable Qs we did NOT wrongly abstain on."""
        if not self.abst_n_answerable:
            return None
        return self.abst_kept_answerable / self.abst_n_answerable


class MetricsAccumulator:
    """Per-category + overall accumulation over a run."""

    def __init__(self):
        self.overall = Aggregate()
        self.by_category: Dict[str, Aggregate] = {}

    def _cat(self, category: str) -> Aggregate:
        return self.by_category.setdefault(category, Aggregate())

    def add_retrieval(self, s: RetrievalScore) -> None:
        self.overall.add_retrieval(s)
        self._cat(s.question_type).add_retrieval(s)

    def add_judgement(self, category: str, correct: bool, tokens: int) -> None:
        self.overall.add_judgement(correct, tokens)
        self._cat(category).add_judgement(correct, tokens)

    def add_abstention(
        self, category: str, is_abstention: bool, decided_abstain: bool
    ) -> None:
        self.overall.add_abstention(is_abstention, decided_abstain)
        self._cat(category).add_abstention(is_abstention, decided_abstain)

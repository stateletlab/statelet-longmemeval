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
"""Stage 1a — LongMemEval dataset loader.

Loads the 500-question LongMemEval set: each question carries a multi-session
chat history (the "haystack") and a set of labeled gold-evidence session ids.
The loader is the single source of truth for the gold mapping used by hit@k /
recall@k downstream — it resolves `answer_session_ids` to haystack session
*indices* so the metric never depends on the order results come back in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Question:
    """One LongMemEval question + its haystack."""

    question_id: str
    question: str
    answer: str
    question_type: str
    question_date: str
    # haystack_sessions[i] is a list of {"role","content"} messages.
    sessions: List[List[dict]]
    # Per-session ISO date strings (parallel to sessions); may be empty.
    session_dates: List[str]
    # Session ids parallel to `sessions`.
    haystack_session_ids: List[str]
    # The subset of session ids that contain gold evidence.
    answer_session_ids: List[str]

    @property
    def gold_session_indices(self) -> List[int]:
        """Resolve gold session ids to haystack indices (sorted)."""
        gold = set(self.answer_session_ids)
        out = []
        for i, sid in enumerate(self.haystack_session_ids):
            if sid in gold:
                out.append(i)
        # Fallback for datasets that don't carry per-session ids but do carry
        # `answer_session_ids` as integer indices.
        if not out and self.answer_session_ids:
            for sid in self.answer_session_ids:
                if isinstance(sid, int) and 0 <= sid < len(self.sessions):
                    out.append(sid)
        return sorted(set(out))


def _coerce(d: dict) -> Question:
    sessions = d.get("haystack_sessions", [])
    haystack_ids = d.get("haystack_session_ids", [])
    if not haystack_ids:
        # Synthesize stable ids from index so the gold mapping still works when
        # the dataset only labels answers by index.
        haystack_ids = [f"sess_{i}" for i in range(len(sessions))]
    return Question(
        question_id=str(d["question_id"]),
        question=str(d["question"]),
        answer=str(d.get("answer", "")),
        question_type=str(d["question_type"]),
        question_date=str(d.get("question_date", "")),
        sessions=sessions,
        session_dates=list(d.get("haystack_dates", [])),
        haystack_session_ids=list(haystack_ids),
        answer_session_ids=list(d.get("answer_session_ids", [])),
    )


def load_questions(
    dataset_dir: str,
    limit: int = 0,
    only_ids: Optional[List[str]] = None,
    start: int = 0,
    with_total: bool = False,
):
    """Load LongMemEval questions from `dataset_dir`.

    Supports two on-disk layouts:
      * one JSON file per question (the eval/ harness layout), or
      * a single JSON array file (`longmemeval_*.json`).

    `start`/`limit` slice the sorted list so `--start N` resumes from the same
    deterministic order printed as `[N/total]`. `start` is 1-based; `limit`
    caps the count after the offset.

    With `with_total=True`, returns `(questions, full_total)` where
    `full_total` is the count before the start/limit window.
    """
    root = Path(dataset_dir).expanduser()
    raw: List[dict] = []

    if root.is_dir():
        files = sorted(root.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No question JSON files in {root}")
        for f in files:
            with open(f) as fh:
                obj = json.load(fh)
            if isinstance(obj, list):
                raw.extend(obj)
            else:
                raw.append(obj)
    elif root.is_file():
        with open(root) as fh:
            obj = json.load(fh)
        raw = obj if isinstance(obj, list) else [obj]
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    questions = [_coerce(d) for d in raw]

    if only_ids:
        wanted = set(only_ids)
        questions = [q for q in questions if q.question_id in wanted]

    # Deterministic order for reproducibility.
    questions.sort(key=lambda q: q.question_id)
    full_total = len(questions)

    if start and start > 0:
        questions = questions[start - 1:]
    if limit and limit > 0:
        questions = questions[:limit]

    if with_total:
        return questions, full_total
    return questions


def category_counts(questions: List[Question]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for q in questions:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    return counts

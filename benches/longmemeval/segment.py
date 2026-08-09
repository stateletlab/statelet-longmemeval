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
"""Stage 1b — rule-based sentence segmentation for the injector.

LLM-free (per #741): a lightweight, deterministic segmenter that mirrors the
gateway's server-side `STATELET_SENTENCE_UNITS` rule (~16-48 tok units). We
split a message into sentences, then:
  * merge units shorter than `min_unit_tokens` into the following unit, and
  * split units longer than `max_unit_tokens` at clause boundaries.

Token counting is whitespace-based (an approximation that needs no tokenizer
and stays reproducible across machines).
"""

from __future__ import annotations

import re
from typing import List


# Sentence boundary: terminal punctuation followed by whitespace + capital/quote/digit.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
# Soft clause boundaries for splitting over-long sentences.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def _tok_count(s: str) -> int:
    return len(s.split())


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_long(unit: str, max_tokens: int) -> List[str]:
    if _tok_count(unit) <= max_tokens:
        return [unit]
    clauses = _CLAUSE_SPLIT.split(unit)
    out: List[str] = []
    buf: List[str] = []
    buf_tok = 0
    for c in clauses:
        ct = _tok_count(c)
        if buf and buf_tok + ct > max_tokens:
            out.append(" ".join(buf))
            buf, buf_tok = [], 0
        buf.append(c)
        buf_tok += ct
    if buf:
        out.append(" ".join(buf))
    # Hard word-window fallback for clause-free runs over the limit.
    final: List[str] = []
    for piece in out:
        if _tok_count(piece) <= max_tokens:
            final.append(piece)
            continue
        words = piece.split()
        for i in range(0, len(words), max_tokens):
            final.append(" ".join(words[i : i + max_tokens]))
    return final


def segment(text: str, min_unit_tokens: int = 8, max_unit_tokens: int = 64) -> List[str]:
    """Segment `text` into retrieval units.

    Returns a list of unit strings in source order. Deterministic.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Split over-long sentences first.
    expanded: List[str] = []
    for s in sentences:
        expanded.extend(_split_long(s, max_unit_tokens))

    # Merge short units forward.
    merged: List[str] = []
    carry = ""
    for u in expanded:
        cand = (carry + " " + u).strip() if carry else u
        if _tok_count(cand) < min_unit_tokens:
            carry = cand
        else:
            merged.append(cand)
            carry = ""
    if carry:
        if merged and _tok_count(merged[-1]) + _tok_count(carry) <= max_unit_tokens:
            merged[-1] = (merged[-1] + " " + carry).strip()
        else:
            # Keep a slightly-short tail rather than blow past the max cap.
            merged.append(carry)
    return merged

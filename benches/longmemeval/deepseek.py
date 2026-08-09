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
"""Stages 4 & 5 — DeepSeek answerer + LLM judge.

DeepSeek is the ONLY LLM in the loop. It is purely the reader (answers from
retrieved context) and the judge (compares answer vs gold -> correct/incorrect).
The memory layer (extraction + retrieval) stays LLM-free per #741.

Prompt templates mirror mem0's / LongMemEval's answer + judge prompts so the
accuracy number is comparable to mem0's published ~94.4 (parity).
References: https://mem0.ai/research ,
            https://github.com/mem0ai/memory-benchmarks ,
            https://arxiv.org/pdf/2410.10813
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import List

from .config import DeepSeekConfig


# Approximate token estimate (whitespace-based) so the report can show
# mean-tokens/query without depending on a tokenizer.
def estimate_tokens(*texts: str) -> int:
    return sum(len(t.split()) for t in texts if t)


@dataclass
class Judgement:
    correct: bool
    label: str
    explanation: str


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        import requests  # lazy: not needed for retrieval-only runs.

        self._requests = requests

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _call(self, prompt: str, role: str) -> str:
        model = self.cfg.model_for(role)
        is_reasoner = ("reasoner" in model) or ("v4-pro" in model)
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000 if is_reasoner else 2000,
        }
        if not is_reasoner:
            body["temperature"] = self.cfg.temperature

        timeout = 300 if is_reasoner else self.cfg.timeout_s
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._requests.post(url, headers=headers, json=body, timeout=timeout)
                if resp.status_code == 200:
                    msg = resp.json()["choices"][0]["message"]
                    return (msg.get("content") or "").strip()
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(5 if resp.status_code == 429 else 2)
                    continue
                return f"Error: DeepSeek {resp.status_code}: {resp.text[:200]}"
            except self._requests.Timeout:
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(2)
                    continue
                return ""
            except Exception as e:  # noqa: BLE001
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(2)
                    continue
                return f"Error: {e}"
        return ""

    # ── Stage 4: answer ──
    def answer(
        self,
        question: str,
        question_date: str,
        context: List[str],
    ) -> str:
        ctx = "\n\n".join(f"Memory {i + 1}: {c}" for i, c in enumerate(context)) or "(no memories)"
        date_line = ""
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", question_date or "")
        if m:
            date_line = (
                f"The current date is {m.group(1)}-{m.group(2)}-{m.group(3)}. "
                "Resolve all relative dates ('yesterday', 'last week') against it.\n"
            )
        prompt = (
            "You are a helpful assistant answering a question using ONLY the "
            "memories retrieved from a user's past conversations.\n"
            f"{date_line}\n"
            "Memories:\n"
            f"{ctx}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and directly using only the memories. If the "
            "memories do not contain the answer, say you don't have enough "
            "information. Provide ONLY the final answer."
        )
        return self._call(prompt, role="answer")

    # ── Stage 5: judge ──
    def judge(self, question: str, gold: str, hypothesis: str) -> Judgement:
        prompt = (
            "You are grading whether a model's answer to a question is correct, "
            "given the gold answer. The answer is CORRECT if it conveys the same "
            "factual information as the gold answer, even if phrased differently. "
            "Minor wording, formatting, or extra-but-consistent detail does not "
            "make it wrong. It is INCORRECT if it contradicts the gold answer, "
            "omits the key fact, or says it lacks information when the gold gives "
            "a concrete answer.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Model answer: {hypothesis}\n\n"
            'Respond with a JSON object ONLY: {"correct": true|false, '
            '"explanation": "<one sentence>"}'
        )
        raw = self._call(prompt, role="judge")
        return _parse_judgement(raw)


def _parse_judgement(raw: str) -> Judgement:
    text = (raw or "").strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if isinstance(obj, dict) and "correct" in obj:
        correct = bool(obj.get("correct"))
        return Judgement(
            correct=correct,
            label="CORRECT" if correct else "INCORRECT",
            explanation=str(obj.get("explanation", "")),
        )
    # Fallback heuristic on free text.
    low = text.lower()
    correct = ("true" in low or "correct" in low) and "incorrect" not in low
    return Judgement(
        correct=correct,
        label="CORRECT" if correct else "INCORRECT",
        explanation=text[:200],
    )

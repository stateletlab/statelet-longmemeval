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
"""Stage 6 — Reporter.

Emits the per-category + overall table (hit@k, recall@k, accuracy, mean
tokens/query) plus a side-by-side row vs mem0's published numbers, and writes a
machine-readable JSON artifact for reproducibility.
"""

from __future__ import annotations

import json
from typing import List, Optional

from .config import CATEGORIES, MEM0_ACCURACY, HarnessConfig
from .metrics import Aggregate, MetricsAccumulator


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v * 100:5.1f}" if v is not None else "  -- "


def _fmt_num(v: Optional[float]) -> str:
    return f"{v:6.1f}" if v is not None else "   -- "


def render_table(acc: MetricsAccumulator, k: int, judged: bool) -> str:
    lines: List[str] = []
    header = (
        f"{'category':<28} {'n':>4} {'hit@'+str(k):>7} {'recall@'+str(k):>9} "
        f"{'mrr':>6} {'span@'+str(k):>8} {'f1':>6} {'rougeL':>6} "
        f"{'acc%':>6} {'mem0%':>6} {'tok':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(name: str, agg: Aggregate, mem0: Optional[float]) -> str:
        return (
            f"{name:<28} {agg.n:>4} {_fmt_pct(agg.hit_at_k):>7} "
            f"{_fmt_pct(agg.recall_at_k):>9} {agg.mrr:>6.3f} "
            f"{_fmt_pct(agg.answer_span_recall_at_k):>8} "
            f"{agg.best_token_f1:>6.3f} {agg.best_rouge_l:>6.3f} "
            f"{_fmt_pct(agg.accuracy):>6} {_fmt_num(mem0):>6} "
            f"{_fmt_num(agg.mean_tokens):>7}"
        )

    # Per-category in canonical order, then any unexpected categories.
    seen = set()
    for cat in CATEGORIES:
        if cat in acc.by_category:
            seen.add(cat)
            lines.append(row(cat, acc.by_category[cat], MEM0_ACCURACY.get(cat)))
    for cat in sorted(acc.by_category):
        if cat not in seen:
            lines.append(row(cat, acc.by_category[cat], MEM0_ACCURACY.get(cat)))

    lines.append("-" * len(header))
    lines.append(row("OVERALL", acc.overall, MEM0_ACCURACY.get("overall")))

    if not judged:
        lines.append("")
        lines.append("(accuracy / mem0% / tok blank: retrieval-only run — no DEEPSEEK_API_KEY)")

    # Phase 6 — abstention sub-table (only when an abstention policy ran).
    if acc.overall.abst_n_abstention or acc.overall.abst_n_answerable:
        lines.append("")
        abst_header = (
            f"{'abstention (Phase 6)':<28} {'#abst':>6} {'abst-acc':>9} "
            f"{'#ans':>6} {'ans-keep':>9}"
        )
        lines.append(abst_header)
        lines.append("-" * len(abst_header))

        def abst_row(name: str, agg: Aggregate) -> str:
            return (
                f"{name:<28} {agg.abst_n_abstention:>6} "
                f"{_fmt_pct(agg.abstention_accuracy):>9} "
                f"{agg.abst_n_answerable:>6} "
                f"{_fmt_pct(agg.answerable_retention):>9}"
            )

        for cat in CATEGORIES:
            a = acc.by_category.get(cat)
            if a and (a.abst_n_abstention or a.abst_n_answerable):
                lines.append(abst_row(cat, a))
        lines.append("-" * len(abst_header))
        lines.append(abst_row("OVERALL", acc.overall))
    return "\n".join(lines)


def build_report(
    cfg: HarnessConfig,
    acc: MetricsAccumulator,
    judged: bool,
    elapsed_s: float,
    abstention: Optional[dict] = None,
) -> dict:
    def agg_dict(a: Aggregate) -> dict:
        d = {
            "n": a.n,
            "hit_at_k": a.hit_at_k,
            "recall_at_k": a.recall_at_k,
            "mrr": a.mrr,
            "answer_span_hit_at_k": a.answer_span_hit_at_k,
            "answer_span_recall_at_k": a.answer_span_recall_at_k,
            "best_token_f1": a.best_token_f1,
            "avg_token_f1": a.avg_token_f1,
            "best_rouge_l": a.best_rouge_l,
            "accuracy": a.accuracy,
            "mean_tokens": a.mean_tokens,
        }
        # Phase 6 — only emit abstention fields when a policy actually ran.
        if a.abst_n_abstention or a.abst_n_answerable:
            d["abstention"] = {
                "n_abstention": a.abst_n_abstention,
                "n_answerable": a.abst_n_answerable,
                "abstention_accuracy": a.abstention_accuracy,
                "answerable_retention": a.answerable_retention,
            }
        return d

    report = {
        "config": cfg.to_dict(),
        "k": cfg.k,
        "judged": judged,
        "elapsed_s": elapsed_s,
        "mem0_baseline": MEM0_ACCURACY,
        "overall": agg_dict(acc.overall),
        "by_category": {c: agg_dict(a) for c, a in acc.by_category.items()},
    }
    if abstention is not None:
        report["abstention"] = abstention
    return report


def write_report(path: str, report: dict) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

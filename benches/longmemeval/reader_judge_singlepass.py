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
"""SINGLE-PASS reader/judge pass over a LongMemEval detail.log (Claude Code CLI).

WARNING — this backend's accuracy is NOT comparable to the other two.

One `claude -p` call per batch produces `predicted_answer` AND `correct`
together, from a prompt that contains the gold answer. The model therefore
grades itself with the answer key in front of it. What that measures is
"does the retrieved memory pack support the gold answer", NOT "can a reader
recover the answer without seeing it". Numbers from this script must never be
compared against, or reported alongside, the two-stage backends:

    reader_judge.py      — Codex CLI,  blind reader → separate gold judge
    reader_judge_api.py  — chat API,   blind reader → separate gold judge

Both of those withhold gold from the reader and introduce it only in the
judge_chunk files, so their accuracy is gold-verified. Prefer them. This
module is kept for reproducing historical single-pass runs.

Mechanics: Claude Code has no `--output-schema` flag, and granting it file
tools in headless mode triggers permission prompts, so each batch inlines its
chunk + the JSON schema into the prompt, forbids tool use, and parses a single
bare JSON object from stdout. The result_*.json files are written in the exact
same shape as the other backends, so the shared `summarize()` consumes them
unchanged.

`--backend codex` re-runs the two-stage Codex flow that `reader_judge.py`
already implements; it exists for callers that want both protocols behind one
entrypoint and is a duplicate of that module's main flow.

All detail.log parsing, chunking, schema, and summary logic is imported from
`reader_judge` — this module only swaps the per-batch runner.

Usage:
    python -m benches.longmemeval.reader_judge_singlepass RUN/detail.log \
        [--top-k 5] [--batch-size 25] [--model opus] [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

from .reader_judge import (
    batched,
    extract_json_object,
    judge_result_path_for_chunk,
    load_reader_results,
    materialize_final_result,
    normalize_reader_predictions,
    parse_detail_log,
    reader_result_path_for_chunk,
    result_path_for_chunk,
    run_chunk,
    select_records,
    summarize,
    trace_path_for_chunk,
    write_chunk,
    write_judge_chunk,
    write_judge_schema,
    write_records_jsonl,
    write_reader_schema,
    write_schema,
)

# Same reader/judge contract as the Codex backend (reader_judge.build_prompt),
# kept self-contained here because the Claude prompt inlines the chunk instead
# of asking the model to read a file. Keep the two in sync if the rubric changes.
_READER_JUDGE_RULES = (
    "你是 LongMemEval reader 和 judge。不要编辑文件，不要使用外部资料。"
    "每题只能使用该题 Question、Question Date、Computed Memory 和 Top-k Memory Pack 作答。"
    "The memories block has two sections: Evidence (quoted excerpts) and Answer aids "
    "(structured summaries — ordered lists, timelines, counts — computed by the memory "
    "system from the full store, not limited to the excerpts shown). Both sections are "
    "part of the retrieved memory."
    "Before answering, check the question's presupposition against the memory: if the "
    "premise itself (the event, role, purchase, or attribute the question assumes) is "
    "not supported by the evidence, answer that the memory is insufficient rather than "
    "adapting a similar-but-different fact to fit the question."
    "Question Date 是该题的 today/reference date；遇到 today、currently、now、latest、recent "
    "等相对时间表达时必须以它为准。"
    "Computed Memory 是 gateway 压缩后的答案候选/操作数/直接记忆；Top-k Memory "
    "是检索到的原始 session memory。memory_plan 只描述本题需要收集的操作数/"
    "时间约束；operand_memory 汇总数字/金额/百分比等操作数；timeline_memory 汇总"
    "日期和事件；assistant_memory 保留上一轮 assistant 输出/列表/具体标识信息/引用等长上下文；"
    "temporal_memory 给出相对时间解析后的目标日期及命中；preference_memory 保留用户约束。"
    "answer_result/direct_answer_span/current_state_direct 优先级高于 derived answer card。"
    "任何 primary_answer_result_untrusted 都只能当作低可信提示，不能单独作为"
    "答案依据。若 Computed Memory 和 Top-k Memory 冲突，以能被原始记忆支持的事实为准。"
    "对 how many/how much/total/average/percentage/order/before/after/since 类问题，"
    "先列出所有可见操作数再计算；缺一个关键操作数且没有直接记忆时才判 memory 不足。然后将你的 "
    "predicted_answer 与该题 Gold A 做语义一致性判断。人名、日期、数量、地点等"
    "关键事实必须一致；措辞不同但语义等价算 correct；证据不足或答案和 A 关键事实"
    "冲突算 incorrect。返回 JSON，items 中每题一个对象，字段 idx, "
    "predicted_answer, correct, reason。"
)


def build_claude_prompt(chunk_text: str, schema_text: str) -> str:
    return (
        _READER_JUDGE_RULES
        + "\n\n下面三反引号代码块内是本批题目（Markdown，已内联，无需也不要读取任何文件）：\n```\n"
        + chunk_text
        + "\n```\n\n"
        + "严格要求：只输出一个 JSON 对象，必须符合下面的 JSON Schema；不要使用任何工具、"
        "不要读写文件、不要输出 JSON 以外的任何文字、解释或 Markdown 代码围栏。\nJSON Schema:\n"
        + schema_text
    )


def run_claude(
    *,
    claude_bin: str,
    model: str,
    chunk_path: Path,
    schema_path: Path,
    result_path: Path,
    trace_path: Path,
    timeout: int,
    call_delay: float,
    max_retries: int,
) -> int:
    """Answer + judge one batch with `claude -p`. Returns 0 on success, non-zero
    on timeout / non-zero exit / unparseable output. Writes result_path (same
    shape as the Codex backend) and a trace_path with the raw stdout/stderr."""
    chunk_text = chunk_path.read_text(encoding="utf-8")
    schema_text = schema_path.read_text(encoding="utf-8")
    prompt = build_claude_prompt(chunk_text, schema_text)
    cmd = [claude_bin, "-p", "--model", model] if model else [claude_bin, "-p"]

    last_stdout = ""
    last_stderr = ""
    last_rc = -1
    for attempt in range(max_retries + 1):
        if call_delay > 0:
            time.sleep(call_delay)
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_stdout, last_stderr, last_rc = "", "TIMEOUT", 124
            if attempt < max_retries:
                time.sleep(5)
                continue
            break
        last_stdout = proc.stdout or ""
        last_stderr = proc.stderr or ""
        last_rc = proc.returncode
        lowered = last_stdout.lower()
        if "hit your limit" in lowered or "usage limit" in lowered:
            wait = 30 * (attempt + 1)
            if attempt < max_retries:
                print(
                    f"    rate limited ({model}); waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries + 1})",
                    flush=True,
                )
                time.sleep(wait)
                continue
        data = extract_json_object(last_stdout)
        if proc.returncode == 0 and isinstance(data, dict) and isinstance(
            data.get("items"), list
        ):
            result_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _write_trace(trace_path, cmd, last_rc, last_stdout, last_stderr)
            return 0
        if attempt < max_retries:
            time.sleep(5)
            continue

    _write_trace(trace_path, cmd, last_rc, last_stdout, last_stderr)
    return last_rc if last_rc != 0 else 1


def _write_trace(
    trace_path: Path, cmd: Sequence[str], rc: int, stdout: str, stderr: str
) -> None:
    with trace_path.open("w", encoding="utf-8") as trace:
        trace.write(f"$ {' '.join(cmd)}\nrc={rc}\n")
        trace.write("=== STDOUT ===\n")
        trace.write(stdout)
        trace.write("\n=== STDERR ===\n")
        trace.write(stderr)
        trace.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benches.longmemeval.reader_judge_singlepass",
        description=(
            "Run a Claude Code (`claude -p`) reader/judge pass over a LongMemEval "
            "detail.log using Question + top-k evidence from the log."
        ),
    )
    parser.add_argument("detail_log", help="path to benches/longmemeval/runs/.../detail.log")
    parser.add_argument(
        "--backend",
        choices=("claude", "codex"),
        default="claude",
        help=(
            "runner backend. claude preserves the original single-pass Claude "
            "runner; codex uses Codex blind reader plus separate semantic judge"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="number of evidence hits per question")
    parser.add_argument("--batch-size", type=int, default=25, help="questions per Claude call")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("LME_READER_CONCURRENCY", "5")),
        help="Codex backend only: number of batch calls to run concurrently",
    )
    parser.add_argument("--start-index", type=int, default=1, help="first 1-based question index to include")
    parser.add_argument("--limit", type=int, default=0, help="max questions to include; 0 = all after start")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "output directory; default: <detail-log-dir>/reader_judge_<BACKEND>_top<TOP_K>"
        ),
    )
    parser.add_argument("--claude-bin", default="claude", help="Claude Code CLI binary")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI binary")
    parser.add_argument("--repo-root", default=".", help="Codex backend only: working directory passed to codex exec")
    parser.add_argument("--model", default="opus", help="Claude model alias/name (default: opus)")
    parser.add_argument(
        "--codex-model",
        default=None,
        help=(
            "Codex backend only: optional Codex model name. If omitted, Codex "
            "uses its configured default instead of the Claude-oriented --model default"
        ),
    )
    parser.add_argument("--timeout", type=int, default=300, help="per-batch CLI timeout in seconds")
    parser.add_argument(
        "--call-delay",
        type=float,
        default=1.0,
        help="seconds to sleep before each claude call (rate-limit cushion)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="retries per batch on failure / rate limit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun batches even if result JSON files already exist",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only parse detail.log and write chunk/schema files; do not call Claude",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="do not call Claude; summarize existing result_*.json files",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue later batches after a Claude failure; summary will show missing idx",
    )
    parser.add_argument(
        "--max-record-chars",
        type=int,
        default=int(os.environ.get("LME_READER_MAX_RECORD_CHARS", "2600")),
        help="Codex backend only: soft character budget per question in generated reader chunks",
    )
    parser.add_argument(
        "--raw-topk-only",
        action="store_true",
        help="Codex backend only: omit Computed Memory and feed only raw top-k memory trunks",
    )
    return parser.parse_args(argv)


def _codex_model_from_args(args: argparse.Namespace) -> str:
    if args.codex_model is not None:
        return args.codex_model
    # `--model` historically defaults to Claude's "opus". Do not pass that
    # value through to Codex unless the caller explicitly uses --codex-model.
    return "" if args.model == "opus" else args.model


def run_codex_backend(
    *,
    args: argparse.Namespace,
    records,
    chunks: Sequence[Path],
    out_dir: Path,
    detail_log: Path,
) -> int:
    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2

    reader_schema_path = write_reader_schema(out_dir)
    judge_schema_path = write_judge_schema(out_dir)
    codex_model = _codex_model_from_args(args)

    if not args.prepare_only and not args.summarize_only:
        repo_root = Path(args.repo_root).resolve()
        pending_chunks: list[Path] = []
        for chunk_path in chunks:
            reader_result_path = reader_result_path_for_chunk(chunk_path)
            if reader_result_path.exists() and not args.force:
                print(f"skip {chunk_path.name}: {reader_result_path.name} exists")
                continue
            pending_chunks.append(chunk_path)

        if pending_chunks:
            workers = min(args.concurrency, len(pending_chunks))
            print(f"running codex reader {len(pending_chunks)} batches with concurrency={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_chunk,
                        chunk_path=chunk_path,
                        codex_bin=args.codex_bin,
                        repo_root=repo_root,
                        schema_path=reader_schema_path,
                        model=codex_model,
                        stage="reader",
                    )
                    for chunk_path in pending_chunks
                ]
                for future in as_completed(futures):
                    try:
                        _name, rc, trace_path = future.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"Codex worker failed: {exc}", file=sys.stderr)
                        if not args.keep_going:
                            for other in futures:
                                other.cancel()
                            return 1
                        continue
                    if rc != 0 and not args.keep_going:
                        print(f"Codex failed; see {trace_path}", file=sys.stderr)
                        for other in futures:
                            other.cancel()
                        return rc

        reader_items = load_reader_results(out_dir)
        normalized_count = normalize_reader_predictions(records, reader_items)
        if normalized_count:
            print(f"normalized reader predictions: {normalized_count}", flush=True)

        judge_chunks: list[Path] = []
        for batch, chunk_path in zip(batched(records, args.batch_size), chunks):
            if not any(record.idx in reader_items for record in batch):
                continue
            judge_chunks.append(write_judge_chunk(batch, out_dir, reader_items))

        pending_judge_chunks: list[Path] = []
        for judge_chunk in judge_chunks:
            final_path = result_path_for_chunk(judge_chunk)
            judge_result_path = judge_result_path_for_chunk(judge_chunk)
            if final_path.exists() and judge_result_path.exists() and not args.force:
                print(f"skip {judge_chunk.name}: {final_path.name} exists")
                continue
            pending_judge_chunks.append(judge_chunk)

        if pending_judge_chunks:
            workers = min(args.concurrency, len(pending_judge_chunks))
            print(f"running codex judge {len(pending_judge_chunks)} batches with concurrency={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_chunk,
                        chunk_path=chunk_path,
                        codex_bin=args.codex_bin,
                        repo_root=repo_root,
                        schema_path=judge_schema_path,
                        model=codex_model,
                        stage="judge",
                    )
                    for chunk_path in pending_judge_chunks
                ]
                for future in as_completed(futures):
                    try:
                        _name, rc, trace_path = future.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"Codex worker failed: {exc}", file=sys.stderr)
                        if not args.keep_going:
                            for other in futures:
                                other.cancel()
                            return 1
                        continue
                    if rc != 0 and not args.keep_going:
                        print(f"Codex failed; see {trace_path}", file=sys.stderr)
                        for other in futures:
                            other.cancel()
                        return rc

        for judge_chunk in judge_chunks:
            materialize_final_result(chunk_path=judge_chunk, reader_items=reader_items)

    if args.prepare_only:
        return 0

    summary = summarize(
        detail_log=detail_log,
        out_dir=out_dir,
        records=records,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    summary["method"] = (
        "Codex CLI blind reader plus separate semantic judge; reader chunks contain "
        "Question, Question Date, Computed Memory, and Top-k Memory only. Gold A is "
        "introduced only in judge_chunk files after reader_result files are written."
    )
    summary_path = out_dir / f"summary_reader_judge_codex_top{args.top_k}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "accuracy "
        f"{summary['correct']}/{summary['judged_total']} = {summary['accuracy']:.2%}; "
        f"summary: {summary_path}"
    )
    if summary["missing_idx"]:
        print(f"missing idx count: {len(summary['missing_idx'])}", file=sys.stderr)
        return 1
    if summary["duplicate_idx"] or summary["unexpected_idx"]:
        print("duplicate or unexpected idx present", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    detail_log = Path(args.detail_log)
    if not detail_log.exists():
        print(f"detail log not found: {detail_log}", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k must be >= 1", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.prepare_only and args.summarize_only:
        print("--prepare-only and --summarize-only are mutually exclusive", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else detail_log.parent / f"reader_judge_{args.backend}_top{args.top_k}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = select_records(
        parse_detail_log(detail_log, args.top_k),
        start_index=max(1, args.start_index),
        limit=max(0, args.limit),
    )
    if not records:
        print("no records selected", file=sys.stderr)
        return 2

    schema_path = write_schema(out_dir)
    write_records_jsonl(records, out_dir)
    chunks = [
        write_chunk(
            batch,
            out_dir,
            max_record_chars=max(1200, args.max_record_chars),
            raw_topk_only=args.raw_topk_only,
        )
        for batch in batched(records, args.batch_size)
    ]
    print(f"prepared {len(records)} records in {len(chunks)} batches under {out_dir}")

    if args.backend == "codex":
        return run_codex_backend(
            args=args,
            records=records,
            chunks=chunks,
            out_dir=out_dir,
            detail_log=detail_log,
        )

    if not args.prepare_only and not args.summarize_only:
        for chunk_path in chunks:
            result_path = result_path_for_chunk(chunk_path)
            trace_path = trace_path_for_chunk(chunk_path)
            if result_path.exists() and not args.force:
                print(f"skip {chunk_path.name}: {result_path.name} exists")
                continue
            print(f"start {chunk_path.name} (claude --model {args.model})", flush=True)
            rc = run_claude(
                claude_bin=args.claude_bin,
                model=args.model,
                chunk_path=chunk_path,
                schema_path=schema_path,
                result_path=result_path,
                trace_path=trace_path,
                timeout=args.timeout,
                call_delay=args.call_delay,
                max_retries=args.max_retries,
            )
            print(f"done {chunk_path.name} rc={rc}", flush=True)
            if rc != 0 and not args.keep_going:
                print(f"Claude failed; see {trace_path}", file=sys.stderr)
                return rc

    if args.prepare_only:
        return 0

    summary = summarize(
        detail_log=detail_log,
        out_dir=out_dir,
        records=records,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    # summarize() stamps a Codex-flavored method string; correct it for this
    # backend without touching reader_judge.py.
    summary["method"] = (
        f"Claude Code CLI (`claude -p --model {args.model}`) reader/judge; each item "
        "uses only Question + top-k Memory Pack from the detail log; semantic match "
        "against the log A line."
    )
    summary_path = out_dir / f"summary_reader_judge_singlepass_top{args.top_k}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "accuracy "
        f"{summary['correct']}/{summary['judged_total']} = {summary['accuracy']:.2%}; "
        f"summary: {summary_path}"
    )
    if summary["missing_idx"]:
        print(f"missing idx count: {len(summary['missing_idx'])}", file=sys.stderr)
        return 1
    if summary["duplicate_idx"] or summary["unexpected_idx"]:
        print("duplicate or unexpected idx present", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

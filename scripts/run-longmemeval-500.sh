#!/usr/bin/env bash
# run-longmemeval-500.sh — the full 500-question pipeline, end to end.
#
# This is the recorded path to 91.60% accuracy (458/500) on LongMemEval_S.
# Every default below is pinned to that run; override via the LME_* / judge
# env vars if you are deliberately measuring something else.
#
#   Stage 1  retrieval, NO LLM        →  $OUT_DIR/detail.log
#            gateway search only; writes the memory packs the reader will see.
#
#   Stage 2  reader + judge, 2 stages →  $OUT_DIR/$JUDGE_SUBDIR/summary_*.json
#            blind reader (Question + Question Date + top-k pack, no gold)
#            → separate judge that sees gold. Accuracy is gold-verified;
#            the reader never sees the answer key.
#
# The two stages are decoupled through detail.log, so you can re-judge an old
# retrieval run without touching the cluster (LME_STAGE=judge + LME_DETAIL_LOG).
#
# Usage:
#   DEEPSEEK_API_KEY=<openai key> scripts/run-longmemeval-500.sh
#   LME_STAGE=retrieval scripts/run-longmemeval-500.sh          # no API key needed
#   LME_STAGE=judge LME_DETAIL_LOG=path/to/detail.log \
#       DEEPSEEK_API_KEY=... scripts/run-longmemeval-500.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# all | retrieval | judge
STAGE="${LME_STAGE:-all}"
case "$STAGE" in
  all|retrieval|judge) ;;
  *) echo "LME_STAGE must be one of: all, retrieval, judge (got '$STAGE')" >&2; exit 2 ;;
esac

TS="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${LME_OUT_DIR:-benches/longmemeval/runs/$TS}"

REPORT="$OUT_DIR/report.json"
DETAIL="${LME_DETAIL_LOG:-$OUT_DIR/detail.log}"
STDOUT_LOG="$OUT_DIR/stdout.log"
JUDGE_DIR="$OUT_DIR/${LME_JUDGE_SUBDIR:-rj_final500}"
JUDGE_LOG="$JUDGE_DIR/judge_stdout.log"

# ── Stage 1: retrieval ──────────────────────────────────────────────────────
# k=5 (NOT the harness default 10): the reader was fed top-5 packs.
ADDR="${STATELET_ADDR:-127.0.0.1:9379}"
GRAPH_PREFIX="${LME_GRAPH_PREFIX:-lme-test}"
START="${LME_START:-1}"
LIMIT="${LME_LIMIT:-0}"
K="${LME_K:-5}"
SEED="${LME_SEED:-1337}"
WORKERS="${LME_WORKERS:-${STATELET_INGEST_WORKERS:-8}}"
RETRIEVE_K="${LME_RETRIEVE_K:-50}"
REFRESH_WAIT_S="${LME_REFRESH_WAIT_S:-${STATELET_LME_REFRESH_WAIT_S:-0}}"
PER_QUESTION_WAIT_S="${LME_PER_QUESTION_WAIT_S:-${STATELET_LME_PER_QUESTION_WAIT_S:-0}}"
GRANULARITY="${LME_GRANULARITY:-session}"
DETAIL_MAX_CHARS="${LME_DETAIL_MAX_CHARS:-0}"

# Answer-bundle budgets. These match config.py's defaults; pinned explicitly so
# a later config change cannot silently move the reader's input out from under
# a rerun that is meant to reproduce 91.6%.
EVIDENCE_MAX="${LME_ANSWER_EVIDENCE_MAX_CHARS:-800}"
EVIDENCE_TOTAL_MAX="${LME_ANSWER_EVIDENCE_TOTAL_MAX_CHARS:-3000}"
COMPLEX_EVIDENCE_MAX="${LME_ANSWER_COMPLEX_EVIDENCE_MAX_CHARS:-1000}"
COMPLEX_EVIDENCE_TOTAL_MAX="${LME_ANSWER_COMPLEX_EVIDENCE_TOTAL_MAX_CHARS:-6000}"

# ── Stage 2: reader + judge ─────────────────────────────────────────────────
# Env vars keep their DEEPSEEK_* names but are provider-agnostic: with a base
# URL containing openai.com the client switches to OpenAI's native dialect
# (reasoning_effort / max_completion_tokens instead of thinking / max_tokens).
JUDGE_MODEL="${LME_JUDGE_MODEL:-gpt-5.6-sol}"
JUDGE_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.openai.com}"
JUDGE_CONCURRENCY="${LME_JUDGE_CONCURRENCY:-2}"
JUDGE_TOP_K="${LME_JUDGE_TOP_K:-$K}"
# batch=1 for both stages: reasoning depth is per call, so one question per
# call keeps the budget from being split across a batch.
JUDGE_BATCH="${LME_JUDGE_BATCH:-1}"
JUDGE_READER_BATCH="${LME_JUDGE_READER_BATCH:-1}"
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-medium}"

# Fail before the (long) retrieval stage rather than after it.
if [[ "$STAGE" != "retrieval" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required for the judge stage." >&2
  echo "  It holds the key for whatever --base-url points at; with the default" >&2
  echo "  $JUDGE_BASE_URL that is an OpenAI key." >&2
  echo "  Retrieval only (no key, no LLM): LME_STAGE=retrieval $0" >&2
  exit 2
fi

if [[ "$STAGE" == "judge" && ! -f "$DETAIL" ]]; then
  echo "LME_STAGE=judge needs an existing detail log; not found: $DETAIL" >&2
  echo "  point at one with LME_DETAIL_LOG=path/to/detail.log" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "LongMemEval 500-question pipeline (stage: $STAGE)"
echo "  out dir: $OUT_DIR"
echo "  detail:  $DETAIL"
echo

if [[ "$STAGE" == "all" || "$STAGE" == "retrieval" ]]; then
  cmd=(
    python3 -m benches.longmemeval.run
    --addr "$ADDR"
    --graph-prefix "$GRAPH_PREFIX"
    --start "$START"
    --limit "$LIMIT"
    --k "$K"
    --seed "$SEED"
    --workers "$WORKERS"
    --retrieve-k "$RETRIEVE_K"
    --refresh-wait-s "$REFRESH_WAIT_S"
    --per-question-wait-s "$PER_QUESTION_WAIT_S"
    --granularity "$GRANULARITY"
    --answer-evidence-max-chars "$EVIDENCE_MAX"
    --answer-evidence-total-max-chars "$EVIDENCE_TOTAL_MAX"
    --answer-complex-evidence-max-chars "$COMPLEX_EVIDENCE_MAX"
    --answer-complex-evidence-total-max-chars "$COMPLEX_EVIDENCE_TOTAL_MAX"
    --out "$REPORT"
    --detail-log "$DETAIL"
    --detail-max-chars "$DETAIL_MAX_CHARS"
    # Stage 2 does the answering; keep stage 1 LLM-free even if a key is set.
    --retrieval-only
  )

  # Default is to reuse the already-ingested 500-question store (the 91.6% run
  # did). Set LME_REINGEST=1 to ingest, and LME_RESET=1 to drop graphs first.
  if [[ "${LME_REINGEST:-0}" != "1" ]]; then
    cmd+=(--no-ingest)
  elif [[ "${LME_RESET:-0}" == "1" ]]; then
    cmd+=(--reset)
  fi

  if [[ "${LME_ABSTENTION:-0}" == "1" ]]; then
    cmd+=(--abstention)
  fi

  if [[ -n "${LONGMEMEVAL_QUESTIONS_DIR:-}" ]]; then
    cmd+=(--dataset-dir "$LONGMEMEVAL_QUESTIONS_DIR")
  fi

  echo "── Stage 1/2: retrieval (LLM-free) ──"
  echo "  command: ${cmd[*]}"
  echo
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" "${cmd[@]}" | tee "$STDOUT_LOG"

  echo
  echo "Retrieval category metrics:"
  awk '
    /^category[[:space:]]+n[[:space:]]+hit@/ { printing = 1 }
    printing { print }
  ' "$STDOUT_LOG"
  echo
fi

if [[ "$STAGE" == "all" || "$STAGE" == "judge" ]]; then
  mkdir -p "$JUDGE_DIR"

  judge_cmd=(
    python3 -m benches.longmemeval.reader_judge_api
    "$DETAIL"
    --model "$JUDGE_MODEL"
    --base-url "$JUDGE_BASE_URL"
    --top-k "$JUDGE_TOP_K"
    --batch-size "$JUDGE_BATCH"
    --reader-batch-size "$JUDGE_READER_BATCH"
    --concurrency "$JUDGE_CONCURRENCY"
    --no-reader-thinking
    --out-dir "$JUDGE_DIR"
  )

  echo "── Stage 2/2: blind reader + gold-verified judge ──"
  echo "  model:   $JUDGE_MODEL @ $JUDGE_BASE_URL (effort=$OPENAI_REASONING_EFFORT)"
  echo "  out dir: $JUDGE_DIR"
  echo "  command: ${judge_cmd[*]}"
  echo
  # Completed batches are skipped on rerun (reader stage keys on
  # reader_result_*.json, judge stage on result_* + judge_result_*), so an
  # interrupted run resumes by re-invoking this script with the same LME_OUT_DIR.
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" "${judge_cmd[@]}" | tee "$JUDGE_LOG"
  echo
fi

echo "Artifacts:"
[[ -f "$REPORT" ]]     && echo "  $REPORT"
[[ -f "$DETAIL" ]]     && echo "  $DETAIL"
[[ -f "$STDOUT_LOG" ]] && echo "  $STDOUT_LOG"
if [[ -d "$JUDGE_DIR" ]]; then
  echo "  $JUDGE_DIR/"
  for f in "$JUDGE_DIR"/summary_*.json; do
    [[ -e "$f" ]] && echo "    $f"
  done
fi
exit 0

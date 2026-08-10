# statelet-longmemeval

LongMemEval benchmark harness for [Statelet](https://github.com/stateletlab/statelet)
— agent-memory **retrieval + QA** measured over all 500 questions of
`LongMemEval_S`, mirroring
[mem0's benchmark methodology](https://github.com/mem0ai/memory-benchmarks)
(the run reporting ~94.4% accuracy).

Split out of the main repository so evaluation work does not churn the
database's history, and so a run's ~250 MB of output has somewhere to live that
is not the engine repo.

**Best recorded result: 91.60% accuracy (458/500).** The exact recipe is in
[Recorded run](#recorded-run-9160--458500) and is encoded in
`scripts/run-longmemeval-500.sh`.

## Quick start

```bash
# 0. Install the Statelet Python client (`import statelet`; the harness will
#    not start without it). `statelet` pulls it in and adds the server
#    binaries; `statelet-sdk` is the client on its own. Either works here:
pip install statelet
pip install -e /path/to/statelet-sdk/python

# 1. Start a cluster (in the Statelet checkout — this also sets the ingest
#    env vars described under Phase A below):
scripts/cluster-741.sh

# 2a. Retrieval only — no API key, no LLM, the hit@k gate:
LME_STAGE=retrieval scripts/run-longmemeval-500.sh

# 2b. Both phases, defaults pinned to the 91.6% recipe:
DEEPSEEK_API_KEY=<openai key> scripts/run-longmemeval-500.sh

# 2c. Re-judge an existing detail.log without touching the cluster:
LME_STAGE=judge LME_DETAIL_LOG=path/to/detail.log \
    DEEPSEEK_API_KEY=... scripts/run-longmemeval-500.sh
```

## Layout

| Path | What |
|---|---|
| `benches/longmemeval/` | The harness proper — ingest, query, judging, metrics, reporting. A Python package with **relative imports**, invoked as `python -m benches.longmemeval`. |
| `eval/` | Standalone retrieval/answer harnesses (`longmemeval_retrieval_eval.py` and the per-model drivers) plus `sessiondoc_postprocess.py`, which imports the retrieval harness directly. Independent of the `benches/` package. |
| `scripts/` | Shell driver for the full-500 run (`run-longmemeval-500.sh`). |
| `.github/workflows/` | `publish-statelet-pypi.yml` — builds and publishes the `statelet` server wheel. See [Publishing](#publishing-the-statelet-server-wheel-to-pypi). |
| `.github/scripts/` | `manylinux-build.sh` — the Linux half of that build, run inside a manylinux container. |
| `packaging/pypi/` | The server wheel's packaging skeleton: `pyproject.toml`, the `statelet_server` package and its CLI shims. Binaries are dropped in at build time. |

The `benches/` prefix is kept rather than flattened to a top-level
`longmemeval/`: the package uses relative imports and every documented command
is `python -m benches.longmemeval …`, so moving it would invalidate this README
and the shell driver for no functional gain.

## Publishing the `statelet` server wheel to PyPI

`pip install statelet` is served by
[`.github/workflows/publish-statelet-pypi.yml`](.github/workflows/publish-statelet-pypi.yml).

**What the package is.** `statelet` on PyPI is the **server** distribution: the
three Rust executables, the console-script shims that exec them, and
`statelet-cluster` — a Python port of the engine's bash launcher, so
`pip install statelet && statelet-cluster start` brings a local cluster up in
one step on Windows as well — all under a `statelet_server` import package. The client library is *not* bundled — it is
declared as a dependency on `statelet-sdk`, published from
[stateletlab/statelet-sdk](https://github.com/stateletlab/statelet-sdk). So
`pip install statelet` is still one step and still gives you both, while only
one distribution owns the `statelet` import package.

That split is not cosmetic. Two distributions cannot both ship `statelet/`:
pip lets the second overwrite the first's files, and uninstalling either one
then breaks the other. The dependency edge keeps a single authoritative copy of
the client code in the SDK repository.

**Where the pieces come from.** The wheel's packaging skeleton lives here, in
[`packaging/pypi/`](packaging/pypi) — `pyproject.toml`, the `statelet_server`
package and its CLI shims. The engine checkout contributes compiled binaries
and nothing else, so the SDKs moving out of `stateletlab/statelet` cannot break
this build. Per target the workflow drops `gateway` / `metadata_service` /
`raft_engine` into `statelet_server/bin/` as `statelet-gateway` / `-metadata` /
`-datanode`, builds a wheel, and retags it from `py3-none-any` to the real
platform tag.

| Target | Built on | Wheel tag |
|---|---|---|
| `aarch64-apple-darwin` | `macos-14` | `macosx_11_0_arm64` |
| `x86_64-apple-darwin` | `macos-14`, cross | `macosx_10_12_x86_64` |
| `x86_64-unknown-linux-gnu` | `manylinux2014_x86_64` container | `manylinux2014_x86_64` |
| `aarch64-unknown-linux-gnu` | `manylinux2014_aarch64` container on an ARM runner | `manylinux2014_aarch64` |
| `x86_64-pc-windows-msvc` | `windows-latest` | `win_amd64` |

**Why two Linux rows cover every distribution.** `manylinux` is a cross-distro
ABI contract, not a distro: one `manylinux2014` wheel serves any glibc ≥ 2.17
machine — CentOS/RHEL 7 and up, Debian 8+, Ubuntu 14.04+, Fedora, Arch,
openSUSE, Rocky, Alma. That is why the Linux binaries compile *inside* the
official manylinux images. Building on `ubuntu-latest` and relabelling the
wheel would emit a binary needing glibc 2.39 under a tag promising 2.17, which
pip installs happily and which then dies at startup with `GLIBC_2.39 not
found`. A `Verify the glibc floor` step reads the actual symbol versions back
out of the binaries with `objdump` and fails the build if they exceed what the
tag claims, so the promise cannot silently rot.

The image is a `docker run` inside the job rather than the job's `container:`.
CentOS 7 predates glibc 2.28, which the Actions runner's own node20 requires,
so `actions/checkout` and the other JavaScript actions stay on the host and
only `cargo` goes inside the image.

**What is not covered.** No sdist and no pure-Python fallback wheel are
published, so anything outside the table gets "no matching distribution" and
must install from a checkout. That leaves Alpine and other musl distributions,
which would need `musllinux` wheels, and Windows on ARM.

**One-time setup:**

1. `secrets.STATELET_REPO_TOKEN` on this repository — a fine-grained PAT with
   `Contents: read` on `stateletlab/statelet`, used only by `actions/checkout`.
2. A GitHub environment named `pypi` here, registered on PyPI as a Trusted
   Publisher for project `statelet`: owner `stateletlab`, repository
   `statelet-longmemeval`, workflow `publish-statelet-pypi.yml`, environment
   `pypi`. For the first ever release this is a *pending* publisher, created
   from the PyPI account page before the project exists.
3. `statelet-sdk` published from `stateletlab/statelet-sdk`. Until it exists
   on PyPI, `pip install statelet` resolves the dependency and fails — publish
   the client first. (The build itself does not care: its smoke test installs
   with `--no-deps`.)

**Running it.** Manually via *Actions → Publish statelet to PyPI → Run
workflow*, taking a `ref` of the engine repository (default `main`), an
optional `version` override, and a `dry_run` box that builds and smoke-tests
the wheels without publishing. Or automatically by pushing a
`statelet-v<version>` tag here, which builds engine tag `<version>` —
`statelet-v0.1.1` → `v0.1.1`.

The version comes from [`packaging/pypi/pyproject.toml`](packaging/pypi/pyproject.toml)
unless overridden; publishing uses `skip-existing`, so re-running against a
version already on PyPI is a no-op rather than an error. Keep the triggers as
they are — `pull_request` would hand the private-repo token to fork
contributors.

**A note on the container build script.** The manylinux build lives in
[`.github/scripts/manylinux-build.sh`](.github/scripts/manylinux-build.sh)
rather than inline in the workflow. It was inline once, inside
`docker run ... bash -c '...'`, and a single apostrophe in a comment closed the
quote early and silently ran the rest of the script *on the host* — where the
failure surfaced as `yum: command not found`. A file has no such failure mode.

## The three phases, and who owns each

The thing under test spans two processes. Only the middle phase lives in this
repo, which is the most common source of confusion when a rerun does not
reproduce a number:

| Phase | Runs where | Controlled by | LLM |
|---|---|---|---|
| **A. Ingest** — conversations → memory graph | Statelet gateway | gateway env vars (`STATELET_LLM_*`), set by the cluster launcher | **Yes**, server-side, if `STATELET_LLM_EXTRACT=1` |
| **B. Retrieval** — question → top-k memory packs | this harness | `run.py` CLI flags | No |
| **C. Reader + judge** — packs → answer → correct? | this harness | `reader_judge_api.py` CLI flags | Yes, the reader/judge model |

Nothing in this repo can set, read back, or record phase A's configuration.
A detail.log records what was *retrieved*, never how the memory it came from
was *built*. Encode that in the run directory name (as
`reingest500_dsv4_20260714` does) — it is the only durable record you get.

### Phase A — ingest (gateway-side)

The harness writes through the normal gateway write path. Whether the gateway
distills that text with an LLM or with its rule-based extractor is a
**server-side** decision (`src/engine/embedding/llm_extract.rs` in the Statelet
checkout):

| Var | Default | Purpose |
|---|---|---|
| `STATELET_LLM_EXTRACT` | unset | `1` / `true` to enable LLM extraction |
| `STATELET_LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `STATELET_LLM_MODEL` | `deepseek-chat` | extraction model |
| `STATELET_LLM_API_KEY` | required | key for that endpoint |
| `STATELET_LLM_EXTRACT_THRESHOLD` | `500` | min chars before extraction triggers |
| `STATELET_LLM_CONFLICT_THRESHOLD` | `0.35` | cosine distance that counts as a conflict |

Enabled, it does two things: **fact extraction** (long conversational turns →
self-contained dated facts) and **conflict resolution** (a new fact is checked
against existing memories by vector similarity; the LLM decides
UPDATE-with-supersedes-edge / DELETE / NONE). Disabled, the gateway uses its
rule-based `LocalExtractor` plus the local ONNX model stack, and ingestion is
fully LLM-free.

`scripts/cluster-741.sh` in the Statelet checkout — the launcher used for this
epic — turns extraction **on by default** and points it at DeepSeek. So a
cluster started that way ingests with DeepSeek even though this harness never
asks it to.

> Retrieval stays local either way: query rewriting, embedding, reranking and
> fusion run on the ONNX stack. The ingest LLM shapes *what is stored*, never
> *how it is searched*.

### Phase B — retrieval (this repo, LLM-free)

| Stage | Module | What |
|---|---|---|
| 1. Inject | `inject.py` + `segment.py` | Load history, sentence-segment (merge-short / split-long), ingest every question via the gateway write path with bitemporal timestamps from session/message order. One namespaced graph per question. Skipped with `--no-ingest`. |
| 2. Query | `query.py` | Top-k retrieval for every question (over-fetch `--retrieve-k`, drop fragment hits, dedup, keep `k`). |
| 3. hit@k | `metrics.py` | `hit@k` / `recall@k` / `mrr` vs labeled gold-evidence sessions, per category + overall. **The LLM-free gate.** |
| 4. Report | `report.py` | hit@k, recall@k, accuracy, mean tokens/query (per-category + overall) + a side-by-side row vs mem0. |

Output: `report.json` plus **`detail.log`**, which carries the per-question
memory packs. detail.log is the seam between phase B and phase C — phase C
never talks to the gateway.

### Phase C — reader + judge (this repo)

Three backends, all writing byte-compatible
`reader_result_*` / `judge_result_*` / `result_*.json` consumed by one shared
`summarize()`. **They do not measure the same thing:**

| Module | Protocol | Transport | Comparable? |
|---|---|---|---|
| `reader_judge_api.py` | two-stage, gold-verified | HTTP chat-completions | ✅ **use this** |
| `reader_judge.py` | two-stage, gold-verified | Codex CLI subprocess | ✅ |
| `reader_judge_singlepass.py` | single-pass **self-assessment** | Claude Code CLI | ❌ never compare |

Two-stage means: stage 1 the **reader** sees only Question + Question Date +
top-k memory pack and writes `reader_result_*.json` — the gold answer is not in
its prompt. Stage 2 the **judge** gets that prediction plus gold (introduced for
the first time in `judge_chunk_*.md`) and rules on semantic equivalence.
Accuracy is therefore gold-verified.

Single-pass produces the answer and the verdict in one call from a prompt that
contains gold, i.e. the model grades itself with the answer key visible. That
measures "does the pack support gold", not "can a reader recover the answer".
Keep those numbers out of any table that also has two-stage numbers.

## Recorded run: 91.60% (458/500)

Full 500 questions, `LongMemEval_S`.

**Phase A — re-ingest with DeepSeek v4 extraction (2026-07-14).** Gateway
started with `STATELET_LLM_EXTRACT=1` against DeepSeek v4; the store was rebuilt
from scratch, which is what the run directory name records:
`runs/reingest500_dsv4_20260714/`. Every later phase reuses this store.

**Phase B — retrieval only, 2026-07-22** → `final500_v7_detail.log`:

```
gateway   127.0.0.1:9379          granularity  session
k         5                       retrieve_k   50
seed      1337                    filter_fragments + dedup   on
answer bundle   800 / 3000 per-evidence / total chars
   complex      1000 / 6000
ingest    SKIPPED (--no-ingest, reusing the phase-A store)
LLM       DISABLED (--retrieval-only)
```

**Phase C — blind reader + gold judge, `gpt-5.6-sol` at medium effort:**

```bash
OPENAI_REASONING_EFFORT=medium \
DEEPSEEK_API_KEY=<openai key> \
DEEPSEEK_BASE_URL=https://api.openai.com \
python -m benches.longmemeval.reader_judge_api \
  benches/longmemeval/runs/reingest500_dsv4_20260714/final500_v7_detail.log \
  --model gpt-5.6-sol --top-k 5 --batch-size 1 --reader-batch-size 1 \
  --no-reader-thinking --concurrency 2 \
  --out-dir benches/longmemeval/runs/reingest500_dsv4_20260714/rj_final500_solplat_med
```

`--batch-size 1 --reader-batch-size 1` (one question per call, so reasoning
budget is not split across a batch) × 500 questions × 2 stages = 1000 API
calls at concurrency 2. The run was interrupted twice and resumed by
re-issuing the same command: the reader stage skips a chunk when its
`reader_result_*.json` exists, the judge stage when `result_*` **and**
`judge_result_*` both exist.

The `DEEPSEEK_*` env names are historical and provider-agnostic — a base URL
containing `openai.com` switches the client to OpenAI's native dialect
(`reasoning_effort` / `max_completion_tokens` instead of `thinking` /
`max_tokens`), so the OpenAI key goes in `DEEPSEEK_API_KEY`.

**Result: `accuracy 458/500 = 91.60%`.**

> The artifacts of this run are not in this repo (`runs/` is gitignored) and are
> no longer on the original machine. Per-category numbers are recoverable only
> if the out-dir resurfaces, via `--summarize-only`, which re-derives everything
> from the `result_*.json` files without spending API calls.

## Running it directly

```bash
python -m benches.longmemeval               # LLM-free, all 500
python -m benches.longmemeval --limit 20    # smoke
python -m benches.longmemeval --only smoke_0001,smoke_0002
```

Key flags: `--addr`, `--dataset-dir`, `--limit`, `--only`, `--k` (default 5),
`--graph-prefix`, `--seed`, `--no-ingest`, `--refresh-wait-s` (NRT visibility
barrier), `--retrieval-only`, `--out`.

`scripts/run-longmemeval-500.sh` pins `k=5`, `seed=1337`,
`granularity=session`, `retrieve_k=50`, `--no-ingest`, `--retrieval-only`, the
four answer-bundle char budgets, and the phase-C model/effort/batching.
Override any of them with the `LME_*` env vars listed at the top of the script.
Note `LME_REINGEST=1` to actually ingest (`LME_RESET=1` to drop graphs first) —
the default reuses the existing store, and if that store is gone you get a
valid-looking run with near-zero retrieval rather than an error.

## Evidence packing

The harness requests gateway answer bundles by default and writes only compact
`Computed Evidence` entries into `detail.log` (`evidence_plan`, trusted/direct
`operand_table`, `timeline_highlight`, `answer_results`, trusted
`primary_answer_result`, answer-bundle fields, and a small reranked fact/chunk
sample plus `top_hit_highlight` snippets from raw top-k hits). Low-confidence
primary answers are marked `primary_answer_result_untrusted` and should not be
used alone. This keeps reader prompts bounded while surfacing operands,
relative-time hints, and direct answer spans that top-k session packs may omit.

Disable with `--no-answer-bundle`, or tune prompt size with
`--answer-results-limit`, `--answer-evidence-max-chars`,
`--answer-evidence-total-max-chars`, `--answer-fact-k`, `--answer-chunk-k`.
Operand/temporal/list/preference questions use larger adaptive caps controlled
by the matching `--answer-complex-*` flags. Complex-question query fanout is
handled inside the gateway planner by default; the harness sends the public
question date as the gateway-supported `[Context date: YYYY-MM-DD]` query
prefix so temporal rewrites share the reader's reference date. Disable gateway
fanout service-side with `STATELET_ADAPTIVE_EVIDENCE_QUERIES=0` or tune it with
`STATELET_ADAPTIVE_EVIDENCE_EXTRA_QUERIES`.

## Phase 6 — calibrated abstention (LLM-free)

LongMemEval has an *abstention* category: questions unanswerable from memory. A
pure top-k retriever always returns something, so it answers — and fails —
every abstention question. `abstention.py` recovers the category with a
**calibrated confidence threshold**, no LLM:

* Two signals come from the retrieval result alone — `top_score` (top fused
  score, distance mapped to a bounded similarity) and `score_gap` (rank-1 minus
  rank-2 similarity, the decision margin).
* Thresholds are **calibrated on a held-out dev split**, never hardcoded: a grid
  search over candidates drawn from the observed score distribution maximises a
  balanced objective (abstention accuracy *and* answerable retention), subject
  to a retention guardrail so the tradeoff cannot be gamed by abstaining on
  everything.
* The fitted policy abstains iff `top_score < tau_top` OR `score_gap < tau_gap`,
  and is applied to the held-out test split.

```bash
python -m benches.longmemeval --abstention
python -m benches.longmemeval --abstention --abstention-dev-fraction 0.5 \
    --abstention-min-retention 0.95
```

The report gains an `abstention` block (fitted policy + dev/test stats) and
per-category abstention-accuracy / answerable-retention rows.

## Dataset

Resolution order (see `benches/longmemeval/config.py`):
1. `LONGMEMEVAL_QUESTIONS_DIR`
2. the `memorybench` sibling checkout (`../memorybench/.../questions`)
3. the bundled smoke set under `benches/longmemeval/data/questions/`
   (2 questions, offline dev/CI).

Each question JSON carries `haystack_sessions`, `haystack_session_ids`,
`answer_session_ids` (gold), `haystack_dates`, plus `question` / `answer` /
`question_type` / `question_date`.

## Config / reproducibility

Everything affecting phase B is pinned in `config.py` (dataset path, gateway,
`k`, embedder, seed, mem0 baseline) — a phase-B run is reproducible from
(config + seed + dataset + gateway env). The report is emitted as JSON
(`--out`, default `last_report.json`) and never contains an API key.

Phase C reproducibility lives in `reader_judge_api.py` alone: its reader/judge
rules (`_READER_RULES_EN` / `_JUDGE_RULES_EN`) and schema (`READER_SCHEMA_V2`)
are deliberately **local** to that module rather than shared with
`reader_judge.py`, so a recorded run is reproducible from that one file and
changes to the Codex backend cannot silently move it.

Phase A reproducibility is **not** captured anywhere in this repo — see the
phase table above. Record it in the run directory name.

## Tests (offline, no gateway / no LLM)

```bash
python benches/longmemeval/tests/test_offline.py
python benches/longmemeval/tests/test_run_e2e.py
python benches/longmemeval/tests/test_abstention.py
```

## References

- mem0 methodology — https://mem0.ai/research · https://github.com/mem0ai/memory-benchmarks
- LongMemEval — https://arxiv.org/pdf/2410.10813

# benches/longmemeval

The LongMemEval harness package. Invoked as `python -m benches.longmemeval` —
it uses relative imports, so run it from the repository root, not from here.

**Documentation lives in the [repository README](../../README.md)**, kept in one
place so the two cannot drift: the three-phase model (ingest / retrieval /
reader+judge) and who owns each, the recorded 91.60% (458/500) recipe, the
reader/judge backend comparison, evidence packing, abstention, dataset
resolution, and the tests.

Module map:

| Module | Role |
|---|---|
| `run.py` / `__main__.py` | Entrypoint: inject → query → metrics → report |
| `inject.py` / `segment.py` | Phase B stage 1 — sentence segmentation + gateway writes |
| `query.py` / `store.py` | Phase B stage 2 — retrieval and the SDK wrapper |
| `metrics.py` / `report.py` | hit@k / recall@k / mrr, and the report table |
| `config.py` / `dataset.py` | Pinned run config; question loading |
| `abstention.py` | Calibrated LLM-free abstention |
| `reader_judge.py` | Shared reader/judge library **and** the Codex two-stage backend |
| `reader_judge_api.py` | Two-stage backend over an OpenAI-compatible chat API — **the one that produced 91.6%** |
| `reader_judge_singlepass.py` | Single-pass self-assessment backend; its numbers are **not** comparable to the two-stage ones |
| `deepseek.py` | Optional in-pipeline DeepSeek reader/judge (bypassed by `--retrieval-only`) |

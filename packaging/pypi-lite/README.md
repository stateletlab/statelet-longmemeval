# statelet-lite

The whole [Statelet](https://github.com/stateletlab/statelet) database in ONE
single-node process — the metadata plane, the data node and the gateway fused
into one binary. Agent memory with KV, vector search, and temporal causal
graphs, without running a cluster.

```bash
pip install statelet-lite
statelet-lite                   # data in ~/.statelet/lite, or pass a directory
```

```python
from statelet import Client
c = Client("127.0.0.1:9379")
c.put(b"hello", b"world")
print(c.get(b"hello"))          # b'world'
```

Then open the admin UI at **http://127.0.0.1:9380** — databases, KV and graph
consoles, metrics. It is served out of this package; nothing else to install.

## Semantic search

Text embedding runs locally (ONNX, no API keys). ONNX Runtime itself comes in
through the `onnxruntime` wheel this package depends on — nothing to install.
The retrieval models are a one-time download, and the **first run fetches them
automatically** (~1.4 GB) into `~/.statelet/models/`:

```bash
statelet-lite                  # first run downloads the models, then starts
statelet-lite --fetch-models   # or fetch them up front, without starting
```

Four models are installed, all published by this project as English-pruned ONNX
builds:

| directory | role | from |
|---|---|---|
| `gte-base-en-v1.5` | dense embeddings (768-dim) | [`statelet/gte-base-en-v1.5`](https://huggingface.co/statelet/gte-base-en-v1.5) |
| `opensearch-neural-sparse-encoding-multilingual-v1` | learned sparse retrieval | [`statelet/…-v1-en`](https://huggingface.co/statelet/opensearch-neural-sparse-encoding-multilingual-v1-en) |
| `mmarco-mMiniLMv2` | cross-encoder reranking | [`statelet/mmarco-mMiniLMv2-en`](https://huggingface.co/statelet/mmarco-mMiniLMv2-en) |
| `nli-mdeberta` | NLI (conflict detection, relevance) | [`statelet/nli-mdeberta-en`](https://huggingface.co/statelet/nli-mdeberta-en) |

Set `STATELET_NO_FETCH_MODELS=1` to skip the download and start immediately.
`--help` and `--version` never download anything.

Without the models, KV / graph / explicit-vector calls all work, but any
text-embedding call answers `FAILED_PRECONDITION` and retrieval falls back to
keyword-only recall. Any single model can be overridden with its own variable
(`STATELET_EMBEDDING_MODEL`, `STATELET_SPARSE_MODEL`, `STATELET_RERANKER_MODEL`,
`STATELET_NLI_MODEL`); `STATELET_MODEL_ROOT` repoints all of them at once.

## Configuration

Configuration is by environment. Without `GATEWAY_JWT_SECRET` every listener
binds loopback only — set the secret (which also flips the default bind to
`0.0.0.0`) before exposing statelet-lite to a network:

| Env | Default | Description |
|---|---|---|
| `GATEWAY_ADDR` | `127.0.0.1:9379` (`0.0.0.0:9379` with auth) | gRPC address |
| `GATEWAY_MGMT_ADDR` | `127.0.0.1:9380` (`0.0.0.0:9380` with auth) | Management HTTP / admin UI |
| `GATEWAY_REDIS_ADDR` | `127.0.0.1:6380` (`0.0.0.0:6380` with auth) | Redis-compatible listener (6380 keeps clear of a local Redis) |
| `GATEWAY_JWT_SECRET` | unset (no auth) | Enable JWT auth |
| `STATELET_MODEL_ROOT` | `~/.statelet/models` | Directory holding every model directory |
| `STATELET_EMBEDDING_MODEL` | `~/.statelet/models/gte-base-en-v1.5` | Embedding model directory (overrides the root for this one model) |
| `STATELET_NO_FETCH_MODELS` | unset | Set to `1` to skip the first-run model download |
| `ORT_DYLIB_PATH` | from the `onnxruntime` wheel | ONNX Runtime shared library |
| `STATELET_LITE_DIR` | `~/.statelet/lite` | Data directory |

`statelet-lite` is hard-locked to a single node. Growing to a multi-node
cluster is a configuration change, not a data migration — the cluster binaries
(`pip install statelet`) run the same library functions over the same storage
format.

Platform wheels are published for macOS (Apple Silicon and Intel), Linux
(x86_64 and aarch64, glibc 2.17 and newer), and Windows x64. Anywhere else,
build from a checkout.

Licensed under FSL-1.1-ALv2: free for any use other than a competing service,
and each version becomes Apache-2.0 two years after its release.

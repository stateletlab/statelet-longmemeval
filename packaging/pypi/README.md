# statelet

Server binaries for [Statelet](https://github.com/stateletlab/statelet) — agent
memory with KV, vector search, and temporal causal graphs in one database.

```bash
pip install statelet
```

Then bring a local cluster up:

```bash
statelet-cluster start          # --nodes N, default 3
statelet-cluster status
statelet-cluster stop           # --clean also deletes the data directory
```

```python
from statelet import Client
c = Client("127.0.0.1:9379", username="admin", password="admin")
c.put(b"hello", b"world")
print(c.get(b"hello"))          # b'world'
```

Then open the admin UI at **http://127.0.0.1:9380** — cluster, databases, KV
and graph consoles, users, metrics. It is served by the gateway out of this
package; nothing else to install or configure.

Data and logs live under `~/.statelet/cluster`, or `$STATELET_DATA_DIR`.

## Semantic search

Text embedding runs locally in the gateway (ONNX, no API keys). ONNX Runtime
itself comes in through the `onnxruntime` wheel this package depends on —
nothing to install. The embedding model is a one-time download:

```bash
statelet-gateway --fetch-models   # multilingual-e5-small → ~/.statelet/models/
statelet-cluster start            # the gateway auto-discovers it there
```

Without the model, KV / graph / explicit-vector calls all work, but any
text-embedding call answers `FAILED_PRECONDITION`. A custom model directory
(`tokenizer.json` + `model.onnx`) can be pointed at with
`STATELET_EMBEDDING_MODEL`.

The individual executables are on your `PATH` too, for anyone driving them
under systemd, Docker or a supervisor of their own —

| Command | Role |
|---|---|
| `statelet-gateway` | stateless routing gateway; what clients connect to |
| `statelet-metadata` | metadata-plane Raft group |
| `statelet-datanode` | data-plane storage engine |

— and pulls in `statelet-sdk`, so `import statelet` works in the same
environment.

Platform wheels are published for macOS (Apple Silicon and Intel), Linux
(x86_64 and aarch64, glibc 2.17 and newer), and Windows x64. Anywhere else,
build from a checkout.

Licensed under Apache-2.0.

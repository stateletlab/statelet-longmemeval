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

Configuration is by environment:

| Env | Default | Description |
|---|---|---|
| `GATEWAY_ADDR` | `0.0.0.0:9379` | Public gRPC address |
| `GATEWAY_MGMT_ADDR` | `0.0.0.0:9380` | Management HTTP / admin UI |
| `GATEWAY_REDIS_ADDR` | `0.0.0.0:6379` | Redis-compatible listener |
| `GATEWAY_JWT_SECRET` | unset (no auth) | Enable JWT auth |
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

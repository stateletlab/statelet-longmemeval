# statelet

Server binaries for [Statelet](https://github.com/stateletlab/statelet) — agent
memory with KV, vector search, and temporal causal graphs in one database.

```bash
pip install statelet
```

That installs three executables onto your `PATH` —

| Command | Role |
|---|---|
| `statelet-gateway` | stateless routing gateway; what clients connect to |
| `statelet-metadata` | metadata-plane Raft group |
| `statelet-datanode` | data-plane storage engine |

— and pulls in `statelet-client`, so `import statelet` works in the same
environment.

Platform wheels are published for macOS (Apple Silicon and Intel), Linux
(x86_64 and aarch64, glibc 2.17 and newer), and Windows x64. Anywhere else,
build from a checkout.

Licensed under Apache-2.0.

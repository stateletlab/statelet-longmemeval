"""`statelet-cluster` — start, stop and inspect a local Statelet cluster.

The engine ships a bash script of the same name for the Homebrew and
deb/rpm installs. This is the pip equivalent, in Python because the wheel
also has to work on Windows, and resolving the binaries through
`statelet_server.binary_path` rather than `$PATH` so it runs the executables
from *this* installation instead of whichever ones happen to be earlier in
the path.

Topology, matching the bash script so the two agree:

    metadata          127.0.0.1:8379          metrics 9091
    data node i       127.0.0.1:7379+2(i-1)   raft +1, metrics 9092+i-1
    gateway           127.0.0.1:9379          mgmt 9380, redis 6379, metrics 9095
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ._cli import binary_path

METADATA_PORT = 8379
GATEWAY_PORT = 9379
GATEWAY_MGMT_PORT = 9380
GATEWAY_REDIS_PORT = 6379
MAX_NODES = 5


def data_dir() -> Path:
    return Path(os.environ.get("STATELET_DATA_DIR", Path.home() / ".statelet" / "cluster"))


def _node_ports(i: int) -> Dict[str, int]:
    return {
        "data": 7379 + (i - 1) * 2,
        "raft": 7380 + (i - 1) * 2,
        "metrics": 9092 + i - 1,
    }


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _wait_for_port(port: int, timeout: float, label: str) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.3)
    print(f"  ! {label} did not come up on port {port} within {timeout:.0f}s", file=sys.stderr)
    return False


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(40):          # 10s for a clean shutdown
        if not _pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _spawn(name: str, argv: List[str], env: Dict[str, str], root: Path) -> int:
    """Start one server detached, so it outlives this launcher."""
    log_path = root / "logs" / f"{name}.log"
    log = open(log_path, "ab")
    kwargs: Dict[str, object] = {"stdout": log, "stderr": subprocess.STDOUT, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(argv, **kwargs)  # type: ignore[arg-type]
    (root / "pids" / f"{name}.pid").write_text(str(proc.pid))
    print(f"  started {name} (pid {proc.pid}, log {log_path})")
    return proc.pid


def _read_pids(root: Path) -> Dict[str, int]:
    pids: Dict[str, int] = {}
    pid_dir = root / "pids"
    if not pid_dir.is_dir():
        return pids
    for f in sorted(pid_dir.glob("*.pid")):
        try:
            pids[f.stem] = int(f.read_text().strip())
        except ValueError:
            continue
    return pids


def _busy_ports(nodes: int) -> List[str]:
    checks = [
        (METADATA_PORT, "metadata"),
        (9091, "metadata metrics"),
        (GATEWAY_PORT, "gateway"),
        (GATEWAY_MGMT_PORT, "gateway management"),
        (GATEWAY_REDIS_PORT, "gateway redis"),
        (9095, "gateway metrics"),
    ]
    for i in range(1, nodes + 1):
        p = _node_ports(i)
        checks += [
            (p["data"], f"data node {i}"),
            (p["raft"], f"raft node {i}"),
            (p["metrics"], f"data node {i} metrics"),
        ]
    return [f"{label} port {port}" for port, label in checks if _port_open(port)]


def do_start(nodes: int) -> int:
    if not 1 <= nodes <= MAX_NODES:
        print(f"error: --nodes must be between 1 and {MAX_NODES}", file=sys.stderr)
        return 2

    for name in ("statelet-metadata", "statelet-datanode", "statelet-gateway"):
        if not os.path.isfile(binary_path(name)):
            print(f"error: {name} is missing from this installation "
                  f"({binary_path(name)})", file=sys.stderr)
            return 1

    root = data_dir()
    if any(_pid_alive(pid) for pid in _read_pids(root).values()):
        print("A cluster is already running. Stop it first with `statelet-cluster stop`.",
              file=sys.stderr)
        return 1

    busy = _busy_ports(nodes)
    if busy:
        print("error: these ports are already in use:", file=sys.stderr)
        for b in busy:
            print(f"  {b}", file=sys.stderr)
        print("A Statelet cluster started some other way will hold exactly these; "
              "`statelet-cluster stop` only knows about one it started itself.",
              file=sys.stderr)
        return 1

    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "pids").mkdir(parents=True, exist_ok=True)
    log_level = os.environ.get("RUST_LOG", "info")

    base = dict(os.environ)
    meta_addr = f"127.0.0.1:{METADATA_PORT}"

    print(f"Starting Statelet ({nodes} data node{'s' if nodes > 1 else ''}) in {root}")

    _spawn("metadata", [binary_path("statelet-metadata")], {
        **base, "RUST_LOG": log_level, "META_NODE_ID": "1",
        "META_ADDR": meta_addr, "META_PEERS": f"1@{meta_addr}",
        "METRICS_ADDR": "127.0.0.1:9091",
    }, root)
    if not _wait_for_port(METADATA_PORT, 20, "metadata service"):
        print(f"  see {root / 'logs' / 'metadata.log'}", file=sys.stderr)
        return 1

    for i in range(1, nodes + 1):
        p = _node_ports(i)
        node_root = root / f"node{i}"
        node_root.mkdir(parents=True, exist_ok=True)
        _spawn(f"node{i}", [
            binary_path("statelet-datanode"), str(node_root), f"0.0.0.0:{p['data']}",
        ], {
            **base, "RUST_LOG": log_level,
            "META_SERVER": f"http://127.0.0.1:{METADATA_PORT}",
            "DATA_NODE_ID": str(i),
            "DATA_ADDR": f"http://127.0.0.1:{p['data']}",
            "RAFT_ADDR": f"127.0.0.1:{p['raft']}",
            "METRICS_ADDR": f"127.0.0.1:{p['metrics']}",
        }, root)
    _wait_for_port(_node_ports(1)["data"], 20, "data node 1")

    _spawn("gateway", [binary_path("statelet-gateway")], {
        **base, "RUST_LOG": log_level,
        "GATEWAY_ADDR": f"0.0.0.0:{GATEWAY_PORT}",
        "GATEWAY_META": f"http://127.0.0.1:{METADATA_PORT}",
        "GATEWAY_MGMT_ADDR": f"0.0.0.0:{GATEWAY_MGMT_PORT}",
        "GATEWAY_REDIS_ADDR": f"0.0.0.0:{GATEWAY_REDIS_PORT}",
        "GATEWAY_JWT_SECRET": os.environ.get("GATEWAY_JWT_SECRET", "dev-secret"),
        "GATEWAY_USERS": os.environ.get("GATEWAY_USERS", "admin:admin:admin"),
        "METRICS_ADDR": "127.0.0.1:9095",
    }, root)
    if not _wait_for_port(GATEWAY_PORT, 20, "gateway"):
        print(f"  see {root / 'logs' / 'gateway.log'}", file=sys.stderr)
        return 1

    print(f"""
Statelet is up.

  gRPC        127.0.0.1:{GATEWAY_PORT}
  management  http://127.0.0.1:{GATEWAY_MGMT_PORT}
  redis       127.0.0.1:{GATEWAY_REDIS_PORT}
  data        {root}
  logs        {root / 'logs'}

  from statelet import Client
  c = Client("127.0.0.1:{GATEWAY_PORT}", username="admin", password="admin")
  c.put(b"hello", b"world"); print(c.get(b"hello"))

Stop it with `statelet-cluster stop`.""")
    return 0


def do_stop(clean: bool) -> int:
    root = data_dir()
    pids = _read_pids(root)
    if not pids:
        print("No cluster recorded in", root)
    # Gateway first, metadata last: the reverse of startup.
    order = sorted(pids, key=lambda n: {"gateway": 0, "metadata": 2}.get(n, 1))
    for name in order:
        pid = pids[name]
        if _pid_alive(pid):
            print(f"  stopping {name} (pid {pid})")
            _kill(pid)
        (root / "pids" / f"{name}.pid").unlink(missing_ok=True)

    if clean:
        if root.exists():
            shutil.rmtree(root)
            print(f"  removed {root}")
    return 0


def do_status() -> int:
    root = data_dir()
    pids = _read_pids(root)
    if not pids:
        print(f"No cluster recorded in {root}")
        return 1
    print(f"{'process':<12} {'pid':>8}  state")
    any_dead = False
    for name, pid in pids.items():
        alive = _pid_alive(pid)
        any_dead |= not alive
        print(f"{name:<12} {pid:>8}  {'running' if alive else 'not running'}")
    listening = [
        (METADATA_PORT, "metadata"),
        (GATEWAY_PORT, "gateway grpc"),
        (GATEWAY_MGMT_PORT, "gateway mgmt"),
        (GATEWAY_REDIS_PORT, "gateway redis"),
    ]
    print()
    for port, label in listening:
        print(f"  {label:<14} :{port}  {'open' if _port_open(port) else 'closed'}")
    return 1 if any_dead else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="statelet-cluster",
        description="Start, stop or inspect a local Statelet cluster.",
        epilog="STATELET_DATA_DIR overrides the data directory "
               "(default ~/.statelet/cluster); RUST_LOG sets the log level.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_start = sub.add_parser("start", help="start a cluster")
    p_start.add_argument("--nodes", type=int, default=3,
                         help=f"number of data nodes (1-{MAX_NODES}, default 3)")
    p_stop = sub.add_parser("stop", help="stop the cluster")
    p_stop.add_argument("--clean", action="store_true",
                        help="delete the data directory afterwards")
    sub.add_parser("status", help="show process and port status")

    args = parser.parse_args()
    if args.command == "start":
        sys.exit(do_start(args.nodes))
    if args.command == "stop":
        sys.exit(do_stop(args.clean))
    sys.exit(do_status())

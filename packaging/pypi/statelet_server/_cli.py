"""Thin wrappers that exec the bundled Statelet server binaries."""

import os
import subprocess
import sys

_BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")


def binary_path(name: str) -> str:
    """Absolute path to a bundled binary, `.exe` suffix included on Windows."""
    if sys.platform == "win32":
        name += ".exe"
    return os.path.join(_BIN_DIR, name)


def _exec(name: str) -> None:
    binary = binary_path(name)
    if not os.path.isfile(binary):
        print(f"error: binary not found: {binary}", file=sys.stderr)
        print(
            "This wheel was built without server binaries for this platform.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(subprocess.call([binary] + sys.argv[1:]))


def gateway() -> None:
    _exec("statelet-gateway")


def metadata() -> None:
    _exec("statelet-metadata")


def datanode() -> None:
    _exec("statelet-datanode")

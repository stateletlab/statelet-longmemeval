"""Thin wrappers that exec the bundled Statelet server binaries."""

import os
import subprocess
import sys
from typing import Optional

_BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


def ui_dir() -> Optional[str]:
    """The bundled admin UI, or None if this wheel was built without one."""
    return _UI_DIR if os.path.isfile(os.path.join(_UI_DIR, "index.html")) else None


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
    # The gateway serves the admin UI out of GATEWAY_UI_DIR, whose default is
    # the relative path `ui/dist` — which resolves to nothing unless you happen
    # to be standing in an engine checkout. Point it at the copy in this wheel,
    # unless the caller has already chosen one.
    bundled = ui_dir()
    if bundled and not os.environ.get("GATEWAY_UI_DIR"):
        os.environ["GATEWAY_UI_DIR"] = bundled
    _exec("statelet-gateway")


def metadata() -> None:
    _exec("statelet-metadata")


def datanode() -> None:
    _exec("statelet-datanode")

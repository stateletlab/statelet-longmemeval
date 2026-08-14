"""Thin wrapper that execs the bundled statelet-lite binary."""

import os
import subprocess
import sys
from typing import Optional

_BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


def ui_dir() -> Optional[str]:
    """The bundled admin UI, or None if this wheel was built without one."""
    return _UI_DIR if os.path.isfile(os.path.join(_UI_DIR, "index.html")) else None


def binary_path(name: str = "statelet-lite") -> str:
    """Absolute path to the bundled binary, `.exe` suffix included on Windows."""
    if sys.platform == "win32":
        name += ".exe"
    return os.path.join(_BIN_DIR, name)


def main() -> None:
    # The fused gateway serves the admin UI out of GATEWAY_UI_DIR, whose
    # default is resolved relative to the executable — which sits inside
    # site-packages here, not next to a ui/ directory. Point it at the copy in
    # this wheel, unless the caller has already chosen one.
    bundled = ui_dir()
    if bundled and not os.environ.get("GATEWAY_UI_DIR"):
        os.environ["GATEWAY_UI_DIR"] = bundled

    binary = binary_path()
    if not os.path.isfile(binary):
        print(f"error: binary not found: {binary}", file=sys.stderr)
        print(
            "This wheel was built without a statelet-lite binary for this platform.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(subprocess.call([binary] + sys.argv[1:]))

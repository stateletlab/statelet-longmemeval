"""Thin wrapper that execs the bundled statelet-lite binary."""

import glob
import os
import subprocess
import sys
from typing import Optional

_BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")

# The dense embedding model the engine's semantic search expects, in the two
# files it loads (EmbeddingModel::load accepts tokenizer.json + model.onnx at
# the directory root).
_MODEL_NAME = "multilingual-e5-small"
_MODEL_FILES = {
    "tokenizer.json": (
        "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json"
    ),
    "model.onnx": (
        "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx"
    ),
}


def ui_dir() -> Optional[str]:
    """The bundled admin UI, or None if this wheel was built without one."""
    return _UI_DIR if os.path.isfile(os.path.join(_UI_DIR, "index.html")) else None


def binary_path(name: str = "statelet-lite") -> str:
    """Absolute path to the bundled binary, `.exe` suffix included on Windows."""
    if sys.platform == "win32":
        name += ".exe"
    return os.path.join(_BIN_DIR, name)


def model_dir() -> str:
    """Where `--fetch-models` puts the embedding model; the binary looks here."""
    return os.path.join(os.path.expanduser("~"), ".statelet", "models", _MODEL_NAME)


def ort_dylib_path() -> Optional[str]:
    """The ONNX Runtime shared library inside the `onnxruntime` wheel.

    The engine loads libonnxruntime dynamically; pointing ORT_DYLIB_PATH at
    the copy pip already installed saves the user a separate ONNX Runtime
    install.
    """
    try:
        import onnxruntime
    except ImportError:
        return None
    capi = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
    for pattern in ("libonnxruntime*.dylib", "libonnxruntime.so*", "onnxruntime.dll"):
        hits = sorted(glob.glob(os.path.join(capi, pattern)))
        if hits:
            return hits[0]
    return None


def fetch_models() -> str:
    """Download the multilingual-e5-small ONNX model into `model_dir()`."""
    import urllib.request

    dest = model_dir()
    os.makedirs(dest, exist_ok=True)
    for name, url in _MODEL_FILES.items():
        target = os.path.join(dest, name)
        if os.path.isfile(target):
            print(f"already present: {target}")
            continue
        print(f"downloading {url} ...")
        tmp = target + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, target)
        except Exception as e:  # noqa: BLE001 — any failure gets the same cleanup
            if os.path.isfile(tmp):
                os.remove(tmp)
            raise SystemExit(f"error: failed to download {url}: {e}") from e
        print(f"saved {target}")
    print(f"embedding model ready: {dest}")
    print("statelet-lite auto-discovers it there — just restart the server.")
    return dest


def main() -> None:
    args = sys.argv[1:]
    if "--fetch-models" in args:
        fetch_models()
        args = [a for a in args if a != "--fetch-models"]
        if not args:
            return

    # The fused gateway serves the admin UI out of GATEWAY_UI_DIR, whose
    # default is resolved relative to the executable — which sits inside
    # site-packages here, not next to a ui/ directory. Point it at the copy in
    # this wheel, unless the caller has already chosen one.
    bundled = ui_dir()
    if bundled and not os.environ.get("GATEWAY_UI_DIR"):
        os.environ["GATEWAY_UI_DIR"] = bundled

    # Same for ONNX Runtime: unless the caller chose a library, hand the
    # binary the one from the `onnxruntime` wheel this package depends on.
    if not os.environ.get("ORT_DYLIB_PATH"):
        dylib = ort_dylib_path()
        if dylib:
            os.environ["ORT_DYLIB_PATH"] = dylib

    binary = binary_path()
    if not os.path.isfile(binary):
        print(f"error: binary not found: {binary}", file=sys.stderr)
        print(
            "This wheel was built without a statelet-lite binary for this platform.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(subprocess.call([binary] + args))

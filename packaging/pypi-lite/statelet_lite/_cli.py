"""Thin wrapper that execs the bundled statelet-lite binary."""

import glob
import os
import subprocess
import sys
from typing import Optional

_BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
_UI_DIR = os.path.join(os.path.dirname(__file__), "ui")

# The model set the engine loads for its English retrieval pipeline. Each entry
# is downloaded into `<model root>/<dest>` — and `dest` is the directory name
# the ENGINE looks for, which is not always the name of the repo it comes from:
# the published repos carry an `-en` suffix marking them English-pruned, while
# the engine's built-in defaults (CrossEncoderReranker::DEFAULT_MODEL_DIR,
# NliModel::DEFAULT_MODEL_DIR, sparse's MODEL_DIR_NAME) do not. Landing them
# under the engine's names is what lets STATELET_MODEL_ROOT alone wire up the
# whole set with no per-model variables.
#
# `files` lists only what the loaders actually open (model.onnx +
# tokenizer.json at the directory root, plus config.json where the loader reads
# it). fp16/int8/GPU variants are deliberately not fetched: they are opt-in.
_MODEL_ROOT_NAME = os.path.join(".statelet", "models")
_HF = "https://huggingface.co/{repo}/resolve/main/{name}"

_MODELS = (
    # The dense embedder. `dest` keeps the model's own name rather than the
    # engine's `multilingual-e5-small` default, because it is a DIFFERENT model
    # at a different dimension (768, not 384) — naming it e5 would make a
    # mismatched index look like a matching one. main() points
    # STATELET_EMBEDDING_MODEL straight at it instead.
    dict(dest="gte-base-en-v1.5", repo="statelet/gte-base-en-v1.5", embedding=True,
         files=("model.onnx", "tokenizer.json", "config.json", "pooling_config.json")),
    dict(dest="opensearch-neural-sparse-encoding-multilingual-v1",
         repo="statelet/opensearch-neural-sparse-encoding-multilingual-v1-en",
         files=("model.onnx", "tokenizer.json")),
    dict(dest="mmarco-mMiniLMv2", repo="statelet/mmarco-mMiniLMv2-en",
         files=("model.onnx", "tokenizer.json")),
    dict(dest="nli-mdeberta", repo="statelet/nli-mdeberta-en",
         files=("model.onnx", "tokenizer.json", "config.json")),
)


def ui_dir() -> Optional[str]:
    """The bundled admin UI, or None if this wheel was built without one."""
    return _UI_DIR if os.path.isfile(os.path.join(_UI_DIR, "index.html")) else None


def binary_path(name: str = "statelet-lite") -> str:
    """Absolute path to the bundled binary, `.exe` suffix included on Windows."""
    if sys.platform == "win32":
        name += ".exe"
    return os.path.join(_BIN_DIR, name)


def model_root() -> str:
    """The one directory every model is downloaded into.

    Absolute, and derived from `$HOME` rather than the working directory: the
    engine's own `models/` fallbacks are relative to wherever the server was
    started from, so a wrapper that leaves the root implicit makes `statelet-lite`
    behave differently depending on the directory it is launched in.
    """
    return os.path.join(os.path.expanduser("~"), _MODEL_ROOT_NAME)


def model_dir(entry: dict) -> str:
    """Absolute directory one model in `_MODELS` is installed to."""
    return os.path.join(model_root(), entry["dest"])


def embedding_dir() -> str:
    """Absolute directory of the dense embedding model."""
    return model_dir(next(m for m in _MODELS if m.get("embedding")))


def missing_models() -> list:
    """Entries from `_MODELS` that are not fully present on disk."""
    return [
        m for m in _MODELS
        if not all(os.path.isfile(os.path.join(model_dir(m), f)) for f in m["files"])
    ]


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


def _download(url: str, target: str) -> None:
    """Fetch one file to `target`, atomically, with a progress line."""
    import urllib.request

    tmp = target + ".part"
    label = os.path.join(os.path.basename(os.path.dirname(target)), os.path.basename(target))

    def progress(blocks: int, block_size: int, total: int) -> None:
        if not sys.stderr.isatty():
            return
        got = blocks * block_size
        if total > 0:
            pct = min(100, got * 100 // total)
            print(f"\r  {label:58} {pct:3d}%  {got / 1048576:7.1f}/{total / 1048576:.1f} MiB",
                  end="", file=sys.stderr, flush=True)
        else:
            print(f"\r  {label:58}       {got / 1048576:7.1f} MiB",
                  end="", file=sys.stderr, flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        # Only rename once the body is fully written, so an interrupted run
        # leaves a .part behind rather than a truncated model the loader would
        # accept as present and then fail on.
        os.replace(tmp, target)
    except BaseException:
        # BaseException, not Exception: Ctrl-C during a 700 MB download must not
        # leave a half-written .part lying around either.
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    finally:
        if sys.stderr.isatty():
            print(file=sys.stderr)


def fetch_models(entries=None) -> str:
    """Download `entries` (default: everything missing) into `model_root()`."""
    entries = list(_MODELS if entries is None else entries)
    if not entries:
        print(f"all models already present: {model_root()}")
        return model_root()

    print(f"statelet-lite: fetching {len(entries)} model(s) into {model_root()}")
    for entry in entries:
        dest = model_dir(entry)
        os.makedirs(dest, exist_ok=True)
        for name in entry["files"]:
            target = os.path.join(dest, name)
            if os.path.isfile(target):
                continue
            url = _HF.format(repo=entry["repo"], name=name)
            try:
                _download(url, target)
            except KeyboardInterrupt:
                raise SystemExit("\naborted; rerun `statelet-lite --fetch-models` to resume")
            except Exception as e:  # noqa: BLE001 — every failure reports the same way
                raise SystemExit(f"error: failed to download {url}: {e}") from e
    print(f"models ready: {model_root()}")
    return model_root()


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    args = sys.argv[1:]

    forced = "--fetch-models" in args
    if forced:
        args = [a for a in args if a != "--fetch-models"]

    # `--help` / `--version` must stay instant and offline: they are how you
    # inspect a broken install, so they must not be gated behind a 1.4 GB
    # download.
    informational = any(a in ("-h", "--help", "-V", "--version") for a in args)

    missing = missing_models()
    if forced:
        fetch_models(missing)
        if not args:
            return
    elif missing and not informational and not _truthy(os.environ.get("STATELET_NO_FETCH_MODELS")):
        # First run: the engine degrades to keyword-only recall without these,
        # which looks like bad retrieval rather than a missing install, so fetch
        # them once instead of starting up quietly crippled. Announce the size
        # first — this is a surprising amount of network for `statelet-lite`
        # with no arguments. Set STATELET_NO_FETCH_MODELS=1 to skip.
        print(
            f"statelet-lite: {len(missing)} of {len(_MODELS)} models are not installed yet.\n"
            f"  Downloading them once into {model_root()} (~1.4 GB total).\n"
            f"  Set STATELET_NO_FETCH_MODELS=1 to skip and run without semantic search.",
            file=sys.stderr,
        )
        fetch_models(missing)

    # Point the engine at the download location explicitly, as absolute paths.
    # Without this the engine falls back to a `models/` directory resolved
    # against the process working directory, so the same command loads the
    # models from one directory and silently skips them from another.
    os.environ.setdefault("STATELET_MODEL_ROOT", model_root())
    # The embedder does not sit under the name the engine defaults to, so it
    # needs naming outright rather than being found by directory name.
    if not os.environ.get("STATELET_EMBEDDING_MODEL"):
        embedding = embedding_dir()
        if os.path.isdir(embedding):
            os.environ["STATELET_EMBEDDING_MODEL"] = embedding

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
        elif not informational:
            # Reachable when the dependency was skipped (`--no-deps`) or the
            # wheel predates it: the engine then reports the same thing per
            # model, but only after the models have already been downloaded.
            print(
                "statelet-lite: WARNING: no ONNX Runtime library found in this "
                "environment — semantic search will be unavailable. Reinstall with "
                "`pip install --force-reinstall statelet-lite`, which depends on "
                "onnxruntime, or point ORT_DYLIB_PATH at a libonnxruntime.",
                file=sys.stderr,
            )

    binary = binary_path()
    if not os.path.isfile(binary):
        print(f"error: binary not found: {binary}", file=sys.stderr)
        print(
            "This wheel was built without a statelet-lite binary for this platform.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(subprocess.call([binary] + args))

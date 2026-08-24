"""Statelet server binaries, packaged for pip.

The client library lives in the separate `statelet-sdk` distribution and is
imported as `statelet`; this package carries only the `statelet-gateway`,
`statelet-metadata` and `statelet-datanode` executables and the console-script
shims that exec them.
"""

from ._cli import binary_path, fetch_models, model_dir, ort_dylib_path, ui_dir

__all__ = ["binary_path", "fetch_models", "model_dir", "ort_dylib_path", "ui_dir"]

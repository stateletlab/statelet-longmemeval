"""Statelet Lite, packaged for pip.

The whole database in one single-node process: this package carries the
`statelet-lite` executable and its bundled admin UI, and pulls the client
library in as the separate `statelet-sdk` distribution (imported as
`statelet`).
"""

from ._cli import binary_path, fetch_models, model_dir, ort_dylib_path, ui_dir

__all__ = ["binary_path", "fetch_models", "model_dir", "ort_dylib_path", "ui_dir"]

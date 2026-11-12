"""
services/native_compat.py
-------------------------
Stand-ins for native modules that Windows Application Control refuses to load.

spaCy imports srsly, whose bundled ujson and msgpack extensions are .pyd files.
Whisper imports numba, whose _internal extension is also blocked on some PCs.
Both libraries ship a pure-Python path that is enough for Clario:

- srsly JSON calls go through the standard library.
- srsly msgpack uses the pure-Python fallback that already ships with srsly.
- If numba itself cannot be imported, a no-op ``jit`` decorator is installed so
  Whisper's timing helpers run as ordinary NumPy code.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
import types

_INSTALLED = False

_SRSLY_TARGETS = {
    "srsly.ujson.ujson",
    "srsly.msgpack._epoch",
    "srsly.msgpack._packer",
    "srsly.msgpack._unpacker",
    "thinc.layers.premap_ids",
}


class _CompatLoader(importlib.abc.Loader):
    """Load a pure-Python replacement for one blocked extension."""

    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec):  # noqa: ANN001
        return None

    def exec_module(self, module) -> None:  # noqa: ANN001
        if self.fullname == "srsly.ujson.ujson":
            _fill_ujson(module)
        elif self.fullname == "srsly.msgpack._epoch":
            module.utc = None
            module.epoch = None
        elif self.fullname == "srsly.msgpack._packer":
            from srsly.msgpack.fallback import Packer
            module.Packer = Packer
        elif self.fullname == "srsly.msgpack._unpacker":
            from srsly.msgpack.fallback import Unpacker, unpackb
            module.Unpacker = Unpacker
            module.unpackb = unpackb
        elif self.fullname == "thinc.layers.premap_ids":
            module.premap_ids = _premap_ids()


class _CompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        if fullname not in _SRSLY_TARGETS:
            return None
        loader = _CompatLoader(fullname)
        return importlib.machinery.ModuleSpec(fullname, loader)


def _fill_ujson(module) -> None:
    def dumps(obj, indent=0, escape_forward_slashes=False, **kwargs):
        # ujson treats indent=0 as compact output. The stdlib uses None.
        del escape_forward_slashes
        if indent == 0:
            indent = None
        kwargs.pop("encode_html_chars", None)
        return json.dumps(obj, indent=indent, **kwargs)

    module.dumps = dumps
    module.loads = json.loads
    module.dump = json.dump
    module.load = json.load
    module.encode = dumps
    module.decode = json.loads


def _premap_ids():
    """Pure-Python stand-in for thinc's compiled id-remapping layer."""
    import numpy as np
    from thinc.config import registry
    from thinc.model import Model

    @registry.layers("premap_ids.v1")
    def premap_ids(mapping_table, default: int = 0, *, column: int | None = None):
        table = {int(key): int(value) for key, value in mapping_table.items()}
        return Model(
            "premap_ids",
            forward,
            attrs={"mapping_table": table, "default": int(default), "column": column},
        )

    def forward(model, inputs, is_train: bool):
        table = model.attrs["mapping_table"]
        default = model.attrs["default"]
        column = model.attrs["column"]
        idx = inputs if column is None else inputs[:, column]
        if not isinstance(idx, np.ndarray):
            idx = model.ops.to_numpy(idx)
        flat = idx.reshape(-1)
        result = np.fromiter(
            (table.get(int(key), default) for key in flat),
            dtype=np.int64,
            count=flat.size,
        )
        output = model.ops.asarray2i(result)
        output = model.ops.reshape2i(output, -1, 1)

        def backprop(dY):
            return model.ops.xp.empty(dY.shape)

        return output, backprop

    return premap_ids


def _numba_is_usable() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _install_numba_fallback() -> None:
    """Replace a blocked numba import with a decorator that returns the function."""
    for name in list(sys.modules):
        if name == "numba" or name.startswith("numba."):
            del sys.modules[name]

    module = types.ModuleType("numba")

    def jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    module.jit = jit
    module.njit = jit
    sys.modules["numba"] = module


def _install_torchaudio_stub() -> None:
    """Hide torchaudio when Windows Application Control blocks its .pyd.

    Transformers imports torchaudio while loading text models such as Marian.
    A stub keeps that import from aborting Urdu-to-English translation.
    """
    try:
        import torchaudio  # noqa: F401
        return
    except Exception:
        pass
    for name in list(sys.modules):
        if name == "torchaudio" or name.startswith("torchaudio."):
            del sys.modules[name]
    module = types.ModuleType("torchaudio")
    module.__version__ = "0.0.0"
    module.__path__ = []
    module.__package__ = "torchaudio"
    module.__spec__ = importlib.machinery.ModuleSpec(
        "torchaudio",
        loader=None,
        is_package=True,
    )

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("torchaudio native library is blocked on this machine")

    module.load = _unavailable
    module.info = _unavailable
    sys.modules["torchaudio"] = module
    extension = types.ModuleType("torchaudio._extension")
    extension.__package__ = "torchaudio._extension"
    extension.__spec__ = importlib.machinery.ModuleSpec("torchaudio._extension", loader=None)
    extension._IS_TORCHAUDIO_EXT_AVAILABLE = False
    sys.modules["torchaudio._extension"] = extension


def install() -> None:
    """Install the compatibility hooks once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    sys.meta_path.insert(0, _CompatFinder())
    if not _numba_is_usable():
        _install_numba_fallback()
    _install_torchaudio_stub()
    _INSTALLED = True

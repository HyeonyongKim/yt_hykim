"""
Universal loader for yt datasets.

Entry point: load_universal(path, **kwargs)

Strategy:
  1. Try yt.load(path) — reuse existing frontends first.
  2. If yt recognizes it, attach universal field aliases and return.
  3. Inspect the file, find a matching adapter, build IR, convert to yt dataset.
"""

import logging

import yt

from yt_universal.adapters import registry
from yt_universal.builders.stream import build_from_ir
from yt_universal.fields.aliases import attach_universal_aliases
from yt_universal.inspectors.base import inspect_path

mylog = logging.getLogger(__name__)


def load_universal(path, **kwargs):
    """Load a dataset through yt with universal field aliases.

    This is the main entry point for yt_universal. It tries yt's native
    frontend discovery first. If that fails, it inspects the file,
    selects a universal adapter, builds a canonical IR, and converts
    it into a yt stream dataset.

    Parameters
    ----------
    path : str or path-like
        Path to the dataset file or directory.
    **kwargs
        Additional keyword arguments. Recognized keys:
        - hint : str — frontend hint for yt.load()
        - length_unit, mass_unit, time_unit, velocity_unit, magnetic_unit : str
        - bbox : array-like, shape (3,2)
        All other kwargs are passed to yt.load().

    Returns
    -------
    ds : yt Dataset
        The loaded dataset with universal aliases attached.

    Examples
    --------
    >>> from yt_universal import load_universal
    >>> ds = load_universal("snapshot_000.hdf5")
    >>> ad = ds.all_data()
    >>> ad["gas", "density"]  # works regardless of native field names
    """
    # Phase 1: try yt.load() with existing frontends
    hint = kwargs.pop("hint", None)
    yt_load_exc = None
    try:
        if hint is not None:
            ds = yt.load(path, hint=hint, **kwargs)
        else:
            ds = yt.load(path, **kwargs)
        mylog.info("yt_universal: loaded via yt frontend '%s'", type(ds).__name__)
        return attach_universal_aliases(ds)
    except Exception as exc:
        yt_load_exc = exc
        mylog.debug("yt_universal: yt.load() failed: %s", exc)

    # Phase 2: inspect -> adapt -> build from IR
    sig = inspect_path(str(path))

    # If inspection suggests a known yt frontend, retry with hint
    if sig.yt_hint is not None:
        try:
            ds = yt.load(path, hint=sig.yt_hint, **kwargs)
            mylog.info(
                "yt_universal: loaded via yt frontend '%s' (hint=%s)",
                type(ds).__name__,
                sig.yt_hint,
            )
            return attach_universal_aliases(ds)
        except Exception as exc:
            mylog.debug(
                "yt_universal: yt.load(hint=%s) also failed: %s",
                sig.yt_hint,
                exc,
            )

    # Try universal adapters
    adapter = registry.match(sig)
    if adapter is not None:
        # Extract unit/bbox kwargs for the adapter, pass the rest through
        adapter_kwargs = {}
        for key in ("length_unit", "mass_unit", "time_unit",
                     "velocity_unit", "magnetic_unit", "bbox", "grid_dims"):
            if key in kwargs:
                adapter_kwargs[key] = kwargs.pop(key)

        ir = adapter.build_ir(str(path), sig, **adapter_kwargs)
        ds = build_from_ir(ir)
        mylog.info(
            "yt_universal: loaded via adapter '%s'",
            type(adapter).__name__,
        )
        return attach_universal_aliases(ds)

    # No adapter matched — raise with helpful context
    from yt_universal.diagnostics import format_load_failure

    raise RuntimeError(
        format_load_failure(str(path), yt_load_exc, sig)
    ) from yt_load_exc

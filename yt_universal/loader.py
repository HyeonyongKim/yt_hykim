"""
Universal loader for yt datasets.

Entry point: load_universal(path, **kwargs)

Strategy:
  1. Try yt.load(path) — reuse existing frontends first.
  2. If yt recognizes it, attach universal field aliases and return.
  3. (Future phases) Inspect, adapt, and build from canonical IR.
"""

import logging

import yt

from yt_universal.fields.aliases import attach_universal_aliases

mylog = logging.getLogger(__name__)


def load_universal(path, **kwargs):
    """Load a dataset through yt with universal field aliases.

    This is the main entry point for yt_universal. It tries yt's native
    frontend discovery first. If successful, universal field aliases
    (e.g. ("gas", "density"), ("all", "particle_mass")) are attached
    so that downstream analysis code can use a single field vocabulary
    regardless of the simulation format.

    Parameters
    ----------
    path : str or path-like
        Path to the dataset file or directory.
    **kwargs
        Additional keyword arguments passed to yt.load().

    Returns
    -------
    ds : yt Dataset
        The loaded dataset with universal aliases attached.

    Raises
    ------
    YTUnidentifiedDataType
        If yt cannot identify the dataset and no universal adapter
        matches (adapter fallback is not yet implemented).

    Examples
    --------
    >>> from yt_universal import load_universal
    >>> ds = load_universal("snapshot_000.hdf5")
    >>> ad = ds.all_data()
    >>> ad["gas", "density"]  # works regardless of native field names
    """
    # Phase 1: try yt.load() with existing frontends
    hint = kwargs.pop("hint", None)
    try:
        if hint is not None:
            ds = yt.load(path, hint=hint, **kwargs)
        else:
            ds = yt.load(path, **kwargs)
        mylog.info("yt_universal: loaded via yt frontend '%s'", type(ds).__name__)
        return attach_universal_aliases(ds)
    except Exception as exc:
        mylog.debug("yt_universal: yt.load() failed: %s", exc)

    # Phase 2+ (future): inspect -> adapt -> build from IR
    # For now, re-raise so the user gets a clear error
    raise type(exc)(
        f"yt_universal: could not load '{path}'. "
        f"yt.load() failed with: {exc}\n"
        "Universal adapter fallback is not yet implemented (coming in Phase 2+)."
    ) from exc

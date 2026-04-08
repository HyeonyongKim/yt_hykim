"""
Diagnostic utilities for yt_universal.

Provides structured warnings and helpful error messages that guide
users toward fixing issues (e.g., creating a manifest file).
"""

import logging
import warnings

mylog = logging.getLogger(__name__)

_MANIFEST_TEMPLATE = """\
# yt_universal.yaml — place this file next to your data file
# See yt_universal documentation for full specification

format: binary_particle   # or: binary_grid
endianness: little         # or: big

dtype:
  x: float32
  y: float32
  z: float32
  mass: float32

units:
  length_unit: kpc
  mass_unit: Msun

field_map:
  x: particle_position_x
  y: particle_position_y
  z: particle_position_z
  mass: particle_mass

bbox:
  - [0, 1000]
  - [0, 1000]
  - [0, 1000]
"""


def warn_no_units(source_path: str):
    """Emit a structured warning when loading without unit information."""
    warnings.warn(
        f"yt_universal: loading '{source_path}' with no unit information. "
        "Data will be in code units. Pass length_unit, mass_unit, etc. "
        "to load_universal() for physical units.",
        UserWarning,
        stacklevel=4,
    )


def warn_uncertain_classification(source_path: str, family: str, confidence: float):
    """Warn when the dataset classification is low-confidence."""
    if confidence < 0.5:
        warnings.warn(
            f"yt_universal: classification of '{source_path}' as '{family}' "
            f"has low confidence ({confidence:.0%}). Results may be incorrect. "
            "Consider providing a yt_universal.yaml manifest file.",
            UserWarning,
            stacklevel=4,
        )


def format_load_failure(path: str, yt_error, sig) -> str:
    """Format a helpful error message when loading fails."""
    lines = [
        f"yt_universal: could not load '{path}'.",
        "",
        f"  yt.load() error: {yt_error}",
        "",
        "  Inspection results:",
        f"    Container type: {sig.container_type.name}",
        f"    Layout type:    {sig.layout_type.name}",
        f"    Family:         {sig.candidate_family}",
        f"    Confidence:     {sig.confidence:.0%}",
    ]

    if sig.hdf5_root_groups:
        groups_str = ", ".join(sig.hdf5_root_groups[:10])
        if len(sig.hdf5_root_groups) > 10:
            groups_str += f" ... (+{len(sig.hdf5_root_groups) - 10} more)"
        lines.append(f"    Root groups:    {groups_str}")

    lines.extend([
        "",
        "  No universal adapter matched this file.",
        "",
        "  To load this file, create a yt_universal.yaml manifest next to it.",
        "  Template:",
        "",
    ])

    for template_line in _MANIFEST_TEMPLATE.strip().split("\n"):
        lines.append(f"    {template_line}")

    return "\n".join(lines)


def summarize_dataset(ds) -> str:
    """Return a concise summary of a loaded dataset for diagnostics."""
    lines = [
        f"Dataset: {ds}",
        f"  Domain: {ds.domain_left_edge} -> {ds.domain_right_edge}",
    ]

    # Field count
    n_fields = len(ds.field_list)
    n_derived = len(ds.derived_field_list)
    lines.append(f"  Fields: {n_fields} on-disk, {n_derived} derived")

    # Particle types
    if hasattr(ds, "particle_types") and ds.particle_types:
        lines.append(f"  Particle types: {', '.join(ds.particle_types)}")

    return "\n".join(lines)

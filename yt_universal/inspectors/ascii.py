"""
ASCII file inspection: detect columnar tables and structured text data.
"""

import logging
import re
from pathlib import Path

from yt_universal.inspectors.base import DatasetSignature, LayoutType

mylog = logging.getLogger(__name__)


def inspect_ascii(path: str, sig: DatasetSignature):
    """Populate a DatasetSignature with ASCII-specific metadata.

    Reads the first few lines to detect:
    - Column headers
    - Number of columns
    - Whether it looks like particle data or grid data
    """
    try:
        with open(path, "r") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i >= 100:
                    break
    except (OSError, UnicodeDecodeError):
        return

    if not lines:
        return

    # Find the header line and data start
    header_line = None
    data_start = 0
    comment_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!") or stripped.startswith("%"):
            comment_lines.append(stripped)
            data_start = i + 1
            # Check if this comment contains column names
            content = stripped.lstrip("#!% ")
            if _looks_like_header(content):
                header_line = content
        else:
            # First non-comment line
            if header_line is None and _looks_like_header(stripped):
                header_line = stripped
                data_start = i + 1
            else:
                data_start = i
                break

    # Count columns from first data line
    if data_start < len(lines):
        first_data = lines[data_start].strip()
        # Try different delimiters
        for delim in [None, ",", "\t", "|"]:  # None = whitespace
            parts = first_data.split(delim)
            n_cols = len([p for p in parts if p.strip()])
            if n_cols >= 2:
                break
    else:
        n_cols = 0

    # Count data lines (approximate from sample)
    n_data_lines = sum(
        1 for line in lines[data_start:]
        if line.strip() and not line.strip().startswith("#")
    )

    # Classify layout
    if n_cols >= 3:
        # Could be particle data (x, y, z, ...) or grid slices
        sig.layout_type = LayoutType.PARTICLE_ONLY
        sig.candidate_family = "ascii_particle"
        sig.confidence = 0.50
    elif n_cols >= 1:
        sig.candidate_family = "ascii_table"
        sig.confidence = 0.30
    else:
        sig.candidate_family = "ascii_unknown"
        sig.confidence = 0.10

    # Store parsed info in the signature for the adapter
    sig.hdf5_datasets = {}  # Repurpose for column info
    if header_line:
        sig.hdf5_root_groups = _parse_column_names(header_line)
    else:
        sig.hdf5_root_groups = [f"col{i}" for i in range(n_cols)]

    mylog.debug(
        "inspect_ascii: %d columns, %d data lines, header=%s",
        n_cols,
        n_data_lines,
        header_line,
    )


def _looks_like_header(text):
    """Check if a text line looks like column headers (not numeric data)."""
    parts = text.split()
    if not parts:
        return False
    # If most parts are non-numeric, it's likely a header
    non_numeric = 0
    for p in parts:
        p = p.strip(",;|")
        try:
            float(p)
        except ValueError:
            non_numeric += 1
    return non_numeric > len(parts) / 2


def _parse_column_names(header_line):
    """Parse column names from a header line."""
    # Try common delimiters
    for delim in [",", "\t", "|"]:
        if delim in header_line:
            return [c.strip() for c in header_line.split(delim) if c.strip()]
    # Default: whitespace
    return header_line.split()

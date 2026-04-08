"""
Base adapter interface.

All adapters convert a file + DatasetSignature into a DatasetIR.
"""

from abc import ABC, abstractmethod

from yt_universal.inspectors.base import DatasetSignature
from yt_universal.schema.ir import DatasetIR


class BaseAdapter(ABC):
    """Base class for all dataset adapters."""

    @staticmethod
    @abstractmethod
    def can_handle(sig: DatasetSignature) -> bool:
        """Return True if this adapter can handle the given signature."""
        ...

    @staticmethod
    @abstractmethod
    def priority() -> int:
        """Higher priority adapters are tried first. Range 0-100."""
        ...

    @abstractmethod
    def build_ir(self, path: str, sig: DatasetSignature, **kwargs) -> DatasetIR:
        """Parse the file and produce a canonical DatasetIR.

        Parameters
        ----------
        path : str
            Path to the dataset.
        sig : DatasetSignature
            Pre-computed signature from the inspector.
        **kwargs
            Additional user-provided hints (units, bbox, etc.)

        Returns
        -------
        DatasetIR
        """
        ...

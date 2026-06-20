"""
Re-export of kernel.state_space for backward compatibility.

The KernelStateSpace class used to be duplicated here; it has been unified
into kernel/state_space.py. This module preserves the import path
`from generation.target import KernelStateSpace` for older callers, but the
implementation lives in kernel/state_space.py.
"""

from kernel.state_space import KernelStateSpace, FAMILIES, DTYPES

__all__ = ["KernelStateSpace", "FAMILIES", "DTYPES"]

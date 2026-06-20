"""
Re-export of the canonical kernel.state_space.KernelStateSpace.

Historically this file held a separate copy of KernelStateSpace that the
runtime fuzzer loaded via a PYTHONPATH side effect. The class has been
moved into kernel/state_space.py so the runtime fuzzer and evaluation
scripts share a single implementation. This file remains only to keep
the import path `from kernel_state_space import KernelStateSpace`
working for any rq1_bugsindlls script that still relies on it.
"""

# Ensure the package root is on sys.path so `kernel.state_space` resolves
# regardless of how this evaluation script was launched.
import os as _os
import sys as _sys
_PKG_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

from kernel.state_space import KernelStateSpace, FAMILIES, DTYPES

__all__ = ["KernelStateSpace", "FAMILIES", "DTYPES"]

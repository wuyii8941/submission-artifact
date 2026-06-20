"""
Program template registry.

Template families target specific codegen decisions:
  D1: Pointwise-Reduction Fusion
  D2: Epilogue Fusion into GEMM/Conv
  D3: Layout Propagation
  D4: Cross-op Data Dependency
  D5: Shape/Dtype Boundary + Alias/Autograd
  D6+: Extended coverage (scatter, linalg, etc.)
"""
from templates.registry import PROGRAM_TEMPLATES, _TPL_CATEGORIES

"""
Stage 1: Generating Test Programs (paper Section 4.2, Algorithm 1).

Each generation step decides three pieces, then assembles a test program:

Modules:
  ops         -- Operator pools per scheduling family (path entries)
  conditions  -- Boundary condition sampling (shapes, dtypes, context)
  compose    -- Program composition: combines path entries + boundary
                conditions + composition constraints into an executable
                test program
  target      -- Re-export of kernel.state_space.KernelStateSpace (kept
                for backward import compatibility)
"""

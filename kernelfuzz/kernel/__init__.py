"""
Kernel observation (paper Stage 2: Execution and Kernel Observation).

Modules:
  coverage    -- GraphCoverageTracker: 3-level coverage (L1 kernel family,
                 L2 fusion patterns + extern variants, L3 kernel hashes).
                 Sets last_was_novel after each update() — this is the
                 matcher predicate (M_gamma) in paper Algorithm 1.
  state_space -- KernelStateSpace: simpler (family, dtype) coverage report
                 used by the target sampler for under-reached state lookup.
                 The runtime fuzzer.py uses a richer version exposed at the
                 top-level `kernel_state_space` module (TODO: unify).
"""

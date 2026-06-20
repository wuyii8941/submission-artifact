"""
TensorFlow/XLA differential fuzzer.

Instantiates the KernelFuzz methodology for TensorFlow's compilation pipeline:
  - Execution seams: eager vs Grappler, eager vs XLA, CPU vs GPU
  - Oracles: crash, numerical divergence, gradient inconsistency, math invariant
  - Coverage: HLO op/fusion tracking, kernel state space guidance

Modules:
  fuzzer       -- Main fuzzing loop (two-phase: scan + subprocess verify)
  graph_gen    -- Random TF computation graph generator (150+ ops)
  oracles      -- 10 differential oracles
  kernel_state -- Kernel state space and epsilon-greedy scheduler
  coverage     -- HLO/GraphDef coverage tracking and bug basin dedup
  poison       -- Adversarial input generation (11 modes)
"""

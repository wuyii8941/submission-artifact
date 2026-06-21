"""
DeepOpFuzz: compiler-operation-guided fuzzing for deep learning compilers.

Package structure (maps to paper Algorithm 1 stages):

  generation/        -- Stage 1: Generating Test Programs
    target           -- Re-export of kernel.state_space.KernelStateSpace
    ops              -- Operator pools by scheduling family
    compose          -- Program composition: path entries + boundary conditions
                        + composition constraints (paper Section 4.2)
    conditions       -- Boundary condition sampling

  kernel/            -- Stage 2: Execution and Kernel Observation
    coverage         -- GraphCoverageTracker: matcher M_gamma (sets
                        last_was_novel) + 3-level coverage
    state_space      -- KernelStateSpace: (family, dtype) coverage report

  oracle/            -- Stage 3: Oracles and Validation
    (re-exports from bug_result: result classification, divergence checking,
     fresh-process verification, backward gradient comparison)

  templates/         -- Program template registry (D1-D7 template families)
    registry         -- 150+ template functions targeting codegen decisions

  tensorflow/        -- TensorFlow/XLA cross-framework instantiation
    fuzzer           -- Main fuzzing loop (two-phase: scan + subprocess verify)
    graph_gen        -- Random TF computation graph generator (150+ ops)
    oracles          -- 10 differential oracles (eager/Grappler/XLA/CPU-GPU)
    kernel_state     -- TF kernel state space and epsilon-greedy scheduler
    coverage         -- HLO/GraphDef coverage tracking and bug basin dedup
    poison           -- Adversarial input generation (11 modes)

  fuzzer             -- Main fuzzing loop (Algorithm 1)
  program            -- Program representation and mutation
  corpus             -- Corpus management: basin tracking, diversity scheduling
  config             -- Configuration and CLI
  random_generator   -- Fair random baseline (ablation)
  targeted_generator -- Family-targeted program generation (Stage 1 helper)
"""

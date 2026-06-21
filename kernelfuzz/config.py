#!/usr/bin/env python3
"""
DeepOpFuzz configuration. No hardcoded paths.
"""

import argparse
import os


# Environment — override via CLI or env vars, never hardcode
NIGHTLY_ENV = os.environ.get("KERNELFUZZ_NIGHTLY_ENV", "")
OUTPUT_DIR = os.environ.get("KERNELFUZZ_OUTPUT", "./kernelfuzz_output")
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Fuzzing parameters
DEFAULT_MAX_ITERATIONS = 50000
DEFAULT_TIMEOUT_MINUTES = 300
DEFAULT_COMPILE_TIMEOUT = 60

# Oracle thresholds
FP32_ATOL = 1e-3
FP64_ATOL = 1e-6
FP64_BACKOFF_THRESHOLD = 1e-6
SUSPICIOUS_THRESHOLD = 10.0

# Coverage parameters
KSS_TOTAL_STATES = 46
WARMUP_ITERATIONS = 50

# Oracle sampling rates
BACKWARD_ORACLE_RATE = 0.25
DOUBLE_BACKWARD_RATE = 0.15   # raised from 0.05 to improve bug_019 coverage
META_STRIDE_RATE = 0.10
GUARD_ORACLE_RATE = 0.08
FINITE_DIFF_RATE = 0.03
CONFIG_VS_CONFIG_RATE = 0.10
AGGRESSIVE_SCHEDULE_RATE = 0.05
DYNAMIC_MULTI_SHAPE_RATE = 0.05
AOTI_RATE = 0.08


def setup_environment():
    """Set up paths from config, not hardcoded."""
    if NIGHTLY_ENV and os.path.isdir(NIGHTLY_ENV):
        env_bin = os.path.join(NIGHTLY_ENV, "bin")
        env_lib = os.path.join(NIGHTLY_ENV, "lib")
        if env_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = env_bin + ":" + os.environ.get("PATH", "")
        if env_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = env_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def get_parser():
    """CLI argument parser for DeepOpFuzz."""
    p = argparse.ArgumentParser(description="DeepOpFuzz: Kernel-space guided fuzzer")
    p.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MINUTES,
                   help="Wall-clock timeout in minutes")
    p.add_argument("--compile-timeout", type=int, default=DEFAULT_COMPILE_TIMEOUT,
                   help="Per-compilation timeout in seconds")
    p.add_argument("--nightly-env", default=NIGHTLY_ENV,
                   help="Path to PyTorch nightly environment")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return p

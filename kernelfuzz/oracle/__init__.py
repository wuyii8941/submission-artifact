"""
Oracle pipeline (paper Section 3.3).

Three oracle classes:
  Crash     -- compilation failures, runtime exceptions, assertion failures
  Differential -- eager vs compiled, cross-config, backward gradient comparison
  Metamorphic  -- property (P), equivalence (E), metamorphic (M), gradient (G)

Implementation notes:
  - Result classification and divergence checking: oracle.result
  - Oracle execution orchestration: embedded in fuzzer.KernelFuzzer
  - Bug verification (fresh-process re-run): oracle.result._verify_bug_in_fresh_process
  - Config preset sampling: oracle.result._sample_config_preset
"""

from bug_result import (
    # Result types
    GraphBugResult,
    GraphBugResultV2,
    DivergenceLevel,
    # Analysis functions
    analyze_graph_results,
    analyze_graph_results_v2,
    check_divergence,
    # Verification
    _verify_bug_in_fresh_process,
    _run_backward_verification,
    # Config
    _sample_config_preset,
    _build_config_injection_code,
    # I/O
    _write_bug_file,
    _write_bug_file_v2,
    _save_crash_report,
    # Constants
    GRAPH_DIFF_BOUNDS,
)

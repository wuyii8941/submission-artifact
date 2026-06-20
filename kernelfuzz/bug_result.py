#!/usr/bin/env python3
"""
Bug result classification and reporting.

Divergence levels:
  SUCCESS: within tolerance
  SUSPICIOUS: above 10x tolerance but ≤ 10.0
  CATASTROPHIC: above 10.0 max diff or >1% violation ratio
"""

import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
import traceback
from enum import Enum
from os.path import join

import torch

from kernel.coverage import _is_unsupported_err, _is_env_error

import sys as _sys
_FREEFUZZ_SRC = os.environ.get(
    "KERNELFUZZ_FREEFUZZ_SRC",
    os.path.join(os.path.dirname(__file__), "..", "FreeFuzz-baseline", "kernel-guided", "src"),
)
if os.path.isdir(_FREEFUZZ_SRC) and _FREEFUZZ_SRC not in _sys.path:
    _sys.path.insert(0, _FREEFUZZ_SRC)
from classes.torch_library import TorchLibrary

GRAPH_DIFF_BOUNDS = {"torch.float32": 1e-3, "torch.float64": 1e-6}

class GraphBugResult:
    def __init__(self):
        self.is_bug=False; self.bug_type=None; self.priority=0
        self.trigger_config=None; self.error_msg=None; self.max_diff=None
        self.category=None  # D1/D2/D3/D4


def _diff_tensors(t1, t2, atol):
    """Compare tensors using TorchLibrary.is_equal() and compute max_diff."""
    try:
        equal = TorchLibrary.is_equal(t1, t2, diff_bound=atol)
    except Exception:
        equal = False
    max_diff = None
    try:
        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
            if t1.shape == t2.shape:
                c1, c2 = t1.detach().cpu().float(), t2.detach().cpu().float()
                if c1.is_floating_point():
                    max_diff = float((c1 - c2).abs().max())
        elif isinstance(t1, (list, tuple)) and isinstance(t2, (list, tuple)):
            if len(t1) != len(t2):
                return False, float('inf')
            ds = [_diff_tensors(a, b, atol)[1] for a, b in zip(t1, t2)]
            max_diff = max((d for d in ds if d is not None), default=None)
    except Exception:
        pass
    return equal, max_diff


# ---------------------------------------------------------------------------
# AOTI environment error detector
# Filters out system-level AOTI failures (libstdc++ / GLIBCXX / dlopen
# version mismatches) that are not real bugs but operational issues.
# ---------------------------------------------------------------------------

_AOTI_ENV_ERROR_MARKERS = (
    "GLIBCXX_",
    "libstdc++",
    "Error in dlopen",
    "version `GLIBC",
    "GLIBC_",
    "libtorch_python.so",
    "cannot allocate memory in static TLS block",
    "CppCompileError: C++ compile error",
    "No space left on device",
    "CUDA out of memory",
)


def _is_aoti_env_error(err_msg):
    """True if the AOTI failure is a system/env issue, not a real compiler bug."""
    if not err_msg:
        return False
    return any(marker in err_msg for marker in _AOTI_ENV_ERROR_MARKERS)


# ---------------------------------------------------------------------------
# Fresh-process bug verifier
# Re-runs a reported diff bug in a clean child process to filter out
# long-process state-pollution false positives. Crash bugs are returned
# as verified=True immediately because crash failure modes are deterministic.
# ---------------------------------------------------------------------------

# Diff-class bug types that need fresh-process verification.
# backward_only and backward_amplified are also verified:
# the verify driver sets requires_grad, runs backward with a fixed seed grad,
# and compares gradients instead of forward outputs.
_VERIFY_DIFF_BUG_TYPES = (
    "frontend_diff", "inductor_diff", "compile_diff", "aoti_diff",
    "backward_only", "backward_amplified", "recompile_diff",
    "cpu_inductor_diff",
    "cross_fusion_diff", "dynamic_reuse_diff",
    "functionalize_diff", "guard_diff", "finite_diff_grad",
    "dynamic_multi_shape_diff", "aggressive_schedule_diff",
    "cudagraph_replay_diff",
    # Note: meta_stride_diff is NOT here — it runs via standalone TorchLibrary.run_code
    # and doesn't use program body_code, so fresh-process verify is N/A.
)


def _verify_bug_in_fresh_process(bug, program, timeout_sec=240):
    """Re-run the bug's compile path in a fresh subprocess and confirm the
    diff signal still fires. Returns (verified: bool, reason: str).

    Default timeout is 240s because cold CUDA context init alone takes ~65s
    on some systems (sm_75 + driver-bound), so a tight timeout would
    spuriously mark every diff bug as "verify_timeout"."""
    if not getattr(bug, "is_bug", False):
        return True, "not_a_bug"
    bug_type = getattr(bug, "bug_type", None)
    if bug_type not in _VERIFY_DIFF_BUG_TYPES:
        return True, f"crash_class_skipped:{bug_type}"

    import os, sys, subprocess, tempfile

    body = getattr(program, "body_code", "") or ""
    if not body.strip():
        return True, "no_body_code"
    cfg_preset = getattr(program, "config_preset", "default")
    trigger = getattr(bug, "trigger_config", "") or ""
    cfg_lbl = trigger[len("inductor_"):] if trigger.startswith("inductor_") else trigger

    # Build config injection code early — needed by all paths
    cfg_inject_lines = _build_config_injection_code(cfg_preset)
    cfg_inject = "\n".join(cfg_inject_lines)
    atol = GRAPH_DIFF_BOUNDS.get(getattr(program, "dtype", "torch.float32"), 1e-3)

    # Backward verify: compare eager vs inductor gradients in child process
    if bug_type in ("backward_only", "backward_amplified"):
        bwd_driver = (
            "import sys\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "torch.cuda.manual_seed(0)\n"
            f"{cfg_inject}\n"
            f"{body}\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "# Set requires_grad on float inputs\n"
            "_program_inputs = tuple(\n"
            "    t.requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t\n"
            "    for t in _program_inputs)\n"
            "# Eager backward\n"
            "try:\n"
            "    out_e = program(*_program_inputs)\n"
            "    if not (isinstance(out_e, torch.Tensor) and out_e.requires_grad):\n"
            "        print('VERIFY:not_differentiable')\n"
            "        sys.exit(0)\n"
            "    _grad = torch.randn_like(out_e)\n"
            "    _grad.manual_seed = None  # ensure same grad\n"
            "    out_e.backward(_grad)\n"
            "    _eager_grads = [p.grad.clone() for p in _program_inputs\n"
            "                    if isinstance(p, torch.Tensor) and p.grad is not None]\n"
            "except Exception as e:\n"
            "    print('VERIFY:eager_fail:' + type(e).__name__)\n"
            "    sys.exit(0)\n"
            "# Inductor backward\n"
            "try:\n"
            "    torch._dynamo.reset()\n"
            "    _bwd_inputs = tuple(\n"
            "        t.detach().clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else\n"
            "        (t.clone() if isinstance(t, torch.Tensor) else t)\n"
            "        for t in _program_inputs)\n"
            "    _ifn = torch.compile(program, backend='inductor')\n"
            "    out_i = _ifn(*_bwd_inputs)\n"
            "    out_i.backward(_grad)\n"
            "    _ind_grads = [p.grad.clone() for p in _bwd_inputs\n"
            "                  if isinstance(p, torch.Tensor) and p.grad is not None]\n"
            "except Exception as e:\n"
            "    print('VERIFY:compile_fail:' + type(e).__name__)\n"
            "    sys.exit(0)\n"
            "finally:\n"
            "    torch._dynamo.reset()\n"
            "# Compare gradients\n"
            "if len(_eager_grads) != len(_ind_grads):\n"
            "    print('VERIFY:diff:inf')\n"
            "    sys.exit(0)\n"
            "max_gd = 0.0\n"
            "for eg, ig in zip(_eager_grads, _ind_grads):\n"
            "    if eg.shape != ig.shape:\n"
            "        max_gd = float('inf')\n"
            "        break\n"
            "    gd = float((eg.detach().cpu().float() - ig.detach().cpu().float()).abs().max())\n"
            "    max_gd = max(max_gd, gd)\n"
            "print('VERIFY:diff:%.6e' % max_gd)\n"
        )
        tmp_dir = "${ARTIFACT_ROOT}/graph_exp/tmp_verify"
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except OSError:
            pass
        fd, path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
        try:
            os.write(fd, bwd_driver.encode("utf-8"))
            os.close(fd)
            try:
                res = subprocess.run(
                    [sys.executable, path],
                    capture_output=True, timeout=timeout_sec, text=True,
                )
            except subprocess.TimeoutExpired:
                return False, "verify_timeout_backward"
            out_lines = (res.stdout or "").strip().splitlines()
            verify_line = next((l for l in reversed(out_lines) if l.startswith("VERIFY:")), None)
            if verify_line is None:
                err_tail = (res.stderr or "")[-200:]
                return False, f"no_verify_line_bwd_rc{res.returncode}:{err_tail}"
            if verify_line.startswith("VERIFY:eager_fail"):
                return False, "child_eager_failed_bwd"
            if verify_line.startswith("VERIFY:compile_fail"):
                return False, "child_compile_crashed_bwd"
            if verify_line.startswith("VERIFY:not_differentiable"):
                return False, "not_differentiable"
            if verify_line.startswith("VERIFY:diff:"):
                try:
                    child_diff = float(verify_line.split(":", 2)[2])
                except (ValueError, IndexError):
                    return False, "verify_parse_error_bwd"
                parent_diff = getattr(bug, "max_diff", 0.0) or 0.0
                # Backward threshold is 100×atol (not 10×) because backward
                # chains naturally amplify fp32 noise — a 5-layer matmul
                # backward can produce ~0.01-0.02 diff that's normal.
                # Real backward bugs (bug_007: grad=0) have diff >> 0.1.
                if child_diff > atol * 100 and child_diff >= parent_diff * 0.3:
                    return True, f"verified_bwd:child_diff={child_diff:.2e}"
                return False, f"diff_too_small_bwd:child={child_diff:.2e}_parent={parent_diff:.2e}"
            return False, f"unknown_verify_line_bwd:{verify_line[:80]}"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # AOTI verify: export + compile + load + compare in the child process.
    if cfg_lbl == "aoti":
        aoti_driver = (
            "import os as _os\n"
            "import sys\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            f"{cfg_inject}\n"
            f"{body}\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "# Preload conda libstdc++ for AOTI dlopen\n"
            "try:\n"
            "    import ctypes as _ctl\n"
            "    _conda = _os.environ.get('CONDA_PREFIX', '')\n"
            "    _ctl.CDLL(_os.path.join(_conda, 'lib', 'libstdc++.so.6') if _conda else 'libstdc++.so.6', mode=_ctl.RTLD_GLOBAL)\n"
            "except OSError:\n"
            "    pass\n"
            "try:\n"
            "    out_eager = program(*_program_inputs)\n"
            "except Exception as e:\n"
            "    print('VERIFY:eager_fail:' + type(e).__name__)\n"
            "    sys.exit(0)\n"
            "try:\n"
            "    torch._dynamo.reset()\n"
            "    class _PM(torch.nn.Module):\n"
            "        def forward(self, *args):\n"
            "            return program(*args)\n"
            "    _ep = torch.export.export(_PM(), _program_inputs)\n"
            "    import torch._inductor as _ind_v\n"
            "    _path = _ind_v.aoti_compile_and_package(_ep, package_path='${ARTIFACT_ROOT}/graph_exp/tmp_aoti/verify_aoti.pt2')\n"
            "    _loaded = _ind_v.aoti_load_package(_path)\n"
            "    out_aoti = _loaded(*_program_inputs)\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.synchronize()\n"
            "    import os\n"
            "    if os.path.exists(_path): os.remove(_path)\n"
            "except Exception as e:\n"
            "    print('VERIFY:compile_fail:' + type(e).__name__)\n"
            "    sys.exit(0)\n"
            "def _diff(a, b):\n"
            "    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):\n"
            "        if a.shape != b.shape:\n"
            "            return float('inf')\n"
            "        c1 = a.detach().cpu().float()\n"
            "        c2 = b.detach().cpu().float()\n"
            "        return float((c1 - c2).abs().max())\n"
            "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
            "        if len(a) != len(b):\n"
            "            return float('inf')\n"
            "        return max((_diff(x, y) for x, y in zip(a, b)), default=0.0)\n"
            "    return float('inf')\n"
            "md = _diff(out_eager, out_aoti)\n"
            "print('VERIFY:diff:%.6e' % md)\n"
        )
        # Override the main driver string for AOTI path
        driver = aoti_driver
        tmp_dir = "${ARTIFACT_ROOT}/graph_exp/tmp_verify"
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except OSError:
            pass
        fd, path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
        try:
            os.write(fd, driver.encode("utf-8"))
            os.close(fd)
            try:
                res = subprocess.run(
                    [sys.executable, path],
                    capture_output=True, timeout=timeout_sec, text=True,
                    env={**os.environ,
                         "LD_LIBRARY_PATH": os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib") + ":"
                         + os.environ.get("LD_LIBRARY_PATH", "")},
                )
            except subprocess.TimeoutExpired:
                return False, "verify_timeout_aoti"
            out_lines = (res.stdout or "").strip().splitlines()
            verify_line = next((l for l in reversed(out_lines) if l.startswith("VERIFY:")), None)
            if verify_line is None:
                err_tail = (res.stderr or "")[-200:]
                return False, f"no_verify_line_aoti_rc{res.returncode}:{err_tail}"
            if verify_line.startswith("VERIFY:eager_fail"):
                return False, "child_eager_failed_aoti"
            if verify_line.startswith("VERIFY:compile_fail"):
                return False, "child_aoti_compile_crashed"
            if verify_line.startswith("VERIFY:diff:"):
                try:
                    child_diff = float(verify_line.split(":", 2)[2])
                except (ValueError, IndexError):
                    return False, "verify_parse_error_aoti"
                parent_diff = getattr(bug, "max_diff", 0.0) or 0.0
                if child_diff > atol * 10 and child_diff >= parent_diff * 0.5:
                    return True, f"verified_aoti:child_diff={child_diff:.2e}"
                return False, f"diff_too_small_aoti:child={child_diff:.2e}_parent={parent_diff:.2e}"
            return False, f"unknown_verify_line_aoti:{verify_line[:80]}"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # CPU inductor verify: run on CPU only, no CUDA
    if trigger == "cpu_inductor":
        cpu_driver = (
            "import sys\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            f"{cfg_inject}\n"
            f"{body}\n"
            "# Keep inputs on CPU\n"
            "try:\n"
            "    out_eager = program(*_program_inputs)\n"
            "except Exception as e:\n"
            "    print('VERIFY:eager_fail:' + type(e).__name__)\n"
            "    sys.exit(0)\n"
            "try:\n"
            "    torch._dynamo.reset()\n"
            "    fn = torch.compile(program, backend='inductor')\n"
            "    _inp_clone = tuple(t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "    out_compiled = fn(*_inp_clone)\n"
            "except Exception as e:\n"
            "    print('VERIFY:compile_fail:' + type(e).__name__)\n"
            "    sys.exit(0)\n"
            "def _diff(a, b):\n"
            "    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):\n"
            "        if a.shape != b.shape:\n"
            "            return float('inf')\n"
            "        c1 = a.detach().float()\n"
            "        c2 = b.detach().float()\n"
            "        return float((c1 - c2).abs().max())\n"
            "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
            "        if len(a) != len(b):\n"
            "            return float('inf')\n"
            "        return max((_diff(x, y) for x, y in zip(a, b)), default=0.0)\n"
            "    return float('inf')\n"
            "md = _diff(out_eager, out_compiled)\n"
            "print('VERIFY:diff:%.6e' % md)\n"
        )
        tmp_dir = "${ARTIFACT_ROOT}/graph_exp/tmp_verify"
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except OSError:
            pass
        fd, path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
        try:
            os.write(fd, cpu_driver.encode("utf-8"))
            os.close(fd)
            try:
                res = subprocess.run(
                    [sys.executable, path],
                    capture_output=True, timeout=timeout_sec, text=True,
                )
            except subprocess.TimeoutExpired:
                return False, "verify_timeout_cpu"
            out_lines = (res.stdout or "").strip().splitlines()
            verify_line = next((l for l in reversed(out_lines) if l.startswith("VERIFY:")), None)
            if verify_line is None:
                err_tail = (res.stderr or "")[-200:]
                return False, f"no_verify_line_cpu_rc{res.returncode}:{err_tail}"
            if verify_line.startswith("VERIFY:eager_fail"):
                return False, "child_eager_failed_cpu"
            if verify_line.startswith("VERIFY:compile_fail"):
                return False, "child_cpu_compile_crashed"
            if verify_line.startswith("VERIFY:diff:"):
                try:
                    child_diff = float(verify_line.split(":", 2)[2])
                except (ValueError, IndexError):
                    return False, "verify_parse_error_cpu"
                parent_diff = getattr(bug, "max_diff", 0.0) or 0.0
                if child_diff > atol * 10 and child_diff >= parent_diff * 0.5:
                    return True, f"verified_cpu:child_diff={child_diff:.2e}"
                return False, f"diff_too_small_cpu:child={child_diff:.2e}_parent={parent_diff:.2e}"
            return False, f"unknown_verify_line_cpu:{verify_line[:80]}"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    if cfg_lbl == "aot_eager":
        compile_call = "torch.compile(program, backend='aot_eager', fullgraph=False)"
    elif cfg_lbl == "default":
        compile_call = "torch.compile(program, backend='inductor')"
    elif cfg_lbl == "reduce_overhead":
        compile_call = "torch.compile(program, backend='inductor', mode='reduce-overhead')"
    elif cfg_lbl == "fullgraph":
        compile_call = "torch.compile(program, backend='inductor', fullgraph=True)"
    elif cfg_lbl == "dynamic":
        compile_call = "torch.compile(program, backend='inductor', dynamic=True)"
    else:
        compile_call = "torch.compile(program, backend='inductor')"

    driver = (
        "import sys\n"
        "import torch\n"
        "torch.manual_seed(0)\n"
        f"{cfg_inject}\n"
        f"{body}\n"
        "if torch.cuda.is_available():\n"
        "    _program_inputs = tuple(\n"
        "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
        "try:\n"
        "    out_eager = program(*_program_inputs)\n"
        "except Exception as e:\n"
        "    print('VERIFY:eager_fail:' + type(e).__name__)\n"
        "    sys.exit(0)\n"
        "try:\n"
        "    torch._dynamo.reset()\n"
        f"    fn = {compile_call}\n"
        "    _inp_clone = tuple(t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
        "    out_compiled = fn(*_inp_clone)\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.synchronize()\n"
        "except Exception as e:\n"
        "    print('VERIFY:compile_fail:' + type(e).__name__)\n"
        "    sys.exit(0)\n"
        "def _diff(a, b):\n"
        "    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):\n"
        "        if a.shape != b.shape:\n"
        "            return float('inf')\n"
        "        c1 = a.detach().cpu().float()\n"
        "        c2 = b.detach().cpu().float()\n"
        "        return float((c1 - c2).abs().max())\n"
        "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
        "        if len(a) != len(b):\n"
        "            return float('inf')\n"
        "        return max((_diff(x, y) for x, y in zip(a, b)), default=0.0)\n"
        "    return float('inf')\n"
        "md = _diff(out_eager, out_compiled)\n"
        "print('VERIFY:diff:%.6e' % md)\n"
    )

    tmp_dir = "${ARTIFACT_ROOT}/graph_exp/tmp_verify"
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except OSError:
        pass
    fd, path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
    try:
        os.write(fd, driver.encode("utf-8"))
        os.close(fd)
        try:
            res = subprocess.run(
                [sys.executable, path],
                capture_output=True, timeout=timeout_sec, text=True,
            )
        except subprocess.TimeoutExpired:
            return False, "verify_timeout"
        out_lines = (res.stdout or "").strip().splitlines()
        verify_line = next((l for l in reversed(out_lines) if l.startswith("VERIFY:")), None)
        if verify_line is None:
            err_tail = (res.stderr or "")[-200:]
            return False, f"no_verify_line_rc{res.returncode}:{err_tail}"
        if verify_line.startswith("VERIFY:eager_fail"):
            return False, "child_eager_failed"
        if verify_line.startswith("VERIFY:compile_fail"):
            # If parent saw a diff but child compile crashes, signature shifted
            # → not the same FP class, but also not a clean reproduction.
            return False, "child_compile_crashed"
        if verify_line.startswith("VERIFY:diff:"):
            try:
                child_diff = float(verify_line.split(":", 2)[2])
            except (ValueError, IndexError):
                return False, "verify_parse_error"
            parent_diff = getattr(bug, "max_diff", 0.0) or 0.0
            # Verified iff child diff exceeds 10×atol AND is at least half the
            # parent diff. The 0.5× factor allows some inter-run jitter while
            # still catching the d1_logcumsumexp class (parent diff ~5, child
            # diff ~0).
            if child_diff > atol * 10 and child_diff >= parent_diff * 0.5:
                return True, f"verified:child_diff={child_diff:.2e}"
            return False, f"diff_too_small:child={child_diff:.2e}_parent={parent_diff:.2e}"
        return False, f"unknown_verify_line:{verify_line[:80]}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def analyze_graph_results(results, program, outer_error):
    """Three-way oracle: eager vs aot_eager vs inductor (multiple configs).

    When program.has_poison is True, only crash/compile errors are reported
    (no numerical comparison) to avoid NaN/Inf false positive diffs.
    """
    bug  = GraphBugResult()
    atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)

    if outer_error and TorchLibrary.is_crash_msg(outer_error): return bug
    if not results.get("eager_ok", False): return bug

    eager_out = results.get("eager")

    # aot_eager crash -> frontend_crash -- always report crashes
    if results.get("aot_eager_ok") is False:
        err = results.get("aot_eager_error", "")
        if err and TorchLibrary.is_crash_msg(err) and not _is_unsupported_err(err):
            bug.is_bug=True; bug.bug_type="frontend_crash"; bug.priority=2
            bug.error_msg=err; bug.trigger_config="aot_eager"; bug.category=program.category
            return bug

    # aot_eager diff -> frontend_diff -- skip if poisoned
    aot_out = results.get("aot_eager")
    if (not program.has_poison
            and aot_out is not None and eager_out is not None
            and program.is_deterministic):
        is_eq, max_d = _diff_tensors(eager_out, aot_out, atol)
        if not is_eq:
            bug.is_bug=True; bug.bug_type="frontend_diff"; bug.priority=2
            bug.max_diff=max_d; bug.trigger_config="aot_eager"; bug.category=program.category
            bug.error_msg=f"eager vs aot_eager max_diff={max_d}"
            return bug

    # inductor configs
    for cfg_lbl, cfg_res in results.get("inductor_results", {}).items():
        # Crash/compile errors — always report
        if not cfg_res.get("ok", False):
            err = cfg_res.get("error", "")
            if not err or _is_unsupported_err(err): continue
            bug.is_bug=True; bug.bug_type="backend_crash"; bug.category=program.category
            bug.priority=3 if TorchLibrary.is_crash_msg(err) else 2
            bug.error_msg=err; bug.trigger_config=f"inductor_{cfg_lbl}"
            return bug

        # Diff checks — skip if poisoned (NaN/Inf causes false positive diffs)
        if program.has_poison: continue
        if not program.is_deterministic: continue
        ind_out = cfg_res.get("result")
        if ind_out is None or eager_out is None: continue
        is_eq_eager, max_d = _diff_tensors(eager_out, ind_out, atol)
        if is_eq_eager: continue

        is_inductor_specific = False
        if aot_out is not None:
            is_eq_aot, _ = _diff_tensors(aot_out, ind_out, atol)
            is_inductor_specific = not is_eq_aot

        bug.is_bug=True
        bug.bug_type="inductor_diff" if is_inductor_specific else "compile_diff"
        bug.priority=3 if is_inductor_specific else 1
        bug.max_diff=max_d; bug.trigger_config=f"inductor_{cfg_lbl}"
        bug.category=program.category
        bug.error_msg=(f"eager vs inductor({cfg_lbl}) max_diff={max_d:.2e} "
                       f"[{'inductor-specific' if is_inductor_specific else 'aot-level'}]")
        return bug
    return bug


# ---------------------------------------------------------------------------
# Graded divergence detection oracle
# ---------------------------------------------------------------------------

class DivergenceLevel:
    """Three-level divergence classification."""
    SUCCESS      = "SUCCESS"       # within tolerance
    SUSPICIOUS   = "SUSPICIOUS"    # above base threshold, needs further verification
    CATASTROPHIC = "CATASTROPHIC"  # extreme divergence, almost certainly a bug


def _check_divergence_tensor(t1, t2, base_atol):
    """Graded divergence check for a single tensor pair.

    Returns (level, max_abs, violation_ratio).
    """
    try:
        c1 = t1.detach().cpu().float()
        c2 = t2.detach().cpu().float()
        if c1.shape != c2.shape:
            return DivergenceLevel.CATASTROPHIC, float("inf"), 1.0

        abs_diff = (c1 - c2).abs()
        max_abs = float(abs_diff.max())

        # fraction of elements exceeding 10x atol
        numel = max(c1.numel(), 1)
        violation_ratio = float((abs_diff > base_atol * 10).sum()) / numel

        # CATASTROPHIC: single-point disaster (>10.0) or widespread (>1% elements exceed 10x atol)
        if max_abs > 10.0 or violation_ratio > 0.01:
            return DivergenceLevel.CATASTROPHIC, max_abs, violation_ratio

        # SUSPICIOUS: above base threshold but not catastrophic
        if max_abs > base_atol * 10:
            return DivergenceLevel.SUSPICIOUS, max_abs, violation_ratio

        return DivergenceLevel.SUCCESS, max_abs, violation_ratio
    except Exception:
        return DivergenceLevel.CATASTROPHIC, float("inf"), 1.0


def check_divergence(t1, t2, dtype):
    """Graded divergence check for tensor / list[tensor] / tuple[tensor].

    Returns (level, max_abs, violation_ratio).
    """
    base_atol = GRAPH_DIFF_BOUNDS.get(dtype, 1e-3)

    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return _check_divergence_tensor(t1, t2, base_atol)

    if isinstance(t1, (list, tuple)) and isinstance(t2, (list, tuple)):
        if len(t1) != len(t2):
            return DivergenceLevel.CATASTROPHIC, float("inf"), 1.0
        worst_level = DivergenceLevel.SUCCESS
        worst_abs = 0.0
        worst_vr = 0.0
        for a, b in zip(t1, t2):
            lv, ma, vr = check_divergence(a, b, dtype)
            if ma > worst_abs:
                worst_abs = ma
            if vr > worst_vr:
                worst_vr = vr
            if lv == DivergenceLevel.CATASTROPHIC:
                worst_level = DivergenceLevel.CATASTROPHIC
            elif lv == DivergenceLevel.SUSPICIOUS and worst_level == DivergenceLevel.SUCCESS:
                worst_level = DivergenceLevel.SUSPICIOUS
        return worst_level, worst_abs, worst_vr

    # type mismatch
    return DivergenceLevel.CATASTROPHIC, float("inf"), 1.0


class GraphBugResultV2:
    """Bug result with divergence_level and violation_ratio fields."""
    def __init__(self):
        self.is_bug = False
        self.bug_type = None        # frontend_crash / frontend_diff / backend_crash /
                                    # inductor_diff / compile_diff / dynamic_crash /
                                    # backward_amplified
        self.priority = 0           # 3=FIRE, 2=BUG, 1=LOW
        self.trigger_config = None
        self.error_msg = None
        self.max_diff = None
        self.violation_ratio = None
        self.divergence_level = DivergenceLevel.SUCCESS  # CATASTROPHIC / SUSPICIOUS / SUCCESS
        self.category = None        # D1/D2/D3/D4/D5
        self.needs_backward = False  # True = triggers backward verification


def analyze_graph_results_v2(results, program, outer_error):
    """Three-way oracle with graded divergence detection.

    Differences from analyze_graph_results:
      1. Uses check_divergence() instead of _diff_tensors() for graded detection
      2. CATASTROPHIC -> is_bug=True (high confidence)
      3. SUSPICIOUS -> is_bug=True with needs_backward=True (pending backward verification)
      4. Supports dynamic config crash-only detection
    """
    bug = GraphBugResultV2()

    if outer_error and TorchLibrary.is_crash_msg(outer_error):
        return bug
    if not results.get("eager_ok", False):
        return bug

    eager_out = results.get("eager")

    # --- aot_eager crash -> frontend_crash ---
    if results.get("aot_eager_ok") is False:
        err = results.get("aot_eager_error", "")
        if err and TorchLibrary.is_crash_msg(err) and not _is_unsupported_err(err):
            bug.is_bug = True
            bug.bug_type = "frontend_crash"
            bug.priority = 2
            bug.error_msg = err
            bug.trigger_config = "aot_eager"
            bug.category = program.category
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            return bug

    # --- aot_eager diff (graded) ---
    aot_out = results.get("aot_eager")
    if (not program.has_poison
            and aot_out is not None and eager_out is not None
            and program.is_deterministic):
        level, max_d, vr = check_divergence(eager_out, aot_out, program.dtype)
        if level == DivergenceLevel.CATASTROPHIC:
            bug.is_bug = True
            bug.bug_type = "frontend_diff"
            bug.priority = 2
            bug.max_diff = max_d
            bug.violation_ratio = vr
            bug.divergence_level = level
            bug.trigger_config = "aot_eager"
            bug.category = program.category
            bug.error_msg = f"eager vs aot_eager max_diff={max_d:.2e} vr={vr:.4f} [CATASTROPHIC]"
            return bug
        elif level == DivergenceLevel.SUSPICIOUS:
            bug.is_bug = True
            bug.bug_type = "frontend_diff"
            bug.priority = 1
            bug.max_diff = max_d
            bug.violation_ratio = vr
            bug.divergence_level = level
            bug.trigger_config = "aot_eager"
            bug.category = program.category
            bug.needs_backward = True
            bug.error_msg = f"eager vs aot_eager max_diff={max_d:.2e} vr={vr:.4f} [SUSPICIOUS]"
            return bug

    # --- inductor configs (including dynamic) ---
    for cfg_lbl, cfg_res in results.get("inductor_results", {}).items():
        is_dynamic = ("dynamic" in cfg_lbl)

        # Crash/compile errors — report unless env/resource error
        if not cfg_res.get("ok", False):
            err = cfg_res.get("error", "")
            if not err or _is_unsupported_err(err) or _is_env_error(err):
                continue
            bug.is_bug = True
            bug.bug_type = "dynamic_crash" if is_dynamic else "backend_crash"
            bug.category = program.category
            bug.priority = 3 if TorchLibrary.is_crash_msg(err) else 2
            bug.error_msg = err
            bug.trigger_config = f"inductor_{cfg_lbl}"
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            return bug

        # Dynamic mode: crash-only, no numerical diff
        if is_dynamic:
            continue

        # Diff checks — skip if poisoned
        if program.has_poison:
            continue
        if not program.is_deterministic:
            continue
        ind_out = cfg_res.get("result")
        if ind_out is None or eager_out is None:
            continue

        level, max_d, vr = check_divergence(eager_out, ind_out, program.dtype)
        if level == DivergenceLevel.SUCCESS:
            continue

        # determine if inductor-specific
        is_inductor_specific = False
        if aot_out is not None:
            aot_level, _, _ = check_divergence(aot_out, ind_out, program.dtype)
            is_inductor_specific = (aot_level != DivergenceLevel.SUCCESS)

        bug.is_bug = True
        bug.bug_type = "inductor_diff" if is_inductor_specific else "compile_diff"
        bug.max_diff = max_d
        bug.violation_ratio = vr
        bug.divergence_level = level
        bug.trigger_config = f"inductor_{cfg_lbl}"
        bug.category = program.category

        if level == DivergenceLevel.CATASTROPHIC:
            bug.priority = 3 if is_inductor_specific else 2
            bug.error_msg = (
                f"eager vs inductor({cfg_lbl}) max_diff={max_d:.2e} vr={vr:.4f} "
                f"[CATASTROPHIC|{'inductor-specific' if is_inductor_specific else 'aot-level'}]")
        else:  # SUSPICIOUS
            bug.priority = 1
            bug.needs_backward = True
            bug.error_msg = (
                f"eager vs inductor({cfg_lbl}) max_diff={max_d:.2e} vr={vr:.4f} "
                f"[SUSPICIOUS|{'inductor-specific' if is_inductor_specific else 'aot-level'}]")
        return bug

    # --- AOTI oracle target (3-state filter) ---
    # Reach here only if all JIT inductor configs succeeded with no diff bugs.
    # (If any JIT config failed or produced a diff, we've already returned above.)
    # State table:
    #   JIT default OK  & AOTI FAIL  -> aoti_specific_crash (record, unless env err)
    #   JIT default OK  & AOTI OK    -> check eager-vs-AOTI numerical diff (record if diff)
    #   JIT default FAIL & AOTI FAIL -> already handled by JIT path above (skip)
    #   AOTI export failed           -> not a fuzzer-target bug class (skip)
    if not results.get("aoti_export_failed", False):
        aoti_ok = results.get("aoti_ok", False)
        jit_default = results.get("inductor_results", {}).get("default", {})
        jit_default_ok = jit_default.get("ok", False)

        if jit_default_ok and not aoti_ok:
            # JIT_OK & AOTI_FAIL -> AOTI-specific crash/compile error
            err = results.get("aoti_error", "")
            if err and not _is_unsupported_err(err) and not _is_aoti_env_error(err):
                bug.is_bug = True
                bug.bug_type = "aoti_specific_crash"
                bug.priority = 3 if TorchLibrary.is_crash_msg(err) else 2
                bug.error_msg = "AOTI failed but JIT inductor succeeded: " + err[:600]
                bug.trigger_config = "aoti"
                bug.category = program.category
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                return bug
        elif jit_default_ok and aoti_ok:
            # JIT_OK & AOTI_OK -> check numerical diff (eager vs AOTI)
            if (not program.has_poison
                    and program.is_deterministic
                    and eager_out is not None):
                aoti_out = results.get("aoti_result")
                if aoti_out is not None:
                    if isinstance(aoti_out, (int, float)):
                        _atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
                        a_max = float(aoti_out)
                        if a_max > _atol * 10:
                            a_level = DivergenceLevel.CATASTROPHIC
                        elif a_max > _atol:
                            a_level = DivergenceLevel.SUSPICIOUS
                        else:
                            a_level = DivergenceLevel.SUCCESS
                        a_vr = 1.0 if a_level != DivergenceLevel.SUCCESS else 0.0
                    else:
                        a_level, a_max, a_vr = check_divergence(eager_out, aoti_out, program.dtype)
                    if a_level != DivergenceLevel.SUCCESS:
                        bug.is_bug = True
                        bug.bug_type = "aoti_diff"
                        bug.max_diff = a_max
                        bug.violation_ratio = a_vr
                        bug.divergence_level = a_level
                        bug.trigger_config = "aoti"
                        bug.category = program.category
                        if a_level == DivergenceLevel.CATASTROPHIC:
                            bug.priority = 3
                            bug.error_msg = (
                                f"eager vs AOTI max_diff={a_max:.2e} vr={a_vr:.4f} [CATASTROPHIC]")
                        else:  # SUSPICIOUS
                            bug.priority = 1
                            bug.needs_backward = True
                            bug.error_msg = (
                                f"eager vs AOTI max_diff={a_max:.2e} vr={a_vr:.4f} [SUSPICIOUS]")
                        return bug

    # --- Recompilation oracle: detect guard/recompilation bugs ---
    if (results.get('recompile_ok') and not program.has_poison
            and program.is_deterministic):
        rc_diff = results.get('recompile_diff')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if rc_diff is not None and rc_diff > atol * 100:
            bug.is_bug = True
            bug.bug_type = "recompile_diff"
            bug.priority = 3
            bug.max_diff = rc_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "inductor_recompile"
            bug.category = program.category
            bug.error_msg = (f"recompile_diff: different shape → max_diff={rc_diff:.2e} "
                             f"(atol={atol:.0e})")
            return bug

    # --- CPU inductor oracle: detect C++ codegen bugs ---
    cpu_err = results.get('cpu_inductor_error')
    if cpu_err and not _is_unsupported_err(cpu_err):
        bug.is_bug = True
        bug.bug_type = "cpu_backend_crash"
        bug.priority = 2
        bug.error_msg = cpu_err
        bug.trigger_config = "cpu_inductor"
        bug.category = program.category
        bug.divergence_level = DivergenceLevel.CATASTROPHIC
        return bug

    if (results.get('cpu_inductor_ok') and not program.has_poison
            and program.is_deterministic):
        cpu_diff = results.get('cpu_inductor_diff')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if cpu_diff is not None and cpu_diff > atol * 10:
            bug.is_bug = True
            bug.bug_type = "cpu_inductor_diff"
            bug.priority = 3
            bug.max_diff = cpu_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "cpu_inductor"
            bug.category = program.category
            bug.error_msg = (f"CPU eager vs CPU inductor max_diff={cpu_diff:.2e} "
                             f"(atol={atol:.0e})")
            return bug

    # --- Cross-fusion-level oracle: detect fusion pattern bugs ---
    if (results.get('cross_fusion_ok') and not program.has_poison
            and program.is_deterministic):
        cf_diff = results.get('cross_fusion_diff')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if cf_diff is not None and cf_diff > atol * 10:
            bug.is_bug = True
            bug.bug_type = "cross_fusion_diff"
            bug.priority = 3
            bug.max_diff = cf_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "inductor_nofusion_vs_default"
            bug.category = program.category
            bug.error_msg = (f"inductor(max_fusion_size=1) vs inductor(default) "
                             f"max_diff={cf_diff:.2e} (atol={atol:.0e})")
            return bug

    # --- Cross-dynamic-shape oracle: detect symbolic shape bugs ---
    if (results.get('dynamic_reuse_ok') and not program.has_poison
            and program.is_deterministic):
        dr_diff = results.get('dynamic_reuse_diff')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if dr_diff is not None and dr_diff > atol * 100:
            bug.is_bug = True
            bug.bug_type = "dynamic_reuse_diff"
            bug.priority = 3
            bug.max_diff = dr_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "inductor_dynamic_reuse"
            bug.category = program.category
            bug.error_msg = (f"dynamic compiled(shape_A) vs eager(shape_B) "
                             f"max_diff={dr_diff:.2e} (atol={atol:.0e})")
            return bug

    # --- Functionalization oracle: detect lowering bugs ---
    func_err = results.get('functionalize_error')
    if func_err and not _is_unsupported_err(func_err):
        bug.is_bug = True
        bug.bug_type = "functionalize_crash"
        bug.priority = 2
        bug.error_msg = func_err
        bug.trigger_config = "functionalize"
        bug.category = program.category
        bug.divergence_level = DivergenceLevel.CATASTROPHIC
        return bug

    if (results.get('functionalize_ok') and not program.has_poison
            and program.is_deterministic):
        func_diff = results.get('functionalize_diff')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if func_diff is not None and func_diff > atol * 10:
            bug.is_bug = True
            bug.bug_type = "functionalize_diff"
            bug.priority = 3
            bug.max_diff = func_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "functionalize"
            bug.category = program.category
            bug.error_msg = (f"eager vs functionalize max_diff={func_diff:.2e} "
                             f"(atol={atol:.0e})")
            return bug

    # --- Guard oracle: detect Dynamo guard correctness bugs ---
    if (results.get('guard_ok') and not program.has_poison
            and program.is_deterministic):
        guard_diff = results.get('guard_diff')
        guard_trigger = results.get('guard_trigger')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if guard_diff is not None and guard_diff > atol * 100:
            bug.is_bug = True
            bug.bug_type = "guard_diff"
            bug.priority = 3
            bug.max_diff = guard_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = f"guard_{guard_trigger}"
            bug.category = program.category
            bug.error_msg = (f"guard oracle: {guard_trigger} → max_diff={guard_diff:.2e} "
                             f"(atol={atol:.0e})")
            return bug

    # --- Finite-difference gradient oracle: detect backward formula bugs ---
    # This catches bugs invisible to ALL differential oracles: when autograd
    # itself returns wrong gradients (eager AND compiled both wrong).
    if results.get('finite_diff_ok') and program.is_deterministic:
        fd_max = results.get('finite_diff_max')
        if fd_max is not None and fd_max > 1e-2:
            bug.is_bug = True
            bug.bug_type = "finite_diff_grad"
            bug.priority = 3
            bug.max_diff = fd_max
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "autograd_finite_diff"
            bug.category = program.category
            bug.error_msg = (f"finite-diff vs autograd: max_rel_diff={fd_max:.2e} "
                             f"(threshold=1e-2)")
            return bug

    # --- CUDAGraph multi-replay oracle ---
    if (results.get('cudagraph_replay_ok') and not program.has_poison
            and program.is_deterministic):
        cg_diff = results.get('cudagraph_replay_diff')
        atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
        if cg_diff is not None and cg_diff > atol * 10:
            bug.is_bug = True
            bug.bug_type = "cudagraph_replay_diff"
            bug.priority = 3
            bug.max_diff = cg_diff
            bug.violation_ratio = 1.0
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.trigger_config = "reduce_overhead_replay"
            bug.category = program.category
            bug.error_msg = (f"CUDAGraph replay diff: max_diff={cg_diff:.2e} "
                             f"over 5 replays (atol={atol:.0e})")
            return bug

    return bug


# ---------------------------------------------------------------------------
# Inductor Config Presets
# ---------------------------------------------------------------------------

INDUCTOR_CONFIG_PRESETS = {
    "default": {},
    "aggressive_fusion": {
        "epilogue_fusion": True,
        "split_reductions": True,
        "pattern_matcher": True,
    },
    "no_fusion": {
        "epilogue_fusion": False,
        "pattern_matcher": False,
    },
    "freezing": {
        "freezing": True,
    },
    "max_autotune": {
        "max_autotune": True,
        "coordinate_descent_tuning": True,
    },
    "no_cudagraph": {
        "triton.cudagraphs": False,
    },
    # --- 2.7.0 new code paths ---
    "no_prologue_fusion": {
        # Prologue Fusion is ON by default in 2.7. Disabling it forces a different
        # codegen path for pointwise ops before GEMM/reduction kernels.
        "prologue_fusion": False,
    },
    "no_online_softmax": {
        # Online softmax (single-pass) is ON by default in 2.7. Disabling it
        # forces the legacy two-pass softmax codegen. Comparing results between
        # online and legacy paths can expose precision/correctness bugs.
        "online_softmax": False,
    },
    "emulate_precision": {
        # New in 2.7: controls whether low-precision casts are emulated.
        # Turning it ON changes how bf16/fp16 round-trips are handled in Triton.
        "emulate_precision_casts": True,
    },
    "combo_kernels": {
        # Off by default. Enables experimental kernel combination.
        "combo_kernels": True,
    },

    # --- 2.8 - 2.11 new code paths ---
    "disable_decompose_k": {
        # 2.8: DecomposeK is the new GEMM-K-axis decomposition for large K.
        # Default-on; turning it off forces the legacy matmul lowering.
        # Differential testing on/off catches DecomposeK miscompilations.
        "disable_decompose_k": True,
    },
    "min_split_high": {
        # 2.8: forces more aggressive reduction splitting.
        "min_num_split": 8,
    },
    "max_fusion_low": {
        # 2.8: limits how far apart in the schedule fusion candidates can be.
        # Tightening this changes which kernels get fused.
        "max_fusion_buffer_group_pairwise_attempts": 16,
    },
    "no_32bit_indexing": {
        # 2.10: assume_32bit_indexing toggles whether inductor uses 32-bit
        # indexing math. Forcing it OFF (use 64-bit) changes the codegen.
        "assume_32bit_indexing": False,
    },
    "combo_pointwise_only": {
        # 2.11: when combo_kernels is on, this restricts to pointwise only.
        "combo_kernels": True,
        "combo_kernels_pointwise_only": True,
    },
    "deterministic_inductor": {
        # 2.10: deterministic mode disables many non-deterministic optimizations.
        "deterministic": True,
    },
    # CUDAGraph mode: different memory management, known fragile
    # with dynamic shapes, in-place mutations, and tensor resizing.
    "reduce_overhead": {
        "triton.cudagraphs": True,
    },
}

_CONFIG_PRESET_NAMES = list(INDUCTOR_CONFIG_PRESETS.keys())
# Weights: higher = more likely to be selected.
# 2.8-2.11 presets get high weight because their code paths are the freshest.
# [default, aggressive_fusion, no_fusion, freezing, max_autotune, no_cudagraph,
#  no_prologue_fusion, no_online_softmax, emulate_precision, combo_kernels,
#  disable_decompose_k, min_split_high, max_fusion_low, no_32bit_indexing,
#  combo_pointwise_only, deterministic_inductor, reduce_overhead]
_CONFIG_PRESET_WEIGHTS = [1, 2, 3, 3, 2, 2, 2, 2, 2, 2,
                          4, 4, 4, 4, 4, 3, 3]


def _enumerate_inductor_bool_flags():
    """Extract all toggle-able bool flags from torch._inductor.config.

    Returns list of flag names that are safe to set to True/False.
    Uses PyTorch's ConfigEntry metadata to ensure type-correctness.
    """
    safe = []
    try:
        import torch._inductor.config as _ic
        d = _ic._config
        for name, entry in d.items():
            # Only top-level bool flags (no nested submodules — those need special handling)
            if type(entry).__name__ != "_ConfigEntry":
                continue
            if entry.value_type is bool:
                # Skip flags that are known to be dangerous or test-only
                DANGEROUS = {
                    "debug", "verbose_progress", "trace", "profile",
                    "TYPE_CHECKING",
                    # disabling caches breaks compilation
                    "fx_graph_cache",  # we already set this in eval code
                    "fx_graph_remote_cache",
                    # enabling benchmark mode causes autotune on every kernel — too slow
                    "benchmark_kernel", "benchmark_epilogue_fusion",
                    # torch.compile's low-level tracing / debugging flags
                }
                if name in DANGEROUS: continue
                safe.append(name)
    except Exception:
        pass
    return sorted(safe)


_INDUCTOR_BOOL_FLAGS = _enumerate_inductor_bool_flags()


def _sample_random_bool_flag_config():
    """Pick 1-3 random bool flags and force them to non-default values.

    Returns a preset-like dict suitable for _build_config_injection_code.
    """
    if not _INDUCTOR_BOOL_FLAGS:
        return {}
    n = random.choice([1, 1, 2, 3])
    picked = random.sample(_INDUCTOR_BOOL_FLAGS,
                           min(n, len(_INDUCTOR_BOOL_FLAGS)))
    # Force each to the opposite of default (strongest signal)
    import torch._inductor.config as _ic
    cfg_dict = {}
    for name in picked:
        try:
            default = _ic._config[name].default
            cfg_dict[name] = not default
        except Exception:
            continue
    return cfg_dict


def _sample_config_preset(combine_prob=0.05, random_flag_prob=0.05):
    """Sample a config preset. With probability `combine_prob`, combine
    two presets by '+'-joining their names (e.g. 'freezing+max_autotune').
    Only combines presets that are 'orthogonal' (don't conflict).

    With probability `random_flag_prob`, return a special marker
    'RAND_FLAGS:<json>' that tells _build_config_injection_code to set
    random bool flags from the full inductor config space.
    """
    # Random bool flag combo (5% of the time)
    if random.random() < random_flag_prob:
        cfg = _sample_random_bool_flag_config()
        if cfg:
            import json
            return "RAND_FLAGS:" + json.dumps(cfg)

    primary = random.choices(_CONFIG_PRESET_NAMES, weights=_CONFIG_PRESET_WEIGHTS, k=1)[0]
    if random.random() >= combine_prob or primary == "default":
        return primary
    # Pick a compatible second preset
    # Avoid combining presets that set the same key (conflict semantics)
    compatible = [p for p in _CONFIG_PRESET_NAMES
                  if p != primary and p != "default"
                  and not (INDUCTOR_CONFIG_PRESETS.get(primary, {}).keys() &
                           INDUCTOR_CONFIG_PRESETS.get(p, {}).keys())]
    if not compatible:
        return primary
    secondary = random.choice(compatible)
    return f"{primary}+{secondary}"


def _build_config_injection_code(preset_name):
    """Build inductor config injection code lines for the given preset.

    Returns an empty list for "default". preset_name can be a '+'-joined
    string (e.g. "freezing+max_autotune") to combine presets.
    """
    # RAND_FLAGS:<json> marker: decode to dict of bool flag overrides
    if preset_name.startswith("RAND_FLAGS:"):
        import json
        try:
            preset = json.loads(preset_name[len("RAND_FLAGS:"):])
        except Exception:
            preset = {}
    elif "+" in preset_name:
        parts = preset_name.split("+")
        preset = {}
        for p in parts:
            preset.update(INDUCTOR_CONFIG_PRESETS.get(p, {}))
    else:
        preset = INDUCTOR_CONFIG_PRESETS.get(preset_name, {})
    if not preset:
        return []
    lines = ["import torch._inductor.config as _ic"]
    for k, v in preset.items():
        lines.append(f"try:\n    _ic.{k} = {v!r}\nexcept AttributeError:\n    pass")
    return lines


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _sanitize_path_component(s):
    """Remove path traversal characters from a string used in file paths."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s) if s else "unknown"


def _atomic_write_next(type_dir, content):
    """Atomically create the next numbered .py file in type_dir, avoiding TOCTOU races."""
    os.makedirs(type_dir, exist_ok=True)
    for _attempt in range(100):
        existing = [int(f[:-3]) for f in os.listdir(type_dir)
                    if f.endswith(".py") and f[:-3].isdigit()]
        dest = join(type_dir, f"{max(existing, default=0) + 1}.py")
        try:
            fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "w") as f:
                f.write(content)
            return dest
        except FileExistsError:
            continue
    # Fallback: use timestamp-based name
    import uuid
    dest = join(type_dir, f"bug_{uuid.uuid4().hex[:12]}.py")
    with open(dest, "w") as f:
        f.write(content)
    return dest


def _write_bug_file(graph_dir, program, bug):
    safe_type = _sanitize_path_component(bug.bug_type)
    type_dir = join(graph_dir, "potential-bug", safe_type)
    content = (
        f"# GraphFuzzer Bug  [{bug.category}]\n"
        f"# type={bug.bug_type} P{bug.priority} config={bug.trigger_config}\n"
        f"# max_diff={bug.max_diff}\n"
        f"# error: {(bug.error_msg or '')[:200]}\n"
        f"# program: {program.signature}\n\nimport torch\n\n"
        + (program.body_code or "")
    )
    dest = _atomic_write_next(type_dir, content)

    TorchLibrary._save_diff_detail(dest, {
        "oracle": "graph", "api": program.name, "trigger": bug.bug_type,
        "max_diff": bug.max_diff, "mean_diff": None,
        "cpu_sample": None, "gpu_sample": None, "cpu_shape": None, "gpu_shape": None,
        "dtype": program.dtype,
        "has_nan_cpu": False, "has_nan_gpu": False, "has_inf_cpu": False, "has_inf_gpu": False,
        "error_msg": bug.error_msg,
        "program_name": program.name, "bug_type": bug.bug_type,
        "category": bug.category, "priority": bug.priority,
        "trigger_config": bug.trigger_config,
        "shape": f"{program.batch}x{program.seq}x{program.hidden}",
        "is_deterministic": program.is_deterministic,
    })
    return dest


def _write_bug_file_v2(graph_dir, program, bug, subdir="potential-bug"):
    """Write bug file with weak-signal subdirectory support.

    subdir: "potential-bug" (CATASTROPHIC) or "weak-signal" (SUSPICIOUS).
    """
    safe_subdir = _sanitize_path_component(subdir)
    safe_type = _sanitize_path_component(bug.bug_type)
    type_dir = join(graph_dir, safe_subdir, safe_type)
    content = (
        f"# GraphFuzzerV2 Bug  [{bug.category}]\n"
        f"# type={bug.bug_type} P{bug.priority} config={bug.trigger_config}\n"
        f"# divergence={bug.divergence_level} max_diff={bug.max_diff} vr={bug.violation_ratio}\n"
        f"# config_preset={program.config_preset} dynamic={program.use_dynamic}\n"
        f"# error: {(bug.error_msg or '')[:200]}\n"
        f"# program: {program.signature}\n\nimport torch\n\n"
        + (program.body_code or "")
    )
    dest = _atomic_write_next(type_dir, content)

    TorchLibrary._save_diff_detail(dest, {
        "oracle": "graph_v2", "api": program.name, "trigger": bug.bug_type,
        "max_diff": bug.max_diff, "mean_diff": None,
        "violation_ratio": bug.violation_ratio,
        "divergence_level": bug.divergence_level,
        "cpu_sample": None, "gpu_sample": None, "cpu_shape": None, "gpu_shape": None,
        "dtype": program.dtype,
        "has_nan_cpu": False, "has_nan_gpu": False, "has_inf_cpu": False, "has_inf_gpu": False,
        "error_msg": bug.error_msg,
        "program_name": program.name, "bug_type": bug.bug_type,
        "category": bug.category, "priority": bug.priority,
        "trigger_config": bug.trigger_config,
        "config_preset": program.config_preset,
        "use_dynamic": program.use_dynamic,
        "shape": f"{program.batch}x{program.seq}x{program.hidden}",
        "is_deterministic": program.is_deterministic,
    })
    return dest


def _save_crash_report(graph_dir, program, error_type, error_msg, iteration, source):
    """Save a CUDA crash or RuntimeError as a report file for post-mortem analysis."""
    safe_type = _sanitize_path_component(error_type)
    type_dir = join(graph_dir, "crash", safe_type)
    header = (
        f"# CRASH REPORT [{error_type}]\n"
        f"# iteration={iteration} source={source}\n"
        f"# error: {str(error_msg)[:500]}\n"
        f"# program: {getattr(program, 'signature', 'unknown')}\n"
        f"# config_preset={getattr(program, 'config_preset', 'unknown')}\n"
        f"# dynamic={getattr(program, 'use_dynamic', False)}\n"
        f"# shape: batch={program.batch} seq={program.seq} hidden={program.hidden}\n"
        f"# dtype: {program.dtype}\n\n"
        f"import torch\n\n"
    )
    content = header + (program.body_code or "")
    try:
        dest = _atomic_write_next(type_dir, content)
        # Also save JSON detail
        detail = {
            "error_type": error_type,
            "error_msg": str(error_msg)[:2000],
            "iteration": iteration,
            "source": source,
            "program_name": program.name,
            "program_signature": getattr(program, 'signature', ''),
            "config_preset": getattr(program, 'config_preset', 'default'),
            "use_dynamic": getattr(program, 'use_dynamic', False),
            "dtype": str(program.dtype),
            "batch": program.batch, "seq": program.seq, "hidden": program.hidden,
            "has_poison": getattr(program, 'has_poison', False),
        }
        TorchLibrary._save_diff_detail(dest, detail)
        print(f"  [CRASH] saved: {dest}")
        return dest
    except Exception as e:
        print(f"  [CRASH] failed to save report: {e}")
        return None


def _run_backward_verification(program, dtype):
    """Run backward differential verification.

    Returns (is_amplified_bug, grad_level, grad_max_diff, error_msg,
    bwd_kernel_heuristics) 5-tuple.
    """
    try:
        bwd_code = program.build_backward_eval_code()
        results, outer_error, _ = TorchLibrary.run_code(bwd_code)

        bwd_kheurs = results.get("_bwd_kernel_heuristics") if isinstance(results, dict) else None

        if outer_error:
            return False, None, None, f"backward outer error: {outer_error}", bwd_kheurs

        if not results.get("backward_eager_ok"):
            return False, None, None, results.get("backward_eager_error", "eager backward failed"), bwd_kheurs

        if not results.get("backward_inductor_ok"):
            err = results.get("backward_inductor_error", "")
            if err and TorchLibrary.is_crash_msg(err) and not _is_unsupported_err(err):
                return True, DivergenceLevel.CATASTROPHIC, None, f"inductor backward crash: {err}", bwd_kheurs
            return False, None, None, f"inductor backward failed: {err}", bwd_kheurs

        eager_grads = results.get("eager_grads", [])
        inductor_grads = results.get("inductor_grads", [])

        if not eager_grads or not inductor_grads:
            return False, None, None, "no gradients to compare", bwd_kheurs

        if len(eager_grads) != len(inductor_grads):
            return True, DivergenceLevel.CATASTROPHIC, float("inf"), \
                f"gradient count mismatch: eager={len(eager_grads)} inductor={len(inductor_grads)}", bwd_kheurs

        worst_level = DivergenceLevel.SUCCESS
        worst_abs = 0.0
        for eg, ig in zip(eager_grads, inductor_grads):
            level, ma, _ = check_divergence(eg, ig, dtype)
            if ma > worst_abs:
                worst_abs = ma
            if level == DivergenceLevel.CATASTROPHIC:
                worst_level = DivergenceLevel.CATASTROPHIC
            elif level == DivergenceLevel.SUSPICIOUS and worst_level == DivergenceLevel.SUCCESS:
                worst_level = DivergenceLevel.SUSPICIOUS

        is_amplified = (worst_level == DivergenceLevel.CATASTROPHIC)
        return is_amplified, worst_level, worst_abs, None, bwd_kheurs

    except Exception as e:
        return False, None, None, f"backward verification exception: {e}", None


# ---------------------------------------------------------------------------
# GraphFuzzer
# ---------------------------------------------------------------------------


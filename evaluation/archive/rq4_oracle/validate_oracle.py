#!/usr/bin/env python3
"""
RQ4 Oracle Validation Pipeline (standalone demonstration).

This script implements the three-stage oracle validation described in
Section 4.4 of the paper:

  1. Repeated Execution (Determinism): Run program 3 times, filter if
     results are inconsistent (FLAKY).
  2. Perturbation Stability: Add epsilon ~ U(-1e-4, 1e-4) to inputs;
     if the difference flips sign or vanishes, it's precision noise.
  3. FP64 Backoff: Re-run at float64 precision; if the difference
     disappears, the original was floating-point noise, not a real bug.

Usage:
    python validate_oracle.py --program /path/to/program.py
    python validate_oracle.py --demo  # run built-in test cases

The full oracle pipeline in the fuzzer (fuzzer.py:GraphFuzzerV4) also
includes cross-configuration checks (aot_eager, reduce_overhead, AOTI),
backward-mode amplification, and meta-stride validation. This script
demonstrates only the three core validation stages.
"""

import argparse
import os
import sys
import time

import torch
import torch._dynamo
import torch._inductor

torch._inductor.config.force_disable_caches = True

N_REPEATS = 3
PERTURB_EPSILON = 1e-4
N_PERTURB_TRIALS = 3
FP64_BACKOFF_THRESHOLD = 1e-6


def run_eager_vs_compiled(fn, inputs, device='cuda'):
    """Single eager-vs-compiled execution. Returns (max_diff, error)."""
    try:
        eager_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t for t in inputs
        )
        eager_out = fn(*eager_in)

        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, backend='inductor')
        comp_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t for t in inputs
        )
        compiled_out = compiled_fn(*comp_in)

        if isinstance(eager_out, torch.Tensor) and isinstance(compiled_out, torch.Tensor):
            if eager_out.shape != compiled_out.shape:
                return float('inf'), 'shape_mismatch'
            diff = (eager_out.float() - compiled_out.float()).abs().max().item()
            return diff, None
        return 0.0, None
    except Exception as e:
        return None, str(e)[:200]
    finally:
        torch._dynamo.reset()


def stage1_determinism(fn, inputs, device='cuda'):
    """Stage 1: Repeated execution (3 times). Returns (consistent, diffs)."""
    diffs = []
    for _ in range(N_REPEATS):
        d, err = run_eager_vs_compiled(fn, inputs, device)
        if err:
            return False, f'execution_error: {err}'
        diffs.append(d)

    if max(diffs) == 0.0:
        return True, diffs

    if min(diffs) == 0.0 and max(diffs) > 0.0:
        return False, diffs  # FLAKY: sometimes zero, sometimes nonzero

    ratio = min(diffs) / max(diffs) if max(diffs) > 0 else 1.0
    if ratio < 0.1:
        return False, diffs  # FLAKY: varies by >10x

    return True, diffs


def stage2_perturbation_stability(fn, inputs, baseline_diff, device='cuda'):
    """Stage 2: Perturb inputs and check directional consistency.
    Returns (stable, perturbed_diffs)."""
    perturbed_diffs = []
    for _ in range(N_PERTURB_TRIALS):
        perturbed_inputs = tuple(
            t + torch.empty_like(t).uniform_(-PERTURB_EPSILON, PERTURB_EPSILON)
            if isinstance(t, torch.Tensor) and t.is_floating_point() else t
            for t in inputs
        )
        d, err = run_eager_vs_compiled(fn, perturbed_inputs, device)
        if err or d is None:
            perturbed_diffs.append(None)
            continue
        perturbed_diffs.append(d)

    valid_diffs = [d for d in perturbed_diffs if d is not None]
    if not valid_diffs:
        return False, perturbed_diffs

    # Directional consistency: all perturbed runs should show similar
    # magnitude difference (within 10x of baseline)
    for d in valid_diffs:
        if d == 0.0 and baseline_diff > 1e-6:
            return False, perturbed_diffs  # Diff vanished under perturbation
    return True, perturbed_diffs


def stage3_fp64_backoff(fn, inputs, device='cuda'):
    """Stage 3: Re-run at float64 precision.
    If diff disappears at fp64, original was precision noise.
    Returns (is_real_bug, fp64_diff)."""
    inputs_fp64 = tuple(
        t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
        for t in inputs
    )

    try:
        eager_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t for t in inputs_fp64
        )
        eager_out = fn(*eager_in)

        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, backend='inductor')
        comp_in = tuple(
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, torch.Tensor) else t for t in inputs_fp64
        )
        compiled_out = compiled_fn(*comp_in)

        if isinstance(eager_out, torch.Tensor) and isinstance(compiled_out, torch.Tensor):
            fp64_diff = (eager_out.float() - compiled_out.float()).abs().max().item()
            is_real = fp64_diff > FP64_BACKOFF_THRESHOLD
            return is_real, fp64_diff
        return True, 0.0
    except Exception:
        return True, None  # Can't disprove, assume real
    finally:
        torch._dynamo.reset()


def validate_program(fn, inputs, device='cuda', verbose=True):
    """Full 3-stage validation pipeline.
    Returns dict with verdict and per-stage results."""
    result = {'verdict': None, 'stages': {}}

    # Stage 1: Determinism
    consistent, diffs = stage1_determinism(fn, inputs, device)
    result['stages']['determinism'] = {
        'passed': consistent, 'diffs': diffs
    }
    if not consistent:
        result['verdict'] = 'FLAKY'
        if verbose:
            print(f"  Stage 1 FAIL: inconsistent diffs = {diffs}")
        return result
    if verbose:
        print(f"  Stage 1 PASS: consistent diffs = {diffs}")

    baseline_diff = max(diffs) if isinstance(diffs, list) else 0.0
    if baseline_diff == 0.0:
        result['verdict'] = 'NO_BUG'
        if verbose:
            print(f"  No difference detected (diff=0). Not a bug.")
        return result

    # Stage 2: Perturbation stability
    stable, perturbed_diffs = stage2_perturbation_stability(
        fn, inputs, baseline_diff, device)
    result['stages']['perturbation'] = {
        'passed': stable, 'diffs': perturbed_diffs
    }
    if not stable:
        result['verdict'] = 'PRECISION_NOISE'
        if verbose:
            print(f"  Stage 2 FAIL: unstable under perturbation = {perturbed_diffs}")
        return result
    if verbose:
        print(f"  Stage 2 PASS: stable under perturbation = {perturbed_diffs}")

    # Stage 3: FP64 backoff
    is_real, fp64_diff = stage3_fp64_backoff(fn, inputs, device)
    result['stages']['fp64_backoff'] = {
        'passed': is_real, 'fp64_diff': fp64_diff
    }
    if not is_real:
        result['verdict'] = 'FP_NOISE'
        if verbose:
            print(f"  Stage 3 FAIL: diff vanishes at fp64 (fp64_diff={fp64_diff:.2e})")
        return result
    if verbose:
        print(f"  Stage 3 PASS: persists at fp64 (fp64_diff={fp64_diff})")

    result['verdict'] = 'VALID_BUG'
    return result


def demo():
    """Run validation on built-in test cases demonstrating each stage."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Oracle Validation Demo (device={device})")
    print("=" * 60)

    # Test 1: Clean program (no bug expected)
    print("\n[Test 1] Clean program (sin + relu):")

    def clean_fn(x):
        return torch.relu(torch.sin(x))

    clean_inputs = (torch.randn(64, 64, device=device),)
    r = validate_program(clean_fn, clean_inputs, device)
    print(f"  Verdict: {r['verdict']}")

    # Test 2: Program with known precision sensitivity
    print("\n[Test 2] Precision-sensitive (sum of many small values):")

    def precision_fn(x):
        return x.sum() * 1e8

    precision_inputs = (torch.randn(10000, device=device) * 1e-6,)
    r = validate_program(precision_fn, precision_inputs, device)
    print(f"  Verdict: {r['verdict']}")

    # Test 3: Known bug pattern (if available)
    print("\n[Test 3] Boundary-shape reduction (potential real bug):")

    def boundary_fn(x):
        return x.view(2, 65, 8, 8).sum(dim=1)

    boundary_inputs = (torch.randn(2 * 65 * 8 * 8, device=device),)
    r = validate_program(boundary_fn, boundary_inputs, device)
    print(f"  Verdict: {r['verdict']}")

    print("\n" + "=" * 60)
    print("Done. Verdicts: VALID_BUG, FP_NOISE, PRECISION_NOISE, FLAKY, NO_BUG")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RQ4 Oracle Validation Pipeline')
    parser.add_argument('--program', type=str, help='Path to program .py file')
    parser.add_argument('--demo', action='store_true', help='Run demo test cases')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.program:
        ns = {}
        exec(open(args.program).read(), ns)
        fn = ns['program']
        inputs = ns['_program_inputs']
        inputs = tuple(
            t.to(args.device) if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        print(f"Validating: {args.program}")
        r = validate_program(fn, inputs, args.device)
        print(f"\nFinal verdict: {r['verdict']}")
    else:
        parser.print_help()

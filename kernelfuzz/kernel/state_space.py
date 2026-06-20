"""
Kernel State Space — the canonical operation space for the paper.

A bounded, enumerable space of Triton kernel codegen configurations,
analogous to FreeFuzz's DispatcherStateSpace but defined over Inductor's
codegen decisions rather than per-API dispatcher state.

Per kernel family, state = (dtype, persistent, n_fused):
  - dtype       : float32, float64, float16, bfloat16            (4)
  - persistent  : True, False                                    (2)
  - n_fused_ops : 1, 2-3, 4+                                     (3)

Per-family raw space = 4 x 2 x 3 = 24
Total raw space      = 8 families x 24 = 192
Total valid space    = 192 - constraint-filtered combos  ~= 46

Kernel families (observable via _get_heuristic + kernel name prefix):
  pointwise, reduction, persistent_reduction, cooperative_reduction,
  split_scan, scan, extern_gemm, extern_conv.

NOTE: GraphCoverageTracker in coverage.py additionally recognizes
`template`, `foreach`, `fixed_config`, and `extern_fmha` as observed
family labels, but those have no dedicated generators yet and are not
counted toward the state-space denominator.

This class is the single, authoritative KernelStateSpace used by both
the runtime fuzzer (`fuzzer.py`) and the evaluation scripts. The matcher
predicate M_gamma (paper Algorithm 1) is exposed as `last_was_novel`
after each `record_from_kernels()` call.
"""

from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional
import re


class KernelStateSpace:
    """Bounded state space for Triton kernel codegen configurations.

    Dimensions per kernel family:
      1. dtype:          float32, float64, float16, bfloat16    (4)
      2. persistent:     True, False                            (2)
      3. n_fused_ops:    1, 2-3, 4+                             (3)

    Per-family raw space = 4 * 2 * 3 = 24.
    Total raw space      = 8 families * 24 = 192.
    After constraint filtering, ~46 valid (family, dtype, persistent,
    n_fused) tuples are reachable.
    """

    # Input dtypes the generator samples; fp16/bf16 appear via dtype_cast
    # templates (e.g., .half().float()) which produce fp16/bf16 Triton
    # kernels even when the program dtype is fp32/fp64.
    DIMENSION_DTYPE = [
        'float32', 'float64', 'float16', 'bfloat16',
    ]

    DIMENSION_PERSISTENT = [True, False]
    DIMENSION_N_FUSED = ['1', '2-3', '4+']

    # Kernel families with dedicated generators.
    KERNEL_FAMILIES = [
        'pointwise',
        'reduction',
        'persistent_reduction',
        'cooperative_reduction',
        'split_scan',
        'scan',           # single-block scan
        'extern_gemm',    # mm/bmm/addmm via extern_kernels
        'extern_conv',    # convolution via extern_kernels
    ]

    _FLOAT_DTYPES = {'float32', 'float64', 'float16', 'bfloat16'}
    # The generator only samples float32/float64 as input dtypes; fp16/bf16
    # appear in kernels via dtype_cast templates (bug_004's .half().float()
    # pattern), so for non-pointwise families we treat the reachable input
    # space as fp32/fp64 only.
    _PRIMARY_DTYPES = {'float32', 'float64'}
    _ALL_DTYPES = set(DIMENSION_DTYPE)

    # Family constraints reflect what the generator CAN reach.
    # - pointwise: all 4 float dtypes (dtype_cast templates produce
    #   fp16/bf16 pointwise kernels)
    # - reduction / scan / GEMM / conv: only fp32/fp64 reachable
    _FAMILY_CONSTRAINTS = {
        'pointwise':              {'dtype': _FLOAT_DTYPES,
                                   'persistent': {False}},

        # extern_gemm/conv: only primary dtypes, standalone (no fusion)
        'extern_gemm':            {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {False},
                                   'n_fused': {'1'}},
        'extern_conv':            {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {False},
                                   'n_fused': {'1'}},

        # scan / split_scan: primary dtypes only
        'scan':                   {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {True}},
        'split_scan':             {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {False}},

        # Reductions: primary dtypes only
        'cooperative_reduction':  {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {False}},
        'persistent_reduction':   {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {True}},
        'reduction':              {'dtype': _PRIMARY_DTYPES,
                                   'persistent': {False}},
    }

    def __init__(self):
        self._family_spaces: Dict[str, int] = {}
        self._hit_states: Dict[str, Set[str]] = {}  # family -> set of state hashes
        self._hit_freq: Dict[str, Dict[str, int]] = {}  # family -> {state_hash: count}
        self._total_valid = 0
        self._total_hits_count = 0

        # Pre-compute valid space per family
        for family in self.KERNEL_FAMILIES:
            constraints = self._FAMILY_CONSTRAINTS.get(family, {})
            valid = self._count_valid_states(constraints)
            self._family_spaces[family] = valid
            self._hit_states[family] = set()
            self._hit_freq[family] = {}
            self._total_valid += valid

        # Matcher signal (M_gamma in paper Algorithm 1): True iff the most
        # recent record_from_kernels() observed at least one new state.
        self.last_was_novel = False

    def _count_valid_states(self, constraints: dict) -> int:
        dtypes = constraints.get('dtype', self.DIMENSION_DTYPE)
        persists = constraints.get('persistent', self.DIMENSION_PERSISTENT)
        n_fused = constraints.get('n_fused', self.DIMENSION_N_FUSED)
        return len(dtypes) * len(persists) * len(n_fused)

    @staticmethod
    def _classify_n_fused(n: int) -> str:
        if n <= 1: return '1'
        if n <= 3: return '2-3'
        return '4+'

    @staticmethod
    def _infer_family_from_kernel_name(kernel_name: str) -> Optional[str]:
        """Infer kernel family from triton kernel function name prefix
        or extern kernel event name."""
        if not kernel_name.startswith('triton_'):
            kn = kernel_name.lower()
            # Extern GEMM: aten::mm/bmm/addmm, sgemm/dgemm/hgemm, cutlass,
            # extern_kernels.mm, etc.
            if ('aten::mm' in kn or 'aten::bmm' in kn or 'aten::addmm' in kn
                    or 'sgemm' in kn or 'dgemm' in kn or 'hgemm' in kn
                    or 'gemm' in kn or 'cutlass' in kn
                    or 'extern_kernels.mm' in kn or 'extern_kernels.bmm' in kn
                    or 'extern_kernels.addmm' in kn):
                return 'extern_gemm'
            # Extern conv: aten::convolution, cudnn_convolution, implicit/
            # winograd conv variants.
            if ('aten::convolution' in kn or 'aten::conv2d' in kn
                    or 'cudnn' in kn or 'cudnn_convolution' in kn
                    or 'extern_kernels.convolution' in kn
                    or ('conv' in kn and ('implicit' in kn or 'winograd' in kn))):
                return 'extern_conv'
            return None

        prefix_map = {
            'poi': 'pointwise',
            'per': 'persistent_reduction',
            'red': 'reduction',
            'hel': 'cooperative_reduction',
        }
        m = re.match(r'triton_(\w{3})_', kernel_name)
        if m:
            return prefix_map.get(m.group(1))
        return None

    @staticmethod
    def _count_fused_ops(kernel_name: str) -> int:
        """Count fused ops from a Triton kernel name like
        triton_per_fused__softmax_exp_sub_sum_0 -> 4."""
        m = re.match(r'triton_\w{3}_fused_(.+?)_\d+$', kernel_name)
        if m:
            ops_str = m.group(1)
            parts = [p for p in ops_str.split('_') if p]
            return len(parts)
        return 1

    def record_hit(self, kernel_name: str, dtype: str,
                   heuristic: str = None,
                   persistent: bool = None) -> bool:
        """Record one kernel observation. Returns True iff this is a
        previously-unseen (dtype, persistent, n_fused) state for the
        inferred family."""
        family = heuristic or self._infer_family_from_kernel_name(kernel_name)
        if not family or family not in self._family_spaces:
            return False

        dtype_str = dtype.replace('torch.', '') if dtype else 'float32'
        if dtype_str not in self.DIMENSION_DTYPE:
            dtype_str = 'float32'

        # Default `persistent` from family identity if caller did not supply.
        if persistent is None:
            if family == 'persistent_reduction':
                persistent = True
            elif family in ('reduction', 'cooperative_reduction', 'split_scan'):
                persistent = False
            else:
                persistent = False

        n_fused = self._classify_n_fused(self._count_fused_ops(kernel_name))
        state_hash = f"{dtype_str}|{persistent}|{n_fused}"

        self._hit_freq[family][state_hash] = (
            self._hit_freq[family].get(state_hash, 0) + 1
        )

        if state_hash not in self._hit_states[family]:
            self._hit_states[family].add(state_hash)
            self._total_hits_count += 1
            return True
        return False

    def record_from_kernels(self, kernel_names, dtype: str,
                            heuristics: list = None) -> int:
        """Record hits from a kernel-event batch (as returned by
        TorchLibrary.run_code). Returns count of newly-discovered states.

        Side effect: sets `self.last_was_novel = (return value > 0)`,
        which is the per-iteration matcher predicate M_gamma in paper
        Algorithm 1.
        """
        new_count = 0
        heur_list = list(heuristics) if heuristics else []
        heur_idx = 0

        for k in kernel_names:
            if not isinstance(k, str):
                continue

            if 'triton_' in k and 'fused_' in k:
                h = heur_list[heur_idx] if heur_idx < len(heur_list) else None
                heur_idx += 1
                if self.record_hit(k, dtype, heuristic=h):
                    new_count += 1
            else:
                fam = self._infer_family_from_kernel_name(k)
                if fam in ('extern_gemm', 'extern_conv'):
                    if self.record_hit(k, dtype, heuristic=fam):
                        new_count += 1

        self.last_was_novel = new_count > 0
        return new_count

    def get_coverage(self, family: str = None) -> float:
        if family:
            total = self._family_spaces.get(family, 0)
            hits = len(self._hit_states.get(family, set()))
            return hits / max(total, 1)
        return self._total_hits_count / max(self._total_valid, 1)

    def get_total_valid(self) -> int:
        return self._total_valid

    def get_total_hits(self) -> int:
        return self._total_hits_count

    def get_uncovered_families(self) -> List[str]:
        return [f for f in self.KERNEL_FAMILIES
                if len(self._hit_states[f]) == 0]

    def get_lowest_coverage_family(self) -> Optional[str]:
        """Return the family with the lowest non-zero coverage (good
        target for guided sampling)."""
        best = None
        best_cov = 1.0
        for family in self.KERNEL_FAMILIES:
            cov = self.get_coverage(family)
            if 0 < cov < best_cov:
                best = family
                best_cov = cov
        return best

    def get_uncovered_state(self, family: str):
        """Return one uncovered (dtype, persistent, n_fused) tuple for
        `family`, or None if family is fully covered / unknown."""
        if family not in self._family_spaces:
            return None
        constraints = self._FAMILY_CONSTRAINTS.get(family, {})
        dtypes = list(constraints.get('dtype', self.DIMENSION_DTYPE))
        persists = list(constraints.get('persistent', self.DIMENSION_PERSISTENT))
        n_fused = list(constraints.get('n_fused', self.DIMENSION_N_FUSED))
        hit = self._hit_states.get(family, set())
        import itertools, random as _r
        all_states = list(itertools.product(dtypes, persists, n_fused))
        _r.shuffle(all_states)
        for dt, p, nf in all_states:
            state_hash = f"{dt}|{p}|{nf}"
            if state_hash not in hit:
                return (dt, p, nf)
        return None

    # Backward-compatible plural alias used by some evaluation scripts.
    def get_uncovered_states(self) -> Set[tuple]:
        """All uncovered (family, dtype, persistent, n_fused) tuples."""
        import itertools
        uncovered = set()
        for family in self.KERNEL_FAMILIES:
            constraints = self._FAMILY_CONSTRAINTS.get(family, {})
            dtypes = list(constraints.get('dtype', self.DIMENSION_DTYPE))
            persists = list(constraints.get('persistent', self.DIMENSION_PERSISTENT))
            n_fused = list(constraints.get('n_fused', self.DIMENSION_N_FUSED))
            hit = self._hit_states.get(family, set())
            for dt, p, nf in itertools.product(dtypes, persists, n_fused):
                state_hash = f"{dt}|{p}|{nf}"
                if state_hash not in hit:
                    uncovered.add((family, dt, p, nf))
        return uncovered

    def get_summary(self) -> Dict:
        per_family = {}
        for family in self.KERNEL_FAMILIES:
            hits = len(self._hit_states[family])
            total = self._family_spaces[family]
            per_family[family] = {
                'hits': hits,
                'total': total,
                'coverage': hits / max(total, 1),
            }
        return {
            'total_hits': self._total_hits_count,
            'total_valid': self._total_valid,
            'coverage': self.get_coverage(),
            'per_family': per_family,
        }

    def get_hit_frequency(self) -> Dict[str, Dict[str, int]]:
        """Per-state hit frequency for heatmap visualization."""
        return {f: dict(counts) for f, counts in self._hit_freq.items() if counts}

    def save_checkpoint(self, path: str):
        """Save hit states to disk for cross-worker accumulation."""
        import json
        data = {
            "version": 1,
            "hit_states": {fam: sorted(hits)
                           for fam, hits in self._hit_states.items()},
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  [KernelSS] save_checkpoint failed: {e}")

    def load_checkpoint(self, path: str) -> int:
        """Load hit states from disk. Returns count of restored states."""
        import os, json
        if not os.path.exists(path):
            return 0
        try:
            with open(path) as f:
                data = json.load(f)
            hit_states = data.get("hit_states", {})
            restored = 0
            for fam, hits in hit_states.items():
                if fam in self._hit_states:
                    for h in hits:
                        if h not in self._hit_states[fam]:
                            self._hit_states[fam].add(h)
                            self._total_hits_count += 1
                            restored += 1
            return restored
        except Exception as e:
            print(f"  [KernelSS] load_checkpoint failed: {e}")
            return 0

    def print_coverage(self):
        total_cov = self.get_coverage()
        print(f"Kernel State Space Coverage: "
              f"{self._total_hits_count}/{self._total_valid} "
              f"({total_cov*100:.1f}%)")
        for family in self.KERNEL_FAMILIES:
            hits = len(self._hit_states[family])
            total = self._family_spaces[family]
            cov = hits / max(total, 1)
            bar_w = 20
            filled = int(bar_w * cov)
            bar = '#' * filled + '.' * (bar_w - filled)
            print(f"  {family:25s}: {hits:3d}/{total:3d} "
                  f"({cov*100:5.1f}%) [{bar}]")


# Module-level constants for backward-compat with old `generation/target.py`
# callers that did `from kernel.state_space import FAMILIES, DTYPES`.
FAMILIES = list(KernelStateSpace.KERNEL_FAMILIES)
DTYPES = [d for d in KernelStateSpace.DIMENSION_DTYPE]

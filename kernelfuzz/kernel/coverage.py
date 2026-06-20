#!/usr/bin/env python3
"""
Kernel-space coverage tracking (paper §3.1).

Three-level coverage model:
  L1: Kernel family (8 families) — coarse targeting
  L2: Fusion pattern (unique op sets per Triton kernel) — codegen diversity
  L3: Kernel hash (SHA256 of kernel event) — fine-grained dedup

Plus saturation detection and mutation scheduling.
"""

import hashlib
import random
import re
import time
from collections import defaultdict, deque

class GraphSaturationDetector:
    def __init__(self, patience=100, check_interval=50):
        self.patience=patience; self.check_interval=check_interval
        self.no_discovery=0; self.last_count=0; self.last_check=0; self.expansion_mode=False

    def update(self, current_kernels, iteration):
        if current_kernels > self.last_count:
            self.no_discovery=0; self.last_count=current_kernels
            if self.expansion_mode:
                self.expansion_mode=False; return False, "New kernels, returning to normal"
        else:
            self.no_discovery += 1
        if (iteration-self.last_check >= self.check_interval
                and self.no_discovery >= self.patience and not self.expansion_mode):
            self.last_check=iteration; self.expansion_mode=True
            return True, f"Saturation: {self.no_discovery} iters, entering expansion"
        return False, ""


# Adversarial shape pools: 2^n +/- 1 to stress Triton tile masking
_ADV_BATCH  = [1, 3, 7, 15, 17]
_ADV_SEQ    = [31, 33, 63, 65, 127, 129, 255, 257]
_ADV_HIDDEN = [31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1023]

_STD_BATCH  = [1, 2, 3, 4, 7]
_STD_SEQ    = [7, 8, 13, 16, 32, 33, 64, 65, 127]
_STD_HIDDEN = [7, 8, 13, 16, 32, 33, 64, 65, 127, 128, 256]

# Inductor codegen decision boundaries: shapes +/-1 from exact thresholds
# where Inductor switches codegen strategy.
_BOUNDARY_SHAPES = [
    # Persistent-reduction threshold: rnumel > 1024 goes non-persistent
    1022, 1023, 1024, 1025, 1026,
    # Non-persistent reduction block size commonly 2048
    2046, 2047, 2048, 2049, 2050,
    # SplitScan threshold: scan_dim >= 129
    126, 127, 128, 129, 130, 131,
    # fp64 persistent reduction boundary
    511, 512, 513, 514,
    # Cooperative reduction / large reduction
    32766, 32768, 32770,
    # Triton block sizes — tile masking edges
    16, 32, 33, 64, 65,
]
_BOUNDARY_SEQ = [x for x in _BOUNDARY_SHAPES if x <= 1024]
_BOUNDARY_HIDDEN = [x for x in _BOUNDARY_SHAPES if x <= 2048]


def _sample_shapes(adversarial_ratio=0.0):
    """Sample batch/seq/hidden. adversarial_ratio > 0 enables adversarial shapes.

    20% of adversarial samples pick exactly one dim from codegen decision
    boundary shapes.
    """
    if adversarial_ratio > 0 and random.random() < adversarial_ratio:
        # 20% chance: pick one dim from codegen decision boundary
        if random.random() < 0.20:
            b = random.choice(_ADV_BATCH)
            # Pick which dim gets the boundary shape
            target_dim = random.choice(['seq', 'hidden'])
            if target_dim == 'seq':
                return (b, random.choice(_BOUNDARY_SEQ),
                        random.choice(_ADV_HIDDEN))
            else:
                return (b, random.choice(_ADV_SEQ),
                        random.choice(_BOUNDARY_HIDDEN))
        return (random.choice(_ADV_BATCH),
                random.choice(_ADV_SEQ),
                random.choice(_ADV_HIDDEN))
    return (random.choice(_STD_BATCH),
            random.choice(_STD_SEQ),
            random.choice(_STD_HIDDEN))


class GraphCoverageTracker:
    """Kernel codegen coverage tracking.

    Coverage = |F| + |E|
      F = unique Triton fusion patterns (codegen paths)
      E = unique extern kernel variants (cuBLAS/CUTLASS dispatch paths)

    Level 1 — Kernel family: coarse categories (pointwise, reduction, ...)
    Level 2 — Codegen path: fusion patterns (Triton) + extern variants (cuBLAS/CUTLASS)
    Level 3 — Kernel hash: SHA256 dedup (internal, not reported as coverage)
    """

    # Level 1: All known kernel families
    ALL_KERNEL_FAMILIES = frozenset([
        "pointwise", "reduction", "persistent_reduction",
        "cooperative_reduction", "split_scan", "template",
        "foreach", "fixed_config",
        "extern_gemm", "extern_fmha", "extern_conv",
    ])

    # Prefix → family mapping for Triton kernel names
    _PREFIX_TO_FAMILY = {
        "poi": "pointwise",
        "per": "persistent_reduction",
        "red": "reduction",
        "hel": "cooperative_reduction",
    }

    # Extern kernel classification patterns
    _EXTERN_PATTERNS = [
        (re.compile(r'(volta|ampere|turing|sm\d+)_[shd]gemm'), "extern_gemm"),
        (re.compile(r'gemm[A-Z]'), "extern_gemm"),
        (re.compile(r'cublas'), "extern_gemm"),
        (re.compile(r'fmha_cutlass'), "extern_fmha"),
        (re.compile(r'fmha_'), "extern_fmha"),
        (re.compile(r'flash_'), "extern_fmha"),
        (re.compile(r'(implicit|explicit)_gemm.*conv', re.IGNORECASE), "extern_conv"),
        (re.compile(r'cudnn.*conv', re.IGNORECASE), "extern_conv"),
    ]

    @staticmethod
    def _classify_extern(name):
        for pat, family in GraphCoverageTracker._EXTERN_PATTERNS:
            if pat.search(name):
                base = re.sub(r'_\d+x\d+(?:x\d+)?', '', name)
                base = re.sub(r'_[a-f0-9]{8,}', '', base)
                return family, base
        return None, None

    def __init__(self, name):
        self.name = name
        # Level 3: kernel hash tracking (internal dedup)
        self.all_kernel_hashes = set()
        self.kernel_count = 0
        self.history = deque(maxlen=500)
        self._recent_kernels = deque(maxlen=1000)
        self.saturation = GraphSaturationDetector()
        # Level 1: kernel family coverage
        self._kernel_families_seen = set()
        self._kernel_family_counts = defaultdict(int)
        # Level 2: fusion pattern coverage (Triton)
        self._fusion_patterns_seen = set()  # set of frozenset(fused_ops)
        self._fusion_pattern_first_iter = {}
        # Level 2: extern kernel variants
        self._extern_variants_seen = set()  # set of normalized extern kernel names
        # Matcher signal (M_gamma in paper Algorithm 1): True iff the most
        # recent update() observed a new family, fusion pattern, or extern
        # variant. Acts as the per-iteration "target hit" predicate.
        self.last_was_novel = False

    @staticmethod
    def _parse_triton_kernel_name(name):
        """Extract (family_prefix, fused_op_set) from a Triton kernel name.

        Example: 'triton_per_fused__softmax_exp_sub_sum_0'
          → prefix='per', fused_ops={'softmax', 'exp', 'sub', 'sum'}
        """
        import re
        m = re.match(r'triton_(\w{3})_fused_(.+?)_(\d+)$', name)
        if m:
            prefix = m.group(1)
            ops_str = m.group(2)
            # Split on '_' but handle ops starting with '_' (like _softmax)
            ops = set()
            for op in re.split(r'(?<=[a-z0-9])_(?=[a-z])', ops_str):
                op = op.strip('_')
                if op:
                    ops.add(op)
            return prefix, frozenset(ops)
        return None, None

    def update(self, new_kernels, iteration, config_preset=None,
               kernel_heuristics=None):
        if not isinstance(new_kernels, set):
            new_kernels = set(new_kernels) if new_kernels else set()

        # Level 3: kernel hash tracking
        fresh = 0
        for k in new_kernels:
            h = hashlib.sha256(k.encode()).hexdigest()[:32]
            if h not in self.all_kernel_hashes:
                self.all_kernel_hashes.add(h)
                self._recent_kernels.append(k)
                fresh += 1
        self.kernel_count = len(self.all_kernel_hashes)

        # Level 1: kernel family from heuristic types
        new_families = set()
        if kernel_heuristics:
            for h in kernel_heuristics:
                self._kernel_family_counts[h] += 1
                if h not in self._kernel_families_seen:
                    self._kernel_families_seen.add(h)
                    new_families.add(h)

        # Level 2a: fusion patterns from Triton kernel names
        new_fusions = []
        for k in new_kernels:
            if 'triton_' in k and 'fused_' in k:
                prefix, fused_ops = self._parse_triton_kernel_name(k)
                if fused_ops and fused_ops not in self._fusion_patterns_seen:
                    self._fusion_patterns_seen.add(fused_ops)
                    self._fusion_pattern_first_iter[fused_ops] = iteration
                    new_fusions.append(fused_ops)
                if prefix and prefix in self._PREFIX_TO_FAMILY:
                    fam = self._PREFIX_TO_FAMILY[prefix]
                    if fam not in self._kernel_families_seen:
                        self._kernel_families_seen.add(fam)
                        new_families.add(fam)

        # Level 2b: extern kernel variants (cuBLAS/CUTLASS/cuDNN)
        new_externs = []
        for k in new_kernels:
            if 'triton_' in k or k.startswith('aten::') or k.startswith('inductor_'):
                continue
            ext_family, ext_variant = self._classify_extern(k)
            if ext_family:
                if ext_variant not in self._extern_variants_seen:
                    self._extern_variants_seen.add(ext_variant)
                    new_externs.append(ext_variant)
                if ext_family not in self._kernel_families_seen:
                    self._kernel_families_seen.add(ext_family)
                    new_families.add(ext_family)

        # Combined coverage = fusion patterns + extern variants
        new_coverage = len(new_fusions) + len(new_externs)

        # Matcher signal: any new L1 family or L2 codegen path counts as a
        # target hit. Used as the per-iteration M_gamma predicate in the
        # paper's Algorithm 1.
        self.last_was_novel = bool(new_families) or new_coverage > 0

        # Log coverage changes
        if fresh > 0 or new_families or new_fusions or new_externs:
            parts = []
            if new_fusions:
                parts.append(f"+{len(new_fusions)} fusions")
            if new_externs:
                parts.append(f"+{len(new_externs)} externs({','.join(new_externs[:3])})")
            if new_families:
                parts.append(f"+families:{','.join(sorted(new_families))}")
            cov = len(self._fusion_patterns_seen) + len(self._extern_variants_seen)
            print(f"  [GraphCov] iter {iteration}: {' '.join(parts)} "
                  f"L1={len(self._kernel_families_seen)} "
                  f"cov={cov}(F{len(self._fusion_patterns_seen)}+E{len(self._extern_variants_seen)})")

        # Saturation tracks combined coverage
        cov_total = len(self._fusion_patterns_seen) + len(self._extern_variants_seen)
        self.history.append((iteration, cov_total))
        sat_sig = self.saturation.update(cov_total, iteration)
        if sat_sig[0]:
            print(f"  [GraphCov] {sat_sig[1]}")
        return new_coverage, sat_sig

    def get_total(self):
        return len(self._fusion_patterns_seen) + len(self._extern_variants_seen)

    def get_coverage_summary(self):
        return {
            "L1_families": sorted(self._kernel_families_seen),
            "L1_coverage": f"{len(self._kernel_families_seen)}/{len(self.ALL_KERNEL_FAMILIES)}",
            "L2_fusion_patterns": len(self._fusion_patterns_seen),
            "L2_extern_variants": len(self._extern_variants_seen),
            "L2_total": len(self._fusion_patterns_seen) + len(self._extern_variants_seen),
            "L3_kernel_hashes": self.kernel_count,
        }


class GraphMutationController:
    def __init__(self, base_epsilon=0.10, expansion_epsilon=0.30,
                 surgical_range=16, exploration_range=64):
        self.base_epsilon=base_epsilon; self.expansion_epsilon=expansion_epsilon
        self.surgical_range=surgical_range; self.exploration_range=exploration_range
        self.current_epsilon=base_epsilon; self.current_range=surgical_range
        self.expansion_mode=False; self.exploration_count=0; self.exploitation_count=0
        self.warmup_count=0

    def record_warmup(self):
        self.warmup_count += 1

    def should_explore(self):
        if random.random() < self.current_epsilon:
            self.exploration_count += 1; return True
        self.exploitation_count += 1; return False

    def enter_expansion_mode(self):
        self.expansion_mode=True; self.current_epsilon=self.expansion_epsilon
        self.current_range=self.exploration_range
        print(f"  [GraphMut] Expansion: eps={self.current_epsilon} range=+/-{self.current_range}")

    def exit_expansion_mode(self):
        self.expansion_mode=False; self.current_epsilon=self.base_epsilon
        self.current_range=self.surgical_range
        print(f"  [GraphMut] Normal: eps={self.current_epsilon} range=+/-{self.current_range}")

    def get_mutation_range(self): return self.current_range

    def get_status(self):
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return f"Warmup phase ({self.warmup_count} programs)"
        pct = self.exploration_count / total * 100
        return (f"explore={self.exploration_count}/{total}({pct:.1f}%) "
                f"mode={'expansion' if self.expansion_mode else 'normal'} "
                f"eps={self.current_epsilon:.2f} range=+/-{self.current_range} "
                f"warmup={self.warmup_count}")


# Differential oracle only compares float32/float64 (half-precision
# numerical differences are spec-permitted and produce false positives).
GRAPH_DIFF_BOUNDS = {"torch.float32": 1e-3, "torch.float64": 1e-6}
_DTYPES        = ["torch.float32", "torch.float64"]
_DTYPE_WEIGHTS = [3, 3]  # balanced for better KernelSS dtype-dim coverage

_UNSUPPORTED_PATTERNS = [
    "unsupported", "nyi", "not implemented", "not supported",
    "torch._dynamo.exc.unsupported", "data-dependent",
    "graph break", "cannot trace", "tracing",
]

_ENV_ERROR_PATTERNS = [
    "no space left on device",
    "errno 28",
    "cuda out of memory",
    "cuda driver error: out of memory",
    "too many blocks in cooperative launch",
    "does not support bfloat16 compilation natively",
    "skipping cudagraphs due to cpp-wrapper",
    "both a fallback and a decomp for same op",
    "c++ wrapper codegen error: c++ codegen is disabled",
    "found no compiled frames",
]

def _is_env_error(msg):
    if not msg: return False
    m = msg.lower()
    return any(p in m for p in _ENV_ERROR_PATTERNS)

def _is_unsupported_err(msg):
    if not msg: return False
    return any(p in msg.lower() for p in _UNSUPPORTED_PATTERNS)




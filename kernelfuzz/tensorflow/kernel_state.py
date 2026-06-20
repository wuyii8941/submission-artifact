"""TF kernel state space and coverage-guided scheduler.

State definition:
    s = (op_family, dtype_class, execution_seam, shape_regime)

    op_family       in {activation, arithmetic, reduction, normalize, linalg, structural}
    dtype_class     in {float32, float64, complex64, complex128}
    execution_seam  in {eager_vs_grappler, eager_vs_xla, cpu_vs_gpu, backward, mixed_precision}
    shape_regime    in {standard, boundary, degenerate, magnitude}

Total theoretical: 6x4x5x4 = 480.  After constraint filtering: ~120 reachable.
"""
import random
from collections import defaultdict

OP_FAMILIES = ["activation", "arithmetic", "reduction", "normalize", "linalg", "structural"]
DTYPE_CLASSES = ["float32", "float64", "complex64", "complex128"]
EXECUTION_SEAMS = ["eager_vs_grappler", "eager_vs_xla", "cpu_vs_gpu", "backward", "mixed_precision"]
SHAPE_REGIMES = ["standard", "boundary", "degenerate", "magnitude"]

_OP_TO_FAMILY = {}
_ACTIVATION_OPS = {
    "k3_sin", "k3_cos", "k3_tanh", "k3_exp", "k3_log1p", "k3_sigmoid",
    "k3_relu", "k3_relu6", "k3_silu", "k3_gelu", "k3_elu", "k3_softplus",
    "k3_softsign", "k3_abs", "k3_square", "k3_rsqrt", "k3_erf", "k3_expm1",
    "k3_sign", "k3_ceil", "k3_floor", "k3_arctan", "k3_arcsinh", "k3_log_sigmoid",
    "nn_relu", "nn_elu", "nn_selu", "nn_gelu", "nn_swish", "nn_silu",
    "nn_sigmoid", "nn_softplus", "nn_tanh",
    "raw_sigmoid", "raw_tanh", "raw_softplus", "raw_relu", "raw_rsqrt",
    "raw_erf", "raw_elu",
    "where_pos", "clip", "nan_to_num",
}
_ARITHMETIC_OPS = {
    "k3_add", "k3_sub", "k3_mul", "k3_div", "k3_max", "k3_min",
    "k3_power", "k3_hypot", "k3_arctan2", "k3_logaddexp",
    "k3_maximum", "k3_minimum",
}
_REDUCTION_OPS = {
    "k3_sum_last", "k3_mean_last", "k3_max_last", "k3_min_last",
    "k3_prod_last", "k3_logsumexp", "k3_std_last", "k3_var_last",
    "k3_norm_last", "k3_cumsum", "k3_cumprod", "k3_sum_0",
    "k3_reduce_max", "k3_reduce_min",
    "amp_argmax", "amp_argmin",
}
_NORMALIZE_OPS = {
    "k3_softmax", "k3_log_softmax", "k3_l2_norm", "k3_layernorm", "k3_rmsnorm",
    "nn_softmax", "nn_log_softmax",
    "amp_softmax_floor",
}
_LINALG_OPS = {"matmul"}
_STRUCTURAL_OPS = {
    "cast_bf16_f32", "cast_f16_f32", "cast_f64_f32",
    "idx_slice_half", "idx_reverse", "idx_roll", "idx_sort", "idx_tile2", "idx_argsort",
    "m_sin", "m_cos", "m_tan", "m_asinh", "m_erfc", "m_bessel_i0e",
    "m_digamma", "m_lgamma", "m_rint", "m_round", "m_sign", "m_log",
    "m_igamma", "m_igammac",
    "amp_floor", "amp_ceil", "amp_round", "amp_rint",
    "amp_eq_1", "amp_gt_half", "amp_lt_zero",
    "bs_check_numerics", "bs_resize_bilinear", "bs_searchsorted",
    "bs_eigvalsh", "bs_segment_min", "bs_segment_max",
    "bs_fake_quant", "bs_xlogy", "bs_xlog1py", "bs_pow_raw",
    "bs_floordiv", "bs_floormod", "bs_reciprocal", "bs_batchnorm",
}
for _ops, _fam in [
    (_ACTIVATION_OPS, "activation"), (_ARITHMETIC_OPS, "arithmetic"),
    (_REDUCTION_OPS, "reduction"), (_NORMALIZE_OPS, "normalize"),
    (_LINALG_OPS, "linalg"), (_STRUCTURAL_OPS, "structural"),
]:
    for op in _ops:
        _OP_TO_FAMILY[op] = _fam

_POISON_TO_REGIME = {
    "none": "standard", "light": "standard", "heavy": "standard", "extreme": "standard",
    "simd_boundary": "boundary",
    "size_1_inner": "degenerate", "zero_dim": "degenerate",
    "huge": "magnitude", "tiny": "magnitude", "cancellation": "magnitude",
    "mag_huge_size1": "magnitude",
}

_DTYPE_TO_CLASS = {
    "tf.float32": "float32", "tf.float64": "float64",
    "tf.complex64": "complex64", "tf.complex128": "complex128",
}


def _is_reachable(op_family, dtype_class, seam, shape_regime):
    if dtype_class in ("complex64", "complex128"):
        if seam == "mixed_precision":
            return False
        if op_family == "linalg" and seam != "eager_vs_grappler":
            return False
    if op_family == "linalg" and shape_regime == "degenerate":
        return False
    return True


class TFKernelStateSpace:
    def __init__(self):
        self.all_states = set()
        self.reachable_states = set()
        self.covered_states = set()
        self._hit_counts = defaultdict(int)

        for fam in OP_FAMILIES:
            for dt in DTYPE_CLASSES:
                for seam in EXECUTION_SEAMS:
                    for sr in SHAPE_REGIMES:
                        s = (fam, dt, seam, sr)
                        self.all_states.add(s)
                        if _is_reachable(fam, dt, seam, sr):
                            self.reachable_states.add(s)

    @property
    def total_reachable(self):
        return len(self.reachable_states)

    def state_from_program(self, prog_info, active_oracles):
        ops = prog_info.get("op_fingerprint", [])
        family_counts = defaultdict(int)
        for op in ops:
            fam = _OP_TO_FAMILY.get(op)
            if fam:
                family_counts[fam] += 1
        if "matmul" in ops:
            op_family = "linalg"
        else:
            op_family = max(family_counts, key=family_counts.get) if family_counts else "activation"

        dtype_class = _DTYPE_TO_CLASS.get(prog_info.get("dtype", "tf.float32"), "float32")
        poison = prog_info.get("poison_mode", "none")
        shape_regime = _POISON_TO_REGIME.get(poison, "standard")

        if "xla" in active_oracles:
            seam = "eager_vs_xla"
        elif "cpu_gpu" in active_oracles or "eager_cpu_gpu" in active_oracles:
            seam = "cpu_vs_gpu"
        elif "grad" in active_oracles or "eager_grad_cpu_gpu" in active_oracles:
            seam = "backward"
        elif "mixed" in active_oracles:
            seam = "mixed_precision"
        else:
            seam = "eager_vs_grappler"

        return (op_family, dtype_class, seam, shape_regime)

    def mark_covered(self, state):
        if state in self.reachable_states:
            self.covered_states.add(state)
            self._hit_counts[state] += 1

    def uncovered_states(self):
        return sorted(self.reachable_states - self.covered_states)

    def lowest_hit_states(self, n=10):
        candidates = []
        for s in self.reachable_states:
            candidates.append((self._hit_counts.get(s, 0), s))
        candidates.sort()
        return [s for _, s in candidates[:n]]

    def coverage_ratio(self):
        return len(self.covered_states) / max(self.total_reachable, 1)

    def coverage_by_dimension(self):
        by_family = defaultdict(int)
        by_dtype = defaultdict(int)
        by_seam = defaultdict(int)
        by_shape = defaultdict(int)
        for fam, dt, seam, sr in self.covered_states:
            by_family[fam] += 1
            by_dtype[dt] += 1
            by_seam[seam] += 1
            by_shape[sr] += 1
        return {
            "op_family": dict(by_family),
            "dtype_class": dict(by_dtype),
            "execution_seam": dict(by_seam),
            "shape_regime": dict(by_shape),
        }

    def stats(self):
        return {
            "total_reachable": self.total_reachable,
            "covered": len(self.covered_states),
            "coverage_ratio": round(self.coverage_ratio(), 3),
            "by_dimension": self.coverage_by_dimension(),
        }


_FAMILY_TO_WEIGHTS = {
    "activation": {
        "keras3_unary": 35, "tfnn": 25, "raw": 15,
        "keras3_binary": 3, "keras3_red": 3, "keras3_norm": 3,
        "tfmath": 5, "indexing": 2, "cast": 2, "where": 5,
        "semantic": 1, "amplifier": 1,
    },
    "arithmetic": {
        "keras3_binary": 50, "where": 15,
        "keras3_unary": 5, "keras3_red": 5, "keras3_norm": 3,
        "tfnn": 3, "tfmath": 5, "indexing": 3, "cast": 3,
        "raw": 3, "semantic": 2, "amplifier": 3,
    },
    "reduction": {
        "keras3_red": 40, "amplifier": 25,
        "keras3_unary": 5, "keras3_binary": 5, "keras3_norm": 5,
        "tfnn": 3, "tfmath": 3, "indexing": 3, "cast": 3,
        "where": 3, "raw": 3, "semantic": 2,
    },
    "normalize": {
        "keras3_norm": 45, "tfnn": 15,
        "keras3_unary": 5, "keras3_binary": 5, "keras3_red": 5,
        "tfmath": 3, "indexing": 3, "cast": 3, "where": 3,
        "raw": 5, "semantic": 3, "amplifier": 5,
    },
    "linalg": {
        "keras3_unary": 8, "keras3_binary": 5, "keras3_red": 8,
        "keras3_norm": 8, "tfnn": 5, "tfmath": 3, "indexing": 3,
        "cast": 3, "where": 3, "raw": 3, "semantic": 1, "amplifier": 5,
    },
    "structural": {
        "cast": 20, "indexing": 20, "tfmath": 25,
        "keras3_unary": 5, "keras3_binary": 3, "keras3_red": 5,
        "keras3_norm": 3, "tfnn": 3, "raw": 5, "where": 3,
        "semantic": 3, "amplifier": 5,
    },
}

_REGIME_TO_POISON_POOL = {
    "standard": ["none", "light", "heavy", "extreme"],
    "boundary": ["simd_boundary"],
    "degenerate": ["size_1_inner", "zero_dim"],
    "magnitude": ["huge", "tiny", "cancellation", "mag_huge_size1"],
}

_SEAM_TO_ORACLES = {
    "eager_vs_grappler": {"eager"},
    "eager_vs_xla": {"eager", "xla"},
    "cpu_vs_gpu": {"eager", "eager_cpu_gpu"},
    "backward": {"eager", "grad", "eager_grad_cpu_gpu"},
    "mixed_precision": {"eager", "mixed"},
}


class EpsilonGreedyScheduler:
    def __init__(self, state_space, epsilon=0.3, seed=None):
        self.state_space = state_space
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def select_target(self):
        uncovered = self.state_space.uncovered_states()
        if not uncovered or self.rng.random() < self.epsilon:
            return None
        low_hit = self.state_space.lowest_hit_states(20)
        uncovered_set = set(uncovered)
        prefer_uncovered = [s for s in low_hit if s in uncovered_set]
        if prefer_uncovered:
            return self.rng.choice(prefer_uncovered)
        return self.rng.choice(uncovered)

    def target_to_gen_params(self, target):
        if target is None:
            return {}
        op_family, dtype_class, seam, shape_regime = target
        params = {}
        weights = _FAMILY_TO_WEIGHTS.get(op_family)
        if weights:
            params["category_weights"] = dict(weights)
        if op_family == "linalg":
            params["force_matmul"] = True
        params["dtype_override"] = f"tf.{dtype_class}"
        poison_pool = _REGIME_TO_POISON_POOL.get(shape_regime, ["none"])
        params["poison_mode"] = self.rng.choice(poison_pool)
        params["_target_seam"] = seam
        return params

    def oracle_flags_for_target(self, target, base_flags):
        if target is None:
            return base_flags
        seam = target[2]
        seam_oracles = _SEAM_TO_ORACLES.get(seam, {"eager"})
        flags = dict(base_flags)
        flags["xla"] = "xla" in seam_oracles
        flags["grad"] = "grad" in seam_oracles
        flags["cpu_gpu"] = "cpu_gpu" in seam_oracles
        flags["eager_cpu_gpu"] = "eager_cpu_gpu" in seam_oracles
        flags["eager_grad_cpu_gpu"] = "eager_grad_cpu_gpu" in seam_oracles
        flags["mixed"] = "mixed" in seam_oracles
        return flags

    def active_oracle_names(self, oracle_flags):
        names = {"eager"}
        for k, v in oracle_flags.items():
            if v:
                names.add(k)
        return names

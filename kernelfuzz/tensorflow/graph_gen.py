"""Random TF computation graph generator.

Generates random pure-functional TF programs for differential fuzzing.
Op categories:
  - tf.keras.ops.*    (Keras 3, highest priority — new code paths)
  - tf.nn.*           (activations, conv, reductions)
  - tf.raw_ops.*      (low-level, stress lowering)
  - tf.math.*         (scalar math, special functions)
  - Amplifiers        (floor/ceil/argmax — turn 1-ULP diffs into observable bugs)

Design:
  - Ops categorized by shape rule
  - Random chains of 5-25 ops
  - Adversarial shapes (2^n +/- 1, primes, <= 500k elems)
  - Composition presets (matmul chains, cast chains, deep Keras3)
  - Op fingerprint tracking for bug basin dedup
"""
import os
import random
import numpy as np

# ---- Op definitions ----
# Each tuple: (name, template, shape_rule)

KERAS3_UNARY = [
    ("k3_sin",      "tf.keras.ops.sin({x})",            "same"),
    ("k3_cos",      "tf.keras.ops.cos({x})",            "same"),
    ("k3_tanh",     "tf.keras.ops.tanh({x})",           "same"),
    ("k3_exp",      "tf.keras.ops.exp(tf.clip_by_value({x}, -10.0, 10.0))", "same"),
    ("k3_log1p",    "tf.keras.ops.log1p(tf.abs({x}) + 1e-7)", "same"),
    ("k3_sigmoid",  "tf.keras.ops.sigmoid({x})",        "same"),
    ("k3_relu",     "tf.keras.ops.relu({x})",           "same"),
    ("k3_relu6",    "tf.keras.ops.relu6({x})",          "same"),
    ("k3_silu",     "tf.keras.ops.silu({x})",           "same"),
    ("k3_gelu",     "tf.keras.ops.gelu({x})",           "same"),
    ("k3_elu",      "tf.keras.ops.elu({x})",            "same"),
    ("k3_softplus", "tf.keras.ops.softplus({x})",       "same"),
    ("k3_softsign", "tf.keras.ops.softsign({x})",       "same"),
    ("k3_abs",      "tf.keras.ops.abs({x})",            "same"),
    ("k3_square",   "tf.keras.ops.square({x})",         "same"),
    ("k3_rsqrt",    "tf.keras.ops.rsqrt(tf.abs({x}) + 1e-7)", "same"),
    ("k3_erf",      "tf.keras.ops.erf({x})",            "same"),
    ("k3_expm1",    "tf.keras.ops.expm1(tf.clip_by_value({x}, -10.0, 10.0))", "same"),
    ("k3_sign",     "tf.keras.ops.sign({x})",           "same"),
    ("k3_ceil",     "tf.keras.ops.ceil({x})",           "same"),
    ("k3_floor",    "tf.keras.ops.floor({x})",          "same"),
    ("k3_arctan",   "tf.keras.ops.arctan({x})",         "same"),
    ("k3_arcsinh",  "tf.keras.ops.arcsinh({x})",        "same"),
    ("k3_log_sigmoid", "tf.keras.ops.log_sigmoid({x})", "same"),
]

KERAS3_BINARY = [
    ("k3_add",      "({a} + {b})",                                   "same"),
    ("k3_sub",      "({a} - {b})",                                   "same"),
    ("k3_mul",      "({a} * {b})",                                   "same"),
    ("k3_div",      "({a} / (tf.abs({b}) + 1e-7))",                  "same"),
    ("k3_max",      "tf.keras.ops.maximum({a}, {b})",                "same"),
    ("k3_min",      "tf.keras.ops.minimum({a}, {b})",                "same"),
    ("k3_power",    "tf.keras.ops.power(tf.abs({a})+1e-7, tf.clip_by_value(tf.abs({b}), 0.1, 3.0))", "same"),
    ("k3_hypot",    "tf.keras.ops.hypot({a}, {b})",                  "same"),
    ("k3_arctan2",  "tf.keras.ops.arctan2({a}, {b} + 1e-7)",        "same"),
    ("k3_logaddexp","tf.keras.ops.logaddexp({a}, {b})",              "same"),
    ("k3_maximum",  "tf.math.maximum({a}, {b})",                     "same"),
    ("k3_minimum",  "tf.math.minimum({a}, {b})",                     "same"),
]

KERAS3_REDUCTION = [
    ("k3_sum_last",   "tf.keras.ops.sum({x}, axis=-1, keepdims=True)",  "keepdim"),
    ("k3_mean_last",  "tf.keras.ops.mean({x}, axis=-1, keepdims=True)", "keepdim"),
    ("k3_max_last",   "tf.keras.ops.max({x}, axis=-1, keepdims=True)",  "keepdim"),
    ("k3_min_last",   "tf.keras.ops.min({x}, axis=-1, keepdims=True)",  "keepdim"),
    ("k3_prod_last",  "tf.keras.ops.prod(tf.clip_by_value({x}, 0.5, 1.5), axis=-1, keepdims=True)", "keepdim"),
    ("k3_logsumexp",  "tf.keras.ops.logsumexp({x}, axis=-1, keepdims=True)", "keepdim"),
    ("k3_std_last",   "tf.keras.ops.std({x}, axis=-1, keepdims=True) + 1e-7",  "keepdim"),
    ("k3_var_last",   "tf.keras.ops.var({x}, axis=-1, keepdims=True) + 1e-7",  "keepdim"),
    ("k3_norm_last",  "tf.keras.ops.norm({x}, axis=-1, keepdims=True)", "keepdim"),
    ("k3_cumsum",     "tf.keras.ops.cumsum({x}, axis=-1)",              "same"),
    ("k3_cumprod",    "tf.keras.ops.cumprod(tf.clip_by_value({x}, 0.5, 1.5), axis=-1)", "same"),
    ("k3_sum_0",      "tf.keras.ops.sum({x}, axis=0, keepdims=True)",   "keepdim_0"),
    ("k3_reduce_max", "tf.reduce_max({x}, axis=-1, keepdims=True)",     "keepdim"),
    ("k3_reduce_min", "tf.reduce_min({x}, axis=-1, keepdims=True)",     "keepdim"),
]

KERAS3_NORMALIZE = [
    ("k3_softmax",     "tf.keras.ops.softmax({x}, axis=-1)",            "same"),
    ("k3_log_softmax", "tf.keras.ops.log_softmax({x}, axis=-1)",        "same"),
    ("k3_l2_norm",     "({x} / (tf.keras.ops.norm({x}, axis=-1, keepdims=True) + 1e-7))", "same"),
    ("k3_layernorm",   "(lambda _t: (_t - tf.keras.ops.mean(_t, axis=-1, keepdims=True)) / (tf.keras.ops.std(_t, axis=-1, keepdims=True) + 1e-6))({x})", "same"),
    ("k3_rmsnorm",     "(lambda _t: _t / (tf.keras.ops.sqrt(tf.keras.ops.mean(tf.keras.ops.square(_t), axis=-1, keepdims=True)) + 1e-6))({x})", "same"),
]

TFNN_OPS = [
    ("nn_relu",    "tf.nn.relu({x})",         "same"),
    ("nn_elu",     "tf.nn.elu({x})",          "same"),
    ("nn_selu",    "tf.nn.selu({x})",         "same"),
    ("nn_gelu",    "tf.nn.gelu({x})",         "same"),
    ("nn_swish",   "tf.nn.swish({x})",        "same"),
    ("nn_silu",    "tf.nn.silu({x})",         "same"),
    ("nn_sigmoid", "tf.nn.sigmoid({x})",      "same"),
    ("nn_softplus","tf.nn.softplus({x})",     "same"),
    ("nn_tanh",    "tf.nn.tanh({x})",         "same"),
    ("nn_softmax", "tf.nn.softmax({x}, axis=-1)", "same"),
    ("nn_log_softmax","tf.nn.log_softmax({x}, axis=-1)", "same"),
]

TFMATH_OPS = [
    ("m_sin",      "tf.math.sin({x})",        "same"),
    ("m_cos",      "tf.math.cos({x})",        "same"),
    ("m_tan",      "tf.math.tan(tf.clip_by_value({x}, -1.5, 1.5))", "same"),
    ("m_asinh",    "tf.math.asinh({x})",      "same"),
    ("m_erfc",     "tf.math.erfc({x})",       "same"),
    ("m_bessel_i0e","tf.math.bessel_i0e({x})","same"),
    ("m_digamma",  "tf.math.digamma(tf.abs({x}) + 1.5)", "same"),
    ("m_lgamma",   "tf.math.lgamma(tf.abs({x}) + 1.5)",  "same"),
    ("m_rint",     "tf.math.rint({x})",       "same"),
    ("m_round",    "tf.math.round({x})",      "same"),
    ("m_sign",     "tf.math.sign({x})",       "same"),
    ("m_log",      "tf.math.log(tf.abs({x}) + 1e-7)", "same"),
    ("m_igamma",   "tf.math.igamma(tf.abs({x}) + 0.5, tf.abs({x}) + 0.5)", "same"),
    ("m_igammac",  "tf.math.igammac(tf.abs({x}) + 0.5, tf.abs({x}) + 0.5)", "same"),
]

CAST_OPS = [
    ("cast_bf16_f32", "tf.cast(tf.cast({x}, tf.bfloat16), tf.float32)", "same"),
    ("cast_f16_f32",  "tf.cast(tf.cast({x}, tf.float16), tf.float32)",  "same"),
    ("cast_f64_f32",  "tf.cast(tf.cast({x}, tf.float64), tf.float32)",  "same"),
]

INDEXING_OPS = [
    ("idx_slice_half", "{x}[..., :({x}.shape[-1]//2)]",                 "half_last"),
    ("idx_reverse",    "tf.reverse({x}, axis=[-1])",                     "same"),
    ("idx_roll",       "tf.roll({x}, shift=3, axis=-1)",                 "same"),
    ("idx_sort",       "tf.sort({x}, axis=-1)",                          "same"),
    ("idx_tile2",      "tf.tile({x}, tf.concat([tf.ones([tf.rank({x})-1], dtype=tf.int32), [2]], axis=0))", "double_last"),
    ("idx_argsort",    "tf.cast(tf.argsort({x}, axis=-1), tf.float32)", "same"),
]

WHERE_OPS = [
    ("where_pos", "tf.where({x} > 0, {x}, -{x} * 0.1)",          "same"),
    ("clip",      "tf.clip_by_value({x}, -5.0, 5.0)",             "same"),
    ("nan_to_num","tf.where(tf.math.is_finite({x}), {x}, tf.zeros_like({x}))", "same"),
]

RAW_OPS = [
    ("raw_sigmoid",   "tf.raw_ops.Sigmoid(x={x})",              "same"),
    ("raw_tanh",      "tf.raw_ops.Tanh(x={x})",                 "same"),
    ("raw_softplus",  "tf.raw_ops.Softplus(features={x})",      "same"),
    ("raw_relu",      "tf.raw_ops.Relu(features={x})",          "same"),
    ("raw_rsqrt",     "tf.raw_ops.Rsqrt(x=tf.abs({x}) + 1e-7)", "same"),
    ("raw_erf",       "tf.raw_ops.Erf(x={x})",                  "same"),
    ("raw_elu",       "tf.raw_ops.Elu(features={x})",           "same"),
]

AMPLIFIER_OPS = [
    ("amp_floor",       "tf.math.floor({x})",                       "same"),
    ("amp_ceil",        "tf.math.ceil({x})",                        "same"),
    ("amp_round",       "tf.math.round({x})",                       "same"),
    ("amp_rint",        "tf.math.rint({x})",                        "same"),
    ("amp_eq_1",        "tf.cast(tf.math.equal({x}, 1.0), tf.float32)", "same"),
    ("amp_gt_half",     "tf.cast({x} > 0.5, tf.float32)",           "same"),
    ("amp_lt_zero",     "tf.cast({x} < 0.0, tf.float32)",           "same"),
    ("amp_argmax",      "tf.cast(tf.argmax({x}, axis=-1), tf.float32)", "reduce_last"),
    ("amp_argmin",      "tf.cast(tf.argmin({x}, axis=-1), tf.float32)", "reduce_last"),
    ("amp_softmax_floor", "tf.math.floor(tf.nn.softmax({x}, axis=-1))", "same"),
]

SEMANTIC_MUTATIONS = [
    ("sem_identity", "tf.identity({x})",                      "same"),
    ("sem_clone",    "({x} + 0.0)",                           "same"),
    ("sem_reshape",  "tf.reshape({x}, tf.shape({x}))",        "same"),
    ("sem_stop_grad","tf.stop_gradient({x})",                 "same"),
    ("sem_cast_nop", "tf.cast({x}, tf.float32)",              "same"),
    ("sem_ensure",   "tf.ensure_shape({x}, {x}.shape)",       "same"),
]

# Bug-specific ops: operators that triggered confirmed bugs not covered above.
# T5/T6/T7 categories from paper appendix.
BUGSPEC_OPS = [
    # #117319: check_numerics silently passes NaN/Inf under XLA
    ("bs_check_numerics", "tf.debugging.check_numerics({x}, 'check')", "same"),
    # #117803: image.resize bilinear over-propagates NaN under XLA
    ("bs_resize_bilinear",
     "(lambda _t: tf.image.resize(tf.reshape(_t, [1] + list(_t.shape) + [1]), "
     "[max(1, _t.shape[0]//2 if _t.shape.rank==1 else _t.shape[-2]//2), "
     "max(1, _t.shape[-1]//2)], method='bilinear')[0,:,:,0])({x})",
     "quarter_last"),
    # #117804: searchsorted returns wrong position for NaN under XLA
    ("bs_searchsorted",
     "tf.cast(tf.searchsorted(tf.sort({x}, axis=-1), {x}, side='right'), tf.float32)",
     "same"),
    # #117807, #117864: eigvalsh wrong for large/complex matrices
    ("bs_eigvalsh",
     "(lambda _t: tf.linalg.eigvalsh(tf.linalg.matmul(_t, _t, transpose_b=True)))({x})",
     "keepdim"),
    # #117862: segment_min/max NaN-position-dependent results
    ("bs_segment_min",
     "(lambda _t: tf.math.unsorted_segment_min(_t, tf.zeros(tf.shape(_t)[0], dtype=tf.int32), 1))({x})",
     "scalar"),
    ("bs_segment_max",
     "(lambda _t: tf.math.unsorted_segment_max(_t, tf.zeros(tf.shape(_t)[0], dtype=tf.int32), 1))({x})",
     "scalar"),
    # #117962: fake_quant nudge rounding mismatch
    ("bs_fake_quant",
     "tf.quantization.fake_quant_with_min_max_vars({x}, min=-6.0, max=6.0, num_bits=8)",
     "same"),
    # #119476: xlogy/xlog1py gradient at zero
    ("bs_xlogy",
     "tf.math.xlogy({x}, tf.abs({x}) + 1e-7)",
     "same"),
    ("bs_xlog1py",
     "tf.math.xlog1py({x}, tf.abs({x}))",
     "same"),
    # #117802: complex pow(0+0j, 0+0j) returns NaN (unguarded power)
    ("bs_pow_raw",
     "tf.math.pow(tf.abs({x}), {x} * 0.5)",
     "same"),
    # #118642: floordiv/floormod identity violation under XLA
    ("bs_floordiv",
     "tf.cast(tf.math.floordiv({x}, tf.ones_like({x}) * 2.0 + 1e-7), {x}.dtype)",
     "same"),
    ("bs_floormod",
     "tf.math.floormod({x}, tf.ones_like({x}) * 3.0 + 1e-7)",
     "same"),
    # #118645: reciprocal flush-to-zero on GPU
    ("bs_reciprocal",
     "tf.math.reciprocal({x} + 1e-38)",
     "same"),
    # #118701: fused_batch_norm with negative variance under XLA
    ("bs_batchnorm",
     "(lambda _t: tf.nn.batch_normalization("
     "tf.reshape(_t, [1, 1, 1, -1]), "
     "tf.zeros([tf.shape(_t)[-1]]), tf.ones([tf.shape(_t)[-1]]), "
     "tf.zeros([tf.shape(_t)[-1]]), tf.ones([tf.shape(_t)[-1]]), 1e-5)"
     "[0, 0, 0])({x})",
     "same"),
]


# ---- Shape utilities ----

def _shape_after_op(shape, rule):
    if rule == "same" or rule == "keepdim":
        return shape
    elif rule == "scalar":
        return ()
    elif rule == "half_last":
        if not shape:
            return shape
        return shape[:-1] + (max(1, shape[-1] // 2),)
    elif rule == "double_last":
        if not shape:
            return shape
        return shape[:-1] + (shape[-1] * 2,)
    elif rule == "quarter_last":
        if not shape:
            return shape
        return shape[:-1] + (max(1, shape[-1] // 4),)
    elif rule == "keepdim_0":
        if len(shape) < 2:
            return shape
        return (1,) + shape[1:]
    return shape


# ---- Adversarial shape generation ----

BOUNDARY_SIZES = [7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513]
PRIMES = [13, 29, 53, 97, 127, 257, 521]
TYPICAL = [32, 64, 128, 256, 512, 1024]
ODD_SMALL = [3, 5, 9, 11, 19, 37, 71, 131]


class _SkipOp(Exception):
    pass


class GraphGenerator:
    """Generate random TF computation graphs."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._op_fingerprint = set()

    def gen_shape(self, max_elems=500_000):
        ndim = self.rng.choices([1, 2, 3, 4], weights=[10, 40, 35, 15], k=1)[0]
        sizes = []
        for _ in range(ndim):
            kind = self.rng.choice(["boundary", "prime", "typical", "odd"])
            if kind == "boundary":
                sizes.append(self.rng.choice(BOUNDARY_SIZES))
            elif kind == "prime":
                sizes.append(self.rng.choice(PRIMES))
            elif kind == "typical":
                sizes.append(self.rng.choice(TYPICAL))
            else:
                sizes.append(self.rng.choice(ODD_SMALL))
        total = 1
        for s in sizes:
            total *= s
        while total > max_elems and len(sizes) > 1:
            sizes.pop(0)
            total = 1
            for s in sizes:
                total *= s
        return tuple(sizes)

    def gen_matmul_shapes(self):
        use_batch = self.rng.random() < 0.6
        M = self.rng.choice(BOUNDARY_SIZES[:8] + TYPICAL[:3] + PRIMES[:4])
        K = self.rng.choice(BOUNDARY_SIZES[:8] + TYPICAL[:3] + PRIMES[:4])
        N = self.rng.choice(BOUNDARY_SIZES[:8] + TYPICAL[:3] + PRIMES[:4])
        if use_batch:
            B = self.rng.choice([1, 2, 4, 8, 16])
            return (B, M, K), (B, K, N), (B, M, N)
        return (M, K), (K, N), (M, N)

    def _apply_template(self, template, var_name):
        return template.replace("{x}", var_name)

    def _apply_binary_template(self, template, var_a, var_b):
        return template.replace("{a}", var_a).replace("{b}", var_b)

    def _pick_var(self, vars_available, min_ndim=0, same_shape_as=None):
        candidates = [v for v in vars_available if len(v[1]) >= min_ndim]
        if same_shape_as:
            exact = [v for v in candidates if v[1] == same_shape_as]
            if exact:
                return self.rng.choice(exact)
        if not candidates:
            candidates = vars_available
        return self.rng.choice(candidates)

    def gen_program(self, n_ops=None, force_matmul=False, cast_heavy=False,
                    category_weights=None, dtype_override=None):
        if n_ops is None:
            n_ops = self.rng.randint(5, 25)

        if dtype_override:
            dtype = dtype_override
        else:
            dtype = self.rng.choices(
                ["tf.float32", "tf.float64", "tf.complex64", "tf.complex128"],
                weights=[70, 15, 10, 5], k=1
            )[0]
        self._is_complex = "complex" in dtype

        has_matmul = force_matmul or (self.rng.random() < 0.25)

        if has_matmul:
            shape_a, shape_b, shape_out = self.gen_matmul_shapes()
            n_inputs = 2
            input_shapes = [shape_a, shape_b]
        else:
            base_shape = self.gen_shape()
            n_inputs = self.rng.choices([1, 2, 3], weights=[40, 40, 20], k=1)[0]
            input_shapes = [base_shape] * n_inputs

        lines = []
        lines.append("def fn({}):".format(", ".join(f"x{i}" for i in range(n_inputs))))

        vars_available = [(f"x{i}", input_shapes[i]) for i in range(n_inputs)]
        var_counter = 0
        self._op_fingerprint = set()

        matmul_done = False
        matmul_insert_at = self.rng.randint(0, min(3, n_ops - 1)) if has_matmul else -1

        for step in range(n_ops):
            if has_matmul and not matmul_done and step >= matmul_insert_at:
                matmul_done = True
                vname = f"v{var_counter}"
                var_counter += 1
                expr = "tf.linalg.matmul(x0, x1)"
                lines.append(f"    {vname} = {expr}")
                vars_available.append((vname, shape_out))
                self._op_fingerprint.add("matmul")
                continue

            graph_preset = os.environ.get("TF_FUZZER_GRAPH_PRESET", "v2")
            if category_weights is not None:
                weights = category_weights
            elif graph_preset == "v3_tf_only":
                weights = {
                    "keras3_unary":    6,
                    "keras3_binary":   4,
                    "keras3_red":     18,
                    "keras3_norm":    22,
                    "tfnn":           14,
                    "tfmath":          8,
                    "indexing":        4,
                    "cast":            2,
                    "where":           3,
                    "raw":             8,
                    "semantic":        3,
                    "amplifier":       8,
                    "bugspec":         5,
                }
            else:
                weights = {
                    "keras3_unary":   18,
                    "keras3_binary":   8,
                    "keras3_red":     10,
                    "keras3_norm":     8,
                    "tfnn":            6,
                    "tfmath":          5,
                    "indexing":        5,
                    "cast":            6 if cast_heavy else 2,
                    "where":           5,
                    "raw":             5,
                    "semantic":        3,
                    "amplifier":      10,
                    "bugspec":         5,
                }
            cat = self.rng.choices(list(weights.keys()),
                                   weights=list(weights.values()), k=1)[0]

            try:
                expr, out_shape = self._gen_op(cat, vars_available)
            except _SkipOp:
                continue

            actual_vname = f"v{var_counter}"
            var_counter += 1
            lines.append(f"    {actual_vname} = {expr}")
            vars_available.append((actual_vname, out_shape))

            if len(vars_available) > 10:
                keep_inputs = [(n, s) for n, s in vars_available if n.startswith("x")]
                keep_recent = vars_available[-6:]
                seen = set()
                merged = []
                for v in keep_inputs + keep_recent:
                    if v[0] not in seen:
                        seen.add(v[0])
                        merged.append(v)
                vars_available = merged

        last_var = vars_available[-1][0]
        last_shape = vars_available[-1][1]
        if last_shape == ():
            lines.append(f"    return {last_var}")
        else:
            lines.append(f"    return tf.reduce_sum({last_var})")

        code = "\n".join(lines)
        return code, input_shapes, dtype, n_inputs, frozenset(self._op_fingerprint)

    _COMPLEX_UNSAFE_CATS = {"amplifier", "where", "indexing"}
    _COMPLEX_SAFE_OPS = {"k3_sin", "k3_cos", "k3_tanh", "k3_exp", "k3_sigmoid",
                         "k3_abs", "k3_square", "k3_sign", "k3_erf", "k3_log1p",
                         "nn_relu", "nn_elu", "nn_selu", "nn_sigmoid", "nn_tanh",
                         "m_sin", "m_cos", "m_exp", "m_abs", "m_sign", "m_conj",
                         "m_log",
                         "matmul"}

    def _gen_op(self, cat, vars_available):
        if self._is_complex and cat in self._COMPLEX_UNSAFE_CATS:
            raise _SkipOp()
        if cat == "keras3_unary":
            name, tmpl, rule = self.rng.choice(KERAS3_UNARY)
            src = self._pick_var(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "keras3_binary":
            name, tmpl, rule = self.rng.choice(KERAS3_BINARY)
            src1 = self._pick_var(vars_available)
            src2 = self._pick_var(vars_available, same_shape_as=src1[1])
            expr = self._apply_binary_template(tmpl, src1[0], src2[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src1[1], rule)
        elif cat == "keras3_red":
            name, tmpl, rule = self.rng.choice(KERAS3_REDUCTION)
            src = self._pick_var(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if rule == "keepdim_0" and len(src[1]) < 2:
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "keras3_norm":
            name, tmpl, rule = self.rng.choice(KERAS3_NORMALIZE)
            src = self._pick_var(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if src[1][-1] < 2 and name not in ("k3_softmax", "k3_log_softmax"):
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "tfnn":
            name, tmpl, rule = self.rng.choice(TFNN_OPS)
            src = self._pick_var(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "tfmath":
            _TFMATH_COMPLEX_OK = {"m_sin", "m_cos", "m_asinh", "m_sign", "m_log"}
            pool = TFMATH_OPS
            if self._is_complex:
                pool = [op for op in TFMATH_OPS if op[0] in _TFMATH_COMPLEX_OK]
                if not pool:
                    raise _SkipOp()
            name, tmpl, rule = self.rng.choice(pool)
            src = self._pick_var(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "indexing":
            name, tmpl, rule = self.rng.choice(INDEXING_OPS)
            src = self._pick_var(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if name == "idx_slice_half" and src[1][-1] < 2:
                raise _SkipOp()
            if name == "idx_tile2" and src[1][-1] > 256:
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "cast":
            name, tmpl, rule = self.rng.choice(CAST_OPS)
            src = self._pick_var(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "where":
            name, tmpl, rule = self.rng.choice(WHERE_OPS)
            src = self._pick_var(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "raw":
            name, tmpl, rule = self.rng.choice(RAW_OPS)
            src = self._pick_var(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "semantic":
            name, tmpl, rule = self.rng.choice(SEMANTIC_MUTATIONS)
            src = self._pick_var(vars_available)
            expr = self._apply_template(tmpl, src[0])
            return expr, _shape_after_op(src[1], rule)
        elif cat == "amplifier":
            name, tmpl, rule = self.rng.choice(AMPLIFIER_OPS)
            src = self._pick_var(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if rule == "reduce_last" and src[1][-1] < 2:
                raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        elif cat == "bugspec":
            pool = BUGSPEC_OPS
            if self._is_complex:
                pool = [op for op in pool if op[0] in ("bs_eigvalsh", "bs_pow_raw")]
            if not pool:
                raise _SkipOp()
            name, tmpl, rule = self.rng.choice(pool)
            src = self._pick_var(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if name == "bs_eigvalsh" and len(src[1]) < 2:
                raise _SkipOp()
            if name == "bs_resize_bilinear" and len(src[1]) < 2:
                raise _SkipOp()
            if name in ("bs_segment_min", "bs_segment_max") and len(src[1]) != 1:
                src = self._pick_var(vars_available, min_ndim=1)
                if len(src[1]) != 1:
                    raise _SkipOp()
            expr = self._apply_template(tmpl, src[0])
            self._op_fingerprint.add(name)
            return expr, _shape_after_op(src[1], rule)
        raise _SkipOp()


# ---- Composition presets ----

class CompositionGenerator:
    STAGE_PRESETS = [
        ("matmul_heavy", {"force_matmul": True}, (5, 10)),
        ("matmul_then_norm", {"force_matmul": True}, (3, 7)),
        ("deep_keras3", {}, (15, 25)),
        ("cast_stress", {"cast_heavy": True}, (8, 15)),
        ("norm_chain", {}, (8, 15)),
    ]

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.gen = GraphGenerator(seed=seed)

    def gen_composed_program(self):
        preset_name, kwargs, (lo, hi) = self.rng.choice(self.STAGE_PRESETS)
        n_ops = self.rng.randint(lo, hi)
        code, shapes, dtype, n_inputs, fingerprint = self.gen.gen_program(
            n_ops=n_ops, **kwargs
        )
        return code, shapes, dtype, n_inputs, fingerprint, preset_name


# ---- Public API ----

def generate_program_with_inputs(seed, use_composition=False, poison_mode=None,
                                  dtype_override=None, category_weights=None,
                                  force_matmul=False):
    """Generate complete program + input tensors as a dict for eval."""
    if use_composition and not (dtype_override or category_weights or force_matmul):
        cgen = CompositionGenerator(seed=seed)
        code, shapes, dtype, n_inputs, fingerprint, preset = cgen.gen_composed_program()
    else:
        gen = GraphGenerator(seed=seed)
        code, shapes, dtype, n_inputs, fingerprint = gen.gen_program(
            force_matmul=force_matmul,
            dtype_override=dtype_override,
            category_weights=category_weights,
        )
        preset = None

    if poison_mode is None:
        preset_env = os.environ.get("TF_FUZZER_POISON_PRESET", "v2")
        prng = random.Random(seed * 7919)
        if preset_env == "v3":
            r = prng.random()
            if   r < 0.35: poison_mode = "none"
            elif r < 0.40: poison_mode = "light"
            elif r < 0.43: poison_mode = "heavy"
            elif r < 0.50: poison_mode = "huge"
            elif r < 0.53: poison_mode = "tiny"
            elif r < 0.56: poison_mode = "zero_dim"
            elif r < 0.58: poison_mode = "extreme"
            elif r < 0.78: poison_mode = "size_1_inner"
            elif r < 0.90: poison_mode = "simd_boundary"
            elif r < 0.95: poison_mode = "mag_huge_size1"
            else:          poison_mode = "cancellation"
        else:
            r = prng.random()
            if   r < 0.70: poison_mode = "none"
            elif r < 0.80: poison_mode = "light"
            elif r < 0.85: poison_mode = "heavy"
            elif r < 0.90: poison_mode = "huge"
            elif r < 0.95: poison_mode = "tiny"
            elif r < 0.98: poison_mode = "zero_dim"
            else:          poison_mode = "extreme"

    setup_lines = [
        "import tensorflow as tf",
        "import numpy as np",
        "import os, sys",
        f"sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tensorflow'))",
        "from poison import gen_poisoned_input",
    ]
    for i in range(n_inputs):
        shape_str = repr(shapes[i])
        setup_lines.append(
            f"x{i} = tf.constant(gen_poisoned_input({shape_str}, {dtype!r}, {seed + i}, {poison_mode!r}), dtype={dtype})"
        )

    setup = "\n".join(setup_lines)
    call_args = ", ".join(f"x{i}" for i in range(n_inputs))

    return {
        "setup": setup,
        "fn_code": code,
        "call_args": call_args,
        "n_inputs": n_inputs,
        "shapes": [list(s) for s in shapes],
        "dtype": dtype,
        "seed": seed,
        "poison_mode": poison_mode,
        "op_fingerprint": sorted(fingerprint),
        "composition_preset": preset,
    }

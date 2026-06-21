"""Random JAX computation graph generator — v2.

Generates random pure-functional JAX programs for differential fuzzing.
Compared to v1: broader op pool (matmul, conv, gather/scatter, scan),
multi-shape tracking, composition chains, semantic mutation.

Design informed by graph_exp's GraphFuzzerV4 but adapted for JAX/XLA.
"""
import random
import numpy as np

# ─────────────────────────────────────────────────────────────
# Op definitions
# ─────────────────────────────────────────────────────────────
# Each op: (name, template, shape_rule)
#   shape_rule: 'same'         — output shape == input shape
#               'keepdim'      — reduction with keepdims=True (same shape)
#               'reduce_last'  — drops last dim
#               'scalar'       — returns scalar
#               'half_last'    — last dim halved
#               'cumulative'   — same shape (cumsum etc)
#               'matmul'       — special handling
#               'conv'         — special handling
#               'gather'       — special handling

POINTWISE_UNARY = [
    ("sin",        "jnp.sin({x})",                        "same"),
    ("cos",        "jnp.cos({x})",                        "same"),
    ("tan",        "jnp.tan(jnp.clip({x}, -1.5, 1.5))",  "same"),
    ("tanh",       "jnp.tanh({x})",                       "same"),
    ("exp",        "jnp.exp(jnp.clip({x}, -10, 10))",     "same"),
    ("log1p",      "jnp.log1p(jnp.abs({x}) + 1e-7)",     "same"),
    ("abs",        "jnp.abs({x})",                        "same"),
    ("neg",        "jnp.negative({x})",                   "same"),
    ("sigmoid",    "jax.nn.sigmoid({x})",                 "same"),
    ("relu",       "jax.nn.relu({x})",                    "same"),
    ("relu6",      "jax.nn.relu6({x})",                   "same"),
    ("silu",       "jax.nn.silu({x})",                    "same"),
    ("gelu",       "jax.nn.gelu({x})",                    "same"),
    ("elu",        "jax.nn.elu({x})",                     "same"),
    ("softplus",   "jax.nn.softplus({x})",                "same"),
    ("rsqrt",      "jax.lax.rsqrt(jnp.abs({x}) + 1e-7)", "same"),
    ("erf",        "jax.lax.erf({x})",                    "same"),
    ("erfc",       "jax.lax.erfc({x})",                   "same"),
    ("square",     "jnp.square({x})",                     "same"),
    ("reciprocal", "jax.lax.reciprocal(jnp.abs({x}) + 1e-7)", "same"),
    ("sign",       "jnp.sign({x})",                       "same"),
    ("ceil",       "jnp.ceil({x})",                       "same"),
    ("floor",      "jnp.floor({x})",                      "same"),
    ("expm1",      "jnp.expm1(jnp.clip({x}, -10, 10))",  "same"),
    ("cbrt",       "jnp.cbrt(jnp.abs({x}) + 1e-7)",      "same"),
    ("sinh",       "jnp.sinh(jnp.clip({x}, -5, 5))",     "same"),
    ("cosh",       "jnp.cosh(jnp.clip({x}, -5, 5))",     "same"),
    ("arctan",     "jnp.arctan({x})",                     "same"),
    ("arcsinh",    "jnp.arcsinh({x})",                    "same"),
    ("log_sigmoid","jax.nn.log_sigmoid({x})",             "same"),
    # bitwise-style through float
    ("round",      "jnp.round({x})",                      "same"),
    ("rint",       "jax.lax.round({x}, jax.lax.RoundingMode.TO_NEAREST_EVEN)", "same"),
]

POINTWISE_BINARY = [
    ("add",     "({a} + {b})",                                           "same"),
    ("sub",     "({a} - {b})",                                           "same"),
    ("mul",     "({a} * {b})",                                           "same"),
    ("div",     "({a} / (jnp.abs({b}) + 1e-7))",                        "same"),
    ("max2",    "jnp.maximum({a}, {b})",                                 "same"),
    ("min2",    "jnp.minimum({a}, {b})",                                 "same"),
    ("pow2",    "jnp.power(jnp.abs({a})+1e-7, jnp.clip(jnp.abs({b}), 0.1, 3.0))", "same"),
    ("hypot",   "jnp.hypot({a}, {b})",                                  "same"),
    ("atan2",   "jnp.arctan2({a}, {b} + 1e-7)",                         "same"),
    ("logaddexp","jnp.logaddexp({a}, {b})",                              "same"),
    ("where_gt","jnp.where({a} > {b}, {a}, {b} * 0.1)",                 "same"),
]

REDUCTION = [
    ("sum_last",   "jnp.sum({x}, axis=-1, keepdims=True)",              "keepdim"),
    ("mean_last",  "jnp.mean({x}, axis=-1, keepdims=True)",             "keepdim"),
    ("max_last",   "jnp.max({x}, axis=-1, keepdims=True)",              "keepdim"),
    ("min_last",   "jnp.min({x}, axis=-1, keepdims=True)",              "keepdim"),
    ("prod_last",  "jnp.prod(jnp.clip({x}, 0.5, 1.5), axis=-1, keepdims=True)", "keepdim"),
    ("logsumexp",  "jax.scipy.special.logsumexp({x}, axis=-1, keepdims=True)", "keepdim"),
    ("std_last",   "jnp.std({x}, axis=-1, keepdims=True) + 1e-7",      "keepdim"),
    ("var_last",   "jnp.var({x}, axis=-1, keepdims=True) + 1e-7",      "keepdim"),
    ("norm_last",  "jnp.linalg.norm({x}, axis=-1, keepdims=True)",      "keepdim"),
    ("cumsum",     "jnp.cumsum({x}, axis=-1)",                           "same"),
    ("cumprod",    "jnp.cumprod(jnp.clip({x}, 0.5, 1.5), axis=-1)",     "same"),
    # reduce to scalar
    ("sum_all",    "jnp.sum({x})",                                       "scalar"),
    ("mean_all",   "jnp.mean({x})",                                      "scalar"),
    # multi-axis reduction (for ndim >= 2)
    ("sum_01",     "jnp.sum({x}, axis=0, keepdims=True)",                "keepdim_0"),
]

NORMALIZE = [
    ("softmax",     "jax.nn.softmax({x}, axis=-1)",                      "same"),
    ("log_softmax", "jax.nn.log_softmax({x}, axis=-1)",                  "same"),
    ("l2_norm",     "({x} / (jnp.linalg.norm({x}, axis=-1, keepdims=True) + 1e-7))", "same"),
    ("layernorm",   "(lambda _t: (_t - jnp.mean(_t, axis=-1, keepdims=True)) / (jnp.std(_t, axis=-1, keepdims=True) + 1e-6))({x})", "same"),
    ("rmsnorm",     "(lambda _t: _t / (jnp.sqrt(jnp.mean(jnp.square(_t), axis=-1, keepdims=True)) + 1e-6))({x})", "same"),
]

INDEXING = [
    ("slice_half",  "{x}[..., :({x}.shape[-1]//2)]",                     "half_last"),
    ("flip",        "jnp.flip({x}, axis=-1)",                            "same"),
    ("roll",        "jnp.roll({x}, 3, axis=-1)",                         "same"),
    ("sort",        "jnp.sort({x}, axis=-1)",                            "same"),
    ("reverse",     "jax.lax.rev({x}, dimensions=[{x}.ndim - 1])",      "same"),
    ("tile2",       "jnp.tile({x}, (1,)*({x}.ndim-1) + (2,))",          "double_last"),
    ("repeat2",     "jnp.repeat({x}, 2, axis=-1)",                      "double_last"),
]

CAST = [
    ("to_bf16_f32", "{x}.astype(jnp.bfloat16).astype(jnp.float32)",     "same"),
    ("to_f16_f32",  "{x}.astype(jnp.float16).astype(jnp.float32)",      "same"),
    ("to_f64_f32",  "{x}.astype(jnp.float64).astype(jnp.float32)",      "same"),
    ("to_i32_f32",  "jnp.round({x}).astype(jnp.int32).astype(jnp.float32)", "same"),
]

WHERE_COND = [
    ("where_pos",   "jnp.where({x} > 0, {x}, -{x} * 0.1)",             "same"),
    ("clamp",       "jnp.clip({x}, -5.0, 5.0)",                         "same"),
    ("nan_to_num",  "jnp.nan_to_num({x}, nan=0.0, posinf=1e4, neginf=-1e4)", "same"),
    ("select_topk", "(lambda _t: jnp.where(jnp.argsort(jnp.argsort(_t, axis=-1), axis=-1) >= _t.shape[-1]//2, _t, 0.0))({x})", "same"),
]

# --- New categories for v2 ---

MATMUL_OPS = [
    # (name, template_fn, description)
    # These are handled specially in the generator — template is a function
    # that takes (var_a, var_b, shape_a, shape_b) and returns (expr, out_shape)
    "matmul",
    "batched_matmul",
]

LAX_PRIMITIVES = [
    ("lax_clamp",     "jax.lax.clamp(jnp.float32(-3.0), {x}, jnp.float32(3.0))", "same"),
    ("lax_shift_left","jax.lax.shift_left(jnp.round({x}).astype(jnp.int32), jnp.int32(1)).astype(jnp.float32)", "same"),
    ("lax_population_count", "jax.lax.population_count(jnp.round(jnp.abs({x})).astype(jnp.int32)).astype(jnp.float32)", "same"),
    ("lax_select",    "jax.lax.select({x} > 0, {x}, {x} * 0.01)",      "same"),
    ("lax_sort",      "jax.lax.sort({x}, dimension=-1)",                 "same"),
    ("lax_top_k_val", "jax.lax.top_k({x}, max(1, {x}.shape[-1]//4))[0]", "quarter_last"),
]

SCAN_OPS = [
    # associative_scan is interesting because XLA must lower it to efficient parallel prefix
    ("assoc_scan_add", "jax.lax.associative_scan(jax.lax.add, {x}, axis=-1)", "same"),
    ("assoc_scan_mul", "jax.lax.associative_scan(lambda a, b: a * b, jnp.clip({x}, 0.9, 1.1), axis=-1)", "same"),
    ("assoc_scan_max", "jax.lax.associative_scan(jnp.maximum, {x}, axis=-1)", "same"),
]

# Semantic mutations: neutral ops that change compiler scheduling
SEMANTIC_MUTATIONS = [
    ("clone",    "({x} + 0.0)",              "same"),
    ("identity", "jnp.array({x})",           "same"),
    ("reshape",  "{x}.reshape({x}.shape)",   "same"),
    ("cast_nop", "{x}.astype(jnp.float32)",  "same"),
    ("broadcast","jnp.broadcast_to({x}, {x}.shape)", "same"),
    ("transpose_back", "(lambda _t: jnp.transpose(jnp.transpose(_t)) if _t.ndim >= 2 else _t)({x})", "same"),
    ("stop_grad","jax.lax.stop_gradient({x})", "same"),
]


# ─────────────────────────────────────────────────────────────
# Shape utilities
# ─────────────────────────────────────────────────────────────

def _shape_after_op(shape, rule):
    """Compute output shape given input shape and shape rule."""
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
    elif rule == "reduce_last":
        if not shape:
            return shape
        return shape[:-1] if len(shape) > 1 else (1,)
    return shape


# ─────────────────────────────────────────────────────────────
# Adversarial shape generation
# ─────────────────────────────────────────────────────────────

BOUNDARY_SIZES = [7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513]
PRIMES = [13, 29, 53, 97, 127, 257, 521]
TYPICAL = [32, 64, 128, 256, 512, 1024]
ODD_SMALL = [3, 5, 9, 11, 19, 37, 71, 131]


class GraphGenerator:
    """Generate random JAX computation graphs — v2."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._op_fingerprint = set()  # track which ops are used

    def gen_shape(self, max_elems=500_000):
        """Generate an adversarial input shape, capped at max_elems total."""
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
        # Cap total elements to avoid OOM
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
        """Generate compatible shapes for matmul: (B, M, K) @ (B, K, N) -> (B, M, N)."""
        use_batch = self.rng.random() < 0.6
        M = self.rng.choice(BOUNDARY_SIZES[:8] + TYPICAL[:3] + PRIMES[:4])
        K = self.rng.choice(BOUNDARY_SIZES[:8] + TYPICAL[:3] + PRIMES[:4])
        N = self.rng.choice(BOUNDARY_SIZES[:8] + TYPICAL[:3] + PRIMES[:4])
        if use_batch:
            B = self.rng.choice([1, 2, 4, 8, 16])
            return (B, M, K), (B, K, N), (B, M, N)
        return (M, K), (K, N), (M, N)

    def _pick_var_with_shape(self, vars_available, min_ndim=1, preferred_shape=None):
        """Pick a variable, optionally matching shape constraints."""
        candidates = [v for v in vars_available if len(v[1]) >= min_ndim]
        if not candidates:
            candidates = vars_available
        if preferred_shape:
            exact = [v for v in candidates if v[1] == preferred_shape]
            if exact:
                return self.rng.choice(exact)
        return self.rng.choice(candidates)

    def _apply_template(self, template, var_name):
        """Apply a unary op template, replacing {x} with var_name."""
        return template.replace("{x}", var_name)

    def _apply_binary_template(self, template, var_a, var_b):
        """Apply a binary op template."""
        return template.replace("{a}", var_a).replace("{b}", var_b)

    def gen_program(self, n_ops=None, force_matmul=False, force_scan=False):
        """Generate a random program.

        Returns: (code_str, input_shapes_list, dtype_str, n_inputs, op_fingerprint)
        """
        if n_ops is None:
            n_ops = self.rng.randint(5, 25)

        dtype = self.rng.choices(
            ["jnp.float32", "jnp.float64"],
            weights=[70, 30], k=1
        )[0]

        # Decide if this program has a matmul component
        has_matmul = force_matmul or (self.rng.random() < 0.3)
        has_scan = force_scan or (self.rng.random() < 0.2)

        # Generate input shapes
        if has_matmul:
            shape_a, shape_b, shape_out = self.gen_matmul_shapes()
            n_inputs = 2
            input_shapes = [shape_a, shape_b]
            # We'll create the matmul early in the program
        else:
            base_shape = self.gen_shape()
            n_inputs = self.rng.choices([1, 2, 3], weights=[40, 40, 20], k=1)[0]
            input_shapes = [base_shape] * n_inputs

        lines = []
        lines.append("def fn({}):".format(", ".join(f"x{i}" for i in range(n_inputs))))

        # Track available variables: (name, shape)
        vars_available = [(f"x{i}", input_shapes[i]) for i in range(n_inputs)]
        var_counter = 0
        self._op_fingerprint = set()

        # If matmul program, insert matmul at a random early position
        matmul_done = False
        matmul_insert_at = self.rng.randint(0, min(3, n_ops - 1)) if has_matmul else -1

        for step in range(n_ops):
            # Insert matmul at designated step
            if has_matmul and not matmul_done and step >= matmul_insert_at:
                matmul_done = True
                vname = f"v{var_counter}"
                var_counter += 1
                is_batched = len(shape_a) == 3
                if is_batched:
                    expr = f"jnp.matmul(x0, x1)"
                else:
                    expr = f"x0 @ x1"
                lines.append(f"    {vname} = {expr}")
                vars_available.append((vname, shape_out))
                self._op_fingerprint.add("matmul")
                continue

            # Pick operation category
            weights = {
                "pw_unary": 25,
                "pw_binary": 15,
                "reduction": 15,
                "normalize": 10,
                "indexing": 8,
                "cast": 2,  # low weight: cast bugs are mostly known (#anon-1)
                "where": 8,
                "lax": 5,
                "scan": 8 if has_scan else 2,
                "semantic_mut": 4,
            }
            cat = self.rng.choices(
                list(weights.keys()),
                weights=list(weights.values()),
                k=1
            )[0]

            try:
                vname, expr, out_shape = self._gen_op(cat, vars_available)
            except _SkipOp:
                continue

            if vname is None:
                continue

            actual_vname = f"v{var_counter}"
            var_counter += 1
            lines.append(f"    {actual_vname} = {expr}")
            vars_available.append((actual_vname, out_shape))

            # Prune old vars to force reuse
            if len(vars_available) > 10:
                # Keep inputs and recent vars
                keep_inputs = [(n, s) for n, s in vars_available if n.startswith("x")]
                keep_recent = vars_available[-6:]
                seen = set()
                merged = []
                for v in keep_inputs + keep_recent:
                    if v[0] not in seen:
                        seen.add(v[0])
                        merged.append(v)
                vars_available = merged

        # Final output: reduce to scalar for easy comparison
        last_var = vars_available[-1][0]
        last_shape = vars_available[-1][1]
        if last_shape == ():
            lines.append(f"    return {last_var}")
        else:
            lines.append(f"    return jnp.sum({last_var})")

        code = "\n".join(lines)
        return code, input_shapes, dtype, n_inputs, frozenset(self._op_fingerprint)

    def _gen_op(self, cat, vars_available):
        """Generate a single op. Returns (placeholder_vname, expr, out_shape) or raises _SkipOp."""
        if cat == "pw_unary":
            name, template, rule = self.rng.choice(POINTWISE_UNARY)
            src = self.rng.choice(vars_available)
            if src[1] == ():  # skip scalar for some ops
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "pw_binary":
            name, template, rule = self.rng.choice(POINTWISE_BINARY)
            src1 = self.rng.choice(vars_available)
            # Find a var with matching shape for binary ops
            same_shape = [v for v in vars_available if v[1] == src1[1]]
            src2 = self.rng.choice(same_shape) if same_shape else src1
            expr = self._apply_binary_template(template, src1[0], src2[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src1[1], rule)

        elif cat == "reduction":
            name, template, rule = self.rng.choice(REDUCTION)
            src = self._pick_var_with_shape(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            # sum_01 needs ndim >= 2
            if rule == "keepdim_0" and len(src[1]) < 2:
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "normalize":
            name, template, rule = self.rng.choice(NORMALIZE)
            src = self._pick_var_with_shape(vars_available, min_ndim=1)
            if src[1] == () or src[1][-1] < 2:
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "indexing":
            name, template, rule = self.rng.choice(INDEXING)
            src = self._pick_var_with_shape(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if name == "slice_half" and src[1][-1] < 2:
                raise _SkipOp()
            if name in ("tile2", "repeat2") and src[1][-1] > 256:
                raise _SkipOp()  # avoid OOM
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "cast":
            name, template, rule = self.rng.choice(CAST)
            src = self.rng.choice(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "where":
            name, template, rule = self.rng.choice(WHERE_COND)
            src = self.rng.choice(vars_available)
            if src[1] == ():
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "lax":
            name, template, rule = self.rng.choice(LAX_PRIMITIVES)
            src = self._pick_var_with_shape(vars_available, min_ndim=1)
            if src[1] == ():
                raise _SkipOp()
            if rule == "quarter_last" and src[1][-1] < 4:
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "scan":
            name, template, rule = self.rng.choice(SCAN_OPS)
            src = self._pick_var_with_shape(vars_available, min_ndim=1)
            if src[1] == () or src[1][-1] < 2:
                raise _SkipOp()
            expr = self._apply_template(template, src[0])
            self._op_fingerprint.add(name)
            return "v", expr, _shape_after_op(src[1], rule)

        elif cat == "semantic_mut":
            name, template, rule = self.rng.choice(SEMANTIC_MUTATIONS)
            src = self.rng.choice(vars_available)
            expr = self._apply_template(template, src[0])
            # don't add to fingerprint — these are neutral
            return "v", expr, _shape_after_op(src[1], rule)

        raise _SkipOp()


class _SkipOp(Exception):
    """Sentinel to skip an op that can't be applied to current state."""
    pass


# ─────────────────────────────────────────────────────────────
# Program composition: chain 2-3 stages from different categories
# ─────────────────────────────────────────────────────────────

class CompositionGenerator:
    """Generate multi-stage programs that combine different op categories.

    E.g., normalize → reduce → pointwise chain, or matmul → normalize → reduce.
    These compositions force XLA to fuse across category boundaries.
    """

    STAGE_PRESETS = [
        # (name, forced_ops, n_ops_range)
        ("matmul_then_norm", {"force_matmul": True}, (3, 6)),
        ("matmul_then_reduce", {"force_matmul": True}, (5, 10)),
        ("scan_heavy", {"force_scan": True}, (5, 12)),
        ("deep_pointwise", {}, (15, 25)),
        ("cast_stress", {}, (8, 15)),
        ("norm_chain", {}, (8, 15)),
    ]

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.gen = GraphGenerator(seed=seed)

    def gen_composed_program(self):
        """Generate a program by composing stages."""
        preset_name, kwargs, (lo, hi) = self.rng.choice(self.STAGE_PRESETS)
        n_ops = self.rng.randint(lo, hi)
        code, shapes, dtype, n_inputs, fingerprint = self.gen.gen_program(
            n_ops=n_ops, **kwargs
        )
        return code, shapes, dtype, n_inputs, fingerprint, preset_name


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def generate_program_with_inputs(seed, use_composition=False):
    """Generate a complete program + input arrays as strings for eval.

    Returns dict with keys: setup, fn_code, call_args, n_inputs, shapes, dtype, seed,
                            op_fingerprint, composition_preset
    """
    if use_composition:
        cgen = CompositionGenerator(seed=seed)
        code, shapes, dtype, n_inputs, fingerprint, preset = cgen.gen_composed_program()
    else:
        gen = GraphGenerator(seed=seed)
        code, shapes, dtype, n_inputs, fingerprint = gen.gen_program()
        preset = None

    setup_lines = [
        "import jax",
        "import jax.numpy as jnp",
        "import jax.scipy.special",
        "import numpy as np",
        f"rng = np.random.default_rng({seed})",
    ]
    for i in range(n_inputs):
        shape_str = repr(shapes[i])
        setup_lines.append(
            f"x{i} = jnp.asarray(rng.standard_normal({shape_str}), dtype={dtype})"
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
        "op_fingerprint": sorted(fingerprint),
        "composition_preset": preset,
    }

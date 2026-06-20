#!/usr/bin/env python3
"""
Corpus management for the fuzzing loop.

Components:
  BugBasinTracker    -- identifies structurally similar bugs, prevents re-drilling
  DiversityScheduler -- forces exploration of under-represented categories
  TemplateCompositor -- cross-category graph composition (see also generation/compose.py)
  DiversityAwareCorpus -- novelty-weighted seed selection with age decay
"""

import hashlib
import json
import os
import random
import time
from collections import defaultdict

from templates.registry import PROGRAM_TEMPLATES

_FINGERPRINT_OPS = [
    ".sum(", ".mean(", ".amax(", ".amin(", ".prod(", ".std(", ".var(",
    ".argmax(", ".argmin(", ".cumsum(", ".cumprod(",
    "F.relu", "F.gelu", "F.silu", "F.leaky_relu", "torch.sigmoid", "torch.tanh",
    "F.softmax", "F.log_softmax",
    "torch.matmul", "torch.bmm", "torch.mm", "F.linear", "torch.addmm",
    "torch.baddbmm", "torch.einsum",
    "F.conv2d", "F.conv1d", "F.conv_transpose2d",
    "F.layer_norm", "F.batch_norm", "F.group_norm", "F.instance_norm",
    ".permute(", ".transpose(", ".reshape(", ".view(", ".contiguous(",
    ".expand(", ".chunk(", ".split(", ".cat(",
    "F.pad(", "F.interpolate(",
    ".gather(", ".scatter(", ".scatter_add(", ".index_put(",
    "torch.where(", ".masked_fill(",
    "F.avg_pool", "F.max_pool", "F.adaptive_avg_pool",
    ".sort(", ".topk(", ".triu(", ".tril(",
]


def _extract_op_fingerprint(body_code):
    """Extract a frozenset of ops present in the code -> structural identity."""
    return frozenset(op for op in _FINGERPRINT_OPS if op in body_code)


class BugBasinTracker:
    """
    Tracks "bug basins" — clusters of structurally similar programs that
    trigger the same root cause. When a bug is found, its op fingerprint
    is registered. Future programs with high fingerprint overlap are
    suppressed (not added to corpus, reward downweighted).

    This is the core "anti-exploitation" mechanism:
    once you've found a bug basin, stop drilling into it.
    """

    def __init__(self, similarity_threshold=0.7):
        self.threshold = similarity_threshold
        self._basins = []  # list of (fingerprint: frozenset, bug_type, count, template_name)
        self._suppressed_count = 0

    def register_bug(self, program, bug):
        """Register a new bug's structural fingerprint as a known basin."""
        fp = _extract_op_fingerprint(program.body_code)
        # Check if it matches an existing basin
        for i, (basin_fp, _, count, _) in enumerate(self._basins):
            sim = self._jaccard(fp, basin_fp)
            if sim >= self.threshold:
                # Keep original basin fingerprint to prevent inflation
                self._basins[i] = (basin_fp, bug.bug_type, count + 1,
                                   self._basins[i][3])
                return
        # New basin
        self._basins.append((fp, getattr(bug, 'bug_type', 'unknown'), 1, program.name))

    def is_in_known_basin(self, program):
        """Check if a program's structure falls into a known bug basin."""
        fp = _extract_op_fingerprint(program.body_code)
        for basin_fp, _, count, _ in self._basins:
            sim = self._jaccard(fp, basin_fp)
            if sim >= self.threshold and count >= 2:
                return True
        return False

    def get_novelty_score(self, program):
        """
        Returns 0.0 (duplicate of known basin) to 1.0 (completely novel).
        Used as a multiplier on reward.
        """
        fp = _extract_op_fingerprint(program.body_code)
        if not self._basins:
            return 1.0
        max_sim = max(self._jaccard(fp, b[0]) for b in self._basins)
        return max(0.0, 1.0 - max_sim)

    @staticmethod
    def _jaccard(a, b):
        if not a and not b:
            return 1.0
        union = len(a | b)
        if union == 0:
            return 1.0
        return len(a & b) / union

    @property
    def suppressed_count(self):
        return self._suppressed_count

    def record_suppression(self):
        self._suppressed_count += 1

    def get_status(self):
        return (f"basins={len(self._basins)} suppressed={self._suppressed_count} "
                f"basin_sizes={[b[2] for b in self._basins]}")


class DiversityScheduler:
    """
    Forces exploration of under-represented categories and templates.
    Implements two mechanisms:
      1. Category rotation: every N iterations, force a specific under-explored category
      2. Template anti-repetition: adaptively downweight frequently-used templates
    """

    def __init__(self, rotation_interval=30):
        self.rotation_interval = rotation_interval
        self._category_hits = defaultdict(int)
        self._template_hits = defaultdict(int)
        self._category_bugs = defaultdict(int)  # bugs found per category
        self._iteration = 0
        self._forced_category = None

    def step(self):
        self._iteration += 1
        if self._iteration % self.rotation_interval == 0:
            # Force the least-explored category
            all_cats = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
            min_cat = min(all_cats, key=lambda c: self._category_hits.get(c, 0))
            self._forced_category = min_cat
        else:
            self._forced_category = None

    def record(self, category, template_name):
        self._category_hits[category] += 1
        self._template_hits[template_name] += 1

    def record_bug(self, category):
        self._category_bugs[category] += 1

    def get_forced_category(self):
        return self._forced_category

    def get_adaptive_weights(self):
        """
        Return adjusted template weights that downweight over-explored templates
        and upweight under-explored ones.
        """
        weights = []
        for name, _, base_w in PROGRAM_TEMPLATES:
            hits = self._template_hits.get(name, 0)
            # Inverse frequency bonus: less explored → higher weight
            freq_factor = 1.0 / (1.0 + hits * 0.1)
            # Bug saturation penalty: if this template's category already has many bugs,
            # reduce weight (diminishing returns)
            cat = name.split("_")[0].upper()
            bug_count = self._category_bugs.get(cat, 0)
            bug_penalty = 1.0 / (1.0 + bug_count * 0.5)
            weights.append(base_w * freq_factor * bug_penalty)
        return weights

    def get_status(self):
        return (f"cat_hits={dict(self._category_hits)} "
                f"cat_bugs={dict(self._category_bugs)} "
                f"forced={self._forced_category}")


class TemplateCompositor:
    """
    Creates novel computation graphs by composing fragments from
    different template categories. This forces structural diversity
    that cannot be achieved by mutating within a single template.

    Key idea: pick 2 "stages" from different categories and chain them.
    This creates computation patterns the compiler has never seen,
    increasing the chance of hitting new optimization paths.
    """

    # Composable "stages" — each is a code fragment that transforms x → x
    _STAGES = [
        # Reduction-based stages
        ("reduce_norm", "D1", lambda b, s, h, d: (
            f"    _mean = x.mean(dim=-1, keepdim=True)\n"
            f"    _var = ((x - _mean) ** 2).mean(dim=-1, keepdim=True)\n"
            f"    x = (x - _mean) / torch.sqrt(_var + 1e-5)\n"
        )),
        ("reduce_softmax", "D1", lambda b, s, h, d: (
            f"    _xmax = x.amax(dim=-1, keepdim=True)\n"
            f"    x = torch.exp(x - _xmax)\n"
            f"    x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)\n"
        )),
        ("reduce_scale", "D1", lambda b, s, h, d: (
            f"    _scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)\n"
            f"    x = x / _scale\n"
        )),
        # GEMM-based stages (weights injected via _program_inputs, not created inside program())

        ("gemm_project", "D2", None),   # handled specially in compose()
        ("gemm_bilinear", "D2", None),  # handled specially in compose()
        # Layout-disrupting stages
        ("layout_permute", "D3", lambda b, s, h, d: (
            f"    x = x.transpose(-1, -2).contiguous().transpose(-1, -2)\n"
        )),
        ("layout_chunk_cat", "D3", lambda b, s, h, d: (
            f"    _chunks = x.chunk(2, dim=-1)\n"
            f"    x = torch.cat([_chunks[1], _chunks[0]], dim=-1)\n"
        )),
        # Layout disruption: non-contiguous + contiguous round-trip
        ("layout_permute_contig", "D3", lambda b, s, h, d: (
            f"    x = x.transpose(-1, -2).contiguous()\n"
            f"    x = x.transpose(-1, -2)\n"
        )),
        # Strided slice + pad back: stresses Triton indexing math
        ("layout_slice_stride", "D3", lambda b, s, h, d: (
            f"    _half = {h} // 2 + {h} % 2\n"
            f"    x = x[..., ::2]\n"
            f"    x = F.pad(x, (0, {h} - _half), value=0.0)\n"
        )),
        # Dependency stages
        ("dep_gating", "D4", lambda b, s, h, d: (
            f"    _gate = torch.sigmoid(x.mean(dim=-1, keepdim=True))\n"
            f"    x = x * _gate\n"
        )),
        ("dep_residual_scale", "D4", lambda b, s, h, d: (
            f"    _res = x.clone()\n"
            f"    _s = x.std(dim=-1, keepdim=True).clamp(min=1e-5)\n"
            f"    x = _res + x / _s\n"
        )),
        # Boundary-probing stages
        ("bound_dtype_cast", "D5", lambda b, s, h, d: (
            f"    x = x.to(torch.float64).to({d})\n"
        )),
        ("bound_narrow_expand", "D5", lambda b, s, h, d: (
            f"    _n = min(x.shape[-1], max(1, x.shape[-1] - 1))\n"
            f"    x = x.narrow(-1, 0, _n)\n"
            f"    x = F.pad(x, (0, {h} - _n), value=0.0)\n"
        )),
        # Broadcast + reduction cross
        ("reduce_broadcast_cross", "D1", lambda b, s, h, d: (
            f"    _xm = x.mean(dim=-1, keepdim=True)\n"
            f"    x = (x - _xm) * x.std(dim=-1, keepdim=True).clamp(min=1e-5)\n"
            f"    x = x.sum(dim=1, keepdim=True).expand_as(x) + x\n"
        )),
        # Dtype cast round-trip in fusion chain
        ("bound_half_roundtrip", "D5", lambda b, s, h, d: (
            f"    x = x.half().float()\n"
            f"    x = torch.relu(x)\n"
        )),
        # Logit with high eps
        ("bound_logit_eps", "D5", lambda b, s, h, d: (
            f"    x = x.sigmoid()\n"
            f"    x = torch.logit(x, eps=0.9)\n"
        )),
    ]

    # Special GEMM stage code generators that add weights to _program_inputs
    _GEMM_GENERATORS = {
        "gemm_project": lambda b, s, h, d: {
            "input_code": (
                f"_w_proj = torch.randn({h}, {h}, dtype={d})\n"
                f"if torch.cuda.is_available(): _w_proj = _w_proj.cuda()\n"
            ),
            "input_names": ["_w_proj"],
            "body_code": (
                f"    x = torch.matmul(x, _w_proj)\n"
            ),
        },
        "gemm_bilinear": lambda b, s, h, d: {
            "input_code": (
                f"_w_bil = torch.randn({h}, {h}, dtype={d})\n"
                f"_b_bil = torch.randn({h}, dtype={d})\n"
                f"if torch.cuda.is_available(): _w_bil = _w_bil.cuda(); _b_bil = _b_bil.cuda()\n"
            ),
            "input_names": ["_w_bil", "_b_bil"],
            "body_code": (
                f"    x = torch.addmm(_b_bil, x.reshape(-1, {h}), _w_bil).reshape(x.shape)\n"
            ),
        },
    }

    @staticmethod
    def compose(batch, seq, hidden, dtype):
        """
        Generate a novel program by chaining 2-3 random stages from
        different categories.
        """
        import torch.nn.functional as F  # noqa — needed for generated code
        # Pick 2-3 stages from different categories
        n_stages = random.choice([2, 2, 3])
        available = list(TemplateCompositor._STAGES)
        random.shuffle(available)

        chosen = []
        used_cats = set()
        for stage_name, cat, gen_fn in available:
            if cat not in used_cats or len(chosen) < 2:
                chosen.append((stage_name, cat, gen_fn))
                used_cats.add(cat)
            if len(chosen) >= n_stages:
                break

        # Build the composed program
        stage_code = ""
        stage_names = []
        extra_input_code = ""
        extra_input_names = []
        for stage_name, cat, gen_fn in chosen:
            if gen_fn is None and stage_name in TemplateCompositor._GEMM_GENERATORS:
                # GEMM stages: weights go to _program_inputs for determinism
                gemm = TemplateCompositor._GEMM_GENERATORS[stage_name](
                    batch, seq, hidden, dtype)
                extra_input_code += gemm["input_code"]
                extra_input_names.extend(gemm["input_names"])
                stage_code += gemm["body_code"]
            elif gen_fn is not None:
                stage_code += gen_fn(batch, seq, hidden, dtype)
            stage_names.append(f"{cat}:{stage_name}")

        name = "composed_" + "_".join(s[0] for s in chosen)

        # Build function signature with extra weight args
        all_params = ["x"] + extra_input_names
        func_sig = ", ".join(all_params)
        all_inputs = ["_x"] + extra_input_names

        body = (
            f"import torch.nn.functional as F\n"
            f"_x = torch.randn({batch}, {seq}, {hidden}, dtype={dtype})\n"
            f"{extra_input_code}"
            f"_program_inputs = ({', '.join(all_inputs)},)\n\n"
            f"# Composed: {' -> '.join(stage_names)}\n"
            f"def program({func_sig}):\n"
            f"{stage_code}"
            f"    return x\n"
        )
        return name, body, True  # (name, body_code, is_deterministic)


class DiversityAwareCorpus:
    """
    Enhanced corpus that:
    1. Penalizes seeds in known bug basins (via BugBasinTracker)
    2. Uses age-based decay (older seeds lose weight)
    3. Ensures minimum category diversity
    """

    def __init__(self, max_size=60, basin_tracker=None):
        self.max_size = max_size
        self.basin_tracker = basin_tracker
        self._entries = []  # list of (program, base_score, add_iteration)

    def add_seed(self, program, kernel_count, iteration=0):
        if kernel_count == 0:
            return
        # Check for duplicate signature
        for prog, _, _ in self._entries:
            if prog.signature == program.signature:
                return
        # Don't add if it's in a known bug basin
        if self.basin_tracker and self.basin_tracker.is_in_known_basin(program):
            self.basin_tracker.record_suppression()
            return
        self._entries.append((program, kernel_count, iteration))
        # Evict oldest low-score entries if over capacity
        if len(self._entries) > self.max_size:
            # Sort by score ascending, remove lowest
            self._entries.sort(key=lambda e: e[1])
            self._entries = self._entries[1:]

    def select_parent(self, current_iteration=0):
        if not self._entries:
            return None
        # Score = base_kernel_count * novelty * age_decay
        scores = []
        for prog, base_score, add_iter in self._entries:
            novelty = (self.basin_tracker.get_novelty_score(prog)
                       if self.basin_tracker else 1.0)
            # Age decay: halve weight every 500 iterations
            age = max(0, current_iteration - add_iter)
            age_factor = 0.5 ** (age / 500.0)
            score = max(0.01, base_score * (0.3 + 0.7 * novelty) * age_factor)
            scores.append(score)
        total = sum(scores)
        if total == 0:
            return random.choice([e[0] for e in self._entries])
        return random.choices([e[0] for e in self._entries],
                              weights=[s / total for s in scores], k=1)[0]

    def size(self):
        return len(self._entries)

    def category_coverage(self):
        return set(p.category for p, _, _ in self._entries)

    def save_checkpoint(self, path):
        """Serialize corpus to JSON for persistence across runs."""
        entries = []
        for prog, score, add_iter in self._entries:
            entries.append({
                'name': prog.name, 'body_code': prog.body_code,
                'is_deterministic': prog.is_deterministic,
                'dtype': prog.dtype, 'batch': prog.batch,
                'seq': prog.seq, 'hidden': prog.hidden,
                'has_poison': prog.has_poison,
                'config_preset': prog.config_preset,
                'use_dynamic': prog.use_dynamic,
                'supports_backward': prog.supports_backward,
                'score': score, 'add_iteration': add_iter,
            })
        try:
            tmp = path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({'entries': entries, 'max_size': self.max_size}, f)
            os.replace(tmp, path)
        except Exception as e:
            print(f"  [Corpus] checkpoint save failed: {e}")

    def load_checkpoint(self, path):
        """Restore corpus from a previous checkpoint."""
        if not os.path.exists(path):
            return 0
        try:
            with open(path) as f:
                data = json.load(f)
            loaded = 0
            for e in data.get('entries', []):
                prog = GraphProgram(
                    name=e['name'], body_code=e['body_code'],
                    is_deterministic=e['is_deterministic'],
                    dtype=e['dtype'], batch=e['batch'],
                    seq=e['seq'], hidden=e['hidden'],
                    has_poison=e.get('has_poison', False),
                    config_preset=e.get('config_preset', 'default'),
                    use_dynamic=e.get('use_dynamic', False),
                    supports_backward=e.get('supports_backward', True),
                )
                # Bypass basin check for restored entries
                self._entries.append((prog, e['score'], e['add_iteration']))
                loaded += 1
            if loaded:
                print(f"  [Corpus] restored {loaded} entries from {path}")
            return loaded
        except Exception as e:
            print(f"  [Corpus] checkpoint load failed: {e}")
            return 0



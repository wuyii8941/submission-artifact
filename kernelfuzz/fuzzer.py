#!/usr/bin/env python3
"""
DeepOpFuzz: Kernel-space guided graph fuzzer for PyTorch/Inductor.

Main fuzzing loop (Algorithm 1 in the paper).

Usage:
  python fuzzer.py --output /path/to/output --max-iterations 5000 --timeout 60
"""

import gc
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
import traceback
from collections import defaultdict, deque
from os.path import join
from typing import Dict, List, Optional, Set, Tuple

import torch

# Kernel-space coverage tracking.
#  - GraphCoverageTracker is the matcher (M_gamma in paper Algorithm 1):
#    each update() sets tracker.last_was_novel iff a new L1 family or L2
#    codegen path was observed.
#  - The runtime KernelStateSpace (imported in GraphFuzzerV4.__init__) is a
#    separate richer (family, dtype, persistent, n_fused) coverage report
#    used by the target sampler to find under-reached states.
from kernel.coverage import (
    GraphSaturationDetector, GraphCoverageTracker, GraphMutationController,
    _DTYPES, _DTYPE_WEIGHTS,
    _ADV_BATCH, _ADV_SEQ, _ADV_HIDDEN,
    _STD_BATCH, _STD_SEQ, _STD_HIDDEN,
    _BOUNDARY_SHAPES, GRAPH_DIFF_BOUNDS,
    _sample_shapes,
)

# Rate at which target sampling is steered by kernel-state coverage feedback
# (vs. unguided template sampling). The KSS-guided branch reads the running
# coverage of compiler decision states and biases generation toward
# under-covered (family, dtype) states. The branch is gated off entirely when
# guidance_mode == "random" (the ablation baseline), disabling the loop.
KSS_GUIDANCE_RATE = 0.30

# Program generation
from templates.registry import (
    GraphPoisonInjector, PROGRAM_TEMPLATES, _TPL_CATEGORIES,
    _INPUT_VALUE_STRATEGIES, _tpl_category,
    _TPL_WEIGHTS, _TPL_MAP, ENRICHED_CLUSTER_BOOST,
    _apply_value_diversification,
)
from program import (
    GraphProgram, ProgramGenerator, GraphMutator, _NO_BACKWARD_TEMPLATES,
    _apply_semantic_mutation, _apply_backward_stress_mutation,
)

# Bug classification
from bug_result import (
    DivergenceLevel, GraphBugResultV2, GraphBugResult,
    analyze_graph_results, analyze_graph_results_v2,
    check_divergence, _write_bug_file, _write_bug_file_v2,
    _save_crash_report, _verify_bug_in_fresh_process,
    _sample_config_preset, _build_config_injection_code,
    _run_backward_verification,
)

# Corpus management
from corpus import (
    BugBasinTracker, DiversityScheduler, TemplateCompositor,
    DiversityAwareCorpus, _extract_op_fingerprint,
)

# Configuration
from config import setup_environment, get_parser

# Oracle execution engine (from FreeFuzz baseline)
import sys
_FREEFUZZ_SRC = os.environ.get(
    "KERNELFUZZ_FREEFUZZ_SRC",
    os.path.join(os.path.dirname(__file__), "..", "FreeFuzz-baseline", "kernel-guided", "src"),
)
if os.path.isdir(_FREEFUZZ_SRC) and _FREEFUZZ_SRC not in sys.path:
    sys.path.insert(0, _FREEFUZZ_SRC)
from classes.torch_library import TorchLibrary

setup_environment()


class GraphFuzzerV3:
    """
    Base graph fuzzer with anti-exploitation mechanisms.

    Prevents bug-basin lock-in via:
      - BugBasinTracker: structural similarity dedup to avoid re-hitting same root cause
      - DiversityScheduler: forced category rotation with adaptive template weights
      - TemplateCompositor: cross-category composition for novel graph topologies
      - DiversityAwareCorpus: novelty-weighted seed selection with age decay
      - Structural dedup based on op fingerprint rather than error hash
    """

    def __init__(self, output_dir, strategy_name="GraphV3", warmup_ratio=0.08,
                 composition_ratio=0.20, basin_similarity=0.7):
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.warmup_ratio = warmup_ratio
        self.composition_ratio = composition_ratio  # % of iterations using composed templates

        # Core components
        self.basin_tracker = BugBasinTracker(similarity_threshold=basin_similarity)
        self.diversity_scheduler = DiversityScheduler(rotation_interval=30)
        self.coverage = GraphCoverageTracker(strategy_name)
        self.corpus = DiversityAwareCorpus(max_size=60, basin_tracker=self.basin_tracker)
        self.mutation_ctrl = GraphMutationController(
            base_epsilon=0.15,
            expansion_epsilon=0.40,
        )
        self.generator = ProgramGenerator()
        self.iterations_done = 0

        self._graph_dir = join(output_dir, "graph-oracle")
        for sub in ["potential-bug", "weak-signal", "success", "fail"]:
            os.makedirs(join(self._graph_dir, sub), exist_ok=True)

        self._seen_bugs = {}
        self._structural_dedup = {}  # op_fingerprint -> list of bug_types
        self._BUG_COOLING_THRESHOLD = 3
        self._bug_traces = {}
        self.total_bugs = 0
        self.total_novel_bugs = 0  # Bugs with novel fingerprints
        self.total_basin_suppressed = 0
        self.total_composed = 0
        self.total_suspicious = 0
        self.total_backward_amplified = 0
        self.bugs_by_type = defaultdict(int)
        self.bugs_by_category = defaultdict(int)
        self.bugs_by_divergence = defaultdict(int)

    def _execute(self, program):
        results, outer_error, kernels = TorchLibrary.run_code(program.build_eval_code_v2())
        return results, outer_error, kernels

    def _structural_dedup_key(self, program):
        """Dedup by op fingerprint — catches same root cause across different templates."""
        fp = _extract_op_fingerprint(program.body_code)
        return fp

    def _is_structurally_novel(self, program, bug):
        """Check if this bug has a genuinely new structural fingerprint."""
        fp = self._structural_dedup_key(program)
        existing = self._structural_dedup.get(fp, [])
        bt = getattr(bug, 'bug_type', None)
        if bt in existing:
            return False
        return True

    def _generate_program(self, iteration, inject_poison):
        """
        Program generation with diversity forcing:
          1. Check DiversityScheduler for forced category
          2. Periodically use TemplateCompositor for novel topologies
          3. Use adaptive weights instead of static weights
        """
        self.diversity_scheduler.step()

        # 20% of the time: use TemplateCompositor for structural jumps
        if random.random() < self.composition_ratio:
            batch = random.choice([1, 2, 3, 4, 7])
            seq = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127])
            hidden = random.choice([7, 8, 13, 16, 32, 33, 64, 128, 256])
            dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
            name, body_code, is_det = TemplateCompositor.compose(batch, seq, hidden, dtype)
            self.total_composed += 1

            poisoned = False
            if inject_poison:
                new_code = GraphPoisonInjector.inject(body_code, dtype)
                poisoned = (new_code != body_code)
                body_code = new_code

            config_preset = _sample_config_preset()
            use_dynamic = (random.random() < 0.30)
            prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                has_poison=poisoned, config_preset=config_preset,
                                use_dynamic=use_dynamic, supports_backward=True)
            return prog, "composed"

        # Check for forced category
        forced_cat = self.diversity_scheduler.get_forced_category()
        if forced_cat and forced_cat in _TPL_CATEGORIES and _TPL_CATEGORIES[forced_cat]:
            name = random.choice(_TPL_CATEGORIES[forced_cat])
            batch = random.choice([1, 2, 3, 4, 7])
            seq = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127])
            hidden = random.choice([7, 8, 13, 16, 32, 33, 64, 128, 256])
            dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
            try:
                body_code, is_det = _TPL_MAP[name](batch, seq, hidden, dtype)
            except Exception as e:
                print(f"  forced template {name} failed: {type(e).__name__}: {e}")
                body_code, is_det = _tpl_d2_linear_bias_act(batch, seq, hidden, dtype)
                name = "d2_linear_bias_act"
            poisoned = False
            if inject_poison:
                new_code = GraphPoisonInjector.inject(body_code, dtype)
                poisoned = (new_code != body_code)
                body_code = new_code
            config_preset = _sample_config_preset()
            use_dynamic = (random.random() < 0.30)
            prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                has_poison=poisoned, config_preset=config_preset,
                                use_dynamic=use_dynamic,
                                supports_backward=(name not in _NO_BACKWARD_TEMPLATES))
            self.diversity_scheduler.record(forced_cat, name)
            return prog, f"forced_{forced_cat}"

        # Normal generation with adaptive weights — only sample params, not a full program
        adaptive_weights = self.diversity_scheduler.get_adaptive_weights()
        name, _, _ = random.choices(
            [(n, f, w) for n, f, w in PROGRAM_TEMPLATES],
            weights=adaptive_weights, k=1)[0]
        batch  = random.choice([1, 2, 3, 4, 7])
        seq    = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127])
        hidden = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127, 128, 256])
        dtype  = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
        try:
            body_code, is_det = _TPL_MAP[name](batch, seq, hidden, dtype)
        except Exception as e:
            print(f"  adaptive template {name} failed: {type(e).__name__}: {e}")
            body_code, is_det = _tpl_d2_linear_bias_act(batch, seq, hidden, dtype)
            name = "d2_linear_bias_act"
        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code
        config_preset = _sample_config_preset()
        use_dynamic = (random.random() < 0.15)
        result_prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                   has_poison=poisoned, config_preset=config_preset,
                                   use_dynamic=use_dynamic,
                                   supports_backward=(name not in _NO_BACKWARD_TEMPLATES))
        self.diversity_scheduler.record(result_prog.category, name)
        return result_prog, "adaptive"

    def _handle_bug(self, program, bug, source, kernels):
        """Bug handling with structural dedup and basin tracking."""
        # Structural novelty check
        is_novel = self._is_structurally_novel(program, bug)
        fp = self._structural_dedup_key(program)

        # Traditional dedup (for cooling)
        err_norm = ""
        if bug.error_msg:
            cleaned = re.sub(r"0x[0-9a-f]+", "ADDR",
                             re.sub(r"\d+", "N", bug.error_msg.lower()))[:80]
            err_norm = hashlib.sha256(cleaned.encode()).hexdigest()[:16]
        key = f"{program.name}|{bug.bug_type}|{err_norm}"
        hits = self._seen_bugs.get(key, 0) + 1
        self._seen_bugs[key] = hits

        if hits > self._BUG_COOLING_THRESHOLD:
            if hits == self._BUG_COOLING_THRESHOLD + 1:
                print(f"  [COOLED] [{bug.category}:{bug.bug_type}] "
                      f"key={key} hit {hits}x, suppressing further reports")
            return

        # Register in basin tracker (always, even for repeats)
        self.basin_tracker.register_bug(program, bug)
        self.diversity_scheduler.record_bug(bug.category)

        if hits > 1 and not is_novel:
            # Repeat + structurally known → minimal handling
            print(f"  [REPEAT x{hits}] [{bug.category}:{bug.bug_type}] "
                  f"prog={program.signature} [basin-known]")
            return

        # Backward verification for SUSPICIOUS signals
        if (getattr(bug, 'needs_backward', False)
                and getattr(bug, 'divergence_level', None) == DivergenceLevel.SUSPICIOUS
                and program.supports_backward and not program.has_poison):
            is_amplified, grad_level, grad_max, bwd_err, _ = _run_backward_verification(
                program, program.dtype)
            if is_amplified:
                bug.bug_type = "backward_amplified"
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                bug.priority = 3
                bug.error_msg = (f"[BWD AMPLIFIED] forward={bug.error_msg} "
                                 f"grad_max={grad_max:.2e}")
                bug.needs_backward = False
                self.total_backward_amplified += 1

        # Record bug
        self.total_bugs += 1
        self.bugs_by_type[bug.bug_type] += 1
        self.bugs_by_category[bug.category] += 1
        div_level = getattr(bug, 'divergence_level', DivergenceLevel.CATASTROPHIC)
        self.bugs_by_divergence[div_level] += 1

        # Track structural dedup
        bt = getattr(bug, 'bug_type', 'unknown')
        if fp not in self._structural_dedup:
            self._structural_dedup[fp] = []
        self._structural_dedup[fp].append(bt)

        if is_novel:
            self.total_novel_bugs += 1

        # Write to appropriate directory
        if div_level == DivergenceLevel.CATASTROPHIC:
            subdir = "potential-bug"
        else:
            subdir = "weak-signal"
            self.total_suspicious += 1

        try:
            dest = _write_bug_file_v2(self._graph_dir, program, bug, subdir=subdir)
            self._bug_traces[dest] = {
                "type": bug.bug_type, "msg": bug.error_msg or "",
                "priority": bug.priority, "config": bug.trigger_config,
                "category": bug.category,
                "divergence_level": div_level,
                "config_preset": program.config_preset,
                "use_dynamic": program.use_dynamic,
                "is_novel": is_novel,
                "op_fingerprint": sorted(fp) if fp else [],
            }
        except Exception as e:
            print(f"  [GraphFuzzerV3] write_bug failed: {e}")

        tag = {3: "[FIRE]", 2: "[BUG]", 1: "[LOW]"}.get(bug.priority, "")
        div_tag = {"CATASTROPHIC": "CAT", "SUSPICIOUS": "SUS"}.get(div_level, "?")
        novel_tag = " [NOVEL]" if is_novel else ""
        print(f"  {tag}[{div_tag}]{novel_tag} [{bug.category}:{bug.bug_type}] P{bug.priority} "
              f"src={source} config={bug.trigger_config} "
              f"preset={program.config_preset} dyn={program.use_dynamic} "
              f"diff={bug.max_diff} kernels={len(kernels)} "
              f"prog={program.signature}")

    def run_fuzzing_loop(self, max_iterations=5000, result_callback=None,
                         checkpoint_interval=100):
        warmup_threshold = min(50, int(max_iterations * self.warmup_ratio))

        print(f"\n{'='*70}")
        print(f"[GraphFuzzerV3] Fuzzer")
        print(f"  strategy={self.strategy_name} max_iter={max_iterations}")
        print(f"  Templates: {len(PROGRAM_TEMPLATES)} + TemplateCompositor")
        print(f"  Composition ratio: {self.composition_ratio:.0%}")
        print(f"  Basin similarity threshold: {self.basin_tracker.threshold}")
        print(f"  Category rotation interval: {self.diversity_scheduler.rotation_interval}")
        print(f"  Oracle: graded (CATASTROPHIC/SUSPICIOUS/SUCCESS)")
        print(f"  Dedup: structural fingerprint + traditional")
        print(f"  Warmup: {warmup_threshold}  Base ε: {self.mutation_ctrl.base_epsilon}")
        print(f"{'='*70}\n")

        start = time.time()

        for i in range(max_iterations):
            if i < warmup_threshold:
                program = self.generator.generate_v2(inject_poison=False)
                source = "warmup"
                self.mutation_ctrl.record_warmup()
            else:
                if i == warmup_threshold:
                    print(f"\n  warmup -> evolution (iter {i})\n")
                    if self.corpus.size() == 0:
                        for _ in range(5):
                            seed = self.generator.generate_v2(inject_poison=False)
                            self.corpus.add_seed(seed, 1, iteration=i)

                if self.mutation_ctrl.should_explore():
                    # Exploration: use diversity-aware generation
                    program, source = self._generate_program(i, inject_poison=True)
                elif self.corpus.size() > 0 and random.random() < 0.60:
                    # Exploitation from corpus (reduced from 0.70)
                    parent = self.corpus.select_parent(current_iteration=i)
                    program = GraphMutator.mutate_v2(
                        parent, self.mutation_ctrl.get_mutation_range(), inject_poison=True)
                    source = "corpus"
                    self.diversity_scheduler.record(program.category, program.name)
                else:
                    # Random with diversity
                    program, source = self._generate_program(i, inject_poison=True)

                # Basin suppression: skip programs in known basins (but only during exploitation)
                if source == "corpus" and self.basin_tracker.is_in_known_basin(program):
                    self.total_basin_suppressed += 1
                    # Force a fresh exploration instead
                    program, source = self._generate_program(i, inject_poison=True)
                    source = "basin_escape"

            try:
                results, outer_error, kernels = self._execute(program)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [GraphFuzzerV3] iter {i}: CUDA OOM, cleared cache")
                continue
            except Exception as e:
                print(f"  [GraphFuzzerV3] iter {i}: execute failed: {type(e).__name__}: {e}")
                continue

            new_count, (should_expand, _) = self.coverage.update(kernels, i)
            if should_expand:
                self.mutation_ctrl.enter_expansion_mode()
            if new_count > 0:
                if self.mutation_ctrl.expansion_mode:
                    self.mutation_ctrl.exit_expansion_mode()
                self.corpus.add_seed(program, new_count, iteration=i)

            bug = analyze_graph_results_v2(results, program, outer_error)
            if bug.is_bug:
                _verified, _vreason = _verify_bug_in_fresh_process(bug, program)
                if _verified:
                    self._handle_bug(program, bug, source, kernels)
                else:
                    print(f"  [verify] FILTERED {bug.bug_type} reason={_vreason[:120]}")
                    bug.is_bug = False

            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start
                cat_cov = self.corpus.category_coverage()
                print(f"\n  [{i+1:5d}/{max_iterations}] "
                      f"programs={self.generator.generated_count} "
                      f"kernels={self.coverage.get_total()} "
                      f"bugs={self.total_bugs} (novel={self.total_novel_bugs} "
                      f"CAT={self.bugs_by_divergence.get(DivergenceLevel.CATASTROPHIC,0)} "
                      f"SUS={self.total_suspicious} BWD={self.total_backward_amplified}) "
                      f"corpus={self.corpus.size()} cats={cat_cov} "
                      f"composed={self.total_composed} "
                      f"basin_suppressed={self.total_basin_suppressed} "
                      f"speed={(i+1)/max(elapsed,1e-9):.1f}/s "
                      f"elapsed={elapsed/60:.1f}m")
                print(f"  {self.mutation_ctrl.get_status()}")
                print(f"  diversity: {self.diversity_scheduler.get_status()}")
                print(f"  basins: {self.basin_tracker.get_status()}")
                print(f"  by_type={dict(self.bugs_by_type)}")
                print(f"  by_category={dict(self.bugs_by_category)}\n")
                if result_callback:
                    result_callback(self._get_result())

            self.iterations_done = i + 1

        self._save_bug_traces()
        elapsed = time.time() - start
        print(f"\n{'='*70}")
        print(f"[GraphFuzzerV3] done: programs={self.generator.generated_count} "
              f"kernels={self.coverage.get_total()} "
              f"bugs={self.total_bugs} (novel={self.total_novel_bugs}) "
              f"composed={self.total_composed} basin_suppressed={self.total_basin_suppressed} "
              f"elapsed={elapsed/60:.1f}m")
        print(f"  by_type={dict(self.bugs_by_type)}")
        print(f"  by_category={dict(self.bugs_by_category)}")
        print(f"  basins: {self.basin_tracker.get_status()}")
        print(f"  diversity: {self.diversity_scheduler.get_status()}")
        print(f"{'='*70}\n")
        return self._get_result()

    def _get_result(self):
        return {
            "phase": "GraphV3", "success": True,
            "total_programs": self.generator.generated_count,
            "total_kernels": self.coverage.get_total(),
            "total_bugs": self.total_bugs,
            "total_novel_bugs": self.total_novel_bugs,
            "total_composed": self.total_composed,
            "total_basin_suppressed": self.total_basin_suppressed,
            "total_catastrophic": self.bugs_by_divergence.get(DivergenceLevel.CATASTROPHIC, 0),
            "total_suspicious": self.total_suspicious,
            "total_backward_amplified": self.total_backward_amplified,
            "bugs_by_type": dict(self.bugs_by_type),
            "bugs_by_category": dict(self.bugs_by_category),
            "bugs_by_divergence": dict(self.bugs_by_divergence),
            "unique_bugs": len(self._seen_bugs),
            "unique_fingerprints": len(self._structural_dedup),
            "corpus_size": self.corpus.size(),
            "coverage_history": list(self.coverage.history),
            "mutation_status": self.mutation_ctrl.get_status(),
            "basin_status": self.basin_tracker.get_status(),
            "diversity_status": self.diversity_scheduler.get_status(),
            "bugs": {
                "crash": (self.bugs_by_type.get("backend_crash", 0)
                          + self.bugs_by_type.get("frontend_crash", 0)
                          + self.bugs_by_type.get("dynamic_crash", 0)),
                "compile": (self.bugs_by_type.get("inductor_diff", 0)
                            + self.bugs_by_type.get("frontend_diff", 0)
                            + self.bugs_by_type.get("compile_diff", 0)),
                "backward": self.bugs_by_type.get("backward_amplified", 0),
                "cuda": 0, "precision": 0,
                "total": self.total_bugs, "unique": len(self._seen_bugs),
                "novel": self.total_novel_bugs,
            },
        }

    def _save_bug_traces(self):
        out = join(self.output_dir, "graph_v3_bug_traces.json")
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self._bug_traces, f, indent=2, ensure_ascii=False)
            print(f"  [GraphFuzzerV3] {len(self._bug_traces)} traces -> {out}")
        except Exception as e:
            print(f"  [GraphFuzzerV3] trace save failed: {e}")


# =============================================================================
# Process-Isolated Runner (GraphFuzzerV3)
# =============================================================================

def _run_graph_worker_v3(output_dir, max_iterations, result_file, config_path):
    result = {
        "phase": "GraphV3", "success": False, "error": None,
        "total_programs": 0, "total_kernels": 0, "total_bugs": 0,
        "total_novel_bugs": 0, "total_composed": 0, "total_basin_suppressed": 0,
        "bugs_by_type": {}, "bugs_by_category": {}, "bugs_by_divergence": {},
        "unique_bugs": 0, "coverage_history": [], "corpus_size": 0,
        "mutation_status": "",
        "bugs": {"crash": 0, "compile": 0, "backward": 0, "cuda": 0,
                 "precision": 0, "total": 0, "novel": 0},
    }
    if config_path and os.path.exists(config_path):
        try:
            import configparser
            from classes.database import TorchDatabase
            cfg = configparser.ConfigParser(); cfg.read(config_path)
            TorchDatabase.database_config(
                cfg["mongodb"]["host"], int(cfg["mongodb"]["port"]),
                cfg["mongodb"]["torch_database"])
        except Exception as e:
            print(f"  [GraphFuzzerV3] DB config failed: {e}")

    def _save():
        try:
            tmp = result_file + ".tmp"
            with open(tmp, "w") as f: json.dump(result, f, indent=2)
            os.replace(tmp, result_file)
        except Exception as e:
            print(f"  [GraphFuzzerV3] save failed: {e}")

    try:
        fuzzer = GraphFuzzerV3(output_dir=output_dir, strategy_name="GraphV3")
        r = fuzzer.run_fuzzing_loop(
            max_iterations=max_iterations,
            result_callback=lambda rr: (result.update(rr), _save()))
        result.update(r); result["success"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n[GraphFuzzerV3] worker failed: {e}")
    _save()


def run_graph_phase_isolated_v3(output_dir, max_iterations=5000,
                                 timeout_minutes=60, config_path=None):
    os.makedirs(output_dir, exist_ok=True)
    result_file = join(output_dir, "graph_v3_result.json")
    print(f"\n{'='*70}\nLAUNCHING ISOLATED PHASE: GraphV3  "
          f"max_iter={max_iterations:,}  timeout={timeout_minutes}min\n{'='*70}\n")

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_run_graph_worker_v3,
                    args=(output_dir, max_iterations, result_file, config_path))
    t0 = time.time(); p.start(); p.join(timeout=timeout_minutes * 60)
    elapsed = time.time() - t0

    result = {"phase": "GraphV3", "success": False, "error": None,
              "elapsed_seconds": elapsed,
              "total_programs": 0, "total_bugs": 0, "total_novel_bugs": 0,
              "bugs": {"crash": 0, "compile": 0, "backward": 0,
                       "cuda": 0, "precision": 0, "total": 0, "novel": 0}}

    if p.is_alive():
        p.terminate(); p.join(timeout=10)
        if p.is_alive(): p.kill(); p.join()
        result["error"] = f"Timeout after {timeout_minutes} minutes"
    elif p.exitcode != 0:
        result["error"] = f"Process exited with code {p.exitcode}"

    if os.path.exists(result_file):
        try:
            with open(result_file) as f: saved = json.load(f)
            result.update(saved); result["elapsed_seconds"] = elapsed
        except Exception as e:
            print(f"  Failed to read result file: {e}")

    print(f"\nGraphV3 phase {'COMPLETED' if result.get('success') else 'FAILED'}  "
          f"programs={result.get('total_programs',0)}  "
          f"kernels={result.get('total_kernels',0)}  "
          f"bugs={result.get('total_bugs',0)}  "
          f"novel={result.get('total_novel_bugs',0)}  "
          f"composed={result.get('total_composed',0)}  "
          f"basin_suppressed={result.get('total_basin_suppressed',0)}  "
          f"elapsed={elapsed/60:.1f}m")
    return result


# =============================================================================
# GraphFuzzerV4: Targeted Exploration
# =============================================================================

def _infer_dtype_from_body(body):
    for dtype in ("torch.bfloat16", "torch.float16", "torch.float64", "torch.float32"):
        if dtype in body:
            return dtype
    return "torch.float32"


def _infer_first_dim_from_body(body):
    m = re.search(r"torch\.(?:randn|zeros|ones|empty|randint)\(([^)]{1,120})\)", body)
    if not m:
        return 1
    nums = re.findall(r"\b\d+\b", m.group(1))
    return int(nums[0]) if nums else 1



class GraphFuzzerV4(GraphFuzzerV3):
    """
    Targeted exploration fuzzer, extending GraphFuzzerV3.

    Additional capabilities:
      1. Adversarial shapes: 2^n +/- 1 and primes to stress Triton tile masking
      2. Semantic mutation: insert .clone()/.contiguous()/+0/*1 etc. that
         don't change math but change compiler alias analysis & fusion decisions
      3. Layout disruption stages: permute+contiguous, strided slice+pad
      4. Config preset: no_cudagraph to expose dynamic-shape bugs
      5. Multiple secondary oracles (backward, config-vs-config, guard, etc.)
    """

    _V4_ADV_RATIO = 0.5          # 50% adversarial shapes
    _V4_SEMANTIC_PROB = 0.20     # 20% semantic mutation
    # Widened from 1e-8 → 1e-6: fp64 reduction noise on compounded matmul
    # stacks (~1e3 elements deep) can still produce ~1e-9 diffs that aren't bugs.
    _V4_FP64_BACKOFF_THRESHOLD = 1e-6
    # Primary backward oracle: probability of running backward on a program
    # whose forward looked clean. ~25% adds modest runtime cost (~25%) but
    # opens a whole new bug surface (gradient-only bugs that forward misses).
    _V4_PRIMARY_BWD_PROB = 0.25
    # Backward stress mutation — probability of applying backward-stressing
    # ops to generated programs. These programs always get backward oracle
    # applied (100%, not the normal 25%).
    _V4_BWD_STRESS_PROB = 0.15
    # Secondary config-vs-config oracle: when eager-vs-inductor agrees, we
    # sometimes also compare two different inductor configs against each other.
    # Disagreements go ONLY to weak-signal (never CATASTROPHIC) because neither
    # side is ground truth. ~10% sample to keep runtime cost low.
    _V4_CFG_CFG_PROB = 0.10
    _V4_CFG_CFG_THRESHOLD = 1e-3  # max abs diff above which we flag

    # Known-bug fingerprints: programs matching these patterns are routed to
    # weak-signal/ instead of potential-bug/. Update this list as bugs get
    # confirmed and reported upstream so the fuzzer doesn't keep re-reporting
    # them and crowding out new findings.
    _KNOWN_BUG_PATTERNS = [
        # bug_004: bf16/half cast elision
        # Any program that does .to(bf16/half).float() is in this family.
        ("bug_001_bf16_cast",
         lambda body: (".to(torch.bfloat16)" in body or ".bfloat16(" in body)
                      and ".float()" in body),
        ("bug_004_half_cast",
         lambda body: (".to(torch.float16)" in body or ".half(" in body)
                      and ".float()" in body),
        # bug_007: slice_scatter / select_scatter backward stride-0
        ("bug_007_scatter_bwd",
         lambda body: "select_scatter" in body or "slice_scatter" in body),
    ]

    def __init__(self, output_dir, strategy_name="GraphV4",
                 guidance_mode="guided", novelty_batch_path=None,
                 novelty_ratio=0.35, **kwargs):
        super().__init__(output_dir, strategy_name=strategy_name, **kwargs)
        self.guidance_mode = guidance_mode
        self.novelty_batch_path = novelty_batch_path
        self.novelty_ratio = max(0.0, min(float(novelty_ratio or 0.0), 1.0))
        if guidance_mode == "random":
            self._V4_ADV_RATIO = 0.0
        self._coverage_timeseries = []
        self.total_adversarial_shapes = 0
        self.total_semantic_mutations = 0
        self.total_cuda_crashes = 0
        self.total_fp64_backoff_downgraded = 0
        self.total_fp64_backoff_confirmed = 0
        self.total_known_bug_filtered = 0
        self.total_backward_only_bugs = 0
        self.total_primary_bwd_runs = 0
        # Op-level backward coverage — track (template, dtype) pairs
        self._bwd_tested_pairs = set()  # set of "template_name|dtype" strings
        self._bwd_valid_pairs = set()   # denominator: all valid (template, dtype) combos
        _bwd_dtypes = [d.replace('torch.', '') for d in _DTYPES]
        for _tname, _tfn, _tw in PROGRAM_TEMPLATES:
            if _tname not in _NO_BACKWARD_TEMPLATES:
                for _dt in _bwd_dtypes:
                    self._bwd_valid_pairs.add(f"{_tname}|{_dt}")
        self.total_cfg_cfg_runs = 0
        self.total_cfg_cfg_diffs = 0
        self.total_cross_sched_runs = 0
        self.total_cross_sched_diffs = 0
        self._aoti_disabled = False  # set True after AOTI causes CUDA crash
        self.total_aoti_cuda_crashes = 0
        self.total_bwd_stress_mutations = 0
        self.total_kss_guided = 0  # count KSS-guided template selections
        self.total_guard_oracle_runs = 0
        self.total_guard_diffs = 0
        self.total_finite_diff_runs = 0
        self.total_finite_diff_bugs = 0
        self.total_dyn_multi_runs = 0
        self.total_dyn_multi_bugs = 0
        self.total_pattern_matcher_runs = 0
        self.total_pattern_matcher_bugs = 0
        self.total_aggr_sched_runs = 0
        self.total_aggr_sched_bugs = 0
        self.total_novelty_guided = 0
        self._novelty_seed_pool = self._load_novelty_seed_pool(novelty_batch_path)
        if self._novelty_seed_pool:
            print(f"  [NoveltyGuided] Loaded {len(self._novelty_seed_pool)} seeds "
                  f"from {novelty_batch_path} ratio={self.novelty_ratio:.0%}")
        # Kernel state space coverage (paper Section 4.3). Canonical
        # KernelStateSpace with (dtype, persistent, n_fused) per family.
        # Shared with evaluation scripts under evaluation/.
        from kernel.state_space import KernelStateSpace
        self.kernel_ss = KernelStateSpace()
        # Cross-worker KSS state persistence — survives worker restart
        self._kss_checkpoint_path = os.path.join(
            self._graph_dir, "kernel_ss_checkpoint.json")
        _restored = self.kernel_ss.load_checkpoint(self._kss_checkpoint_path)
        if _restored > 0:
            print(f"  [KernelSS] Restored {_restored} states from previous worker "
                  f"→ starting at {self.kernel_ss.get_coverage()*100:.1f}%")
        # Backward KSS — separate coverage tracker for backward codegen
        self.backward_kss = KernelStateSpace()
        self._bwd_kss_checkpoint_path = os.path.join(
            self._graph_dir, "backward_kss_checkpoint.json")
        _bwd_restored = self.backward_kss.load_checkpoint(self._bwd_kss_checkpoint_path)
        if _bwd_restored > 0:
            print(f"  [BwdKSS] Restored {_bwd_restored} states from previous worker "
                  f"→ starting at {self.backward_kss.get_coverage()*100:.1f}%")
        # Create crash report directory
        os.makedirs(join(self._graph_dir, "crash"), exist_ok=True)

    def _load_novelty_seed_pool(self, path):
        if not path:
            return []
        try:
            if not os.path.exists(path):
                print(f"  [NoveltyGuided] batch file not found: {path}")
                return []
            with open(path, "r") as f:
                data = json.load(f)
            selected = data.get("selected", []) if isinstance(data, dict) else []
            seeds = []
            for idx, item in enumerate(selected):
                body = item.get("body_code", "")
                if not body or "def program(" not in body or "_program_inputs" not in body:
                    continue
                sig = item.get("signature", {}) or {}
                meta = item.get("metadata", {}) or {}
                name = item.get("name") or f"novelty_seed_{idx}"
                seeds.append({
                    "name": name,
                    "body_code": body,
                    "dtype": meta.get("dtype") or _infer_dtype_from_body(body),
                    "batch": int(meta.get("batch") or _infer_first_dim_from_body(body) or 1),
                    "seq": int(meta.get("seq") or 1),
                    "hidden": int(meta.get("hidden") or 1),
                    "use_dynamic": bool(meta.get("use_dynamic") or "dynamic" in sig.get("mode_pair", "")),
                    "supports_backward": bool(meta.get("supports_backward", True)),
                })
            return seeds
        except Exception as e:
            print(f"  [NoveltyGuided] failed to load {path}: {type(e).__name__}: {e}")
            return []

    def _generate_novelty_program(self, inject_poison):
        if not self._novelty_seed_pool:
            return None
        seed = random.choice(self._novelty_seed_pool)
        body_code = seed["body_code"]
        dtype = seed.get("dtype") or _infer_dtype_from_body(body_code)
        name = seed.get("name", "novelty_seed")

        body_code = self._v4_maybe_semantic_mutate(body_code)
        body_code = self._v4_maybe_non_contiguous(body_code)
        body_code, val_strategy = _apply_value_diversification(body_code, dtype)
        if val_strategy:
            self.total_value_diverse = getattr(self, 'total_value_diverse', 0) + 1

        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code

        config_preset = _sample_config_preset()
        use_dynamic = bool(seed.get("use_dynamic")) or (random.random() < 0.25)
        prog = GraphProgram(
            name, body_code, True, dtype,
            int(seed.get("batch") or 1),
            int(seed.get("seq") or 1),
            int(seed.get("hidden") or 1),
            has_poison=poisoned,
            config_preset=config_preset,
            use_dynamic=use_dynamic,
            supports_backward=bool(seed.get("supports_backward", True)),
        )
        # Keep existing progress/result counters meaningful when the program
        # comes from an external novelty seed rather than ProgramGenerator.
        self.generator.generated_count += 1
        self.total_novelty_guided += 1
        return prog

    def _v4_sample_shapes(self):
        """Sample shapes with adversarial ratio."""
        b, s, h = _sample_shapes(adversarial_ratio=self._V4_ADV_RATIO)
        if b in _ADV_BATCH and s in _ADV_SEQ:
            self.total_adversarial_shapes += 1
        return b, s, h

    @staticmethod
    def _kss_directed_shapes(family):
        """Construct shapes that trigger a specific kernel family.

        The thresholds below are source-derived: each is read from Inductor's
        codegen decision logic (e.g. _get_heuristic / scheduler predicates), so a
        sampled shape lands on the boundary that selects the target family
        (rnumel<=2048 -> persistent, scan_dim>=129 -> split_scan, ...). This is
        static source-derived targeting, not a learned/adaptive shape search."""
        if family == 'cooperative_reduction':
            # xnumel small + rnumel huge → cooperative
            return (1, random.choice([2, 3, 4]), random.choice([8192, 16384, 32768]))
        elif family == 'persistent_reduction':
            # rnumel ≤ 2048 → persistent
            return (random.randint(1, 8), random.randint(16, 128),
                    random.choice([64, 128, 256, 512]))
        elif family == 'split_scan':
            # scan_dim ≥ 129 triggers split_scan
            return (random.randint(1, 4), random.choice([129, 130, 255, 256, 512]),
                    random.randint(16, 64))
        elif family == 'scan':
            # scan_dim < 129 → single-block scan
            return (random.randint(1, 4), random.choice([32, 64, 96, 127, 128]),
                    random.randint(16, 64))
        elif family == 'extern_gemm':
            # mm/addmm needs 2D matmul-compatible shapes
            return (random.randint(1, 8), random.choice([64, 128, 256]),
                    random.choice([64, 128, 256]))
        elif family == 'extern_conv':
            # conv needs spatial dims
            return (random.randint(1, 4), random.choice([16, 32, 64]),
                    random.choice([16, 32, 64]))
        else:
            # pointwise / reduction — standard
            return (random.randint(1, 8),
                    random.choice([33, 63, 127, 255]),
                    random.choice([33, 63, 127, 255]))

    def _v4_maybe_semantic_mutate(self, body_code):
        """Apply semantic mutation with configured probability."""
        if random.random() < self._V4_SEMANTIC_PROB:
            mutated, changed = _apply_semantic_mutation(body_code)
            if changed:
                self.total_semantic_mutations += 1
                return mutated
        return body_code

    # Layout class definitions for systematic stride fuzzing.
    # Each class produces a different stride pattern that stresses
    # Triton kernel indexing (ptr + stride_i * index_i).
    _LAYOUT_CLASSES = [
        # L1: permuted — stride order shuffled
        ('permuted', '{var} = {var}.transpose(-1, -2).contiguous().transpose(-1, -2)'),
        # L2: sliced — non-aligned offset, step > 1
        ('sliced', '{var} = torch.cat([{var}, {var}], dim=-1)[..., ::2]'),
        # L3: broadcasted — stride=0 on one dim (most dangerous for indexing)
        ('broadcast', '{var} = {var}.mean(dim=-1, keepdim=True).expand_as({var}).clone().requires_grad_({var}.requires_grad)'),
        # L4: channels_last — 4D memory format change (when applicable)
        ('channels_last', 'if {var}.dim() == 4: {var} = {var}.to(memory_format=torch.channels_last)'),
    ]

    def _v4_maybe_non_contiguous(self, body_code):
        """Layout class fuzzing — 25% chance to apply systematic
        stride mutations to input tensors. Instead of simple transpose,
        apply one of 4 layout classes (permuted/sliced/broadcast/channels_last).
        Key: mixed layout — different tensors get different classes."""
        if random.random() >= 0.25:
            return body_code
        lines = body_code.split("\n")
        mutated_any = False
        input_vars = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith("_") and "torch.randn" in line
                    and "=" in line and "_program_inputs" not in line):
                var = stripped.split("=")[0].strip()
                if var.startswith("_") and not var.startswith("_program"):
                    input_vars.append((i, var, line))

        if not input_vars:
            return body_code

        # Each tensor gets an independent layout class (mixed layout)
        for idx, (line_idx, var, line) in enumerate(reversed(input_vars)):
            lc_name, lc_tpl = random.choice(self._LAYOUT_CLASSES)
            indent = " " * (len(line) - len(line.lstrip()))
            guard = f"if {var}.dim() >= 2: " if lc_name != 'channels_last' else ''
            mutation_line = f"{indent}{guard}{lc_tpl.format(var=var)}"
            lines.insert(line_idx + 1, mutation_line)
            mutated_any = True

        if mutated_any:
            self.total_non_contig = getattr(self, 'total_non_contig', 0) + 1
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # FP64 backoff: re-run a bug-triggering program at fp64 to filter
    # out fp32 reduction-noise false positives.
    #
    # Returns the fp64 max_diff (float) or None if the check could not
    # be performed (program already at fp64, complex dtype, exec error,
    # etc.). The caller decides what to do with the value.
    # ------------------------------------------------------------------
    def _fp64_backoff_check(self, program, trigger_config_label):
        """
        Re-run `program` with all float dtypes promoted to float64 and compare
        eager vs the same inductor config that originally produced the diff.

        Returns:
            float or None — fp64 max_diff between eager and inductor (None if
            the backoff could not be evaluated).
        """
        body = program.body_code
        # Skip programs with explicit half/bf16 cast in the math
        # (those are bug_004 family — the cast IS the bug, fp64 doesn't apply).
        # The known-bug filter should already have downgraded these, but keep
        # this guard so the backoff path stays sane if called directly.
        if ".half(" in body or ".bfloat16(" in body or ".to(torch.bfloat16)" in body \
                or ".to(torch.float16)" in body:
            return None
        # Skip if program is already at fp64 and has no fp32/complex paths
        # to promote — backoff would re-run identical code.
        if ("torch.float32" not in body and "torch.float16" not in body
                and "torch.bfloat16" not in body and "torch.complex" not in body
                and "torch.fft.rfft" not in body):
            return None

        body64 = (body
                  .replace("torch.float32", "torch.float64")
                  .replace("torch.float16", "torch.float64")
                  .replace("torch.bfloat16", "torch.float64"))
        # For complex inputs: promote cfloat → cdouble by ensuring the real
        # parts are float64 before torch.complex(...). The substitution above
        # already handles `torch.randn(..., dtype=torch.float32)` lines.

        # Strip the inductor config preset label down to the actual config name
        # used in build_eval_code_v2. The label is e.g. "inductor_default" or
        # "inductor_reduce_overhead"; we always use the simplest 'default' here.
        # NOTE: do NOT rebind `results` — TorchLibrary.run_code passes the dict
        # in via the exec globals. Reassigning would silently drop our writes.
        eval_code = "\n".join([
            "import torch",
            "try:",
            "    import torch._inductor.config as _ic",
            "    _ic.fx_graph_cache = False",
            "    _ic.fx_graph_remote_cache = False",
            "except Exception:",
            "    pass",
            "",
            body64.strip(),
            "",
            "if torch.cuda.is_available():",
            "    _program_inputs = tuple(",
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)",
            "",
            "try:",
            "    results['eager'] = program(*_program_inputs)",
            "    results['eager_ok'] = True",
            "except Exception as _e:",
            "    results['eager_ok'] = False",
            "    results['err'] = str(_e)",
            "",
            "if results.get('eager_ok'):",
            "    try:",
            "        torch._dynamo.reset()",
            "        _ifn = torch.compile(program, backend='inductor')",
            "        _inp = tuple(t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs)",
            "        results['inductor'] = _ifn(*_inp)",
            "        results['inductor_ok'] = True",
            "    except Exception as _e:",
            "        results['inductor_ok'] = False",
            "        results['err'] = str(_e)",
            "    finally:",
            "        torch._dynamo.reset()",
        ])

        try:
            r, err, _ = TorchLibrary.run_code(eval_code)
        except Exception as e:
            print(f"  [FP64-Backoff] run_code raised: {type(e).__name__}: {e}")
            return None
        if err:
            return None
        if not r.get("eager_ok") or not r.get("inductor_ok"):
            return None

        eager = r.get("eager")
        ind = r.get("inductor")
        if eager is None or ind is None:
            return None
        try:
            # Handle tuple/list outputs by reducing each leaf and taking the max
            def _max_diff(a, b):
                if isinstance(a, (list, tuple)):
                    return max(_max_diff(x, y) for x, y in zip(a, b))
                if not isinstance(a, torch.Tensor):
                    return 0.0
                return (a.detach() - b.detach()).abs().max().item()
            return _max_diff(eager, ind)
        except Exception as e:
            print(f"  [FP64-Backoff] diff computation failed: {type(e).__name__}: {e}")
            return None

    def _run_primary_backward_oracle(self, program):
        """
        Primary backward oracle.

        Run on programs whose FORWARD looked clean, to catch bugs that only
        manifest in the backward pass. Many real Inductor numerical bugs
        live in the autograd-generated backward graph and never appear in
        forward — this oracle is the only way to find them at scale.

        Returns a GraphBugResultV2 if a catastrophic gradient diff is found,
        otherwise None. Skips programs that:
          - have non-deterministic ops (atomics, dropout, etc.)
          - have poison inputs (NaN/Inf would give false-positive grads)
          - are in the no-backward template list (in-place mutation, etc.)
          - have non-floating-point inputs only
        """
        if not program.supports_backward:
            return None
        if program.has_poison or not program.is_deterministic:
            return None
        if program.name in _NO_BACKWARD_TEMPLATES:
            return None

        try:
            self.total_primary_bwd_runs += 1
            is_amplified, grad_level, grad_max, bwd_err, bwd_kheurs = _run_backward_verification(
                program, program.dtype)
        except Exception as e:
            return None

        # Record op-level backward coverage (template x dtype)
        if hasattr(self, '_bwd_tested_pairs'):
            _dt = program.dtype.replace('torch.', '') if program.dtype else 'float32'
            self._bwd_tested_pairs.add(f"{program.name}|{_dt}")

        # Feed backward kernel heuristics into backward KSS
        if bwd_kheurs and hasattr(self, 'backward_kss'):
            for h in bwd_kheurs:
                self.backward_kss.record_hit('', program.dtype, heuristic=h)

        # Only report if we got a clean catastrophic signal — anything else
        # (errors, "output not differentiable", crashes during backward) is
        # ambiguous and we don't want noise here. Inductor backward crashes
        # ARE reported as backward_only with priority 3.
        if grad_level is None:
            return None
        if grad_level == DivergenceLevel.SUCCESS:
            return None
        if grad_level == DivergenceLevel.SUSPICIOUS:
            # Don't promote SUSPICIOUS gradient diffs — too many fp32 noise
            # cases (long reduction chains in backward), would crowd dedup.
            return None
        # CATASTROPHIC backward diff
        bug = GraphBugResultV2()
        bug.is_bug = True
        bug.bug_type = "backward_only"
        bug.priority = 3
        bug.divergence_level = DivergenceLevel.CATASTROPHIC
        bug.max_diff = grad_max if grad_max is not None else float("inf")
        bug.violation_ratio = 1.0
        bug.trigger_config = "inductor_backward"
        bug.category = program.category
        bug.error_msg = (f"[BACKWARD_ONLY] forward clean but gradient diverges: "
                         f"grad_max={bug.max_diff:.2e}"
                         + (f" err={bwd_err}" if bwd_err else ""))
        self.total_backward_only_bugs += 1
        return bug

    def _run_double_backward_oracle(self, program):
        """
        Double-backward (grad-of-grad) oracle.

        Tests `torch.autograd.grad(create_graph=True)` path — a completely
        separate bug surface from single backward. Inductor's lowering for
        higher-order derivatives has its own fusion decisions and has been
        shown to be fragile in areas like stride-0 backward (bug_007 family
        extended to 2nd order).

        Runs on ~5% of clean-forward programs. Skips same programs as
        primary backward oracle + a few extras (HOP requires more invariants).
        """
        if not program.supports_backward:
            return None
        if program.has_poison or not program.is_deterministic:
            return None
        if program.name in _NO_BACKWARD_TEMPLATES:
            return None

        dbwd_code = program.body_code.strip() + "\n"
        driver = (
            "import torch\n"
            + dbwd_code +
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "_program_inputs = tuple(\n"
            "    t.requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t\n"
            "    for t in _program_inputs)\n"
            "\n"
            "def _double_bwd(fn, inputs):\n"
            "    out = fn(*inputs)\n"
            "    if not (isinstance(out, torch.Tensor) and out.requires_grad):\n"
            "        return None\n"
            "    # First backward — with create_graph=True to keep the grad graph differentiable\n"
            "    _g = torch.autograd.grad(out.sum(), [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],\n"
            "                             create_graph=True, retain_graph=True)\n"
            "    # Second backward — grad-of-grad\n"
            "    _gg = torch.autograd.grad(sum(g.sum() for g in _g),\n"
            "                              [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],\n"
            "                              retain_graph=False, allow_unused=True)\n"
            "    return [g.clone() if g is not None else None for g in _gg]\n"
            "\n"
            "results['eager_gg_ok'] = False\n"
            "try:\n"
            "    results['eager_gg'] = _double_bwd(program, _program_inputs)\n"
            "    results['eager_gg_ok'] = (results['eager_gg'] is not None)\n"
            "except Exception as _e:\n"
            "    results['eager_gg_error'] = str(_e)[:300]\n"
            "\n"
            "results['ind_gg_ok'] = False\n"
            "if results.get('eager_gg_ok'):\n"
            "    try:\n"
            "        torch._dynamo.reset()\n"
            "        _inp_fresh = tuple(\n"
            "            t.detach().clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t\n"
            "            for t in _program_inputs)\n"
            "        _ifn = torch.compile(program, backend='inductor')\n"
            "        results['ind_gg'] = _double_bwd(_ifn, _inp_fresh)\n"
            "        results['ind_gg_ok'] = (results['ind_gg'] is not None)\n"
            "    except Exception as _e:\n"
            "        results['ind_gg_error'] = str(_e)[:300]\n"
            "    finally:\n"
            "        torch._dynamo.reset()\n"
        )

        try:
            self.total_double_bwd_runs = getattr(self, 'total_double_bwd_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error:
                return None
            if not results.get('eager_gg_ok') or not results.get('ind_gg_ok'):
                # Eager unable to compute grad-of-grad → not differentiable at 2nd order
                # Inductor crash → could be a real bug but often means unsupported op
                ind_err = results.get('ind_gg_error', '')
                if results.get('eager_gg_ok') and ind_err and TorchLibrary.is_crash_msg(ind_err):
                    # Eager OK but inductor crashes on double-backward — real bug
                    bug = GraphBugResultV2()
                    bug.is_bug = True
                    bug.bug_type = "double_backward_crash"
                    bug.priority = 3
                    bug.divergence_level = DivergenceLevel.CATASTROPHIC
                    bug.max_diff = float('inf')
                    bug.trigger_config = "inductor_double_backward"
                    bug.category = program.category
                    bug.error_msg = f"[DOUBLE_BWD_CRASH] inductor crashes on 2nd-order backward: {ind_err[:200]}"
                    return bug
                return None

            # Compare grad-of-grad values
            eager_gg = results.get('eager_gg', [])
            ind_gg = results.get('ind_gg', [])
            if not eager_gg or not ind_gg or len(eager_gg) != len(ind_gg):
                return None

            worst_diff = 0.0
            for eg, ig in zip(eager_gg, ind_gg):
                if eg is None or ig is None:
                    continue
                if not (isinstance(eg, torch.Tensor) and isinstance(ig, torch.Tensor)):
                    continue
                if eg.shape != ig.shape:
                    worst_diff = float('inf')
                    break
                try:
                    d = (eg.detach().cpu().float() - ig.detach().cpu().float()).abs().max().item()
                    worst_diff = max(worst_diff, d)
                except Exception:
                    continue

            # Threshold: grad-of-grad diffs accumulate 2× through fp32 chain
            # Use atol x 100 (same threshold as backward verify)
            atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
            if worst_diff > atol * 100:
                bug = GraphBugResultV2()
                bug.is_bug = True
                bug.bug_type = "double_backward_diff"
                bug.priority = 3
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                bug.max_diff = worst_diff
                bug.violation_ratio = 1.0
                bug.trigger_config = "inductor_double_backward"
                bug.category = program.category
                bug.error_msg = (f"[DOUBLE_BWD] grad-of-grad diverges: "
                                 f"max_diff={worst_diff:.2e} (threshold {atol*100:.1e})")
                self.total_double_bwd_bugs = getattr(self, 'total_double_bwd_bugs', 0) + 1
                return bug
        except Exception:
            return None
        return None

    # =========================================================================
    # Compiled Autograd Oracle
    # =========================================================================
    # Compiles the backward pass itself via torch._dynamo.config.compiled_autograd.
    # This exercises a completely different codegen path from standard inductor.
    _V4_COMPILED_AUTOGRAD_PROB = 0.10

    def _run_compiled_autograd_oracle(self, program):
        """Test compiled autograd: compile both forward AND backward passes.
        Compares gradient from standard inductor vs compiled autograd inductor."""
        if not program.supports_backward or not program.is_deterministic:
            return None
        if program.has_poison or program.name in _NO_BACKWARD_TEMPLATES:
            return None

        body = program.body_code.strip() + "\n"
        driver = (
            "import torch\n"
            "import torch._dynamo\n"
            + body +
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "_program_inputs = tuple(\n"
            "    t.requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t\n"
            "    for t in _program_inputs)\n"
            "\n"
            "# Standard inductor backward\n"
            "results['std_ok'] = False\n"
            "try:\n"
            "    torch._dynamo.reset()\n"
            "    _inp1 = tuple(t.detach().clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in _program_inputs)\n"
            "    _fn1 = torch.compile(program, backend='inductor')\n"
            "    _out1 = _fn1(*_inp1)\n"
            "    if isinstance(_out1, torch.Tensor) and _out1.requires_grad:\n"
            "        _out1.sum().backward()\n"
            "        results['std_grads'] = [t.grad.clone() for t in _inp1 if isinstance(t, torch.Tensor) and t.grad is not None]\n"
            "        results['std_ok'] = True\n"
            "    torch._dynamo.reset()\n"
            "except Exception as _e:\n"
            "    results['std_error'] = str(_e)[:300]\n"
            "    torch._dynamo.reset()\n"
            "\n"
            "# Compiled autograd backward\n"
            "results['ca_ok'] = False\n"
            "if results.get('std_ok'):\n"
            "    try:\n"
            "        torch._dynamo.reset()\n"
            "        torch._dynamo.config.compiled_autograd = True\n"
            "        _inp2 = tuple(t.detach().clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in _program_inputs)\n"
            "        _fn2 = torch.compile(program, backend='inductor')\n"
            "        _out2 = _fn2(*_inp2)\n"
            "        if isinstance(_out2, torch.Tensor) and _out2.requires_grad:\n"
            "            _out2.sum().backward()\n"
            "            results['ca_grads'] = [t.grad.clone() for t in _inp2 if isinstance(t, torch.Tensor) and t.grad is not None]\n"
            "            results['ca_ok'] = True\n"
            "        torch._dynamo.config.compiled_autograd = False\n"
            "        torch._dynamo.reset()\n"
            "    except Exception as _e:\n"
            "        results['ca_error'] = str(_e)[:300]\n"
            "        torch._dynamo.config.compiled_autograd = False\n"
            "        torch._dynamo.reset()\n"
        )

        try:
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
        except Exception:
            return None

        std_ok = results.get('std_ok', False)
        ca_ok = results.get('ca_ok', False)

        # Case 1: standard OK but compiled autograd crashes
        if std_ok and not ca_ok:
            ca_err = results.get('ca_error', '')
            if ca_err and not _is_unsupported_err(ca_err) and not _is_env_error(ca_err):
                bug = GraphBugResultV2()
                bug.is_bug = True
                bug.bug_type = "compiled_autograd_crash"
                bug.priority = 2
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                bug.max_diff = None
                bug.violation_ratio = 0.0
                bug.trigger_config = "compiled_autograd"
                bug.category = program.category
                bug.error_msg = f"[COMPILED_AUTOGRAD] crashes but standard inductor backward OK: {ca_err[:200]}"
                return bug

        # Case 2: both OK, compare gradients
        if std_ok and ca_ok:
            std_grads = results.get('std_grads', [])
            ca_grads = results.get('ca_grads', [])
            if len(std_grads) != len(ca_grads):
                bug = GraphBugResultV2()
                bug.is_bug = True
                bug.bug_type = "compiled_autograd_grad_count"
                bug.priority = 2
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                bug.max_diff = None
                bug.violation_ratio = 1.0
                bug.trigger_config = "compiled_autograd"
                bug.category = program.category
                bug.error_msg = f"[COMPILED_AUTOGRAD] gradient count mismatch: std={len(std_grads)} ca={len(ca_grads)}"
                return bug

            worst_diff = 0.0
            for sg, cg in zip(std_grads, ca_grads):
                worst_diff = max(worst_diff, (sg - cg).abs().max().item())

            atol = GRAPH_DIFF_BOUNDS.get(program.dtype, 1e-3)
            if worst_diff > atol * 10:
                bug = GraphBugResultV2()
                bug.is_bug = True
                bug.bug_type = "compiled_autograd_diff"
                bug.priority = 3
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                bug.max_diff = worst_diff
                bug.violation_ratio = 1.0
                bug.trigger_config = "compiled_autograd"
                bug.category = program.category
                bug.error_msg = f"[COMPILED_AUTOGRAD] grad diverges: max_diff={worst_diff:.2e}"
                return bug

        return None

    # Semantic invariant probes — each probe is a short program whose
    # output must satisfy a mathematical invariant. If inductor breaks the
    # invariant, we have a bug WITHOUT needing eager as ground truth.
    #
    # FP protection: all invariants use fp64 internally to avoid fp32 noise,
    # and tolerances are set per-invariant based on expected numerical behavior.
    _SEMANTIC_INVARIANTS = [
        # (name, program_str, invariant_check_str, tolerance)
        # softmax output must sum to 1 along its dim
        ("softmax_sums_to_1",
         "import torch\n_x = torch.randn(4, 32, 128, dtype=torch.float64)\n_program_inputs = (_x,)\n"
         "def program(x): return torch.softmax(x, dim=-1)",
         "out.sum(dim=-1)",  # should be ~1.0 everywhere
         1e-6,
         "1.0"),
        # cumsum last element equals total sum
        ("cumsum_last_eq_sum",
         "import torch\n_x = torch.randn(4, 64, dtype=torch.float64)\n_program_inputs = (_x,)\n"
         "def program(x): return x.cumsum(dim=-1)",
         "out[..., -1]",  # should equal x.sum(-1)
         1e-8,
         "_program_inputs[0].sum(dim=-1)"),
        # contiguous is idempotent
        ("contig_idempotent",
         "import torch\n_x = torch.randn(3, 17, 11, dtype=torch.float64).transpose(0, 1)\n_program_inputs = (_x,)\n"
         "def program(x): return x.contiguous().contiguous()",
         "out",
         1e-10,
         "_program_inputs[0].contiguous()"),
        # relu is idempotent
        ("relu_idempotent",
         "import torch\n_x = torch.randn(4, 32, dtype=torch.float64)\n_program_inputs = (_x,)\n"
         "def program(x): return torch.relu(torch.relu(x))",
         "out",
         1e-10,
         "torch.relu(_program_inputs[0])"),
        # double transpose is identity
        ("transpose_roundtrip",
         "import torch\n_x = torch.randn(7, 5, 11, dtype=torch.float64)\n_program_inputs = (_x,)\n"
         "def program(x): return x.transpose(-1, -2).transpose(-1, -2)",
         "out",
         1e-10,
         "_program_inputs[0]"),
    ]

    def _run_semantic_invariant_oracle(self):
        """Run a random semantic invariant check under inductor.
        Returns GraphBugResultV2 if invariant is violated, else None.

        Independent of program under test — picks one of the built-in probes.
        The probes are fp64 to avoid precision-noise false positives.
        """
        probe_name, body, invariant_expr, tol, expected_expr = random.choice(self._SEMANTIC_INVARIANTS)
        driver = (
            f"{body}\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "results['invariant_ok'] = False\n"
            "try:\n"
            "    torch._dynamo.reset()\n"
            "    _fn = torch.compile(program, backend='inductor')\n"
            "    out = _fn(*_program_inputs)\n"
            "    if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "    torch._dynamo.reset()\n"
            f"    actual = {invariant_expr}\n"
            f"    expected = {expected_expr}\n"
            "    if isinstance(expected, (int, float)):\n"
            "        expected_t = torch.full_like(actual, float(expected))\n"
            "    else:\n"
            "        expected_t = expected\n"
            "    diff = (actual.float() - expected_t.float()).abs().max().item()\n"
            "    results['invariant_diff'] = diff\n"
            "    results['invariant_ok'] = True\n"
            "except Exception as _e:\n"
            "    results['invariant_error'] = str(_e)[:200]\n"
        )

        try:
            self.total_semantic_invariant_runs = getattr(self, 'total_semantic_invariant_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not results.get('invariant_ok'):
                return None
            diff = results.get('invariant_diff', 0.0)
            if diff > tol:
                bug = GraphBugResultV2()
                bug.is_bug = True
                bug.bug_type = "semantic_invariant_violation"
                bug.priority = 3
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
                bug.max_diff = diff
                bug.violation_ratio = 1.0
                bug.trigger_config = "inductor_invariant"
                bug.category = "INVARIANT"
                bug.error_msg = (f"[INVARIANT] {probe_name} violated: "
                                 f"diff={diff:.2e} (tol={tol:.0e})")
                self.total_semantic_invariant_bugs = getattr(self, 'total_semantic_invariant_bugs', 0) + 1
                return bug
        except Exception:
            return None
        return None

    def _run_cfg_cfg_oracle(self, results, program):
        """
        Secondary config-vs-config oracle.

        When eager-vs-inductor agrees on all configs, we additionally compare
        the OUTPUTS of two different inductor configs against each other.
        Disagreements indicate one config is wrong (or both are wrong but
        differently) — neither side is ground truth, so this oracle is
        STRICTLY weak-signal: bugs go to weak-signal/, never catastrophic.
        Empirically these surface fusion-order/cudagraph bugs that eager
        cannot catch because eager isn't on the buggy code path at all.

        Returns a GraphBugResultV2 (priority=1, divergence=SUSPICIOUS) or None.
        """
        if not program.is_deterministic or program.has_poison:
            return None
        ind_results = results.get("inductor_results") or {}
        # Collect successful inductor outputs from non-dynamic configs
        outs = {}
        for lbl, cr in ind_results.items():
            if "dynamic" in lbl:
                continue
            if not cr.get("ok", False):
                continue
            r = cr.get("result")
            if r is None:
                continue
            outs[lbl] = r
        if len(outs) < 2:
            return None
        self.total_cfg_cfg_runs += 1

        labels = list(outs.keys())
        worst_lbl_a = worst_lbl_b = None
        worst_max = 0.0
        # All-pairs would be quadratic but only 2-3 configs, fine.
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                _, ma, _ = check_divergence(outs[la], outs[lb], program.dtype)
                if ma is not None and ma > worst_max:
                    worst_max = ma
                    worst_lbl_a, worst_lbl_b = la, lb

        if worst_max < self._V4_CFG_CFG_THRESHOLD:
            return None

        self.total_cfg_cfg_diffs += 1
        bug = GraphBugResultV2()
        bug.is_bug = True
        bug.bug_type = "cfg_cfg_diff"
        bug.priority = 1
        bug.divergence_level = DivergenceLevel.SUSPICIOUS  # never catastrophic
        bug.max_diff = worst_max
        bug.violation_ratio = 0.0
        bug.trigger_config = f"inductor_{worst_lbl_a}_vs_{worst_lbl_b}"
        bug.category = program.category
        bug.error_msg = (f"[CFG_VS_CFG] eager agrees with both but "
                         f"inductor({worst_lbl_a}) vs inductor({worst_lbl_b}) "
                         f"max_diff={worst_max:.2e}")
        return bug

    # =========================================================================
    # Config Differential Oracle: recompile same program with two orthogonal
    # inductor configs and compare outputs. Unlike cfg_cfg which passively
    # compares already-executed configs, this actively picks two configs that
    # exercise DIFFERENT codegen paths (e.g., fusion vs no-fusion).
    # =========================================================================
    _V4_CONFIG_DIFF_PROB = 0.15
    _V4_CONFIG_DIFF_PAIRS = [
        # (config_a, config_b) — each is a dict of inductor config overrides
        ({"epilogue_fusion": True, "pattern_matcher": True},
         {"epilogue_fusion": False, "pattern_matcher": False}),
        ({"combo_kernels": True},
         {"combo_kernels": False}),
        ({"freezing": True},
         {"freezing": False}),
        ({"max_autotune": True, "coordinate_descent_tuning": True},
         {}),
        ({"prologue_fusion": True},
         {"prologue_fusion": False}),
        ({"online_softmax": True},
         {"online_softmax": False}),
        ({"split_reductions": True},
         {"split_reductions": False}),
        # Buffer reuse / inplace — different memory layouts can expose aliasing bugs
        ({"allow_buffer_reuse": True, "inplace_buffers": True},
         {"allow_buffer_reuse": False, "inplace_buffers": False}),
        # cpp_wrapper uses different codegen path entirely
        ({"cpp_wrapper": True},
         {"cpp_wrapper": False}),
        # Aggressive fusion changes what ops get fused together
        ({"aggressive_fusion": True},
         {"aggressive_fusion": False}),
        # Shape padding affects Triton tile masking
        ({"comprehensive_padding": True, "force_shape_pad": True},
         {"comprehensive_padding": False, "shape_padding": False}),
        # Constant folding can change graph structure
        ({"joint_graph_constant_folding": True},
         {"joint_graph_constant_folding": False}),
        # Combo: freezing + max_autotune (two codegen paths at once)
        ({"freezing": True, "max_autotune": True},
         {"freezing": False, "max_autotune": False}),
        # fallback_random: different random number generation path
        ({"fallback_random": True},
         {"fallback_random": False}),
    ]

    def _run_config_diff_oracle(self, program):
        """Recompile same program with two different inductor configs.
        Returns GraphBugResultV2 if outputs diverge, else None.
        """
        if not program.is_deterministic or program.has_poison:
            return None

        pair = random.choice(self._V4_CONFIG_DIFF_PAIRS)
        cfg_a, cfg_b = pair

        def _cfg_lines(cfg):
            lines = ["import torch._inductor.config as _ic"]
            for k, v in cfg.items():
                lines.append(
                    f"try:\n    _ic.{k} = {v!r}\n"
                    f"except AttributeError:\n    pass")
            return "\n".join(lines)

        body = program.body_code
        driver = (
            "import torch\n"
            "import torch._dynamo\n"
            "results = {}\n\n"
            # Run A
            f"{_cfg_lines(cfg_a)}\n"
            "torch._dynamo.reset()\n"
            f"{body}\n"
            "try:\n"
            "    _compiled_a = torch.compile(program)\n"
            "    _out_a = _compiled_a(*[t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs])\n"
            "    if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "    results['out_a'] = _out_a.detach().clone() if isinstance(_out_a, torch.Tensor) else _out_a\n"
            "    results['ok_a'] = True\n"
            "except Exception as _ea:\n"
            "    results['ok_a'] = False\n"
            "    results['err_a'] = str(_ea)[:500]\n"
            "torch._dynamo.reset()\n\n"
            # Run B
            f"{_cfg_lines(cfg_b)}\n"
            f"{body}\n"
            "try:\n"
            "    _compiled_b = torch.compile(program)\n"
            "    _out_b = _compiled_b(*[t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs])\n"
            "    if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "    results['out_b'] = _out_b.detach().clone() if isinstance(_out_b, torch.Tensor) else _out_b\n"
            "    results['ok_b'] = True\n"
            "except Exception as _eb:\n"
            "    results['ok_b'] = False\n"
            "    results['err_b'] = str(_eb)[:500]\n"
            "torch._dynamo.reset()\n"
        )

        try:
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
        except Exception:
            return None

        ok_a = results.get('ok_a', False)
        ok_b = results.get('ok_b', False)

        cfg_a_str = ",".join(f"{k}={v}" for k, v in cfg_a.items()) or "default"
        cfg_b_str = ",".join(f"{k}={v}" for k, v in cfg_b.items()) or "default"

        # Case 1: one crashes, the other doesn't
        if ok_a != ok_b:
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "config_diff_crash"
            bug.priority = 2
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = None
            bug.violation_ratio = 0.0
            bug.trigger_config = f"config_diff_{cfg_a_str}_vs_{cfg_b_str}"
            bug.category = program.category
            crash_side = "A" if not ok_a else "B"
            crash_cfg = cfg_a_str if not ok_a else cfg_b_str
            crash_err = results.get('err_a', '') if not ok_a else results.get('err_b', '')
            bug.error_msg = (
                f"[CONFIG_DIFF] config {crash_cfg} crashes but "
                f"{'default' if crash_side == 'A' else cfg_a_str} succeeds: "
                f"{crash_err[:200]}")
            return bug

        if not ok_a or not ok_b:
            return None

        # Case 2: both succeed, compare outputs
        out_a = results.get('out_a')
        out_b = results.get('out_b')
        if out_a is None or out_b is None:
            return None
        if not isinstance(out_a, torch.Tensor) or not isinstance(out_b, torch.Tensor):
            return None

        level, max_abs, vr = check_divergence(out_a, out_b, program.dtype)
        if level == DivergenceLevel.SUCCESS:
            return None

        bug = GraphBugResultV2()
        bug.is_bug = True
        bug.bug_type = "config_diff_divergence"
        bug.priority = 2 if level == DivergenceLevel.CATASTROPHIC else 1
        bug.divergence_level = level
        bug.max_diff = max_abs
        bug.violation_ratio = vr
        bug.trigger_config = f"config_diff_{cfg_a_str}_vs_{cfg_b_str}"
        bug.category = program.category
        bug.error_msg = (
            f"[CONFIG_DIFF] {cfg_a_str} vs {cfg_b_str}: "
            f"max_diff={max_abs:.2e} vr={vr:.4f} [{level}]")
        return bug

    # Pattern Matcher Oracle: isolate pattern_matcher rewrites by toggling
    # only that flag. Pattern matcher silently rewrites subgraphs (matmul+add
    # -> addmm, softmax decomp, etc.) — if a rewrite is buggy, outputs diverge.
    _V4_PATTERN_MATCHER_PROB = 0.18

    def _run_pattern_matcher_oracle(self, program):
        """Toggle pattern_matcher on/off and compare outputs."""
        if not program.is_deterministic or program.has_poison:
            return None

        self.total_pattern_matcher_runs += 1

        def _cfg_lines(pm_val):
            return (
                "import torch._inductor.config as _ic\n"
                f"try:\n    _ic.pattern_matcher = {pm_val!r}\n"
                f"except AttributeError:\n    pass"
            )

        body = program.body_code
        driver = (
            "import torch\n"
            "import torch._dynamo\n"
            "results = {}\n\n"
            f"{_cfg_lines(True)}\n"
            "torch._dynamo.reset()\n"
            f"{body}\n"
            "try:\n"
            "    _compiled_a = torch.compile(program)\n"
            "    _out_a = _compiled_a(*[t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs])\n"
            "    if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "    results['out_a'] = _out_a.detach().clone() if isinstance(_out_a, torch.Tensor) else _out_a\n"
            "    results['ok_a'] = True\n"
            "except Exception as _ea:\n"
            "    results['ok_a'] = False\n"
            "    results['err_a'] = str(_ea)[:500]\n"
            "torch._dynamo.reset()\n\n"
            f"{_cfg_lines(False)}\n"
            f"{body}\n"
            "try:\n"
            "    _compiled_b = torch.compile(program)\n"
            "    _out_b = _compiled_b(*[t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs])\n"
            "    if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "    results['out_b'] = _out_b.detach().clone() if isinstance(_out_b, torch.Tensor) else _out_b\n"
            "    results['ok_b'] = True\n"
            "except Exception as _eb:\n"
            "    results['ok_b'] = False\n"
            "    results['err_b'] = str(_eb)[:500]\n"
            "torch._dynamo.reset()\n"
        )

        try:
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
        except Exception:
            return None

        ok_a = results.get('ok_a', False)
        ok_b = results.get('ok_b', False)

        if ok_a != ok_b:
            self.total_pattern_matcher_bugs += 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "pattern_matcher_crash"
            bug.priority = 3
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = None
            bug.violation_ratio = 0.0
            bug.trigger_config = "pattern_matcher_on_vs_off"
            bug.category = program.category
            crash_side = "ON" if not ok_a else "OFF"
            crash_err = results.get('err_a', '') if not ok_a else results.get('err_b', '')
            bug.error_msg = (
                f"[PATTERN_MATCHER] pattern_matcher={crash_side} crashes but "
                f"{'OFF' if crash_side == 'ON' else 'ON'} succeeds: "
                f"{crash_err[:200]}")
            return bug

        if not ok_a or not ok_b:
            return None

        out_a = results.get('out_a')
        out_b = results.get('out_b')
        if out_a is None or out_b is None:
            return None
        if not isinstance(out_a, torch.Tensor) or not isinstance(out_b, torch.Tensor):
            return None

        level, max_abs, vr = check_divergence(out_a, out_b, program.dtype)
        if level == DivergenceLevel.SUCCESS:
            return None

        self.total_pattern_matcher_bugs += 1
        bug = GraphBugResultV2()
        bug.is_bug = True
        bug.bug_type = "pattern_matcher_divergence"
        bug.priority = 3 if level == DivergenceLevel.CATASTROPHIC else 2
        bug.divergence_level = level
        bug.max_diff = max_abs
        bug.violation_ratio = vr
        bug.trigger_config = "pattern_matcher_on_vs_off"
        bug.category = program.category
        bug.error_msg = (
            f"[PATTERN_MATCHER] ON vs OFF: "
            f"max_diff={max_abs:.2e} vr={vr:.4f} [{level}]")
        return bug

    # Cross-Schedule Oracle: force two different kernel heuristics on the same
    # graph and compare outputs. If they disagree, one codegen path has a bug.
    # This is more targeted than cfg_cfg (which varies config flags) — here we
    # directly control which kernel template the compiler emits.
    _V4_CROSS_SCHED_PROB = 0.05  # 5% of clean iterations
    _V4_CROSS_SCHED_PAIRS = [
        ('persistent_reduction', 'reduction'),
        ('persistent_reduction', 'cooperative_reduction'),
        ('reduction', 'cooperative_reduction'),
    ]

    def _run_cross_schedule_oracle(self, program):
        """Force two different reduction heuristics on the same program.
        Returns GraphBugResultV2 if outputs disagree, else None.

        Mechanism: monkeypatch TritonKernel._get_heuristic to return a fixed
        string, recompile with each forced heuristic, compare outputs.
        Only meaningful for programs that contain reduction kernels.
        """
        if not program.is_deterministic or program.has_poison:
            return None
        if not hasattr(program, 'body_code') or not program.body_code:
            return None

        pair = random.choice(self._V4_CROSS_SCHED_PAIRS)
        heur_a, heur_b = pair

        body_and_inputs = program.body_code
        driver = (
            "import torch\n"
            "import torch._inductor.config as inductor_config\n"
            + body_and_inputs + "\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "\n"
            "def _force_compile(forced_heuristic):\n"
            "    torch._dynamo.reset()\n"
            "    from torch._inductor.codegen.triton import TritonKernel as _TK\n"
            "    _orig = _TK._get_heuristic\n"
            "    def _forced(_self):\n"
            "        return forced_heuristic\n"
            "    _TK._get_heuristic = _forced\n"
            "    try:\n"
            "        fn = torch.compile(program, backend='inductor', fullgraph=False)\n"
            "        out = fn(*_program_inputs)\n"
            "        if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "        return out\n"
            "    except Exception as e:\n"
            "        return str(e)[:200]\n"
            "    finally:\n"
            "        _TK._get_heuristic = _orig\n"
            "        torch._dynamo.reset()\n"
            "\n"
            f"out_a = _force_compile('{heur_a}')\n"
            f"out_b = _force_compile('{heur_b}')\n"
            "results['cross_sched_ok'] = False\n"
            "if isinstance(out_a, str) or isinstance(out_b, str):\n"
            "    results['cross_sched_error'] = f'a={type(out_a).__name__} b={type(out_b).__name__}'\n"
            "elif isinstance(out_a, torch.Tensor) and isinstance(out_b, torch.Tensor):\n"
            "    diff = (out_a.float() - out_b.float()).abs().max().item()\n"
            "    results['cross_sched_diff'] = diff\n"
            "    results['cross_sched_ok'] = True\n"
            "else:\n"
            "    results['cross_sched_error'] = 'non-tensor output'\n"
        )

        try:
            self.total_cross_sched_runs = getattr(self, 'total_cross_sched_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('cross_sched_ok'):
                return None
            diff = results.get('cross_sched_diff', 0.0)
            if diff is None or diff < self._V4_CFG_CFG_THRESHOLD:
                return None

            self.total_cross_sched_diffs = getattr(self, 'total_cross_sched_diffs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "cross_schedule_diff"
            bug.priority = 2
            bug.divergence_level = DivergenceLevel.SUSPICIOUS
            if diff > 1.0:
                bug.priority = 3
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = diff
            bug.violation_ratio = 0.0
            bug.trigger_config = f"forced_{heur_a}_vs_{heur_b}"
            bug.category = program.category
            bug.error_msg = (
                f"[CROSS_SCHEDULE] forced {heur_a} vs {heur_b}: "
                f"max_diff={diff:.2e}")
            return bug
        except Exception:
            return None

    # Cross-Schedule Backward Oracle: force different kernel heuristics on the
    # same backward graph and compare gradients.
    _V4_CROSS_SCHED_BWD_PROB = 0.03  # 3% of backward-eligible clean iterations

    def _run_cross_schedule_backward_oracle(self, program):
        """Force two different reduction heuristics on the backward pass.
        Returns GraphBugResultV2 if gradients disagree, else None."""
        if not program.supports_backward:
            return None
        if not program.is_deterministic or program.has_poison:
            return None
        if program.name in _NO_BACKWARD_TEMPLATES:
            return None
        if not hasattr(program, 'body_code') or not program.body_code:
            return None

        pair = random.choice(self._V4_CROSS_SCHED_PAIRS)
        heur_a, heur_b = pair

        body_and_inputs = program.body_code.strip()
        driver = (
            "import torch\n"
            "import torch._inductor.config as inductor_config\n"
            + body_and_inputs + "\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "\n"
            "def _bwd_with_heuristic(forced_heuristic, inputs_clone):\n"
            "    torch._dynamo.reset()\n"
            "    from torch._inductor.codegen.triton import TritonKernel as _TK\n"
            "    _orig = _TK._get_heuristic\n"
            "    def _forced(_self):\n"
            "        return forced_heuristic\n"
            "    _TK._get_heuristic = _forced\n"
            "    try:\n"
            "        inputs = tuple(\n"
            "            t.clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point()\n"
            "            else t for t in inputs_clone)\n"
            "        fn = torch.compile(program, backend='inductor', fullgraph=False)\n"
            "        out = fn(*inputs)\n"
            "        if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "        if not isinstance(out, torch.Tensor) or not out.requires_grad:\n"
            "            return None\n"
            "        grad_out = torch.randn_like(out)\n"
            "        out.backward(grad_out)\n"
            "        if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "        grads = [t.grad.clone() for t in inputs if isinstance(t, torch.Tensor) and t.grad is not None]\n"
            "        return grads, grad_out\n"
            "    except Exception:\n"
            "        return None\n"
            "    finally:\n"
            "        _TK._get_heuristic = _orig\n"
            "        torch._dynamo.reset()\n"
            "\n"
            f"res_a = _bwd_with_heuristic('{heur_a}', _program_inputs)\n"
            "if res_a is None:\n"
            "    results['cross_sched_bwd_ok'] = False\n"
            "else:\n"
            f"    res_b = _bwd_with_heuristic('{heur_b}', _program_inputs)\n"
            "    if res_b is None:\n"
            "        results['cross_sched_bwd_ok'] = False\n"
            "    else:\n"
            "        grads_a, _ = res_a\n"
            "        grads_b, _ = res_b\n"
            "        if len(grads_a) != len(grads_b) or len(grads_a) == 0:\n"
            "            results['cross_sched_bwd_ok'] = False\n"
            "        else:\n"
            "            max_diff = max((ga.float() - gb.float()).abs().max().item()\n"
            "                           for ga, gb in zip(grads_a, grads_b))\n"
            "            results['cross_sched_bwd_diff'] = max_diff\n"
            "            results['cross_sched_bwd_ok'] = True\n"
        )

        try:
            self.total_cross_sched_bwd_runs = getattr(self, 'total_cross_sched_bwd_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('cross_sched_bwd_ok'):
                return None
            diff = results.get('cross_sched_bwd_diff', 0.0)
            if diff is None or diff < self._V4_CFG_CFG_THRESHOLD:
                return None

            self.total_cross_sched_bwd_diffs = getattr(self, 'total_cross_sched_bwd_diffs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "cross_schedule_bwd_diff"
            bug.priority = 2
            bug.divergence_level = DivergenceLevel.SUSPICIOUS
            if diff > 1.0:
                bug.priority = 3
                bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = diff
            bug.violation_ratio = 0.0
            bug.trigger_config = f"bwd_forced_{heur_a}_vs_{heur_b}"
            bug.category = program.category
            bug.error_msg = (
                f"[CROSS_SCHEDULE_BWD] forced {heur_a} vs {heur_b}: "
                f"grad_max_diff={diff:.2e}")
            return bug
        except Exception:
            return None

    # =========================================================================
    # AOTI subprocess isolation
    # =========================================================================
    # Run AOTI in a child process so C++ compile errors / CUDA crashes
    # cannot corrupt the main fuzzer's CUDA context.
    _V4_AOTI_PROB = 0.30
    _V4_AOTI_TIMEOUT = 120

    def _run_aoti_subprocess(self, program):
        """Run AOTI export+compile+load+run in a subprocess.
        Returns dict with aoti_ok, aoti_error, aoti_export_failed, aoti_diff."""
        if program.skip_aoti or program.dtype == "torch.float64":
            return None
        if getattr(self, '_aoti_disabled', False):
            return None

        body = program.body_code
        if not body or "_program_inputs" not in body:
            return None

        aoti_dir = os.path.join(self._graph_dir, "tmp_aoti")
        os.makedirs(aoti_dir, exist_ok=True)

        driver = (
            "import torch, torch._dynamo, sys, os, json\n"
            "os.environ.setdefault('TORCH_LOGS', '')\n"
            "\n"
            + body.strip() + "\n"
            "\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "\n"
            "r = {'aoti_ok': False, 'aoti_export_failed': False}\n"
            "try:\n"
            "    _eager_out = program(*tuple(\n"
            "        t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs))\n"
            "except Exception as e:\n"
            "    r['eager_error'] = str(e)[:200]\n"
            "    print('AOTI_RESULT:' + json.dumps(r)); sys.exit(0)\n"
            "\n"
            "class _M(torch.nn.Module):\n"
            "    def forward(self, *args): return program(*args)\n"
            "try:\n"
            "    _ep = torch.export.export(_M(), tuple(\n"
            "        t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs))\n"
            "except Exception as e:\n"
            "    r['aoti_export_failed'] = True\n"
            "    r['aoti_export_error'] = str(e)[:300]\n"
            "    print('AOTI_RESULT:' + json.dumps(r)); sys.exit(0)\n"
            "\n"
            f"_pkg = os.path.join('{aoti_dir}', f'aoti_sub_{{os.getpid()}}.pt2')\n"
            "try:\n"
            "    import torch._inductor as _ind\n"
            "    _actual = _ind.aoti_compile_and_package(_ep, package_path=_pkg)\n"
            "    _loaded = _ind.aoti_load_package(_actual)\n"
            "    _aoti_inp = tuple(\n"
            "        t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "    _aoti_out = _loaded(*_aoti_inp)\n"
            "    if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "    r['aoti_ok'] = True\n"
            "    if isinstance(_eager_out, torch.Tensor) and isinstance(_aoti_out, torch.Tensor):\n"
            "        r['aoti_diff'] = (_eager_out.float() - _aoti_out.float()).abs().max().item()\n"
            "except Exception as e:\n"
            "    r['aoti_error'] = str(e)[:1000]\n"
            "finally:\n"
            "    try:\n"
            "        if os.path.exists(_pkg): os.remove(_pkg)\n"
            "        if '_actual' in dir() and os.path.exists(_actual): os.remove(_actual)\n"
            "    except: pass\n"
            "print('AOTI_RESULT:' + json.dumps(r))\n"
        )

        import tempfile
        import subprocess
        import sys
        tmp_path = os.path.join(aoti_dir, f"aoti_driver_{os.getpid()}_{id(program)}.py")
        try:
            with open(tmp_path, 'w') as f:
                f.write(driver)
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True,
                timeout=self._V4_AOTI_TIMEOUT,
                env={**os.environ},
            )
            for line in result.stdout.split('\n'):
                if line.startswith('AOTI_RESULT:'):
                    return json.loads(line[len('AOTI_RESULT:'):])
            if result.returncode != 0:
                return {'aoti_ok': False,
                        'aoti_error': f'subprocess exit {result.returncode}: {result.stderr[-500:]}'}
            return None
        except subprocess.TimeoutExpired:
            return {'aoti_ok': False, 'aoti_error': 'AOTI subprocess timeout'}
        except Exception as e:
            return {'aoti_ok': False, 'aoti_error': f'subprocess error: {e}'}
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    # =========================================================================
    # Dynamic Shape Multi-Call Oracle
    # =========================================================================
    # Compile with dynamic=True, then call the SAME compiled fn with a
    # *sequence* of shapes including boundary values (2^n ± 1).  Compares each
    # call's output against eager ground truth.  This catches:
    #   - Guard system too loose → compiled code reused for incompatible shape
    #   - Symbolic shape specialization wrong → kernel tile/mask error
    #   - Recompilation path produces different result than first compile
    _V4_DYN_MULTI_PROB = 0.08

    _DYN_MULTI_SHAPE_OFFSETS = [-2, -1, 0, 1, 2, 3]

    def _run_dynamic_shape_multi_oracle(self, program):
        """Call the same compiled fn with 3-4 different shapes sequentially.
        Compare each against eager.  Returns bug if any call disagrees."""
        body = program.body_code
        if not body or "_program_inputs" not in body:
            return None
        if (torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] < 8
                and ('bfloat16' in body or 'bf16' in body)):
            return None

        driver = (
            "import torch\n"
            "import torch._dynamo\n"
            "torch._dynamo.reset()\n"
            "\n"
            + body + "\n"
            "\n"
            "import random as _rnd\n"
            "_shapes_tried = []\n"
            "_max_diff = 0.0\n"
            "_worst_shape = None\n"
            "_n_ok = 0\n"
            "\n"
            "try:\n"
            "    _compiled_fn = torch.compile(program, backend='inductor', dynamic=True)\n"
            "    # First call: establish guards with original shape\n"
            "    _orig_inp = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "    _compiled_fn(*_orig_inp)\n"
            "\n"
            "    # Shape variants: boundary shapes (2^n ± 1), scaled dims\n"
            "    _boundary_deltas = [-2, -1, 0, 1, 2, 3, 7, -3]\n"
            "    _rnd.shuffle(_boundary_deltas)\n"
            "    for _delta in _boundary_deltas[:4]:\n"
            "        try:\n"
            "            _var_inp = []\n"
            "            _new_shapes = []\n"
            "            for _t in _program_inputs:\n"
            "                if isinstance(_t, torch.Tensor) and _t.dim() >= 1:\n"
            "                    _ns = list(_t.shape)\n"
            "                    _dim = _rnd.randint(0, len(_ns) - 1)\n"
            "                    _ns[_dim] = max(1, _ns[_dim] + _delta)\n"
            "                    _new_t = torch.randn(_ns, dtype=_t.dtype, device=_t.device)\n"
            "                    _var_inp.append(_new_t)\n"
            "                    _new_shapes.append(tuple(_ns))\n"
            "                else:\n"
            "                    _var_inp.append(_t.clone() if isinstance(_t, torch.Tensor) else _t)\n"
            "            _var_inp = tuple(_var_inp)\n"
            "            # Eager ground truth\n"
            "            _eager_out = program(*tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _var_inp))\n"
            "            # Compiled (may reuse existing guard or recompile)\n"
            "            _comp_out = _compiled_fn(*tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _var_inp))\n"
            "            if isinstance(_eager_out, torch.Tensor) and isinstance(_comp_out, torch.Tensor):\n"
            "                if _eager_out.shape == _comp_out.shape:\n"
            "                    _d = (_eager_out.float() - _comp_out.float()).abs().max().item()\n"
            "                    _shapes_tried.append(str(_new_shapes))\n"
            "                    if _d > _max_diff:\n"
            "                        _max_diff = _d\n"
            "                        _worst_shape = str(_new_shapes)\n"
            "                    _n_ok += 1\n"
            "        except Exception:\n"
            "            continue\n"
            "    results['dyn_multi_ok'] = _n_ok >= 2\n"
            "    results['dyn_multi_max_diff'] = _max_diff\n"
            "    results['dyn_multi_worst_shape'] = _worst_shape\n"
            "    results['dyn_multi_n_shapes'] = _n_ok\n"
            "except Exception as _e:\n"
            "    results['dyn_multi_ok'] = False\n"
            "    results['dyn_multi_error'] = str(_e)[:300]\n"
            "finally:\n"
            "    torch._dynamo.reset()\n"
        )

        try:
            self.total_dyn_multi_runs = getattr(self, 'total_dyn_multi_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('dyn_multi_ok'):
                return None

            max_diff = results.get('dyn_multi_max_diff', 0.0)
            dtype = program.dtype or "torch.float32"
            base_tol = GRAPH_DIFF_BOUNDS.get(dtype, 1e-4)
            threshold = base_tol * 100

            if max_diff <= threshold:
                return None

            self.total_dyn_multi_bugs = getattr(self, 'total_dyn_multi_bugs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "dynamic_multi_shape_diff"
            bug.priority = 3
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = max_diff
            bug.violation_ratio = 1.0
            bug.trigger_config = "dynamic_multi_shape"
            bug.category = program.category
            bug.error_msg = (
                f"[DYN_MULTI_SHAPE] max_diff={max_diff:.3e} at shape={results.get('dyn_multi_worst_shape', '?')} "
                f"n_shapes={results.get('dyn_multi_n_shapes', 0)} template={program.name}")
            return bug
        except Exception:
            return None

    # =========================================================================
    # Aggressive Cross-Schedule Traversal
    # =========================================================================
    # Instead of randomly picking one pair of heuristics, systematically try
    # ALL heuristics against EAGER ground truth.  Any heuristic that produces
    # a different result from eager is a real bug (not just a scheduling diff).
    _V4_AGGR_SCHED_PROB = 0.05

    _REDUCTION_HEURISTICS = [
        'reduction', 'persistent_reduction', 'cooperative_reduction',
    ]

    def _run_aggressive_schedule_oracle(self, program):
        """Force each reduction heuristic on reduction kernels only and compare
        against eager.  Pointwise kernels keep their natural heuristic so that
        mixed programs (pointwise + reduction) don't crash spuriously."""
        body = program.body_code
        if not body or "_program_inputs" not in body:
            return None

        heuristics_str = repr(self._REDUCTION_HEURISTICS)
        driver = (
            "import torch\n"
            "import torch._dynamo\n"
            "import torch._inductor.config as inductor_config\n"
            + body + "\n"
            "\n"
            "if torch.cuda.is_available():\n"
            "    _program_inputs = tuple(\n"
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "\n"
            "_eager_inp = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "try:\n"
            "    _eager_out = program(*_eager_inp)\n"
            "except Exception:\n"
            "    _eager_out = None\n"
            "\n"
            "if _eager_out is not None and isinstance(_eager_out, torch.Tensor):\n"
            f"    for _heur in {heuristics_str}:\n"
            "        torch._dynamo.reset()\n"
            "        try:\n"
            "            from torch._inductor.codegen.triton import TritonKernel as _TK\n"
            "            _orig_h = _TK._get_heuristic\n"
            "            def _make_forced(h):\n"
            "                def _f(_self):\n"
            "                    if _self.inside_reduction:\n"
            "                        return h\n"
            "                    return _orig_h(_self)\n"
            "                return _f\n"
            "            _TK._get_heuristic = _make_forced(_heur)\n"
            "            _forced_fn = torch.compile(program, backend='inductor', fullgraph=False)\n"
            "            _forced_inp = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "            _forced_out = _forced_fn(*_forced_inp)\n"
            "            if torch.cuda.is_available(): torch.cuda.synchronize()\n"
            "            _TK._get_heuristic = _orig_h\n"
            "            if isinstance(_forced_out, torch.Tensor) and _forced_out.shape == _eager_out.shape:\n"
            "                _d = (_eager_out.float() - _forced_out.float()).abs().max().item()\n"
            "                if _d > results.get('aggr_sched_max_diff', 0.0):\n"
            "                    results['aggr_sched_max_diff'] = _d\n"
            "                    results['aggr_sched_worst_heur'] = _heur\n"
            "                results['aggr_sched_ok'] = True\n"
            "                results.setdefault('aggr_sched_tried', []).append(_heur)\n"
            "        except Exception as _e:\n"
            "            results.setdefault('aggr_sched_errors', []).append(f'{_heur}:{type(_e).__name__}')\n"
            "            try: _TK._get_heuristic = _orig_h\n"
            "            except: pass\n"
            "            continue\n"
            "        finally:\n"
            "            torch._dynamo.reset()\n"
        )

        try:
            self.total_aggr_sched_runs = getattr(self, 'total_aggr_sched_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('aggr_sched_ok'):
                return None

            max_diff = results.get('aggr_sched_max_diff', 0.0)
            dtype = program.dtype or "torch.float32"
            base_tol = GRAPH_DIFF_BOUNDS.get(dtype, 1e-4)
            threshold = base_tol * 10

            if max_diff <= threshold:
                return None

            worst_heur = results.get('aggr_sched_worst_heur', '?')
            tried = results.get('aggr_sched_tried', [])

            self.total_aggr_sched_bugs = getattr(self, 'total_aggr_sched_bugs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "aggressive_schedule_diff"
            bug.priority = 3
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = max_diff
            bug.violation_ratio = 1.0
            bug.trigger_config = f"forced_{worst_heur}_vs_eager"
            bug.category = program.category
            bug.error_msg = (
                f"[AGGR_SCHED] forced {worst_heur} vs eager: "
                f"max_diff={max_diff:.2e} tried={len(tried)} heuristics "
                f"template={program.name}")
            return bug
        except Exception:
            return None

    # Meta stride oracle — compare FakeTensor stride/shape against eager
    # reality for ops that have @register_meta. Catches bugs where
    # meta implementations return wrong stride for expanded tensors.
    _V4_META_STRIDE_PROB = 0.10  # 10% of clean iterations (cheap: no GPU/compile)

    # Ops with complex hand-written meta implementations that warrant stride testing.
    # Focus on ops that use clone_preserve_strides, as_strided, expand, or reshape
    # in their meta registration — these are the most likely to have stride bugs.
    _META_STRIDE_OPS = [
        ("slice_scatter", "torch.slice_scatter({x}, {src}, dim=1, start=0, end={e})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "src": f"torch.randn({b},{s//3},{h})", "e": s // 3}),
        ("select_scatter", "torch.select_scatter({x}, {src}, dim=1, index=0)",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "src": f"torch.randn({b},{h})"}),
        ("as_strided", "torch.as_strided({x}, [{b},{s}], [{s},1])",
         lambda b, s, h: {"x": f"torch.randn({b*s})"}),
        ("expand", "{x}.expand({b},{s},{h})",
         lambda b, s, h: {"x": f"torch.randn(1,1,{h})"}),
        ("repeat", "{x}.repeat(1,{s},1)",
         lambda b, s, h: {"x": f"torch.randn({b},1,{h})"}),
        ("unflatten", "{x}.unflatten(1, ({g}, {s_g}))",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "g": 2, "s_g": s // 2} if s >= 4 and s % 2 == 0 else None),
        ("diagonal_scatter", "torch.diagonal_scatter({x}, {src})",
         lambda b, s, h: {"x": f"torch.randn({s},{s})", "src": f"torch.randn({s})"}),
        ("view_as_real", "torch.view_as_real({x})",
         lambda b, s, h: {"x": f"torch.randn({b},{s}, dtype=torch.complex64)"}),
        ("scatter", "torch.scatter({x}, 1, {idx}, {src})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "idx": f"torch.zeros({b},1,{h}, dtype=torch.long)", "src": f"torch.randn({b},1,{h})"}),
        ("index_select", "torch.index_select({x}, 1, {idx})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "idx": f"torch.zeros({min(3,s)}, dtype=torch.long)"}),
        ("narrow", "{x}.narrow(1, 0, {n})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "n": max(1, s // 3)}),
        ("permute_view", "{x}.permute(0,2,1).contiguous().view({b},{h},{s})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})"}),
        ("chunk_first", "torch.chunk({x}, {n}, dim=1)[0]",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "n": min(3, s)} if s >= 3 else None),
        ("transpose_expand", "{x}.transpose(1,2).expand({b},{h},{s})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})"}),
        ("cat_mixed", "torch.cat([{x}, {x2}], dim=1)",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})", "x2": f"torch.randn({b},{s//2},{h})"}
         if s >= 4 else None),
        ("stack", "torch.stack([{x}, {x}], dim=0)",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})"}),
        ("flip", "torch.flip({x}, dims=[1])",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})"}),
        ("roll", "torch.roll({x}, shifts=2, dims=1)",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})"}),
        ("diagonal", "torch.diagonal({x}, dim1=0, dim2=1)",
         lambda b, s, h: {"x": f"torch.randn({s},{s},{h})"}),
        ("gather_idx", "torch.gather({x}, 1, {idx})",
         lambda b, s, h: {"x": f"torch.randn({b},{s},{h})",
                          "idx": f"torch.zeros({b},1,{h}, dtype=torch.long)"}),
        ("triu", "torch.triu({x})",
         lambda b, s, h: {"x": f"torch.randn({s},{s})"}),
    ]

    _META_STRIDE_LAYOUTS = [
        ("contiguous", "{tensor}"),
        ("permuted", "{tensor}.permute({perm}).contiguous().permute({inv_perm})"),
        ("expanded", "{tensor}.unsqueeze(0).expand(2, {expand_args})"),
        ("sliced", "{tensor}[..., ::2]"),
        ("channels_last", "{tensor}"),
    ]

    def _run_meta_stride_oracle(self, program):
        """Test meta registration stride correctness for a random op+layout combo.
        Compares FakeTensor metadata (shape, stride) vs eager real execution.
        Returns GraphBugResultV2 if stride mismatch found, else None."""
        import random as rng
        op_name, op_expr, param_fn = rng.choice(self._META_STRIDE_OPS)
        b, s, h = rng.choice([3, 7]), rng.choice([8, 16, 33, 65]), rng.choice([4, 16, 64])
        params = param_fn(b, s, h)
        if params is None:
            return None

        # Build the input tensors and the op call
        input_lines = []
        for k, v in params.items():
            if isinstance(v, str) and (v.startswith("torch.randn") or v.startswith("torch.zeros")):
                input_lines.append(f"{k} = {v}")
        expr = op_expr.format(**params)

        # Build layout variants for inputs
        layout_name = rng.choice(["contiguous", "expanded", "sliced", "permuted", "channels_last"])

        driver = (
            "import torch\n"
            "from torch._subclasses.fake_tensor import FakeTensorMode\n"
            "from torch._subclasses.fake_tensor import FakeTensor\n"
            "\n"
            + "\n".join(input_lines) + "\n"
            "\n"
            "# Apply layout mutation to first input\n"
        )
        first_input = list(params.keys())[0]
        if layout_name == "expanded" and "x" in params:
            driver += f"x = x.unsqueeze(0).expand(2, *x.shape)\n"
        elif layout_name == "sliced" and "x" in params:
            driver += f"if x.dim() >= 2 and x.shape[-1] > 2: x = x[..., ::2]\n"
        elif layout_name == "permuted" and "x" in params:
            driver += f"if x.dim() >= 2: x = x.transpose(-1, -2).contiguous().transpose(-1, -2)\n"
        elif layout_name == "channels_last" and "x" in params:
            driver += f"if x.dim() == 4: x = x.to(memory_format=torch.channels_last)\n"

        driver += (
            "\n"
            "# Eager execution — ground truth\n"
            f"try:\n"
            f"    eager_out = {expr}\n"
            f"    results['eager_shape'] = list(eager_out.shape)\n"
            f"    results['eager_stride'] = list(eager_out.stride())\n"
            f"    results['eager_offset'] = eager_out.storage_offset()\n"
            f"    results['eager_ok'] = True\n"
            f"except Exception as e:\n"
            f"    results['eager_ok'] = False\n"
            f"    results['eager_error'] = str(e)[:300]\n"
            "\n"
            "# Meta execution — FakeTensor mode\n"
            "if results.get('eager_ok'):\n"
            "    try:\n"
            "        with FakeTensorMode() as fake_mode:\n"
        )
        # Re-create inputs as FakeTensors from the ORIGINAL (pre-layout-mutation) tensor,
        # then apply the same layout mutation to the fake tensor.
        # Note: we create fake from the original, not the mutated tensor, because
        # from_tensor() on the already-mutated tensor would double-apply the mutation.
        for k, v in params.items():
            if isinstance(v, str) and (v.startswith("torch.randn") or v.startswith("torch.zeros")):
                driver += f"            _orig_{k} = {v}\n"
                driver += f"            fake_{k} = fake_mode.from_tensor(_orig_{k})\n"

        # Apply same layout mutation to fake tensors
        if layout_name == "expanded" and "x" in params:
            driver += f"            fake_x = fake_x.unsqueeze(0).expand(2, *fake_x.shape[1:])\n"
        elif layout_name == "sliced" and "x" in params:
            driver += f"            if fake_x.dim() >= 2 and fake_x.shape[-1] > 2: fake_x = fake_x[..., ::2]\n"
        elif layout_name == "permuted" and "x" in params:
            driver += f"            if fake_x.dim() >= 2: fake_x = fake_x.transpose(-1, -2).contiguous().transpose(-1, -2)\n"
        elif layout_name == "channels_last" and "x" in params:
            driver += f"            if fake_x.dim() == 4: fake_x = fake_x.to(memory_format=torch.channels_last)\n"

        import re as _re
        fake_expr = op_expr
        non_tensor_params = {}
        for k, v in params.items():
            if isinstance(v, str) and (v.startswith("torch.randn") or v.startswith("torch.zeros")):
                fake_expr = fake_expr.replace(f"{{{k}}}", f"fake_{k}")
                fake_expr = _re.sub(rf'\b{_re.escape(k)}\b', f'fake_{k}', fake_expr)
            else:
                non_tensor_params[k] = v
        if non_tensor_params:
            fake_expr = fake_expr.format(**non_tensor_params)
        driver += (
            f"            meta_out = {fake_expr}\n"
            f"            results['meta_shape'] = list(meta_out.shape)\n"
            f"            results['meta_stride'] = list(meta_out.stride())\n"
            f"            results['meta_offset'] = meta_out.storage_offset()\n"
            f"            results['meta_ok'] = True\n"
            f"    except Exception as e:\n"
            f"        results['meta_ok'] = False\n"
            f"        results['meta_error'] = str(e)[:300]\n"
            "\n"
            f"results['meta_op'] = '{op_name}'\n"
            f"results['meta_layout'] = '{layout_name}'\n"
        )

        try:
            self.total_meta_stride_runs = getattr(self, 'total_meta_stride_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('eager_ok') or not results.get('meta_ok'):
                return None

            eager_shape = results.get('eager_shape')
            meta_shape = results.get('meta_shape')
            eager_stride = results.get('eager_stride')
            meta_stride = results.get('meta_stride')

            shape_match = (eager_shape == meta_shape)
            stride_match = (eager_stride == meta_stride)

            if shape_match and stride_match:
                return None

            self.total_meta_stride_diffs = getattr(self, 'total_meta_stride_diffs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "meta_stride_diff"
            bug.priority = 3
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = 0.0
            bug.violation_ratio = 1.0
            mismatch = []
            if not shape_match:
                mismatch.append(f"shape:eager={eager_shape} meta={meta_shape}")
            if not stride_match:
                mismatch.append(f"stride:eager={eager_stride} meta={meta_stride}")
            bug.trigger_config = f"meta_{op_name}_{layout_name}"
            bug.category = program.category if program else "meta"
            bug.error_msg = (
                f"[META_STRIDE] {op_name} layout={layout_name}: "
                + " | ".join(mismatch))
            return bug
        except Exception:
            return None

    # =========================================================================
    # Guard Oracle — test if Dynamo guards are too loose
    # =========================================================================
    _V4_GUARD_ORACLE_PROB = 0.08

    def _run_guard_oracle(self, program):
        """Test Dynamo guard correctness: compile with input A, call with
        mutated input B (dtype/contiguity/requires_grad change). If guard
        is too loose, compiled code is reused incorrectly → silent wrong result."""
        body = program.body_code
        if not body or "_program_inputs" not in body:
            return None

        mutation_axes = [
            ("dtype_f64", "x = x.to(torch.float64)"),
            ("contiguity", "x = x.permute(*reversed(range(x.dim()))).contiguous() if x.dim() >= 2 else x"),
            ("requires_grad", "x = x.detach().requires_grad_(not x.requires_grad)"),
            ("unsqueeze_rank", "x = x.unsqueeze(0)"),
            ("squeeze_rank", "x = x.squeeze(0) if x.shape[0] == 1 else x"),
            ("stride_expand", "x = x.unsqueeze(-1).expand(*x.shape, 2).contiguous()[..., 0] if x.is_floating_point() else x"),
            ("non_contiguous_slice", "x = torch.cat([x, x], dim=-1)[..., ::2] if x.dim() >= 1 else x"),
            ("shape_plus1", "x = torch.nn.functional.pad(x, (0, 1)) if x.dim() >= 1 and x.is_floating_point() else x"),
            ("channels_last", "x = x.to(memory_format=torch.channels_last) if x.dim() == 4 else x"),
        ]
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            mutation_axes.append(("dtype_f16", "x = x.to(torch.float16)"))
            mutation_axes.append(("dtype_bf16", "x = x.to(torch.bfloat16)"))
        import random as rng
        axis_name, mutate_code = rng.choice(mutation_axes)

        driver = (
            "import torch\n"
            "import torch._dynamo\n"
            "torch._dynamo.reset()\n"
            "\n"
            + body + "\n"
            "\n"
            "# Phase A: compile with original inputs\n"
            "_fn = program\n"
            "_original_inputs = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _program_inputs)\n"
            "try:\n"
            "    _compiled = torch.compile(_fn, backend='inductor')\n"
            "    _out_a = _compiled(*_original_inputs)\n"
            "    results['phase_a_ok'] = True\n"
            "except Exception as _e:\n"
            "    results['phase_a_ok'] = False\n"
            "    results['phase_a_error'] = str(_e)[:300]\n"
            "\n"
            "# Phase B: mutate inputs, call compiled (may reuse) vs eager\n"
            "if results.get('phase_a_ok'):\n"
            "    try:\n"
            "        _mutated_inputs = []\n"
            "        for x in _program_inputs:\n"
            "            if isinstance(x, torch.Tensor):\n"
            f"                {mutate_code}\n"
            "                _mutated_inputs.append(x)\n"
            "            else:\n"
            "                _mutated_inputs.append(x)\n"
            "        _mutated_inputs = tuple(_mutated_inputs)\n"
            "        # Eager ground truth\n"
            "        _eager_b = _fn(*tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _mutated_inputs))\n"
            "        # Compiled with mutated inputs — guard should trigger recompile\n"
            "        _compiled_b = _compiled(*tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _mutated_inputs))\n"
            "        if isinstance(_eager_b, torch.Tensor) and isinstance(_compiled_b, torch.Tensor):\n"
            "            if _eager_b.shape == _compiled_b.shape and _eager_b.dtype == _compiled_b.dtype:\n"
            "                _diff = (_eager_b.float() - _compiled_b.float()).abs().max().item()\n"
            "                results['guard_diff'] = _diff\n"
            "                results['guard_eager_sum'] = _eager_b.float().sum().item()\n"
            "                results['guard_compiled_sum'] = _compiled_b.float().sum().item()\n"
            "            else:\n"
            "                results['guard_shape_mismatch'] = f'{_eager_b.shape}vs{_compiled_b.shape}'\n"
            "        results['guard_ok'] = True\n"
            "    except Exception as _e:\n"
            "        results['guard_ok'] = False\n"
            "        results['guard_error'] = str(_e)[:300]\n"
            "\n"
            f"results['guard_axis'] = '{axis_name}'\n"
            "torch._dynamo.reset()\n"
        )

        try:
            self.total_guard_oracle_runs = getattr(self, 'total_guard_oracle_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('phase_a_ok') or not results.get('guard_ok'):
                return None

            guard_diff = results.get('guard_diff')
            if guard_diff is None:
                return None

            dtype = program.dtype or "torch.float32"
            base_tol = GRAPH_DIFF_BOUNDS.get(dtype, 1e-4)
            threshold = base_tol * 100

            if guard_diff <= threshold:
                return None

            self.total_guard_diffs = getattr(self, 'total_guard_diffs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "guard_diff"
            bug.priority = 3
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = guard_diff
            bug.violation_ratio = 1.0
            bug.trigger_config = f"guard_{axis_name}"
            bug.category = program.category if program else "guard"
            bug.error_msg = (
                f"[GUARD_TOO_LOOSE] axis={axis_name}: "
                f"max_diff={guard_diff:.3e} threshold={threshold:.3e} "
                f"eager_sum={results.get('guard_eager_sum', '?')} "
                f"compiled_sum={results.get('guard_compiled_sum', '?')}")
            return bug
        except Exception:
            return None

    # =========================================================================
    # Finite-Difference Gradient Check
    # =========================================================================
    _V4_FINITE_DIFF_PROB = 0.03
    _V4_FINITE_DIFF_EPS = 1e-5
    _V4_FINITE_DIFF_RTOL = 1e-3
    _V4_FINITE_DIFF_MAX_ELEMS = 50

    _FD_NONDIFF_OPS = re.compile(
        r'torch\.sort\s*\(|torch\.argsort\s*\(|torch\.topk\s*\(|'
        r'torch\.randint\s*\(|torch\.rand\s*\(|torch\.randn\s*\(|'
        r'torch\.randperm\s*\(|torch\.bernoulli\s*\(|'
        r'\.half\s*\(\s*\)|\.to\s*\(\s*torch\.float16\s*\)|\.to\s*\(\s*torch\.bfloat16\s*\)|'
        r'torch\.scatter\s*\(|\.scatter_\s*\(|\.scatter_add\s*\(|'
        r'torch\.unique\s*\(|torch\.nonzero\s*\('
    )

    _FD_DEGENERATE_INPUTS = re.compile(
        r'torch\.zeros\s*\(|torch\.ones\s*\(|torch\.full\s*\(|'
        r'\*\s*1e[3-9]\b|\*\s*1e[1-9]\d'
    )

    def _run_finite_diff_gradient_oracle(self, program):
        """Compare autograd.grad vs finite-difference approximation.
        Catches bugs where BOTH eager and compiled produce wrong gradients."""
        body = program.body_code
        if not body or "_program_inputs" not in body:
            return None

        if self._FD_NONDIFF_OPS.search(body):
            return None
        if self._FD_DEGENERATE_INPUTS.search(body):
            return None

        eps = self._V4_FINITE_DIFF_EPS
        max_elems = self._V4_FINITE_DIFF_MAX_ELEMS
        rtol = self._V4_FINITE_DIFF_RTOL

        driver = (
            "import torch\n"
            "import random as _rng\n"
            "\n"
            + body + "\n"
            "\n"
            "# Move to fp64 and enable grad\n"
            "_inputs_fd = []\n"
            "for _t in _program_inputs:\n"
            "    if isinstance(_t, torch.Tensor) and _t.is_floating_point():\n"
            "        _inputs_fd.append(_t.detach().double().requires_grad_(True))\n"
            "    else:\n"
            "        _inputs_fd.append(_t)\n"
            "\n"
            "try:\n"
            "    _out = program(*_inputs_fd)\n"
            "    if isinstance(_out, torch.Tensor):\n"
            "        _scalar = _out.sum()\n"
            "    else:\n"
            "        results['fd_ok'] = False\n"
            "        results['fd_error'] = 'non-tensor output'\n"
            "        _scalar = None\n"
            "except Exception as _e:\n"
            "    results['fd_ok'] = False\n"
            "    results['fd_error'] = str(_e)[:300]\n"
            "    _scalar = None\n"
            "\n"
            "if _scalar is not None and _scalar.requires_grad:\n"
            "    # Autograd gradient\n"
            "    _grad_inputs = [_t for _t in _inputs_fd if isinstance(_t, torch.Tensor) and _t.requires_grad]\n"
            "    if _grad_inputs:\n"
            "        try:\n"
            "            _ag_grads = torch.autograd.grad(_scalar, _grad_inputs, create_graph=False)\n"
            "        except Exception as _e:\n"
            "            _ag_grads = None\n"
            "            results['fd_ok'] = False\n"
            "            results['fd_error'] = f'autograd failed: {str(_e)[:200]}'\n"
            "        if _ag_grads is not None:\n"
            "            _max_rel_err = 0.0\n"
            "            _worst_idx = -1\n"
            "            _n_checked = 0\n"
            f"            _eps = {eps}\n"
            f"            _max_elems = {max_elems}\n"
            "            for _gi, (_inp, _ag) in enumerate(zip(_grad_inputs, _ag_grads)):\n"
            "                _flat_inp = _inp.detach().view(-1)\n"
            "                _flat_ag = _ag.detach().view(-1)\n"
            "                _n = _flat_inp.numel()\n"
            "                _indices = _rng.sample(range(_n), min(_max_elems, _n))\n"
            "                for _idx in _indices:\n"
            "                    if abs(_flat_ag[_idx].item()) > 1e10:\n"
            "                        continue  # ill-conditioned, skip\n"
            "                    _xp = _inp.detach().clone().view(-1)\n"
            "                    _xp[_idx] += _eps\n"
            "                    _xm = _inp.detach().clone().view(-1)\n"
            "                    _xm[_idx] -= _eps\n"
            "                    # Rebuild inputs with perturbed tensor\n"
            "                    def _rebuild(_inputs_fd, _gi, _new_flat, _shape):\n"
            "                        _out = []\n"
            "                        _j = 0\n"
            "                        for _t in _inputs_fd:\n"
            "                            if isinstance(_t, torch.Tensor) and _t.requires_grad:\n"
            "                                if _j == _gi:\n"
            "                                    _out.append(_new_flat.view(_shape))\n"
            "                                else:\n"
            "                                    _out.append(_t.detach())\n"
            "                                _j += 1\n"
            "                            else:\n"
            "                                _out.append(_t)\n"
            "                        return _out\n"
            "                    try:\n"
            "                        _fp = program(*_rebuild(_inputs_fd, _gi, _xp, _inp.shape)).sum().item()\n"
            "                        _fm = program(*_rebuild(_inputs_fd, _gi, _xm, _inp.shape)).sum().item()\n"
            "                    except Exception:\n"
            "                        continue\n"
            "                    if not (torch.isfinite(torch.tensor(_fp)) and torch.isfinite(torch.tensor(_fm))):\n"
            "                        continue\n"
            "                    _fd_val = (_fp - _fm) / (2 * _eps)\n"
            "                    _ag_val = _flat_ag[_idx].item()\n"
            "                    _denom = max(abs(_ag_val), _eps)\n"
            "                    _rel = abs(_fd_val - _ag_val) / _denom\n"
            "                    if _rel > _max_rel_err:\n"
            "                        _max_rel_err = _rel\n"
            "                        _worst_idx = _idx\n"
            "                    _n_checked += 1\n"
            "            results['fd_max_rel_err'] = _max_rel_err\n"
            "            results['fd_worst_idx'] = _worst_idx\n"
            "            results['fd_n_checked'] = _n_checked\n"
            "            results['fd_ok'] = True\n"
            "    else:\n"
            "        results['fd_ok'] = False\n"
            "        results['fd_error'] = 'no grad inputs'\n"
            "elif _scalar is not None:\n"
            "    results['fd_ok'] = False\n"
            "    results['fd_error'] = 'no requires_grad'\n"
        )

        try:
            self.total_finite_diff_runs = getattr(self, 'total_finite_diff_runs', 0) + 1
            results, outer_error, _ = TorchLibrary.run_code(driver)
            if outer_error or not isinstance(results, dict):
                return None
            if not results.get('fd_ok'):
                return None
            if results.get('fd_n_checked', 0) < 5:
                return None

            max_rel_err = results.get('fd_max_rel_err', 0.0)
            if max_rel_err <= rtol:
                return None

            self.total_finite_diff_bugs = getattr(self, 'total_finite_diff_bugs', 0) + 1
            bug = GraphBugResultV2()
            bug.is_bug = True
            bug.bug_type = "finite_diff_grad"
            bug.priority = 3
            bug.divergence_level = DivergenceLevel.CATASTROPHIC
            bug.max_diff = max_rel_err
            bug.violation_ratio = 1.0
            bug.trigger_config = f"finite_diff_{program.name}"
            bug.category = program.category if program else "finite_diff"
            bug.error_msg = (
                f"[FINITE_DIFF] autograd vs FD max_rel_err={max_rel_err:.3e} "
                f"at elem_idx={results.get('fd_worst_idx', '?')} "
                f"n_checked={results.get('fd_n_checked', 0)} "
                f"template={program.name}")
            return bug
        except Exception:
            return None

    def _known_bug_match(self, program):
        """Return tag of known-bug family matched, or None."""
        body = program.body_code or ""
        for tag, pred in self._KNOWN_BUG_PATTERNS:
            try:
                if pred(body):
                    return tag
            except Exception:
                continue
        return None

    def _handle_bug(self, program, bug, source, kernels):
        """Bug handling: known-bug filter + fp64 backoff."""
        # Known-bug filter: drop bugs that match a known-bug fingerprint
        # straight to weak-signal/ so they don't crowd out new findings.
        if (bug.divergence_level == DivergenceLevel.CATASTROPHIC
                and bug.bug_type in ("inductor_diff", "compile_diff", "frontend_diff")):
            kb_tag = self._known_bug_match(program)
            if kb_tag is not None:
                bug.divergence_level = DivergenceLevel.SUSPICIOUS
                bug.priority = 1
                bug.error_msg = (
                    f"[KNOWN_BUG:{kb_tag}] " + (bug.error_msg or ""))
                self.total_known_bug_filtered += 1
                # No need to do fp64 backoff for known bugs
                return super()._handle_bug(program, bug, source, kernels)

        # FP64 backoff: only on numeric-diff bugs at CATASTROPHIC level
        if (bug.divergence_level == DivergenceLevel.CATASTROPHIC
                and bug.bug_type in ("inductor_diff", "compile_diff", "frontend_diff")
                and not program.has_poison
                and program.is_deterministic
                and bug.max_diff is not None):
            fp64_diff = self._fp64_backoff_check(program, bug.trigger_config or "default")
            if fp64_diff is not None:
                if fp64_diff < self._V4_FP64_BACKOFF_THRESHOLD:
                    # fp32 diff was reduction noise — downgrade to weak signal
                    bug.divergence_level = DivergenceLevel.SUSPICIOUS
                    bug.priority = 1
                    bug.error_msg = (
                        f"[FP64_BACKOFF noise|fp64_diff={fp64_diff:.2e}] "
                        + (bug.error_msg or ""))
                    self.total_fp64_backoff_downgraded += 1
                    print(f"  [FP64-Backoff] DOWNGRADE [{bug.category}:{bug.bug_type}] "
                          f"fp32_diff={bug.max_diff:.2e} fp64_diff={fp64_diff:.2e} "
                          f"prog={program.signature}")
                else:
                    # fp64 diff is also large — confirmed real bug
                    self.total_fp64_backoff_confirmed += 1
                    bug.error_msg = (
                        f"[FP64_BACKOFF confirmed|fp64_diff={fp64_diff:.2e}] "
                        + (bug.error_msg or ""))
                    print(f"  [FP64-Backoff] CONFIRMED [{bug.category}:{bug.bug_type}] "
                          f"fp32_diff={bug.max_diff:.2e} fp64_diff={fp64_diff:.2e} "
                          f"prog={program.signature}")

        # Delegate to base handler for dedup, basin tracking, write
        return super()._handle_bug(program, bug, source, kernels)

    # Mapping: kernel family → template names that target it
    _KSS_FAMILY_TO_TEMPLATES = {
        'pointwise':             ['d1_softmax_fused', 'd5_dtype_cast_chain',
                                  'real_stochastic_depth', 'kt_long_fusion_chain',
                                  'kt_complex_fft_chain',
                                  'idx_bool_mask_reduce', 'idx_expand_inplace_pattern'],
        'reduction':             ['kt_split_reduction', 'd1_reduce_cascade',
                                  'kt_reduction_axis_sweep', 'kt_searchsorted_bucketize',
                                  'idx_gather_noncontiguous', 'idx_scatter_broadcast'],
        'persistent_reduction':  ['d1_layernorm_manual', 'real_rmsnorm_swiglu'],
        'cooperative_reduction': ['kt_cooperative_reduction', 'kt_cooperative_fused'],
        'split_scan':            ['real_scan_gate'],  # S≥129 triggers split_scan
        'scan':                  ['kt_scan_single_block'],
        'extern_gemm':           ['kt_extern_gemm_standalone', 'real_cross_attention',
                                  'kt_sparse_coo_dense'],
        'extern_conv':           ['kt_extern_conv_standalone', 'real_conv_bn_relu_pool'],
    }

    def _generate_program(self, iteration, inject_poison):
        """
        Program generation with adversarial shapes and semantic mutation.
        """
        self.diversity_scheduler.step()

        # Novelty-guided seed injection: consume high-value static candidates
        # Apply mutations and configs to externally-selected programs
        # the old long-running entrypoint keeps exploring around those seeds.
        if (self._novelty_seed_pool
                and random.random() < self.novelty_ratio):
            prog = self._generate_novelty_program(inject_poison=inject_poison)
            if prog is not None:
                self.diversity_scheduler.record(prog.category, prog.name)
                return prog, "novelty_guided"

        # Coverage-feedback target sampling (the closed loop): with probability
        # KSS_GUIDANCE_RATE, steer generation toward a compiler decision state
        # that the running campaign has under-covered. Coverage is the same
        # kernel-state observation that defines a target hit, accumulated in
        # self.kernel_ss by record_from_kernels() after each compilation; the
        # sampler reads it back here via get_uncovered_families/get_uncovered_state.
        # Gated off when guidance_mode == "random" (ablation baseline), which
        # turns this into unguided template sampling.
        if (self.guidance_mode != "random"
                and random.random() < KSS_GUIDANCE_RATE
                and hasattr(self, 'kernel_ss')):
            uncovered = self.kernel_ss.get_uncovered_families()
            if not uncovered:
                uncovered_fam = self.kernel_ss.get_lowest_coverage_family()
            else:
                uncovered_fam = random.choice(uncovered)
            if uncovered_fam and uncovered_fam in self._KSS_FAMILY_TO_TEMPLATES:
                candidates = self._KSS_FAMILY_TO_TEMPLATES[uncovered_fam]
                valid = [c for c in candidates if c in _TPL_MAP]
                if valid:
                    name = random.choice(valid)
                    # Heuristic-directed shapes — construct shapes
                    # that actually trigger the target kernel family,
                    # instead of random sampling.
                    batch, seq, hidden = self._kss_directed_shapes(uncovered_fam)
                    # Pick the specific dtype that covers an uncovered state
                    # within this family (more targeted than random).
                    uncov_state = self.kernel_ss.get_uncovered_state(uncovered_fam)
                    if uncov_state is not None:
                        target_dtype = uncov_state[0]
                        dtype = f"torch.{target_dtype}" if f"torch.{target_dtype}" in _DTYPES else random.choice(_DTYPES)
                    else:
                        dtype = random.choice(_DTYPES)
                    try:
                        body_code, is_det = _TPL_MAP[name](batch, seq, hidden, dtype)
                        body_code = self._v4_maybe_semantic_mutate(body_code)
                        body_code = self._v4_maybe_non_contiguous(body_code)
                        # Input value diversification
                        body_code, _ = _apply_value_diversification(body_code, dtype)
                        # force backward-stress mutation on KSS-guided templates.
                        # bug_012 was triggered only when stress mutation added
                        # chunk+cat and narrow+pad on top of nan_to_num — composition
                        # is the main path to new bug discovery.
                        if (name not in _NO_BACKWARD_TEMPLATES):
                            mutated, changed = _apply_backward_stress_mutation(body_code)
                            if changed:
                                body_code = mutated
                                self.total_bwd_stress_mutations += 1
                        config_preset = _sample_config_preset()
                        # KSS-guided gets dynamic=True 50% of the time (vs
                        # normal 15%). Inductor's symbolic-shape codegen is a
                        # separate bug surface (bug_009 lived here).
                        use_dynamic = (random.random() < 0.50)
                        prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                            config_preset=config_preset,
                                            use_dynamic=use_dynamic,
                                            supports_backward=(name not in _NO_BACKWARD_TEMPLATES))
                        self.total_kss_guided += 1
                        if self.total_kss_guided % 10 == 0:
                            print(f"  [KSS-guided] #{self.total_kss_guided}: family={uncovered_fam} "
                                  f"template={name} dyn={use_dynamic}")
                        return prog, f"kss_guided_{uncovered_fam}"
                    except Exception:
                        pass  # fall through to normal generation

        # 20% of the time: use TemplateCompositor for structural jumps
        if random.random() < self.composition_ratio:
            batch, seq, hidden = self._v4_sample_shapes()
            dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
            name, body_code, is_det = TemplateCompositor.compose(batch, seq, hidden, dtype)
            self.total_composed += 1

            body_code = self._v4_maybe_semantic_mutate(body_code)
            body_code = self._v4_maybe_non_contiguous(body_code)

            # Input value diversification
            body_code, val_strategy = _apply_value_diversification(body_code, dtype)
            if val_strategy:
                self.total_value_diverse = getattr(self, 'total_value_diverse', 0) + 1

            poisoned = False
            if inject_poison:
                new_code = GraphPoisonInjector.inject(body_code, dtype)
                poisoned = (new_code != body_code)
                body_code = new_code

            config_preset = _sample_config_preset()
            use_dynamic = (random.random() < 0.30)
            prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                has_poison=poisoned, config_preset=config_preset,
                                use_dynamic=use_dynamic, supports_backward=True)
            return prog, "composed"

        # Check for forced category
        forced_cat = self.diversity_scheduler.get_forced_category()
        if forced_cat and forced_cat in _TPL_CATEGORIES and _TPL_CATEGORIES[forced_cat]:
            name = random.choice(_TPL_CATEGORIES[forced_cat])
            batch, seq, hidden = self._v4_sample_shapes()
            dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
            try:
                body_code, is_det = _TPL_MAP[name](batch, seq, hidden, dtype)
            except Exception as e:
                print(f"  [DeepOpFuzz] forced template {name} failed: {type(e).__name__}: {e}")
                body_code, is_det = _tpl_d2_linear_bias_act(batch, seq, hidden, dtype)
                name = "d2_linear_bias_act"

            body_code = self._v4_maybe_semantic_mutate(body_code)
            body_code = self._v4_maybe_non_contiguous(body_code)

            # Input value diversification
            body_code, val_strategy = _apply_value_diversification(body_code, dtype)
            if val_strategy:
                self.total_value_diverse = getattr(self, 'total_value_diverse', 0) + 1

            poisoned = False
            if inject_poison:
                new_code = GraphPoisonInjector.inject(body_code, dtype)
                poisoned = (new_code != body_code)
                body_code = new_code
            config_preset = _sample_config_preset()
            use_dynamic = (random.random() < 0.30)
            prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                has_poison=poisoned, config_preset=config_preset,
                                use_dynamic=use_dynamic,
                                supports_backward=(name not in _NO_BACKWARD_TEMPLATES))
            self.diversity_scheduler.record(forced_cat, name)
            return prog, f"forced_{forced_cat}"

        # Normal generation with adaptive weights + enrichment cluster boost
        adaptive_weights = self.diversity_scheduler.get_adaptive_weights()
        # Apply enrichment cluster boost: templates in enriched clusters get higher weight
        boosted_weights = [w * ENRICHED_CLUSTER_BOOST.get(n, 1.0)
                           for w, (n, _, _) in zip(adaptive_weights, PROGRAM_TEMPLATES)]
        name, _, _ = random.choices(
            [(n, f, w) for n, f, w in PROGRAM_TEMPLATES],
            weights=boosted_weights, k=1)[0]
        batch, seq, hidden = self._v4_sample_shapes()
        dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
        try:
            body_code, is_det = _TPL_MAP[name](batch, seq, hidden, dtype)
        except Exception as e:
            print(f"  [DeepOpFuzz] adaptive template {name} failed: {type(e).__name__}: {e}")
            body_code, is_det = _tpl_d2_linear_bias_act(batch, seq, hidden, dtype)
            name = "d2_linear_bias_act"

        body_code = self._v4_maybe_semantic_mutate(body_code)
        body_code = self._v4_maybe_non_contiguous(body_code)

        # Input value diversification
        body_code, val_strategy = _apply_value_diversification(body_code, dtype)
        if val_strategy:
            self.total_value_diverse = getattr(self, 'total_value_diverse', 0) + 1

        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code
        config_preset = _sample_config_preset()
        use_dynamic = (random.random() < 0.30)
        if name.startswith("d38_"):
            use_dynamic = True
        result_prog = GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                                   has_poison=poisoned, config_preset=config_preset,
                                   use_dynamic=use_dynamic,
                                   supports_backward=(name not in _NO_BACKWARD_TEMPLATES))
        self.diversity_scheduler.record(result_prog.category, name)
        return result_prog, "adaptive"

    def run_fuzzing_loop(self, max_iterations=5000, result_callback=None,
                         checkpoint_interval=100):
        warmup_threshold = min(50, int(max_iterations * self.warmup_ratio))

        print(f"\n{'='*70}")
        print(f"[GraphFuzzerV4] Targeted Exploration Fuzzer "
              f"(guidance_mode={self.guidance_mode})")
        print(f"  strategy={self.strategy_name} max_iter={max_iterations}")
        print(f"  Templates: {len(PROGRAM_TEMPLATES)} + TemplateCompositor")
        print(f"  [DeepOpFuzz] Adversarial shapes: {self._V4_ADV_RATIO:.0%} "
              f"(batch={_ADV_BATCH}, seq={_ADV_SEQ[:4]}..., hidden={_ADV_HIDDEN[:4]}...)")
        print(f"  [DeepOpFuzz] Semantic mutation: {self._V4_SEMANTIC_PROB:.0%} "
              f"(.clone, .contiguous, +0, *1, reshape, dtype cast)")
        print(f"  [DeepOpFuzz] Layout stages: layout_permute_contig, layout_slice_stride")
        print(f"  [DeepOpFuzz] Config: +no_cudagraph preset")
        print(f"  [DeepOpFuzz] FP64 backoff: enabled "
              f"(downgrade if fp64_diff < {self._V4_FP64_BACKOFF_THRESHOLD:.0e})")
        print(f"  [DeepOpFuzz] Coverage: kernel hash")
        print(f"  [Enrichment] Enrichment cluster boost: "
              f"{len(ENRICHED_CLUSTER_BOOST)} templates in 3 clusters (A/B/C)")
        if self._novelty_seed_pool:
            print(f"  [NoveltyGuided] seeds={len(self._novelty_seed_pool)} "
                  f"ratio={self.novelty_ratio:.0%} source={self.novelty_batch_path}")
        print(f"  Backward stress mutation: {self._V4_BWD_STRESS_PROB:.0%} "
              f"(view/stride/scatter/expand patterns, 100% backward oracle)")
        print(f"  Composition ratio: {self.composition_ratio:.0%}")
        print(f"  Basin similarity threshold: {self.basin_tracker.threshold}")
        print(f"  Category rotation interval: {self.diversity_scheduler.rotation_interval}")
        print(f"  Oracle: graded (CATASTROPHIC/SUSPICIOUS/SUCCESS)")
        print(f"  Dedup: structural fingerprint + traditional")
        print(f"  Warmup: {warmup_threshold}  Base e: {self.mutation_ctrl.base_epsilon}")
        print(f"{'='*70}\n")

        # Corpus persistence: try to restore from previous run
        self._corpus_checkpoint_path = join(self._graph_dir, "corpus_checkpoint.json")
        restored = self.corpus.load_checkpoint(self._corpus_checkpoint_path)
        if restored > 0:
            print(f"  [DeepOpFuzz] Resumed corpus with {restored} entries")

        start = time.time()

        for i in range(max_iterations):
            if i < warmup_threshold:
                # Warmup also uses adversarial shapes
                program, source = self._generate_program(i, inject_poison=False)
                source = "warmup"
                self.mutation_ctrl.record_warmup()
            else:
                if i == warmup_threshold:
                    print(f"\n  [DeepOpFuzz] warmup -> evolution (iter {i})\n")
                    if self.corpus.size() == 0:
                        for _ in range(5):
                            seed, _ = self._generate_program(i, inject_poison=False)
                            self.corpus.add_seed(seed, 1, iteration=i)

                if self.mutation_ctrl.should_explore():
                    program, source = self._generate_program(i, inject_poison=True)
                elif self.corpus.size() > 0 and random.random() < 0.60:
                    parent = self.corpus.select_parent(current_iteration=i)
                    program = GraphMutator.mutate_v2(
                        parent, self.mutation_ctrl.get_mutation_range(), inject_poison=True)
                    # Apply semantic mutation to corpus-mutated programs
                    if random.random() < self._V4_SEMANTIC_PROB:
                        mutated, changed = _apply_semantic_mutation(program.body_code)
                        if changed:
                            program.body_code = mutated
                            self.total_semantic_mutations += 1
                    source = "corpus"
                    self.diversity_scheduler.record(program.category, program.name)
                else:
                    program, source = self._generate_program(i, inject_poison=True)

                if source == "corpus" and self.basin_tracker.is_in_known_basin(program):
                    self.total_basin_suppressed += 1
                    program, source = self._generate_program(i, inject_poison=True)
                    source = "basin_escape"

            # Backward stress mutation — inject view/stride/scatter patterns
            # that create complex backward graphs for inductor to handle
            _bwd_stressed = False
            if (random.random() < self._V4_BWD_STRESS_PROB
                    and program.supports_backward
                    and not program.has_poison
                    and program.name not in _NO_BACKWARD_TEMPLATES):
                mutated, changed = _apply_backward_stress_mutation(program.body_code)
                if changed:
                    program.body_code = mutated
                    self.total_bwd_stress_mutations += 1
                    _bwd_stressed = True

            # Disable AOTI on this program if AOTI previously caused CUDA crash
            if self._aoti_disabled:
                program.skip_aoti = True

            try:
                results, outer_error, kernels = self._execute(program)
            except torch.cuda.OutOfMemoryError:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"  [GraphFuzzerV4] iter {i}: CUDA OOM, cleared cache")
                continue
            except RuntimeError as e:
                err_str = str(e)
                is_cuda_err = ("CUDA" in err_str or "cuda" in err_str
                               or "illegal memory" in err_str.lower()
                               or "device-side assert" in err_str.lower())
                if is_cuda_err:
                    print(f"  [GraphFuzzerV4] iter {i}: CUDA RuntimeError: {err_str[:200]}")
                    _save_crash_report(self._graph_dir, program,
                                       "cuda_runtime_error", err_str, i, source)
                    self.total_cuda_crashes += 1
                    # Try to reset CUDA state
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Verify CUDA is actually usable with a real operation
                    try:
                        torch.zeros(1, device='cuda')
                    except Exception:
                        print(f"  [GraphFuzzerV4] CUDA unrecoverable at iter {i}, "
                              f"saving checkpoint and exiting for restart.")
                        self.corpus.save_checkpoint(self._corpus_checkpoint_path)
                        self.kernel_ss.save_checkpoint(self._kss_checkpoint_path)
                        self.backward_kss.save_checkpoint(self._bwd_kss_checkpoint_path)
                        self._save_bug_traces()
                        raise RuntimeError(f"CUDA_UNRECOVERABLE at iter {i}")
                    continue
                else:
                    print(f"  [GraphFuzzerV4] iter {i}: RuntimeError: {err_str[:200]}")
                    _save_crash_report(self._graph_dir, program,
                                       "runtime_error", err_str, i, source)
                    continue
            except Exception as e:
                print(f"  [GraphFuzzerV4] iter {i}: execute failed: {type(e).__name__}: {e}")
                _save_crash_report(self._graph_dir, program,
                                   type(e).__name__, str(e), i, source)
                continue

            # AOTI subprocess: run in child process to avoid CUDA context corruption.
            if (isinstance(results, dict)
                    and results.get('eager_ok')
                    and not self._aoti_disabled
                    and random.random() < self._V4_AOTI_PROB):
                try:
                    _aoti_res = self._run_aoti_subprocess(program)
                    if _aoti_res and isinstance(results, dict):
                        results['aoti_ok'] = _aoti_res.get('aoti_ok', False)
                        results['aoti_error'] = _aoti_res.get('aoti_error', '')
                        results['aoti_export_failed'] = _aoti_res.get('aoti_export_failed', False)
                        if _aoti_res.get('aoti_diff') is not None:
                            results['aoti_result'] = _aoti_res['aoti_diff']
                except Exception as _aoti_e:
                    pass

            # Extract kernel heuristic types from results
            _kheurs = results.get('_kernel_heuristics') if isinstance(results, dict) else None

            new_count, (should_expand, _) = self.coverage.update(
                kernels, i, config_preset=program.config_preset,
                kernel_heuristics=_kheurs)
            if should_expand:
                self.mutation_ctrl.enter_expansion_mode()
            if new_count > 0:
                if self.mutation_ctrl.expansion_mode:
                    self.mutation_ctrl.exit_expansion_mode()
                self.corpus.add_seed(program, new_count, iteration=i)

            # Kernel state space coverage (FreeFuzz-style)
            # detect fp16/bf16 kernel dtypes from template name.
            # dtype_cast templates produce fp16/bf16 Triton kernels even
            # when the program dtype is fp32/fp64. We record additional
            # hits so KSS counts these correctly.
            _prog_dtype = program.dtype
            _kss_new = self.kernel_ss.record_from_kernels(
                kernels, _prog_dtype, heuristics=_kheurs)
            _cast_dtypes = []
            _pname = program.name if hasattr(program, 'name') else ''
            if 'half' in _pname or 'fp16' in _pname:
                _cast_dtypes.append('float16')
            if 'bf16' in _pname or 'bfloat' in _pname:
                _cast_dtypes.append('bfloat16')
            if '.half()' in program.body_code or 'float16' in program.body_code:
                _cast_dtypes.append('float16')
            if '.bfloat16()' in program.body_code or 'bfloat16' in program.body_code:
                _cast_dtypes.append('bfloat16')
            for _cd in set(_cast_dtypes):
                _kss_new += self.kernel_ss.record_from_kernels(
                    kernels, _cd, heuristics=_kheurs)
            if _kss_new > 0 and i % 50 == 0:
                print(f"  [KernelSS] +{_kss_new} new states, "
                      f"coverage={self.kernel_ss.get_coverage()*100:.1f}% "
                      f"({self.kernel_ss.get_total_hits()}/{self.kernel_ss.get_total_valid()})")

            # Per-iteration coverage timeseries (for evaluation figures)
            _cov_summary = self.coverage.get_coverage_summary()
            self._coverage_timeseries.append({
                "iter": i,
                "l2": _cov_summary["L2_fusion_patterns"],
                "l3": _cov_summary["L3_kernel_hashes"],
                "kss": self.kernel_ss.get_total_hits(),
                "bugs": self.total_bugs,
                "source": source,
            })

            bug = analyze_graph_results_v2(results, program, outer_error)
            if bug.is_bug:
                _verified, _vreason = _verify_bug_in_fresh_process(bug, program)
                if not _verified:
                    print(f"  [verify] FILTERED {bug.bug_type} reason={_vreason[:120]}")
                    bug.is_bug = False
            if bug.is_bug:
                self._handle_bug(program, bug, source, kernels)
            else:
                # Primary backward oracle: forward was clean — sample some
                # programs and run backward to catch grad-only bugs.
                # Backward-stressed programs always get backward oracle (100%)
                _bwd_prob = 1.0 if _bwd_stressed else self._V4_PRIMARY_BWD_PROB
                # boost backward prob if this (template, dtype) not yet tested
                _dt = program.dtype.replace('torch.', '') if program.dtype else 'float32'
                if f"{program.name}|{_dt}" not in self._bwd_tested_pairs:
                    _bwd_prob = 1.0
                if (random.random() < _bwd_prob
                        and program.supports_backward
                        and program.is_deterministic
                        and not program.has_poison):
                    bwd_bug = self._run_primary_backward_oracle(program)
                    if bwd_bug is not None:
                        self._handle_bug(program, bwd_bug, source, kernels)
                        # If backward triggered, skip cfg-cfg this iter
                        bug = bwd_bug
                # semantic invariant oracle — 2% of iter runs an independent
                # probe program (softmax sums to 1, cumsum[-1] == sum, etc).
                # Catches bugs where eager and inductor both produce a result
                # but inductor violates a mathematical invariant.
                if not bug.is_bug and random.random() < 0.02:
                    inv_bug = self._run_semantic_invariant_oracle()
                    if inv_bug is not None:
                        self._handle_bug(program, inv_bug, "invariant_probe", kernels)
                        bug = inv_bug
                # double-backward oracle (grad-of-grad) — 5% of the time.
                # Tests torch.autograd.grad(create_graph=True) path, a separate
                # bug surface from single backward.
                if (not bug.is_bug
                        and random.random() < 0.15
                        and program.supports_backward
                        and program.is_deterministic
                        and not program.has_poison):
                    dbwd_bug = self._run_double_backward_oracle(program)
                    if dbwd_bug is not None:
                        self._handle_bug(program, dbwd_bug, source, kernels)
                        bug = dbwd_bug
                # Secondary cfg-vs-cfg oracle:
                if (not bug.is_bug
                        and random.random() < self._V4_CFG_CFG_PROB):
                    cfg_bug = self._run_cfg_cfg_oracle(results, program)
                    if cfg_bug is not None:
                        self._handle_bug(program, cfg_bug, source, kernels)
                # Config differential oracle: recompile with two different configs
                if (not bug.is_bug
                        and random.random() < self._V4_CONFIG_DIFF_PROB
                        and program.is_deterministic
                        and not program.has_poison):
                    cd_bug = self._run_config_diff_oracle(program)
                    if cd_bug is not None:
                        self._handle_bug(program, cd_bug, source, kernels)
                # Pattern matcher oracle: dedicated toggle, higher prob
                if (not bug.is_bug
                        and random.random() < self._V4_PATTERN_MATCHER_PROB
                        and program.is_deterministic
                        and not program.has_poison):
                    pm_bug = self._run_pattern_matcher_oracle(program)
                    if pm_bug is not None:
                        self._handle_bug(program, pm_bug, source, kernels)
                # Cross-schedule oracle: force different heuristics on same graph
                if (not bug.is_bug
                        and random.random() < self._V4_CROSS_SCHED_PROB
                        and program.is_deterministic
                        and not program.has_poison):
                    cs_bug = self._run_cross_schedule_oracle(program)
                    if cs_bug is not None:
                        self._handle_bug(program, cs_bug, source, kernels)
                # Cross-schedule backward oracle: force different heuristics on backward
                if (not bug.is_bug
                        and random.random() < self._V4_CROSS_SCHED_BWD_PROB
                        and program.supports_backward
                        and program.is_deterministic
                        and not program.has_poison
                        and program.name not in _NO_BACKWARD_TEMPLATES):
                    cs_bwd_bug = self._run_cross_schedule_backward_oracle(program)
                    if cs_bwd_bug is not None:
                        self._handle_bug(program, cs_bwd_bug, source, kernels)
                # Compiled autograd oracle — compile backward pass itself
                if (not bug.is_bug
                        and random.random() < self._V4_COMPILED_AUTOGRAD_PROB
                        and program.supports_backward
                        and program.is_deterministic
                        and not program.has_poison
                        and program.name not in _NO_BACKWARD_TEMPLATES):
                    try:
                        ca_bug = self._run_compiled_autograd_oracle(program)
                        if ca_bug is not None:
                            self._handle_bug(program, ca_bug, source, kernels)
                    except Exception as _ca_e:
                        if i < 10 or i % 100 == 0:
                            print(f"  [COMPILED_AUTOGRAD] oracle error: {_ca_e}")
                # Meta stride oracle — test @register_meta stride correctness
                if (not bug.is_bug
                        and random.random() < self._V4_META_STRIDE_PROB):
                    try:
                        meta_bug = self._run_meta_stride_oracle(program)
                        if meta_bug is not None:
                            self._handle_bug(program, meta_bug, source, kernels)
                    except Exception as _meta_e:
                        if i < 10 or i % 100 == 0:
                            print(f"  [META_STRIDE] oracle error: {_meta_e}")
                # Guard oracle — test Dynamo guard looseness
                if (not bug.is_bug
                        and random.random() < self._V4_GUARD_ORACLE_PROB
                        and program.is_deterministic
                        and not program.has_poison):
                    try:
                        guard_bug = self._run_guard_oracle(program)
                        if guard_bug is not None:
                            self._handle_bug(program, guard_bug, source, kernels)
                    except Exception as _guard_e:
                        if i < 10 or i % 100 == 0:
                            print(f"  [GUARD] oracle error: {_guard_e}")
                # Finite-diff gradient oracle — catch wrong autograd formulas
                if (not bug.is_bug
                        and random.random() < self._V4_FINITE_DIFF_PROB
                        and program.supports_backward
                        and program.is_deterministic
                        and not program.has_poison
                        and program.name not in _NO_BACKWARD_TEMPLATES):
                    try:
                        fd_bug = self._run_finite_diff_gradient_oracle(program)
                        if fd_bug is not None:
                            self._handle_bug(program, fd_bug, source, kernels)
                    except Exception as _fd_e:
                        if i < 10 or i % 100 == 0:
                            print(f"  [FINITE_DIFF] oracle error: {_fd_e}")
                # Dynamic shape multi-call oracle
                if (not bug.is_bug
                        and random.random() < self._V4_DYN_MULTI_PROB
                        and program.is_deterministic
                        and not program.has_poison):
                    try:
                        dyn_bug = self._run_dynamic_shape_multi_oracle(program)
                        if dyn_bug is not None:
                            self._handle_bug(program, dyn_bug, source, kernels)
                    except Exception as _dyn_e:
                        if i < 10 or i % 100 == 0:
                            print(f"  [DYN_MULTI] oracle error: {_dyn_e}")
                # Aggressive cross-schedule traversal
                if (not bug.is_bug
                        and random.random() < self._V4_AGGR_SCHED_PROB
                        and program.is_deterministic
                        and not program.has_poison):
                    try:
                        aggr_bug = self._run_aggressive_schedule_oracle(program)
                        if aggr_bug is not None:
                            self._handle_bug(program, aggr_bug, source, kernels)
                    except Exception as _aggr_e:
                        if i < 10 or i % 100 == 0:
                            print(f"  [AGGR_SCHED] oracle error: {_aggr_e}")

            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        # CUDA is in bad state — verify with a real operation
                        try:
                            torch.zeros(1, device='cuda')
                        except Exception:
                            print(f"  [GraphFuzzerV4] CUDA unrecoverable at iter {i}, "
                                  f"saving checkpoint and exiting for restart.")
                            self.corpus.save_checkpoint(self._corpus_checkpoint_path)
                            self.kernel_ss.save_checkpoint(self._kss_checkpoint_path)
                            self.backward_kss.save_checkpoint(self._bwd_kss_checkpoint_path)
                            self._save_bug_traces()
                            raise RuntimeError(f"CUDA_UNRECOVERABLE at iter {i}") from e

            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start
                cat_cov = self.corpus.category_coverage()
                print(f"\n  [{i+1:5d}/{max_iterations}] "
                      f"programs={self.generator.generated_count} "
                      f"kernels={self.coverage.get_total()} "
                      f"bugs={self.total_bugs} (novel={self.total_novel_bugs} "
                      f"CAT={self.bugs_by_divergence.get(DivergenceLevel.CATASTROPHIC,0)} "
                      f"SUS={self.total_suspicious} BWD_AMP={self.total_backward_amplified} "
                      f"BWD_ONLY={self.total_backward_only_bugs} "
                      f"CFGCFG={self.total_cfg_cfg_diffs} "
                      f"XSCHED={self.total_cross_sched_diffs} "
                      f"XSCHED_BWD={getattr(self, 'total_cross_sched_bwd_diffs', 0)}) "
                      f"corpus={self.corpus.size()} cats={cat_cov} "
                      f"composed={self.total_composed} "
                      f"novelty={self.total_novelty_guided} "
                      f"basin_suppressed={self.total_basin_suppressed} "
                      f"adv_shapes={self.total_adversarial_shapes} "
                      f"sem_mut={self.total_semantic_mutations} "
                      f"bwd_stress={self.total_bwd_stress_mutations} "
                      f"crashes={self.total_cuda_crashes} "
                      f"primary_bwd_runs={self.total_primary_bwd_runs} "
                      f"cfg_cfg_runs={self.total_cfg_cfg_runs} "
                      f"cross_sched_runs={self.total_cross_sched_runs} "
                      f"GUARD={self.total_guard_diffs}/{self.total_guard_oracle_runs} "
                      f"FD={self.total_finite_diff_bugs}/{self.total_finite_diff_runs} "
                      f"DYN_MULTI={getattr(self,'total_dyn_multi_bugs',0)}/{getattr(self,'total_dyn_multi_runs',0)} "
                      f"AGGR_SCHED={getattr(self,'total_aggr_sched_bugs',0)}/{getattr(self,'total_aggr_sched_runs',0)} "
                      f"speed={(i+1)/max(elapsed,1e-9):.1f}/s "
                      f"elapsed={elapsed/60:.1f}m")
                print(f"  {self.mutation_ctrl.get_status()}")
                print(f"  diversity: {self.diversity_scheduler.get_status()}")
                print(f"  basins: {self.basin_tracker.get_status()}")
                cov = self.coverage.get_coverage_summary()
                print(f"  coverage: L1={cov['L1_coverage']} families={cov['L1_families']} "
                      f"L2={cov['L2_fusion_patterns']} fusion_patterns "
                      f"L3={cov['L3_kernel_hashes']} kernel_hashes")
                kss = self.kernel_ss.get_summary()
                print(f"  kernel_ss: {kss['total_hits']}/{kss['total_valid']} "
                      f"({kss['coverage']*100:.1f}%)")
                bwd_kss = self.backward_kss.get_summary()
                _bwd_op_hits = len(self._bwd_tested_pairs & self._bwd_valid_pairs)
                _bwd_op_total = len(self._bwd_valid_pairs)
                print(f"  bwd_kss: {bwd_kss['total_hits']}/{bwd_kss['total_valid']} "
                      f"({bwd_kss['coverage']*100:.1f}%)  "
                      f"bwd_op_cov: {_bwd_op_hits}/{_bwd_op_total} "
                      f"({_bwd_op_hits/max(_bwd_op_total,1)*100:.1f}%)")
                print(f"  by_type={dict(self.bugs_by_type)}")
                print(f"  by_category={dict(self.bugs_by_category)}\n")
                if result_callback:
                    result_callback(self._get_result())
                # Save corpus checkpoint periodically
                if (i + 1) % 500 == 0:
                    self.corpus.save_checkpoint(self._corpus_checkpoint_path)
                    self.kernel_ss.save_checkpoint(self._kss_checkpoint_path)
                    self.backward_kss.save_checkpoint(self._bwd_kss_checkpoint_path)

            self.iterations_done = i + 1

        # Final corpus checkpoint
        self.corpus.save_checkpoint(self._corpus_checkpoint_path)
        self.kernel_ss.save_checkpoint(self._kss_checkpoint_path)
        self.backward_kss.save_checkpoint(self._bwd_kss_checkpoint_path)
        self._save_bug_traces()
        elapsed = time.time() - start
        print(f"\n{'='*70}")
        print(f"[GraphFuzzerV4] done: programs={self.generator.generated_count} "
              f"kernels={self.coverage.get_total()} "
              f"bugs={self.total_bugs} (novel={self.total_novel_bugs}) "
              f"composed={self.total_composed} novelty={self.total_novelty_guided} "
              f"basin_suppressed={self.total_basin_suppressed} "
              f"adv_shapes={self.total_adversarial_shapes} "
              f"sem_mut={self.total_semantic_mutations} "
              f"crashes={self.total_cuda_crashes} "
              f"elapsed={elapsed/60:.1f}m")
        print(f"  by_type={dict(self.bugs_by_type)}")
        print(f"  by_category={dict(self.bugs_by_category)}")
        print(f"  basins: {self.basin_tracker.get_status()}")
        print(f"  diversity: {self.diversity_scheduler.get_status()}")
        print(f"{'='*70}\n")
        return self._get_result()

    def _get_result(self):
        result = super()._get_result()
        result["phase"] = "GraphV4"
        result["total_adversarial_shapes"] = self.total_adversarial_shapes
        result["total_semantic_mutations"] = self.total_semantic_mutations
        result["total_cuda_crashes"] = self.total_cuda_crashes
        result["total_fp64_backoff_downgraded"] = self.total_fp64_backoff_downgraded
        result["total_fp64_backoff_confirmed"] = self.total_fp64_backoff_confirmed
        result["total_known_bug_filtered"] = self.total_known_bug_filtered
        result["total_backward_only_bugs"] = self.total_backward_only_bugs
        result["total_primary_bwd_runs"] = self.total_primary_bwd_runs
        result["total_cfg_cfg_runs"] = self.total_cfg_cfg_runs
        result["total_cfg_cfg_diffs"] = self.total_cfg_cfg_diffs
        result["total_cross_sched_runs"] = self.total_cross_sched_runs
        result["total_cross_sched_diffs"] = self.total_cross_sched_diffs
        result["total_novelty_guided"] = self.total_novelty_guided
        result["novelty_batch_path"] = self.novelty_batch_path
        result["novelty_ratio"] = self.novelty_ratio
        result["iterations_done"] = self.iterations_done
        result["guidance_mode"] = self.guidance_mode
        result["coverage_timeseries"] = self._coverage_timeseries
        result["kss_hit_frequency"] = self.kernel_ss.get_hit_frequency()
        result["kss_summary"] = self.kernel_ss.get_summary()
        return result


# =============================================================================
# Process-Isolated Runner (GraphFuzzerV4)
# =============================================================================

def _run_graph_worker_v4(output_dir, max_iterations, result_file, config_path,
                         novelty_batch_path=None, novelty_ratio=0.35):
    result = {
        "phase": "GraphV4", "success": False, "error": None,
        "total_programs": 0, "total_kernels": 0, "total_bugs": 0,
        "total_novel_bugs": 0, "total_composed": 0, "total_basin_suppressed": 0,
        "total_adversarial_shapes": 0, "total_semantic_mutations": 0,
        "total_cuda_crashes": 0,
        "bugs_by_type": {}, "bugs_by_category": {}, "bugs_by_divergence": {},
        "unique_bugs": 0, "coverage_history": [], "corpus_size": 0,
        "mutation_status": "",
        "bugs": {"crash": 0, "compile": 0, "backward": 0, "cuda": 0,
                 "precision": 0, "total": 0, "novel": 0},
    }
    if config_path and os.path.exists(config_path):
        try:
            import configparser
            from classes.database import TorchDatabase
            cfg = configparser.ConfigParser(); cfg.read(config_path)
            TorchDatabase.database_config(
                cfg["mongodb"]["host"], int(cfg["mongodb"]["port"]),
                cfg["mongodb"]["torch_database"])
        except Exception as e:
            print(f"  [GraphFuzzerV4] DB config failed: {e}")

    def _save():
        try:
            tmp = result_file + ".tmp"
            with open(tmp, "w") as f: json.dump(result, f, indent=2)
            os.replace(tmp, result_file)
        except Exception as e:
            print(f"  [GraphFuzzerV4] save failed: {e}")

    try:
        fuzzer = GraphFuzzerV4(
            output_dir=output_dir,
            strategy_name="GraphV4",
            novelty_batch_path=novelty_batch_path,
            novelty_ratio=novelty_ratio,
        )
        r = fuzzer.run_fuzzing_loop(
            max_iterations=max_iterations,
            result_callback=lambda rr: (result.update(rr), _save()))
        result.update(r); result["success"] = True
    except RuntimeError as e:
        err_msg = str(e)
        result["error"] = err_msg
        result["iterations_done"] = getattr(fuzzer, 'iterations_done', 0)
        print(f"\n[GraphFuzzerV4] worker failed: {err_msg[:200]}")
        _save()
        if "CUDA_UNRECOVERABLE" in err_msg:
            # Exit with code 2 to signal parent to restart
            import sys
            sys.exit(2)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result["iterations_done"] = getattr(fuzzer, 'iterations_done', 0)
        print(f"\n[GraphFuzzerV4] worker failed: {e}")
    _save()


def run_graph_phase_isolated_v4(output_dir, max_iterations=5000,
                                 timeout_minutes=60, config_path=None,
                                 novelty_batch_path=None, novelty_ratio=0.35):
    os.makedirs(output_dir, exist_ok=True)
    result_file = join(output_dir, "graph_v4_result.json")
    print(f"\n{'='*70}\nLAUNCHING ISOLATED PHASE: GraphV4  "
          f"max_iter={max_iterations:,}  timeout={timeout_minutes}min\n{'='*70}\n")
    if novelty_batch_path:
        print(f"  [NoveltyGuided] batch={novelty_batch_path} ratio={novelty_ratio:.0%}")

    t0 = time.time()
    remaining_iter = max_iterations
    total_restarts = 0
    max_restarts = 10  # prevent infinite restart loops

    while remaining_iter > 0 and total_restarts <= max_restarts:
        elapsed_so_far = time.time() - t0
        remaining_minutes = timeout_minutes - elapsed_so_far / 60
        if remaining_minutes < 1:
            print(f"  [Isolated] No time left, stopping.")
            break

        if total_restarts > 0:
            print(f"\n{'='*70}")
            print(f"  [Isolated] RESTARTING worker (attempt {total_restarts}/{max_restarts})")
            print(f"  [Isolated] remaining_iter={remaining_iter}, "
                  f"remaining_time={remaining_minutes:.1f}min")
            print(f"  [Isolated] Corpus checkpoint will be auto-loaded by worker")
            print(f"{'='*70}\n")

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_run_graph_worker_v4,
                        args=(output_dir, remaining_iter, result_file, config_path,
                              novelty_batch_path, novelty_ratio))
        p.start()
        p.join(timeout=remaining_minutes * 60)

        if p.is_alive():
            p.terminate(); p.join(timeout=10)
            if p.is_alive(): p.kill(); p.join()
            break  # timeout — done

        # Worker exited. Check if it was a CUDA crash (non-zero exit)
        # and if so, read how many iterations it completed.
        if p.exitcode != 0:
            iters_done = 0
            error_msg = ''
            saved = {}
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        saved = json.load(f)
                    iters_done = saved.get('iterations_done', 0)
                    error_msg = saved.get('error', '') or ''
                except Exception as _e:
                    print(f"  [Isolated] Failed to read result file: {_e}")

            # Treat ANY non-zero exit as possibly-recoverable CUDA issue.
            # The worker uses exit(2) for CUDA_UNRECOVERABLE, but other
            # unhandled segfaults (C10 aborts, cudagraph failures) also
            # exit non-zero without writing the result file — those ARE
            # exactly the cases we want to auto-restart.
            is_cuda_related = (
                'CUDA' in error_msg or 'cuda' in error_msg
                or p.exitcode in (2, -6, -11, 134, 139)  # SIGABRT/SIGSEGV + exit(2)
                or not error_msg  # no result file written → likely C-level abort
            )
            if is_cuda_related and total_restarts < max_restarts:
                print(f"  [Isolated] Worker exited (code={p.exitcode}) after "
                      f"{iters_done} iters, err={error_msg[:120]!r} — "
                      f"restarting with fresh CUDA context "
                      f"({total_restarts + 1}/{max_restarts})")
                remaining_iter -= max(iters_done, 1)
                total_restarts += 1
                continue
            else:
                print(f"  [Isolated] Non-recoverable worker exit "
                      f"(code={p.exitcode}, restarts={total_restarts}) — stopping")
                break
        else:
            # Normal exit
            break

    elapsed = time.time() - t0

    result = {"phase": "GraphV4", "success": False, "error": None,
              "elapsed_seconds": elapsed,
              "total_programs": 0, "total_bugs": 0, "total_novel_bugs": 0,
              "total_restarts": total_restarts,
              "bugs": {"crash": 0, "compile": 0, "backward": 0,
                       "cuda": 0, "precision": 0, "total": 0, "novel": 0}}

    if os.path.exists(result_file):
        try:
            with open(result_file) as f: saved = json.load(f)
            result.update(saved); result["elapsed_seconds"] = elapsed
            result["total_restarts"] = total_restarts
        except Exception as e:
            print(f"  Failed to read result file: {e}")

    print(f"\nGraphV4 phase {'COMPLETED' if result.get('success') else 'FAILED'}  "
          f"programs={result.get('total_programs',0)}  "
          f"kernels={result.get('total_kernels',0)}  "
          f"bugs={result.get('total_bugs',0)}  "
          f"novel={result.get('total_novel_bugs',0)}  "
          f"composed={result.get('total_composed',0)}  "
          f"basin_suppressed={result.get('total_basin_suppressed',0)}  "
          f"adv_shapes={result.get('total_adversarial_shapes',0)}  "
          f"sem_mut={result.get('total_semantic_mutations',0)}  "
          f"crashes={result.get('total_cuda_crashes',0)}  "
          f"elapsed={elapsed/60:.1f}m")
    return result


# =============================================================================
# Process-Isolated Runner (Random Baseline)
# =============================================================================

def _run_graph_worker_v4_random(output_dir, max_iterations, result_file, config_path):
    result = {
        "phase": "GraphV4Random", "success": False, "error": None,
        "total_programs": 0, "total_kernels": 0, "total_bugs": 0,
        "total_novel_bugs": 0, "total_composed": 0, "total_basin_suppressed": 0,
        "total_adversarial_shapes": 0, "total_semantic_mutations": 0,
        "total_cuda_crashes": 0,
        "bugs_by_type": {}, "bugs_by_category": {}, "bugs_by_divergence": {},
        "unique_bugs": 0, "coverage_history": [], "corpus_size": 0,
        "mutation_status": "",
        "bugs": {"crash": 0, "compile": 0, "backward": 0, "cuda": 0,
                 "precision": 0, "total": 0, "novel": 0},
    }
    if config_path and os.path.exists(config_path):
        try:
            import configparser
            from classes.database import TorchDatabase
            cfg = configparser.ConfigParser(); cfg.read(config_path)
            TorchDatabase.database_config(
                cfg["mongodb"]["host"], int(cfg["mongodb"]["port"]),
                cfg["mongodb"]["torch_database"])
        except Exception as e:
            print(f"  [GraphV4Random] DB config failed: {e}")

    def _save():
        try:
            tmp = result_file + ".tmp"
            with open(tmp, "w") as f: json.dump(result, f, indent=2)
            os.replace(tmp, result_file)
        except Exception as e:
            print(f"  [GraphV4Random] save failed: {e}")

    try:
        fuzzer = GraphFuzzerV4(
            output_dir=output_dir, strategy_name="GraphV4Random",
            guidance_mode="random")
        r = fuzzer.run_fuzzing_loop(
            max_iterations=max_iterations,
            result_callback=lambda rr: (result.update(rr), _save()))
        result.update(r); result["success"] = True
    except RuntimeError as e:
        err_msg = str(e)
        result["error"] = err_msg
        result["iterations_done"] = getattr(fuzzer, 'iterations_done', 0)
        print(f"\n[GraphV4Random] worker failed: {err_msg[:200]}")
        _save()
        if "CUDA_UNRECOVERABLE" in err_msg:
            import sys
            sys.exit(2)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result["iterations_done"] = getattr(fuzzer, 'iterations_done', 0)
        print(f"\n[GraphV4Random] worker failed: {e}")
    _save()


def run_graph_phase_isolated_v4_random(output_dir, max_iterations=5000,
                                        timeout_minutes=60, config_path=None):
    os.makedirs(output_dir, exist_ok=True)
    result_file = join(output_dir, "graph_v4_random_result.json")
    print(f"\n{'='*70}\nLAUNCHING ISOLATED PHASE: GraphV4 Random Baseline  "
          f"max_iter={max_iterations:,}  timeout={timeout_minutes}min\n{'='*70}\n")

    t0 = time.time()
    remaining_iter = max_iterations
    total_restarts = 0
    max_restarts = 10

    while remaining_iter > 0 and total_restarts <= max_restarts:
        elapsed_so_far = time.time() - t0
        remaining_minutes = timeout_minutes - elapsed_so_far / 60
        if remaining_minutes < 1:
            break

        if total_restarts > 0:
            print(f"\n  [Isolated] RESTARTING random worker "
                  f"(attempt {total_restarts}/{max_restarts})")

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_run_graph_worker_v4_random,
                        args=(output_dir, remaining_iter, result_file, config_path))
        p.start()
        p.join(timeout=remaining_minutes * 60)

        if p.is_alive():
            p.terminate(); p.join(timeout=10)
            if p.is_alive(): p.kill(); p.join()
            break

        if p.exitcode != 0:
            iters_done = 0
            error_msg = ''
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        saved = json.load(f)
                    iters_done = saved.get('iterations_done', 0)
                    error_msg = saved.get('error', '') or ''
                except Exception:
                    pass

            is_cuda_related = (
                'CUDA' in error_msg or 'cuda' in error_msg
                or p.exitcode in (2, -6, -11, 134, 139)
                or not error_msg
            )
            if is_cuda_related and total_restarts < max_restarts:
                remaining_iter -= max(iters_done, 1)
                total_restarts += 1
                continue
            else:
                break
        else:
            break

    elapsed = time.time() - t0
    result = {"phase": "GraphV4Random", "success": False, "error": None,
              "elapsed_seconds": elapsed,
              "total_programs": 0, "total_bugs": 0, "total_novel_bugs": 0,
              "total_restarts": total_restarts,
              "bugs": {"crash": 0, "compile": 0, "backward": 0,
                       "cuda": 0, "precision": 0, "total": 0, "novel": 0}}

    if os.path.exists(result_file):
        try:
            with open(result_file) as f: saved = json.load(f)
            result.update(saved); result["elapsed_seconds"] = elapsed
            result["total_restarts"] = total_restarts
        except Exception as e:
            print(f"  Failed to read result file: {e}")

    print(f"\nGraphV4Random phase {'COMPLETED' if result.get('success') else 'FAILED'}  "
          f"programs={result.get('total_programs',0)}  "
          f"kernels={result.get('total_kernels',0)}  "
          f"bugs={result.get('total_bugs',0)}  "
          f"elapsed={elapsed/60:.1f}m")
    return result


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GraphFuzzer: academically-grounded graph-level PyTorch fuzzing")
    parser.add_argument("--output", "-o", default="graph_output")
    parser.add_argument("--max-iterations", "-n", type=int, default=5000)
    parser.add_argument("--timeout", "-t", type=int, default=60)
    parser.add_argument("--conf", default=None)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--v3", action="store_true",
                        help="Use base fuzzer with bug basin tracking, "
                             "diversity forcing, template composition")
    parser.add_argument("--v4", action="store_true",
                        help="Use extended fuzzer with targeted exploration: adversarial shapes, "
                             "semantic mutation, layout disruption, no_cudagraph config")
    args = parser.parse_args()

    config_path = None
    if args.conf:
        for p in [args.conf, join("config", args.conf), join("..", "config", args.conf)]:
            if os.path.exists(p):
                config_path = os.path.abspath(p); break

    if args.v4:
        if args.isolated:
            result = run_graph_phase_isolated_v4(
                args.output, args.max_iterations, args.timeout, config_path)
        else:
            fuzzer = GraphFuzzerV4(output_dir=args.output)
            result = fuzzer.run_fuzzing_loop(max_iterations=args.max_iterations)
    elif args.v3:
        if args.isolated:
            result = run_graph_phase_isolated_v3(
                args.output, args.max_iterations, args.timeout, config_path)
        else:
            fuzzer = GraphFuzzerV3(output_dir=args.output)
            result = fuzzer.run_fuzzing_loop(max_iterations=args.max_iterations)
    else:
        # Default to full fuzzer
        if args.isolated:
            result = run_graph_phase_isolated_v4(
                args.output, args.max_iterations, args.timeout, config_path)
        else:
            fuzzer = GraphFuzzerV4(output_dir=args.output)
            result = fuzzer.run_fuzzing_loop(max_iterations=args.max_iterations)

    print("\nResult:")
    for k, v in result.items():
        if k != "coverage_history": print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

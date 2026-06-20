#!/usr/bin/env python3
"""
Program representation, generation, mutation, and corpus management.
"""

import hashlib
import os
import random
import re
import time
from collections import defaultdict
from os.path import join

import torch

from templates.registry import (
    GraphPoisonInjector, PROGRAM_TEMPLATES, _TPL_CATEGORIES, _TPL_WEIGHTS,
    _TPL_MAP, _tpl_d2_linear_bias_act,
)
from kernel.coverage import _DTYPES, _DTYPE_WEIGHTS
from bug_result import _build_config_injection_code, _sample_config_preset

# =============================================================================
# GraphProgram
# =============================================================================

class GraphProgram:
    _BODY_NONDETERMINISTIC_RE = re.compile(
        r'torch\.randint\s*\(|torch\.rand\s*\(|torch\.randn\s*\(|'
        r'torch\.randperm\s*\(|torch\.bernoulli\s*\(|'
        r'torch\.multinomial\s*\(|torch\.normal\s*\(|'
        r'torch\.poisson\s*\(|random\.random\s*\(')

    def __init__(self, name, body_code, is_deterministic, dtype, batch, seq, hidden,
                 has_poison=False, config_preset="default", use_dynamic=False,
                 supports_backward=True):
        self.name=name; self.body_code=body_code
        self.dtype=dtype; self.batch=batch; self.seq=seq; self.hidden=hidden
        self.has_poison=has_poison
        self.config_preset = config_preset
        self.use_dynamic = use_dynamic
        self.supports_backward = supports_backward
        self.skip_aoti = False  # set True to skip AOTI block in eval code
        self.category = name.split("_")[0].upper() if "_" in name else "UNK"
        if is_deterministic and body_code:
            func_body = body_code.split('def program(')[1] if 'def program(' in body_code else ''
            if self._BODY_NONDETERMINISTIC_RE.search(func_body):
                is_deterministic = False
        self.is_deterministic = is_deterministic

    @property
    def signature(self):
        cfg = f"|cfg={self.config_preset}" if self.config_preset != "default" else ""
        dyn = "|dyn" if self.use_dynamic else ""
        return f"{self.name}|{self.dtype}|b{self.batch}s{self.seq}h{self.hidden}{cfg}{dyn}"

    def build_eval_code_v2(self):
        """Build evaluation code with config preset injection, dynamic shapes, and
        multiple inductor configuration variants."""
        # Config injection lines
        config_lines = _build_config_injection_code(self.config_preset)

        # Build inductor config list (base configs + preset variant + optional dynamic)
        inductor_configs = [
            "('default',        {'backend':'inductor'})",
            "('reduce_overhead', {'backend':'inductor','mode':'reduce-overhead'})",
            "('fullgraph',      {'backend':'inductor','fullgraph':True})",
        ]
        if self.use_dynamic:
            inductor_configs.append(
                "('dynamic',        {'backend':'inductor','dynamic':True})")

        lines = [
            "import torch", "import random", "",
            "# -- Kernel heuristic tracing (lightweight, in-process) --",
            "_kernel_heuristics = []",
            "try:",
            "    from torch._inductor.codegen.triton import TritonKernel as _TK",
            "    _orig_gh = _TK._get_heuristic",
            "    def _traced_gh(_self):",
            "        _h = _orig_gh(_self)",
            "        _kernel_heuristics.append(_h)",
            "        return _h",
            "    _TK._get_heuristic = _traced_gh",
            "    # Also patch SplitScan kernel if available",
            "    try:",
            "        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel as _TSK",
            "        _orig_gh_ss = _TSK._get_heuristic",
            "        def _traced_gh_ss(_self):",
            "            _h = _orig_gh_ss(_self)",
            "            _kernel_heuristics.append(_h)",
            "            return _h",
            "        _TSK._get_heuristic = _traced_gh_ss",
            "    except Exception: pass",
            "    # Capture 'template' family -- TritonTemplateKernel for max_autotune GEMM",
            "    try:",
            "        from torch._inductor.select_algorithm import TritonTemplateKernel as _TTK",
            "        _orig_ttk_init = _TTK.__init__",
            "        def _traced_ttk_init(_self, *args, **kwargs):",
            "            _kernel_heuristics.append('template')",
            "            return _orig_ttk_init(_self, *args, **kwargs)",
            "        _TTK.__init__ = _traced_ttk_init",
            "    except Exception: pass",
            "    # Capture 'foreach' family -- TritonComboKernel.select_combo_heuristics",
            "    try:",
            "        from torch._inductor.codegen.triton_combo_kernel import TritonComboKernel as _TCK",
            "        _orig_sch = _TCK.select_combo_heuristics",
            "        def _traced_sch(_self, *args, **kwargs):",
            "            _r = _orig_sch(_self, *args, **kwargs)",
            "            if _r and len(_r) >= 1:",
            "                _kernel_heuristics.append(_r[0])",
            "            return _r",
            "        _TCK.select_combo_heuristics = _traced_sch",
            "    except Exception: pass",
            "except Exception: pass",
            "",
        ]
        # Inject inductor config before any compilation
        if config_lines:
            lines.extend(config_lines)
            lines.append("")

        lines.extend([
            self.body_code.strip(), "",

            "# -- Move inputs to CUDA so Inductor generates Triton kernels --",
            "if torch.cuda.is_available():",
            "    _program_inputs = tuple(",
            "        t.cuda() if isinstance(t, torch.Tensor) else t for t in _program_inputs)",
            "",

            "# -- 1. Eager (baseline, no inductor config affects this) --",
            "try:",
            "    results['eager'] = program(*_program_inputs)",
            "    results['eager_ok'] = True",
            "except Exception as _e:",
            "    results['eager_error'] = str(_e)",
            "    results['eager_ok'] = False",
            "",

            "# -- 2. aot_eager (frontend only, no kernel fusion) --",
            "if results.get('eager_ok'):",
            "    try:",
            "        torch._dynamo.reset()",
            "        _aot_fn = torch.compile(program, backend='aot_eager', fullgraph=False)",
            "        _aot_inp = tuple(t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs)",
            "        results['aot_eager'] = _aot_fn(*_aot_inp)",
            "        results['aot_eager_ok'] = True",
            "    except Exception as _e:",
            "        results['aot_eager_error'] = str(_e)",
            "        results['aot_eager_ok'] = False",
            "    finally:",
            "        torch._dynamo.reset()",
            "",

            f"# -- 3. Inductor (configs + preset={self.config_preset}"
            f"{' + dynamic' if self.use_dynamic else ''}) --",
            "if results.get('eager_ok'):",
            "    results['inductor_results'] = {}",
            "    for _lbl, _kw in [",
        ])
        for cfg in inductor_configs:
            lines.append(f"        {cfg},")
        lines.extend([
            "    ]:",
            "        try:",
            "            torch._dynamo.reset()",
            "            _ifn = torch.compile(program, **_kw)",
            "            _inp = tuple(t.clone().detach() if isinstance(t, torch.Tensor) else t for t in _program_inputs)",
            "            results['inductor_results'][_lbl] = {'result': _ifn(*_inp), 'ok': True}",
            "        except Exception as _e:",
            "            results['inductor_results'][_lbl] = {'error': str(_e), 'ok': False}",
            "        finally:",
            "            torch._dynamo.reset()",
            "",

            "# -- 3b. Recompilation oracle: call the SAME compiled fn",
            "# with a DIFFERENT shape to test shape guard correctness. --",
            "results['recompile_ok'] = False",
            "results['recompile_diff'] = None",
            f"_do_recompile = {not self.use_dynamic!r} and random.random() < 0.15",
            "if results.get('eager_ok') and _do_recompile:",
            "    try:",
            "        torch._dynamo.reset()",
            "        import random as _rnd",
            "        # Build shape-varied inputs: scale each tensor dim by a small factor",
            "        _recomp_inputs = []",
            "        for _t in _program_inputs:",
            "            if isinstance(_t, torch.Tensor) and _t.dim() >= 2:",
            "                _new_shape = list(_t.shape)",
            "                _dim = _rnd.randint(0, len(_new_shape) - 1)",
            "                _new_shape[_dim] = max(1, _new_shape[_dim] + _rnd.choice([-1, 1, 2, -2]))",
            "                _recomp_inputs.append(torch.randn(_new_shape, dtype=_t.dtype, device=_t.device))",
            "            else:",
            "                _recomp_inputs.append(_t.clone() if isinstance(_t, torch.Tensor) else _t)",
            "        _recomp_inputs = tuple(_recomp_inputs)",
            "        # Eager with new shape",
            "        _recomp_eager = program(*tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _recomp_inputs))",
            "        # Compiled with new shape (reuses compiled fn, tests guard/recompilation)",
            "        _recomp_fn = torch.compile(program, backend='inductor')",
            "        _recomp_out = _recomp_fn(*tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _recomp_inputs))",
            "        if isinstance(_recomp_eager, torch.Tensor) and isinstance(_recomp_out, torch.Tensor):",
            "            _rc_diff = (_recomp_eager.float() - _recomp_out.float()).abs().max().item()",
            "            results['recompile_diff'] = _rc_diff",
            "            results['recompile_ok'] = True",
            "    except Exception as _e:",
            "        results['recompile_error'] = str(_e)[:200]",
            "    finally:",
            "        torch._dynamo.reset()",
            "",
            "# -- 4. AOTI -- now runs in subprocess (see _run_aoti_subprocess) --",
            "results['aoti_export_failed'] = False",
            "results['aoti_ok'] = False",
            "",
            "# -- 5. CPU Inductor oracle: eager CPU vs inductor CPU --",
            "# CPU codegen (C++ OpenMP) is a completely different path from Triton.",
            "results['cpu_inductor_ok'] = False",
            "results['cpu_inductor_diff'] = None",
            "results['cpu_inductor_error'] = None",
            "_do_cpu = results.get('eager_ok') and random.random() < 0.15",
            "if _do_cpu:",
            "    try:",
            "        torch._dynamo.reset()",
            "        _cpu_inputs = tuple(",
            "            t.detach().cpu().clone() if isinstance(t, torch.Tensor) else t",
            "            for t in _program_inputs)",
            "        _cpu_eager = program(*_cpu_inputs)",
            "        torch._dynamo.reset()",
            "        _cpu_fn = torch.compile(program, backend='inductor')",
            "        _cpu_inp2 = tuple(",
            "            t.clone() if isinstance(t, torch.Tensor) else t",
            "            for t in _cpu_inputs)",
            "        _cpu_compiled = _cpu_fn(*_cpu_inp2)",
            "        results['cpu_inductor_ok'] = True",
            "        # Compute max diff",
            "        if isinstance(_cpu_eager, torch.Tensor) and isinstance(_cpu_compiled, torch.Tensor):",
            "            _cd = (_cpu_eager.float() - _cpu_compiled.float()).abs().max().item()",
            "            results['cpu_inductor_diff'] = _cd",
            "        elif isinstance(_cpu_eager, (tuple, list)):",
            "            _cd = 0.0",
            "            for _ce, _cc in zip(_cpu_eager, _cpu_compiled):",
            "                if isinstance(_ce, torch.Tensor) and isinstance(_cc, torch.Tensor):",
            "                    _cd = max(_cd, (_ce.float() - _cc.float()).abs().max().item())",
            "            results['cpu_inductor_diff'] = _cd",
            "    except Exception as _cpu_e:",
            "        _cpu_err = str(_cpu_e)[:500]",
            "        results['cpu_inductor_error'] = _cpu_err",
            "        results['cpu_inductor_ok'] = False",
            "    finally:",
            "        torch._dynamo.reset()",
            "",
            "# -- 6. Cross-fusion-level oracle: same program at --",
            "# max_fusion_size=1 vs default. Diff indicates fusion pattern bug.",
            "results['cross_fusion_ok'] = False",
            "results['cross_fusion_diff'] = None",
            "results['cross_fusion_error'] = None",
            f"_do_cross_fusion = results.get('eager_ok') and random.random() < {0.80 if self.name.startswith('cf_') else 0.10}",
            "if _do_cross_fusion:",
            "    try:",
            "        torch._dynamo.reset()",
            "        import torch._inductor.config as _icfg",
            "        # Run with no fusion (max_fusion_size=1)",
            "        with _icfg.patch(max_fusion_size=1):",
            "            _nf_fn = torch.compile(program, backend='inductor')",
            "            _nf_inp = tuple(",
            "                t.clone().detach() if isinstance(t, torch.Tensor) else t",
            "                for t in _program_inputs)",
            "            _nf_out = _nf_fn(*_nf_inp)",
            "        torch._dynamo.reset()",
            "        # Run with default fusion",
            "        _df_fn = torch.compile(program, backend='inductor')",
            "        _df_inp = tuple(",
            "            t.clone().detach() if isinstance(t, torch.Tensor) else t",
            "            for t in _program_inputs)",
            "        _df_out = _df_fn(*_df_inp)",
            "        results['cross_fusion_ok'] = True",
            "        if isinstance(_nf_out, torch.Tensor) and isinstance(_df_out, torch.Tensor):",
            "            _cf_d = (_nf_out.float() - _df_out.float()).abs().max().item()",
            "            results['cross_fusion_diff'] = _cf_d",
            "        elif isinstance(_nf_out, (tuple, list)):",
            "            _cf_d = 0.0",
            "            for _nfe, _dfe in zip(_nf_out, _df_out):",
            "                if isinstance(_nfe, torch.Tensor) and isinstance(_dfe, torch.Tensor):",
            "                    _cf_d = max(_cf_d, (_nfe.float() - _dfe.float()).abs().max().item())",
            "            results['cross_fusion_diff'] = _cf_d",
            "    except Exception as _cf_e:",
            "        results['cross_fusion_error'] = str(_cf_e)[:500]",
            "        results['cross_fusion_ok'] = False",
            "    finally:",
            "        torch._dynamo.reset()",
            "",
            "# -- 7. Cross-dynamic-shape oracle: compile with dynamic=True, --",
            "# call with shape A, then call SAME compiled fn with shape B (no recompile).",
            "# Tests symbolic shape guard generalization correctness.",
            "results['dynamic_reuse_ok'] = False",
            "results['dynamic_reuse_diff'] = None",
            "results['dynamic_reuse_error'] = None",
            f"_do_dyn_reuse = results.get('eager_ok') and random.random() < {0.80 if self.name.startswith('dyn_') else 0.10}",
            "if _do_dyn_reuse:",
            "    try:",
            "        torch._dynamo.reset()",
            "        import random as _rnd2",
            "        # Compile with dynamic=True",
            "        _dyn_fn = torch.compile(program, backend='inductor', dynamic=True)",
            "        # First call with original shape (establishes guards)",
            "        _dyn_inp1 = tuple(",
            "            t.clone().detach() if isinstance(t, torch.Tensor) else t",
            "            for t in _program_inputs)",
            "        _dyn_out1 = _dyn_fn(*_dyn_inp1)",
            "        # Build shape-varied inputs (within guard range -- scale dims by small factor)",
            "        _dyn_inp2 = []",
            "        for _t in _program_inputs:",
            "            if isinstance(_t, torch.Tensor) and _t.dim() >= 1:",
            "                _ns = list(_t.shape)",
            "                for _di in range(len(_ns)):",
            "                    if _ns[_di] > 2:",
            "                        _ns[_di] = max(1, _ns[_di] + _rnd2.choice([-2, -1, 1, 2, 3]))",
            "                _dyn_inp2.append(torch.randn(_ns, dtype=_t.dtype, device=_t.device))",
            "            else:",
            "                _dyn_inp2.append(_t.clone() if isinstance(_t, torch.Tensor) else _t)",
            "        _dyn_inp2 = tuple(_dyn_inp2)",
            "        # Eager ground truth with shape B",
            "        _dyn_eager2 = program(*tuple(",
            "            t.clone() if isinstance(t, torch.Tensor) else t for t in _dyn_inp2))",
            "        # Compiled with shape B (reuses compiled fn -- may or may not recompile)",
            "        _dyn_out2 = _dyn_fn(*tuple(",
            "            t.clone() if isinstance(t, torch.Tensor) else t for t in _dyn_inp2))",
            "        results['dynamic_reuse_ok'] = True",
            "        if isinstance(_dyn_eager2, torch.Tensor) and isinstance(_dyn_out2, torch.Tensor):",
            "            _dr_d = (_dyn_eager2.float() - _dyn_out2.float()).abs().max().item()",
            "            results['dynamic_reuse_diff'] = _dr_d",
            "        elif isinstance(_dyn_eager2, (tuple, list)):",
            "            _dr_d = 0.0",
            "            for _dre, _drc in zip(_dyn_eager2, _dyn_out2):",
            "                if isinstance(_dre, torch.Tensor) and isinstance(_drc, torch.Tensor):",
            "                    _dr_d = max(_dr_d, (_dre.float() - _drc.float()).abs().max().item())",
            "            results['dynamic_reuse_diff'] = _dr_d",
            "    except Exception as _dr_e:",
            "        results['dynamic_reuse_error'] = str(_dr_e)[:500]",
            "        results['dynamic_reuse_ok'] = False",
            "    finally:",
            "        torch._dynamo.reset()",
            "",
            "# -- 8. Functionalization oracle: test functionalize(fn) vs fn --",
            "# Catches Stage 2 bugs: in-place op on aliased views not correctly handled.",
            "results['functionalize_ok'] = False",
            "results['functionalize_diff'] = None",
            "results['functionalize_error'] = None",
            "_do_func = results.get('eager_ok') and random.random() < 0.15",
            "if _do_func:",
            "    try:",
            "        from torch.func import functionalize as _functionalize",
            "        _func_inp = tuple(",
            "            t.clone().detach() if isinstance(t, torch.Tensor) else t",
            "            for t in _program_inputs)",
            "        _func_out = _functionalize(program)(*_func_inp)",
            "        _eager_inp2 = tuple(",
            "            t.clone().detach() if isinstance(t, torch.Tensor) else t",
            "            for t in _program_inputs)",
            "        _eager_out2 = program(*_eager_inp2)",
            "        results['functionalize_ok'] = True",
            "        if isinstance(_eager_out2, torch.Tensor) and isinstance(_func_out, torch.Tensor):",
            "            _fn_d = (_eager_out2.float() - _func_out.float()).abs().max().item()",
            "            results['functionalize_diff'] = _fn_d",
            "        elif isinstance(_eager_out2, (tuple, list)):",
            "            _fn_d = 0.0",
            "            for _fe, _fc in zip(_eager_out2, _func_out):",
            "                if isinstance(_fe, torch.Tensor) and isinstance(_fc, torch.Tensor):",
            "                    _fn_d = max(_fn_d, (_fe.float() - _fc.float()).abs().max().item())",
            "            results['functionalize_diff'] = _fn_d",
            "    except Exception as _func_e:",
            "        results['functionalize_error'] = str(_func_e)[:500]",
            "        results['functionalize_ok'] = False",
            "",
            "# -- 9. Guard oracle: test Dynamo shape guard correctness --",
            "# Compile with shape A, then call SAME compiled fn (no dynamo reset) with",
            "# shape B that crosses Inductor decision boundaries. If guard is too",
            "# permissive, stale compiled code runs on shape B producing silent wrong results.",
            "results['guard_ok'] = False",
            "results['guard_diff'] = None",
            "results['guard_error'] = None",
            "results['guard_trigger'] = None",
            "_do_guard = results.get('eager_ok') and random.random() < 0.12",
            "if _do_guard:",
            "    try:",
            "        torch._dynamo.reset()",
            "        _guard_fn = torch.compile(program, backend='inductor')",
            "        # First call: establish guards with original shape",
            "        _guard_inp1 = tuple(",
            "            t.clone().detach() if isinstance(t, torch.Tensor) else t",
            "            for t in _program_inputs)",
            "        _guard_out1 = _guard_fn(*_guard_inp1)",
            "        # Shape mutations that cross Inductor decision thresholds.",
            "        # These are the shapes most likely to expose guard-too-permissive bugs:",
            "        # e.g. a guard that specializes on 'x.shape[1] >= 2' when the kernel",
            "        # actually assumed 'x.shape[1] % 16 == 0'.",
            "        import random as _grnd",
            "        _guard_shape_muts = [",
            "            # Cross power-of-2 boundary (tiling decisions change)",
            "            ('pow2_cross', lambda s: max(1, (s | (s-1)) + 1) if s > 2 else s + 1),",
            "            # Cross SplitScan threshold (128->129 or reverse)",
            "            ('scan_thresh', lambda s: 129 if s <= 128 else 128),",
            "            # Cross persistent reduction threshold (512->513 or reverse)",
            "            ('red_thresh', lambda s: 513 if s <= 512 else 512),",
            "            # Odd <-> even (Triton tile masking boundary)",
            "            ('parity', lambda s: s + 1 if s % 2 == 0 else s - 1 if s > 1 else 2),",
            "            # 2x scale (changes tile count, reduction strategy)",
            "            ('scale_2x', lambda s: max(1, s * 2)),",
            "            # Shrink to 1 (degenerate dimension, different codegen)",
            "            ('dim_to_1', lambda s: 1),",
            "        ]",
            "        results['guard_ok'] = True",
            "        _max_guard_diff = 0.0",
            "        _worst_trigger = None",
            "        # Pick 3 random mutations to limit compile overhead",
            "        for _gm_name, _gm_fn in _grnd.sample(_guard_shape_muts, min(3, len(_guard_shape_muts))):",
            "            try:",
            "                _gm_inp = []",
            "                _gm_changed = False",
            "                for _t in _program_inputs:",
            "                    if isinstance(_t, torch.Tensor) and _t.dim() >= 2:",
            "                        _ns = list(_t.shape)",
            "                        _dim = _grnd.randint(0, len(_ns) - 1)",
            "                        _old = _ns[_dim]",
            "                        _ns[_dim] = _gm_fn(_old)",
            "                        if _ns[_dim] != _old:",
            "                            _gm_changed = True",
            "                        _gm_inp.append(torch.randn(_ns, dtype=_t.dtype, device=_t.device))",
            "                    elif isinstance(_t, torch.Tensor):",
            "                        _gm_inp.append(torch.randn_like(_t))",
            "                    else:",
            "                        _gm_inp.append(_t)",
            "                if not _gm_changed:",
            "                    continue",
            "                _gm_inp = tuple(_gm_inp)",
            "                # Eager ground truth with new shape",
            "                _gm_eager = program(*tuple(",
            "                    t.clone() if isinstance(t, torch.Tensor) else t for t in _gm_inp))",
            "                # Reuse compiled fn -- do NOT reset dynamo. This tests",
            "                # whether guards correctly detect the shape change.",
            "                _gm_compiled = _guard_fn(*tuple(",
            "                    t.clone() if isinstance(t, torch.Tensor) else t for t in _gm_inp))",
            "                if isinstance(_gm_eager, torch.Tensor) and isinstance(_gm_compiled, torch.Tensor):",
            "                    _gd = (_gm_eager.float() - _gm_compiled.float()).abs().max().item()",
            "                    if _gd > _max_guard_diff:",
            "                        _max_guard_diff = _gd",
            "                        _worst_trigger = _gm_name",
            "            except Exception:",
            "                pass",
            "        results['guard_diff'] = _max_guard_diff",
            "        results['guard_trigger'] = _worst_trigger",
            "    except Exception as _guard_e:",
            "        results['guard_error'] = str(_guard_e)[:500]",
            "        results['guard_ok'] = False",
            "    finally:",
            "        torch._dynamo.reset()",
            "",
            "# -- 10. Finite-difference gradient oracle --",
            "# The ONLY oracle that can catch bugs where eager AND compiled both",
            "# return wrong gradients (e.g. decomposition backward formula is wrong).",
            "# All other oracles are differential (eager vs compiled) and blind to this.",
            "# fp64 only (fp32 finite-diff is too noisy). Scalar output only.",
            "results['finite_diff_ok'] = False",
            "results['finite_diff_max'] = None",
            "results['finite_diff_error'] = None",
            f"_do_fd = (results.get('eager_ok') and '{self.dtype}' == 'torch.float64'",
            "          and random.random() < 0.10)",
            "if _do_fd:",
            "    try:",
            "        # Check if output is scalar (or reducible to scalar)",
            "        _fd_out = results.get('eager')",
            "        _fd_is_scalar = (isinstance(_fd_out, torch.Tensor)",
            "                         and _fd_out.dim() == 0)",
            "        if not _fd_is_scalar and isinstance(_fd_out, torch.Tensor):",
            "            # Wrap program to produce scalar output via .sum()",
            "            def _fd_program(*args):",
            "                return program(*args).sum()",
            "            _fd_is_scalar = True",
            "        else:",
            "            _fd_program = program",
            "        if _fd_is_scalar:",
            "            _fd_eps = 1e-5",
            "            _fd_n_samples = 50",
            "            _fd_max_diff = 0.0",
            "            # Pick the first floating-point tensor input for gradient check",
            "            _fd_param_idx = None",
            "            for _fi, _ft in enumerate(_program_inputs):",
            "                if isinstance(_ft, torch.Tensor) and _ft.is_floating_point():",
            "                    _fd_param_idx = _fi",
            "                    break",
            "            if _fd_param_idx is not None:",
            "                _fd_x = _program_inputs[_fd_param_idx]",
            "                _fd_x_flat = _fd_x.detach().clone().view(-1)",
            "                _fd_numel = _fd_x_flat.numel()",
            "                # Autograd gradient",
            "                _fd_x_ag = _fd_x.detach().clone().requires_grad_(True)",
            "                _fd_ag_inputs = list(_program_inputs)",
            "                _fd_ag_inputs[_fd_param_idx] = _fd_x_ag",
            "                _fd_ag_out = _fd_program(*_fd_ag_inputs)",
            "                _fd_ag_grad = torch.autograd.grad(_fd_ag_out, _fd_x_ag)[0].view(-1)",
            "                # Finite-difference gradient on sampled indices",
            "                import random as _fdrnd",
            "                _fd_indices = _fdrnd.sample(",
            "                    range(_fd_numel), min(_fd_n_samples, _fd_numel))",
            "                for _fdi in _fd_indices:",
            "                    _fd_xp = _fd_x_flat.clone()",
            "                    _fd_xm = _fd_x_flat.clone()",
            "                    _fd_xp[_fdi] += _fd_eps",
            "                    _fd_xm[_fdi] -= _fd_eps",
            "                    _fd_inp_p = list(_program_inputs)",
            "                    _fd_inp_m = list(_program_inputs)",
            "                    _fd_inp_p[_fd_param_idx] = _fd_xp.view(_fd_x.shape)",
            "                    _fd_inp_m[_fd_param_idx] = _fd_xm.view(_fd_x.shape)",
            "                    _fd_fp = _fd_program(*_fd_inp_p).item()",
            "                    _fd_fm = _fd_program(*_fd_inp_m).item()",
            "                    _fd_grad_i = (_fd_fp - _fd_fm) / (2 * _fd_eps)",
            "                    _fd_ag_i = _fd_ag_grad[_fdi].item()",
            "                    _fd_diff_i = abs(_fd_grad_i - _fd_ag_i)",
            "                    # Relative tolerance: |fd - ag| / max(|fd|, |ag|, 1)",
            "                    _fd_scale = max(abs(_fd_grad_i), abs(_fd_ag_i), 1.0)",
            "                    _fd_rel = _fd_diff_i / _fd_scale",
            "                    _fd_max_diff = max(_fd_max_diff, _fd_rel)",
            "                results['finite_diff_ok'] = True",
            "                results['finite_diff_max'] = _fd_max_diff",
            "    except Exception as _fd_e:",
            "        results['finite_diff_error'] = str(_fd_e)[:500]",
            "        results['finite_diff_ok'] = False",
            "",
            "# -- 11. CUDAGraph multi-replay oracle: compile with",
            "# reduce-overhead, replay 5 times with fresh inputs. --",
            "# CUDAGraph bugs often appear on the 2nd+ replay (memory aliasing,",
            "# data-dependent shapes fixed at capture time, tensor mutation).",
            "results['cudagraph_replay_ok'] = False",
            "results['cudagraph_replay_diff'] = None",
            "results['cudagraph_replay_error'] = None",
            "_do_cg = (results.get('eager_ok') and torch.cuda.is_available()",
            "          and random.random() < 0.08)",
            "if _do_cg:",
            "    try:",
            "        torch._dynamo.reset()",
            "        _cg_fn = torch.compile(program, mode='reduce-overhead')",
            "        _cg_max_diff = 0.0",
            "        _cg_n_ok = 0",
            "        for _cg_i in range(5):",
            "            _cg_inp = tuple(",
            "                torch.randn_like(t) if isinstance(t, torch.Tensor) else t",
            "                for t in _program_inputs)",
            "            _cg_eager = program(*tuple(",
            "                t.clone() if isinstance(t, torch.Tensor) else t for t in _cg_inp))",
            "            _cg_compiled = _cg_fn(*tuple(",
            "                t.clone() if isinstance(t, torch.Tensor) else t for t in _cg_inp))",
            "            if torch.cuda.is_available(): torch.cuda.synchronize()",
            "            if isinstance(_cg_eager, torch.Tensor) and isinstance(_cg_compiled, torch.Tensor):",
            "                _cg_d = (_cg_eager.float() - _cg_compiled.float()).abs().max().item()",
            "                _cg_max_diff = max(_cg_max_diff, _cg_d)",
            "                _cg_n_ok += 1",
            "        if _cg_n_ok > 0:",
            "            results['cudagraph_replay_ok'] = True",
            "            results['cudagraph_replay_diff'] = _cg_max_diff",
            "    except Exception as _cg_e:",
            "        results['cudagraph_replay_error'] = str(_cg_e)[:500]",
            "        results['cudagraph_replay_ok'] = False",
            "    finally:",
            "        torch._dynamo.reset()",
            "",
            "# -- Restore kernel heuristic tracing and collect results --",
            "try:",
            "    _TK._get_heuristic = _orig_gh",
            "except Exception: pass",
            "results['_kernel_heuristics'] = list(_kernel_heuristics)",
        ])
        return "\n".join(lines)

    def build_backward_eval_code(self):
        """Build backward differential evaluation code. Called when forward shows
        suspicious signals. Compares eager vs inductor backward gradients and
        traces backward kernel heuristics."""
        return "\n".join([
            "import torch", "",
            self.body_code.strip(), "",

            "# -- Backward kernel heuristic tracing --",
            "_bwd_kernel_heuristics = []",
            "try:",
            "    from torch._inductor.codegen.triton import TritonKernel as _BWD_TK",
            "    _bwd_orig_gh = _BWD_TK._get_heuristic",
            "    def _bwd_traced_gh(_self):",
            "        _h = _bwd_orig_gh(_self)",
            "        _bwd_kernel_heuristics.append(_h)",
            "        return _h",
            "    _BWD_TK._get_heuristic = _bwd_traced_gh",
            "except Exception: pass",
            "",

            "# -- Move inputs to CUDA with requires_grad --",
            "if torch.cuda.is_available():",
            "    _program_inputs = tuple(",
            "        t.cuda().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else",
            "        (t.cuda() if isinstance(t, torch.Tensor) else t)",
            "        for t in _program_inputs)",
            "else:",
            "    _program_inputs = tuple(",
            "        t.requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t",
            "        for t in _program_inputs)",
            "",

            "# -- Eager backward --",
            "try:",
            "    _out_eager = program(*_program_inputs)",
            "    if isinstance(_out_eager, torch.Tensor) and _out_eager.requires_grad:",
            "        _grad = torch.randn_like(_out_eager)",
            "        _out_eager.backward(_grad)",
            "        results['eager_grads'] = [",
            "            p.grad.clone() for p in _program_inputs",
            "            if isinstance(p, torch.Tensor) and p.grad is not None]",
            "        results['backward_grad'] = _grad",
            "        results['backward_eager_ok'] = True",
            "    else:",
            "        results['backward_eager_ok'] = False",
            "        results['backward_eager_error'] = 'output not differentiable'",
            "except Exception as _e:",
            "    results['backward_eager_ok'] = False",
            "    results['backward_eager_error'] = str(_e)",
            "",

            "# -- Inductor backward --",
            "if results.get('backward_eager_ok'):",
            "    try:",
            "        torch._dynamo.reset()",
            "        _bwd_inputs = tuple(",
            "            t.detach().clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else",
            "            (t.clone() if isinstance(t, torch.Tensor) else t)",
            "            for t in _program_inputs)",
            "        _ifn = torch.compile(program, backend='inductor')",
            "        _out_ind = _ifn(*_bwd_inputs)",
            "        _out_ind.backward(results['backward_grad'])",
            "        results['inductor_grads'] = [",
            "            p.grad.clone() for p in _bwd_inputs",
            "            if isinstance(p, torch.Tensor) and p.grad is not None]",
            "        results['backward_inductor_ok'] = True",
            "    except Exception as _e:",
            "        results['backward_inductor_ok'] = False",
            "        results['backward_inductor_error'] = str(_e)",
            "    finally:",
            "        torch._dynamo.reset()",
            "",

            "# -- Restore backward heuristic tracing and collect --",
            "try:",
            "    _BWD_TK._get_heuristic = _bwd_orig_gh",
            "except Exception: pass",
            "results['_bwd_kernel_heuristics'] = list(_bwd_kernel_heuristics)",
        ])


# =============================================================================
# ProgramGenerator
# =============================================================================

# Templates known to use in-place or non-differentiable ops (backward unsafe)
_NO_BACKWARD_TEMPLATES = {
    "d4_scatter_reduce_dup", "d4_index_put_accumulate", "d2_pool_int_dtype",
    "d3_inplace_view_mutation", "d6_multi_margin_loss",
    # D8 inplace / non-differentiable indexing
    "d8_scatter_add_overlap", "d8_index_put_accumulate_3d",
    "d8_take_indexing",
    # D10 templates with non-differentiable paths
    "d10_sort_with_nan",  # nan_to_num + where chain breaks autograd
    "d10_ldexp_chain",    # ldexp w/ integer exponent
    # D11 -- non-differentiable / data-dependent shape
    "d11_cummax_noncontig",       # cummax returns indices, breaks autograd nicely
    "d11_cummin_with_neg",        # cummin same
    "d11_unique_consecutive",     # data-dependent output shape
    # D12 -- data-dependent shape / non-differentiable
    "d12_nonzero_reduce",         # nonzero returns int indices, no autograd
    "d12_unique_sum",             # unique is non-differentiable
    "d12_unique_inverse_compose", # unique w/ inverse -- non-differentiable
    "d12_unique_return_counts",   # unique w/ counts -- non-differentiable
    "d12_bincount_reduce",        # int-only
    "d12_sort_threshold_reduce",  # data-dependent slice
    # Indexing templates with non-differentiable random indices
    "idx_gather_noncontiguous",
    "idx_scatter_broadcast",
    "idx_advanced_index_chain",
    "idx_scatter_view_chain",
}
# Add real-model no-backward entries
try:
    from real_model_templates import REAL_MODEL_NO_BACKWARD
    _NO_BACKWARD_TEMPLATES.update(REAL_MODEL_NO_BACKWARD)
except ImportError:
    pass  # real_model_templates optional
# Cross-fusion template with hardcoded bf16 ignores dtype param
_NO_BACKWARD_TEMPLATES.add("cf_mixed_precision_chain")
# D14: graph break templates use .item() which breaks autograd graph
_NO_BACKWARD_TEMPLATES.update({
    "d14_graph_break_data_dep", "d14_graph_break_attr_mutate",
    "d14_graph_break_loop", "d14_graph_break_cond",
    "d14_graph_break_print", "d14_graph_break_multi_segment",
})
# D15-D28: new templates added for 2.12-2.13 coverage
_NO_BACKWARD_TEMPLATES.update({
    "d15_while_loop", "d15_cond_nested",
    "d15_scan_cummax_chain", "d15_flex_attention_score_mod",
    "d17_compiled_autograd_linear", "d17_compiled_autograd_norm_chain",
    "d17_compiled_autograd_attention", "d17_compiled_autograd_inplace",
    "d18_nested_linear", "d18_nested_softmax", "d18_nested_norm",
    "d19_sparse_mm", "d19_sparse_softmax", "d19_sparse_csr_mv",
    "d20_dropout_train", "d20_bernoulli_mask", "d20_multinomial_sample",
    "d21_linalg_eigh", "d21_linalg_svd", "d21_lstsq",
    "d22_large_numel_pointwise", "d22_large_reduction",
    "d27_bool_indexing_compile",
    # D29: CPU templates -- skip backward for simplicity (different device)
    "d29_cpu_scatter_reduce",
    # D30: integer ops -- non-differentiable
    "d30_bitwise_and_or", "d30_shift_ops", "d30_int_div_mod",
    "d30_int_comparison_reduce", "d30_int64_large_values",
    # D31: zero-element tensors -- autograd edge cases
    "d31_zero_dim_reduce", "d31_zero_cat", "d31_zero_matmul",
    "d31_zero_conv", "d31_zero_softmax",
    # D33: no_grad/inference_mode -- breaks autograd by design
    "d33_inference_mode_compile",
    # D34: type promotion -- mixed integer types not differentiable
    "d34_int_float_add", "d34_int64_float64_matmul",
    # D35: out= parameter -- in-place semantics complicate autograd
    "d35_add_out", "d35_mm_out", "d35_out_noncontiguous", "d35_reduce_out",
    # D36: amp backward -- already calls backward internally
    "d36_amp_forward_backward",
    # D40: higher-order grad templates handle backward internally
    "d40_double_backward_matmul", "d40_double_backward_norm",
    "d40_grad_of_grad_softmax", "d40_vjp_jvp_matmul",
    "d40_backward_through_where", "d40_backward_retain_graph",
})
# D41: as_strided has undefined autograd with overlapping strides
_NO_BACKWARD_TEMPLATES.add("d41_as_strided_pointwise")
# D42: indexing templates use integer indices -- non-differentiable
_NO_BACKWARD_TEMPLATES.update({
    "d42_gather_prime_dims", "d42_scatter_reduce_dim1",
    "d42_index_put_bool_mask", "d42_advanced_int_index_chain",
    "d42_gather_noncontig", "d42_scatter_add_broadcast",
    "d42_index_put_accumulate",
})


class ProgramGenerator:
    def __init__(self):
        self.generated_count = 0

    def generate(self, inject_poison=False):
        dtype  = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
        batch  = random.choice([1, 2, 3, 4, 7])
        seq    = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127])
        hidden = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127, 128, 256])

        name, tpl_fn, _ = random.choices(
            [(n, f, w) for n, f, w in PROGRAM_TEMPLATES], weights=_TPL_WEIGHTS, k=1)[0]

        try:
            body_code, is_det = tpl_fn(batch, seq, hidden, dtype)
        except Exception as e:
            print(f"  [ProgramGenerator] template {name} failed: {type(e).__name__}: {e}")
            body_code, is_det = _tpl_d2_linear_bias_act(batch, seq, hidden, dtype)
            name = "d2_linear_bias_act"

        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code

        self.generated_count += 1
        return GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                            has_poison=poisoned)

    def generate_v2(self, inject_poison=False):
        """Generate a program with config preset injection and dynamic shapes."""
        dtype  = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]
        batch  = random.choice([1, 2, 3, 4, 7])
        seq    = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127])
        hidden = random.choice([7, 8, 13, 16, 32, 33, 64, 65, 127, 128, 256])

        name, tpl_fn, _ = random.choices(
            [(n, f, w) for n, f, w in PROGRAM_TEMPLATES], weights=_TPL_WEIGHTS, k=1)[0]

        try:
            body_code, is_det = tpl_fn(batch, seq, hidden, dtype)
        except Exception as e:
            print(f"  [ProgramGenerator] template {name} failed: {type(e).__name__}: {e}")
            body_code, is_det = _tpl_d2_linear_bias_act(batch, seq, hidden, dtype)
            name = "d2_linear_bias_act"

        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code

        # Config preset selection
        config_preset = _sample_config_preset()

        # Dynamic shapes: 15% probability
        use_dynamic = (random.random() < 0.15)

        supports_backward = (name not in _NO_BACKWARD_TEMPLATES)

        self.generated_count += 1
        return GraphProgram(name, body_code, is_det, dtype, batch, seq, hidden,
                            has_poison=poisoned, config_preset=config_preset,
                            use_dynamic=use_dynamic,
                            supports_backward=supports_backward)


# =============================================================================
# Graph Structure Mutation
# =============================================================================

# Op replacement map: each op can be replaced by same-family alternatives
_OP_REPLACEMENTS = {
    # Activation replacements
    "F.relu": ["F.gelu", "F.silu", "F.leaky_relu", "torch.sigmoid", "torch.tanh"],
    "F.gelu": ["F.relu", "F.silu", "F.leaky_relu", "torch.sigmoid"],
    "F.silu": ["F.relu", "F.gelu", "F.leaky_relu", "torch.sigmoid"],
    "torch.relu": ["F.gelu", "F.silu", "torch.sigmoid", "torch.tanh"],
    "torch.sigmoid": ["F.relu", "F.gelu", "F.silu", "torch.tanh"],
    "torch.tanh": ["F.relu", "F.gelu", "F.silu", "torch.sigmoid"],
    # Reduction replacements
    ".sum(": [".mean(", ".amax("],
    ".mean(": [".sum(", ".amax("],
    # Norm replacements
    "F.layer_norm": ["F.layer_norm"],  # keep but change eps later
    # Softmax
    "F.softmax": ["F.log_softmax"],
    "F.log_softmax": ["F.softmax"],
}

# Intermediate ops that can be inserted
_INSERTABLE_OPS = [
    ".contiguous()",
    ".clone()",
    ".to(memory_format=torch.contiguous_format)",
]

# Numeric parameter regex patterns and their mutation types
_NUMERIC_PARAM_PATTERNS = [
    # padding values like pad=(1, 1, 1, 1) -- tweak individual values
    (r"pad=\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", "pad"),
    # kernel_size, stride etc
    (r"kernel_size=(\d+)", "kernel_size"),
    (r"stride=(\d+)", "stride"),
]


def _mutate_graph_structure(body_code, dtype="torch.float32"):
    """Apply graph structure mutation to existing body_code.
    Randomly selects one mutation type: op replacement (40%),
    intermediate op insertion (30%), or numeric parameter perturbation (30%).
    Returns (mutated_code, changed)."""
    mutated = body_code
    roll = random.random()

    if roll < 0.40:
        # Op replacement
        candidates = []
        for op, replacements in _OP_REPLACEMENTS.items():
            if op in mutated:
                candidates.append((op, replacements))
        if candidates:
            op, replacements = random.choice(candidates)
            replacement = random.choice(replacements)
            # Only replace first occurrence to avoid breaking the entire program
            mutated = mutated.replace(op, replacement, 1)

    elif roll < 0.70:
        # Intermediate op insertion: insert .contiguous()/.clone() after tensor assignments
        lines = mutated.split("\n")
        # Find "x = ..." style assignment lines
        insert_candidates = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith("x = ") or stripped.startswith("x=") or
                    stripped.startswith("out = ") or stripped.startswith("out=")):
                insert_candidates.append(idx)
        if insert_candidates:
            target_idx = random.choice(insert_candidates)
            target_line = lines[target_idx]
            # Extract variable name
            var_name = target_line.strip().split("=")[0].strip()
            indent = " " * (len(target_line) - len(target_line.lstrip()))
            insert_op = random.choice(_INSERTABLE_OPS)
            new_line = f"{indent}{var_name} = {var_name}{insert_op}"
            lines.insert(target_idx + 1, new_line)
            mutated = "\n".join(lines)

    else:
        # Numeric parameter perturbation: modify padding, kernel_size, etc.
        import re as _re
        for pattern, param_type in _NUMERIC_PARAM_PATTERNS:
            matches = list(_re.finditer(pattern, mutated))
            if matches:
                match = random.choice(matches)
                if param_type == "pad":
                    # Perturb padding values
                    vals = [int(match.group(i+1)) for i in range(4)]
                    new_vals = [max(0, v + random.randint(-1, 1)) for v in vals]
                    old_str = match.group(0)
                    new_str = f"pad=({new_vals[0]}, {new_vals[1]}, {new_vals[2]}, {new_vals[3]})"
                    mutated = mutated[:match.start()] + new_str + mutated[match.end():]
                else:
                    # Perturb kernel_size/stride
                    val = int(match.group(1))
                    new_val = max(1, val + random.randint(-1, 1))
                    old_str = match.group(0)
                    new_str = f"{param_type}={new_val}"
                    mutated = mutated[:match.start()] + new_str + mutated[match.end():]
                break  # Only perturb one parameter at a time

    changed = (mutated != body_code)
    return mutated, changed


# Semantic mutation: insert semantically-neutral ops that change compiler behavior
_SEMANTIC_TRANSFORMS = [
    ("{var} = {var}.clone()",              4),   # break alias analysis
    ("{var} = {var}.contiguous()",          4),   # change stride assumptions
    ("{var} = {var}.reshape({var}.shape)",  2),   # trigger view/reshape path
    ("{var} = {var} + 0",                  1),   # disrupt constant folding
    ("{var} = {var} * 1",                  1),   # disrupt mul optimization
    ("{var} = {var}.to({var}.dtype)",       1),   # trigger dtype cast no-op
]
_SEM_TRANSFORM_TEMPLATES = [t for t, _ in _SEMANTIC_TRANSFORMS]
_SEM_TRANSFORM_WEIGHTS   = [w for _, w in _SEMANTIC_TRANSFORMS]


def _apply_semantic_mutation(body_code):
    """Insert 1-2 semantically-neutral ops that change compiler behavior without changing math.
    Returns (mutated_code, changed)."""
    lines = body_code.split("\n")
    # Find "x = ..." assignment lines inside program()
    candidates = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("x = ") or stripped.startswith("x=")
                or stripped.startswith("out = ") or stripped.startswith("out=")):
            candidates.append(idx)
    if not candidates:
        return body_code, False

    n_inserts = random.choice([1, 1, 2])
    targets = random.sample(candidates, min(n_inserts, len(candidates)))
    # Sort descending so insertions don't shift later indices
    targets.sort(reverse=True)
    for target_idx in targets:
        target_line = lines[target_idx]
        var_name = target_line.strip().split("=")[0].strip()
        indent = " " * (len(target_line) - len(target_line.lstrip()))
        tpl = random.choices(_SEM_TRANSFORM_TEMPLATES, weights=_SEM_TRANSFORM_WEIGHTS, k=1)[0]
        new_line = indent + tpl.format(var=var_name)
        lines.insert(target_idx + 1, new_line)

    mutated = "\n".join(lines)
    return mutated, (mutated != body_code)


# =============================================================================
# Backward-Stressing Mutations
# =============================================================================
# These mutations inject op patterns whose BACKWARD graphs stress inductor's
# codegen for gradient computation. The key insight from bug_007: inductor's
# backward handling of view/stride/scatter ops is where most silent bugs live.

_BWD_STRESS_TRANSFORMS = [
    # 1. View chain: backward must reconstruct strides
    ("{var} = {var}.view({var}.shape[0], -1, {var}.shape[-1])\n"
     "    {var} = {var}.transpose(-1, -2).contiguous().transpose(-1, -2)"),

    # 2. Expand + use: backward of expand is sum (stride-0 gradient accumulation)
    ("_bwd_expanded = {var}.mean(dim=-1, keepdim=True).expand_as({var})\n"
     "    {var} = {var} + _bwd_expanded * 0.1"),

    # 3. Narrow + pad: backward of narrow zeros-out, backward of pad slices
    ("_bwd_half = {var}.shape[-1] // 2 + 1\n"
     "    {var} = torch.nn.functional.pad({var}.narrow(-1, 0, _bwd_half), "
     "(0, {var}.shape[-1] - _bwd_half))"),

    # 4. slice_scatter: backward creates stride-0 pattern (bug_007 family)
    ("_bwd_slice = {var}[..., :1]\n"
     "    {var} = torch.slice_scatter({var}, _bwd_slice * 0.5, dim=-1, start=0, end=1)"),

    # 5. Repeat + sum: backward of repeat sums, backward of sum broadcasts
    ("{var} = {var}.repeat(1, 1, 1)[..., :{var}.shape[-1]]"),

    # 6. Multi-use: forces gradient accumulation from multiple paths
    ("_bwd_branch = {var}.detach().clone().requires_grad_(False)\n"
     "    {var} = {var} * torch.sigmoid(_bwd_branch) + {var} * 0.1"),

    # 7. unflatten + flatten: backward must invert reshape
    ("_bwd_last = {var}.shape[-1]\n"
     "    {var} = {var}.flatten(-2) if {var}.dim() > 2 else {var}\n"
     "    {var} = {var}.unflatten(-1, (-1, _bwd_last)) if {var}.dim() > 1 else {var}"),

    # 8. Chunk + cat reversed: backward splits and re-orders
    ("_bwd_chunks = {var}.chunk(2, dim=-1)\n"
     "    {var} = torch.cat([_bwd_chunks[-1], _bwd_chunks[0]], dim=-1)"),
]

_BWD_STRESS_WEIGHTS = [3, 3, 2, 4, 1, 2, 2, 3]  # slice_scatter highest (bug_007 family)


def _apply_backward_stress_mutation(body_code):
    """Inject 1-2 ops whose backward graphs stress inductor codegen.
    Returns (mutated_code, changed)."""
    lines = body_code.split("\n")
    candidates = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("x = ") or stripped.startswith("x=")
                or stripped.startswith("out = ") or stripped.startswith("out=")):
            candidates.append(idx)
    if not candidates:
        return body_code, False

    n_inserts = random.choice([1, 1, 2])
    targets = random.sample(candidates, min(n_inserts, len(candidates)))
    targets.sort(reverse=True)
    for target_idx in targets:
        target_line = lines[target_idx]
        var_name = target_line.strip().split("=")[0].strip()
        indent = " " * (len(target_line) - len(target_line.lstrip()))
        tpl = random.choices(_BWD_STRESS_TRANSFORMS, weights=_BWD_STRESS_WEIGHTS, k=1)[0]
        new_lines = tpl.format(var=var_name).split("\n")
        for i, nl in enumerate(reversed(new_lines)):
            lines.insert(target_idx + 1, indent + nl.strip())

    mutated = "\n".join(lines)
    return mutated, (mutated != body_code)


# =============================================================================
# GraphMutator
# =============================================================================

class GraphMutator:
    """Mutation strategies:
    1. Shape mutation (surgical range controlled by the scheduler)
    2. 70% same-category mutation, 30% cross-category (context-sensitive principle)
    3. dtype mutation (20% probability)
    4. Optional poison injection
    """
    @staticmethod
    def mutate(parent, mutation_range, inject_poison=True):
        r = mutation_range
        new_batch  = max(1, min(8, parent.batch  + random.randint(-max(1,r//8), max(1,r//8))))
        new_seq    = max(4, min(256, parent.seq    + random.randint(-r, r)))
        new_hidden = max(7, min(512, parent.hidden + random.randint(-r, r)))

        dtype = parent.dtype
        if random.random() < 0.20:
            dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]

        # Prefer same-category mutation (context-sensitive principle)
        cat = parent.category
        if random.random() < 0.70 and cat in _TPL_CATEGORIES and _TPL_CATEGORIES[cat]:
            # Pick a random template within the same category
            name = random.choice(_TPL_CATEGORIES[cat])
        else:
            # Cross-category
            name, _, _ = random.choices(
                [(n, f, w) for n, f, w in PROGRAM_TEMPLATES], weights=_TPL_WEIGHTS, k=1)[0]

        try:
            body_code, is_det = _TPL_MAP[name](new_batch, new_seq, new_hidden, dtype)
        except Exception as e:
            print(f"  [Mutator] template {name} failed: {type(e).__name__}: {e}")
            body_code, is_det = _tpl_d2_linear_bias_act(new_batch, new_seq, new_hidden, dtype)
            name = "d2_linear_bias_act"

        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code
        return GraphProgram(name, body_code, is_det, dtype, new_batch, new_seq, new_hidden,
                            has_poison=poisoned)

    @staticmethod
    def mutate_v2(parent, mutation_range, inject_poison=True):
        """Mutate with shape, graph structure, config preset, and dynamic shape variation."""
        r = mutation_range
        new_batch  = max(1, min(8, parent.batch  + random.randint(-max(1,r//8), max(1,r//8))))
        new_seq    = max(4, min(256, parent.seq    + random.randint(-r, r)))
        new_hidden = max(7, min(512, parent.hidden + random.randint(-r, r)))

        dtype = parent.dtype
        if random.random() < 0.20:
            dtype = random.choices(_DTYPES, weights=_DTYPE_WEIGHTS, k=1)[0]

        # 50% chance: apply graph structure mutation to parent's body_code
        # 50% chance: resample from templates
        use_structural_mutation = (random.random() < 0.50 and hasattr(parent, 'body_code')
                                   and parent.body_code)

        if use_structural_mutation:
            name = parent.name
            body_code = parent.body_code
            is_det = parent.is_deterministic
            # Graph structure mutation
            body_code, changed = _mutate_graph_structure(body_code, dtype)
            if not changed:
                # Structural mutation had no effect, fall back to template resampling
                use_structural_mutation = False

        if not use_structural_mutation:
            cat = parent.category
            if random.random() < 0.70 and cat in _TPL_CATEGORIES and _TPL_CATEGORIES[cat]:
                name = random.choice(_TPL_CATEGORIES[cat])
            else:
                name, _, _ = random.choices(
                    [(n, f, w) for n, f, w in PROGRAM_TEMPLATES], weights=_TPL_WEIGHTS, k=1)[0]

            try:
                body_code, is_det = _TPL_MAP[name](new_batch, new_seq, new_hidden, dtype)
            except Exception as e:
                print(f"  [MutatorV2] template {name} failed: {type(e).__name__}: {e}")
                body_code, is_det = _tpl_d2_linear_bias_act(new_batch, new_seq, new_hidden, dtype)
                name = "d2_linear_bias_act"

        poisoned = False
        if inject_poison:
            new_code = GraphPoisonInjector.inject(body_code, dtype)
            poisoned = (new_code != body_code)
            body_code = new_code

        # Config preset: 30% chance to mutate, otherwise inherit from parent
        config_preset = parent.config_preset
        if random.random() < 0.30:
            config_preset = _sample_config_preset()

        # Dynamic shapes: 15% chance to enable
        use_dynamic = (random.random() < 0.15)

        supports_backward = (name not in _NO_BACKWARD_TEMPLATES)

        return GraphProgram(name, body_code, is_det, dtype, new_batch, new_seq, new_hidden,
                            has_poison=poisoned, config_preset=config_preset,
                            use_dynamic=use_dynamic,
                            supports_backward=supports_backward)


# =============================================================================
# GraphCorpus
# =============================================================================

class GraphCorpus:
    """Corpus with kernel-count-weighted sampling. Tracks per-category seeds
    to ensure broad template category coverage."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self._sigs=deque(maxlen=max_size); self._progs=deque(maxlen=max_size)
        self._scores=deque(maxlen=max_size)

    def add_seed(self, program, kernel_count):
        if kernel_count == 0 or program.signature in self._sigs: return
        self._sigs.append(program.signature)
        self._progs.append(program); self._scores.append(kernel_count)

    def select_parent(self):
        if not self._progs: return None
        total = sum(self._scores)
        if total == 0: return random.choice(list(self._progs))
        return random.choices(list(self._progs), weights=[s/total for s in self._scores], k=1)[0]

    def size(self): return len(self._progs)

    def category_coverage(self):
        cats = set(p.category for p in self._progs)
        return cats


# =============================================================================
# Oracle
# =============================================================================


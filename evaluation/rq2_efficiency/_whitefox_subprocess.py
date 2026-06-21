#!/usr/bin/env python3
"""Subprocess worker for running a single WhiteFox test.
Called by _benchmark_worker.py to isolate memory.

Patched: installs Inductor kernel-tracing hooks inside the subprocess and
returns captured kernel names in the result, so the parent process can
account WhiteFox's generated kernels in kernel_diversity / kernel_templates.
"""
import json
import sys
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = False

# ---------------------------------------------------------------------------
# Kernel tracing (mirrors _install_tracing() in _benchmark_worker.py).
# Must be installed before any torch.compile() invocation in this process.
# ---------------------------------------------------------------------------
_kernel_names = []
_heuristics = []
_tracing_installed = False


def _install_tracing():
    global _tracing_installed
    if _tracing_installed:
        return

    try:
        from torch._inductor.codegen.triton import TritonKernel
        _orig_codegen = TritonKernel.codegen_kernel

        def _traced_codegen(self_, *args, **kwargs):
            if getattr(self_, 'persistent_reduction', False):
                _heuristics.append('persistent_reduction')
            elif getattr(self_, 'inside_reduction', False):
                _heuristics.append('reduction')
            else:
                _heuristics.append('pointwise')
            return _orig_codegen(self_, *args, **kwargs)
        TritonKernel.codegen_kernel = _traced_codegen
    except Exception:
        pass

    try:
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel
        _orig_ss_codegen = TritonSplitScanKernel.codegen_kernel

        def _traced_ss_codegen(self_, *args, **kwargs):
            _heuristics.append('split_scan')
            return _orig_ss_codegen(self_, *args, **kwargs)
        TritonSplitScanKernel.codegen_kernel = _traced_ss_codegen
    except Exception:
        pass

    try:
        try:
            from torch._inductor.codegen.wrapper import PythonWrapperCodegen as WrapperCls
        except ImportError:
            from torch._inductor.codegen.wrapper import WrapperCodeGen as WrapperCls

        _orig_gen = WrapperCls.generate_kernel_call

        def _traced_gen(self_, kernel_name, *args, **kwargs):
            _kernel_names.append(str(kernel_name))
            return _orig_gen(self_, kernel_name, *args, **kwargs)
        WrapperCls.generate_kernel_call = _traced_gen

        _orig_extern_out = WrapperCls.generate_extern_kernel_out
        def _traced_extern_out(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_out:{str(node)[:200]}")
            return _orig_extern_out(self_, node, *args, **kwargs)
        WrapperCls.generate_extern_kernel_out = _traced_extern_out

        _orig_extern_alloc = WrapperCls.generate_extern_kernel_alloc
        def _traced_extern_alloc(self_, node, *args, **kwargs):
            _kernel_names.append(f"extern_alloc:{str(node)[:200]}")
            return _orig_extern_alloc(self_, node, *args, **kwargs)
        WrapperCls.generate_extern_kernel_alloc = _traced_extern_alloc
    except Exception as e:
        print(f"  [WARN] kernel name tracing unavailable: {e}", file=sys.stderr)

    _tracing_installed = True


# Install hooks at import time.
_install_tracing()


def run(test_file, device):
    result = {
        'eager_ok': False, 'compiled_ok': False,
        'max_diff': 0.0, 'bug_type': None, 'error': None,
        'trivial_crash': False, 'dtype': 'float32',
        'kernel_names': [],
        'heuristics': [],
    }
    # Reset per-test capture.
    _kernel_names.clear()
    _heuristics.clear()
    try:
        with open(test_file) as f:
            raw_code = f.read()
        code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n" + raw_code
        ns = {}
        exec(code, ns)
        model = None
        for v in ns.values():
            if isinstance(v, torch.nn.Module):
                model = v
                break
        if model is None and 'Model' in ns:
            model = ns['Model']()
        if model is None:
            result['error'] = 'no model found'
            return result
        model = model.to(device).eval()
        input_tensors = []
        for k, v in ns.items():
            if isinstance(v, torch.Tensor) and k not in ('m', 'model'):
                input_tensors.append(v.to(device))
        if not input_tensors:
            result['error'] = 'no inputs found'
            return result
        for t in input_tensors:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                result['dtype'] = str(t.dtype).replace('torch.', '')
                break
        result['code'] = raw_code
        with torch.no_grad():
            eager_out = model(*input_tensors)
        result['eager_ok'] = True
        torch._dynamo.reset()
        compiled_model = torch.compile(model, backend='inductor')
        with torch.no_grad():
            compiled_out = compiled_model(*input_tensors)
        result['compiled_ok'] = True
        if isinstance(eager_out, (tuple, list)):
            eager_list = list(eager_out)
            compiled_list = list(compiled_out)
        else:
            eager_list = [eager_out]
            compiled_list = [compiled_out]
        max_diff = 0.0
        for eo, co in zip(eager_list, compiled_list):
            if isinstance(eo, torch.Tensor) and isinstance(co, torch.Tensor):
                if eo.shape != co.shape:
                    result['bug_type'] = 'shape_mismatch'
                    result['max_diff'] = float('inf')
                    break
                d = (eo.float() - co.float()).abs().max().item()
                max_diff = max(max_diff, d)
        result['max_diff'] = max_diff
        if result['bug_type'] is None:
            if max_diff > 10.0:
                result['bug_type'] = 'CATASTROPHIC_diff'
            elif max_diff > 1e-2:
                result['bug_type'] = 'SUSPICIOUS_diff'
    except torch._dynamo.exc.BackendCompilerFailed as e:
        result['error'] = str(e)[:300]
        result['bug_type'] = 'compile_crash'
    except RuntimeError as e:
        err = str(e)[:300]
        result['error'] = err
        if result['eager_ok'] and not result['compiled_ok']:
            result['bug_type'] = 'compile_crash'
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)[:200]}'
    # Always return captured kernels, even on partial failures.
    result['kernel_names'] = list(_kernel_names)
    result['heuristics'] = list(_heuristics)
    return result


if __name__ == '__main__':
    test_file = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda'
    r = run(test_file, device)
    # Remove 'code' field to keep output small
    r.pop('code', None)
    print(json.dumps(r))

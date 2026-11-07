"""Compatibility checks and minimal shims for current TorchFuzz on PyTorch 2.2."""

import importlib
from typing import Any, Dict, List


UNSUPPORTED_DTYPE_NAMES = ("uint16", "uint32", "uint64")


def install_torch220_import_shims() -> Dict[str, Any]:
    """Allow upstream registry modules to import on PyTorch 2.2.

    PyTorch main's TorchFuzz registry mentions unsigned dtypes introduced after
    2.2.  The aliases are only an import shim.  Route A never selects these
    names for generated programs on 2.2, and the generated code is checked
    against the actual supported dtype set.
    """
    import torch

    aliases: Dict[str, Any] = {}
    for name in UNSUPPORTED_DTYPE_NAMES:
        if not hasattr(torch, name):
            setattr(torch, name, torch.uint8)
            aliases[name] = "torch.uint8_import_only"
    return aliases


def torch_version() -> str:
    import torch

    return str(torch.__version__)


def qualification_snapshot() -> Dict[str, Any]:
    """Return a machine-readable compatibility snapshot without generation."""
    import torch

    aliases = install_torch220_import_shims()
    importlib.invalidate_caches()
    snapshot: Dict[str, Any] = {
        "torch_version": torch_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "import_shims": aliases,
        "native_dtypes": {},
    }
    for name in ("float16", "bfloat16", "float32", "float64", "int32", "int64", "uint8"):
        snapshot["native_dtypes"][name] = bool(hasattr(torch, name))
    snapshot["observer_hooks"] = _observer_hook_snapshot()
    if torch.cuda.is_available():
        snapshot['gpu_name'] = torch.cuda.get_device_name()
        snapshot['gpu_capability'] = list(torch.cuda.get_device_capability())
    try:
        import torchfuzz  # noqa: F401

        snapshot["torchfuzz_import"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on environment
        snapshot["torchfuzz_import"] = "error"
        snapshot["torchfuzz_import_error"] = "%s: %s" % (type(exc).__name__, exc)
    return snapshot


def _observer_hook_snapshot() -> Dict[str, Any]:
    """Report whether the versioned Inductor hook points are importable."""
    result: Dict[str, Any] = {}
    try:
        from torch._inductor.codegen import triton
        from torch._inductor.codegen.triton import TritonKernel

        result["triton_codegen_kernel"] = callable(
            getattr(TritonKernel, "codegen_kernel", None)
        )
        result["triton_metadata"] = callable(getattr(triton, "get_kernel_metadata", None))
        result["triton_define_kernel"] = callable(getattr(triton.TritonScheduling, "define_kernel", None))
    except Exception as exc:
        result["triton_codegen_kernel"] = False
        result["triton_error"] = "%s: %s" % (type(exc).__name__, exc)
    wrapper = None
    try:
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen

        wrapper = PythonWrapperCodegen
    except Exception:
        try:
            from torch._inductor.codegen.wrapper import WrapperCodeGen

            wrapper = WrapperCodeGen
        except Exception as exc:
            result["wrapper_generate_kernel_call"] = False
            result["wrapper_error"] = "%s: %s" % (type(exc).__name__, exc)
    if wrapper is not None:
        result["wrapper_generate_kernel_call"] = callable(
            getattr(wrapper, "generate_kernel_call", None)
        )
        result["wrapper_extern_alloc"] = callable(
            getattr(wrapper, "generate_extern_kernel_alloc", None)
        )
        result["wrapper_extern_out"] = callable(
            getattr(wrapper, "generate_extern_kernel_out", None)
        )
    try:
        from torch._inductor import ir
        result["ir_metadata"] = callable(getattr(ir, "get_kernel_metadata", None))
    except Exception as exc:
        result["ir_metadata"] = False
        result["ir_error"] = "%s: %s" % (type(exc).__name__, exc)
    return result


def supported_ops_default() -> List[str]:
    """Operators whose semantics are available in the PyTorch 2.2 baseline."""
    return [
        "torch.add",
        "torch.sub",
        "torch.mul",
        "torch.div",
        "torch.relu",
        "torch.sigmoid",
        "torch.tanh",
        "torch.exp",
        "torch.sqrt",
        "torch.abs",
        "torch.sin",
        "torch.cos",
        "torch.sum",
        "torch.mean",
        "torch.amax",
        "torch.amin",
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "torch.nn.functional.gelu",
        "torch.nn.functional.silu",
        "torch.nn.functional.softmax",
    ]


def supported_ops_broad() -> List[str]:
    """Explicit tensor-valued scope, with no prescribed graph connections.

    Version availability and per-program meta validation are still required.
    Shape-changing operators need their own propagation semantics before entry.
    """
    return supported_ops_default() + [
        'torch.neg', 'torch.square', 'torch.reciprocal', 'torch.rsqrt',
        'torch.exp2', 'torch.expm1', 'torch.log', 'torch.log2', 'torch.log10',
        'torch.log1p', 'torch.erf', 'torch.erfc', 'torch.floor', 'torch.ceil',
        'torch.round', 'torch.trunc', 'torch.sign', 'torch.sinh', 'torch.cosh',
        'torch.atan', 'torch.asinh', 'torch.maximum', 'torch.minimum',
        'torch.pow', 'torch.remainder', 'torch.fmod', 'torch.hypot',
        'torch.logaddexp', 'torch.nn.functional.elu',
        'torch.nn.functional.leaky_relu', 'torch.nn.functional.softplus',
        'torch.nn.functional.hardswish', 'torch.nn.functional.mish',
    ]


def operator_pool_snapshot(names=None) -> Dict[str, Any]:
    """Reject absent APIs/registry entries explicitly, before graph sampling.

    This is an availability gate. Dtype/shape legality is checked on every
    emitted node by the renderer; availability alone never means execution
    compatibility for arbitrary inputs.
    """
    import torch
    install_torch220_import_shims()
    from torchfuzz.operators import list_operators
    registry = list_operators()
    accepted, excluded, entries = [], {}, {}
    for name in names if names is not None else supported_ops_broad():
        keys = [key for key, op in registry.items() if key == name or op.torch_op_name == name]
        value = torch
        try:
            for part in name.removeprefix('torch.').split('.'):
                value = getattr(value, part)
            if not callable(value):
                raise AttributeError(name)
        except AttributeError:
            excluded[name] = 'api_unavailable'
            continue
        if not keys:
            excluded[name] = 'registry_unavailable'
            continue
        accepted.append(name)
        entries[name] = keys
    return {'torch_version': str(torch.__version__), 'accepted': accepted,
            'excluded': excluded, 'registry_entries': entries,
            'legality_gate': 'render_graph executes every emitted operator on meta tensors'}

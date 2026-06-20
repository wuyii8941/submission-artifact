# [inductor][aotautograd] `torch.compile` crashes on batched matmul inside `torch.inference_mode()`

> **Status: CONFIRMED** — Filed as [upstream issue #181512 (anonymized)]. Developer (dsashidh) confirmed and is working on a fix.

## 🐛 Describe the bug

`torch.compile` crashes when compiling a function that performs batched matrix multiplication (3D+ inputs) inside `torch.inference_mode()`.

The crash occurs in AOT Autograd's `run_functionalized_fw_and_collect_metadata`, where a weakref to an `UntypedStorage` cannot be resolved.

All of the following work individually — only their combination triggers the crash:
- 2D matmul + `inference_mode` + `torch.compile` → ✅
- 3D matmul + `torch.compile` (no `inference_mode`) → ✅
- 3D matmul + `torch.no_grad()` + `torch.compile` → ✅
- 3D matmul + `inference_mode` (eager, no compile) → ✅
- **3D matmul + `inference_mode` + `torch.compile` → ❌ crash**

## Minimal reproducer

```python
import torch

x = torch.randn(2, 4, 8, device='cuda')
w = torch.randn(8, 8, device='cuda')

def f(x, w):
    with torch.inference_mode():
        return (x @ w).sum()

torch._dynamo.reset()
torch.compile(f)(x, w)
# BackendCompilerFailed: RuntimeError: <weakref at ...; to 'torch.storage.UntypedStorage' at ...>
```

Also crashes on CPU:

```python
x = torch.randn(2, 4, 8)  # CPU
w = torch.randn(8, 8)
# same crash
```

## Controls (all pass)

```python
import torch

x3d = torch.randn(2, 4, 8, device='cuda')
x2d = torch.randn(4, 8, device='cuda')
w = torch.randn(8, 8, device='cuda')

# Control 1: 2D matmul + inference_mode → OK
torch._dynamo.reset()
print(torch.compile(lambda x, w: torch.inference_mode()(lambda: (x @ w).sum())())(x2d, w))

# Control 2: 3D matmul, no inference_mode → OK
torch._dynamo.reset()
print(torch.compile(lambda x, w: (x @ w).sum())(x3d, w))

# Control 3: 3D matmul + no_grad → OK
torch._dynamo.reset()
def g(x, w):
    with torch.no_grad():
        return (x @ w).sum()
print(torch.compile(g)(x3d, w))
```

## Affected ops (systematic test, 31 cases)

Any op that internally dispatches through `aten::matmul` or `aten::linear` on 3D+ input crashes:

| Op | Status |
|----|--------|
| `x @ w` (3D×2D, 3D×3D, 4D×2D) | ❌ crash |
| `F.linear(x_3d, w)` with/without bias | ❌ crash |
| `torch.einsum('bih,hj->bij', ...)` | ❌ crash |
| `nn.TransformerEncoderLayer` | ❌ crash (uses F.linear internally) |
| `nn.MultiheadAttention` | ❌ crash (uses F.linear internally) |
| `torch.bmm` | ✅ OK |
| `torch.baddbmm` | ✅ OK |
| `F.conv1d` / `F.conv2d` | ✅ OK |
| `F.batch_norm` / `F.layer_norm` | ✅ OK |
| `F.softmax` / `torch.sort` / `torch.cumsum` | ✅ OK |
| `F.scaled_dot_product_attention` | ✅ OK |
| `torch.linalg.solve/cholesky/svd/eigh` | ✅ OK |
| `nn.LSTM` / `nn.GRU` | ✅ OK |
| pointwise-only 3D / reduction-only 3D | ✅ OK |
| `torch.addmm` (2D) | ✅ OK |

The crash affects any code using `F.linear` or `@` on batched inputs inside `inference_mode` — this is the standard pattern in Transformer inference.

## Trigger conditions

| Condition | Required? | Notes |
|-----------|-----------|-------|
| `torch.inference_mode()` | **Yes** | `torch.no_grad()` does NOT trigger the crash |
| Input ndim | **≥ 3** | 2D matmul / addmm works fine |
| Device | Any | Both CUDA and CPU crash |
| `dynamic=True` | No | Crashes with both static and dynamic |
| Op | `matmul` / `linear` family | `bmm`, `baddbmm` work fine |
| dtype | Any | Tested fp32, fp64 |

## Error message

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: <weakref at 0x7fda0b097f60; to 'torch.storage.UntypedStorage' at 0x7fda0b0b4190>

While executing %matmul : [num_users=1] = call_function[target=operator.matmul](args = (%l_x_, %l_w_), kwargs = {})
```

## Full stack trace

<details>
<summary>Click to expand</summary>

```
Traceback (most recent call last):
  File "repro.py", line 13, in <module>
    torch.compile(f)(x, w)
  File "/pytorch/torch/_dynamo/eval_frame.py", line 1069, in compile_wrapper
    raise e.remove_dynamo_frames() from None
  File "/pytorch/torch/_dynamo/eval_frame.py", line 1052, in compile_wrapper
    result = fn(*args, **kwargs)
  File "/pytorch/torch/_dynamo/convert_frame.py", line 2480, in __call__
    result = self._torchdynamo_orig_backend(
  File "/pytorch/torch/_dynamo/convert_frame.py", line 2198, in __call__
    result = self._inner_convert(
  File "/pytorch/torch/_dynamo/convert_frame.py", line 758, in __call__
    result = _compile(
  File "/pytorch/torch/_dynamo/convert_frame.py", line 1983, in _compile
    guarded_code, tracer_output = compile_inner(code, one_graph, hooks)
  File "/pytorch/torch/_dynamo/convert_frame.py", line 1593, in compile_inner
    result = _compile_inner(code, one_graph, hooks)
  File "/pytorch/torch/_dynamo/convert_frame.py", line 1652, in _compile_inner
    dynamo_output = compile_frame(
  File "/pytorch/torch/_dynamo/convert_frame.py", line 1500, in compile_frame
    bytecode, tracer_output = transform_code_object(code, transform)
  File "/pytorch/torch/_dynamo/bytecode_transformation.py", line 1626, in transform_code_object
    tracer_output = transformations(instructions, code_options)
  File "/pytorch/torch/_dynamo/convert_frame.py", line 1472, in transform
    tracer_output = trace_frame(
  File "/pytorch/torch/_dynamo/convert_frame.py", line 933, in trace_frame
    run_tracer()
  File "/pytorch/torch/_dynamo/convert_frame.py", line 914, in run_tracer
    tracer.run()
  File "/pytorch/torch/_dynamo/symbolic_convert.py", line 1839, in run
    while self.step():
  File "/pytorch/torch/_dynamo/symbolic_convert.py", line 1506, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/pytorch/torch/_dynamo/symbolic_convert.py", line 5211, in RETURN_VALUE
    self._return(inst)
  File "/pytorch/torch/_dynamo/symbolic_convert.py", line 5193, in _return
    all_stack_locals_metadata = self.output.compile_subgraph(
  File "/pytorch/torch/_dynamo/output_graph.py", line 2114, in compile_subgraph
    self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
  File "/pytorch/torch/_dynamo/output_graph.py", line 2730, in compile_and_call_fx_graph
    compiled_fn = self.call_user_compiler(gm, self.example_inputs())
  File "/pytorch/torch/_dynamo/output_graph.py", line 2897, in call_user_compiler
    return self._call_user_compiler(gm, example_inputs)
  File "/pytorch/torch/_dynamo/output_graph.py", line 2983, in _call_user_compiler
    raise BackendCompilerFailed(
  File "/pytorch/torch/_dynamo/output_graph.py", line 2958, in _call_user_compiler
    compiled_fn = compiler_fn(gm, example_inputs)
  File "/pytorch/torch/_dynamo/repro/after_dynamo.py", line 156, in __call__
    compiled_gm = compiler_fn(gm, example_inputs)
  File "/pytorch/torch/__init__.py", line 2482, in __call__
    return compile_fx(
  File "/pytorch/torch/_inductor/compile_fx.py", line 2743, in compile_fx
    return _maybe_wrap_and_compile_fx_main(
  File "/pytorch/torch/_inductor/compile_fx.py", line 2824, in _maybe_wrap_and_compile_fx_main
    return _compile_fx_main(
  File "/pytorch/torch/_inductor/compile_fx.py", line 3022, in _compile_fx_main
    return dynamo_common.aot_autograd(
  File "/pytorch/torch/_dynamo/backends/common.py", line 123, in __call__
    cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
  File "/pytorch/torch/_functorch/aot_autograd.py", line 1223, in aot_module_simplified
    aot_state = create_aot_state(
  File "/pytorch/torch/_functorch/aot_autograd.py", line 582, in create_aot_state
    fw_metadata = run_functionalized_fw_and_collect_metadata(
  File "/pytorch/torch/_functorch/_aot_autograd/collect_metadata_analysis.py", line 220, in inner
    flat_f_outs = f(*flat_f_args)
  File "/pytorch/torch/_functorch/_aot_autograd/graph_capture_wrappers.py", line 1534, in functional_call
    out = PropagateUnbackedSymInts(mod).run(*args)
  File "/pytorch/torch/fx/interpreter.py", line 224, in run
    raise RuntimeError(*e.args) from e
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: <weakref at 0x7fda0b097f60; to 'torch.storage.UntypedStorage' at 0x7fda0b0b4190>

While executing %matmul : [num_users=1] = call_function[target=operator.matmul](args = (%l_x_, %l_w_), kwargs = {})
```

</details>

## Versions

```
PyTorch: 2.13.0.dev20260425+cu126
Python: 3.11.15
OS: Linux-5.4.0-42-generic-x86_64-with-glibc2.31
CUDA: 12.6
Triton: 3.7.0
GPU: Tesla T4 (sm_75) — also crashes on CPU
```

cc <upstream maintainer> <upstream maintainer> <upstream maintainer>

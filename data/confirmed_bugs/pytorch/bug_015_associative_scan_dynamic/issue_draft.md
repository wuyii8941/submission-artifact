# [inductor] `associative_scan` crashes with `dynamic=True` for ndim >= 2

**Filed:** <upstream issue, anonymized for review>

## Description

`torch.compile(dynamic=True)` crashes when compiling `torch._higher_order_ops.associative_scan` on inputs with 2 or more dimensions. Static compilation works fine. 1D inputs with `dynamic=True` also work.

The error is `LoweringException: RuntimeError: Unable to generate code for associative_scan op, because there are lifted arguments`.

The source code already has a TODO for this (`torch/_higher_order_ops/associative_scan.py` line 46 in my local nightly: "Support lifted arguments in inductor for associative_scan"). I have not found a developer comment or issue confirming that this is an officially tracked limitation, so this is best described as a user-visible crash in an unsupported/TODO lowering path.

I also re-confirmed this on a fuzzer-discovered 3D case on 2026-04-27. In that case, eager and `aot_eager` both run successfully, while Inductor with `dynamic=True` fails with the same lowering error. The failure reproduced 4/4 times on the same nightly build.

## Minimal reproducer

```python
import torch

def combine_fn(a, b):
    return a + b

x = torch.randn(4, 8, device='cuda')

# Static: works
torch._dynamo.reset()
r = torch.compile(
    lambda x: torch._higher_order_ops.associative_scan(combine_fn, x, dim=0)
)(x)
print(f"static: OK, shape={r.shape}")

# Dynamic: crashes
torch._dynamo.reset()
r = torch.compile(
    lambda x: torch._higher_order_ops.associative_scan(combine_fn, x, dim=0),
    dynamic=True
)(x)
# LoweringException: RuntimeError: Unable to generate code for
# associative_scan op, because there are lifted arguments
```

## Trigger conditions

| Condition | Required |
|-----------|----------|
| `dynamic=True` | Yes |
| ndim | >= 2 (1D works) |
| scan dim | Any (tested dim=0 on 2D, dim=1 on 3D) |
| dtype | Any (tested fp32, fp64) |
| device | CUDA |

## Fuzzer-discovered variant

This variant was found by the graph consistency fuzzer in the v211 novelty-guided run:

```python
import torch

x = torch.randn(2, 64, 65, dtype=torch.float64, device="cuda").clamp(-2, 2)

def program(x):
    def combine_fn(a, b):
        return a * b
    return torch._higher_order_ops.associative_scan(combine_fn, x, dim=1)

# eager: OK
program(x)

# aot_eager: OK
torch.compile(program, backend="aot_eager", dynamic=True)(x)

# inductor: crashes
torch.compile(program, backend="inductor", dynamic=True)(x)
```

Observed result:

```text
InductorError: LoweringException: RuntimeError: Unable to generate code for associative_scan op, because there are lifted arguments
  target: associative_scan
```

Local artifacts:

- `minimal_repro.py`: minimized trigger and control cases.
- `fuzzer_case_v211_d15_scan_mul.py`: original fuzzer-discovered program.
- `fuzzer_case_v211_d15_scan_mul.json`: original fuzzer metadata.
- `nightly_repro_2026_04_27*.json`: repeated nightly reproduction evidence, 4/4 failures for Inductor with eager and aot_eager both OK.

## Environment

- PyTorch 2.13.0.dev20260425+cu126
- GPU: Tesla T4 (sm_75)
- Triton 3.7.0

## Root cause

When `associative_scan` operates on ndim >= 2, it permutes the tensor to move the scan dim to the last position. Under `dynamic=True`, the non-scan dimensions become symbolic sizes, which appear as "lifted arguments" in the Inductor lowering for the scan subgraph. The Inductor code generator for `associative_scan` does not yet support lifted arguments in the combine function's subgraph.

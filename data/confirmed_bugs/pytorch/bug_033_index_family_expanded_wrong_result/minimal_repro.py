#!/usr/bin/env python3
"""
NEW BUG: index_add on expanded (stride-0) tensor produces wrong forward
results under torch.compile(backend='inductor').

Pattern: x = zeros(1, N).expand(B, -1)  →  x.index_add(0, idx, src)
The expand creates a stride-0 tensor in dim 0. Under Inductor, the
index_add produces incorrect results (forward mismatch up to 42.7).

Root cause hypothesis: Same class as bug_007/025 (stride-0 / expanded
tensors handled incorrectly in Inductor's codegen or decomposition).
"""

import torch

DEVICE = 'cuda'


def minimal_repro():
    print("=" * 60)
    print("index_add on expanded tensor — Eager vs Compiled")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # ── Minimal reproduction ──
    print("\n--- Minimal Repro ---")

    def fn(src, idx):
        x = torch.zeros(1, 8, device=src.device).expand(4, -1)
        return x.index_add(0, idx, src)

    src = torch.ones(3, 8, device=DEVICE)
    idx = torch.tensor([0, 1, 0], device=DEVICE)

    eager_out = fn(src, idx)
    torch._dynamo.reset()
    compiled_out = torch.compile(fn, backend='inductor')(src, idx)

    print(f"  Eager result:")
    print(f"    row 0 (idx 0 appears 2x): {eager_out[0].tolist()}")
    print(f"    row 1 (idx 1 appears 1x): {eager_out[1].tolist()}")
    print(f"    row 2 (no idx):           {eager_out[2].tolist()}")
    print(f"    row 3 (no idx):           {eager_out[3].tolist()}")
    print(f"  Compiled result:")
    print(f"    row 0: {compiled_out[0].tolist()}")
    print(f"    row 1: {compiled_out[1].tolist()}")
    print(f"    row 2: {compiled_out[2].tolist()}")
    print(f"    row 3: {compiled_out[3].tolist()}")
    diff = (eager_out - compiled_out).abs()
    print(f"  Max diff: {diff.max().item():.6f}")

    # ── Control: non-expanded tensor (should match) ──
    print("\n--- Control: Non-expanded tensor ---")
    torch._dynamo.reset()

    def fn_nonexpanded(src, idx):
        x = torch.zeros(4, 8, device=src.device)  # no expand
        return x.index_add(0, idx, src)

    eager_out = fn_nonexpanded(src, idx)
    compiled_out = torch.compile(fn_nonexpanded, backend='inductor')(src, idx)
    diff = (eager_out - compiled_out).abs()
    print(f"  Max diff (no expand): {diff.max().item():.6f}")

    # ── Varying shapes ──
    print("\n--- Varying shapes ---")
    for B, N, n_idx in [(4, 8, 3), (8, 16, 6), (16, 32, 8), (4, 64, 4), (32, 128, 16)]:
        torch._dynamo.reset()

        def make_fn(b, n):
            def fn(src, idx):
                x = torch.zeros(1, n, device=src.device).expand(b, -1)
                return x.index_add(0, idx, src)
            return fn

        src = torch.randn(n_idx, N, device=DEVICE)
        idx = torch.randint(0, B, (n_idx,), device=DEVICE)
        f = make_fn(B, N)
        eager_out = f(src, idx)
        compiled_out = torch.compile(f, backend='inductor')(src, idx)
        diff = (eager_out - compiled_out).abs().max().item()
        print(f"  B={B:3d}, N={N:3d}, n_idx={n_idx:3d}: max_diff={diff:.6f}")

    # ── Expand on dim 1 instead ──
    print("\n--- Expand on dim 1 ---")
    torch._dynamo.reset()

    def fn_dim1(src, idx):
        x = torch.zeros(4, 1, device=src.device).expand(-1, 8)
        return x.index_add(1, idx, src)

    src = torch.randn(4, 3, device=DEVICE)
    idx = torch.tensor([0, 2, 5], device=DEVICE)
    eager_out = fn_dim1(src, idx)
    compiled_out = torch.compile(fn_dim1, backend='inductor')(src, idx)
    diff = (eager_out - compiled_out).abs().max().item()
    print(f"  Expand dim1: max_diff={diff:.6f}")

    # ── Backward correctness ──
    print("\n--- Backward ---")
    torch._dynamo.reset()

    def fn_bwd(src, idx):
        x = torch.zeros(1, 16, device=src.device).expand(8, -1)
        y = x.index_add(0, idx, src)
        return y.sum()

    src_e = torch.randn(4, 16, device=DEVICE, requires_grad=True)
    src_c = src_e.detach().clone().requires_grad_(True)
    idx = torch.tensor([0, 2, 4, 6], device=DEVICE)

    fn_bwd(src_e, idx).backward()
    torch._dynamo.reset()
    torch.compile(fn_bwd, backend='inductor')(src_c, idx).backward()

    if src_e.grad is not None and src_c.grad is not None:
        grad_diff = (src_e.grad - src_c.grad).abs().max().item()
        print(f"  src.grad max_diff: {grad_diff:.6f}")
    else:
        print(f"  src_e.grad: {src_e.grad is not None}, src_c.grad: {src_c.grad is not None}")

    # ── Float64 ──
    print("\n--- Float64 ---")
    torch._dynamo.reset()

    def fn_f64(src, idx):
        x = torch.zeros(1, 8, device=src.device, dtype=torch.float64).expand(4, -1)
        return x.index_add(0, idx, src)

    src = torch.randn(3, 8, device=DEVICE, dtype=torch.float64)
    idx = torch.tensor([0, 1, 0], device=DEVICE)
    eager_out = fn_f64(src, idx)
    compiled_out = torch.compile(fn_f64, backend='inductor')(src, idx)
    diff = (eager_out - compiled_out).abs().max().item()
    print(f"  Float64: max_diff={diff:.6f}")

    # ── Float16 ──
    print("\n--- Float16 ---")
    torch._dynamo.reset()

    def fn_f16(src, idx):
        x = torch.zeros(1, 8, device=src.device, dtype=torch.float16).expand(4, -1)
        return x.index_add(0, idx, src)

    src = torch.randn(3, 8, device=DEVICE, dtype=torch.float16)
    eager_out = fn_f16(src, idx)
    compiled_out = torch.compile(fn_f16, backend='inductor')(src, idx)
    diff = (eager_out.float() - compiled_out.float()).abs().max().item()
    print(f"  Float16: max_diff={diff:.6f}")

    # ── CPU ──
    print("\n--- CPU (may fail due to GCC version) ---")
    torch._dynamo.reset()

    def fn_cpu(src, idx):
        x = torch.zeros(1, 8).expand(4, -1)
        return x.index_add(0, idx, src)

    src = torch.randn(3, 8)
    idx = torch.tensor([0, 1, 0])
    try:
        eager_out = fn_cpu(src, idx)
        compiled_out = torch.compile(fn_cpu, backend='inductor')(src, idx)
        diff = (eager_out - compiled_out).abs().max().item()
        print(f"  CPU: max_diff={diff:.6f}")
    except Exception as e:
        print(f"  CPU: {str(e)[:100]}")

    # ── Check: is the expand the trigger? ──
    print("\n--- Ablation: expand vs clone ---")
    torch._dynamo.reset()

    def fn_clone(src, idx):
        x = torch.zeros(1, 8, device=src.device).expand(4, -1).clone()  # materialize
        return x.index_add(0, idx, src)

    src = torch.ones(3, 8, device=DEVICE)
    idx = torch.tensor([0, 1, 0], device=DEVICE)
    eager_out = fn_clone(src, idx)
    compiled_out = torch.compile(fn_clone, backend='inductor')(src, idx)
    diff = (eager_out - compiled_out).abs().max().item()
    print(f"  With .clone() after expand: max_diff={diff:.6f}")

    torch._dynamo.reset()
    def fn_expand_only(src, idx):
        x = torch.zeros(4, 8, device=src.device)
        x = x.expand(4, -1)  # no-op expand (already correct size)
        return x.index_add(0, idx, src)

    eager_out = fn_expand_only(src, idx)
    compiled_out = torch.compile(fn_expand_only, backend='inductor')(src, idx)
    diff = (eager_out - compiled_out).abs().max().item()
    print(f"  With no-op expand (same size): max_diff={diff:.6f}")


if __name__ == '__main__':
    minimal_repro()

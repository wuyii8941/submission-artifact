#!/usr/bin/env python3
# === PROVENANCE: Class B — issue-body lifted ===============================
# Code lifted from upstream issue body via web extraction. Not guaranteed
# byte-identical to the original issue post; verify against upstream before
# relying. See BUG_MAPPING.md > Reproducer Provenance Audit.
# Verified on PyTorch 2.13.0.dev20260609+cu126 / Tesla T4: all 4 cases
# crash with AssertionError 'expected size N==N, stride X==Y at dim=...'
# at torch.ops.aten.mul.Tensor — fake/meta stride mismatch as described.
# ==========================================================================
"""
Bug: torch.compile crashes on stft/fftn/rfftn/ifftn backward with
"AssertionError: stride mismatch" — meta kernels for `_fft_r2c` and `_fft_c2c`
return strides inconsistent with eager execution, breaking AOTAutograd.

Upstream issue: <upstream issue/PR, anonymized> (open, assigned)

This reproducer triggers the crash on 4 distinct FFT backward variants.
"""
import torch


def case_stft():
    x = torch.randn(128, device="cuda", requires_grad=True)
    @torch.compile(backend="inductor")
    def f(a):
        return torch.stft(a, n_fft=32, hop_length=16, return_complex=True).abs().sum()
    f(x).backward()


def case_fftn_4d():
    x = torch.randn(2, 4, 4, 4, device="cuda", requires_grad=True)
    @torch.compile(backend="inductor")
    def f(a):
        return torch.fft.fftn(a).abs().sum()
    f(x).backward()


def case_rfftn_3d():
    x = torch.randn(4, 8, 8, device="cuda", requires_grad=True)
    @torch.compile(backend="inductor")
    def f(a):
        return torch.fft.rfftn(a).abs().sum()
    f(x).backward()


def case_ifftn_4d_complex():
    x = torch.randn(2, 8, 8, 8, device="cuda", dtype=torch.cfloat, requires_grad=True)
    @torch.compile(backend="inductor")
    def f(a):
        return torch.fft.ifftn(a).abs().sum()
    f(x).backward()


for case in (case_stft, case_fftn_4d, case_rfftn_3d, case_ifftn_4d_complex):
    name = case.__name__
    try:
        case()
        print(f"[{name}] unexpected: no crash")
    except Exception as e:
        print(f"[{name}] CRASH: {type(e).__name__}: {str(e)[:200]}")

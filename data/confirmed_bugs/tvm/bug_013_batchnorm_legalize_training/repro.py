"""Bug 031: LegalizeOps generates training-mode BatchNorm instead of inference-mode.

When batch_norm is lowered via LegalizeOps alone, the generated TIR code computes
batch statistics (mean/var) from the input data, ignoring the provided running_mean
and running_var parameters. This produces wrong results for inference.

DecomposeOpsForInference correctly decomposes batch_norm using the provided stats.

Root cause: LegalizeOps's batch_norm legalization implements training-mode BN
(computing mean/var from data) regardless of context.

Found by pass-differential testing: LegalizeOps-only vs DecomposeOpsForInference+LegalizeOps.
"""

import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as R


def make_bn_mod(N, C, HW):
    bb = relax.BlockBuilder()
    x = relax.Var('x', relax.TensorStructInfo((N, C, HW, HW), 'float32'))
    gamma = relax.Var('gamma', relax.TensorStructInfo((C,), 'float32'))
    beta = relax.Var('beta', relax.TensorStructInfo((C,), 'float32'))
    mean = relax.Var('mean', relax.TensorStructInfo((C,), 'float32'))
    var = relax.Var('var', relax.TensorStructInfo((C,), 'float32'))
    with bb.function('main', [x, gamma, beta, mean, var]):
        with bb.dataflow():
            bn = bb.emit(R.nn.batch_norm(x, gamma, beta, mean, var,
                                         axis=1, epsilon=1e-5))
            out = bb.emit_output(relax.TupleGetItem(bn, 0))
        bb.emit_func_output(out)
    return bb.finalize()


def run_with_passes(mod, inputs, passes):
    pipeline = tvm.ir.transform.Sequential(passes)
    mod_l = pipeline(mod)
    exe = tvm.relax.build(mod_l, target='llvm')
    vm = tvm.relax.VirtualMachine(exe, device=tvm.cpu())
    tvm_inputs = [tvm.runtime.tensor(x, device=tvm.cpu()) for x in inputs]
    return vm['main'](*tvm_inputs).numpy()


if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.RandomState(42)

    N, C, HW = 1, 4, 4
    mod = make_bn_mod(N, C, HW)

    x_np = np.ones((N, C, HW, HW), dtype=np.float32)
    gamma_np = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    beta_np = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    mean_np = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
    var_np = np.array([1.0, 0.25, 4.0, 0.01], dtype=np.float32)
    inputs = [x_np, gamma_np, beta_np, mean_np, var_np]

    # Numpy reference: (x - mean) / sqrt(var + eps) * gamma + beta
    ref = np.zeros_like(x_np)
    for c in range(C):
        ref[0, c] = (x_np[0, c] - mean_np[c]) / np.sqrt(var_np[c] + 1e-5) * gamma_np[c] + beta_np[c]

    # Pipeline 1: LegalizeOps only (WRONG — uses training-mode BN)
    out_legalize = run_with_passes(mod, inputs, [relax.transform.LegalizeOps()])

    # Pipeline 2: DecomposeOpsForInference + LegalizeOps (CORRECT)
    out_decompose = run_with_passes(mod, inputs,
        [relax.transform.DecomposeOpsForInference(), relax.transform.LegalizeOps()])

    print("=== Bug 031: LegalizeOps batch_norm uses training mode ===")
    print(f"Numpy reference (channel 0): {ref[0, 0, 0, 0]:.6f}")
    print(f"LegalizeOps output (channel 0): {out_legalize[0, 0, 0, 0]:.6f}")
    print(f"Decompose+Legalize output (channel 0): {out_decompose[0, 0, 0, 0]:.6f}")
    print()

    diff_legalize = np.abs(out_legalize - ref).max()
    diff_decompose = np.abs(out_decompose - ref).max()

    for c in range(C):
        print(f"  Channel {c}: ref={ref[0,c,0,0]:.4f}  legalize={out_legalize[0,c,0,0]:.4f}  "
              f"decompose={out_decompose[0,c,0,0]:.4f}")

    print(f"\nmax_diff (LegalizeOps vs ref): {diff_legalize:.6f}")
    print(f"max_diff (Decompose+Legalize vs ref): {diff_decompose:.6f}")

    if diff_legalize > 0.01:
        print("\nBUG CONFIRMED: LegalizeOps batch_norm produces wrong results")
    else:
        print("\nBug NOT reproduced")

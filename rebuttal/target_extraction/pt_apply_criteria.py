#!/usr/bin/env python3
import json, sys
c = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'pt_candidates.json'))
autotune_default = c['config']['max_autotune']
coop_default = c['config']['triton.cooperative_reductions']
rows = []
add = lambda unit, cand, decision, family, rule: rows.append(
    {'unit': unit, 'candidate': cand, 'decision': decision, 'family': family, 'rule': rule})

for h in c['kernel_heuristics']:
    for r in h['returns']:
        if r == 'fixed_config':
            add('kernel heuristic', f"{h['class']}:{r}", 'exclude', None,
                'C2: fixed launch config of user-defined Triton kernels fused with epilogues; not chosen from program conditions')
        elif r == 'cooperative_reduction':
            add('kernel heuristic', f"{h['class']}:{r}", 'include (feature flag)', r,
                f'C1 only with the documented flag triton.cooperative_reductions (default {coop_default}); '
                'C2-C4 hold: chosen from sizes, visible per kernel')
        else:
            add('kernel heuristic', f"{h['class']}:{r}", 'include', r, 'C1-C4: returned by the selector from sizes and program structure')
for n in c['ir_loop_nodes']:
    if n in ('Reduction', 'WelfordReduction', 'OnlineSoftmaxReduction', 'ArgReduction', 'MultiOutputReduction'):
        add('IR loop node', n, 'merge', 'reduction / persistent / cooperative',
            'C3: same reduction code generator; implementation chosen by the reduction heuristics above')
    elif n == 'Pointwise' or n == 'Scatter':
        add('IR loop node', n, 'merge', 'pointwise', 'C3: pointwise code generator (heuristic "pointwise")')
    elif n == 'Scan':
        add('IR loop node', n, 'include', 'scan', 'C1-C4: single-block scan kernel, distinct from reductions')
    elif n == 'SplitScan':
        add('IR loop node', n, 'include', 'split_scan', 'C1-C4: TritonSplitScanKernel; chosen from scan length and rows')
    elif n == 'Sort':
        add('IR loop node', n, 'include (no generator)', 'sort',
            'C1-C4 hold (Triton sort for dims <= 512 by default); no generator in the artifact')
    else:
        add('IR loop node', n, 'review', None, 'not classified by the rules')
hw_specific = {'_scaled_mm': 'fp8 GEMM, compute capability >= 8.9', '_sparse_semi_structured_mm': '2:4 sparse GEMM, compute capability >= 8.0',
               '_grouped_mm': 'grouped GEMM, compute capability >= 9.0', '_scaled_grouped_mm': 'fp8 grouped GEMM, compute capability >= 9.0',
               '_int_mm': 'int8 GEMM (integer dtype, outside the floating-point targets)'}
for e in c['extern_kernel_choices']:
    name = e['name']
    if name in hw_specific:
        add('extern choice', name, 'exclude', None, f'C2: {hw_specific[name]}')
    elif e['module'] == 'conv' or name == 'convolution':
        add('extern choice', name, 'merge', 'extern_conv', 'C3: cuDNN convolution dispatch (backward variants reached through the backward context)')
    else:
        add('extern choice', name, 'merge', 'extern_gemm', 'C3: cuBLAS GEMM dispatch')
add('Triton template', f"{len(c['triton_templates'])} templates", 'exclude', None,
    f'C1: considered only when max_autotune is on (default {autotune_default}) and chosen by timing, not by program conditions')
add('ATen fallback', f"{len(c['aten_fallback_ops'])} operator overloads", 'exclude', None,
    'C2: fixed per operator at registration; no implementation choice to steer (their operators are still used as entries and contexts)')
add('code generator', f"{len(c['non_default_codegens'])} non-default modules", 'exclude', None,
    'C1: CPU, Halide, MPS, Pallas, MTIA, XPU, ROCm, CUTLASS/CuTe backends are not the default CUDA Triton path')
add('kernel class', 'ComboKernel (foreach)', 'include (no generator)', 'foreach',
    'C1-C4 hold for _foreach operators (horizontal fusion); no generator in the artifact')

families = sorted({r['family'] for r in rows if r['decision'] in ('include', 'merge') and r['family'] and '/' not in r['family']})
flag_families = sorted({r['family'] for r in rows if r['decision'] == 'include (feature flag)'})
no_gen = sorted({r['family'] for r in rows if r['decision'] == 'include (no generator)'})
out = {'torch_version': c['torch_version'], 'rows': rows, 'default_path_families': families,
       'feature_flag_families': flag_families, 'families': sorted(families + flag_families), 'no_generator': no_gen}
json.dump(out, open('pt_targets_derived.json', 'w'), indent=1)
print('families on the default path:', families, '| behind a documented feature flag:', flag_families)
print('no generator:', no_gen)
print('rows:', len(rows), {d: sum(r['decision'] == d for r in rows) for d in sorted({r['decision'] for r in rows})})

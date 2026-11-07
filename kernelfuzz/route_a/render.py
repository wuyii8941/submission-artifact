"""Render the upstream operator DAG with one common staged differential harness."""
import json
import torch
from .legalizer import REDUCTIONS, reduction_axes


def render_graph(graph, seed, dynamic=True):
    from torchfuzz.codegen import generate_simple_operation_code
    from torchfuzz.tensor_fuzzer import TensorSpec
    leaves = []
    body = []
    for nid in graph.get_topological_order():
        node = graph.nodes[nid]
        spec = node.output_spec
        if isinstance(spec, TensorSpec):
            storage = 1 + sum(max(0,d-1)*abs(s) for d,s in zip(spec.size, spec.stride))
            if storage > 4_000_000:
                raise ValueError('tensor_storage_budget_exceeded')
        variable = 'var_' + nid
        inputs = ['var_' + source for source in node.input_nodes]
        if not node.input_nodes and node.op_name != 'constant':
            leaves.append((nid, node.output_spec))
            body.append(variable + ' = arg_' + nid)
        elif node.op_name in REDUCTIONS:
            source = graph.nodes[node.input_nodes[0]].output_spec
            axes, keep = reduction_axes(tuple(source.size), tuple(node.output_spec.size))
            kwargs = ', dtype=' + str(node.output_spec.dtype) if node.op_name in ('torch.sum', 'torch.mean') else ''
            body.append(f'{variable} = {node.op_name}({inputs[0]}, dim={axes!r}, keepdim={keep!r}{kwargs})')
        else:
            body.extend(generate_simple_operation_code(variable, inputs, node.op_name, node.output_spec))
    params = ', '.join('arg_' + nid for nid, _ in leaves)
    function = 'def fuzzed_program(' + params + '):\n' + '\n'.join('    '+line for line in body)
    function += '\n    return var_' + graph.root_node_id + '\n'
    # Execute every emitted operator on meta tensors before admitting the case.
    namespace = {'torch': torch}
    validation = function.rsplit('    return ', 1)[0] + '    return locals()\n'
    exec(compile(validation, '<route-a-meta-validation>', 'exec'), namespace)
    meta_args = []
    for _, spec in leaves:
        if isinstance(spec, TensorSpec):
            storage = 1 + sum(max(0,d-1)*abs(s) for d,s in zip(spec.size, spec.stride))
            if storage > 4_000_000:
                raise ValueError('tensor_storage_budget_exceeded')
            meta_args.append(torch.empty_strided(spec.size, spec.stride, dtype=spec.dtype, device='meta'))
        else:
            meta_args.append(spec.constant if spec.constant is not None else 1)
    with torch.device('meta'):
        actual = namespace['fuzzed_program'](*meta_args)
    for nid, node in graph.nodes.items():
        spec = node.output_spec
        val = actual['var_' + nid]
        if isinstance(spec, TensorSpec):
            if spec.dtype == torch.bfloat16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
                raise ValueError('hardware_unsupported_bfloat16')
            if not isinstance(val, torch.Tensor) or tuple(val.shape) != spec.size or val.dtype != spec.dtype:
                raise ValueError('emitted_spec_mismatch:' + nid)

    code = 'import torch, json\ntorch.set_default_device("cuda")\ntorch.manual_seed(' + str(seed) + ')\n'
    code += function + '\nprint("ROUTE_A_STAGE=inputs", flush=True)\n'
    for nid, spec in leaves:
        name = 'arg_' + nid
        if isinstance(spec, TensorSpec):
            storage = 1 + sum(max(0,d-1)*abs(s) for d,s in zip(spec.size, spec.stride))
            code += f'{name}_storage = torch.empty(({storage},), dtype={spec.dtype}, device="cuda")\n'
            if spec.dtype.is_floating_point:
                code += f'{name}_storage.normal_()\n'
            else:
                code += f'{name}_storage.random_(0, 2)\n'
            code += f'{name} = {name}_storage.as_strided({spec.size!r}, {spec.stride!r})\n'
        else:
            code += name + ' = ' + repr(spec.constant if spec.constant is not None else 1) + '\n'
    code += f'args = ({params},)\n' if len(leaves) == 1 else f'args = ({params})\n'
    code += '''print("ROUTE_A_STAGE=eager", flush=True)
reference = fuzzed_program(*args)
torch.cuda.synchronize()
print("ROUTE_A_STAGE=compile", flush=True)
compiled = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result = compiled(*args)
torch.cuda.synchronize()
print("ROUTE_A_STAGE=oracle", flush=True)
if isinstance(reference, torch.Tensor):
    if not (reference.dtype.is_floating_point or reference.dtype.is_complex):
        passed = reference.shape == result.shape and reference.dtype == result.dtype and torch.equal(reference, result)
        oracle = {"status": "pass" if passed else "candidate", "comparison": "exact", "numel": reference.numel()}
    else:
        if reference.dtype in (torch.float16, torch.bfloat16):
            atol, rtol = 1e-2, 1e-2
        elif reference.dtype == torch.float64:
            atol, rtol = 1e-8, 1e-5
        else:
            atol, rtol = 1e-5, 1e-4
        passed = reference.shape == result.shape and reference.dtype == result.dtype and torch.allclose(reference, result, atol=atol, rtol=rtol, equal_nan=True)
        oracle = {"status": "pass" if passed else "candidate", "atol": atol, "rtol": rtol, "numel": reference.numel()}
else:
    oracle = {"status": "pass" if reference == result else "candidate"}
print("ROUTE_A_ORACLE=" + json.dumps(oracle), flush=True)
print("ROUTE_A_STAGE=done", flush=True)
'''
    code = code.replace('dynamic=True)', 'dynamic=' + repr(dynamic) + ')')
    compile(code, '<route-a-render>', 'exec')
    return code

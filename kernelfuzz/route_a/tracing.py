"""Executed verbatim in each isolated PyTorch 2.2 worker (no Route A imports)."""
import hashlib as _ra_hash
import json as _ra_json
import re as _ra_re

_route_a_events = []
_route_a_hook_status = []


def _ra_error(where, error):
    _route_a_events.append('observer_error:' + _ra_json.dumps({
        'where': where, 'error': type(error).__name__ + ': ' + str(error)}))


def _ra_origins(schedule):
    from torch._inductor.utils import aggregate_origins
    result = []
    for origin in aggregate_origins(schedule):
        if origin.op != 'call_function':
            continue
        original = origin.meta.get('original_aten')
        original = str(getattr(original, '_overloadpacket', original))
        from_node = origin.meta.get('from_node', [])
        if from_node:
            source = from_node[0]
            source_name = source.name if hasattr(source, 'name') else source[0]
            result.append({'source_node': str(source_name),
                           'original_aten': original, 'lowered_aten': str(origin.target)})
    return sorted(result, key=lambda x: (x['source_node'], x['original_aten'], x['lowered_aten']))


def _ra_metadata_hook(original):
    def metadata(schedule, wrapper):
        result = original(schedule, wrapper)
        try:
            wrapper._ra_pending_origins = _ra_origins(schedule)
        except Exception as error:
            wrapper._ra_pending_origins = None
            _ra_error('origins', error)
        return result
    return metadata


def _ra_kernel_name(node):
    # PyTorch 2.13 uses get_kernel_name; the 2.2 baseline uses codegen_kernel_name.
    getter = getattr(node, 'codegen_kernel_name', None)
    return getter() if getter is not None else node.get_kernel_name()


def _ra_emit(wrapper, kernel, external=False, schedule=None):
    origins = getattr(wrapper, '_ra_pending_origins', None)
    wrapper._ra_pending_origins = None  # never reuse a preceding call's origins
    definition = getattr(wrapper, '_ra_definitions', {}).get(str(kernel), {})
    if origins is None:
        origins = definition.get('origins')
    if schedule is not None:
        try:
            origins = _ra_origins(schedule)
        except Exception as error:
            _ra_error('extern_origins', error)
    source = definition.get('source', '')
    family = definition.get('family', 'unknown')
    if external:
        name = str(kernel).split('.')[-1]
        family = 'extern_gemm' if name in ('mm', 'bmm', 'addmm', 'matmul') else 'extern_other'
    _route_a_events.append('call:' + _ra_json.dumps({
        'kernel': str(kernel), 'family': family, 'external': external,
        'origins': origins, 'source': source,
        'source_sha256': _ra_hash.sha256(source.encode()).hexdigest() if source else None,
    }))


try:
    from torch._inductor.codegen import triton as _ra_triton
    from torch._inductor import ir as _ra_ir
    try:
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen as _ra_wrapper
    except ImportError:
        from torch._inductor.codegen.wrapper import WrapperCodeGen as _ra_wrapper
    _ra_triton.get_kernel_metadata = _ra_metadata_hook(_ra_triton.get_kernel_metadata)
    _ra_ir.get_kernel_metadata = _ra_metadata_hook(_ra_ir.get_kernel_metadata)
    _route_a_hook_status.append('per_node_origins')

    _ra_original_define = _ra_triton.TritonScheduling.define_kernel
    def _ra_define(self, source, schedule, *extra, **kwargs):
        name = _ra_original_define(self, source, schedule, *extra, **kwargs)
        from torch._inductor.virtualized import V
        wrapper = V.graph.wrapper_code
        if not hasattr(wrapper, '_ra_definitions'):
            wrapper._ra_definitions = {}
        source_text = source.getvalue() if hasattr(source, 'getvalue') else str(source)
        match = _ra_re.search(r'@(?:triton_heuristics\.)?(persistent_reduction|reduction|pointwise)\s*\(', source_text)
        try:
            origins = _ra_origins(schedule)
        except Exception as error:
            origins = None
            _ra_error('define_origins', error)
        wrapper._ra_definitions[name] = {
            'source': source_text, 'family': match.group(1) if match else 'unknown',
            'origins': origins,
        }
        return name
    _ra_triton.TritonScheduling.define_kernel = _ra_define
    _route_a_hook_status.append('kernel_definition')

    _ra_original_call = _ra_wrapper.generate_kernel_call
    def _ra_call(self, name, *args, **kwargs):
        _ra_emit(self, name)
        return _ra_original_call(self, name, *args, **kwargs)
    _ra_wrapper.generate_kernel_call = _ra_call
    _route_a_hook_status.append('kernel_call')

    _ra_original_out = _ra_wrapper.generate_extern_kernel_out
    def _ra_out(self, *args, **kwargs):
        if len(args) == 1:
            node = args[0]
            _ra_emit(self, _ra_kernel_name(node), external=True, schedule=node)
        else:
            _ra_emit(self, args[3] if len(args) >= 4 else kwargs.get('kernel'), external=True)
        return _ra_original_out(self, *args, **kwargs)
    _ra_wrapper.generate_extern_kernel_out = _ra_out
    _ra_original_alloc = _ra_wrapper.generate_extern_kernel_alloc
    def _ra_alloc(self, kernel, *args, **kwargs):
        _ra_emit(self, _ra_kernel_name(kernel), external=True, schedule=kernel)
        return _ra_original_alloc(self, kernel, *args, **kwargs)
    _ra_wrapper.generate_extern_kernel_alloc = _ra_alloc
    _route_a_hook_status.append('extern_call')
except Exception as _ra_install_error:
    _ra_error('hook_install', _ra_install_error)

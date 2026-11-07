"""One explicit input/action domain for every arm; no compiler targets here."""
import hashlib
import inspect
from pathlib import Path
from inventory import ROOT
import torch
from torchfuzz.codegen import make_template
from torchfuzz.tensor_fuzzer import TensorSpec, ScalarSpec

DTYPES=tuple(make_template('default').supported_dtypes())
LAYOUTS=('contiguous','column_major')
# Common action values include zero, although the conservative shape binder
# currently returns unsupported for zero. That failure remains in all costs.
SHAPE_VALUES=(0,1,2,7,16,17,32)
MAX_STORAGE_ELEMENTS=4_000_000


def descriptor():
    template=type(make_template('default'));path=Path(inspect.getsourcefile(template))
    return {'schema':1,'template':'default','device':'cuda','mode':'forward_static_fullgraph',
            'emitted_calls':'forward_functional; out= and mutation rejected by functional_domain.py',
            'fixed_compiler_config':{'coordinate_descent_tuning':False},
            'dtype_domain_meaning':'eager semantic candidates; device compilation support recorded separately,not silently filtered',
            'dtypes':[str(d) for d in DTYPES],'layout_actions':list(LAYOUTS),
            'shape_actions':list(SHAPE_VALUES),'zero_dimensions_legal':True,
            'shape_zero_binding_currently_supported':False,'max_storage_elements':MAX_STORAGE_ELEMENTS,
            'generator_shapes':'unchanged upstream semantic specifications; no decision-specific shape generator',
            'input_values':'per-sample seeded normal floating/complex,0..1 integral/bool; shared value-domain screen rejects unsafe combinations without repair',
            'binding_parameters':'only registered shape,dtype,layout actions; topology and other semantic parameters frozen',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'template_source':str(path),'template_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_dtype(dtype):
    if dtype not in DTYPES:raise ValueError('dtype_outside_common_default_template_domain:'+str(dtype))


def validate_graph(graph):
    for node in graph.nodes.values():
        spec=node.output_spec
        if not isinstance(spec,(TensorSpec,ScalarSpec)):raise ValueError('unsupported_common_spec')
        validate_dtype(spec.dtype)
        if isinstance(spec,TensorSpec):
            if any(type(d) is not int or d<0 for d in spec.size):raise ValueError('invalid_common_shape')
            if len(spec.size)!=len(spec.stride) or any(type(s) is not int or s<0 for s in spec.stride):raise ValueError('invalid_common_stride')
            storage=0 if 0 in spec.size else 1+sum(max(0,d-1)*s for d,s in zip(spec.size,spec.stride))
            if storage>MAX_STORAGE_ELEMENTS:raise ValueError('common_storage_budget_exceeded')
    return True

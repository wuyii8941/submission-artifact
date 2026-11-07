"""Explicit v3 semantic cells. This module does not extend any legacy/v2 pool."""
from dataclasses import asdict, dataclass
import itertools
import math

SCHEMA = 'route-a-semantic-domain-v3.1'
FAMILIES = ('pointwise', 'reduction', 'linear', 'normalization', 'indexing', 'view')
DTYPES = ('float16', 'float32', 'float64', 'complex64')
SHAPES = ('scalar', 'empty', 'singleton', 'prime', 'broadcast', 'large_reduction')
LAYOUTS = ('contiguous', 'transpose', 'slice', 'as_strided', 'view_chain', 'shared', 'mutation')
MODES = ('static', 'dynamic_batch', 'dynamic_inner', 'first_order', 'second_order', 'aot_autograd')
OBJECTIVES = ('operator', 'tensor', 'shape', 'layout_alias', 'dynamic_autograd', 'compiler')
AXES = {'operator': FAMILIES, 'tensor': DTYPES, 'shape': SHAPES,
        'layout_alias': LAYOUTS, 'dynamic_autograd': MODES,
        'compiler': ('fusion', 'decomposition', 'branched')}


@dataclass(frozen=True, order=True)
class SemanticCell:
    family: str = 'pointwise'
    dtype: str = 'float32'
    shape: str = 'prime'
    layout: str = 'contiguous'
    mode: str = 'static'
    variant: int = 0
    context: str = 'fusion'

    def to_dict(self):
        return asdict(self)

    def tags(self):
        return dict(zip(OBJECTIVES, (self.family, self.dtype, self.shape,
                                    self.layout, self.mode, self.context)))


def decision(cell, capability):
    """Conservative declared legal domain, independent of observed compiler outcomes."""
    def no(reason, phase='domain', legal=True):
        return {'admitted': False, 'applicability': 'inapplicable', 'legal': legal,
                'phase': phase, 'reason': reason, 'capability': asdict(capability)}
    if cell.dtype == 'bfloat16' and not capability.native_bf16:
        return no('native_bfloat16_not_supported_on_sm75', 'capability')
    if cell.dtype in ('int32', 'int64', 'bool'):
        return no('integer_boolean_data_excluded_indices_are_bounded_int64', 'domain')
    if (cell.family not in FAMILIES or cell.dtype not in DTYPES + ('bfloat16',)
            or cell.shape not in SHAPES or cell.layout not in LAYOUTS
            or cell.mode not in MODES or cell.context not in AXES['compiler']
            or type(cell.variant) is not int or cell.variant not in range(3)):
        return no('unregistered_semantic_cell', legal=False)
    if cell.dtype == 'complex64' and (cell.family == 'normalization'
            or (cell.family == 'reduction' and cell.variant == 2)):
        return no('complex_ordering_or_normalization_not_qualified')
    if cell.shape == 'scalar' and cell.family not in ('pointwise', 'reduction'):
        return no('operator_requires_positive_rank')
    if cell.shape == 'large_reduction' and cell.family not in ('reduction', 'normalization'):
        return no('large_extent_budget_reserved_for_reduction_families')
    if cell.shape in ('scalar', 'empty') and cell.layout not in ('contiguous', 'view_chain'):
        return no('nontrivial_layout_requires_nonempty_positive_rank')
    if cell.shape == 'singleton' and cell.layout == 'transpose':
        return no('transpose_requires_two_nonunit_dimensions')
    if cell.shape == 'broadcast' and cell.layout in ('shared', 'mutation'):
        return no('shared_pair_requires_identical_input_specs')
    if cell.mode.startswith('dynamic') and cell.shape not in ('prime', 'broadcast'):
        return no('dynamic_families_require_nonunit_positive_extents')
    if cell.mode in ('first_order', 'second_order', 'aot_autograd'):
        if cell.dtype not in ('float32', 'float64'):
            return no('derivative_qualification_requires_fp32_or_fp64')
        if cell.shape == 'empty' or cell.layout == 'mutation':
            return no('derivative_loss_requires_nonempty_nonmutating_inputs')
        if cell.mode == 'second_order' and (cell.family == 'indexing'
                or (cell.family == 'reduction' and cell.variant == 2)):
            return no('nonsmooth_or_indexed_second_order_not_qualified')
    return {'admitted': True, 'applicability': 'applicable', 'legal': True,
            'phase': 'admitted', 'reason': None, 'capability': asdict(capability)}


def compatible_cells(capability):
    for values in itertools.product(FAMILIES, DTYPES, SHAPES, LAYOUTS, MODES, range(3), AXES['compiler']):
        cell = SemanticCell(*values)
        if decision(cell, capability)['admitted']:
            yield cell


def storage_proof(shape, stride, offset=0, limit=262144):
    """Nonnegative strides, separated dimension spans imply injective addresses."""
    if len(shape) != len(stride) or any(type(x) is not int or x < 0 for x in (*shape, *stride, offset)):
        raise ValueError('invalid_storage_descriptor')
    span = 1
    for st, size in sorted((st, size) for size, st in zip(shape, stride) if size > 1):
        if st < span:
            raise ValueError('overlapping_layout_not_proven_safe')
        span += (size - 1) * st
    required = offset + (1 + sum((d - 1) * st for d, st in zip(shape, stride)) if all(shape) else 0)
    allocation = max(1, required)
    if allocation > limit or math.prod(shape) > limit:
        raise ValueError('storage_budget_exceeded')
    return {'allocation_elements': allocation, 'storage_offset': offset,
            'max_address_exclusive': required, 'nonoverlap_proven': True,
            'proof': 'nonnegative_strides_separated_dimension_spans'}

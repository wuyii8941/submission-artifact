"""Scoped operator sampling; the upstream generator owns graph topology."""
from contextlib import contextmanager
import math


@contextmanager
def target_weights(operation_names, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('target_weight_must_be_positive_finite')
    from torchfuzz.operators import list_operators
    names = set(operation_names)
    changed = []
    try:
        for name, operator in list_operators().items():
            if name not in names and operator.torch_op_name not in names:
                continue
            previous = operator.__dict__.get('get_weight')
            method = operator.get_weight
            operator.get_weight = lambda _method=method, **kwargs: multiplier * _method(**kwargs)
            changed.append((operator, previous))
        yield
    finally:
        for operator, previous in changed:
            if previous is None:
                del operator.get_weight
            else:
                operator.get_weight = previous

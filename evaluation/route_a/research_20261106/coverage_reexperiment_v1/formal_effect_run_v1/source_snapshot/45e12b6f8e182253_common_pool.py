"""Frozen candidate pool, identical for generation and every experimental arm."""
from kernelfuzz.route_a.compat import operator_pool_snapshot, supported_ops_broad


def snapshot():
    return operator_pool_snapshot(supported_ops_broad() + ['torch.addmm'])

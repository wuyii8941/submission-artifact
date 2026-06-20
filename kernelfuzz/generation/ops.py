"""
OpsForTarget: operator pools indexed by scheduling family.

Each pool is derived from Inductor's lowering registry and dispatch tables.
An operator's scheduling family is determined by the IR type it lowers to:
  relu  -> ir.Pointwise  -> pointwise template
  cumsum -> ir.Scan      -> scan/split_scan template
  mm     -> extern call  -> cuBLAS dispatch

The mapping is a causal chain (op -> IR type -> scheduling), not
independent dimensions.
"""

import random
from typing import Tuple

FAMILY_OPS = {
    'pointwise': {
        'unary': [
            'torch.relu', 'torch.sigmoid', 'torch.tanh', 'torch.exp',
            'torch.neg', 'torch.abs', 'torch.sin', 'torch.cos',
            'torch.reciprocal', 'torch.sqrt', 'torch.log1p',
            'torch.nn.functional.silu', 'torch.nn.functional.gelu',
        ],
        'binary': [
            ('torch.add', '{a}, {b}'),
            ('torch.mul', '{a}, {b}'),
            ('torch.sub', '{a}, {b}'),
            ('torch.div', '{a} + 1e-6, {b}.abs() + 1e-6'),
            ('torch.pow', '{a}.abs() + 1e-3, 2.0'),
            ('torch.maximum', '{a}, {b}'),
            ('torch.minimum', '{a}, {b}'),
        ],
    },
    'reduction': [
        ('sum',  '.sum(dim=-1, keepdim=True)'),
        ('mean', '.mean(dim=-1, keepdim=True)'),
        ('amax', '.amax(dim=-1, keepdim=True)'),
        ('amin', '.amin(dim=-1, keepdim=True)'),
        ('prod', '.prod(dim=-1, keepdim=True)'),
        ('std',  '.std(dim=-1, keepdim=True)'),
        ('var',  '.var(dim=-1, keepdim=True)'),
    ],
    'persistent_reduction': [
        ('sum',  '.sum(dim=-1, keepdim=True)'),
        ('mean', '.mean(dim=-1, keepdim=True)'),
        ('amax', '.amax(dim=-1, keepdim=True)'),
        ('softmax', 'torch.nn.functional.softmax({x}, dim=-1)'),
        ('log_softmax', 'torch.nn.functional.log_softmax({x}, dim=-1)'),
        ('layer_norm', 'torch.nn.functional.layer_norm({x}, [{hidden}])'),
    ],
    'split_scan': [
        ('cumsum', 'torch.cumsum({x}, dim={dim})'),
        ('cumprod', 'torch.cumprod({x}.sigmoid(), dim={dim})'),
    ],
    'scan': [
        ('cumsum', 'torch.cumsum({x}, dim={dim})'),
        ('cumprod', 'torch.cumprod({x}.sigmoid(), dim={dim})'),
    ],
    'extern_gemm': [
        ('matmul', 'torch.matmul({a}, {b})'),
        ('mm', 'torch.mm({a2d}, {b2d})'),
        ('bmm', 'torch.bmm({a3d}, {b3d})'),
        ('linear', 'torch.nn.functional.linear({a}, {w})'),
        ('addmm', 'torch.addmm({bias}, {a2d}, {b2d})'),
    ],
    'extern_conv': [
        ('conv2d', 'torch.nn.functional.conv2d({x4d}, {w_conv})'),
        ('conv1d', 'torch.nn.functional.conv1d({x3d}, {w_conv1d})'),
    ],
}

COMPOSITION_PATTERNS = [
    'linear_chain',
    'reduce_after_pw',
    'branch_merge',
    'with_backward',
    'pool_flatten_reduce',
]


def ops_for_target(family: str) -> list:
    """Return the operator pool for a given scheduling family."""
    return FAMILY_OPS.get(family, FAMILY_OPS['reduction'])


def pointwise_unary_ops() -> list:
    return FAMILY_OPS['pointwise']['unary']


def pointwise_binary_ops() -> list:
    return FAMILY_OPS['pointwise']['binary']

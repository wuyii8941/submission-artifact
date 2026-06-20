"""
SampleConditions: shape, dtype, value, and context sampling near
compiler decision boundaries.

Boundary shapes are derived from Inductor source code:
  - rnumel > 1024: persistent_reduction -> reduction switch
  - scan_dim >= 129: scan -> split_scan threshold
  - fp64 persistent reduction boundary at 513
  - Triton tile masking edges at powers of 2

These are NOT arbitrary: each value corresponds to an exact threshold
in the Inductor scheduler where the compiler switches codegen strategy.
"""

import random


# Adversarial shapes: 2^n +/- 1 to stress Triton tile masking
ADV_BATCH = [1, 3, 7, 15, 17]
ADV_SEQ = [31, 33, 63, 65, 127, 129, 255, 257]
ADV_HIDDEN = [31, 33, 63, 65, 127, 129, 255, 257, 511, 513, 1023]

# Standard shapes for baseline comparison
STD_BATCH = [1, 2, 3, 4, 7]
STD_SEQ = [7, 8, 13, 16, 32, 33, 64, 65, 127]
STD_HIDDEN = [7, 8, 13, 16, 32, 33, 64, 65, 127, 128, 256]

# Inductor codegen decision boundaries: shapes +/-1 from exact thresholds
# where Inductor switches kernel strategy.
BOUNDARY_SHAPES = [
    # persistent_reduction threshold: rnumel > 1024 -> non-persistent
    1022, 1023, 1024, 1025, 1026,
    # Non-persistent reduction block size 2048
    2046, 2047, 2048, 2049, 2050,
    # SplitScan threshold: scan_dim >= 129
    126, 127, 128, 129, 130, 131,
    # fp64 persistent reduction boundary
    511, 512, 513, 514,
    # Cooperative reduction / large reduction
    32766, 32768, 32770,
    # Triton block sizes -- tile masking edges
    16, 32, 33, 64, 65,
]

BOUNDARY_SEQ = [x for x in BOUNDARY_SHAPES if x <= 1024]
BOUNDARY_HIDDEN = [x for x in BOUNDARY_SHAPES if x <= 2048]

# Family-specific boundary shapes from Inductor scheduling thresholds
FAMILY_BOUNDARY_SHAPES = {
    'persistent_reduction': {
        'threshold_dim': [511, 512, 513, 1023, 1024, 1025],
    },
    'reduction': {
        'threshold_dim': [1025, 2047, 2048, 2049, 4096, 8192],
    },
    'split_scan': {
        'threshold_dim': [16383, 16384, 16385, 32768],
    },
    'scan': {
        'threshold_dim': [63, 64, 65, 127, 128, 129, 256, 512],
    },
    'cooperative_reduction': {
        'threshold_dim': [65536, 131072, 262144],
    },
    'tile_mask': [15, 16, 17, 31, 32, 33, 63, 64, 65],
    'fp64_persistent': [511, 512, 513, 514],
}

# Dtype distribution
DTYPES = [
    'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32',
    'torch.float32',  # 50%
    'torch.float64', 'torch.float64',  # 20%
    'torch.bfloat16',  # 15%
    'torch.float16',   # 15%
]
DTYPE_WEIGHTS = [50, 20, 15, 15]
DTYPE_OPTIONS = ['torch.float32', 'torch.float64', 'torch.bfloat16', 'torch.float16']


def sample_shapes(adversarial_ratio=0.0):
    """Sample (batch, seq, hidden) shapes.

    When adversarial_ratio > 0, 20% of adversarial samples pick exactly
    one dimension from codegen decision boundaries.
    """
    if adversarial_ratio > 0 and random.random() < adversarial_ratio:
        if random.random() < 0.20:
            b = random.choice(ADV_BATCH)
            target_dim = random.choice(['seq', 'hidden'])
            if target_dim == 'seq':
                return (b, random.choice(BOUNDARY_SEQ),
                        random.choice(ADV_HIDDEN))
            else:
                return (b, random.choice(ADV_SEQ),
                        random.choice(BOUNDARY_HIDDEN))
        return (random.choice(ADV_BATCH),
                random.choice(ADV_SEQ),
                random.choice(ADV_HIDDEN))
    return (random.choice(STD_BATCH),
            random.choice(STD_SEQ),
            random.choice(STD_HIDDEN))


def pick_boundary_shape(family: str, dtype_str: str) -> int:
    """Select adversarial shape dimension for the target family."""
    if family == 'persistent_reduction':
        if dtype_str == 'float64':
            return random.choice(FAMILY_BOUNDARY_SHAPES['fp64_persistent'])
        return random.choice(FAMILY_BOUNDARY_SHAPES['persistent_reduction']['threshold_dim'])
    elif family == 'reduction':
        return random.choice(FAMILY_BOUNDARY_SHAPES['reduction']['threshold_dim'])
    elif family == 'split_scan':
        return random.choice(FAMILY_BOUNDARY_SHAPES['split_scan']['threshold_dim'])
    elif family == 'scan':
        return random.choice(FAMILY_BOUNDARY_SHAPES['scan']['threshold_dim'])
    elif family == 'cooperative_reduction':
        return random.choice(FAMILY_BOUNDARY_SHAPES['cooperative_reduction']['threshold_dim'])
    else:
        return random.choice(FAMILY_BOUNDARY_SHAPES['tile_mask'])


def sample_dtype():
    """Sample a dtype from the weighted distribution."""
    return random.choices(DTYPE_OPTIONS, weights=DTYPE_WEIGHTS, k=1)[0]


def directed_shapes(family: str):
    """Construct shapes that target a specific kernel family.

    Uses known codegen decision boundaries instead of random sampling.
    """
    if family == 'cooperative_reduction':
        return (1, random.choice([2, 3, 4]), random.choice([8192, 16384, 32768]))
    elif family == 'persistent_reduction':
        return (random.randint(1, 8), random.randint(16, 128),
                random.choice([64, 128, 256, 512]))
    elif family == 'split_scan':
        return (random.randint(1, 4), random.choice([129, 130, 255, 256, 512]),
                random.randint(16, 64))
    elif family == 'scan':
        return (random.randint(1, 4), random.choice([32, 64, 96, 127, 128]),
                random.randint(16, 64))
    elif family == 'extern_gemm':
        return (random.randint(1, 8), random.choice([64, 128, 256]),
                random.choice([64, 128, 256]))
    elif family == 'extern_conv':
        return (random.randint(1, 4), random.choice([16, 32, 64]),
                random.choice([16, 32, 64]))
    else:
        return (random.randint(1, 8),
                random.choice([33, 63, 127, 255]),
                random.choice([33, 63, 127, 255]))

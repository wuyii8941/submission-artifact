#!/usr/bin/env python3
import json, sys, time
t0 = time.time()
import tensorflow as tf
from tensorflow.python.framework import kernels, op_def_registry, ops as tf_ops
from tf_rules import signature_class

tf.function(lambda v: v + 1, jit_compile=True)(tf.ones([2]))

rows = []
for name in sorted(n for n in dir(tf.raw_ops) if not n.startswith('_')):
    od = op_def_registry.get(name)
    if od is None:
        continue
    try:
        kl = kernels.get_registered_kernels_for_op(name).kernel
    except Exception:
        kl = []
    devices, gpu_types = set(), set()
    for kd in kl:
        devices.add(kd.device_type)
        if kd.device_type in ('GPU', 'XLA_GPU_JIT', 'XLA_CPU_JIT'):
            for c in kd.constraint:
                gpu_types |= {tf.dtypes.as_dtype(t).name for t in c.allowed_values.list.type}
    try:
        has_grad = tf_ops._gradient_registry.lookup(name) is not None
    except LookupError:
        has_grad = False
    rows.append({'op': name, 'signature': signature_class(od), 'stateful': od.is_stateful,
                 'devices': sorted(devices), 'gpu_dtypes': sorted(gpu_types), 'has_gradient': has_grad})

from tensorflow.core.protobuf import rewriter_config_pb2
toggles = [f.name for f in rewriter_config_pb2.RewriterConfig.DESCRIPTOR.fields
           if f.enum_type is not None and f.enum_type.name == 'Toggle']
paths = {
    'grappler_optimizers': toggles,
    'xla': 'tf.function(jit_compile=True)',
    'devices': [d.device_type for d in tf.config.list_physical_devices()],
    'gradient': 'tf.GradientTape',
    'mixed_precision_policies': ['float32', 'mixed_float16', 'mixed_bfloat16'],
}
out = {'tf_version': tf.__version__, 'ops': rows, 'execution_paths': paths, 'seconds': round(time.time() - t0, 1)}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'tf_candidates.json', 'w'), indent=0)
print(json.dumps({'tf_version': tf.__version__, 'ops': len(rows), 'seconds': out['seconds']}))

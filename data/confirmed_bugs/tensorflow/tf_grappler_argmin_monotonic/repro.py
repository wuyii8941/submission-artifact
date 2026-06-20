"""TF V24 semantic/churn-guided compiler finding.
seed=240021
pattern=argmin_non_strict_monotonic
category=algebraic_simplification
tags=['grappler', 'argmin', 'non_strict_monotonic']
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
x0 = tf.constant([[0.33710822463035583, -0.1238815188407898, -0.6368741989135742, -0.9302676916122437, -0.5921591520309448, -0.3998347520828247, 0.6292504072189331, -0.930597722530365], [-0.10000000149011612, -0.5, -2.0, 0.0, 0.0, 0.25, 0.25, 0.5], [1.581595778465271, -1.5841461420059204, 1.9434466361999512, -1.4530607461929321, 1.2971471548080444, -0.7073665857315063, 0.33141595125198364, -0.6119456887245178], [-0.25370317697525024, -0.02746782824397087, 2.9270248413085938, -0.7273091673851013, 1.4972139596939087, 1.0012869834899902, -0.9242722988128662, 1.4943376779556274]], dtype=tf.float32)

def fn(x0):
    y = tf.math.floor(x0)
    return tf.cast(tf.argmax(y, axis=-1), tf.float32)

args = [x0]

out_e = fn(*args)
out_t = tf.function(fn, jit_compile=False)(*args)
print("eager:", np.asarray(out_e.numpy()).reshape(-1)[:16])
print("tf.function:", np.asarray(out_t.numpy()).reshape(-1)[:16])
try:
    out_x = tf.function(fn, jit_compile=True)(*args)
    print("xla:", np.asarray(out_x.numpy()).reshape(-1)[:16])
except Exception as e:
    print("xla_error:", str(e)[:500])

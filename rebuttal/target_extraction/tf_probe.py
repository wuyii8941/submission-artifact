#!/usr/bin/env python3
import importlib.util, json, os, sys, time
import numpy as np
import tensorflow as tf

ART = os.path.join(os.environ.get('ARTIFACT', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))), 'kernelfuzz/tensorflow')
spec = importlib.util.spec_from_file_location('graph_gen', f'{ART}/graph_gen.py')
gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)
EXPR = {n: t for v in vars(gg).values() if isinstance(v, list) and v and isinstance(v[0], tuple) and len(v[0]) == 3
        for n, t, _ in v}
EXPR['matmul'] = 'tf.linalg.matmul({x}, {w})'
SKIP = {'Placeholder', 'Identity', 'Const', 'NoOp', 'Shape', 'Pack', 'StridedSlice', 'Reshape', 'Fill',
        'Range', 'Size', 'Rank', 'ConcatV2', 'ZerosLike', 'OnesLike', 'IdentityN'}


def make(name):
    expr = EXPR[name].replace('{x}', 'x').replace('{a}', 'x').replace('{b}', 'y').replace('{w}', 'w')
    f = eval('lambda x, y, w: ' + expr, {'tf': tf})
    return f, ('{b}' in EXPR[name])


def trace(out):
    rows = []
    rng = np.random.default_rng(0)
    for name in sorted(EXPR):
        row = {'op': name}
        try:
            f, binary = make(name)
            x = tf.constant(rng.standard_normal((4, 16)), tf.float32)
            y = tf.constant(rng.standard_normal((4, 16)), tf.float32)
            w = tf.constant(rng.standard_normal((16, 16)), tf.float32)
            g = tf.function(f).get_concrete_function(x, y, w).graph
            raw = sorted({o.type for o in g.get_operations()})
            base = f(x, y, w).numpy()
            x2 = x.numpy().copy(); x2[0, 5] += 0.7
            moved = np.abs(np.nan_to_num(f(tf.constant(x2), y, w).numpy() - base)) > 1e-6
            if base.shape != (4, 16):
                coupling = 'shape_changes'
            elif moved[1:].any():
                coupling = 'cross_row'
            elif moved[0].all():
                coupling = 'row'
            elif moved.sum() <= 1 and (moved.sum() == 0 or moved[0, 5]):
                coupling = 'local'
            else:
                coupling = 'partial_row'
            row.update(status='ok', raw_ops=raw, core_ops=[r for r in raw if r not in SKIP], binary=binary,
                       out_shape=list(base.shape), coupling=coupling)
        except Exception as e:
            row.update(status='error', error=f'{type(e).__name__}: {str(e)[:120]}')
        rows.append(row)
    json.dump(rows, open(out, 'w'), indent=0)


if __name__ == '__main__':
    t0 = time.time()
    tf.get_logger().setLevel('ERROR')
    trace(sys.argv[2])
    print(f'done in {time.time() - t0:.0f}s', flush=True)

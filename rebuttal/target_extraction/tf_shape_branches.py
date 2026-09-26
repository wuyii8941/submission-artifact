#!/usr/bin/env python3
import json, os, re, sys
site = sys.argv[1]
root = os.path.join(site, 'tensorflow/include/tensorflow/core/kernels')
FILES = ['reduction_gpu_kernels.cu.h', 'reduction_ops_common.h', 'reduction_ops_common_gpu.h', 'redux_functor.h']
SIZE = r'(num_rows|num_cols|num_planes|in_size|extent_[xyz]|NumElements\(\)|num_elements\(\)|dim_size\([^)]*\)|ndims\(\))'
COND = re.compile(r'\bif\s*\((.*' + SIZE + r'\s*(==|<=|<|>=|>|!=)\s*([0-9]+|1\s*<<\s*[0-9]+|TF_RED_WARPSIZE).*)\)')
rows = []
for f in FILES:
    path = os.path.join(root, f)
    if not os.path.exists(path):
        continue
    for i, line in enumerate(open(path), 1):
        m = COND.search(line)
        if m:
            rows.append({'file': f, 'line': i, 'condition': m.group(1).strip()[:120],
                         'variable': m.group(2), 'op': m.group(3), 'constant': m.group(4)})
json.dump(rows, open(sys.argv[2], 'w'), indent=1)
consts = sorted({r['constant'] for r in rows})
print(f'{len(rows)} size branches in {len({r["file"] for r in rows})} files; constants: {consts}')
for r in rows:
    print(f"  {r['file']}:{r['line']}  if ({r['condition']})")

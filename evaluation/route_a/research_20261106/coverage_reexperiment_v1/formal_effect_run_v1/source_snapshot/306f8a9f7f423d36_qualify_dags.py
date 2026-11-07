"""Generated-DAG semantic audit, not an effect or discovery experiment."""
import argparse
from collections import Counter
import hashlib
import json
import random
import time
from inventory import ROOT
from common_pool import snapshot
from kernelfuzz.route_a.model import GraphRequest
from kernelfuzz.route_a.torchfuzz_adapter import TorchFuzzAdapter
from dag_generator import generate
from graph_semantics import manifest, restore
from dag_render import function, validate
from dag_binding import bind


def main():
    p = argparse.ArgumentParser(); p.add_argument('--count', type=int, default=24)
    p.add_argument('--seed', type=int, default=202611060)
    p.add_argument('--output', type=str, required=True); args = p.parse_args()
    out = ROOT / args.output; out.mkdir(exist_ok=False)
    pool = snapshot(); adapter = TorchFuzzAdapter(supported_ops=pool['accepted'])
    rows = []; operators = Counter(); errors = Counter(); binding_status = Counter()
    for i in range(args.count):
        seed = args.seed + i; started = time.monotonic()
        row = {'seed': seed, 'excluded_from_discovery_and_effect_metrics': True}
        try:
            graph, provenance = generate(adapter, GraphRequest(seed=seed, supported_ops=tuple(pool['accepted']),
                                         target_id=None, condition=None, entry_strategy='bounded_filter'))
            row['provenance'] = provenance
            record = manifest(graph); code, leaves = function(graph)
            restored = restore(json.loads(json.dumps(record)))
            restored_code, _ = function(restored)
            rng = random.getstate(); repeat, _ = function(graph)
            assert rng == random.getstate(), 'render_modified_python_rng'
            assert code == repeat == restored_code, 'render_not_stable'
            assert manifest(graph) == manifest(restored) == record, 'roundtrip_changed_graph'
            validate(restored, restored_code)
            row.update(status='qualified', graph=record, code=code,
                       checks={'repeat_render': True, 'json_roundtrip': True, 'rng_unchanged': True, 'meta_valid': True})
            operators.update(node.op_name for node in graph.nodes.values())
            # Same generic candidate actions regardless of target presence.
            binding_rows = []
            nonleaves = [nid for nid in graph.get_topological_order() if graph.nodes[nid].input_nodes]
            if nonleaves:
                nid = nonleaves[0]
                for attribute, value in [('dtype', 'torch.float32'), ('dtype', 'torch.float64'),
                                         ('shape', 7), ('layout', 'column_major')]:
                    bound, receipt = bind(graph, nid, attribute, value)
                    if bound is not None:
                        bound_record = manifest(bound)
                        restored_bound = restore(json.loads(json.dumps(bound_record)))
                        assert function(bound)[0] == function(restored_bound)[0], 'bound_roundtrip_changed_code'
                        receipt['roundtrip_passed'] = True
                    binding_status[receipt['status']] += 1
                    binding_rows.append(receipt)
            row['bindings'] = binding_rows
        except Exception as exc:
            row.update(status='generation_exhausted' if hasattr(exc, 'provenance') else 'error',
                       error_type=type(exc).__name__, error=str(exc)[-2000:])
            if hasattr(exc, 'provenance'): row['provenance'] = exc.provenance
        row['elapsed_seconds'] = time.monotonic() - started
        for attempt in row.get('provenance', {}).get('attempt_trace', []):
            if attempt.get('error_type'): errors[attempt['error_type'] + ':' + attempt.get('error','').split(':')[0]] += 1
        rows.append(row)
        (out / f'case_{i:03d}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({'ordinal': i, 'seed': seed, 'status': row['status'], 'elapsed': round(row['elapsed_seconds'], 3)}), flush=True)
    summary = {'requested': args.count, 'status_counts': dict(Counter(row['status'] for row in rows)),
               'observed_generated_registry_operators': dict(operators),
               'attempt_errors': dict(errors), 'binding_status_counts': dict(binding_status),
               'effect_experiment_started': False, 'denominator_committed': False,
               'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                 ('graph_semantics.py','dag_render.py','dag_generator.py','dag_binding.py','qualify_dags.py','frozen_codegen.py')}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__': main()

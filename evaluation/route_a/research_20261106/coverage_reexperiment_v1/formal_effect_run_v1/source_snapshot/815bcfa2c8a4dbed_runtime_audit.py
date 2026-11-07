"""CUDA execution/FX/source-evidence audit of a serialized general DAG.

Evidence collection only: no percentage, denominator, or arm-effect claim.
Eager/input generation run outside source-observation scope. Source events
retain raw FX provenance; ambiguous events must remain uncounted later.
"""
import argparse
from collections import Counter
import dis
import hashlib
import json
from pathlib import Path
import sys
import time
from inventory import ROOT
from frozen_codegen import encode
from graph_semantics import restore, manifest
from common_domain import validate_graph, descriptor
from opcode_offsets import prefix_aliases
from dag_render import function, scalar_value, storage_size
from torchfuzz.tensor_fuzzer import ScalarSpec
from runtime_oracle_v4 import compare
from value_domain import ValueDomain, ValueDomainViolation
import torch
from torchfuzz.tensor_fuzzer import TensorSpec

AUDIT_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def source_catalog():
    inventory_path = ROOT / 'forward_source_inventory_v1/inventory.json'
    inventory = json.loads(inventory_path.read_text())
    result = {}
    for row in inventory['sources']:
        source = dict(row['source'])
        source['origin_kind'] = row['kind']
        if 'parent_identity' in row: source['parent_identity'] = row['parent_identity']
        key = (source['source'],source['first_line'],source['code_sha256'])
        if hashlib.sha256(Path(key[0]).read_bytes()).hexdigest() != source['source_sha256']:
            raise ValueError('forward_source_inventory_outdated')
        result[key] = {'source':source,'registrations':row['registrations']}
    return result


class SourceAudit:
    def __init__(self, catalog):
        self.catalog = catalog; self.paths = {key[0] for key in catalog}
        self.cache = {}; self.offset_aliases = {}; self.frames = {}; self.calls = []; self.hits = Counter()
    def trace(self, frame, event, arg):
        if frame.f_code.co_filename not in self.paths: return None
        code = frame.f_code
        if code not in self.cache:
            key = (code.co_filename, code.co_firstlineno, hashlib.sha256(code.co_code).hexdigest())
            self.cache[code] = key if key in self.catalog else None
            if key in self.catalog: self.offset_aliases[code] = prefix_aliases(code)
        key = self.cache[code]
        if key is None: return None
        if event == 'call':
            from torch.fx.traceback import get_current_meta
            metadata = get_current_meta()
            provenance = {name: repr(metadata[name]) for name in ('from_node','original_aten','source_fn_stack','stack_trace') if name in metadata}
            if 'original_aten' in metadata:
                provenance['original_aten_id'] = str(metadata['original_aten'])
            if 'from_node' in metadata:
                try:
                    provenance['from_node_structured'] = [source.to_dict() for source in metadata['from_node']]
                except Exception as exc:
                    provenance['structured_provenance_error'] = type(exc).__name__ + ':' + str(exc)
            argument_facts = []
            count = code.co_argcount + code.co_kwonlyargcount
            for name in code.co_varnames[:count]:
                value = frame.f_locals.get(name)
                if isinstance(value,torch.Tensor):
                    argument_facts.append({'name':name,'kind':'tensor','device_type':value.device.type,
                                           'dtype':str(value.dtype),'is_complex':value.dtype.is_complex})
                elif isinstance(value,complex):
                    argument_facts.append({'name':name,'kind':'python_complex','is_complex':True})
            row = {'source_identity': list(key), 'fx_provenance': provenance, 'hit_outcomes': [],
                   'entry_argument_facts':argument_facts}
            self.calls.append(row); self.frames[id(frame)] = (row, None)
            frame.f_trace_opcodes = True; frame.f_trace_lines = False
        elif event == 'opcode':
            row, previous = self.frames.get(id(frame), ({}, None)); raw_now = frame.f_lasti
            now = self.offset_aliases[code].get(raw_now, raw_now)
            outcomes = self.catalog[key]['source']['conditional_outcomes']
            if previous is not None and [previous, raw_now] in outcomes:
                row['hit_outcomes'].append([previous, raw_now]); self.hits[(key, previous, raw_now)] += 1
            self.frames[id(frame)] = (row, now if any(edge[0] == now for edge in outcomes) else None)
        elif event == 'exception':
            if id(frame) in self.frames:
                row, _ = self.frames[id(frame)]; self.frames[id(frame)] = (row, None)
        elif event == 'return':
            self.frames.pop(id(frame), None)
        return self.trace
    def __enter__(self):
        self.prior = sys.gettrace()
        if self.prior is not None: raise ValueError('trace_already_active')
        sys.settrace(self.trace); return self
    def __exit__(self, *args): sys.settrace(self.prior)
    def receipt(self):
        return {'scope': 'current_thread_only_entry_and_nested_source_audit_not_primary_coverage',
                'observer_schema': 4,
                'calls': self.calls, 'distinct_source_outcomes': len(self.hits),
                'catalog': list(self.catalog.values()), 'ambiguous_events_not_counted_as_attributed': True}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    row = json.loads(args.case.read_text()); graph = restore(row['graph'])
    result = {'case': str(args.case.resolve()), 'graph_sha256': manifest(graph)['sha256'],
              'control_only': True, 'effect_experiment_started': False, 'primary_denominator_committed': False,
              'status': 'starting', 'seed': row['seed'], 'stages': []}
    observer = None; value_screen = None
    def stage(name):
        result['stages'].append({'stage': name, 'elapsed_seconds': time.monotonic()-started})
        json_write(args.output/'receipt.json', result)
        print(name, flush=True)
    try:
        import torch._inductor.config as fixed_config
        if fixed_config.coordinate_descent_tuning is not False:raise ValueError('compiler_config_outside_common_domain')
        validate_graph(graph); result['common_domain'] = descriptor()
        torch.set_num_threads(1)
        result['torch'] = torch.__version__; result['commit'] = torch.version.git_version
        result['device'] = torch.cuda.get_device_name(); result['capability'] = list(torch.cuda.get_device_capability())
        source, leaves = function(graph); (args.output/'program.py').write_text(source)
        namespace = {'torch': torch}; exec(compile(source, '<frozen-general-dag-runtime>', 'exec'), namespace)
        torch.manual_seed(row['seed']); actual_args = []; arg_facts = []
        stage('inputs')
        for nid in leaves:
            spec = graph.nodes[nid].output_spec
            if isinstance(spec, TensorSpec):
                storage = torch.empty((storage_size(spec),), dtype=spec.dtype, device='cuda')
                if spec.dtype.is_floating_point or spec.dtype.is_complex: storage.normal_()
                else: storage.random_(0,2)
                value = storage.as_strided(spec.size, spec.stride)
                raw = storage.view(torch.uint8).cpu().numpy().tobytes()
                arg_facts.append({'node_id': nid, 'size': list(value.shape), 'stride': list(value.stride()),
                                  'dtype': str(value.dtype), 'storage_sha256': hashlib.sha256(raw).hexdigest()})
            else:
                value = scalar_value(spec); arg_facts.append({'node_id': nid, 'value': encode(value)})
            actual_args.append(value)
        result['actual_inputs'] = arg_facts
        stage('eager')
        value_screen = ValueDomain()
        eager_probe = source.rsplit('    return ',1)[0] + '    return locals()\n'
        eager_namespace = {'torch':torch}
        exec(compile(eager_probe,'<frozen-general-dag-eager-spec-audit>','exec'),eager_namespace)
        with torch.device('cuda'), value_screen:
            eager_values = eager_namespace['fuzzed_program'](*actual_args)
        actual_node_facts = {}
        for nid in graph.get_topological_order():
            expected = graph.nodes[nid].output_spec; value = eager_values['var_'+nid]
            if isinstance(expected,TensorSpec):
                if not isinstance(value,torch.Tensor) or tuple(value.shape)!=expected.size or value.dtype!=expected.dtype:
                    raise ValueError('cuda_eager_node_spec_mismatch:'+nid)
                actual_node_facts[nid] = {'kind':'tensor','size':list(value.shape),'dtype':str(value.dtype),
                                         'stride':list(value.stride()),'contiguous':value.is_contiguous()}
            elif isinstance(expected,ScalarSpec):
                scalar_type = complex if expected.dtype.is_complex else float if expected.dtype.is_floating_point else bool if expected.dtype==torch.bool else int
                if type(value) is not scalar_type:
                    raise ValueError('cuda_eager_scalar_type_mismatch:'+nid)
                actual_node_facts[nid] = {'kind':'python_scalar','python_type':type(value).__name__}
        result['actual_eager_node_facts'] = actual_node_facts
        result['all_node_eager_specs_verified'] = True
        reference = eager_values['var_'+graph.root_node_id]
        torch.cuda.synchronize()
        catalog = source_catalog(); observer = SourceAudit(catalog)
        def backend(gm, example_inputs):
            result['dynamo_graph_id'] = id(gm.graph)
            result['dynamo_fx'] = [{'name': node.name, 'op': node.op, 'target': str(node.target),
                                   'metadata': {name: repr(node.meta[name]) for name in
                                                ('from_node','original_aten','source_fn_stack','stack_trace') if name in node.meta}}
                                  for node in gm.graph.nodes]
            (args.output/'dynamo_fx.py').write_text(gm.code)
            from torch._inductor.compile_fx import compile_fx
            return compile_fx(gm, example_inputs)
        stage('compile')
        with torch.device('cuda'), observer:
            compiled = torch.compile(namespace['fuzzed_program'], backend=backend, fullgraph=True, dynamic=False)
            actual = compiled(*actual_args)
            torch.cuda.synchronize()
        stage('oracle')
        result['oracle'] = compare(reference, actual)
        result['status'] = 'executed'
    except ValueDomainViolation as exc:
        result.update(status='value_domain_rejected', error_type=type(exc).__name__, value_domain_violation=exc.evidence)
    except Exception as exc:
        result.update(status='execution_error', error_type=type(exc).__name__, error=str(exc)[-6000:])
    finally:
        if value_screen is not None: result['value_domain_screen'] = value_screen.receipt()
        if observer is not None: json_write(args.output/'source_evidence.json', observer.receipt())
        result['elapsed_seconds'] = time.monotonic() - started
        result['source_sha256'] = AUDIT_SOURCE_SHA256
        result['source_unchanged_during_run'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == AUDIT_SOURCE_SHA256
        if not result['source_unchanged_during_run']: result['status'] = 'invalidated_source_changed_during_run'
        json_write(args.output/'receipt.json',result)
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds')},indent=2))
if __name__ == '__main__': main()

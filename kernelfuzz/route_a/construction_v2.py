"""Opt-in condition-aware random construction with bounded, recorded repair.

The native/legacy paths are unchanged. These builders sample a constrained DAG,
not a post-hoc filter of unconstrained native graphs. Repaired candidates become
new fuzz programs; the unchanged oracle compares eager and compiled execution of
that same repaired program, never the unrepaired program's semantics.
"""
from dataclasses import asdict, dataclass, replace
from functools import wraps
import hashlib
import json
import random
import time

from .adapter_v2 import ConfiguredAdapter, ConfiguredGraphRequest
from .conditions_v2 import (Applicability, Capability, ConditionRejected, Dimension, Gate,
    SUPPORTED, bind, check_storage, propagate_meta, reject, static_gate, tensor_records)
from .graph_utils import _contiguous_stride, find_matching_nodes, graph_manifest, validate_graph_specs
from .model import GenerationResult, TestCase
from .targets import target_map

VERSION='condition-aware-construction-v1'
UNARY=('torch.sin','torch.cos','torch.tanh','torch.sigmoid')
BINARY=('add','sub','mul')
QUALIFIED_TARGETS=('pointwise','reduction_sum')
ALLOWED=set(UNARY+BINARY+('torch.sum','arg','constant'))


@dataclass(frozen=True)
class ConstructionRequest(ConfiguredGraphRequest):
    construction_system: str = 'native_v2'
    repair_attempts: int = 3
    repair_budget_seconds: float = 5.

    def to_dict(self):
        data=super().to_dict()
        if self.construction_system!='native_v2':
            data.update(construction_system=self.construction_system,repair_attempts=self.repair_attempts,
                        repair_budget_seconds=self.repair_budget_seconds)
        return data


@dataclass(frozen=True)
class Constraints:
    target: str
    dimension: str
    transformation: str
    dtype: str
    device: str
    shape: tuple
    operators: tuple
    grad_mode: str
    alias_pair: bool
    schema: str = VERSION

    def to_dict(self):
        return asdict(self)


def compatible(target,condition,capability):
    """Filter the candidate domain before scheduler selection/construction."""
    if target not in QUALIFIED_TARGETS:
        return Gate(Applicability.INAPPLICABLE,True,'disabled_extension',('target_condition_builder_not_qualified',))
    if condition.dimension not in SUPPORTED or condition.value not in SUPPORTED[condition.dimension]:
        return Gate(Applicability.INAPPLICABLE,True,'disabled_extension',('condition_builder_not_qualified',))
    if condition.dimension==Dimension.DYNAMIC and (len(condition.family)<3 or len(set(condition.family))!=len(condition.family)
                                                   or any(type(x) is not int or x<=1 for x in condition.family)):
        return Gate(Applicability.INAPPLICABLE,False,'illegal_generation',('dynamic_requires_three_distinct_runtime_shapes',))
    if capability.device not in ('cpu','cuda'):
        return Gate(Applicability.INAPPLICABLE,True,'expected_capability_rejection',('device_not_qualified',))
    return Gate()


def constraints_for(seed,target,condition,capability):
    gate=compatible(target,condition,capability)
    if gate.applicability!=Applicability.APPLICABLE:raise ConditionRejected(gate)
    # Include dimension/value in the stream key but not arm or wall time. The
    # two feedback arms therefore sample the same per-seed candidate menu.
    key=json.dumps([VERSION,seed,target,condition.to_dict()],sort_keys=True)
    rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest(),16))
    rank=rng.choice((2,2,3))
    shape=tuple(rng.randint(2,8) for _ in range(rank-1))+(rng.choice((8,12,16,20,24,28,32,40)),)
    dtype=rng.choice(('torch.float32','torch.float32','torch.float32','torch.float64'))
    grad=condition.value if condition.dimension==Dimension.AUTOGRAD else 'none'
    return Constraints(target,condition.dimension.value,condition.value,dtype,capability.device,shape,
                       UNARY+BINARY+('torch.sum',),grad,condition.value=='shared_inputs'),rng


def _build_dag(constraints,rng,minimum_layers=1):
    import torch
    from torchfuzz.ops_fuzzer import OperationGraph,OperationNode
    from torchfuzz.tensor_fuzzer import TensorSpec
    shape=constraints.shape;dtype=getattr(torch,constraints.dtype.split('.')[-1])
    spec=TensorSpec(shape,_contiguous_stride(shape),dtype)
    nodes={name:OperationNode(name,'arg',[],spec,[],0) for name in ('input_a','input_b')}
    def add(name,op,inputs,output=spec):
        nodes[name]=OperationNode(name,op,[nodes[n].output_spec for n in inputs],output,list(inputs),
                                  1+max(nodes[n].depth for n in inputs))
        return name
    previous=add('mix','mul',('input_a','input_b'))
    # Random operator choices, input branches, rank, extents and DAG depth;
    # external input lineage and bounded smooth arithmetic are invariant.
    for i in range(rng.randint(minimum_layers,3)):
        op=rng.choice(UNARY+BINARY)
        inputs=(previous,) if op in UNARY else (previous,rng.choice(('input_a','input_b','mix')))
        previous=add('context_'+str(i),op,inputs)
    if constraints.target=='reduction_sum':
        output=TensorSpec(shape[:-1],_contiguous_stride(shape[:-1]),dtype)
        add('selected_target','torch.sum',(previous,),output)
    else:
        op=rng.choice(UNARY+('add','mul'))
        inputs=(previous,) if op in UNARY else (previous,rng.choice(('input_a','input_b')))
        add('selected_target',op,inputs)
    return OperationGraph(nodes,'selected_target',nodes['selected_target'].output_spec)


def build_layout(constraints,rng):
    if len(constraints.shape)<2 or sum(x>1 for x in constraints.shape)<2:reject('layout_builder_requires_two_nonunit_dimensions')
    return _build_dag(constraints,rng)


def build_dynamic(constraints,rng):
    # Same-shape pointwise edges and an explicit final reduction make symbolic
    # equalities complete; no data-dependent factory or Tensor.item() is emitted.
    return _build_dag(constraints,rng)


def build_autograd(constraints,rng):
    if constraints.dtype not in ('torch.float32','torch.float64'):reject('autograd_builder_requires_fp32_or_fp64')
    return _build_dag(constraints,rng,minimum_layers=2)


def build_alias(constraints,rng):
    # Both external arguments always have matching specs and reach the target.
    # Existing v2 binding creates actual view/mutation operations and preserves
    # shared storage in the runtime input builder and functionalization oracle.
    return _build_dag(constraints,rng)


BUILDERS={Dimension.LAYOUT:build_layout,Dimension.DYNAMIC:build_dynamic,
          Dimension.AUTOGRAD:build_autograd,Dimension.ALIAS:build_alias}


def scoped_codegen_rng(function):
    """TorchFuzz emits random alpha kwargs; contain that legacy global RNG."""
    @wraps(function)
    def scoped(adapter,request):
        state=random.getstate()
        try:
            seed=json.dumps(request.to_dict(),sort_keys=True)
            random.seed(int(hashlib.sha256(seed.encode()).hexdigest(),16))
            return function(adapter,request)
        finally:
            random.setstate(state)
    return scoped


def require_target(graph,target):
    definition=target_map().get(target)
    matches=find_matching_nodes(graph,definition.operation_names,exact=True) if definition else []
    if not matches:reject('entry_not_found','illegal_generation',False)
    return matches[-1]


def repair_graph(graph,target,condition,constraints,capability=None):
    """Propagate constructor contracts; unsupported operations require resampling.

    Target presence is checked before *any* mutation. Replacing a constant with
    an external fuzz input is an explicit candidate repair, retained in evidence.
    """
    import torch
    from torchfuzz.ops_fuzzer import OperationGraph,OperationNode
    from torchfuzz.tensor_fuzzer import TensorSpec
    selected=require_target(graph,target)
    capability=capability or Capability('cpu')
    capability_gate=static_gate(graph,condition,capability)
    if capability_gate.category=='expected_capability_rejection':
        raise ConditionRejected(capability_gate)
    before=graph_manifest(graph);repairs=[];nodes={}
    dtype=getattr(torch,constraints.dtype.split('.')[-1])
    for nid in graph.get_topological_order():
        node=graph.nodes[nid]
        if node.op_name not in ALLOWED:reject('operator_not_in_condition_contract:'+node.op_name)
        op=node.op_name
        if op=='constant':
            op='arg';repairs.append({'node':nid,'reason':'constant_input_materialized','before_op':'constant','after_op':'arg'})
        if not node.input_nodes:
            spec=TensorSpec(constraints.shape,_contiguous_stride(constraints.shape),dtype)
            if spec!=node.output_spec:repairs.append({'node':nid,'reason':'input_shape_dtype_layout_contract',
                'before':str(node.output_spec),'after':str(spec)})
        else:
            if any(p not in nodes for p in node.input_nodes):reject('missing_input_node','illegal_generation',False)
            shape=tuple(nodes[node.input_nodes[0]].output_spec.size)
            expected_arity=1 if op in UNARY or op=='torch.sum' else 2
            if len(node.input_nodes)!=expected_arity:reject('operator_arity_not_repairable','illegal_generation',False)
            if op=='torch.sum':shape=shape[:-1]
            elif any(tuple(nodes[p].output_spec.size)!=shape for p in node.input_nodes):
                reject('shape_relation_not_repairable','illegal_generation',False)
            spec=TensorSpec(shape,_contiguous_stride(shape),dtype)
            if spec!=node.output_spec:repairs.append({'node':nid,'reason':'output_shape_dtype_propagation',
                'before':str(node.output_spec),'after':str(spec)})
        nodes[nid]=replace(node,op_name=op,output_spec=spec,input_specs=[nodes[p].output_spec for p in node.input_nodes])
    external=sorted(n for n,x in nodes.items() if not x.input_nodes and x.op_name!='constant')
    if not external:reject('no_tensor_input_lineage')
    if condition.value=='shared_inputs' and len(external)<2:
        # Introduce a real paired argument before the already-present target.
        old=external[0];other='repair_pair_input';mix='repair_pair_mix';spec=nodes[old].output_spec
        if other in nodes or mix in nodes:reject('repair_node_collision','illegal_generation',False)
        nodes[other]=OperationNode(other,'arg',[],spec,[],0)
        nodes[mix]=OperationNode(mix,'mul',[spec,spec],spec,[old,other],1)
        for nid,node in list(nodes.items()):
            if nid not in (old,other,mix) and old in node.input_nodes:
                parents=[mix if p==old else p for p in node.input_nodes]
                nodes[nid]=replace(node,input_nodes=parents,input_specs=[nodes[p].output_spec for p in parents])
        repairs.append({'node':mix,'reason':'shared_inputs_requires_matching_leaf_specs','new_argument':other})
    repaired=OperationGraph(nodes,graph.root_node_id,nodes[graph.root_node_id].output_spec)
    repaired,propagation=propagate_meta(repaired)
    ok,reason=validate_graph_specs(repaired)
    if not ok:reject(reason,'illegal_generation',False)
    contracts=[]
    for nid,node in sorted(repaired.nodes.items()):
        contracts.append({'node':nid,'shape':list(node.output_spec.size),'stride':list(node.output_spec.stride),
            'dtype':str(node.output_spec.dtype),'device':constraints.device,
            'requires_grad':constraints.grad_mode!='none' and not node.input_nodes,
            'rule':'meta_shape_dtype_stride_and_request_device_grad_contract'})
    return repaired,selected,{'before_manifest':before,'after_manifest':graph_manifest(repaired),
                             'repairs':repairs,'propagation':propagation,'contracts':contracts}


@scoped_codegen_rng
def generate_constructed(adapter,request):
    from .execution_v2 import render_case
    started=time.monotonic()
    if request.condition_system!='v2' or request.condition_v2 is None or request.condition is not None:
        raise ValueError('condition_aware_construction_requires_explicit_typed_v2_condition')
    if not 1<=request.repair_attempts<=3 or not 0<request.repair_budget_seconds<=5:
        raise ValueError('repair_limit_exceeds_registered_bound')
    capability=Capability.detect(request.runtime_device)
    initial=compatible(request.target_id,request.condition_v2,capability)
    if initial.applicability!=Applicability.APPLICABLE:
        return GenerationResult('invalid',time.monotonic()-started,0,failure_reason=initial.category,
                                failure_detail=';'.join(initial.reasons),metadata={'gate':initial.to_dict(),'attempt_trace':[]})
    trace=[];last=None
    for attempt in range(1,request.repair_attempts+1):
        if time.monotonic()-started>=request.repair_budget_seconds:break
        phase='construct';begin=time.monotonic();construction=None
        try:
            constraints,rng=constraints_for(request.seed*1009+attempt-1,request.target_id,request.condition_v2,capability)
            raw=adapter.build_candidate(constraints,rng,request.condition_v2)
            phase='require_target';require_target(raw,request.target_id)
            phase='repair';prepared,selected,construction=repair_graph(raw,request.target_id,request.condition_v2,constraints,capability)
            phase='bind';graph,evidence=bind(prepared,selected,request.condition_v2,capability)
            evidence.propagation.extend(construction['contracts'])
            phase='render';code=render_case(graph,request.seed,request.condition_v2,evidence,request.runtime_device)
            if time.monotonic()-started>=request.repair_budget_seconds:
                reject('bounded_repair_deadline','generation_timeout',False)
            manifest=graph_manifest(graph)
            manifest.update(entry_node=selected,original_full_signature=graph_manifest(prepared)['full_signature'],
                original_structure_signature=graph_manifest(prepared)['structure_signature'],condition_v2=evidence.to_dict(),
                construction={'version':VERSION,'constraints':constraints.to_dict(),**construction})
            test_id=hashlib.sha256((code+json.dumps(request.to_dict(),sort_keys=True)).encode()).hexdigest()[:24]
            case=TestCase(test_id,code,manifest,[evidence.to_dict()],{'seed':request.seed,'condition_system':'v2',
                'construction_system':VERSION,'runtime_device':request.runtime_device},
                {'generator':'condition-aware-random-DAG','request':request.to_dict(),'attempt':attempt},graph)
            trace.append({'attempt':attempt,'phase':'admitted','gate':Gate().to_dict(),
                'elapsed_seconds':time.monotonic()-begin,'repairs':construction['repairs']})
            return GenerationResult('valid',time.monotonic()-started,attempt,case,metadata={'attempt_trace':trace,
                'construction_system':VERSION,'repair_budget_seconds':request.repair_budget_seconds})
        except ConditionRejected as exc:
            last=exc.gate
        except Exception as exc:
            last=Gate(Applicability.INAPPLICABLE,False,'illegal_generation',(type(exc).__name__+':'+str(exc),))
        trace.append({'attempt':attempt,'phase':phase,'gate':last.to_dict(),'elapsed_seconds':time.monotonic()-begin,
                      'repairs':construction['repairs'] if construction else []})
    if not last or time.monotonic()-started>=request.repair_budget_seconds:
        last=Gate(Applicability.INAPPLICABLE,False,'generation_timeout',('bounded_repair_deadline',))
    return GenerationResult('invalid',time.monotonic()-started,len(trace),failure_reason=last.category,
        failure_detail=';'.join(last.reasons),metadata={'attempt_trace':trace,'gate':last.to_dict(),
            'construction_system':VERSION,'repair_budget_seconds':request.repair_budget_seconds})


class ConstructionAdapter(ConfiguredAdapter):
    def build_candidate(self,constraints,rng,condition):
        return BUILDERS[condition.dimension](constraints,rng)

    def generate(self,request):
        mode=getattr(request,'construction_system','native_v2')
        if mode=='native_v2':return super().generate(request)
        if mode!='condition_aware_v1':raise ValueError('unknown_construction_system')
        return generate_constructed(self,request)

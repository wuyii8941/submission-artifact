"""Serializable per-node operator state; codegen consumes only a private copy.

Capture immediately after fuzz_inputs_specs, before generating child nodes.
This freezes stored parameters and render RNG. It does not certify that a
shape/dtype binding preserves parameters chosen conditionally by codegen;
those transformations still need a semantic guard and exact meta validation.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
from inventory import ROOT
import torch
from torchfuzz.operators import get_operator
from torchfuzz.tensor_fuzzer import TensorSpec,ScalarSpec

def encode(value):
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:return {'tag':'float','value':value.hex()}
    if type(value) is complex:return {'tag':'complex','real':encode(value.real),'imag':encode(value.imag)}
    if isinstance(value,torch.dtype):return {'tag':'dtype','name':str(value).removeprefix('torch.')}
    if isinstance(value,TensorSpec):return {'tag':'TensorSpec','size':list(value.size),'stride':list(value.stride),'dtype':encode(value.dtype)}
    if isinstance(value,ScalarSpec):return {'tag':'ScalarSpec','dtype':encode(value.dtype),'constant':encode(value.constant)}
    if type(value) in (list,tuple):return {'tag':type(value).__name__,'values':[encode(v) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):return {'tag':'dict','values':{k:encode(v) for k,v in value.items()}}
    raise TypeError('unsupported_operator_state:'+type(value).__name__)

def decode(value):
    if not isinstance(value,dict):return value
    tag=value['tag']
    if tag=='float':return float.fromhex(value['value'])
    if tag=='complex':return complex(decode(value['real']),decode(value['imag']))
    if tag=='dtype':
        result=getattr(torch,value['name']);assert isinstance(result,torch.dtype);return result
    if tag=='TensorSpec':return TensorSpec(tuple(value['size']),tuple(value['stride']),decode(value['dtype']))
    if tag=='ScalarSpec':return ScalarSpec(decode(value['dtype']),decode(value['constant']))
    if tag=='list':return [decode(v) for v in value['values']]
    if tag=='tuple':return tuple(decode(v) for v in value['values'])
    if tag=='dict':return {k:decode(v) for k,v in value['values'].items()}
    raise ValueError('unknown_state_tag:'+tag)

def digest(payload):return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def freeze(operator,registry_key,render_seed):
    fn=operator.codegen.__func__;path=Path(inspect.getsourcefile(fn))
    state={k:v for k,v in operator.__dict__.items() if k!='get_weight'}
    payload={'registry_key':registry_key,'class':type(operator).__module__+'.'+type(operator).__qualname__,
        'state':encode(state),'render_seed':int(render_seed),'codegen_source':str(path),
        'codegen_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'excluded_operational_attributes':['get_weight'] if 'get_weight' in operator.__dict__ else []}
    return {'payload':payload,'sha256':digest(payload)}

def render(snapshot,output_name,input_names,output_spec):
    p=snapshot['payload']
    if digest(p)!=snapshot['sha256']:raise ValueError('operator_snapshot_identity_changed')
    original=get_operator(p['registry_key'])
    if original is None or type(original).__module__+'.'+type(original).__qualname__!=p['class']:raise ValueError('operator_class_changed')
    fn=original.codegen.__func__;path=Path(inspect.getsourcefile(fn))
    if str(path)!=p['codegen_source'] or hashlib.sha256(path.read_bytes()).hexdigest()!=p['codegen_source_sha256']:raise ValueError('codegen_source_changed')
    op=copy.deepcopy(original);op.__dict__.clear();op.__dict__.update(decode(p['state']))
    rng=random.getstate()
    try:
        random.seed(p['render_seed'])
        return op.codegen(output_name,list(input_names),output_spec)
    finally:random.setstate(rng)

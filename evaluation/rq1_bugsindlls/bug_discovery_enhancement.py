# -*- coding: utf-8 -*-
import random
import sys
import traceback
import torch

# Stubs to satisfy the main script's import requirements
class EdgeCaseInjector: pass
class DtypeSweepMutator: pass
class CompileWrapperCorpus: 
    def record_bug(self, mode): pass

class BugDiscoveryEnhancer:
    def __init__(self, api_name, enable_edge_case=True, enable_dtype_sweep=True, enable_compile_wrapper=True, **kwargs):
        self.api_name = api_name
        self.enable_compile_wrapper = enable_compile_wrapper
        self.compile_corpus = CompileWrapperCorpus()
        self.stats = {"target_117026_list_iter": 0, "target_116463_itertools": 0, "target_116609_bfloat16": 0, "target_dynamic_shapes": 0}

    def apply(self, api, iteration):
        # Keep the mutation controller untouched; the core logic lives in code generation below.
        return []

    def get_compile_test(self, api):
        # 40% probability of triggering a targeted-sniper test
        if not self.enable_compile_wrapper or random.random() > 0.4:
            return None
            
        original_code = api.to_code()
        roll = random.random()
        
        # Roulette over historical bug templates
        if roll < 0.25:
            # Sniper target 1: Issue #117026 (ListIteratorVariable has no type)
            mode_name = "target_117026_list_iter"
            tail_logic = """
    out = locals().get('results', dict())
    # Deliberately construct a list-iteration scenario
    collected = []
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            collected.append(v)
    # Trigger the iterator
    return [x * 2 for x in collected] if collected else out
"""
            compile_decorator = 'torch.compile(backend="inductor", fullgraph=False)'

        elif roll < 0.50:
            # Sniper target 2: Issue #116463 (missing support for itertools.chain)
            mode_name = "target_116463_itertools"
            tail_logic = """
    import itertools
    out = locals().get('results', dict())
    tensor_list = [v for v in out.values() if isinstance(v, torch.Tensor)]
    if len(tensor_list) >= 2:
        # Deliberately inject an itertools.chain construct
        chained = list(itertools.chain.from_iterable([tensor_list[0].size(), tensor_list[1].size()]))
        return chained
    return out
"""
            compile_decorator = 'torch.compile(backend="inductor", fullgraph=False)'

        elif roll < 0.75:
            # Sniper target 3: Issue #116609 and precision-class bugs
            mode_name = "target_116609_bfloat16"
            tail_logic = """
    out = locals().get('results', dict())
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            # Deliberately inject a BFloat16 conversion
            return torch.nn.functional.relu(v.to(torch.bfloat16))
    return out
"""
            compile_decorator = 'torch.compile(backend="inductor")'

        else:
            # Sniper target 4: generic Inductor dynamic-shapes bugs
            mode_name = "target_dynamic_shapes"
            tail_logic = "    return locals().get('results', dict())"
            compile_decorator = 'torch.compile(backend="inductor", fullgraph=False, dynamic=True)'

        self.stats[mode_name] += 1

        # Assemble the final payload
        payload = f"""
import torch
import sys
import traceback
import torch._dynamo

# Silence log noise
torch._dynamo.config.suppress_errors = True

# 1. Sanity probe: ensure the base API arguments are valid.
def run_eager():
{self._indent(original_code)}
    pass

try:
    run_eager()
except Exception:
    sys.exit(0)

# 2. Targeted-sniper function
def target_func():
{self._indent(original_code)}
{self._indent(tail_logic)}

# 3. Capture deep bugs
try:
    compiled_fn = {compile_decorator}(target_func)
    out = compiled_fn()
except Exception as e:
    err_str = str(e).lower()
    
    # Strict filter: only real crashes count; drop Dynamo's normal fallback notices.
    if "not supported" in err_str and "itertools" not in err_str: 
        sys.exit(0)
    if "graph break" in err_str or "recompilation" in err_str:
        sys.exit(0)
        
    # Emit the stack-trace error to stderr; the outer framework will capture it.
    print(f"COMPILE_CRASH_DETECTED:\\n{{traceback.format_exc()}}", file=sys.stderr)
    sys.exit(139)
"""
        return (payload, mode_name)

    def _indent(self, text):
        return "\n".join(["    " + line for line in text.split("\n")])

    def get_stats(self):
        return self.stats

    def print_stats(self):
        print(f"  [BugDiscovery] 🎯 Template Triggers:")
        for k, v in self.stats.items():
            print(f"      - {k}: {v}")
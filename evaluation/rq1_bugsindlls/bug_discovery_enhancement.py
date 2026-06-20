# -*- coding: utf-8 -*-
import random
import sys
import traceback
import torch

# 占位符，满足主脚本的 Import 需求
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
        # 保持变异控制器原样，核心逻辑全部集中在下方的代码生成
        return []

    def get_compile_test(self, api):
        # 40% 概率触发定向狙击测试
        if not self.enable_compile_wrapper or random.random() > 0.4:
            return None
            
        original_code = api.to_code()
        roll = random.random()
        
        # 🎯 历史 Bug 模板轮盘赌
        if roll < 0.25:
            # 狙击目标 1：Issue #117026 (ListIteratorVariable has no type)
            mode_name = "target_117026_list_iter"
            tail_logic = """
    out = locals().get('results', dict())
    # 定向制造 List Iteration 场景
    collected = []
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            collected.append(v)
    # 触发 iterator
    return [x * 2 for x in collected] if collected else out
"""
            compile_decorator = 'torch.compile(backend="inductor", fullgraph=False)'

        elif roll < 0.50:
            # 狙击目标 2：Issue #116463 (missing support for itertools.chain)
            mode_name = "target_116463_itertools"
            tail_logic = """
    import itertools
    out = locals().get('results', dict())
    tensor_list = [v for v in out.values() if isinstance(v, torch.Tensor)]
    if len(tensor_list) >= 2:
        # 定向注入 itertools.chain 结构
        chained = list(itertools.chain.from_iterable([tensor_list[0].size(), tensor_list[1].size()]))
        return chained
    return out
"""
            compile_decorator = 'torch.compile(backend="inductor", fullgraph=False)'

        elif roll < 0.75:
            # 狙击目标 3：Issue #116609 及精度类 Bug
            mode_name = "target_116609_bfloat16"
            tail_logic = """
    out = locals().get('results', dict())
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            # 定向注入 BFloat16 转换
            return torch.nn.functional.relu(v.to(torch.bfloat16))
    return out
"""
            compile_decorator = 'torch.compile(backend="inductor")'

        else:
            # 狙击目标 4：通用 Inductor Dynamic Shapes Bug
            mode_name = "target_dynamic_shapes"
            tail_logic = "    return locals().get('results', dict())"
            compile_decorator = 'torch.compile(backend="inductor", fullgraph=False, dynamic=True)'

        self.stats[mode_name] += 1

        # 🛠️ 构建最终 Payload
        payload = f"""
import torch
import sys
import traceback
import torch._dynamo

# 关闭日志噪音
torch._dynamo.config.suppress_errors = True

# 1. 探雷：确保基础 API 参数绝对合法
def run_eager():
{self._indent(original_code)}
    pass

try:
    run_eager()
except Exception:
    sys.exit(0)

# 2. 定向狙击函数
def target_func():
{self._indent(original_code)}
{self._indent(tail_logic)}

# 3. 捕获深层 Bug
try:
    compiled_fn = {compile_decorator}(target_func)
    out = compiled_fn()
except Exception as e:
    err_str = str(e).lower()
    
    # 🌟 严格过滤：只关心真的崩溃，过滤掉 Dynamo 正常的 Fallback 提示
    if "not supported" in err_str and "itertools" not in err_str: 
        sys.exit(0)
    if "graph break" in err_str or "recompilation" in err_str:
        sys.exit(0)
        
    # 将包含堆栈信息的报错输出到 stderr，等待外层框架抓取
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
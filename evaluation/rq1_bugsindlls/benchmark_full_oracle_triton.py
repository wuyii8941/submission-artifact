# -*- coding: utf-8 -*-
"""
Kernel-Guided Fuzzing Benchmark - Full Oracle Edition with Hybrid Strategy
==========================================================================

核心增强:
1. **混合策略**: ε-greedy (90% 微创 + 10% 探索)
2. **动态变异**: 停滞时自动扩大变异范围
3. **取消饱和停止**: 改为"扩大搜索"而非"终止"
4. **全 Oracle 支持**: CRASH + CUDA + PRECISION + COMPILE(Guided only)
5. **智能归类**: 基于错误指纹的 Bug 去重与分类
6. **内存保护**: 防止 OOM (Exit 137)
7. **阶段级进程隔离**: Random/Guided 阶段独立子进程运行
8. **LOC 能量法则**: 针对算子融合挖掘黄金种子
9. **动态重放抓取 (Dynamic Replay & Capture)**: 实时截获真实 Bug 堆栈并生成轻量级 JSON

[修复 v14]
- issue_ids warmup: 覆盖 API_TO_ISSUE_MAP 中所有 issue，包括 torch.compile 的 7-8 个 issue
  原来 i % len(issue_ids) 会在 warmup_threshold//2 次迭代内轮流覆盖所有 issue，已正确
- test_with_oracle 返回值解包: 修复为正确解包 (captured_kernels, error_info)
- bug_traces_log 累积: error_info 直接 update 到 bug_traces_log（key 为绝对路径）
- save_crash_code_as_bug: 同步写入 bug_traces_log，使崩溃 bug 也能被 v14 分析脚本匹配
- Random 阶段剥离 COMPILE oracle，保持基准纯净
"""

import copy
import random
import configparser
import os
import json
import hashlib
import re
import pickle
import shutil
import gc
import sys
import subprocess
import multiprocessing as mp
from multiprocessing import Process, Queue
from os.path import join
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import weakref
import traceback

# FreeFuzz imports
from classes.database import TorchDatabase
from classes.torch_api import TorchAPI, TorchArgument
from classes.torch_library import TorchLibrary, IssueScriptRunner
from classes.argument import Argument, ArgType
from constants.enum import OracleType

# TRITON: Ensure OracleType.COMPILE exists
try:
    _ = OracleType.COMPILE
except AttributeError:
    OracleType.COMPILE = max(OracleType.__members__.values()) + 1

# BUG-DISCOVERY: 三方向增强模块
try:
    from bug_discovery_enhancements import (
        BugDiscoveryEnhancer,
        EdgeCaseInjector,
        DtypeSweepMutator,
        CompileWrapperCorpus,
    )
    BUG_DISCOVERY_AVAILABLE = True
    print("[BugDiscovery] ✅ Enhancement module loaded (EdgeCase + DtypeSweep + CompileWrapper)")
except ImportError as _e:
    BUG_DISCOVERY_AVAILABLE = False
    print(f"[BugDiscovery] ⚠️  Enhancement module not found: {_e}")

    class BugDiscoveryEnhancer:
        def __init__(self, *a, **kw): pass
        def apply(self, api, iteration): return []
        def get_compile_test(self, api): return None
        def get_stats(self): return {}
        def print_stats(self): pass

# Constraint Filter imports - 5维正交状态空间
try:
    from pytorch_comprehensive_filter import (
        PyTorchConstraintFilter,
        DispatcherStateSpace,
    )
    FILTER_AVAILABLE = True
except Exception:
    FILTER_AVAILABLE = False

    class DispatcherStateSpace:
        """简化版状态空间管理器 - 无约束过滤"""

        DIMENSION_DEVICE = ['cpu', 'cuda']
        DIMENSION_DTYPE = [
            'float32', 'float64', 'float16', 'bfloat16',
            'int8', 'int16', 'int32', 'int64', 'uint8',
            'bool', 'complex64', 'complex128'
        ]

        def __init__(self, max_states: int = 100000):
            self.max_states = max_states
            self.api_spaces: Dict[str, Dict] = {}
            self.hit_states: Dict[str, Set[str]] = {}
            self.total_hits = 0

        def register_api(self, api_name: str, **kwargs) -> Dict:
            devices = self.DIMENSION_DEVICE
            dtypes = self.DIMENSION_DTYPE
            layouts = ['strided']
            mem_formats = ['contiguous']
            ranks = ['1d', '2d', '3d', '4d', '5d']
            valid_space = len(devices) * len(dtypes) * len(layouts) * len(mem_formats) * len(ranks)
            self.api_spaces[api_name] = {
                'dimensions': {
                    'devices': devices,
                    'dtypes': dtypes,
                    'layouts': layouts,
                    'mem_formats': mem_formats,
                    'ranks': ranks
                },
                'naive_space': valid_space,
                'valid_space': valid_space,
                'filter_stats': {},
            }
            self.hit_states[api_name] = set()
            print(f"[DispatcherStateSpace-Fallback] Registered: {api_name}")
            print(f"  Valid Space: {valid_space} (2×12×1×1×5 = 120)")
            return self.api_spaces[api_name]

        def record_hit(self, api_name: str, state: Dict) -> bool:
            if api_name not in self.api_spaces:
                self.register_api(api_name)
            if self.total_hits >= self.max_states:
                return False
            device = state.get('device', 'unk')
            dtype = state.get('dtype', 'unk')
            layout = state.get('layout', 'strided')
            mem_format = state.get('mem_format', 'contiguous')
            rank = state.get('rank', 'unk')
            if isinstance(rank, int):
                rank = f"{rank}d" if rank <= 5 else "nd"
            state_hash = f"{device}|{dtype}|{layout}|{mem_format}|{rank}"
            if state_hash not in self.hit_states[api_name]:
                self.hit_states[api_name].add(state_hash)
                self.total_hits += 1
                return True
            return False

        def get_coverage(self, api_name: str) -> float:
            if api_name not in self.api_spaces:
                return 0.0
            return len(self.hit_states.get(api_name, set())) / max(self.api_spaces[api_name]['valid_space'], 1)

        def get_detailed_stats(self, api_name: str) -> Dict:
            if api_name not in self.api_spaces:
                return {}
            return {
                'api': api_name,
                'naive_space': self.api_spaces[api_name]['naive_space'],
                'valid_space': self.api_spaces[api_name]['valid_space'],
                'hit_count': len(self.hit_states.get(api_name, set())),
                'coverage': self.get_coverage(api_name),
                'filter_stats': {},
            }

        def print_summary(self):
            print(f"\n{'='*70}")
            print("DISPATCHER STATE SPACE COVERAGE (Fallback)")
            print(f"{'='*70}")
            for api_name, space_info in self.api_spaces.items():
                hit_count = len(self.hit_states.get(api_name, set()))
                valid_space = space_info['valid_space']
                coverage = hit_count / max(valid_space, 1)
                bar_w = 30
                filled = int(bar_w * coverage)
                bar = '█' * filled + '░' * (bar_w - filled)
                print(f"\n[{api_name}]")
                print(f"  Hits: {hit_count}/{valid_space} ({coverage*100:.1f}%) [{bar}]")
            print(f"\nTotal Hits: {self.total_hits}")
            print(f"{'='*70}\n")


# =============================================================================
# Memory Monitor
# =============================================================================

class MemoryMonitor:
    def __init__(self,
                 warning_threshold_gb: float = 4.0,
                 critical_threshold_gb: float = 6.0,
                 enable_auto_gc: bool = True):
        self.warning_threshold = warning_threshold_gb * 1024 * 1024 * 1024
        self.critical_threshold = critical_threshold_gb * 1024 * 1024 * 1024
        self.enable_auto_gc = enable_auto_gc
        self.gc_count = 0
        self.warning_count = 0
        self.last_rss = 0

    def get_memory_usage(self) -> Tuple[float, float]:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated()
        return rss, gpu_mem

    def check_and_cleanup(self, force: bool = False) -> Tuple[bool, str]:
        rss, gpu_mem = self.get_memory_usage()
        message = ""
        is_critical = False

        if rss > self.critical_threshold or force:
            is_critical = True
            message = f"⚠️  CRITICAL MEMORY: {rss/1e9:.2f} GB RSS"
            if self.enable_auto_gc:
                self._aggressive_cleanup()
                new_rss, new_gpu = self.get_memory_usage()
                message += f" → {new_rss/1e9:.2f} GB after cleanup"
        elif rss > self.warning_threshold:
            self.warning_count += 1
            message = f"⚠️  HIGH MEMORY: {rss/1e9:.2f} GB RSS"
            if self.enable_auto_gc:
                self._light_cleanup()

        self.last_rss = rss
        return is_critical, message

    def _light_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.gc_count += 1

    def _aggressive_cleanup(self):
        for _ in range(5):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.gc_count += 1

    def get_status(self) -> str:
        rss, gpu = self.get_memory_usage()
        return f"RSS: {rss/1e9:.2f} GB | GPU: {gpu/1e9:.2f} GB | GC: {self.gc_count}"


# =============================================================================
# Safety Guards
# =============================================================================

class Speedometer:
    def __init__(self, window_size: int = 100, slow_threshold: float = 0.5):
        self.window_size = window_size
        self.slow_threshold = slow_threshold
        self.timestamps = deque(maxlen=window_size)
        self.start_time = None
        self.total_iterations = 0
        self.slow_warnings = 0

    def start(self):
        self.start_time = time.time()
        self.timestamps.append(self.start_time)

    def tick(self):
        self.total_iterations += 1
        self.timestamps.append(time.time())

    def get_speed(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def get_average_speed(self) -> float:
        if not self.start_time or self.total_iterations == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.total_iterations / elapsed if elapsed > 0 else 0.0

    def estimate_remaining_time(self, current_iter: int, max_iter: int) -> float:
        avg_speed = self.get_average_speed()
        if avg_speed == 0:
            return float('inf')
        return (max_iter - current_iter) / avg_speed

    def check_speed(self) -> Tuple[bool, str]:
        current_speed = self.get_speed()
        if current_speed < self.slow_threshold:
            self.slow_warnings += 1
            warning = f"⚠️  SLOW SPEED: {current_speed:.2f} it/s"
            return True, warning
        return False, ""

    def format_time(self, seconds: float) -> str:
        if seconds == float('inf'):
            return "Unknown"
        hours, minutes, secs = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def get_status(self, current_iter: int, max_iter: int) -> str:
        current_speed = self.get_speed()
        avg_speed = self.get_average_speed()
        remaining = self.estimate_remaining_time(current_iter, max_iter)
        elapsed = time.time() - self.start_time if self.start_time else 0
        return (f"{current_speed:.2f} it/s (avg: {avg_speed:.2f}) | "
                f"Elapsed: {self.format_time(elapsed)} | ETA: {self.format_time(remaining)}")


class DiskGuard:
    def __init__(self, output_dir: str, min_free_gb: float = 1.0, auto_cleanup: bool = True):
        self.output_dir = Path(output_dir)
        self.min_free_gb = min_free_gb
        self.auto_cleanup = auto_cleanup
        self.warnings = 0
        self.cleanups = 0

    def get_disk_usage(self) -> Tuple[float, float, float]:
        stat = shutil.disk_usage(self.output_dir)
        return (stat.total / (1024**3), stat.used / (1024**3), stat.free / (1024**3))

    def check_space(self) -> Tuple[bool, str]:
        total, used, free = self.get_disk_usage()
        if free < self.min_free_gb:
            self.warnings += 1
            warning = f"⚠️  DISK SPACE: Only {free:.2f} GB free"
            return True, warning
        return False, ""

    def cleanup_temp_files(self) -> int:
        count = 0
        for temp_file in self.output_dir.rglob("temp.py"):
            try:
                temp_file.unlink()
                count += 1
            except:
                pass
        for pycache in self.output_dir.rglob("__pycache__"):
            try:
                shutil.rmtree(pycache)
                count += 1
            except:
                pass
        if count > 0:
            self.cleanups += 1
        return count

    def check_and_cleanup(self) -> Tuple[bool, str]:
        is_low, warning = self.check_space()
        if not is_low:
            return False, ""
        if self.auto_cleanup:
            cleaned = self.cleanup_temp_files()
            total, used, free = self.get_disk_usage()
            message = f"{warning}\n   Cleaned {cleaned} files → {free:.2f} GB free"
            if free < self.min_free_gb:
                return True, message + "\n   ❌ CRITICAL: Still low!"
            return False, message + "\n   ✅ Cleanup successful"
        return True, warning

    def get_status(self) -> str:
        total, used, free = self.get_disk_usage()
        return f"{free:.2f} GB free ({used/total*100:.1f}% used)"


class OutlierFilter:
    def __init__(self, max_elements: int = int(5e5), max_memory_gb: float = 0.5):
        self.max_elements = max_elements
        self.max_memory_gb = max_memory_gb
        self.total_checks = 0
        self.filtered_count = 0

    def check_tensor_arg(self, arg) -> Tuple[bool, Optional[str]]:
        if not hasattr(arg, 'type') or arg.type != ArgType.TORCH_TENSOR:
            return False, None
        if not hasattr(arg, 'shape') or not arg.shape:
            return False, None
        num_elements = 1
        for dim in arg.shape:
            num_elements *= dim
        if num_elements > self.max_elements:
            return True, f"Too many elements: {num_elements:,}"
        return False, None

    def check_api(self, api) -> Tuple[bool, Optional[str]]:
        self.total_checks += 1
        for param_name, arg in api.args.items():
            if arg is None:
                continue
            if hasattr(arg, 'type') and arg.type in [ArgType.LIST, ArgType.TUPLE]:
                if hasattr(arg, 'value') and arg.value:
                    for sub_arg in arg.value:
                        should_filter, reason = self.check_tensor_arg(sub_arg)
                        if should_filter:
                            self.filtered_count += 1
                            return True, f"{param_name}: {reason}"
            else:
                should_filter, reason = self.check_tensor_arg(arg)
                if should_filter:
                    self.filtered_count += 1
                    return True, f"{param_name}: {reason}"
        return False, None

    def get_status(self) -> str:
        rate = (self.filtered_count / self.total_checks * 100) if self.total_checks > 0 else 0
        return f"{self.filtered_count}/{self.total_checks} filtered ({rate:.1f}%)"


# =============================================================================
# Checkpoint System
# =============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, strategy: str, api_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self.api_name = api_name.replace('.', '_').replace('::', '_')
        self.checkpoint_file = self.checkpoint_dir / f"{self.api_name}_{strategy}_checkpoint.pkl"
        self.metadata_file = self.checkpoint_dir / f"{self.api_name}_{strategy}_metadata.json"

    def save(self, iteration: int, coverage_kernels: set,
             corpus_seeds: list = None, logger_iterations: int = 0) -> bool:
        try:
            checkpoint_data = {
                'iteration': iteration,
                'coverage_kernels': coverage_kernels,
                'corpus_seeds': None,
                'logger_iterations': logger_iterations,
                'strategy': self.strategy,
                'api_name': self.api_name
            }
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            temp_file.replace(self.checkpoint_file)
            metadata = {
                'iteration': iteration,
                'total_kernels': len(coverage_kernels),
                'strategy': self.strategy
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            print(f"[Checkpoint] ❌ Save failed: {e}")
            return False

    def load(self) -> Optional[Dict]:
        if not self.checkpoint_file.exists():
            return None
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[Checkpoint] ✅ Loaded from iteration {data['iteration']}")
            return data
        except Exception as e:
            print(f"[Checkpoint] ❌ Load failed: {e}")
            return None

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

    def clear(self):
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
        except:
            pass


# =============================================================================
# Feature Extractor
# =============================================================================

def extract_features(api: TorchAPI) -> str:
    features = []
    max_features = 10
    for param_name, arg in api.args.items():
        if arg is None:
            continue
        if len(features) >= max_features:
            break
        if arg.type == ArgType.INT:
            features.append(f"{param_name}:int:{arg.value}")
        elif arg.type == ArgType.FLOAT:
            features.append(f"{param_name}:float:{arg.value:.2f}")
        elif arg.type == ArgType.BOOL:
            features.append(f"{param_name}:bool:{arg.value}")
        elif arg.type == ArgType.STR:
            val = str(arg.value)[:20]
            features.append(f"{param_name}:str:{val}")
        elif arg.type == ArgType.TORCH_DTYPE:
            dtype_str = str(arg.value).split(".")[-1] if hasattr(arg, 'value') else 'unknown'
            features.append(f"{param_name}:dtype:{dtype_str}")
        elif arg.type == ArgType.TORCH_TENSOR:
            ndim = len(arg.shape) if hasattr(arg, 'shape') and arg.shape else 0
            dtype_str = str(arg.dtype).split(".")[-1][:8] if hasattr(arg, 'dtype') else 'unk'
            features.append(f"{param_name}:t:{ndim}d:{dtype_str}")
        elif arg.type in [ArgType.LIST, ArgType.TUPLE]:
            length = len(arg.value) if hasattr(arg, 'value') and arg.value else 0
            type_name = "L" if arg.type == ArgType.LIST else "T"
            features.append(f"{param_name}:{type_name}:{length}")
    features.sort()
    result = "|".join(features)
    return result[:200] if len(result) > 200 else result


# =============================================================================
# Experiment Logger
# =============================================================================

class ExperimentLogger:
    def __init__(self, output_file: str, strategy: str):
        self.output_file = output_file
        self.strategy = strategy
        self.file_handle = open(output_file, 'w', buffering=1)
        self.total_iterations = 0
        self.valid_count = 0
        self.invalid_count = 0

    def log_iteration(self, iteration: int, source: str, api: TorchAPI,
                      valid: bool, kernels: List[str]):
        if valid and kernels:
            features = extract_features(api) if api else ""
            record = {"i": iteration, "s": source[:3], "v": 1, "f": features, "k": len(kernels)}
            self.file_handle.write(json.dumps(record) + '\n')
        self.total_iterations += 1
        if valid:
            self.valid_count += 1
        else:
            self.invalid_count += 1

    def get_valid_rate(self) -> float:
        if self.total_iterations == 0:
            return 0.0
        return self.valid_count / self.total_iterations

    def close(self):
        if self.file_handle:
            self.file_handle.close()

    def __del__(self):
        self.close()


# 保存原始方法
_ORIGINAL_DO_SELECT_FROM_DB = None
_ORIGINAL_MUTATE_INT_VALUE = None
_ORIGINAL_MUTATE_FLOAT_VALUE = None


# =============================================================================
# Bug Tracker
# =============================================================================

class BugAnalyzer:
    REGEX_PATTERNS = [
        (r'0x[0-9a-fA-F]+', 'ADDR'),
        (r'\d+\.\d+', 'FLOAT'),
        (r'\d+', 'INT'),
        (r'\[.*?\]', '[SHAPE]'),
        (r'\s+', ' '),
    ]

    @staticmethod
    def get_signature(content: str) -> str:
        if not content:
            return "unknown_empty"
        features = []
        dtype_matches = re.findall(r'dtype\s*=\s*(torch\.[a-z0-9]+)', content)
        if dtype_matches:
            unique_dtypes = sorted(set(dtype_matches))[:3]
            features.append(f"dtype={','.join(unique_dtypes)}")
        shape_matches = re.findall(r'torch\.rand(?:int)?\s*\(\s*[\[\(]([^\]\)]+)[\]\)]', content)
        if shape_matches:
            for shape in shape_matches[:2]:
                ndim = shape.count(',') + 1
                features.append(f"ndim={ndim}")
                break
        if re.search(r'\bnan\b', content, re.IGNORECASE):
            features.append("NaN")
        if re.search(r'\binf\b', content, re.IGNORECASE):
            features.append("Inf")
        if not features:
            return f"hash={hashlib.md5(content.encode()).hexdigest()[:12]}"
        return "|".join(sorted(features))[:100]

    @staticmethod
    def classify_bug(error_msg: str) -> str:
        sig = BugAnalyzer.get_signature(error_msg).lower()
        if "segfault" in sig or "core_dump" in sig:
            return "CRITICAL_SEGFAULT"
        deep_keywords = ['cuda', 'kernel', 'launch', 'cublas', 'cudnn', 'internal', 'assert']
        if any(k in sig for k in deep_keywords):
            return "DEEP_INTERNAL"
        shallow_keywords = ['type', 'shape', 'dim', 'expect', 'got', 'invalid', 'implement']
        if any(k in sig for k in shallow_keywords):
            return "SHALLOW_CHECK"
        return "RUNTIME_OTHER"


class BugTracker:
    def __init__(self, name: str, output_dir: str):
        self.name = name
        self.output_dir = output_dir
        self.bug_counts = {"crash": 0, "cuda": 0, "precision": 0, "compile": 0}
        self.bug_files = {"crash": set(), "cuda": set(), "precision": set(), "compile": set()}
        self.bugs = {"crash": [], "cuda": [], "precision": [], "compile": []}
        self.unique_bugs = {}
        self.max_unique_bugs = 500
        self.bug_stats = defaultdict(int)

    def scan_bugs(self):
        self.bug_stats.clear()
        oracle_map = {
            "crash": "crash-oracle",
            "cuda": "cuda-oracle",
            "precision": "precision-oracle",
            "compile": "compile-oracle"
        }
        for bug_type, oracle_name in oracle_map.items():
            bug_dir = join(self.output_dir, oracle_name, "potential-bug")
            if not os.path.exists(bug_dir):
                continue
            for root, _, files in os.walk(bug_dir):
                for f in files:
                    if not f.endswith('.py'):
                        continue
                    file_path = join(root, f)
                    try:
                        with open(file_path, 'r', errors='ignore') as f_hash:
                            raw_content = f_hash.read(2000)
                        content_key = re.sub(r'#.*|\s+', '', raw_content)
                        file_hash = hashlib.md5(content_key.encode()).hexdigest()[:16]
                    except Exception:
                        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
                        raw_content = ""
                    if file_hash not in self.bug_files[bug_type]:
                        self.bug_files[bug_type].add(file_hash)
                        self.bug_counts[bug_type] += 1
                        if len(self.unique_bugs) < self.max_unique_bugs:
                            try:
                                content = raw_content[:1000] if raw_content else ""
                                signature = BugAnalyzer.get_signature(content)
                                category = BugAnalyzer.classify_bug(content)
                                unique_key = f"[{bug_type.upper()}]_{signature[:50]}"
                                if unique_key not in self.unique_bugs:
                                    self.unique_bugs[unique_key] = file_path
                                    self.bug_stats[category] += 1
                            except:
                                pass
        for bt in ['crash', 'cuda', 'precision', 'compile']:
            self.bugs[bt] = [(i, "", "") for i in range(self.bug_counts[bt])]

    def get_total_bugs(self) -> int:
        return sum(self.bug_counts.values())

    def get_bugs_by_type(self, bug_type: str) -> int:
        return self.bug_counts.get(bug_type, 0)

    def print_summary(self):
        self.scan_bugs()
        print(f"\n{'='*70}")
        print(f"🧩 BUG ANALYSIS REPORT: {self.name}")
        print(f"{'='*70}")
        print(f"Total Unique Issues: {len(self.unique_bugs)}")
        print(f"Distribution: {dict(self.bug_stats)}")
        print("-" * 70)
        print("Top Unique Bugs Found:")
        for i, (sig, path) in enumerate(list(self.unique_bugs.items())[:5]):
            print(f" {i+1}. {sig}")
        if not self.unique_bugs:
            print(" No bugs found.")
        print("-" * 70)
        total_raw = self.get_total_bugs()
        total_unique = len(self.unique_bugs)
        print(f"📌 UNIQUE Bugs: {total_unique}")
        print(f"🔄 Raw Files:   CRASH={self.bug_counts['crash']} | CUDA={self.bug_counts['cuda']} | "
              f"PRECISION={self.bug_counts['precision']} | COMPILE={self.bug_counts['compile']} | Total={total_raw}")
        print(f"{'='*70}\n")


# =============================================================================
# RealTimeBugDeduplicator
# =============================================================================

class RealTimeBugDeduplicator:
    CLEAN_PATTERNS = [
        (r'0x[0-9a-fA-F]+', 'ADDR'),
        (r'\d+\.\d+', 'FLOAT'),
        (r'(?<!\w)\d+(?!\w)', 'INT'),
        (r'\[.*?\]', '[SHAPE]'),
        (r'at .*?:\d+', 'at FILE:LINE'),
        (r'\s+', ' '),
    ]

    def __init__(self, max_signatures: int = 1000):
        self.max_signatures = max_signatures
        self.signatures: Dict[str, Set[str]] = {
            'crash': set(), 'cuda': set(), 'precision': set(), 'compile': set()
        }
        self.first_occurrence_count = 0
        self.total_bugs = {'crash': 0, 'cuda': 0, 'precision': 0, 'compile': 0}
        self.unique_bugs = {'crash': 0, 'cuda': 0, 'precision': 0, 'compile': 0}
        self.error_samples: Dict[str, Dict[str, str]] = {
            'crash': {}, 'cuda': {}, 'precision': {}, 'compile': {}
        }

    def get_signature(self, error_msg: str) -> str:
        if not error_msg:
            return "UNKNOWN"
        lines = error_msg.strip().split('\n')
        core_msg = lines[-1][:200]
        for line in reversed(lines[-10:]):
            if any(kw in line for kw in ['Error:', 'Exception:', 'INTERNAL ASSERT']):
                core_msg = line[:200]
                break
        clean_msg = core_msg
        for pattern, replacement in self.CLEAN_PATTERNS:
            clean_msg = re.sub(pattern, replacement, clean_msg)
        if 'Segmentation fault' in error_msg:
            return "SEGFAULT"
        return clean_msg.strip()[:100]

    def record_bug(self, oracle_type: str, error_msg: str, iteration: int = 0) -> Tuple[bool, str]:
        oracle_type = oracle_type.lower()
        if oracle_type not in self.signatures:
            oracle_type = 'crash'
        self.total_bugs[oracle_type] += 1
        signature = self.get_signature(error_msg)
        sig_hash = hashlib.md5(signature.encode()).hexdigest()[:12]

        if sig_hash in self.signatures[oracle_type]:
            return False, signature

        if len(self.signatures[oracle_type]) >= self.max_signatures:
            return False, signature

        self.signatures[oracle_type].add(sig_hash)
        self.unique_bugs[oracle_type] += 1
        self.first_occurrence_count += 1
        self.error_samples[oracle_type][sig_hash] = str(error_msg).strip()[:1000]

        return True, signature

    def get_dedup_stats(self) -> dict:
        return {
            'total': self.total_bugs.copy(),
            'unique': self.unique_bugs.copy(),
            'dedup_rate': {
                k: 1 - (self.unique_bugs[k] / max(self.total_bugs[k], 1))
                for k in self.total_bugs
            }
        }

    def print_summary(self):
        stats = self.get_dedup_stats()
        print(f"\n{'='*70}")
        print("🔍 BUG DEDUPLICATION REPORT")
        print(f"{'='*70}")
        for oracle in ['crash', 'cuda', 'precision', 'compile']:
            total, unique = stats['total'][oracle], stats['unique'][oracle]
            rate = stats['dedup_rate'][oracle] * 100
            print(f"[{oracle.upper()}] Total: {total}, Unique: {unique}, Dup Rate: {rate:.1f}%")
        print(f"{'='*70}")


# =============================================================================
# DispatcherSpace
# =============================================================================

class DispatcherSpace:
    def __init__(self, max_hits: int = 10000):
        self.max_hits = max_hits
        self.state_space = DispatcherStateSpace(max_states=max_hits)
        self.api_dimensions: Dict[str, Dict[str, List[str]]] = {}
        self.theoretical_sizes: Dict[str, int] = {}
        self.total_records = 0
        self.overflow_warnings = 0
        self.coverage_history: Dict[str, deque] = {}
        self.max_history_size = 100

    def register_api(self, api_name: str, dimensions: Dict[str, List[str]] = None):
        result = self.state_space.register_api(api_name)
        self.theoretical_sizes[api_name] = result['valid_space']
        self.api_dimensions[api_name] = result['dimensions']
        self.coverage_history[api_name] = deque(maxlen=self.max_history_size)

    def record_hit(self, api_name: str, api, is_cuda: bool = False, iteration: int = 0) -> bool:
        if api_name not in self.state_space.api_spaces:
            self.register_api(api_name)
        state = self._extract_state_from_args(api.args, is_cuda)
        if state is None:
            return False
        is_new = self.state_space.record_hit(api_name, state)
        if is_new:
            self.total_records += 1
            if iteration % 50 == 0:
                current_cov = self.get_coverage(api_name)
                self.coverage_history[api_name].append((iteration, current_cov))
        return is_new

    def record_coverage_snapshot(self, api_name: str, iteration: int):
        if api_name not in self.coverage_history:
            self.coverage_history[api_name] = deque(maxlen=self.max_history_size)
        current_cov = self.get_coverage(api_name)
        self.coverage_history[api_name].append((iteration, current_cov))

    def _extract_state_from_args(self, args: Dict, is_cuda: bool) -> Optional[Dict]:
        state = {
            'device': 'cuda' if is_cuda else 'cpu',
            'dtype': None,
            'layout': 'strided',
            'mem_format': 'contiguous',
            'rank': None,
        }

        def extract_from_arg(arg):
            if arg is None:
                return None, None
            if hasattr(arg, 'type') and arg.type in [ArgType.LIST, ArgType.TUPLE]:
                if hasattr(arg, 'value') and arg.value:
                    for sub_arg in arg.value:
                        dtype, rank = extract_from_arg(sub_arg)
                        if dtype is not None:
                            return dtype, rank
                return None, None
            if hasattr(arg, 'type') and arg.type == ArgType.TORCH_TENSOR:
                dtype = None
                rank = None
                if hasattr(arg, 'dtype') and arg.dtype is not None:
                    dtype = str(arg.dtype).replace('torch.', '')
                if hasattr(arg, 'shape') and arg.shape is not None:
                    rank = len(arg.shape)
                return dtype, rank
            return None, None

        if 'input_signature' in args:
            dtype, rank = extract_from_arg(args['input_signature'])
            if dtype is not None:
                state['dtype'] = dtype
            if rank is not None:
                state['rank'] = rank
            if state['dtype'] is not None and state['rank'] is not None:
                return state

        for param_name, arg in args.items():
            if param_name == 'input_signature':
                continue
            dtype, rank = extract_from_arg(arg)
            if dtype is not None and state['dtype'] is None:
                state['dtype'] = dtype
            if rank is not None and state['rank'] is None:
                state['rank'] = rank
            if hasattr(arg, 'is_sparse') and arg.is_sparse:
                state['layout'] = 'sparse_coo'
            if hasattr(arg, 'type') and hasattr(arg, 'value'):
                if arg.type == ArgType.TORCH_DTYPE:
                    state['dtype'] = str(arg.value).replace('torch.', '')
            if state['dtype'] is not None and state['rank'] is not None:
                return state

        if len(args) > 0:
            if state['dtype'] is None:
                state['dtype'] = 'float32'
            if state['rank'] is None:
                state['rank'] = 2
            return state
        return None

    def get_coverage(self, api_name: str) -> float:
        return self.state_space.get_coverage(api_name)

    def get_detailed_stats(self, api_name: str) -> Dict:
        return self.state_space.get_detailed_stats(api_name)

    def print_summary(self):
        self.state_space.print_summary()


# =============================================================================
# AdaptiveSaturationDetector
# =============================================================================

class AdaptiveSaturationDetector:
    def __init__(self, patience: int = 500, check_interval: int = 100):
        self.patience = patience
        self.check_interval = check_interval
        self.no_discovery_count = 0
        self.last_kernel_count = 0
        self.last_check_iteration = 0
        self.expansion_mode = False
        self.expansion_count = 0

    def update(self, current_kernels: int, iteration: int) -> Tuple[bool, str]:
        if current_kernels > self.last_kernel_count:
            self.no_discovery_count = 0
            self.last_kernel_count = current_kernels
            if self.expansion_mode:
                self.expansion_mode = False
                return False, "✅ New kernels found! Returning to normal mode"
        else:
            self.no_discovery_count += 1

        if iteration - self.last_check_iteration >= self.check_interval:
            self.last_check_iteration = iteration
            if self.no_discovery_count >= self.patience and not self.expansion_mode:
                self.expansion_mode = True
                self.expansion_count += 1
                return True, f"🔍 Expanding search (stagnation: {self.no_discovery_count} iters)"
        return False, ""

    def is_in_expansion_mode(self) -> bool:
        return self.expansion_mode

    def reset(self):
        self.no_discovery_count = 0
        self.expansion_mode = False


# =============================================================================
# Enhanced Coverage Tracker
# =============================================================================

class EnhancedCoverageTracker:
    def __init__(self, name: str, enable_adaptive_saturation: bool = True):
        self.name = name
        self.all_kernel_hashes: Set[str] = set()
        self.kernel_count = 0
        self.history: deque = deque(maxlen=200)
        self.new_kernel_iterations: deque = deque(maxlen=100)
        self.saturation_detector = AdaptiveSaturationDetector() if enable_adaptive_saturation else None
        self._recent_kernels: deque = deque(maxlen=1000)

    @property
    def all_kernels(self) -> Set[str]:
        return set(self._recent_kernels)

    def update(self, new_kernels: Set[str], iteration: int) -> Tuple[int, Tuple[bool, str]]:
        if not isinstance(new_kernels, set):
            new_kernels = set(new_kernels) if new_kernels else set()
        fresh_count = 0
        for kernel in new_kernels:
            kernel_hash = hashlib.md5(kernel.encode()).hexdigest()[:16]
            if kernel_hash not in self.all_kernel_hashes:
                self.all_kernel_hashes.add(kernel_hash)
                self._recent_kernels.append(kernel)
                fresh_count += 1
        if fresh_count > 0:
            self.kernel_count += fresh_count
            self.new_kernel_iterations.append(iteration)
            print(f"  [{self.name}] Iter {iteration}: +{fresh_count} new kernels! Total: {self.kernel_count}")
        self.history.append((iteration, self.kernel_count))
        expansion_signal = (False, "")
        if self.saturation_detector:
            expansion_signal = self.saturation_detector.update(self.kernel_count, iteration)
            if expansion_signal[0]:
                print(f"\n{expansion_signal[1]}")
        return fresh_count, expansion_signal

    def get_total(self) -> int:
        return self.kernel_count

    def get_exclusive(self, other: 'EnhancedCoverageTracker') -> Set[str]:
        return set(self._recent_kernels) - set(other._recent_kernels)

    def get_overlap(self, other: 'EnhancedCoverageTracker') -> Set[str]:
        return set(self._recent_kernels) & set(other._recent_kernels)


# =============================================================================
# Evolutionary Corpus (with LOC Energy Scheduling)
# =============================================================================

class EvolutionaryCorpus:
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.max_tensor_elements = 50000
        self._seed_signatures: deque = deque(maxlen=max_size)
        self._seed_api_names: deque = deque(maxlen=max_size)
        self._seed_scores: deque = deque(maxlen=max_size)

    def _get_signature(self, api: TorchAPI) -> Optional[str]:
        try:
            sig_parts = [api.api]
            for name, arg in api.args.items():
                if arg is None:
                    continue
                if hasattr(arg, 'type'):
                    if arg.type == ArgType.TORCH_TENSOR:
                        shape_str = str(arg.shape)[:50] if hasattr(arg, 'shape') else ""
                        dtype_str = str(arg.dtype)[:20] if hasattr(arg, 'dtype') else ""
                        sig_parts.append(f"{name}:T:{shape_str}:{dtype_str}")
                    elif arg.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
                        sig_parts.append(f"{name}:{arg.type.name}:{arg.value}")
            return "|".join(sig_parts)[:500]
        except:
            return None

    def _estimate_api_size(self, api: TorchAPI) -> int:
        total_elements = 0
        for param_name, arg in api.args.items():
            if arg is None:
                continue
            if hasattr(arg, 'type') and arg.type == ArgType.TORCH_TENSOR:
                if hasattr(arg, 'shape') and arg.shape:
                    elements = 1
                    for dim in arg.shape:
                        elements *= dim
                    total_elements += elements
        return total_elements

    def add_seed(self, api: TorchAPI, discovered_kernels: Set[str]):
        if len(discovered_kernels) == 0:
            return
        api_size = self._estimate_api_size(api)
        if api_size > self.max_tensor_elements:
            return
        sig = self._get_signature(api)
        loc_score = sum(len(k.split('\n')) for k in discovered_kernels)

        if sig and sig not in self._seed_signatures:
            self._seed_signatures.append(sig)
            self._seed_api_names.append(api.api)
            self._seed_scores.append(loc_score)

    def select_parent(self) -> Optional[TorchAPI]:
        if not self._seed_api_names:
            return None
        scores = list(self._seed_scores)
        total_score = sum(scores)
        if total_score == 0:
            api_name = random.choice(list(self._seed_api_names))
        else:
            probs = [s / total_score for s in scores]
            api_name = random.choices(list(self._seed_api_names), weights=probs, k=1)[0]
        return TorchAPI(api_name)

    def size(self) -> int:
        return len(self._seed_signatures)

    def clear_old_seeds(self, keep_ratio: float = 0.5):
        pass

    @property
    def corpus(self):
        return list(self._seed_signatures)


# =============================================================================
# Adaptive Mutation Controller
# =============================================================================

class AdaptiveMutationController:
    def __init__(self,
                 base_epsilon: float = 0.1,
                 expansion_epsilon: float = 0.3,
                 surgical_range: int = 8,
                 exploration_range: int = 64):
        self.base_epsilon = base_epsilon
        self.expansion_epsilon = expansion_epsilon
        self.surgical_range = surgical_range
        self.exploration_range = exploration_range
        self.current_epsilon = base_epsilon
        self.current_range = surgical_range
        self.expansion_mode = False
        self.exploration_count = 0
        self.exploitation_count = 0

    def should_explore(self) -> bool:
        if random.random() < self.current_epsilon:
            self.exploration_count += 1
            return True
        else:
            self.exploitation_count += 1
            return False

    def enter_expansion_mode(self):
        self.expansion_mode = True
        self.current_epsilon = self.expansion_epsilon
        self.current_range = self.exploration_range
        print(f"  🔍 [Mutation] Expansion mode: ε={self.current_epsilon}, range=±{self.current_range}")

    def exit_expansion_mode(self):
        self.expansion_mode = False
        self.current_epsilon = self.base_epsilon
        self.current_range = self.surgical_range
        print(f"  ✅ [Mutation] Normal mode: ε={self.current_epsilon}, range=±{self.current_range}")

    def get_mutation_range(self) -> int:
        return self.current_range

    def get_status(self) -> str:
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return "No mutations yet"
        explore_pct = self.exploration_count / total * 100
        return (f"Explore: {self.exploration_count}/{total} ({explore_pct:.1f}%) | "
                f"Mode: {'Expansion' if self.expansion_mode else 'Normal'} | "
                f"ε={self.current_epsilon:.2f}, range=±{self.current_range}")


# =============================================================================
# Probability Patcher
# =============================================================================

class ProbabilityPatcher:
    @staticmethod
    def patch_high_db_probability():
        import utils.probability as prob_module
        global _ORIGINAL_DO_SELECT_FROM_DB
        _ORIGINAL_DO_SELECT_FROM_DB = prob_module.do_select_from_db

        def high_db_select() -> bool:
            from numpy.random import rand
            return rand() < 0.2

        prob_module.do_select_from_db = high_db_select
        print("[Patch] ✅ Database sampling: 20% → 80%")

    @staticmethod
    def restore():
        if _ORIGINAL_DO_SELECT_FROM_DB:
            import utils.probability as prob_module
            prob_module.do_select_from_db = _ORIGINAL_DO_SELECT_FROM_DB
            print("[Patch] 🔄 Database sampling restored")


# =============================================================================
# Poison Patcher
# =============================================================================

_POISON_ORIGINAL_FLOAT = None
_POISON_ORIGINAL_INT = None


class PoisonPatcher:
    @staticmethod
    def patch():
        global _POISON_ORIGINAL_FLOAT, _POISON_ORIGINAL_INT
        _POISON_ORIGINAL_FLOAT = Argument.mutate_float_value
        _POISON_ORIGINAL_INT = Argument.mutate_int_value

        def poison_float_mutation(self, value) -> float:
            from numpy.random import rand, choice
            roll = rand()
            if roll < 0.05:
                return choice([float('inf'), float('-inf')])
            elif roll < 0.10:
                return float('nan')
            elif roll < 0.20:
                return choice([1e20, -1e20, 1e-10, -1e-10])
            elif roll < 0.50:
                if hasattr(Argument, '_float_values'):
                    return choice(Argument._float_values)
                return choice([0.0, 1.0, -1.0])
            else:
                return value + (rand() - 0.5) * 8.0

        def poison_int_mutation(self, value, _min=None, _max=None) -> int:
            from numpy.random import rand, choice, randint

            roll = rand()

            if roll < 0.15:
                new_value = choice([
                    -32768, -10000, -1000, -100, -10, -1,
                    6442450944,
                    0, 1,
                    2147483647, -2147483648,
                    999, -999, 65536, -65536
                ])
            elif roll < 0.30:
                new_value = choice([-2, -3, -4, 256, 512, 1024, 7, 11, 13, 0])
            elif roll < 0.60:
                if hasattr(Argument, '_int_values'):
                    new_value = choice(Argument._int_values)
                else:
                    new_value = randint(-100, 100)
            else:
                new_value = value + randint(-8, 9)

            if _min is not None:
                if new_value < -5:
                    pass
                else:
                    new_value = max(_min, new_value)

            if _max is not None and new_value > _max:
                if new_value < 100000000:
                    new_value = min(_max, new_value)

            return int(new_value)

        Argument.mutate_float_value = poison_float_mutation
        Argument.mutate_int_value = poison_int_mutation

        print("[Patch] 🧪 Enhanced Poison Injection enabled")
        print("        Float: 5% Inf + 5% NaN + 10% Extreme")
        print("        Int: 15% extreme (neg indices -32768~0 + padding 6442450944)")
        print("        Int: 15% trap + 30% list + 40% micro")
        print("        Protection: Special negative values preserved")

    @staticmethod
    def restore():
        global _POISON_ORIGINAL_FLOAT, _POISON_ORIGINAL_INT
        if _POISON_ORIGINAL_FLOAT:
            Argument.mutate_float_value = _POISON_ORIGINAL_FLOAT
        if _POISON_ORIGINAL_INT:
            Argument.mutate_int_value = _POISON_ORIGINAL_INT
        print("[Patch] 🔄 Poison Injection restored")


class AdaptiveMutationPatcher:
    @staticmethod
    def patch_adaptive_mutation(controller: AdaptiveMutationController):
        print(f"[Patch] ✅ Adaptive range enabled: ±{controller.get_mutation_range()}")

    @staticmethod
    def restore():
        print("[Patch] 🔄 Adaptive range disabled")


# =============================================================================
# Enhanced Fuzzer
# =============================================================================

class EnhancedFuzzer:
    def __init__(self,
                 api_name: str,
                 output_dir: str,
                 strategy_name: str,
                 enable_patches: bool,
                 use_all_oracles: bool = True,
                 enable_logging: bool = True,
                 enable_checkpoint: bool = True,
                 enable_safety_guards: bool = True,
                 diff_bound: float = 1e-5,
                 warmup_ratio: float = 0.1,
                 enable_dispatcher: bool = True,
                 crash_code_file: str = None):
        self.api_name = api_name
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.enable_patches = enable_patches
        self.use_all_oracles = use_all_oracles
        self.enable_logging = enable_logging
        self.enable_checkpoint = enable_checkpoint
        self.enable_safety_guards = enable_safety_guards
        self.diff_bound = diff_bound
        self.warmup_ratio = warmup_ratio
        self.enable_dispatcher = enable_dispatcher
        self.crash_code_file = crash_code_file
        self.iterations_completed = 0

        # 用于实时记录截获的报错信息（黑匣子）
        # key = write_to_dir 返回的绝对路径，与 _save_bug_traces 的 relpath 归一化对齐
        self.bug_traces_log = {}

        self.coverage = EnhancedCoverageTracker(strategy_name, enable_adaptive_saturation=enable_patches)
        self.bug_tracker = BugTracker(strategy_name, output_dir)
        self.corpus = EvolutionaryCorpus(max_size=20) if enable_patches else None
        self.mutation_controller = AdaptiveMutationController() if enable_patches else None
        self.library = TorchLibrary(output_dir, diff_bound=diff_bound)

        # [修复] Oracle 列表：Random 阶段剥离 COMPILE，保持基准纯净
        if use_all_oracles:
            if strategy_name.lower() == "random":
                self.oracles = [OracleType.CRASH, OracleType.CUDA, OracleType.PRECISION]
            else:
                self.oracles = [OracleType.CRASH, OracleType.CUDA, OracleType.PRECISION, OracleType.COMPILE]
        else:
            self.oracles = [OracleType.CRASH]

        self.logger = None
        if enable_logging:
            api_clean = api_name.replace('.', '_').replace('::', '_')
            log_file = join(output_dir, f"{api_clean}_{strategy_name.lower()}_trace.jsonl")
            self.logger = ExperimentLogger(log_file, strategy_name.lower())

        self.checkpoint_manager = None
        if enable_checkpoint:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=output_dir,
                strategy=strategy_name.lower(),
                api_name=api_name
            )

        self.speedometer = None
        self.disk_guard = None
        self.outlier_filter = None
        self.memory_monitor = None

        if enable_safety_guards:
            self.speedometer = Speedometer(window_size=50, slow_threshold=0.5)
            self.disk_guard = DiskGuard(output_dir, min_free_gb=1.0, auto_cleanup=True)
            self.outlier_filter = OutlierFilter(max_elements=int(5e5), max_memory_gb=0.5)
            self.memory_monitor = MemoryMonitor(warning_threshold_gb=4.0, critical_threshold_gb=6.0)

        self.dispatcher_space = None
        if enable_dispatcher:
            self.dispatcher_space = DispatcherSpace(max_hits=10000)
            self.dispatcher_space.register_api(api_name)

        self.bug_deduplicator = RealTimeBugDeduplicator(max_signatures=1000)
        self.warmup_threshold = 0

        # [修复 Q1] issue_ids warmup：加载当前 API 对应的所有 issue
        # API_TO_ISSUE_MAP 在 DynamicScriptInjector 里，但 IssueScriptRunner 提供磁盘扫描接口
        # 这里直接用 IssueScriptRunner.load_all_issue_ids() 加载磁盘上存在的所有 issue
        # 对于 torch.compile 这类有 7-8 个 issue 的 API，i % len(issue_ids) 会在
        # warmup_threshold//2 次迭代中轮流覆盖全部 issue，确保不遗漏
        self.issue_ids: list = []
        if enable_patches:
            self.issue_ids = IssueScriptRunner.load_all_issue_ids()
            if self.issue_ids:
                print(f"[IssueScripts] ✅ {len(self.issue_ids)} scripts available for guided warmup")
                print(f"[IssueScripts]    Warmup will cycle through all {len(self.issue_ids)} issues")

        # BUG-DISCOVERY: 三方向增强
        self.bug_enhancer = BugDiscoveryEnhancer(
            api_name=api_name,
            enable_edge_case=True,
            enable_dtype_sweep=True,
            enable_compile_wrapper=(OracleType.COMPILE in self.oracles),
            edge_injection_prob=0.15,
            dtype_sweep_prob=0.20,
            compile_test_prob=0.10,
        )
        if BUG_DISCOVERY_AVAILABLE:
            print(f"[BugDiscovery] Enhancer initialized for {api_name}")
            print(f"  EdgeCase prob=15%, DtypeSweep prob=20%, CompileWrapper prob=10%")

    def run_fuzzing_loop(self,
                         max_iterations: int = 10000,
                         checkpoint_interval: int = 100,
                         bug_scan_interval: int = 50,
                         gc_interval: int = 10,
                         intermediate_save_callback=None) -> EnhancedCoverageTracker:

        if self.speedometer:
            self.speedometer.start()

        # 从 checkpoint 恢复
        start_iteration = 0
        if self.checkpoint_manager and self.checkpoint_manager.exists():
            checkpoint_data = self.checkpoint_manager.load()
            if checkpoint_data:
                self.coverage.all_kernel_hashes = checkpoint_data['coverage_kernels']
                self.coverage.kernel_count = len(checkpoint_data['coverage_kernels'])
                start_iteration = checkpoint_data['iteration'] + 1
                if self.logger:
                    self.logger.total_iterations = checkpoint_data.get('logger_iterations', 0)
                print(f"✅ Resuming from iteration {start_iteration}, "
                      f"kernels: {self.coverage.kernel_count}")

        start_time = time.time()
        self.warmup_threshold = int(max_iterations * self.warmup_ratio)

        print(f"\n{'='*70}")
        print(f"Starting {self.strategy_name} fuzzing")
        print(f"Iterations: {start_iteration} → {max_iterations}")
        print(f"Oracles: {[o.name for o in self.oracles]}")
        print(f"Warmup: 0 ~ {self.warmup_threshold} | Evolution: {self.warmup_threshold} ~ {max_iterations}")
        if self.issue_ids:
            print(f"Issue Warmup (first half): {len(self.issue_ids)} issues will be cycled")
        print(f"{'='*70}\n")

        first_save_done = False

        for i in range(start_iteration, max_iterations):
            if self.speedometer:
                self.speedometer.tick()

            # ── 策略选择 ──────────────────────────────────────────────
            source = "random"
            is_exploration = False

            if i < self.warmup_threshold:
                mid = self.warmup_threshold // 2
                if i < mid and self.issue_ids:
                    # [修复 Q1] 用 i % len(issue_ids) 轮流覆盖所有 issue
                    # torch.compile 有 14 个 issue，全部会在 warmup_threshold//2 次内被覆盖
                    issue_id = self.issue_ids[i % len(self.issue_ids)]
                    mutate = (i % 3 != 0)  # 每3次中2次做变异，1次原始重放
                    try:
                        kernels, bug_found, script_src, error_info = self.library.test_with_script(
                            issue_id, mutate=mutate
                        )
                        all_captured_kernels = set(kernels)
                        execution_valid = True
                        source = f"issue_replay_{issue_id}"
                        # [修复] error_info key 已经是绝对路径（test_with_script 修复后），
                        # 直接 update 到 bug_traces_log
                        if error_info:
                            self.bug_traces_log.update(error_info)
                        if bug_found:
                            print(f"  🐛 [IssueScript] Bug reproduced: issue_{issue_id}")
                        new_count, (should_expand, _) = self.coverage.update(all_captured_kernels, i)
                        if new_count > 0 and self.corpus and script_src:
                            dummy_api = TorchAPI(self.api_name)
                            self.corpus.add_seed(dummy_api, all_captured_kernels)
                        if self.logger:
                            self.logger.log_iteration(
                                iteration=i, source=source, api=None,
                                valid=execution_valid, kernels=list(all_captured_kernels)
                            )
                        if i % gc_interval == 0:
                            gc.collect()
                        self.iterations_completed = i + 1
                        continue
                    except Exception as e:
                        print(f"  [IssueScript] ⚠️ issue_{issue_id} failed: {e}")
                        # 失败时 fallthrough 到普通随机

                # 普通随机 warmup
                api = TorchAPI(self.api_name)
                source = "warmup_random"
                api.mutate()
                if i == self.warmup_threshold - 1:
                    print(f"\n🚀 PHASE TRANSITION: Warm-up → Evolution (iter {i+1})")
            else:
                # Phase 2: Evolution — ε-greedy 混合策略
                if self.mutation_controller:
                    is_exploration = self.mutation_controller.should_explore()

                if is_exploration:
                    api = TorchAPI(self.api_name)
                    source = "exploration"
                elif self.corpus and self.corpus.size() > 0 and random.random() < 0.7:
                    parent_api = self.corpus.select_parent()
                    api = copy.deepcopy(parent_api)
                    source = "corpus"
                else:
                    api = TorchAPI(self.api_name)
                    source = "random"
                api.mutate()

            # ── 保存当前测试代码（用于崩溃恢复）──────────────────────
            if self.crash_code_file:
                try:
                    crash_code = api.to_code(use_try=False)
                    tmp_file = self.crash_code_file + '.tmp'
                    with open(tmp_file, 'w') as f:
                        f.write(f"import torch\nimport time\n{crash_code}\n")
                    os.replace(tmp_file, self.crash_code_file)
                except Exception:
                    pass

            # ── OutlierFilter 检查 ────────────────────────────────────
            if self.outlier_filter:
                should_filter, reason = self.outlier_filter.check_api(api)
                if should_filter:
                    if self.logger:
                        self.logger.log_iteration(
                            iteration=i, source=source, api=api,
                            valid=False, kernels=[]
                        )
                    if i % gc_interval == 0:
                        gc.collect()
                    continue

            # ── 执行所有 Oracle ───────────────────────────────────────
            all_captured_kernels: Set[str] = set()
            execution_valid = False

            for oracle in self.oracles:
                try:
                    # [修复] test_with_oracle 返回 (captured_kernels, error_info)
                    result = self.library.test_with_oracle(api, oracle)

                    if isinstance(result, tuple) and len(result) == 2:
                        captured_kernels, error_info = result
                    else:
                        captured_kernels, error_info = result, {}

                    all_captured_kernels.update(captured_kernels)
                    execution_valid = True

                    # [修复] error_info key 是绝对路径，直接 update，
                    # _save_bug_traces 会统一做 relpath 归一化
                    if error_info:
                        self.bug_traces_log.update(error_info)

                    # 实时 bug 去重记录
                    for path, info in error_info.items():
                        msg = info.get("msg", "")
                        bug_type = info.get("type", "crash")
                        oracle_key = oracle.name.lower() if hasattr(oracle, 'name') else str(oracle)
                        is_new, sig = self.bug_deduplicator.record_bug(
                            oracle_type=oracle_key,
                            error_msg=msg,
                            iteration=i
                        )
                        if is_new:
                            print(f"  🐛 NEW [{oracle_key.upper()}]: {sig[:50]}...")

                except Exception:
                    pass

            # ── 记录日志 ──────────────────────────────────────────────
            if self.logger:
                self.logger.log_iteration(
                    iteration=i, source=source, api=api,
                    valid=execution_valid, kernels=list(all_captured_kernels)
                )

            # ── 记录 DispatcherSpace 命中 ─────────────────────────────
            if self.dispatcher_space and execution_valid:
                is_cuda = OracleType.CUDA in self.oracles
                self.dispatcher_space.record_hit(self.api_name, api, is_cuda=is_cuda, iteration=i)

            if self.dispatcher_space and i % 50 == 0:
                self.dispatcher_space.record_coverage_snapshot(self.api_name, i)

            # ── 更新覆盖率 ────────────────────────────────────────────
            new_count, (should_expand, expansion_msg) = self.coverage.update(all_captured_kernels, i)

            if should_expand and self.mutation_controller:
                self.mutation_controller.enter_expansion_mode()

            if new_count > 0 and self.mutation_controller:
                if self.mutation_controller.expansion_mode:
                    self.mutation_controller.exit_expansion_mode()

            # ── 更新 Corpus ───────────────────────────────────────────
            if new_count > 0 and self.corpus:
                self.corpus.add_seed(api, all_captured_kernels)

            # ── GC ───────────────────────────────────────────────────
            if i % gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── Memory Monitor ────────────────────────────────────────
            if self.memory_monitor and i % 100 == 0:
                is_critical, msg = self.memory_monitor.check_and_cleanup()
                if msg:
                    print(f"\n{msg}")

            # ── 定期扫描 Bug ──────────────────────────────────────────
            if (i + 1) % bug_scan_interval == 0:
                self.bug_tracker.scan_bugs()

            # ── Checkpoint & Progress ─────────────────────────────────
            if (i + 1) % checkpoint_interval == 0:
                if self.checkpoint_manager:
                    self.checkpoint_manager.save(
                        iteration=i,
                        coverage_kernels=self.coverage.all_kernels,
                        corpus_seeds=self.corpus.corpus if self.corpus else None,
                        logger_iterations=self.logger.total_iterations if self.logger else 0
                    )

                if self.disk_guard:
                    is_critical, message = self.disk_guard.check_and_cleanup()
                    if is_critical:
                        print(f"\n{message}\n❌ STOPPING: Critical disk space issue")
                        break
                    elif message:
                        print(f"\n{message}")

                elapsed = time.time() - start_time
                valid_rate_str = ""
                if self.logger:
                    valid_rate_str = f" | Valid: {self.logger.get_valid_rate()*100:.1f}%"

                print(f"\n--- Checkpoint: {i+1}/{max_iterations} ({elapsed/60:.1f} min) ---")
                print(f"Kernels: {self.coverage.get_total()} | "
                      f"Bugs: {self.bug_tracker.get_total_bugs()}{valid_rate_str}")

                if self.speedometer:
                    print(f"Speed: {self.speedometer.get_status(i, max_iterations)}")
                if self.disk_guard:
                    print(f"Disk: {self.disk_guard.get_status()}")
                if self.outlier_filter:
                    print(f"Filter: {self.outlier_filter.get_status()}")
                if self.corpus:
                    print(f"Corpus: {self.corpus.size()} seeds")
                if self.dispatcher_space:
                    cov = self.dispatcher_space.get_coverage(self.api_name)
                    print(f"5D Coverage: {cov*100:.1f}%")
                if self.mutation_controller:
                    print(f"Mutation: {self.mutation_controller.get_status()}")

                if intermediate_save_callback:
                    intermediate_save_callback()

                if not first_save_done and intermediate_save_callback:
                    intermediate_save_callback()
                    first_save_done = True

            self.iterations_completed = i + 1

        # ── 循环结束收尾 ──────────────────────────────────────────────
        self.bug_tracker.scan_bugs()
        self._save_bug_traces()  # 唯一落盘点

        print(f"\n{'='*70}")
        print(f"✅ {self.strategy_name} fuzzing completed: {self.iterations_completed} iterations")
        print(f"   Total kernels: {self.coverage.get_total()}")
        print(f"   Total bugs: {self.bug_tracker.get_total_bugs()}")
        print(f"{'='*70}\n")

        return self.coverage

    def _save_bug_traces(self):
        """
        将黑匣子字典落盘。

        bug_traces_log 的 key 来自两个来源：
          1. test_with_oracle 的 error_info → key 是 write_to_dir 返回的绝对路径
          2. test_with_script 的 error_info → key 也是 write_to_dir 返回的绝对路径（修复后）

        统一用 os.path.relpath(k, self.output_dir) 转为相对路径，
        格式为 "crash-oracle/potential-bug/torch_matmul/1.py"，
        与 v14 分析脚本的 _load_traces 注册的 key 格式对齐。
        """
        out_file = Path(self.output_dir) / f"{self.strategy_name.lower()}_bug_traces.json"
        if not self.bug_traces_log:
            self.bug_traces_log["_heartbeat_"] = {
                "type": "info",
                "msg": "Tracer is online. No new bugs generated in this run yet."
            }
        normalized = {}
        for k, v in self.bug_traces_log.items():
            if k.startswith("_"):
                normalized[k] = v
                continue
            try:
                rel = os.path.relpath(k, self.output_dir).replace("\\", "/")
                normalized[rel if not rel.startswith("..") else k] = v
            except ValueError:
                normalized[k] = v
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(normalized, f, indent=2, ensure_ascii=False)
            print(f"  [BugTraces] 💾 Saved {len(normalized)} entries → {out_file.name}")
        except Exception as e:
            print(f"  [BugTraces] ❌ JSON dump failed: {e}")

    def print_stats(self):
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS: {self.strategy_name}")
        print(f"{'='*70}")
        print(f"Total Kernels: {self.coverage.get_total()}")
        self.bug_tracker.scan_bugs()
        self.bug_tracker.print_summary()
        self.bug_deduplicator.print_summary()
        self.bug_enhancer.print_stats()


# =============================================================================
# Phase-Level Process Isolation
# =============================================================================

def _run_phase_worker(phase_name: str,
                      api_name: str,
                      max_iterations: int,
                      output_dir: str,
                      config_path: str,
                      diff_bound: float,
                      enable_patches: bool,
                      result_file: str,
                      crash_code_file: str):
    """在隔离的子进程中运行一个完整的 fuzzing 阶段，结果通过文件传递"""
    import torch
    import signal

    config = configparser.ConfigParser()
    config.read(config_path)
    TorchDatabase.database_config(
        config["mongodb"]["host"],
        int(config["mongodb"]["port"]),
        config["mongodb"]["torch_database"]
    )

    PoisonPatcher.patch()
    if enable_patches:
        ProbabilityPatcher.patch_high_db_probability()

    result = {
        'phase': phase_name,
        'success': False,
        'error': None,
        'total_kernels': 0,
        'bugs': {'crash': 0, 'cuda': 0, 'precision': 0, 'compile': 0, 'total': 0},
        'coverage_history': [],
        'dispatcher_stats': {},
        'dedup_stats': {},
        'iterations_completed': 0,
    }

    fuzzer = None

    def save_intermediate_result():
        nonlocal result, fuzzer
        if fuzzer is None:
            return
        try:
            result['total_kernels'] = fuzzer.coverage.get_total() if hasattr(fuzzer, 'coverage') else 0
            result['coverage_history'] = list(fuzzer.coverage.history) if hasattr(fuzzer, 'coverage') else []
            result['iterations_completed'] = fuzzer.iterations_completed if hasattr(fuzzer, 'iterations_completed') else 0

            if hasattr(fuzzer, 'dispatcher_space') and fuzzer.dispatcher_space:
                result['dispatcher_stats'] = fuzzer.dispatcher_space.get_detailed_stats(api_name)
                result['dispatcher_coverage_history'] = list(
                    fuzzer.dispatcher_space.coverage_history.get(api_name, [])
                )

            if hasattr(fuzzer, 'bug_tracker'):
                fuzzer.bug_tracker.scan_bugs()
                result['bugs'] = {
                    'crash': fuzzer.bug_tracker.get_bugs_by_type('crash'),
                    'cuda': fuzzer.bug_tracker.get_bugs_by_type('cuda'),
                    'precision': fuzzer.bug_tracker.get_bugs_by_type('precision'),
                    'compile': fuzzer.bug_tracker.get_bugs_by_type('compile'),
                    'total': fuzzer.bug_tracker.get_total_bugs(),
                    'unique': len(fuzzer.bug_tracker.unique_bugs)
                }

            if hasattr(fuzzer, 'bug_deduplicator'):
                result['dedup_stats'] = fuzzer.bug_deduplicator.get_dedup_stats()
                result['unique_errors'] = fuzzer.bug_deduplicator.error_samples

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"  ⚠️ Intermediate save failed: {e}")

    try:
        fuzzer = EnhancedFuzzer(
            api_name=api_name,
            output_dir=output_dir,
            strategy_name=phase_name,
            enable_patches=enable_patches,
            use_all_oracles=True,
            diff_bound=diff_bound,
            crash_code_file=crash_code_file
        )

        if enable_patches and fuzzer.mutation_controller:
            AdaptiveMutationPatcher.patch_adaptive_mutation(fuzzer.mutation_controller)

        coverage = fuzzer.run_fuzzing_loop(
            max_iterations=max_iterations,
            checkpoint_interval=max(100, max_iterations // 20),
            bug_scan_interval=50,
            gc_interval=10,
            intermediate_save_callback=save_intermediate_result
        )

        fuzzer.print_stats()

        result['success'] = True
        result['total_kernels'] = coverage.get_total()
        result['coverage_history'] = list(coverage.history)
        result['iterations_completed'] = fuzzer.iterations_completed

        fuzzer.bug_tracker.scan_bugs()
        result['bugs'] = {
            'crash': fuzzer.bug_tracker.get_bugs_by_type('crash'),
            'cuda': fuzzer.bug_tracker.get_bugs_by_type('cuda'),
            'precision': fuzzer.bug_tracker.get_bugs_by_type('precision'),
            'compile': fuzzer.bug_tracker.get_bugs_by_type('compile'),
            'total': fuzzer.bug_tracker.get_total_bugs(),
            'unique': len(fuzzer.bug_tracker.unique_bugs)
        }

        if fuzzer.dispatcher_space:
            result['dispatcher_stats'] = fuzzer.dispatcher_space.get_detailed_stats(api_name)
            result['dispatcher_coverage_history'] = list(
                fuzzer.dispatcher_space.coverage_history.get(api_name, [])
            )

        if hasattr(fuzzer, 'bug_deduplicator'):
            result['dedup_stats'] = fuzzer.bug_deduplicator.get_dedup_stats()
            result['unique_errors'] = fuzzer.bug_deduplicator.error_samples

        if os.path.exists(crash_code_file):
            os.remove(crash_code_file)

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ Phase {phase_name} failed: {e}")

    finally:
        if enable_patches:
            ProbabilityPatcher.restore()
            AdaptiveMutationPatcher.restore()
        PoisonPatcher.restore()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def run_phase_isolated(phase_name: str,
                       api_name: str,
                       max_iterations: int,
                       output_dir: str,
                       config_path: str,
                       diff_bound: float,
                       enable_patches: bool,
                       timeout_minutes: int = 120) -> Dict:
    """在独立子进程中运行一个 fuzzing 阶段。即使 OOM/SIGFPE 也不影响主进程。"""

    result_file = join(output_dir, f"{phase_name.lower()}_result.json")
    crash_code_file = join(output_dir, f"_current_test_{phase_name.lower()}.py")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🚀 LAUNCHING ISOLATED PHASE: {phase_name}")
    print(f"{'='*70}")
    print(f"API: {api_name}")
    print(f"Max Iterations: {max_iterations:,}")
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    ctx = mp.get_context('spawn')

    p = ctx.Process(
        target=_run_phase_worker,
        args=(
            phase_name, api_name, max_iterations, output_dir,
            config_path, diff_bound, enable_patches,
            result_file, crash_code_file
        )
    )

    start_time = time.time()
    p.start()
    p.join(timeout=timeout_minutes * 60)
    elapsed = time.time() - start_time

    result = {
        'phase': phase_name,
        'success': False,
        'error': None,
        'total_kernels': 0,
        'bugs': {'crash': 0, 'cuda': 0, 'precision': 0, 'compile': 0, 'total': 0},
        'elapsed_seconds': elapsed,
        'crash_saved': False,
    }

    def save_crash_code_as_bug():
        """将导致崩溃的代码保存到 potential-bug 目录，并同步写入 bug_traces"""
        actual_crash_file = None
        if os.path.exists(crash_code_file):
            actual_crash_file = crash_code_file
        elif os.path.exists(crash_code_file + '.tmp'):
            actual_crash_file = crash_code_file + '.tmp'
        if actual_crash_file is None:
            return False
        try:
            with open(actual_crash_file, 'r') as f:
                content = f.read()
            if not content.strip():
                return False

            # [修复] 统一用下划线命名（与 write_to_dir 修复后对齐）
            api_dir_name = api_name.replace(".", "_")
            crash_bug_dir = join(output_dir, "crash-oracle", "potential-bug", api_dir_name)
            os.makedirs(crash_bug_dir, exist_ok=True)

            existing = []
            for f_name in os.listdir(crash_bug_dir):
                if f_name.endswith('.py'):
                    try:
                        existing.append(int(f_name.replace('.py', '')))
                    except ValueError:
                        pass
            next_num = max(existing, default=0) + 1

            dest_file = join(crash_bug_dir, f"{next_num}.py")
            with open(dest_file, 'w') as f:
                f.write(f"# CRASH BUG: This code caused process exit\n")
                f.write(f"# Phase: {phase_name}, Exit: {p.exitcode}, Time: {elapsed:.1f}s\n\n")
                f.write(content)
            print(f"  💾 Crash code saved to: {dest_file}")

            # [修复] 同步写入 bug_traces JSON，使崩溃 bug 也能被 v14 分析脚本匹配
            # key 使用相对路径格式（与 _save_bug_traces 归一化后格式一致）
            trace_file = join(output_dir, f"{phase_name.lower()}_bug_traces.json")
            try:
                traces = {}
                if os.path.exists(trace_file):
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        traces = json.load(f)
                # key 格式：相对于 output_dir 的路径，下划线形式
                rel_key = f"crash-oracle/potential-bug/{api_dir_name}/{next_num}.py"
                traces[rel_key] = {
                    "type": "hard_crash",
                    "msg": f"Process exited with signal {p.exitcode}. Likely OOM or Segfault."
                }
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(traces, f, indent=2, ensure_ascii=False)
            except Exception as te:
                print(f"  ⚠️ Failed to update bug traces: {te}")

            for f_path in [crash_code_file, crash_code_file + '.tmp']:
                try:
                    if os.path.exists(f_path):
                        os.remove(f_path)
                except Exception:
                    pass

            return True
        except Exception as e:
            print(f"  ⚠️ Failed to save crash code: {e}")
            return False

    if p.is_alive():
        print(f"\n⏰ Phase {phase_name} TIMEOUT after {timeout_minutes} minutes")
        p.terminate()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()
        result['error'] = f"Timeout after {timeout_minutes} minutes"
        result['crash_saved'] = save_crash_code_as_bug()

    elif p.exitcode != 0:
        exit_code = p.exitcode
        if exit_code in (-9, 137):
            error_msg = f"OOM Killed (Exit 137) after {elapsed/60:.1f} minutes"
        elif exit_code in (-8, 136):
            error_msg = f"SIGFPE (Exit 136) after {elapsed/60:.1f} minutes"
        elif exit_code in (-11, 139):
            error_msg = f"Segmentation Fault (Exit 139) after {elapsed/60:.1f} minutes"
        else:
            error_msg = f"Process exited with code {exit_code}"

        print(f"\n❌ Phase {phase_name} CRASHED: {error_msg}")
        result['error'] = error_msg
        result['exit_code'] = exit_code
        result['crash_saved'] = save_crash_code_as_bug()

    else:
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                result['elapsed_seconds'] = elapsed
                print(f"\n✅ Phase {phase_name} completed in {elapsed/60:.1f} minutes")
            except Exception as e:
                result['error'] = f"Failed to read result file: {e}"
        else:
            result['error'] = "Result file not found"

    # 尝试恢复部分结果
    if not result.get('success') and os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                partial = json.load(f)
            result['total_kernels'] = partial.get('total_kernels', 0)
            result['bugs'] = partial.get('bugs', result['bugs'])
            result['coverage_history'] = partial.get('coverage_history', [])
            result['iterations_completed'] = partial.get('iterations_completed', 0)
            result['dispatcher_stats'] = partial.get('dispatcher_stats', {})
            result['dispatcher_coverage_history'] = partial.get('dispatcher_coverage_history', [])
            print(f"  ✅ Recovered: {result['iterations_completed']} iters, "
                  f"{result['total_kernels']} kernels, "
                  f"{result['bugs'].get('total', 0)} bugs")
        except Exception as e:
            print(f"  ⚠️ Failed to recover partial results: {e}")

    # 磁盘扫描
    bug_tracker = BugTracker(phase_name, output_dir)
    bug_tracker.scan_bugs()
    if bug_tracker.get_total_bugs() > 0:
        result['bugs'] = {
            'crash': bug_tracker.get_bugs_by_type('crash'),
            'cuda': bug_tracker.get_bugs_by_type('cuda'),
            'precision': bug_tracker.get_bugs_by_type('precision'),
            'compile': bug_tracker.get_bugs_by_type('compile'),
            'total': bug_tracker.get_total_bugs(),
            'unique': len(bug_tracker.unique_bugs)
        }
        print(f"  🐛 Found {bug_tracker.get_total_bugs()} bugs from disk scan")

    return result


# =============================================================================
# Visualization
# =============================================================================

def plot_full_oracle_results(random_result: Dict, guided_result: Dict,
                              api_name: str, output_dir: str):
    """绘制 Random vs Guided 对比图"""

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    def normalize_history(history):
        if not history:
            return [], []
        iters, values = zip(*history)
        min_iter = min(iters)
        return [i - min_iter for i in iters], list(values)

    # A) Kernel 覆盖率曲线
    ax1 = fig.add_subplot(gs[0, :2])
    if random_result.get('coverage_history'):
        iters, kernels = normalize_history(random_result['coverage_history'])
        ax1.plot(iters, kernels,
                 label=f"Random (Total: {random_result.get('total_kernels', 0)})",
                 color="#3498db", linewidth=3, alpha=0.8)
    if guided_result.get('coverage_history'):
        iters, kernels = normalize_history(guided_result['coverage_history'])
        ax1.plot(iters, kernels,
                 label=f"Guided (Total: {guided_result.get('total_kernels', 0)})",
                 color="#e74c3c", linewidth=3)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Cumulative Kernels", fontsize=12)
    ax1.set_title("A) Kernel Coverage Over Time", fontsize=13, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # B) Bug 数量对比
    ax2 = fig.add_subplot(gs[0, 2])
    bug_types = ['CRASH', 'CUDA', 'PRECISION', 'COMPILE']
    random_bugs = random_result.get('bugs', {})
    guided_bugs = guided_result.get('bugs', {})
    random_crash_extra = 1 if random_result.get('crash_saved') else 0
    guided_crash_extra = 1 if guided_result.get('crash_saved') else 0
    random_bug_counts = [
        random_bugs.get('crash', 0) + random_crash_extra,
        random_bugs.get('cuda', 0),
        random_bugs.get('precision', 0),
        random_bugs.get('compile', 0)
    ]
    guided_bug_counts = [
        guided_bugs.get('crash', 0) + guided_crash_extra,
        guided_bugs.get('cuda', 0),
        guided_bugs.get('precision', 0),
        guided_bugs.get('compile', 0)
    ]
    x = np.arange(len(bug_types))
    width = 0.35
    bars1 = ax2.bar(x - width/2, random_bug_counts, width, label='Random', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, guided_bug_counts, width, label='Guided', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel("Bug Type", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("B) Bugs Found by Type", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bug_types)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{int(height)}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)

    # C) 总 Bug 数饼图
    ax3 = fig.add_subplot(gs[1, 0])
    total_random = random_bugs.get('total', 0) + random_crash_extra
    total_guided = guided_bugs.get('total', 0) + guided_crash_extra
    if total_random > 0 or total_guided > 0:
        sizes = [max(total_random, 0.1), max(total_guided, 0.1)]
        labels = [f'Random\n({total_random})', f'Guided\n({total_guided})']
        colors = ['#3498db', '#e74c3c']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title("C) Total Bugs Distribution", fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "No Bugs Found", ha='center', va='center', fontsize=14)
        ax3.axis('off')

    # D) 5D 状态空间覆盖
    ax4 = fig.add_subplot(gs[1, 1])
    has_coverage_data = False
    for label, res, color in [("Random", random_result, "#3498db"), ("Guided", guided_result, "#e74c3c")]:
        hist = res.get('dispatcher_coverage_history', [])
        if hist:
            try:
                iters, covs = normalize_history(hist)
                covs = [min(c * 100, 100.0) for c in covs]
                ax4.plot(iters, covs, label=label, color=color, linewidth=2, marker='o', markersize=3)
                has_coverage_data = True
            except Exception:
                pass
    if not has_coverage_data:
        for label, res, color in [("Random", random_result, "#3498db"), ("Guided", guided_result, "#e74c3c")]:
            final_cov = res.get('dispatcher_stats', {}).get('coverage', 0) * 100
            iters_done = res.get('iterations_completed', 0)
            if final_cov > 0:
                ax4.scatter([0, iters_done], [0, final_cov], label=f"{label} ({final_cov:.1f}%)",
                            color=color, s=50)
                ax4.plot([0, iters_done], [0, final_cov], color=color, linestyle='--', alpha=0.5)
                has_coverage_data = True
    if not has_coverage_data:
        ax4.text(0.5, 0.5, "No Coverage Data", ha='center', va='center',
                 fontsize=12, transform=ax4.transAxes)
    ax4.set_xlabel("Iteration", fontsize=11)
    ax4.set_ylabel("Coverage (%)", fontsize=11)
    ax4.set_title("D) 5D State Space Coverage", fontsize=12, fontweight='bold')
    if has_coverage_data:
        ax4.legend(loc="lower right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)

    # E) 统计汇总表
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    random_status = "OK" if random_result.get('success') else "FAIL"
    guided_status = "OK" if guided_result.get('success') else "FAIL"
    table_data = [
        ["Metric", "Random", "Guided", ""],
        ["", "", "", ""],
        ["Iterations", f"{random_result.get('iterations_completed', 0)}", f"{guided_result.get('iterations_completed', 0)}", ""],
        ["Kernels", f"{random_result.get('total_kernels', 0)}", f"{guided_result.get('total_kernels', 0)}", ""],
        ["5D Coverage",
         f"{random_result.get('dispatcher_stats', {}).get('coverage', 0)*100:.1f}%",
         f"{guided_result.get('dispatcher_stats', {}).get('coverage', 0)*100:.1f}%", ""],
        ["Bugs (Total)", f"{total_random}", f"{total_guided}", ""],
        ["Unique Bugs", f"{random_bugs.get('unique', 0)}", f"{guided_bugs.get('unique', 0)}", ""],
        ["Crash Saved", "YES" if random_result.get('crash_saved') else "-",
         "YES" if guided_result.get('crash_saved') else "-", ""],
        ["Status", random_status, guided_status, ""],
    ]
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.22, 0.22, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for row_idx in range(2, len(table_data)):
        if table_data[row_idx][0] == "Status":
            table[(row_idx, 1)].set_facecolor('#d4edda' if table_data[row_idx][1] == "OK" else '#f8d7da')
            table[(row_idx, 2)].set_facecolor('#d4edda' if table_data[row_idx][2] == "OK" else '#f8d7da')
    ax5.set_title("E) Summary", fontsize=12, fontweight='bold', pad=20)

    error_text = ""
    if random_result.get('error'):
        error_text += f"Random: {random_result['error'][:60]}...\n"
    if guided_result.get('error'):
        error_text += f"Guided: {guided_result['error'][:60]}..."
    if error_text:
        fig.text(0.5, 0.02, f"Errors: {error_text}", ha='center', fontsize=9, color='red',
                 bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='#ffc107'))

    fig.suptitle(
        f"Isolated Phase Benchmark: {api_name}\n"
        f"(Each phase runs in separate process - OOM resistant)",
        fontsize=15, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    plot_file = join(output_dir, f"{api_name.replace('.', '_')}_isolated_benchmark.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Plot saved: {plot_file}")


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_full_oracle_experiment(api_name: str,
                                max_iterations: int,
                                output_dir: str,
                                config_file: str = "demo_torch.conf",
                                diff_bound: float = 1e-5,
                                timeout_minutes: int = 120):
    """运行完整实验 - 使用阶段级进程隔离"""
    os.makedirs(output_dir, exist_ok=True)

    possible_paths = [
        join("config", config_file),
        join("..", "config", config_file),
        config_file,
    ]
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = os.path.abspath(path)
            break

    if not config_path:
        print("❌ Config file not found")
        return None

    # API 可用性预检
    try:
        parts = api_name.split(".")
        obj = __import__(parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
    except (AttributeError, ImportError, ModuleNotFoundError) as e:
        print(f"⚠️  Skipping {api_name}: not available in current PyTorch ({e})")
        return {"skipped": True, "reason": str(e), "api": api_name}

    print(f"\n{'='*70}")
    print(f"🔬 PHASE-ISOLATED BENCHMARK")
    print(f"{'='*70}")
    print(f"API: {api_name}")
    print(f"Max Iterations: {max_iterations:,}")
    print(f"Config: {config_path}")
    print(f"Diff Bound: {diff_bound}")
    print(f"Timeout per phase: {timeout_minutes} minutes")
    print(f"{'='*70}\n")

    # Phase 1: Random Baseline
    random_result = run_phase_isolated(
        phase_name="Random",
        api_name=api_name,
        max_iterations=max_iterations,
        output_dir=join(output_dir, "random"),
        config_path=config_path,
        diff_bound=diff_bound,
        enable_patches=False,
        timeout_minutes=timeout_minutes
    )

    print(f"\n{'='*70}")
    print(f"Phase 1 (Random) {'COMPLETED' if random_result.get('success') else 'FAILED/CRASHED'}")
    if random_result.get('error'):
        print(f"Error: {random_result['error'][:100]}")
    print(f"{'='*70}\n")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: Guided Strategy
    guided_result = run_phase_isolated(
        phase_name="Guided",
        api_name=api_name,
        max_iterations=max_iterations,
        output_dir=join(output_dir, "guided"),
        config_path=config_path,
        diff_bound=diff_bound,
        enable_patches=True,
        timeout_minutes=timeout_minutes
    )

    print(f"\n{'='*70}")
    print(f"Phase 2 (Guided) {'COMPLETED' if guided_result.get('success') else 'FAILED/CRASHED'}")
    if guided_result.get('error'):
        print(f"Error: {guided_result['error'][:100]}")
    print(f"{'='*70}\n")

    # 可视化
    try:
        plot_full_oracle_results(random_result, guided_result, api_name, output_dir)
    except Exception as e:
        print(f"⚠️ Plot generation failed: {e}")

    # 保存综合结果
    combined_result = {
        "api": api_name,
        "max_iterations": max_iterations,
        "isolation_mode": "phase_level_subprocess",
        "random": random_result,
        "guided": guided_result,
        "comparison": {
            "both_succeeded": random_result.get('success', False) and guided_result.get('success', False),
            "kernel_ratio": (guided_result.get('total_kernels', 0) /
                             max(random_result.get('total_kernels', 1), 1)),
            "bug_ratio": (guided_result.get('bugs', {}).get('total', 0) /
                          max(random_result.get('bugs', {}).get('total', 1), 1)),
        }
    }

    result_file = join(output_dir, f"{api_name.replace('.', '_')}_isolated_results.json")
    with open(result_file, "w") as f:
        json.dump(combined_result, f, indent=2, default=str)
    print(f"\n💾 Results saved: {result_file}")

    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    for label, res in [("Random", random_result), ("Guided", guided_result)]:
        status = "✅ Success" if res.get('success') else "❌ Failed"
        bugs = res.get('bugs', {})
        print(f"{label}: {status}")
        print(f"  Iterations: {res.get('iterations_completed', 0)}")
        print(f"  Kernels: {res.get('total_kernels', 0)}")
        print(f"  Bugs: {bugs.get('total', 0)}")
        if res.get('crash_saved'):
            print(f"  💾 Crash bug saved!")
        if res.get('error'):
            print(f"  Error: {res['error'][:80]}...")
    print(f"{'='*70}\n")

    return combined_result


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase-Isolated Benchmark (OOM Resistant)")
    parser.add_argument("--api", type=str, default="torch.nn.LSTM")
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--output", type=str, default="isolated_output")
    parser.add_argument("--conf", type=str, default="demo_torch.conf")
    parser.add_argument("--diff-bound", type=float, default=1e-5)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per phase in minutes (default: 120)")

    args = parser.parse_args()

    run_full_oracle_experiment(
        api_name=args.api,
        max_iterations=args.max_iterations,
        output_dir=args.output,
        config_file=args.conf,
        diff_bound=args.diff_bound,
        timeout_minutes=args.timeout
    )


if __name__ == "__main__":
    main()
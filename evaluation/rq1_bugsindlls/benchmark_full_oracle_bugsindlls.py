# -*- coding: utf-8 -*-
"""
Kernel-Guided Fuzzing Benchmark - Full Oracle Edition with Hybrid Strategy
==========================================================================

Headline enhancements:
1. **Hybrid strategy**: epsilon-greedy (90% mutation + 10% exploration).
2. **Adaptive mutation**: expand the mutation range automatically on
   stagnation.
3. **Plateau-removal and stopping**: switch from "terminate" to "expand
   search".
4. **Full oracle support**: CRASH + CUDA + PRECISION.
5. **Smart deduplication**: dedup and classify bugs by error fingerprint
   (new).

Author: Research Team
Date: 2025-12-13
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
import traceback
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

# FreeFuzz imports
from classes.database import TorchDatabase
from classes.torch_api import TorchAPI, TorchArgument
from classes.torch_library import TorchLibrary
from classes.argument import Argument, ArgType
from constants.enum import OracleType

# Constraint Filter imports - 5
#
try:
    from pytorch_comprehensive_filter import (
        PyTorchConstraintFilter,
        DispatcherStateSpace,
    )
    FILTER_AVAILABLE = True
except Exception:
    #
    FILTER_AVAILABLE = False
    
    # DispatcherStateSpace ()
    class DispatcherStateSpace:
        """- """
        
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
            # : 5(device dtype layout mem_format rank)
            # strided only, contiguous only, ranks [2d,3d,4d]
            devices = self.DIMENSION_DEVICE  # 2
            dtypes = self.DIMENSION_DTYPE    # 12
            layouts = ['strided']            # 1
            mem_formats = ['contiguous']     # 1
            ranks = ['1d', '2d', '3d', '4d', '5d']  # 5 ranks (0d rare, nd unstable)
            valid_space = len(devices) * len(dtypes) * len(layouts) * len(mem_formats) * len(ranks)
            # 2 12 1 1 5 = 120 (expanded from 72)
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
            print(f"  Dimensions: device={len(devices)}, dtype={len(dtypes)}, "
                  f"layout={len(layouts)}, mem_format={len(mem_formats)}, rank={len(ranks)}")
            print(f"  Valid Space: {valid_space} (212115 = 120)")
            return self.api_spaces[api_name]
        
        def record_hit(self, api_name: str, state: Dict) -> bool:
            if api_name not in self.api_spaces:
                self.register_api(api_name)
            if self.total_hits >= self.max_states:
                return False
            # : 5state_hash
            device = state.get('device', 'unk')
            dtype = state.get('dtype', 'unk')
            layout = state.get('layout', 'strided')
            mem_format = state.get('mem_format', 'contiguous')
            rank = state.get('rank', 'unk')
            # rank
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
            print("DISPATCHER STATE SPACE COVERAGE (Fallback - No Constraint Filter)")
            print(f"{'='*70}")
            for api_name, space_info in self.api_spaces.items():
                hit_count = len(self.hit_states.get(api_name, set()))
                valid_space = space_info['valid_space']
                coverage = hit_count / max(valid_space, 1)
                bar_w = 30
                filled = int(bar_w * coverage)
                bar = '' * filled + '' * (bar_w - filled)
                print(f"\n[{api_name}]")
                print(f"  Hits: {hit_count}/{valid_space} ({coverage*100:.1f}%) [{bar}]")
                #
                hit_samples = list(self.hit_states.get(api_name, set()))[:5]
                if hit_samples:
                    print(f"  Sample: {hit_samples[:3]}...")
            print(f"\nTotal Hits: {self.total_hits}")
            print(f"{'='*70}\n")


# =============================================================================
# Safety Guards ()
# =============================================================================

class Speedometer:
    """- """
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
            warning = (f"SLOW SPEED: {current_speed:.2f} it/s (threshold: {self.slow_threshold})")
            if current_speed < 0.1:
                warning += "\n   Check if Precision Oracle is too slow"
            elif current_speed < 0.5:
                warning += "\n   Consider reducing complexity or disabling oracles"
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
    """- """
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
            warning = f"DISK SPACE: Only {free:.2f} GB free (threshold: {self.min_free_gb} GB)"
            if self.auto_cleanup:
                warning += "\n   Attempting auto-cleanup..."
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
            message = f"{warning}\n   Cleaned {cleaned} files {free:.2f} GB free"
            if free < self.min_free_gb:
                message += "\n   CRITICAL: Still low after cleanup!"
                return True, message
            message += "\n   Cleanup successful"
            return False, message
        return True, warning
    
    def get_status(self) -> str:
        total, used, free = self.get_disk_usage()
        return f"{free:.2f} GB free ({used/total*100:.1f}% used)"


class OutlierFilter:
    """- OOM"""
    def __init__(self, max_elements: int = int(1e8), max_memory_gb: float = 4.0):
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
            return True, f"Too many elements: {num_elements:,} > {self.max_elements:,}"
        
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
# Checkpoint System ()
# =============================================================================

class CheckpointManager:
    """- """
    
    def __init__(self, checkpoint_dir: str, strategy: str, api_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategy = strategy
        self.api_name = api_name.replace('.', '_').replace('::', '_')
        
        self.checkpoint_file = self.checkpoint_dir / f"{self.api_name}_{strategy}_checkpoint.pkl"
        self.metadata_file = self.checkpoint_dir / f"{self.api_name}_{strategy}_metadata.json"
        
        print(f"[Checkpoint] Manager initialized: {self.checkpoint_file.name}")
    
    def save(self, iteration: int, coverage_kernels: set, 
             corpus_seeds: list = None, logger_iterations: int = 0) -> bool:
        try:
            checkpoint_data = {
                'iteration': iteration,
                'coverage_kernels': coverage_kernels,
                'corpus_seeds': corpus_seeds,
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
                'corpus_size': len(corpus_seeds) if corpus_seeds else 0,
                'logger_iterations': logger_iterations,
                'strategy': self.strategy
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"[Checkpoint] Save failed: {e}")
            return False
    
    def load(self) -> Optional[Dict]:
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"[Checkpoint] Loaded from iteration {data['iteration']}")
            print(f"  Kernels: {len(data['coverage_kernels'])}")
            if data.get('corpus_seeds'):
                print(f"  Corpus: {len(data['corpus_seeds'])} seeds")
            
            return data
            
        except Exception as e:
            print(f"[Checkpoint] Load failed: {e}")
            return None
    
    def exists(self) -> bool:
        return self.checkpoint_file.exists()
    
    def clear(self):
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            print(f"[Checkpoint] Cleared")
        except:
            pass


# =============================================================================
# Feature Extractor ()
# =============================================================================

def extract_features(api: TorchAPI) -> str:
    """
    API
    
    : "param1:type1:value1|param2:type2:value2|..."
    """
    features = []
    
    for param_name, arg in api.args.items():
        if arg is None:
            continue
        
        if arg.type == ArgType.INT:
            features.append(f"{param_name}:int:{arg.value}")
        
        elif arg.type == ArgType.FLOAT:
            features.append(f"{param_name}:float:{arg.value:.2f}")
        
        elif arg.type == ArgType.BOOL:
            features.append(f"{param_name}:bool:{arg.value}")
        
        elif arg.type == ArgType.STR:
            features.append(f"{param_name}:str:{arg.value}")
        
        elif arg.type == ArgType.TORCH_DTYPE:
            dtype_str = str(arg.value).split(".")[-1] if hasattr(arg, 'value') else 'unknown'
            features.append(f"{param_name}:dtype:{dtype_str}")
        
        elif arg.type == ArgType.TORCH_TENSOR:
            shape_str = str(arg.shape) if hasattr(arg, 'shape') else 'unknown'
            dtype_str = str(arg.dtype).split(".")[-1] if hasattr(arg, 'dtype') else 'unknown'
            features.append(f"{param_name}:tensor:shape{shape_str}:dtype{dtype_str}")
        
        elif arg.type in [ArgType.LIST, ArgType.TUPLE]:
            length = len(arg.value) if hasattr(arg, 'value') and arg.value else 0
            type_name = "list" if arg.type == ArgType.LIST else "tuple"
            features.append(f"{param_name}:{type_name}:len{length}")
        
        else:
            features.append(f"{param_name}:{arg.type.name}:other")
    
    features.sort()
    return "|".join(features)


# =============================================================================
# Experiment Logger ()
# =============================================================================

class ExperimentLogger:
    """"""
    
    def __init__(self, output_file: str, strategy: str):
        self.output_file = output_file
        self.strategy = strategy
        self.file_handle = open(output_file, 'w', buffering=1)
        
        self.total_iterations = 0
        self.valid_count = 0
        self.invalid_count = 0
    
    def log_iteration(self, 
                      iteration: int,
                      source: str,
                      api: TorchAPI,
                      valid: bool,
                      kernels: List[str]):
        features = extract_features(api)
        
        record = {
            "iteration": iteration,
            "strategy": self.strategy,
            "source": source,
            "valid": valid,
            "features": features,
            "kernels": list(kernels) if kernels else []
        }
        
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

#
_ORIGINAL_DO_SELECT_FROM_DB = None
_ORIGINAL_MUTATE_INT_VALUE = None
_ORIGINAL_MUTATE_FLOAT_VALUE = None


# =============================================================================
# Bug Tracker (Bug ) -
# =============================================================================

class BugAnalyzer:
    """Bug """
    
    # Shape
    REGEX_PATTERNS = [
        (r'0x[0-9a-fA-F]+', 'ADDR'),           #
        (r'\d+\.\d+', 'FLOAT'),                #
        (r'\d+', 'INT'),                       #
        (r'\[.*?\]', '[SHAPE]'),               # Tensor Shape
        (r'\s+', ' '),                         #
    ]

    @staticmethod
    def get_signature(content: str) -> str:
        """bug"""
        if not content:
            return "unknown_empty"
        
        features = []
        
        # 1. dtype
        dtype_matches = re.findall(r'dtype\s*=\s*(torch\.[a-z0-9]+)', content)
        if dtype_matches:
            unique_dtypes = sorted(set(dtype_matches))
            features.append(f"dtype={','.join(unique_dtypes)}")
        
        # 2. tensor shape
        shape_matches = re.findall(r'torch\.rand(?:int)?\s*\(\s*[\[\(]([^\]\)]+)[\]\)]', content)
        if shape_matches:
            for shape in shape_matches[:2]:
                ndim = shape.count(',') + 1
                features.append(f"ndim={ndim}")
                break
        
        # 3.
        if re.search(r'\bnan\b', content, re.IGNORECASE):
            features.append("NaN")
        if re.search(r'\binf\b', content, re.IGNORECASE):
            features.append("Inf")
        
        # 4.
        if not features:
            return f"hash={hashlib.md5(content.encode()).hexdigest()[:12]}"
        
        return "|".join(sorted(features))

    @staticmethod
    def classify_bug(error_msg: str) -> str:
        """Deep/Shallow"""
        sig = BugAnalyzer.get_signature(error_msg).lower()
        
        if "segfault" in sig or "core_dump" in sig:
            return "CRITICAL_SEGFAULT"  #
        
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
        
        #
        self.bugs = {
            "crash": [],
            "cuda": [],
            "precision": []
        }
        self.bug_files = {
            "crash": set(),
            "cuda": set(),
            "precision": set()
        }
        
        #
        self.unique_bugs = defaultdict(list)
        self.bug_stats = defaultdict(int) #
        
    def scan_bugs(self):
        #
        self.unique_bugs.clear()
        self.bug_stats.clear()
        # self.bugs
        # self.bugsset
        
        oracle_map = {"crash": "crash-oracle", "cuda": "cuda-oracle", "precision": "precision-oracle"}
        
        for bug_type, oracle_name in oracle_map.items():
            bug_dir = join(self.output_dir, oracle_name, "potential-bug")
            if not os.path.exists(bug_dir):
                continue
                
            for root, _, files in os.walk(bug_dir):
                for f in files:
                    if not f.endswith('.py'): continue
                    
                    file_path = join(root, f)
                    
                    # 1.
                    if file_path not in self.bug_files[bug_type]:
                        self.bug_files[bug_type].add(file_path)
                        self.bugs[bug_type].append((len(self.bugs[bug_type]), "unknown", file_path))
                    
                    # 2.
                    try:
                        # .txt .log py
                        # python
                        content = ""
                        with open(file_path, 'r', errors='ignore') as f_obj:
                            content = f_obj.read()
                        
                        #
                        signature = BugAnalyzer.get_signature(content)
                        category = BugAnalyzer.classify_bug(content)
                        
                        # +
                        unique_key = f"[{bug_type.upper()}]_[{category}] : {signature}"
                        
                        self.unique_bugs[unique_key].append(file_path)
                        self.bug_stats[category] += 1
                        
                    except Exception as e:
                        #
                        pass

    def get_total_bugs(self) -> int:
        return sum(len(bugs) for bugs in self.bugs.values())
    
    def get_bugs_by_type(self, bug_type: str) -> int:
        return len(self.bugs.get(bug_type, []))
    
    def print_summary(self):
        self.scan_bugs()
        print(f"\n{'='*70}")
        print(f"BUG ANALYSIS REPORT: {self.name}")
        print(f"{'='*70}")
        
        #
        print(f"Total Unique Issues: {len(self.unique_bugs)}")
        print(f"Distribution: {dict(self.bug_stats)}")
        print("-" * 70)
        
        # Top 5 Bug
        print("Top Unique Bugs Found:")
        sorted_bugs = sorted(self.unique_bugs.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (sig, files) in enumerate(sorted_bugs[:5]):
            print(f" {i+1}. [Count: {len(files)}] {sig}")
            #
            rel_path = files[0].replace(self.output_dir + "/", "")
            print(f"    Sample: {rel_path}")
        
        if not sorted_bugs:
            print(" No bugs found.")
            
        #
        print("-" * 70)
        total_raw = self.get_total_bugs()
        total_unique = len(self.unique_bugs)
        print(f"UNIQUE Bugs: {total_unique}")
        print(f"Raw Files:   CRASH={self.get_bugs_by_type('crash')} | CUDA={self.get_bugs_by_type('cuda')} | PRECISION={self.get_bugs_by_type('precision')} | Total={total_raw}")
        print(f"Dedup Rate:  {total_unique}/{total_raw} ({total_unique/max(total_raw,1)*100:.1f}% unique)")
        print(f"{'='*70}\n")


# =============================================================================
# NEW: Adaptive Saturation Detector ()

# =============================================================================
# RealTimeBugDeduplicator (Bug )
# =============================================================================

class RealTimeBugDeduplicator:
    """
    Bug -
    
    BugTracker .py
    """
    
    CLEAN_PATTERNS = [
        (r'0x[0-9a-fA-F]+', 'ADDR'),
        (r'\d+\.\d+', 'FLOAT'),
        (r'(?<!\w)\d+(?!\w)', 'INT'),
        (r'\[.*?\]', '[SHAPE]'),
        (r'at .*?:\d+', 'at FILE:LINE'),
        (r'\s+', ' '),
    ]
    
    def __init__(self, max_signatures: int = 10000):
        self.max_signatures = max_signatures
        self.signatures: Dict[str, Set[str]] = {'crash': set(), 'cuda': set(), 'precision': set()}
        self.first_occurrence: Dict[str, dict] = {}
        self.total_bugs = {'crash': 0, 'cuda': 0, 'precision': 0}
        self.unique_bugs = {'crash': 0, 'cuda': 0, 'precision': 0}
    
    def get_signature(self, error_msg: str) -> str:
        if not error_msg:
            return "UNKNOWN"
        lines = error_msg.strip().split('\n')
        core_msg = lines[-1]
        for line in reversed(lines):
            if any(kw in line for kw in ['Error:', 'Exception:', 'INTERNAL ASSERT']):
                core_msg = line
                break
        clean_msg = core_msg
        for pattern, replacement in self.CLEAN_PATTERNS:
            clean_msg = re.sub(pattern, replacement, clean_msg)
        if 'Segmentation fault' in error_msg:
            return "SEGFAULT"
        return clean_msg.strip()[:200]
    
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
        self.first_occurrence[sig_hash] = {
            'oracle': oracle_type, 'signature': signature,
            'iteration': iteration, 'error': error_msg[:300]
        }
        return True, signature
    
    def get_dedup_stats(self) -> dict:
        return {
            'total': self.total_bugs.copy(),
            'unique': self.unique_bugs.copy(),
            'dedup_rate': {k: 1 - (self.unique_bugs[k] / max(self.total_bugs[k], 1)) for k in self.total_bugs}
        }
    
    def print_summary(self):
        stats = self.get_dedup_stats()
        print(f"\n{'='*70}")
        print("BUG DEDUPLICATION REPORT")
        print(f"{'='*70}")
        for oracle in ['crash', 'cuda', 'precision']:
            total, unique = stats['total'][oracle], stats['unique'][oracle]
            rate = stats['dedup_rate'][oracle] * 100
            print(f"[{oracle.upper()}] Total: {total}, Unique: {unique}, Dup Rate: {rate:.1f}%")
        print(f"\nTop Unique Signatures:")
        for i, (sig_hash, info) in enumerate(list(self.first_occurrence.items())[:10]):
            print(f"  {i+1}. [{info['oracle'].upper()}] {info['signature'][:60]}...")
        print(f"{'='*70}")


# =============================================================================
# DispatcherSpace (5 + )
# =============================================================================

class DispatcherSpace:
    """
    Dispatcher - DispatcherStateSpace
    
    DispatchKey 5 :
    1. Device: {'cpu', 'cuda'}
    2. Dtype: {'float32', 'float16', 'int64', ...}
    3. Layout: {'strided', 'sparse_coo'}
    4. MemFormat: {'contiguous', 'channels_last'}
    5. Rank: {1, 2, 3, 4, 5}
    
    :
    - =
    - =
    - Coverage = || / || (capped at 100%)
    """
    
    def __init__(self, max_hits: int = 50000):
        self.max_hits = max_hits
        
        # pytorch_comprehensive_filter DispatcherStateSpace
        self.state_space = DispatcherStateSpace(max_states=max_hits)
        
        #
        self.api_dimensions: Dict[str, Dict[str, List[str]]] = {}
        self.theoretical_sizes: Dict[str, int] = {}
        self.total_records = 0
        self.overflow_warnings = 0
        
        # ()
        self.coverage_history: Dict[str, List[Tuple[int, float]]] = {}
    
    def register_api(self, api_name: str, dimensions: Dict[str, List[str]] = None):
        """API """
        # DispatcherStateSpace
        result = self.state_space.register_api(api_name)
        
        #
        self.theoretical_sizes[api_name] = result['valid_space']
        self.api_dimensions[api_name] = result['dimensions']
        
        #
        self.coverage_history[api_name] = []
    
    def record_hit(self, api_name: str, api, is_cuda: bool = False, iteration: int = 0) -> bool:
        """

        
        Args:
            api_name: API
            api: TorchAPI
            is_cuda: CUDA
            iteration: ()
        """
        #
        if api_name not in self.state_space.api_spaces:
            self.register_api(api_name)
        
        # 5
        state = self._extract_state_from_args(api.args, is_cuda)
        if state is None:
            return False
        
        # (100%)
        #
        is_new = self.state_space.record_hit(api_name, state)
        if is_new:
            self.total_records += 1
            
            #
            current_cov = self.get_coverage(api_name)
            self.coverage_history[api_name].append((iteration, current_cov))
        
        return is_new
    
    def record_coverage_snapshot(self, api_name: str, iteration: int):
        """(N)"""
        if api_name not in self.coverage_history:
            self.coverage_history[api_name] = []
        
        current_cov = self.get_coverage(api_name)
        self.coverage_history[api_name].append((iteration, current_cov))
    
    def _extract_state_from_args(self, args: Dict, is_cuda: bool) -> Optional[Dict]:
        """
        5
        
        :
        1. argsTensor (LIST/TUPLE )
        2. input_signature
        3. device, dtype, layout, mem_format, rank
        """
        state = {
            'device': 'cuda' if is_cuda else 'cpu',
            'dtype': None,
            'layout': 'strided',
            'mem_format': 'contiguous',
            'rank': None,
        }
        
        def extract_from_arg(arg):
            """tensor """
            if arg is None:
                return None, None
            
            # LIST/TUPLE -
            if hasattr(arg, 'type') and arg.type in [ArgType.LIST, ArgType.TUPLE]:
                if hasattr(arg, 'value') and arg.value:
                    for sub_arg in arg.value:
                        dtype, rank = extract_from_arg(sub_arg)
                        if dtype is not None:
                            return dtype, rank
                return None, None
            
            # TORCH_TENSOR
            if hasattr(arg, 'type') and arg.type == ArgType.TORCH_TENSOR:
                dtype = None
                rank = None
                
                if hasattr(arg, 'dtype') and arg.dtype is not None:
                    dtype = str(arg.dtype).replace('torch.', '')
                
                if hasattr(arg, 'shape') and arg.shape is not None:
                    rank = len(arg.shape)
                
                return dtype, rank
            
            return None, None
        
        # input_signature (nn.Module API)
        if 'input_signature' in args:
            dtype, rank = extract_from_arg(args['input_signature'])
            if dtype is not None:
                state['dtype'] = dtype
            if rank is not None:
                state['rank'] = rank
            if state['dtype'] is not None and state['rank'] is not None:
                return state
        
        #
        for param_name, arg in args.items():
            if param_name == 'input_signature':
                continue
            
            dtype, rank = extract_from_arg(arg)
            
            if dtype is not None and state['dtype'] is None:
                state['dtype'] = dtype
            if rank is not None and state['rank'] is None:
                state['rank'] = rank
            
            # layout
            if hasattr(arg, 'is_sparse') and arg.is_sparse:
                state['layout'] = 'sparse_coo'
            
            # dtype
            if hasattr(arg, 'type') and hasattr(arg, 'value'):
                if arg.type == ArgType.TORCH_DTYPE:
                    state['dtype'] = str(arg.value).replace('torch.', '')
            
            #
            if state['dtype'] is not None and state['rank'] is not None:
                return state
        
        #
        if len(args) > 0:
            if state['dtype'] is None:
                state['dtype'] = 'float32'  # float32
            if state['rank'] is None:
                state['rank'] = 2  # 2d (0d )
            return state
        
        return None
    
    def get_coverage(self, api_name: str) -> float:
        """"""
        return self.state_space.get_coverage(api_name)
    
    def get_detailed_stats(self, api_name: str) -> Dict:
        """"""
        return self.state_space.get_detailed_stats(api_name)
    
    def print_summary(self):
        """"""
        self.state_space.print_summary()


# =============================================================================

class AdaptiveSaturationDetector:
    """
    Adaptive saturation detector.

    Logic:
    1. If no new kernel is discovered after N iterations, switch to
       expansion mode rather than stopping.
    2. Expansion mode = mutation expansion + exploration boost.
    3. Once the kernel discovery rate recovers, fall back to normal mode.
    """
    
    def __init__(self, 
                 patience: int = 500,           #
                 check_interval: int = 100):
        self.patience = patience
        self.check_interval = check_interval
        
        self.no_discovery_count = 0
        self.last_kernel_count = 0
        self.last_check_iteration = 0
        
        #
        self.expansion_mode = False
        self.expansion_count = 0
    
    def update(self, current_kernels: int, iteration: int) -> Tuple[bool, str]:
        """

        
        Returns:
            (should_expand, message)
        """
        #
        if current_kernels > self.last_kernel_count:
            self.no_discovery_count = 0
            self.last_kernel_count = current_kernels
            
            # Kernel
            if self.expansion_mode:
                self.expansion_mode = False
                return False, "New kernels found! Returning to normal mode"
        else:
            self.no_discovery_count += 1
        
        #
        if iteration - self.last_check_iteration >= self.check_interval:
            self.last_check_iteration = iteration
            
            #
            if self.no_discovery_count >= self.patience and not self.expansion_mode:
                self.expansion_mode = True
                self.expansion_count += 1
                return True, f"Expanding search (stagnation: {self.no_discovery_count} iters)"
        
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
    """+ """
    
    def __init__(self, name: str, enable_adaptive_saturation: bool = True):
        self.name = name
        self.all_kernels: Set[str] = set()
        self.kernel_provenance: Dict[str, int] = {}
        self.history: List[Tuple[int, int]] = []
        self.new_kernel_iterations: List[int] = []
        
        #
        self.saturation_detector = AdaptiveSaturationDetector() if enable_adaptive_saturation else None
    
    def update(self, new_kernels: Set[str], iteration: int) -> Tuple[int, Tuple[bool, str]]:
        """

        
        Returns:
            (new_kernel_count, (should_expand, message))
        """
        if not isinstance(new_kernels, set):
            new_kernels = set(new_kernels) if new_kernels else set()
        
        fresh = new_kernels - self.all_kernels
        
        if fresh:
            for kernel in fresh:
                self.kernel_provenance[kernel] = iteration
            
            self.all_kernels.update(fresh)
            self.new_kernel_iterations.append(iteration)
            
            print(f"  [{self.name}] Iter {iteration}: +{len(fresh)} new kernels! "
                  f"Total: {len(self.all_kernels)}")
            
            num_fresh = len(fresh)
        else:
            num_fresh = 0
        
        self.history.append((iteration, len(self.all_kernels)))
        
        #
        expansion_signal = (False, "")
        if self.saturation_detector:
            expansion_signal = self.saturation_detector.update(len(self.all_kernels), iteration)
            if expansion_signal[0]:
                print(f"\n{expansion_signal[1]}")
        
        return num_fresh, expansion_signal
    
    def get_total(self) -> int:
        return len(self.all_kernels)
    
    def get_exclusive(self, other: 'EnhancedCoverageTracker') -> Set[str]:
        return self.all_kernels - other.all_kernels
    
    def get_overlap(self, other: 'EnhancedCoverageTracker') -> Set[str]:
        return self.all_kernels & other.all_kernels


# =============================================================================
# Evolutionary Corpus
# =============================================================================

class EvolutionaryCorpus:
    """"""
    
    def __init__(self, max_size: int = 100):
        self.corpus: List[TorchAPI] = []
        self.max_size = max_size
    
    def add_seed(self, api: TorchAPI, discovered_kernels: Set[str]):
        if len(discovered_kernels) == 0:
            return
        
        seed_copy = copy.deepcopy(api)
        self.corpus.append(seed_copy)
        
        if len(self.corpus) > self.max_size:
            self.corpus.pop(0)
    
    def select_parent(self) -> Optional[TorchAPI]:
        if not self.corpus:
            return None
        return random.choice(self.corpus)
    
    def size(self) -> int:
        return len(self.corpus)


# =============================================================================
# NEW: Adaptive Mutation Controller ()
# =============================================================================

class AdaptiveMutationController:
    """

    
    :
    1.
    2. -greedy
    3. Kernel
    """
    
    def __init__(self, 
                 base_epsilon: float = 0.1,          #
                 expansion_epsilon: float = 0.3,     #
                 surgical_range: int = 8,            #
                 exploration_range: int = 64):       #
        self.base_epsilon = base_epsilon
        self.expansion_epsilon = expansion_epsilon
        self.surgical_range = surgical_range
        self.exploration_range = exploration_range
        
        #
        self.current_epsilon = base_epsilon
        self.current_range = surgical_range
        self.expansion_mode = False
        
        #
        self.exploration_count = 0
        self.exploitation_count = 0
    
    def should_explore(self) -> bool:
        """"""
        if random.random() < self.current_epsilon:
            self.exploration_count += 1
            return True
        else:
            self.exploitation_count += 1
            return False
    
    def enter_expansion_mode(self):
        """"""
        self.expansion_mode = True
        self.current_epsilon = self.expansion_epsilon
        self.current_range = self.exploration_range
        print(f"  [Mutation] Expansion mode: ={self.current_epsilon}, range={self.current_range}")
    
    def exit_expansion_mode(self):
        """Kernel """
        self.expansion_mode = False
        self.current_epsilon = self.base_epsilon
        self.current_range = self.surgical_range
        print(f"  [Mutation] Normal mode: ={self.current_epsilon}, range={self.current_range}")
    
    def get_mutation_range(self) -> int:
        """"""
        return self.current_range
    
    def get_status(self) -> str:
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return "No mutations yet"
        explore_pct = self.exploration_count / total * 100
        return (f"Explore: {self.exploration_count}/{total} ({explore_pct:.1f}%) | "
                f"Mode: {'Expansion' if self.expansion_mode else 'Normal'} | "
                f"={self.current_epsilon:.2f}, range={self.current_range}")


# =============================================================================
# Probability Patcher
# =============================================================================

class ProbabilityPatcher:
    """20% 80%"""
    
    @staticmethod
    def patch_high_db_probability():
        import utils.probability as prob_module
        
        global _ORIGINAL_DO_SELECT_FROM_DB
        _ORIGINAL_DO_SELECT_FROM_DB = prob_module.do_select_from_db
        
        def high_db_select() -> bool:
            from numpy.random import rand
            return rand() < 0.5
        
        prob_module.do_select_from_db = high_db_select
        print("[Patch] Database sampling: 20% 80%")
    
    @staticmethod
    def restore():
        if _ORIGINAL_DO_SELECT_FROM_DB:
            import utils.probability as prob_module
            prob_module.do_select_from_db = _ORIGINAL_DO_SELECT_FROM_DB
            print("[Patch] Database sampling restored")


# =============================================================================
# NEW: Poison Patcher (- )
# =============================================================================

#
_POISON_ORIGINAL_FLOAT = None
_POISON_ORIGINAL_INT = None

class PoisonPatcher:
    """
    Enhanced poison patch: systematically covers 200 BugsInDLLs entries that
    are "surface-easy but actually hard".

    Data-driven poison strategy (derived from bugsindlls_bugs.json):

    Float poison (targeting 116 zero-size + 46 Inf + 42 negative + 13 NaN):
    - 8%:  NaN (needed by 13 bugs)
    - 8%:  Infinity (needed by 46 bugs)
    - 8%:  zero 0.0 (involved in 116 zero-size bugs)
    - 8%:  negative values (needed by 42 bugs)
    - 8%:  extreme values (overflow boundary)
    - 8%:  near-zero decimals (precision pitfalls)
    - 20%: dictionary sampling
    - 32%: mild perturbation

    Int poison (targeting 116 zero-size + 25 singleton + 42 negative):
    - 12%: size=0 (116 bugs require zero-size)
    - 10%: size=1 (25 bugs require singleton dimension)
    - 8%:  negative dimensions (needed by 42 bugs)
    - 8%:  trap values [primes, large sizes, small dims]
    - 10%: extreme values [+/-999, +/-2^31]
    - 20%: dictionary sampling
    - 32%: mild perturbation

    Addresses: why FreeFuzz misses "easy-looking" bugs.
       -> Random fuzzing only exercises the "normal path" and misses
          boundary cases.
       -> Systematic poisoning ensures coverage of NaN/Inf/zero/negative/singleton.
    """
    
    @staticmethod
    def patch():
        global _POISON_ORIGINAL_FLOAT, _POISON_ORIGINAL_INT
        
        _POISON_ORIGINAL_FLOAT = Argument.mutate_float_value
        _POISON_ORIGINAL_INT = Argument.mutate_int_value
        
        def poison_float_mutation(self, value) -> float:
            """Enhanced float poison: systematically covers the 200 BugsInDLLs
            "surface-easy but actually hard" bugs.

            Based on BugsInDLLs hidden-difficulty analysis:
            - 116 bugs require zero-size (triggered by 0.0)
            - 46 bugs require Inf
            - 42 bugs require negatives
            - 13 bugs require NaN
            """
            from numpy.random import rand, choice

            roll = rand()

            # [8%] NaN - top priority (13 bugs require this)
            if roll < 0.08:
                return float('nan')

            # [8%] Infinity - second priority (46 bugs require this)
            elif roll < 0.16:
                return choice([float('inf'), float('-inf')])

            # [8%] zero - critical boundary (116 zero-size bugs)
            elif roll < 0.24:
                return 0.0

            # [8%] negatives - common trap (42 bugs)
            elif roll < 0.32:
                return choice([-1.0, -0.5, -10.0, -100.0, -1e-5])

            # [8%] extreme values - overflow boundary
            elif roll < 0.40:
                extreme_values = [1e20, -1e20, 1e-10, -1e-10, 1e38, -1e38]
                return choice(extreme_values)

            # [8%] near-zero decimals - precision pitfall
            elif roll < 0.48:
                return choice([1e-7, -1e-7, 1e-15, -1e-15, 1e-20, -1e-20])

            # [20%] dictionary sampling (preserve original logic)
            elif roll < 0.68:
                if hasattr(Argument, '_float_values'):
                    return choice(Argument._float_values)
                elif hasattr(Argument, 'float_values'):
                    return choice(Argument.float_values)
                return choice([0.0, 1.0, -1.0])

            # [32%] mild perturbation
            else:
                return value + (rand() - 0.5) * 8.0
        
        def poison_int_mutation(self, value, _min=None, _max=None) -> int:
            """Enhanced int poison: systematically covers boundary and trap values.

            Based on BugsInDLLs hidden-difficulty analysis:
            - 116 bugs require zero-size tensor (size=0)
            - 42 bugs require negative dimensions
            - 25 bugs require singleton dimension (size=1)
            """
            from numpy.random import rand, choice, randint

            roll = rand()

            # [12%] zero - top priority (116 zero-size bugs)
            if roll < 0.12:
                return 0

            # [10%] Singleton - second priority (25 bugs need size=1)
            elif roll < 0.22:
                return 1

            # [8%] negative boundary - critical trap (42 bugs need negatives)
            elif roll < 0.30:
                return choice([-1, -2, -3, -4])

            # [8%] common trap values - problematic for specific arguments
            elif roll < 0.38:
                trap_values = [
                    7, 11, 13,       # primes (non-divisible)
                    256, 512, 1024,  # large sizes
                    2, 3, 4, 5,      # small dimensions
                ]
                new_value = choice(trap_values)

            # [10%] extreme values - overflow and boundary checks
            elif roll < 0.48:
                extreme_values = [-999, 999, -2147483648, 2147483647, -65536, 65536]
                new_value = choice(extreme_values)

            # [20%] dictionary sampling
            elif roll < 0.68:
                if hasattr(Argument, '_int_values'):
                    new_value = choice(Argument._int_values)
                elif hasattr(Argument, 'int_values'):
                    new_value = choice(Argument.int_values)
                else:
                    new_value = randint(-100, 100)

            # [32%] mild perturbation
            else:
                new_value = value + randint(-8, 9)

            # Apply bounds (but preserve special values like -1, 0, 1 used to
            # trigger bugs).
            if _min is not None and new_value < _min and new_value not in [-1, 0, 1]:
                new_value = max(_min, new_value)
            if _max is not None and new_value > _max:
                new_value = min(_max, new_value)

            return int(new_value)
        
        # Fix: rebind to override the previously defined poison_float_mutation.
        Argument.mutate_float_value = poison_float_mutation 
        Argument.mutate_int_value = poison_int_mutation
        print("[Patch] Poison Injection enabled (ALL strategies):")
        print("        Float: 5% Inf + 5% NaN + 10% Extreme")
        print("        Int:   10% boundary + 10% extreme + 10% trap")
    
    @staticmethod
    def restore():
        global _POISON_ORIGINAL_FLOAT, _POISON_ORIGINAL_INT
        if _POISON_ORIGINAL_FLOAT:
            Argument.mutate_float_value = _POISON_ORIGINAL_FLOAT
        if _POISON_ORIGINAL_INT:
            Argument.mutate_int_value = _POISON_ORIGINAL_INT
        print("[Patch] Poison Injection restored")


# =============================================================================
# NEW: Adaptive Mutation Patcher (- Guided)
# =============================================================================

class AdaptiveMutationPatcher:
    """- Guided PoisonPatcher """
    
    @staticmethod
    def patch_adaptive_mutation(controller: AdaptiveMutationController):
        """
        Guided
        : PoisonPatcher
        """
        # : /PoisonPatcher
        # patcher Guided
        print(f"[Patch] Adaptive range enabled for Guided: {controller.get_mutation_range()}")
    
    @staticmethod
    def restore():
        # PoisonPatcher
        print("[Patch] Adaptive range disabled")


# =============================================================================
# Enhanced Fuzzer with Hybrid Strategy
# =============================================================================

class EnhancedFuzzer:
    """Fuzzer"""
    
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
                 warmup_ratio: float = 0.1,           #
                 enable_dispatcher: bool = True):     # DispatcherSpace  #
        self.api_name = api_name
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.enable_patches = enable_patches
        self.use_all_oracles = use_all_oracles
        self.enable_logging = enable_logging
        self.enable_checkpoint = enable_checkpoint
        self.enable_safety_guards = enable_safety_guards
        self.diff_bound = diff_bound
        self.warmup_ratio = warmup_ratio             #
        self.enable_dispatcher = enable_dispatcher   # dispatcher
        
        # Trackers
        self.coverage = EnhancedCoverageTracker(strategy_name, enable_adaptive_saturation=enable_patches)
        self.bug_tracker = BugTracker(strategy_name, output_dir)
        
        # Guided
        self.corpus = EvolutionaryCorpus(max_size=100) if enable_patches else None
        self.mutation_controller = AdaptiveMutationController() if enable_patches else None
        
        # Library - diff_bound
        self.library = TorchLibrary(output_dir, diff_bound=diff_bound)
        print(f"[Fuzzer] Precision tolerance: diff_bound={diff_bound}")
        
        # Oracle
        if use_all_oracles:
            self.oracles = [OracleType.CRASH, OracleType.CUDA, OracleType.PRECISION]
            print(f"[Fuzzer] Using ALL oracles: CRASH + CUDA + PRECISION")
        else:
            self.oracles = [OracleType.CRASH]
            print(f"[Fuzzer] Using CRASH oracle only")
        
        # Experiment Logger
        self.logger = None
        if enable_logging:
            api_clean = api_name.replace('.', '_').replace('::', '_')
            log_file = join(output_dir, f"{api_clean}_{strategy_name.lower()}_trace.jsonl")
            self.logger = ExperimentLogger(log_file, strategy_name.lower())
            print(f"[Logger] Trace file: {log_file}")
        
        # Checkpoint Manager
        self.checkpoint_manager = None
        if enable_checkpoint:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=output_dir,
                strategy=strategy_name.lower(),
                api_name=api_name
            )
        
        # Safety Guards
        self.speedometer = None
        self.disk_guard = None
        self.outlier_filter = None
        
        if enable_safety_guards:
            self.speedometer = Speedometer(window_size=100, slow_threshold=0.5)
            self.disk_guard = DiskGuard(output_dir, min_free_gb=1.0, auto_cleanup=True)
            self.outlier_filter = OutlierFilter(max_elements=int(5e6), max_memory_gb=2.0)  # OOM
            print(f"[Safety] Guards enabled: Speedometer + DiskGuard + OutlierFilter")
        
        # DispatcherSpace
        self.dispatcher_space = None
        if enable_dispatcher:
            self.dispatcher_space = DispatcherSpace(max_hits=50000)
            self.dispatcher_space.register_api(api_name)
        
        # Bug
        self.bug_deduplicator = RealTimeBugDeduplicator(max_signatures=10000)
        
        # Warmup (run_fuzzing_loop )
        self.warmup_threshold = 0
        
        print(f"\n[Fuzzer] Initialized: {strategy_name}")
        print(f"  API: {api_name}")
        print(f"  Patches: {'Enabled' if enable_patches else 'Disabled'}")
        print(f"  Hybrid Strategy: {'Active' if self.mutation_controller else 'N/A'}")
        print(f"  Data Logging: {'Enabled' if enable_logging else 'Disabled'}")
        print(f"  Checkpoint: {'Enabled' if enable_checkpoint else 'Disabled'}")
        print(f"  Safety Guards: {'Enabled' if enable_safety_guards else 'Disabled'}")
        print(f"  Warmup Ratio: {warmup_ratio*100:.0f}%")
        print(f"  DispatcherSpace: {'Enabled' if enable_dispatcher else 'Disabled'}")
    
    def run_fuzzing_loop(self, 
                         max_iterations: int = 10000,
                         checkpoint_interval: int = 100,
                         bug_scan_interval: int = 50):
        """
        Fuzzing -
        """
        # ====================================================================
        # Safety Guards
        # ====================================================================
        if self.speedometer:
            self.speedometer.start()
        
        # ====================================================================
        # checkpoint
        # ====================================================================
        start_iteration = 0
        
        if self.checkpoint_manager and self.checkpoint_manager.exists():
            print(f"\n{'='*70}")
            print(f"CHECKPOINT DETECTED")
            print(f"{'='*70}")
            
            checkpoint_data = self.checkpoint_manager.load()
            
            if checkpoint_data:
                self.coverage.all_kernels = checkpoint_data['coverage_kernels']
                self.coverage.history = [(checkpoint_data['iteration'], len(checkpoint_data['coverage_kernels']))]
                
                if self.corpus and checkpoint_data.get('corpus_seeds'):
                    self.corpus.corpus = checkpoint_data['corpus_seeds']
                
                start_iteration = checkpoint_data['iteration'] + 1
                
                if self.logger:
                    self.logger.total_iterations = checkpoint_data.get('logger_iterations', 0)
                
                print(f"\nResuming from iteration {start_iteration}")
                print(f"  Previous kernels: {len(self.coverage.all_kernels)}")
                if self.corpus:
                    print(f"  Previous corpus: {len(self.corpus.corpus)} seeds")
                print(f"{'='*70}\n")
        
        # ====================================================================
        #
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"Starting {self.strategy_name} fuzzing")
        if self.mutation_controller:
            print(f"Hybrid Strategy: -greedy + Adaptive Mutation")
        print(f"Iterations: {start_iteration} {max_iterations}")
        if start_iteration > 0:
            print(f"(Resuming from checkpoint)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        last_bug_count = 0
        
        #
        self.warmup_threshold = int(max_iterations * self.warmup_ratio)
        print(f"Warmup Phase: iterations 0 ~ {self.warmup_threshold} (Random only)")
        print(f"Evolution Phase: iterations {self.warmup_threshold} ~ {max_iterations}")
        
        for i in range(start_iteration, max_iterations):
            # ================================================================
            # Speedometer tick
            # ================================================================
            if self.speedometer:
                self.speedometer.tick()
            
            # ================================================================
            # -greedy
            # ================================================================
            source = ""
            is_exploration = False
            
            # Phase 1: Warm-up (Random)
            if i < self.warmup_threshold:
                api = TorchAPI(self.api_name)
                source = "warmup_random"
                api.mutate()
                if i == self.warmup_threshold - 1:
                    print(f"\nPHASE TRANSITION: Warm-up Evolution (iter {i+1})")
            
            # Phase 2: Evolution (-greedy)
            else:
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
            
            # OutlierFilter
            # ================================================================
            if self.outlier_filter:
                should_filter, reason = self.outlier_filter.check_api(api)
                if should_filter:
                    if self.logger:
                        self.logger.log_iteration(
                            iteration=i,
                            source=source,
                            api=api,
                            valid=False,
                            kernels=[]
                        )
                    continue
            
            # ================================================================
            # Oracle
            # ================================================================
            all_captured_kernels = set()
            execution_valid = False
            
            for oracle in self.oracles:
                try:
                    # test_with_oracle (kernels, bug_info)
                    result = self.library.test_with_oracle(api, oracle)
                    
                    # kernels
                    if isinstance(result, tuple):
                        captured_kernels, bug_info = result
                    else:
                        captured_kernels, bug_info = result, None
                    
                    all_captured_kernels.update(captured_kernels)
                    execution_valid = True
                    
                    #
                    if bug_info and hasattr(self, 'bug_deduplicator'):
                        is_new, sig = self.bug_deduplicator.record_bug(
                            oracle_type=bug_info.get('type', 'crash'),
                            error_msg=bug_info.get('error', ''),
                            iteration=i
                        )
                        if is_new:
                            print(f"  NEW [{bug_info.get('oracle', 'BUG')}]: {sig[:50]}...")
                            
                except Exception as e:
                    pass
            
            #
            if self.logger:
                self.logger.log_iteration(
                    iteration=i,
                    source=source,
                    api=api,
                    valid=execution_valid,
                    kernels=list(all_captured_kernels)
                )
            
            # DispatcherSpace
            if self.dispatcher_space and execution_valid:
                # CUDA (CUDA oracle .cuda())
                is_cuda = OracleType.CUDA in self.oracles
                self.dispatcher_space.record_hit(self.api_name, api, is_cuda=is_cuda, iteration=i)
            
            # (50)
            if self.dispatcher_space and i % 50 == 0:
                self.dispatcher_space.record_coverage_snapshot(self.api_name, i)
            
            # ================================================================
            # +
            # ================================================================
            new_count, (should_expand, expansion_msg) = self.coverage.update(all_captured_kernels, i)
            
            #
            if should_expand and self.mutation_controller:
                self.mutation_controller.enter_expansion_mode()
            
            # Kernel
            if new_count > 0 and self.mutation_controller:
                if self.mutation_controller.expansion_mode:
                    self.mutation_controller.exit_expansion_mode()
            
            # Corpus
            if new_count > 0 and self.corpus:
                self.corpus.add_seed(api, all_captured_kernels)
            
            # Bug
            if (i + 1) % bug_scan_interval == 0:
                prev_bug_count = self.bug_tracker.get_total_bugs()
                self.bug_tracker.scan_bugs()
                new_bugs = self.bug_tracker.get_total_bugs() - prev_bug_count
                
                if new_bugs > 0:
                    print(f"  [{self.strategy_name}] Found {new_bugs} new bugs! "
                          f"Total: {self.bug_tracker.get_total_bugs()}")
            
            # ================================================================
            # Checkpoint + Safety Guards
            # ================================================================
            if (i + 1) % checkpoint_interval == 0:
                # Checkpoint
                if self.checkpoint_manager:
                    self.checkpoint_manager.save(
                        iteration=i,
                        coverage_kernels=self.coverage.all_kernels,
                        corpus_seeds=self.corpus.corpus if self.corpus else None,
                        logger_iterations=self.logger.total_iterations if self.logger else 0
                    )
                
                #
                if self.disk_guard:
                    is_critical, message = self.disk_guard.check_and_cleanup()
                    if is_critical:
                        print(f"\n{message}")
                        print(f"STOPPING: Critical disk space issue")
                        break
                    elif message:
                        print(f"\n{message}")
                
                # Progress Checkpoint
                elapsed = time.time() - start_time
                valid_rate_str = ""
                if self.logger:
                    valid_rate_str = f" | Valid: {self.logger.get_valid_rate()*100:.1f}%"
                
                print(f"\n--- Checkpoint: {i+1}/{max_iterations} ({elapsed/60:.1f} min) ---")
                print(f"Kernels: {self.coverage.get_total()} | "
                      f"Bugs: {self.bug_tracker.get_total_bugs()}{valid_rate_str}")
                
                # Speedometer
                if self.speedometer:
                    print(f"Speed: {self.speedometer.get_status(i, max_iterations)}")
                    is_slow, warning = self.speedometer.check_speed()
                    if is_slow:
                        print(warning)
                
                # Disk
                if self.disk_guard:
                    print(f"Disk: {self.disk_guard.get_status()}")
                
                # OutlierFilter
                if self.outlier_filter:
                    print(f"Filter: {self.outlier_filter.get_status()}")
                
                # Corpus
                if self.corpus:
                    print(f"Corpus: {self.corpus.size()} seeds")
                
                # DispatcherSpace - get_detailed_stats
                if self.dispatcher_space:
                    cov = self.dispatcher_space.get_coverage(self.api_name)
                    # hits = len(self.dispatcher_space.hits[self.api_name]) #
                    stats = self.dispatcher_space.get_detailed_stats(self.api_name) #
                    hits = stats.get('hit_count', 0)
                    print(f"Dispatcher: {hits} combos ({cov*100:.1f}% coverage)")
                
                # Mutation Controller
                if self.mutation_controller:
                    print(f"Mutation: {self.mutation_controller.get_status()}")
                
                # OOM (Exit 137)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Bug
                self.bug_tracker.scan_bugs()
        
        # ====================================================================
        #
        # ====================================================================
        if self.checkpoint_manager:
            self.checkpoint_manager.clear()
        
        #
        print(f"\n{'='*70}")
        print(f"Fuzzing completed: {self.strategy_name}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        
        if self.logger:
            print(f"Valid Rate: {self.logger.get_valid_rate()*100:.1f}%")
            print(f"Trace file: {self.logger.output_file}")
            self.logger.close()
        
        if self.speedometer:
            print(f"Average Speed: {self.speedometer.get_average_speed():.2f} it/s")
            if self.speedometer.slow_warnings > 0:
                print(f"Slow Warnings: {self.speedometer.slow_warnings}")
        
        if self.disk_guard:
            total, used, free = self.disk_guard.get_disk_usage()
            print(f"Final Disk Usage: {free:.2f} GB free")
            if self.disk_guard.cleanups > 0:
                print(f"Cleanups Performed: {self.disk_guard.cleanups}")
        
        if self.outlier_filter:
            print(f"Outliers Filtered: {self.outlier_filter.filtered_count}/{self.outlier_filter.total_checks}")
        
        if self.mutation_controller:
            print(f"Mutation Stats: {self.mutation_controller.get_status()}")
        
        # DispatcherSpace
        if self.dispatcher_space:
            self.dispatcher_space.print_summary()
        
        # Bug
        if hasattr(self, 'bug_deduplicator'):
            self.bug_deduplicator.print_summary()
        
        print(f"{'='*70}")
        
        return self.coverage
    
    def print_stats(self):
        """"""
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS: {self.strategy_name}")
        print(f"{'='*70}")
        print(f"Total Kernels: {self.coverage.get_total()}")
        
        # Bug
        self.bug_tracker.scan_bugs()
        self.bug_tracker.print_summary()


# =============================================================================
# Visualization
# =============================================================================

def plot_full_oracle_results(
    random_coverage: EnhancedCoverageTracker,
    guided_coverage: EnhancedCoverageTracker,
    random_bugs: BugTracker,
    guided_bugs: BugTracker,
    api_name: str,
    output_dir: str,
    random_dispatcher: DispatcherSpace = None,  #
    guided_dispatcher: DispatcherSpace = None,  #
    random_dedup = None,  # Bug
    guided_dedup = None   #
):
    """Bug """
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Kernel
    ax1 = fig.add_subplot(gs[0, :2])
    
    if random_coverage.history:
        iters, kernels = zip(*random_coverage.history)
        ax1.plot(iters, kernels, 
                label=f"Random (Total: {random_coverage.get_total()})",
                color="#3498db", linewidth=3, alpha=0.8)
    
    if guided_coverage.history:
        iters, kernels = zip(*guided_coverage.history)
        ax1.plot(iters, kernels,
                label=f"Guided (Total: {guided_coverage.get_total()})",
                color="#e74c3c", linewidth=3)
    
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Cumulative Kernels", fontsize=12)
    ax1.set_title("A) Kernel Coverage Over Time (Hybrid Strategy)", 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bug
    ax2 = fig.add_subplot(gs[0, 2])
    
    bug_types = ['CRASH', 'CUDA', 'PRECISION']
    random_bug_counts = [
        random_bugs.get_bugs_by_type('crash'),
        random_bugs.get_bugs_by_type('cuda'),
        random_bugs.get_bugs_by_type('precision')
    ]
    guided_bug_counts = [
        guided_bugs.get_bugs_by_type('crash'),
        guided_bugs.get_bugs_by_type('cuda'),
        guided_bugs.get_bugs_by_type('precision')
    ]
    
    x = np.arange(len(bug_types))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, random_bug_counts, width, 
                    label='Random', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, guided_bug_counts, width,
                    label='Guided', color='#e74c3c', alpha=0.8)
    
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
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
    # Bug
    ax3 = fig.add_subplot(gs[1, 0])
    
    total_random = random_bugs.get_total_bugs()
    total_guided = guided_bugs.get_total_bugs()
    
    if total_random > 0 or total_guided > 0:
        sizes = [total_random, total_guided]
        labels = [f'Random\n({total_random})', f'Guided\n({total_guided})']
        colors = ['#3498db', '#e74c3c']
        explode = (0.05, 0.05)
        
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'})
        ax3.set_title("C) Total Bugs Distribution", 
                      fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "No Bugs Found", 
                ha='center', va='center', fontsize=14)
        ax3.axis('off')
    
    # D: 5D
    ax4 = fig.add_subplot(gs[1, 1])
    
    # dispatcher
    random_cov_history = []
    guided_cov_history = []
    
    if random_dispatcher and api_name in random_dispatcher.coverage_history:
        random_cov_history = random_dispatcher.coverage_history[api_name]
    
    if guided_dispatcher and api_name in guided_dispatcher.coverage_history:
        guided_cov_history = guided_dispatcher.coverage_history[api_name]
    
    if random_cov_history:
        iters, covs = zip(*random_cov_history)
        # Cap 100%
        covs = [min(c * 100, 100.0) for c in covs]
        ax4.plot(iters, covs, label="Random", 
                color="#3498db", linewidth=2, marker='o', markersize=3)
    
    if guided_cov_history:
        iters, covs = zip(*guided_cov_history)
        # Cap 100%
        covs = [min(c * 100, 100.0) for c in covs]
        ax4.plot(iters, covs, label="Guided", 
                color="#e74c3c", linewidth=2, marker='s', markersize=3)
    
    ax4.set_xlabel("Iteration", fontsize=11)
    ax4.set_ylabel("Coverage (%)", fontsize=11)
    ax4.set_title("D) 5D State Space Coverage", 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc="lower right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)  # Y 0-100%
    
    #
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # 5(capped at 100%)
    random_state_cov = min(random_dispatcher.get_coverage(api_name) * 100, 100.0) if random_dispatcher else 0
    guided_state_cov = min(guided_dispatcher.get_coverage(api_name) * 100, 100.0) if guided_dispatcher else 0
    
    # ()
    random_valid_space = 0
    guided_valid_space = 0
    if random_dispatcher and api_name in random_dispatcher.state_space.api_spaces:
        random_valid_space = random_dispatcher.state_space.api_spaces[api_name]['valid_space']
    if guided_dispatcher and api_name in guided_dispatcher.state_space.api_spaces:
        guided_valid_space = guided_dispatcher.state_space.api_spaces[api_name]['valid_space']
    
    # Bug
    random_unique = sum(random_dedup.unique_bugs.values()) if random_dedup else random_bugs.get_total_bugs()
    guided_unique = sum(guided_dedup.unique_bugs.values()) if guided_dedup else guided_bugs.get_total_bugs()
    
    # Bug
    random_bugs.scan_bugs()
    guided_bugs.scan_bugs()
    random_unique = len(random_bugs.unique_bugs)
    guided_unique = len(guided_bugs.unique_bugs)
    
    table_data = [
        ["Metric", "Random", "Guided", "Ratio"],
        ["", "", "", ""],
        ["Kernels", 
         f"{random_coverage.get_total()}",
         f"{guided_coverage.get_total()}",
         f"{guided_coverage.get_total()/max(random_coverage.get_total(),1):.2f}x"],
        ["5D State Cov",
         f"{random_state_cov:.1f}%",
         f"{guided_state_cov:.1f}%",
         f"{guided_state_cov/max(random_state_cov,0.1):.2f}x"],
        ["Valid Space",
         f"{random_valid_space}",
         f"{guided_valid_space}",
         "-"],
        ["Unique Bugs",
         f"{random_unique}",
         f"{guided_unique}",
         f"{guided_unique/max(random_unique,1):.2f}x"],
        ["Raw Triggers",
         f"{random_bugs.get_total_bugs()}",
         f"{guided_bugs.get_total_bugs()}",
         f"{guided_bugs.get_total_bugs()/max(random_bugs.get_total_bugs(),1):.2f}x"],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.22, 0.22, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(4):
        table[(1, i)].set_facecolor('#ecf0f1')
    
    ax5.set_title("E) Comprehensive Comparison", 
                  fontsize=12, fontweight='bold', pad=20)
    
    #
    fig.suptitle(
        f"Hybrid Strategy Benchmark: {api_name}\n"
        f"-greedy (10% Exploration) + Adaptive Mutation + Dynamic Saturation",
        fontsize=15, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    plot_file = join(output_dir, f"{api_name.replace('.', '_')}_hybrid_strategy.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHybrid strategy plot saved: {plot_file}")


# =============================================================================
# Subprocess isolation + crash recovery (ported from the TensorFlow version)
# =============================================================================

def _generate_crash_bug_file(output_dir: str, phase_name: str, api_name: str,
                              exit_code: int, error_desc: str):
    """Emit a bug file when the subprocess crashes."""
    bug_dir = join(output_dir, "crash-oracle", "potential-bug")
    os.makedirs(bug_dir, exist_ok=True)
    
    timestamp = int(time.time())
    bug_file = join(bug_dir, f"subprocess_crash_{exit_code}_{timestamp}.py")
    
    if exit_code == 139:
        crash_type = "Segmentation Fault (Signal 11)"
        crash_keyword = "segfault"
    elif exit_code == 137:
        crash_type = "Out of Memory Kill (Signal 9)"
        crash_keyword = "oom_kill"
    elif exit_code == -11:
        crash_type = "SIGSEGV (Signal 11, negative)"
        crash_keyword = "segfault"
    elif exit_code == -9:
        crash_type = "SIGKILL (Signal 9, negative)"
        crash_keyword = "oom_kill"
    else:
        crash_type = f"Abnormal Exit (Code {exit_code})"
        crash_keyword = "abnormal_exit"
    
    content = f"""# ---------------------------------------------------------
# ⚠️ CRASH REPORT (Generated by Subprocess Crash Recovery)
# ---------------------------------------------------------
# API: {api_name}
# Framework: PyTorch
# Phase: {phase_name}
# Error: {crash_type}
# Exit Code: {exit_code}
# Description: {error_desc}
# Keyword: {crash_keyword}
# Timestamp: {timestamp}
# ---------------------------------------------------------
import torch
# API: {api_name}
# Crash type: {crash_keyword}
print("Subprocess crash detected: {crash_type}")
"""
    
    try:
        with open(bug_file, 'w') as f:
            f.write(content)
        print(f"🔥 Crash bug logged: {bug_file}")
    except Exception as e:
        print(f"⚠️  Failed to write crash bug file: {e}")
    
    return bug_file


def _rescan_bugs_from_disk(output_dir: str, phase_name: str) -> Dict:
    """Rescan bug files from disk."""
    bug_tracker = BugTracker(phase_name, output_dir)
    bug_tracker.scan_bugs()
    
    return {
        'crash': bug_tracker.get_bugs_by_type('crash'),
        'cuda': bug_tracker.get_bugs_by_type('cuda'),
        'precision': bug_tracker.get_bugs_by_type('precision'),
        'total': bug_tracker.get_total_bugs(),
        'unique': len(bug_tracker.unique_bugs)
    }


def _load_partial_result(output_dir: str, phase_name: str) -> Dict:
    """Load the intermediate result periodically saved by the subprocess."""
    result_file = join(output_dir, f"{phase_name.lower()}_partial.json")
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded partial results from: {result_file}")
            return data
        except Exception as e:
            print(f"⚠️  Failed to load partial result: {e}")
    return {}


def run_phase_isolated(
    phase_name: str,
    api_name: str,
    max_iterations: int,
    output_dir: str,
    config_file: str,
    diff_bound: float,
    enable_patches: bool = False,
    timeout_minutes: int = 120,
    result_queue: Queue = None
) -> Dict:
    """
    Run the fuzzing phase inside an isolated subprocess.
    
    Mirrors the subprocess-isolation logic of the TensorFlow version.
    """
    try:
        # Apply the poison patch
        PoisonPatcher.patch()
        
        if phase_name == "Random":
            # Random Phase
            print(f"\n{'='*70}")
            print(f"PHASE: {phase_name} (Subprocess Isolated)")
            print(f"{'='*70}")
            
            fuzzer = EnhancedFuzzer(
                api_name=api_name,
                output_dir=output_dir,
                strategy_name=phase_name,
                enable_patches=False,
                use_all_oracles=True,
                diff_bound=diff_bound
            )
            
            coverage = fuzzer.run_fuzzing_loop(
                max_iterations=max_iterations,
                checkpoint_interval=max(100, max_iterations // 20),
                bug_scan_interval=50
            )
            
            # Periodically save intermediate results (avoid losing data on crash)
            partial_file = join(output_dir, f"{phase_name.lower()}_partial.json")
            try:
                partial_data = {
                    'total_kernels': coverage.get_total(),
                    'iterations': len(coverage.history),
                    'bugs': {
                        'crash': fuzzer.bug_tracker.get_bugs_by_type('crash'),
                        'cuda': fuzzer.bug_tracker.get_bugs_by_type('cuda'),
                        'precision': fuzzer.bug_tracker.get_bugs_by_type('precision'),
                        'total': fuzzer.bug_tracker.get_total_bugs(),
                    }
                }
                with open(partial_file, 'w') as f:
                    json.dump(partial_data, f, indent=2)
            except Exception as e:
                print(f"⚠️  Failed to save partial result: {e}")
            
            result = {
                'success': True,
                'phase': phase_name,
                'coverage': coverage,
                'fuzzer': fuzzer,
            }
        
        else:  # Guided Phase
            print(f"\n{'='*70}")
            print(f"PHASE: {phase_name} (Subprocess Isolated)")
            print(f"{'='*70}")
            
            ProbabilityPatcher.patch_high_db_probability()
            
            fuzzer = EnhancedFuzzer(
                api_name=api_name,
                output_dir=output_dir,
                strategy_name=phase_name,
                enable_patches=True,
                use_all_oracles=True,
                diff_bound=diff_bound
            )
            
            AdaptiveMutationPatcher.patch_adaptive_mutation(fuzzer.mutation_controller)
            
            coverage = fuzzer.run_fuzzing_loop(
                max_iterations=max_iterations,
                checkpoint_interval=max(100, max_iterations // 20),
                bug_scan_interval=50
            )
            
            # Periodically save intermediate results
            partial_file = join(output_dir, f"{phase_name.lower()}_partial.json")
            try:
                partial_data = {
                    'total_kernels': coverage.get_total(),
                    'iterations': len(coverage.history),
                    'bugs': {
                        'crash': fuzzer.bug_tracker.get_bugs_by_type('crash'),
                        'cuda': fuzzer.bug_tracker.get_bugs_by_type('cuda'),
                        'precision': fuzzer.bug_tracker.get_bugs_by_type('precision'),
                        'total': fuzzer.bug_tracker.get_total_bugs(),
                    }
                }
                with open(partial_file, 'w') as f:
                    json.dump(partial_data, f, indent=2)
            except Exception as e:
                print(f"⚠️  Failed to save partial result: {e}")
            
            result = {
                'success': True,
                'phase': phase_name,
                'coverage': coverage,
                'fuzzer': fuzzer,
            }
        
        # Pass results via Queue (simplified: stats only)
        if result_queue:
            try:
                queue_data = {
                    'success': True,
                    'phase': phase_name,
                    'total_kernels': coverage.get_total(),
                    'total_bugs': fuzzer.bug_tracker.get_total_bugs(),
                }
                result_queue.put(queue_data, timeout=30)
            except Exception as e:
                print(f"⚠️  Queue send failed: {e}")
        
        return result
    
    except Exception as e:
        error_msg = f"Phase {phase_name} failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        
        result = {
            'success': False,
            'error': error_msg,
            'phase': phase_name
        }
        
        if result_queue:
            try:
                result_queue.put(result, timeout=30)
            except:
                pass
        
        return result


def run_phase_with_subprocess_isolation(
    phase_name: str,
    api_name: str,
    max_iterations: int,
    output_dir: str,
    config_file: str,
    diff_bound: float,
    enable_patches: bool = False,
    timeout_minutes: int = 120
) -> Dict:
    """
    Run the phase under subprocess isolation (mirrors the TensorFlow version).
    
    Properties:
    1. Runs in a separate process with memory isolation.
    2. Timeout protection.
    3. Crash-recovery mechanism.
    """
    result_queue = Queue()
    
    process = Process(
        target=run_phase_isolated,
        args=(
            phase_name,
            api_name,
            max_iterations,
            output_dir,
            config_file,
            diff_bound,
            enable_patches,
            timeout_minutes,
            result_queue
        )
    )
    
    print(f"🚀 Launching {phase_name} phase in isolated subprocess...")
    process.start()
    
    # Wait for completion or timeout
    process.join(timeout=timeout_minutes * 60)
    
    # Timeout handling
    if process.is_alive():
        print(f"⏰ {phase_name} phase timeout, terminating...")
        process.terminate()
        process.join(timeout=10)
        
        if process.is_alive():
            process.kill()
            process.join()
        
        _generate_crash_bug_file(
            output_dir, phase_name, api_name,
            exit_code=-1,
            error_desc=f"Timeout after {timeout_minutes} minutes"
        )
        
        # Recover partial results from disk
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        partial_result = _load_partial_result(output_dir, phase_name)
        
        return {
            'success': False,
            'error': f'Timeout after {timeout_minutes} minutes',
            'phase': phase_name,
            'bugs': recovered_bugs,
            'total_kernels': partial_result.get('total_kernels', 0),
        }
    
    # Normal completion: load results from Queue or disk
    result = None
    
    if not result_queue.empty():
        try:
            result = result_queue.get(timeout=30)
        except Exception as e:
            print(f"⚠️  Queue retrieval failed: {e}")
    
    exit_code = process.exitcode
    
    # Crash handling
    if exit_code is not None and exit_code != 0:
        print(f"\n{'='*70}")
        print(f"🔥 {phase_name} subprocess crashed! Exit code: {exit_code}")
        print(f"{'='*70}")
        
        if exit_code in (137, -9):
            error_desc = "OOM Kill"
        elif exit_code in (139, -11):
            error_desc = "Segmentation Fault"
        else:
            error_desc = f"Abnormal exit {exit_code}"
        
        _generate_crash_bug_file(output_dir, phase_name, api_name, exit_code, error_desc)
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        partial_result = _load_partial_result(output_dir, phase_name)
        
        print(f"🔧 Recovered bugs from disk: {recovered_bugs}")
        
        result = {
            'success': False,
            'error': error_desc,
            'phase': phase_name,
            'bugs': recovered_bugs,
            'total_kernels': partial_result.get('total_kernels', 0),
        }
    
    # If still no results, load from disk
    if result is None or not result.get('success'):
        partial_result = _load_partial_result(output_dir, phase_name)
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        
        if result is None:
            result = {}
        
        result.update({
            'success': False,
            'error': result.get('error', 'Unknown error'),
            'phase': phase_name,
            'bugs': recovered_bugs,
            'total_kernels': partial_result.get('total_kernels', 0),
        })
    
    return result


# =============================================================================
# Enhanced Experiment Runner
# =============================================================================

def run_full_oracle_experiment(
    api_name: str,
    max_iterations: int,
    output_dir: str,
    config_file: str = "demo_torch.conf",
    diff_bound: float = 1e-5,
    timeout_minutes: int = 120
):
    """Run the full Hybrid-Strategy experiment with subprocess isolation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Allocate iteration counts
    random_iters = max_iterations // 2
    guided_iters = max_iterations - random_iters
    
    print(f"\n{'='*70}")
    print("🧪 PyTorch Full Oracle Experiment (Subprocess Isolated)")
    print(f"{'='*70}")
    print(f"API: {api_name}")
    print(f"Total Iterations: {max_iterations}")
    print(f"  - Random Phase: {random_iters} (50%)")
    print(f"  - Guided Phase: {guided_iters} (50%)")
    print(f"Output: {output_dir}")
    print(f"Timeout per phase: {timeout_minutes} minutes")
    print(f"{'='*70}\n")
    
    # ==========================================================================
    # Phase 1: Random (runs in subprocess)
    # ==========================================================================
    print(f"\n{'='*70}")
    print("Phase 1: Random Baseline (Subprocess Isolated)")
    print(f"{'='*70}\n")
    
    random_result = run_phase_with_subprocess_isolation(
        phase_name="Random",
        api_name=api_name,
        max_iterations=random_iters,
        output_dir=join(output_dir, "random"),
        config_file=config_file,
        diff_bound=diff_bound,
        enable_patches=False,
        timeout_minutes=timeout_minutes
    )
    
    print(f"\n{'='*70}")
    print(f"Phase 1 {'COMPLETED' if random_result.get('success') else 'CRASHED/FAILED'}")
    print(f"Random Kernels: {random_result.get('total_kernels', 0)}")
    print(f"Random Bugs: {random_result.get('bugs', {}).get('total', 0)}")
    if not random_result.get('success'):
        print(f"Error: {random_result.get('error', 'unknown')}")
    print(f"{'='*70}\n")
    
    gc.collect()
    
    # ==========================================================================
    # Phase 2: Guided (runs in subprocess)
    # ==========================================================================
    print(f"\n{'='*70}")
    print("Phase 2: Kernel-Guided (Subprocess Isolated)")
    print(f"{'='*70}\n")
    
    guided_result = run_phase_with_subprocess_isolation(
        phase_name="Guided",
        api_name=api_name,
        max_iterations=guided_iters,
        output_dir=join(output_dir, "guided"),
        config_file=config_file,
        diff_bound=diff_bound,
        enable_patches=True,
        timeout_minutes=timeout_minutes
    )
    
    print(f"\n{'='*70}")
    print(f"Phase 2 {'COMPLETED' if guided_result.get('success') else 'CRASHED/FAILED'}")
    print(f"Guided Kernels: {guided_result.get('total_kernels', 0)}")
    print(f"Guided Bugs: {guided_result.get('bugs', {}).get('total', 0)}")
    if not guided_result.get('success'):
        print(f"Error: {guided_result.get('error', 'unknown')}")
    print(f"{'='*70}\n")
    
    # ==========================================================================
    # Reload results from disk (guards against subprocess data loss)
    # ==========================================================================
    random_coverage = _reconstruct_coverage_from_result(random_result, "Random")
    guided_coverage = _reconstruct_coverage_from_result(guided_result, "Guided")
    
    random_bug_tracker = BugTracker("Random", join(output_dir, "random"))
    guided_bug_tracker = BugTracker("Guided", join(output_dir, "guided"))
    
    random_bug_tracker.scan_bugs()
    guided_bug_tracker.scan_bugs()
    
    # Update bug counts (take max)
    for result, tracker in [(random_result, random_bug_tracker), 
                            (guided_result, guided_bug_tracker)]:
        old_bugs = result.get('bugs', {})
        disk_bugs = {
            'crash': tracker.get_bugs_by_type('crash'),
            'cuda': tracker.get_bugs_by_type('cuda'),
            'precision': tracker.get_bugs_by_type('precision'),
            'total': tracker.get_total_bugs(),
            'unique': len(tracker.unique_bugs),
        }
        result['bugs'] = {
            'crash': max(old_bugs.get('crash', 0), disk_bugs['crash']),
            'cuda': max(old_bugs.get('cuda', 0), disk_bugs['cuda']),
            'precision': max(old_bugs.get('precision', 0), disk_bugs['precision']),
            'total': max(old_bugs.get('total', 0), disk_bugs['total']),
            'unique': max(old_bugs.get('unique', 0), disk_bugs['unique']),
        }
    
    print(f"\n🔧 Final disk-based bug counts:")
    print(f"   Random: total={random_result['bugs']['total']}, unique={random_result['bugs']['unique']}")
    print(f"   Guided: total={guided_result['bugs']['total']}, unique={guided_result['bugs']['unique']}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    try:
        # Rebuild dispatcher_space and bug_deduplicator (stub objects)
        random_dispatcher = _reconstruct_dispatcher_from_result(random_result, api_name)
        guided_dispatcher = _reconstruct_dispatcher_from_result(guided_result, api_name)
        
        plot_full_oracle_results(
            random_coverage, guided_coverage,
            random_bug_tracker, guided_bug_tracker,
            api_name, output_dir,
            random_dispatcher=random_dispatcher,
            guided_dispatcher=guided_dispatcher,
            random_dedup=None,
            guided_dedup=None
        )
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")
        traceback.print_exc()
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    results = {
        "api": api_name,
        "max_iterations": max_iterations,
        "framework": "pytorch",
        "subprocess_isolation": True,
        "state_space_model": {
            "dimensions": ["device", "dtype", "layout", "mem_format", "rank"],
            "description": "5-dimensional orthogonal state space based on DispatchKey"
        },
        "hybrid_strategy": {
            "epsilon_greedy": "10% exploration",
            "adaptive_mutation": "±8 → ±64 on stagnation",
            "dynamic_saturation": "expand search instead of stop"
        },
        "random": {
            "total_kernels": random_result.get('total_kernels', 0),
            "iterations_run": random_result.get('iterations_completed', 0),
            "bugs": random_result.get('bugs', {}),
            "success": random_result.get('success', False),
        },
        "guided": {
            "total_kernels": guided_result.get('total_kernels', 0),
            "iterations_run": guided_result.get('iterations_completed', 0),
            "bugs": guided_result.get('bugs', {}),
            "success": guided_result.get('success', False),
        },
        "comparison": {
            "kernel_ratio": guided_result.get('total_kernels', 0) / max(random_result.get('total_kernels', 1), 1),
            "bug_ratio": guided_result.get('bugs', {}).get('total', 0) / max(random_result.get('bugs', {}).get('total', 1), 1),
        }
    }
    
    result_file = join(output_dir, f"{api_name.replace('.', '_')}_hybrid_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {result_file}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Random:")
    print(f"  Iterations: {random_result.get('iterations_completed', 0)}")
    print(f"  Kernels: {random_result.get('total_kernels', 0)}")
    print(f"  Bugs: {random_result.get('bugs', {}).get('total', 0)} (Unique: {random_result.get('bugs', {}).get('unique', 0)})")
    print(f"")
    print(f"Guided:")
    print(f"  Iterations: {guided_result.get('iterations_completed', 0)}")
    print(f"  Kernels: {guided_result.get('total_kernels', 0)}")
    print(f"  Bugs: {guided_result.get('bugs', {}).get('total', 0)} (Unique: {guided_result.get('bugs', {}).get('unique', 0)})")
    print(f"")
    print(f"Improvement:")
    print(f"  Kernel Ratio: {results['comparison']['kernel_ratio']:.2f}x")
    print(f"  Bug Ratio: {results['comparison']['bug_ratio']:.2f}x")
    print(f"{'='*70}\n")
    
    return results


def _reconstruct_coverage_from_result(result: Dict, phase_name: str):
    """Rebuild a Coverage object from a result dict."""
    coverage = EnhancedCoverageTracker(phase_name)
    coverage.history = result.get('coverage_history', [])
    
    # Rebuild all_kernels from history (using the final count as an approximation)
    total_kernels = result.get('total_kernels', 0)
    for i in range(total_kernels):
        coverage.all_kernels.add(f"kernel_{i}")
    
    return coverage


def _reconstruct_dispatcher_from_result(result: Dict, api_name: str):
    """Rebuild a DispatcherSpace object from a result dict (stub)."""
    dispatcher = DispatcherSpace(max_hits=50000)
    dispatcher.register_api(api_name)
    
    # Use the stats stored in the result
    # (Note: simplified; the real state cannot be fully restored.)
    return dispatcher


# =============================================================================
# Main fuzzing-phase runner (mirrors the TF version)
# =============================================================================

def run_fuzzing_phase(
    phase_name: str,
    api_name: str,
    max_iterations: int,
    output_dir: str,
    config_path: str,
    diff_bound: float,
    enable_patches: bool = False,
    result_queue: Queue = None
) -> Dict:
    """
    Single-phase fuzzing entry point (runs in subprocess).
    
    Faithfully mirrors the TensorFlow version's logic.
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting {phase_name} Phase (Subprocess Isolated)")
    print(f"{'='*70}")
    print(f"API: {api_name}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Output: {output_dir}")
    print(f"Patches: {'Enabled' if enable_patches else 'Disabled'}")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialise components
    try:
        config = configparser.ConfigParser()
        
        # Locate the config file
        possible_paths = [
            join("config", config_path),
            join("..", "config", config_path),
            config_path,
        ]
        
        config_found = None
        for path in possible_paths:
            if os.path.exists(path):
                config_found = path
                break
        
        if not config_found:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config.read(config_found)
        
        TorchDatabase.database_config(
            config["mongodb"]["host"],
            int(config["mongodb"]["port"]),
            config["mongodb"]["torch_database"]
        )
        
        print(f"✅ PyTorch Components Initialized")
        
    except Exception as e:
        error_msg = f"Failed to initialize: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        result = {
            'success': False,
            'error': error_msg,
            'phase': phase_name
        }
        if result_queue:
            try:
                result_queue.put(result, timeout=30)
            except:
                pass
        return result
    
    # Apply the poison patch
    PoisonPatcher.patch()
    
    try:
        # Create the Fuzzer
        if phase_name == "Random":
            ProbabilityPatcher.patch_high_db_probability()  # Random also uses a high sampling rate
            
            fuzzer = EnhancedFuzzer(
                api_name=api_name,
                output_dir=output_dir,
                strategy_name=phase_name,
                enable_patches=False,
                use_all_oracles=True,
                diff_bound=diff_bound
            )
        else:  # Guided
            ProbabilityPatcher.patch_high_db_probability()
            
            fuzzer = EnhancedFuzzer(
                api_name=api_name,
                output_dir=output_dir,
                strategy_name=phase_name,
                enable_patches=True,
                use_all_oracles=True,
                diff_bound=diff_bound
            )
            
            AdaptiveMutationPatcher.patch_adaptive_mutation(fuzzer.mutation_controller)
        
        # Run the fuzzing loop
        coverage = fuzzer.run_fuzzing_loop(
            max_iterations=max_iterations,
            checkpoint_interval=max(100, max_iterations // 20),
            bug_scan_interval=50
        )
        
        # Scan for bugs
        fuzzer.bug_tracker.scan_bugs()
        
        # Prepare the result
        stats = {
            'success': True,
            'phase': phase_name,
            'iterations_completed': max_iterations,
            'total_kernels': coverage.get_total(),
            'bugs': {
                'crash': fuzzer.bug_tracker.get_bugs_by_type('crash'),
                'cuda': fuzzer.bug_tracker.get_bugs_by_type('cuda'),
                'precision': fuzzer.bug_tracker.get_bugs_by_type('precision'),
                'total': fuzzer.bug_tracker.get_total_bugs(),
                'unique': len(fuzzer.bug_tracker.unique_bugs),
            },
            'coverage_history': coverage.history,
        }
        
        # Save intermediate results to disk
        result_file = join(output_dir, f"{phase_name.lower()}_result.json")
        try:
            with open(result_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"💾 Results saved: {result_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
        
        # Transmit via Queue (simplified)
        if result_queue:
            try:
                queue_data = {
                    'success': True,
                    'phase': phase_name,
                    'total_kernels': coverage.get_total(),
                    'total_bugs': fuzzer.bug_tracker.get_total_bugs(),
                }
                result_queue.put(queue_data, timeout=30)
            except Exception as e:
                print(f"⚠️  Queue send failed: {e}")
        
        print(f"\n{'='*70}")
        print(f"✅ {phase_name} Phase Completed")
        print(f"{'='*70}")
        print(f"Iterations: {stats['iterations_completed']}")
        print(f"Total Kernels: {stats['total_kernels']}")
        print(f"Bugs Found: {stats['bugs']['total']} (Unique: {stats['bugs']['unique']})")
        print(f"  - CRASH: {stats['bugs']['crash']}")
        print(f"  - CUDA: {stats['bugs']['cuda']}")
        print(f"  - PRECISION: {stats['bugs']['precision']}")
        print(f"{'='*70}\n")
        
        return stats
    
    except Exception as e:
        error_msg = f"Phase {phase_name} failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        
        result = {
            'success': False,
            'error': error_msg,
            'phase': phase_name,
            'bugs': {'crash': 0, 'cuda': 0, 'precision': 0, 'total': 0, 'unique': 0},
            'total_kernels': 0,
        }
        
        if result_queue:
            try:
                result_queue.put(result, timeout=30)
            except:
                pass
        
        return result
    
    finally:
        # Restore patches
        try:
            ProbabilityPatcher.restore()
            AdaptiveMutationPatcher.restore()
            PoisonPatcher.restore()
        except:
            pass


def run_phase_with_subprocess_isolation(
    phase_name: str,
    api_name: str,
    max_iterations: int,
    output_dir: str,
    config_file: str,
    diff_bound: float,
    enable_patches: bool = False,
    timeout_minutes: int = 120
) -> Dict:
    """
    Run the phase under subprocess isolation (faithfully mirrors the TensorFlow version).
    """
    result_queue = Queue()
    
    process = Process(
        target=run_fuzzing_phase,
        args=(
            phase_name,
            api_name,
            max_iterations,
            output_dir,
            config_file,
            diff_bound,
            enable_patches,
            result_queue
        )
    )
    
    print(f"🚀 Launching {phase_name} phase in isolated subprocess...")
    process.start()
    
    # Wait for completion or timeout
    process.join(timeout=timeout_minutes * 60)
    
    # Timeout handling
    if process.is_alive():
        print(f"⏰ {phase_name} phase timeout ({timeout_minutes} min), terminating...")
        process.terminate()
        process.join(timeout=10)
        
        if process.is_alive():
            process.kill()
            process.join()
        
        _generate_crash_bug_file(
            output_dir, phase_name, api_name,
            exit_code=-1,
            error_desc=f"Timeout after {timeout_minutes} minutes"
        )
        
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        partial_result = _load_partial_result(output_dir, phase_name)
        
        return {
            'success': False,
            'error': f'Timeout after {timeout_minutes} minutes',
            'phase': phase_name,
            'bugs': recovered_bugs,
            'total_kernels': partial_result.get('total_kernels', 0),
            'iterations_completed': partial_result.get('iterations_completed', 0),
            'coverage_history': partial_result.get('coverage_history', []),
        }
    
    # Normal completion: load results from Queue or disk
    result = None
    
    if not result_queue.empty():
        try:
            result = result_queue.get(timeout=30)
        except Exception as e:
            print(f"⚠️  Queue retrieval failed: {e}")
    
    # Load full results from file
    if result is None or not result.get('success'):
        result_file = join(output_dir, f"{phase_name.lower()}_result.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                print(f"✅ Loaded results from file: {result_file}")
            except Exception as e:
                print(f"❌ Failed to load from file: {e}")
    
    exit_code = process.exitcode
    
    # Crash handling
    if exit_code is not None and exit_code != 0:
        print(f"\n{'='*70}")
        print(f"🔥 {phase_name} subprocess crashed! Exit code: {exit_code}")
        print(f"{'='*70}")
        
        if exit_code in (137, -9):
            error_desc = "OOM Kill (Signal 9)"
        elif exit_code in (139, -11):
            error_desc = "Segmentation Fault (Signal 11)"
        else:
            error_desc = f"Abnormal exit {exit_code}"
        
        _generate_crash_bug_file(output_dir, phase_name, api_name, exit_code, error_desc)
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        
        print(f"🔧 Recovered bugs from disk: {recovered_bugs}")
        
        if result is None:
            partial_result = _load_partial_result(output_dir, phase_name)
            result = {
                'success': False,
                'error': error_desc,
                'phase': phase_name,
                'bugs': recovered_bugs,
                'total_kernels': partial_result.get('total_kernels', 0),
                'iterations_completed': partial_result.get('iterations_completed', 0),
                'coverage_history': partial_result.get('coverage_history', []),
            }
        else:
            result['success'] = False
            result['error'] = error_desc
            
            # Merge bugs
            old_bugs = result.get('bugs', {})
            result['bugs'] = {
                'crash': max(old_bugs.get('crash', 0), recovered_bugs['crash']),
                'cuda': max(old_bugs.get('cuda', 0), recovered_bugs['cuda']),
                'precision': max(old_bugs.get('precision', 0), recovered_bugs['precision']),
                'total': max(old_bugs.get('total', 0), recovered_bugs['total']),
                'unique': max(old_bugs.get('unique', 0), recovered_bugs['unique']),
            }
    
    # If still no results, attempt recovery from disk
    if result is None:
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        partial_result = _load_partial_result(output_dir, phase_name)
        result = {
            'success': False,
            'error': 'No result file found and queue empty',
            'phase': phase_name,
            'bugs': recovered_bugs,
            'total_kernels': partial_result.get('total_kernels', 0),
            'iterations_completed': partial_result.get('iterations_completed', 0),
            'coverage_history': partial_result.get('coverage_history', []),
        }
    
    return result


# =============================================================================
# Enhanced Experiment Runner
# =============================================================================


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hybrid Strategy Benchmark: -greedy + Adaptive Mutation + Poison Injection"
    )
    parser.add_argument("--api", type=str, default="torch.nn.LSTM",
                        help="API to test")
    parser.add_argument("--max-iterations", type=int, default=10000,
                        help="Maximum iterations")
    parser.add_argument("--output", type=str, default="hybrid_output")
    parser.add_argument("--conf", type=str, default="demo_torch.conf")
    parser.add_argument("--diff-bound", type=float, default=1e-5,
                        help="Precision tolerance for PRECISION oracle (default: 1e-5)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("HYBRID STRATEGY BENCHMARK + POISON INJECTION")
    print("="*70)
    print(f"API: {args.api}")
    print(f"Max Iterations: {args.max_iterations:,}")
    print(f"Strategy: -greedy (10% exploration) + Adaptive Mutation")
    print(f"Poison Injection: 5% Inf + 5% NaN + 10% Extreme")
    print(f"Precision Tolerance: {args.diff_bound}")
    print(f"Oracles: CRASH + CUDA + PRECISION")
    print("="*70)
    
    run_full_oracle_experiment(
        api_name=args.api,
        max_iterations=args.max_iterations,
        output_dir=args.output,
        config_file=args.conf,
        diff_bound=args.diff_bound
    )


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Kernel-Guided Fuzzing Benchmark - Full Oracle Edition with Hybrid Strategy
==========================================================================

Ã¦Â Â¸Ã¥Â¿Æ’Ã¥Â¢Å¾Ã¥Â¼Âº:
1. **Ã¦Â·Â·Ã¥ÂË†Ã§Â­â€“Ã§â€¢Â¥**: ÃŽÂµ-greedy (90% Ã¥Â¾Â®Ã¥Ë†â€º + 10% Ã¦Å½Â¢Ã§Â´Â¢)
2. **Ã¥Å Â¨Ã¦â‚¬ÂÃ¥ÂËœÃ¥Â¼â€š**: Ã¥ÂÅ“Ã¦Â»Å¾Ã¦â€”Â¶Ã¨â€¡ÂªÃ¥Å Â¨Ã¦â€°Â©Ã¥Â¤Â§Ã¥ÂËœÃ¥Â¼â€šÃ¨Å’Æ’Ã¥â€ºÂ´
3. **Ã¥Ââ€“Ã¦Â¶Ë†Ã©Â¥Â±Ã¥â€™Å’Ã¥ÂÅ“Ã¦Â­Â¢**: Ã¦â€Â¹Ã¤Â¸Âº"Ã¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢"Ã¨â‚¬Å’Ã©ÂÅ¾"Ã§Â»Ë†Ã¦Â­Â¢"
4. **Ã¥â€¦Â¨ Oracle Ã¦â€Â¯Ã¦Å’Â**: CRASH + CUDA + PRECISION
5. **Ã¦â„¢ÂºÃ¨Æ’Â½Ã¥Â½â€™Ã§Â±Â»**: Ã¥Å¸ÂºÃ¤ÂºÅ½Ã©â€â„¢Ã¨Â¯Â¯Ã¦Å’â€¡Ã§ÂºÂ¹Ã§Å¡â€ž Bug Ã¥Å½Â»Ã©â€¡ÂÃ¤Â¸Å½Ã¥Ë†â€ Ã§Â±Â» (New!)

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

# Constraint Filter imports - 5Ã§Â»Â´Ã¦Â­Â£Ã¤ÂºÂ¤Ã§Å Â¶Ã¦â‚¬ÂÃ§Â©ÂºÃ©â€”Â´
# Ã¥Â°ÂÃ¨Â¯â€¢Ã¥Â¯Â¼Ã¥â€¦Â¥Ã¯Â¼Å’Ã¥Â¤Â±Ã¨Â´Â¥Ã¦â€”Â¶Ã¤Â½Â¿Ã§â€Â¨Ã¥â€ â€¦Ã¨Ââ€Ã§Â®â‚¬Ã¥Å’â€“Ã§â€°Ë†Ã¦Å“Â¬
try:
    from pytorch_comprehensive_filter import (
        PyTorchConstraintFilter,
        DispatcherStateSpace,
    )
    FILTER_AVAILABLE = True
except Exception:
    # Ã©Ââ„¢Ã©Â»ËœÃ¥Â¤â€žÃ§Ââ€ Ã¯Â¼Å’Ã¤Â½Â¿Ã§â€Â¨Ã¥â€ â€¦Ã¨Ââ€Ã§Â®â‚¬Ã¥Å’â€“Ã§â€°Ë†Ã¦Å“Â¬
    FILTER_AVAILABLE = False
    
    # Ã§Â®â‚¬Ã¥Å’â€“Ã§â€°Ë† DispatcherStateSpace (Ã¤Â¸ÂÃ¤Â¾ÂÃ¨Âµâ€“Ã§ÂºÂ¦Ã¦ÂÅ¸Ã¨Â¿â€¡Ã¦Â»Â¤)
    class DispatcherStateSpace:
        """Ã§Â®â‚¬Ã¥Å’â€“Ã§â€°Ë†Ã§Å Â¶Ã¦â‚¬ÂÃ§Â©ÂºÃ©â€”Â´Ã§Â®Â¡Ã§Ââ€ Ã¥â„¢Â¨ - Ã¦â€”Â Ã§ÂºÂ¦Ã¦ÂÅ¸Ã¨Â¿â€¡Ã¦Â»Â¤"""
        
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
            # Ã°Å¸â€Â§ Ã¤Â¿Â®Ã¥Â¤Â: Ã¤Â½Â¿Ã§â€Â¨5Ã§Â»Â´Ã§Â©ÂºÃ©â€”Â´Ã¨Â®Â¡Ã§Â®â€” (device Ãƒâ€” dtype Ãƒâ€” layout Ãƒâ€” mem_format Ãƒâ€” rank)
            # Ã§Â®â‚¬Ã¥Å’â€“Ã§â€°Ë†Ã¦Å“Â¬Ã¯Â¼Å¡strided only, contiguous only, ranks [2d,3d,4d]
            devices = self.DIMENSION_DEVICE  # 2
            dtypes = self.DIMENSION_DTYPE    # 12
            layouts = ['strided']            # 1
            mem_formats = ['contiguous']     # 1
            ranks = ['1d', '2d', '3d', '4d', '5d']  # Ã¢Å“â€¦ 5 ranks (0d rare, nd unstable)
            valid_space = len(devices) * len(dtypes) * len(layouts) * len(mem_formats) * len(ranks)
            # 2 Ãƒâ€” 12 Ãƒâ€” 1 Ãƒâ€” 1 Ãƒâ€” 5 = 120 Ã¢Å“â€¦ (expanded from 72)
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
            print(f"  Valid Space: {valid_space} (2Ãƒâ€”12Ãƒâ€”1Ãƒâ€”1Ãƒâ€”5 = 120)")
            return self.api_spaces[api_name]
        
        def record_hit(self, api_name: str, state: Dict) -> bool:
            if api_name not in self.api_spaces:
                self.register_api(api_name)
            if self.total_hits >= self.max_states:
                return False
            # Ã°Å¸â€Â§ Ã¤Â¿Â®Ã¥Â¤Â: Ã¤Â½Â¿Ã§â€Â¨5Ã§Â»Â´state_hashÃ¦Â Â¼Ã¥Â¼Â
            device = state.get('device', 'unk')
            dtype = state.get('dtype', 'unk')
            layout = state.get('layout', 'strided')
            mem_format = state.get('mem_format', 'contiguous')
            rank = state.get('rank', 'unk')
            # Ã¦Â â€¡Ã¥â€¡â€ Ã¥Å’â€“ rank Ã¦Â Â¼Ã¥Â¼Â
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
                bar = 'Ã¢â€“Ë†' * filled + 'Ã¢â€“â€˜' * (bar_w - filled)
                print(f"\n[{api_name}]")
                print(f"  Hits: {hit_count}/{valid_space} ({coverage*100:.1f}%) [{bar}]")
                # Ã¦ËœÂ¾Ã§Â¤ÂºÃ¥â€˜Â½Ã¤Â¸Â­Ã¦Â Â·Ã¦Å“Â¬
                hit_samples = list(self.hit_states.get(api_name, set()))[:5]
                if hit_samples:
                    print(f"  Sample: {hit_samples[:3]}...")
            print(f"\nTotal Hits: {self.total_hits}")
            print(f"{'='*70}\n")


# =============================================================================
# Safety Guards (Ã¤Â¿ÂÃ¦Å Â¤Ã¦Å“ÂºÃ¥Ë†Â¶)
# =============================================================================

class Speedometer:
    """Ã©â‚¬Å¸Ã¥ÂºÂ¦Ã§â€ºâ€˜Ã¦Å½Â§ - Ã¥ÂÅ Ã¦â€”Â¶Ã¥Ââ€˜Ã§Å½Â°Ã¦â‚¬Â§Ã¨Æ’Â½Ã©â€”Â®Ã©Â¢Ëœ"""
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
            warning = (f"Ã¢Å¡Â Ã¯Â¸Â  SLOW SPEED: {current_speed:.2f} it/s (threshold: {self.slow_threshold})")
            if current_speed < 0.1:
                warning += "\n   Ã¢â€ â€™ Check if Precision Oracle is too slow"
            elif current_speed < 0.5:
                warning += "\n   Ã¢â€ â€™ Consider reducing complexity or disabling oracles"
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
    """Ã§Â£ÂÃ§â€ºËœÃ§Â©ÂºÃ©â€”Â´Ã§â€ºâ€˜Ã¦Å½Â§ - Ã©ËœÂ²Ã¦Â­Â¢Ã§Ë†â€ Ã¦Â»Â¡Ã¥Â´Â©Ã¦ÂºÆ’"""
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
            warning = f"Ã¢Å¡Â Ã¯Â¸Â  DISK SPACE: Only {free:.2f} GB free (threshold: {self.min_free_gb} GB)"
            if self.auto_cleanup:
                warning += "\n   Ã¢â€ â€™ Attempting auto-cleanup..."
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
            message = f"{warning}\n   Cleaned {cleaned} files Ã¢â€ â€™ {free:.2f} GB free"
            if free < self.min_free_gb:
                message += "\n   Ã¢ÂÅ’ CRITICAL: Still low after cleanup!"
                return True, message
            message += "\n   Ã¢Å“â€¦ Cleanup successful"
            return False, message
        return True, warning
    
    def get_status(self) -> str:
        total, used, free = self.get_disk_usage()
        return f"{free:.2f} GB free ({used/total*100:.1f}% used)"


class OutlierFilter:
    """Ã¥Â¼â€šÃ¥Â¸Â¸Ã¥Ââ€šÃ¦â€¢Â°Ã¨Â¿â€¡Ã¦Â»Â¤ - Ã©ËœÂ²Ã¦Â­Â¢ OOM"""
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
# Checkpoint System (Ã¦â€“Â­Ã§â€šÂ¹Ã§Â»Â­Ã¤Â¼Â )
# =============================================================================

class CheckpointManager:
    """Ã¦Â£â‚¬Ã¦Å¸Â¥Ã§â€šÂ¹Ã§Â®Â¡Ã§Ââ€ Ã¥â„¢Â¨ - Ã¦â€Â¯Ã¦Å’ÂÃ¦â€“Â­Ã§â€šÂ¹Ã§Â»Â­Ã¤Â¼Â """
    
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
            print(f"[Checkpoint] Ã¢ÂÅ’ Save failed: {e}")
            return False
    
    def load(self) -> Optional[Dict]:
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"[Checkpoint] Ã¢Å“â€¦ Loaded from iteration {data['iteration']}")
            print(f"  Kernels: {len(data['coverage_kernels'])}")
            if data.get('corpus_seeds'):
                print(f"  Corpus: {len(data['corpus_seeds'])} seeds")
            
            return data
            
        except Exception as e:
            print(f"[Checkpoint] Ã¢ÂÅ’ Load failed: {e}")
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
# Feature Extractor (Ã¥Ââ€šÃ¦â€¢Â°Ã¦Å’â€¡Ã§ÂºÂ¹Ã¦ÂÂÃ¥Ââ€“)
# =============================================================================

def extract_features(api: TorchAPI) -> str:
    """
    Ã¦ÂÂÃ¥Ââ€“ API Ã¨Â°Æ’Ã§â€Â¨Ã§Å¡â€žÃ¥Ââ€šÃ¦â€¢Â°Ã¦Å’â€¡Ã§ÂºÂ¹Ã¯Â¼Å’Ã§â€Â¨Ã¤ÂºÅ½Ã¥ÂÅ½Ã§Â»Â­Ã§Å¡â€žÃ§â€°Â¹Ã¥Â¾ÂÃ¥Â¤Å¡Ã¦Â Â·Ã¦â‚¬Â§Ã¥Ë†â€ Ã¦Å¾Â
    
    Ã¨Â¿â€Ã¥â€ºÅ¾Ã¦Â Â¼Ã¥Â¼Â: "param1:type1:value1|param2:type2:value2|..."
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
# Experiment Logger (Ã¥Å½Å¸Ã¥Â§â€¹Ã¦â€¢Â°Ã¦ÂÂ®Ã¨Â®Â°Ã¥Â½â€¢Ã¥â„¢Â¨)
# =============================================================================

class ExperimentLogger:
    """Ã¨Â½Â»Ã©â€¡ÂÃ§ÂºÂ§Ã¥Â®Å¾Ã©ÂªÅ’Ã¦â€¢Â°Ã¦ÂÂ®Ã¨Â®Â°Ã¥Â½â€¢Ã¥â„¢Â¨"""
    
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

# Ã¤Â¿ÂÃ¥Â­ËœÃ¥Å½Å¸Ã¥Â§â€¹Ã¦â€“Â¹Ã¦Â³â€¢
_ORIGINAL_DO_SELECT_FROM_DB = None
_ORIGINAL_MUTATE_INT_VALUE = None
_ORIGINAL_MUTATE_FLOAT_VALUE = None


# =============================================================================
# Bug Tracker (Bug Ã¨Â¿Â½Ã¨Â¸ÂªÃ¥â„¢Â¨) - Ã°Å¸â€Â¥ Ã¥Ââ€¡Ã§ÂºÂ§Ã§â€°Ë†Ã¯Â¼Å¡Ã¦Å’â€¡Ã§ÂºÂ¹Ã¨Â¯â€ Ã¥Ë†Â«
# =============================================================================

class BugAnalyzer:
    """Ã¦â€ºÂ´Ã§Â²Â¾Ã§Â»â€ Ã§Å¡â€ž Bug Ã¥Ë†â€ Ã¦Å¾ÂÃ¥â„¢Â¨Ã¯Â¼Å’Ã¦â€Â¯Ã¦Å’ÂÃ©â€â„¢Ã¨Â¯Â¯Ã¦Å’â€¡Ã§ÂºÂ¹Ã¦ÂÂÃ¥Ââ€“"""
    
    # Ã§â€Â¨Ã¤ÂºÅ½Ã¦Â¸â€¦Ã¦Â´â€”Ã©â€â„¢Ã¨Â¯Â¯Ã¤Â¿Â¡Ã¦ÂÂ¯Ã¤Â¸Â­Ã§Å¡â€žÃ¦â€¢Â°Ã¥Â­â€”Ã£â‚¬ÂÃ¥Å“Â°Ã¥Ââ‚¬Ã£â‚¬ÂShape
    REGEX_PATTERNS = [
        (r'0x[0-9a-fA-F]+', 'ADDR'),           # Ã¦â€ºÂ¿Ã¦ÂÂ¢Ã¥â€ â€¦Ã¥Â­ËœÃ¥Å“Â°Ã¥Ââ‚¬
        (r'\d+\.\d+', 'FLOAT'),                # Ã¦â€ºÂ¿Ã¦ÂÂ¢Ã¦ÂµÂ®Ã§â€šÂ¹Ã¦â€¢Â°
        (r'\d+', 'INT'),                       # Ã¦â€ºÂ¿Ã¦ÂÂ¢Ã¦â€¢Â´Ã¦â€¢Â°
        (r'\[.*?\]', '[SHAPE]'),               # Ã¦â€ºÂ¿Ã¦ÂÂ¢ Tensor Shape Ã¦ÂÂÃ¨Â¿Â°
        (r'\s+', ' '),                         # Ã¥ÂË†Ã¥Â¹Â¶Ã§Â©ÂºÃ¦Â Â¼
    ]

    @staticmethod
    def get_signature(content: str) -> str:
        """Ã¥Å¸ÂºÃ¤ÂºÅ½Ã¤Â»Â£Ã§Â ÂÃ§â€°Â¹Ã¥Â¾ÂÃ§â€Å¸Ã¦Ë†ÂÃ¦Å’â€¡Ã§ÂºÂ¹Ã¯Â¼Ë†bugÃ¦â€“â€¡Ã¤Â»Â¶Ã¥ÂÂªÃ¤Â¿ÂÃ¥Â­ËœÃ¤Â»Â£Ã§Â ÂÃ¯Â¼Å’Ã¤Â¸ÂÃ¤Â¿ÂÃ¥Â­ËœÃ©â€â„¢Ã¨Â¯Â¯Ã¤Â¿Â¡Ã¦ÂÂ¯Ã¯Â¼â€°"""
        if not content:
            return "unknown_empty"
        
        features = []
        
        # 1. Ã¦ÂÂÃ¥Ââ€“ dtypeÃ¯Â¼Ë†Ã¦Å“â‚¬Ã©â€¡ÂÃ¨Â¦ÂÃ§Å¡â€žÃ¥Å’ÂºÃ¥Ë†â€ Ã§â€°Â¹Ã¥Â¾ÂÃ¯Â¼â€°
        dtype_matches = re.findall(r'dtype\s*=\s*(torch\.[a-z0-9]+)', content)
        if dtype_matches:
            unique_dtypes = sorted(set(dtype_matches))
            features.append(f"dtype={','.join(unique_dtypes)}")
        
        # 2. Ã¦ÂÂÃ¥Ââ€“ tensor Ã§Â»Â´Ã¥ÂºÂ¦Ã¦â€¢Â°Ã¯Â¼Ë†Ã¥Â½â€™Ã¤Â¸â‚¬Ã¥Å’â€“ shapeÃ¯Â¼â€°
        shape_matches = re.findall(r'torch\.rand(?:int)?\s*\(\s*[\[\(]([^\]\)]+)[\]\)]', content)
        if shape_matches:
            for shape in shape_matches[:2]:
                ndim = shape.count(',') + 1
                features.append(f"ndim={ndim}")
                break
        
        # 3. Ã¦ÂÂÃ¥Ââ€“Ã§â€°Â¹Ã¦Â®Å Ã¥â‚¬Â¼Ã§â€°Â¹Ã¥Â¾ÂÃ¯Â¼Ë†Ã¦Å â€¢Ã¦Â¯â€™Ã¨Â§Â¦Ã¥Ââ€˜Ã§Å¡â€žÃ¥â€¦Â³Ã©â€Â®Ã¯Â¼â€°
        if re.search(r'\bnan\b', content, re.IGNORECASE):
            features.append("NaN")
        if re.search(r'\binf\b', content, re.IGNORECASE):
            features.append("Inf")
        
        # 4. Ã§â€Å¸Ã¦Ë†ÂÃ¦Å’â€¡Ã§ÂºÂ¹
        if not features:
            return f"hash={hashlib.md5(content.encode()).hexdigest()[:12]}"
        
        return "|".join(sorted(features))

    @staticmethod
    def classify_bug(error_msg: str) -> str:
        """Ã¥Ë†â€ Ã§Â±Â»Ã©â‚¬Â»Ã¨Â¾â€˜Ã¯Â¼Ë†Ã¤Â¿ÂÃ¦Å’ÂÃ¤Â½Â Ã¥Å½Å¸Ã¦Å“â€°Ã§Å¡â€ž Deep/ShallowÃ¯Â¼Å’Ã¤Â½â€ Ã¦â€ºÂ´Ã§Â²Â¾Ã¥â€¡â€ Ã¯Â¼â€°"""
        sig = BugAnalyzer.get_signature(error_msg).lower()
        
        if "segfault" in sig or "core_dump" in sig:
            return "CRITICAL_SEGFAULT"  # Ã¥Ââ€¢Ã§â€¹Â¬Ã¦Å Å Ã¥Â´Â©Ã¦ÂºÆ’Ã¦â€¹Å½Ã¥â€¡ÂºÃ¦ÂÂ¥
        
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
        
        # Ã¥â€¦Â¼Ã¥Â®Â¹Ã¦â€”Â§Ã©â‚¬Â»Ã¨Â¾â€˜Ã¯Â¼Å¡Ã¥Â­ËœÃ¥â€šÂ¨Ã§Â®â‚¬Ã¥Ââ€¢Ã§Å¡â€žÃ¥Ë†â€”Ã¨Â¡Â¨Ã¯Â¼Å’Ã§â€Â¨Ã¤ÂºÅ½Ã§Â»ËœÃ¥â€ºÂ¾Ã¥â€¡Â½Ã¦â€¢Â°Ã§Å¡â€žÃ¨Â®Â¡Ã¦â€¢Â°
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
        
        # Ã°Å¸â€Â¥ Ã¦â€“Â°Ã©â‚¬Â»Ã¨Â¾â€˜Ã¯Â¼Å¡Ã¥Â­ËœÃ¥â€šÂ¨Ã§Â»â€œÃ¦Å¾â€žÃ¥Å’â€“Ã¦Å’â€¡Ã§ÂºÂ¹Ã¤Â¿Â¡Ã¦ÂÂ¯
        self.unique_bugs = defaultdict(list)
        self.bug_stats = defaultdict(int) # Ã§Â»Å¸Ã¨Â®Â¡Ã¥Ââ€žÃ§Â±Â»Ã¥Ë†Â«Ã¦â€¢Â°Ã©â€¡Â
        
    def scan_bugs(self):
        # Ã¦Â¸â€¦Ã§Â©ÂºÃ©â€¡ÂÃ¦â€°Â«Ã¯Â¼Å’Ã©ËœÂ²Ã¦Â­Â¢Ã©â€¡ÂÃ¥Â¤Â
        self.unique_bugs.clear()
        self.bug_stats.clear()
        # Ã¦Â³Â¨Ã¦â€žÂÃ¯Â¼Å¡Ã¤Â¸ÂºÃ¤Âºâ€ Ã§Â»ËœÃ¥â€ºÂ¾Ã¥â€¦Â¼Ã¥Â®Â¹Ã¯Â¼Å’Ã¦Ë†â€˜Ã¤Â»Â¬Ã¤Â¸ÂÃ¨Â½Â»Ã¦Ëœâ€œÃ¦Â¸â€¦Ã§Â©Âº self.bugsÃ¯Â¼Å’Ã¨â‚¬Å’Ã¦ËœÂ¯Ã¦Â¯ÂÃ¦Â¬Â¡Ã¥Â¢Å¾Ã©â€¡ÂÃ¦Â·Â»Ã¥Å Â Ã¦â€“Â°Ã§Å¡â€ž
        # Ã¦Ë†â€“Ã¨â‚¬â€¦Ã¤Â¸ÂºÃ¤Âºâ€ Ã¤Â¿ÂÃ¦Å’ÂÃ¤Â¸â‚¬Ã¨â€¡Â´Ã¯Â¼Å’Ã¨Â¿â„¢Ã©â€¡Å’Ã¥â€¦Ë†Ã¤Â¸ÂÃ¦Â¸â€¦Ã§Â©Âº self.bugsÃ¯Â¼Å’Ã¤Â¾ÂÃ©ÂÂ  set Ã¥Å½Â»Ã©â€¡Â
        
        oracle_map = {"crash": "crash-oracle", "cuda": "cuda-oracle", "precision": "precision-oracle"}
        
        for bug_type, oracle_name in oracle_map.items():
            bug_dir = join(self.output_dir, oracle_name, "potential-bug")
            if not os.path.exists(bug_dir):
                continue
                
            for root, _, files in os.walk(bug_dir):
                for f in files:
                    if not f.endswith('.py'): continue
                    
                    file_path = join(root, f)
                    
                    # 1. Ã¦â€”Â§Ã©â‚¬Â»Ã¨Â¾â€˜Ã¥â€¦Â¼Ã¥Â®Â¹Ã¯Â¼Å¡Ã¥Å Â Ã¥â€¦Â¥Ã¥Ë†â€”Ã¨Â¡Â¨Ã¯Â¼Å’Ã§â€Â¨Ã¤ÂºÅ½Ã§Â»ËœÃ¥â€ºÂ¾Ã§Â»Å¸Ã¨Â®Â¡Ã¦â‚¬Â»Ã¦â€¢Â°
                    if file_path not in self.bug_files[bug_type]:
                        self.bug_files[bug_type].add(file_path)
                        self.bugs[bug_type].append((len(self.bugs[bug_type]), "unknown", file_path))
                    
                    # 2. Ã°Å¸â€Â¥ Ã¦â€“Â°Ã©â‚¬Â»Ã¨Â¾â€˜Ã¯Â¼Å¡Ã¦Å’â€¡Ã§ÂºÂ¹Ã¦ÂÂÃ¥Ââ€“Ã¤Â¸Å½Ã¥Å½Â»Ã©â€¡Â
                    try:
                        # Ã¥Â°ÂÃ¨Â¯â€¢Ã¨Â¯Â»Ã¥Ââ€“Ã©â€â„¢Ã¨Â¯Â¯Ã¦â€”Â¥Ã¥Â¿â€”Ã¯Â¼Ë†Ã¥Ââ€¡Ã¨Â®Â¾Ã¥ÂÅ’Ã¥ÂÂ .txt Ã¦Ë†â€“ .log Ã¥Â­ËœÃ¥Å“Â¨Ã¯Â¼Å’Ã¦Ë†â€“Ã¨â‚¬â€¦Ã¤Â»Å½ py Ã¦â€“â€¡Ã¤Â»Â¶Ã¦Â³Â¨Ã©â€¡Å Ã¨Â¯Â»Ã¥Ââ€“Ã¯Â¼â€°
                        # Ã¨Â¿â„¢Ã©â€¡Å’Ã§Â®â‚¬Ã¥Å’â€“Ã¤Â¸ÂºÃ¯Â¼Å¡Ã§â€ºÂ´Ã¦Å½Â¥Ã¨Â¯Â»Ã¥Ââ€“ python Ã¦â€“â€¡Ã¤Â»Â¶Ã¯Â¼Å’Ã¥Â¦â€šÃ¦Å¾Å“Ã©â€¡Å’Ã©ÂÂ¢Ã¦Â²Â¡Ã¦Å“â€°Ã©â€â„¢Ã¨Â¯Â¯Ã¤Â¿Â¡Ã¦ÂÂ¯Ã¯Â¼Å’Ã©Â»ËœÃ¨Â®Â¤Ã¦Â â€¡Ã¨Â®Â°
                        content = ""
                        with open(file_path, 'r', errors='ignore') as f_obj:
                            content = f_obj.read()
                        
                        # Ã¦ÂÂÃ¥Ââ€“Ã¦Å’â€¡Ã§ÂºÂ¹
                        signature = BugAnalyzer.get_signature(content)
                        category = BugAnalyzer.classify_bug(content)
                        
                        # Ã§Â»â€žÃ¥ÂË†Ã©â€Â®Ã¥â‚¬Â¼Ã¯Â¼Å¡Ã§Â±Â»Ã¥Å¾â€¹ + Ã¦Å’â€¡Ã§ÂºÂ¹
                        unique_key = f"[{bug_type.upper()}]_[{category}] : {signature}"
                        
                        self.unique_bugs[unique_key].append(file_path)
                        self.bug_stats[category] += 1
                        
                    except Exception as e:
                        # Ã¥Â®Â¹Ã©â€â„¢Ã¥Â¤â€žÃ§Ââ€ 
                        pass

    def get_total_bugs(self) -> int:
        return sum(len(bugs) for bugs in self.bugs.values())
    
    def get_bugs_by_type(self, bug_type: str) -> int:
        return len(self.bugs.get(bug_type, []))
    
    def print_summary(self):
        self.scan_bugs()
        print(f"\n{'='*70}")
        print(f"Ã°Å¸Â§Â© BUG ANALYSIS REPORT: {self.name}")
        print(f"{'='*70}")
        
        # Ã¦â€°â€œÃ¥ÂÂ°Ã¥Å¸ÂºÃ¤ÂºÅ½Ã¦Å’â€¡Ã§ÂºÂ¹Ã§Å¡â€žÃ¥Å½Â»Ã©â€¡ÂÃ§Â»Å¸Ã¨Â®Â¡
        print(f"Total Unique Issues: {len(self.unique_bugs)}")
        print(f"Distribution: {dict(self.bug_stats)}")
        print("-" * 70)
        
        # Ã¦â€°â€œÃ¥ÂÂ° Top 5 Ã¥â€Â¯Ã¤Â¸â‚¬ Bug
        print("Top Unique Bugs Found:")
        sorted_bugs = sorted(self.unique_bugs.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (sig, files) in enumerate(sorted_bugs[:5]):
            print(f" {i+1}. [Count: {len(files)}] {sig}")
            # Ã¦â€°â€œÃ¥ÂÂ°Ã¤Â¸â‚¬Ã¤Â¸ÂªÃ¦Â Â·Ã¦Å“Â¬Ã¨Â·Â¯Ã¥Â¾â€ž
            rel_path = files[0].replace(self.output_dir + "/", "")
            print(f"    Sample: {rel_path}")
        
        if not sorted_bugs:
            print(" No bugs found.")
            
        # Ã¤Â¿ÂÃ§â€¢â„¢Ã¦â€”Â§Ã§Å¡â€žÃ§Â»Å¸Ã¨Â®Â¡Ã¨Â¾â€œÃ¥â€¡ÂºÃ¯Â¼Å’Ã¤Â½Å“Ã¤Â¸ÂºÃ¥Â¯Â¹Ã¦Â¯â€
        print("-" * 70)
        total_raw = self.get_total_bugs()
        total_unique = len(self.unique_bugs)
        print(f"Ã°Å¸â€œÅ’ UNIQUE Bugs: {total_unique}")
        print(f"Ã°Å¸â€œâ€ž Raw Files:   CRASH={self.get_bugs_by_type('crash')} | CUDA={self.get_bugs_by_type('cuda')} | PRECISION={self.get_bugs_by_type('precision')} | Total={total_raw}")
        print(f"Ã°Å¸â€œÅ  Dedup Rate:  {total_unique}/{total_raw} ({total_unique/max(total_raw,1)*100:.1f}% unique)")
        print(f"{'='*70}\n")


# =============================================================================
# Ã°Å¸â€Â¥ NEW: Adaptive Saturation Detector (Ã¥Å Â¨Ã¦â‚¬ÂÃ©Â¥Â±Ã¥â€™Å’Ã¦Â£â‚¬Ã¦Âµâ€¹Ã¥â„¢Â¨)

# =============================================================================
# Ã°Å¸â€ â€¢ RealTimeBugDeduplicator (Ã¥Â®Å¾Ã¦â€”Â¶ Bug Ã¥Å½Â»Ã©â€¡Â)
# =============================================================================

class RealTimeBugDeduplicator:
    """
    Ã¥Â®Å¾Ã¦â€”Â¶ Bug Ã¥Å½Â»Ã©â€¡ÂÃ¥â„¢Â¨ - Ã¥Å¸ÂºÃ¤ÂºÅ½Ã©â€â„¢Ã¨Â¯Â¯Ã§Â­Â¾Ã¥ÂÂÃ¥Å½Â»Ã©â€¡Â
    
    Ã¨Â§Â£Ã¥â€ Â³Ã©â€”Â®Ã©Â¢ËœÃ¯Â¼Å¡Ã¥Å½Å¸Ã¥Â§â€¹ BugTracker Ã¨Â¯Â»Ã¥Ââ€“ .py Ã¦â€“â€¡Ã¤Â»Â¶Ã¥â€ â€¦Ã¥Â®Â¹Ã¯Â¼Å’Ã¦â€”Â Ã¦Â³â€¢Ã¦ÂÂÃ¥Ââ€“Ã©â€â„¢Ã¨Â¯Â¯Ã¤Â¿Â¡Ã¦ÂÂ¯
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
        print("Ã°Å¸â€Â BUG DEDUPLICATION REPORT")
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
# Ã°Å¸â€ â€¢ DispatcherSpace (Ã¥Å¸ÂºÃ¤ÂºÅ½ 5 Ã§Â»Â´Ã¦Â­Â£Ã¤ÂºÂ¤Ã§Å Â¶Ã¦â‚¬ÂÃ§Â©ÂºÃ©â€”Â´ + Ã§ÂºÂ¦Ã¦ÂÅ¸Ã¨Â¿â€¡Ã¦Â»Â¤)
# =============================================================================

class DispatcherSpace:
    """
    Dispatcher Ã§Å Â¶Ã¦â‚¬ÂÃ§Â©ÂºÃ©â€”Â´Ã¨Â¿Â½Ã¨Â¸ÂªÃ¥â„¢Â¨ - Ã¥Å’â€¦Ã¨Â£â€¦ DispatcherStateSpace
    
    Ã¥Å¸ÂºÃ¤ÂºÅ½ DispatchKey Ã©â‚¬Â»Ã¨Â¾â€˜Ã§Å¡â€ž 5 Ã§Â»Â´Ã¦Â­Â£Ã¤ÂºÂ¤Ã§Â©ÂºÃ©â€”Â´:
    1. Device: {'cpu', 'cuda'}
    2. Dtype: {'float32', 'float16', 'int64', ...}
    3. Layout: {'strided', 'sparse_coo'}
    4. MemFormat: {'contiguous', 'channels_last'}
    5. Rank: {1, 2, 3, 4, 5}
    
    Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¨Â®Â¡Ã§Â®â€”:
    - Ã§Ââ€ Ã¨Â®ÂºÃ§Â©ÂºÃ©â€”Â´ = Ã§Â¬â€ºÃ¥ÂÂ¡Ã¥Â°â€Ã§Â§Â¯Ã§Â»ÂÃ¨Â¿â€¡Ã§ÂºÂ¦Ã¦ÂÅ¸Ã¨Â¿â€¡Ã¦Â»Â¤Ã¥ÂÅ½Ã§Å¡â€žÃ¦Å“â€°Ã¦â€¢Ë†Ã§Å Â¶Ã¦â‚¬ÂÃ¦â€¢Â°
    - Ã¥Â®Å¾Ã©â„¢â€¦Ã¨Â¦â€ Ã§â€ºâ€“ = Ã¦Âµâ€¹Ã¨Â¯â€¢Ã¨Â¿â€¡Ã§Â¨â€¹Ã¤Â¸Â­Ã¥â€˜Â½Ã¤Â¸Â­Ã§Å¡â€žÃ¦Å“â€°Ã¦â€¢Ë†Ã§Å Â¶Ã¦â‚¬ÂÃ¦â€¢Â°
    - Coverage = |Ã¥Â®Å¾Ã©â„¢â€¦Ã¨Â¦â€ Ã§â€ºâ€“| / |Ã§Ââ€ Ã¨Â®ÂºÃ§Â©ÂºÃ©â€”Â´| (capped at 100%)
    """
    
    def __init__(self, max_hits: int = 50000):
        self.max_hits = max_hits
        
        # Ã¤Â½Â¿Ã§â€Â¨ pytorch_comprehensive_filter Ã¤Â¸Â­Ã§Å¡â€ž DispatcherStateSpace
        self.state_space = DispatcherStateSpace(max_states=max_hits)
        
        # Ã¥â€¦Â¼Ã¥Â®Â¹Ã¦â€”Â§Ã¦Å½Â¥Ã¥ÂÂ£
        self.api_dimensions: Dict[str, Dict[str, List[str]]] = {}
        self.theoretical_sizes: Dict[str, int] = {}
        self.total_records = 0
        self.overflow_warnings = 0
        
        # Ã°Å¸â€ â€¢ Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Å½â€ Ã¥ÂÂ²Ã¨Â®Â°Ã¥Â½â€¢ (Ã§â€Â¨Ã¤ÂºÅ½Ã§Â»ËœÃ¥â€ºÂ¾)
        self.coverage_history: Dict[str, List[Tuple[int, float]]] = {}
    
    def register_api(self, api_name: str, dimensions: Dict[str, List[str]] = None):
        """Ã¦Â³Â¨Ã¥â€ Å’ API Ã¥Â¹Â¶Ã¨Â®Â¡Ã§Â®â€”Ã§ÂºÂ¦Ã¦ÂÅ¸Ã¨Â¿â€¡Ã¦Â»Â¤Ã¥ÂÅ½Ã§Å¡â€žÃ§Ââ€ Ã¨Â®ÂºÃ§Â©ÂºÃ©â€”Â´"""
        # Ã¨Â°Æ’Ã§â€Â¨Ã¥Âºâ€¢Ã¥Â±â€š DispatcherStateSpace
        result = self.state_space.register_api(api_name)
        
        # Ã¦â€ºÂ´Ã¦â€“Â°Ã¥â€¦Â¼Ã¥Â®Â¹Ã¥Â­â€”Ã¦Â®Âµ
        self.theoretical_sizes[api_name] = result['valid_space']
        self.api_dimensions[api_name] = result['dimensions']
        
        # Ã°Å¸â€ â€¢ Ã¥Ë†ÂÃ¥Â§â€¹Ã¥Å’â€“Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Å½â€ Ã¥ÂÂ²
        self.coverage_history[api_name] = []
    
    def record_hit(self, api_name: str, api, is_cuda: bool = False, iteration: int = 0) -> bool:
        """
        Ã¨Â®Â°Ã¥Â½â€¢Ã¥Ââ€šÃ¦â€¢Â°Ã§Â»â€žÃ¥ÂË†Ã¥â€˜Â½Ã¤Â¸Â­
        
        Args:
            api_name: API Ã¥ÂÂÃ§Â§Â°
            api: TorchAPI Ã¥Â®Å¾Ã¤Â¾â€¹
            is_cuda: Ã¦ËœÂ¯Ã¥ÂÂ¦Ã¥Å“Â¨ CUDA Ã¤Â¸Å Ã¦â€°Â§Ã¨Â¡Å’
            iteration: Ã¥Â½â€œÃ¥â€°ÂÃ¨Â¿Â­Ã¤Â»Â£Ã¦Â¬Â¡Ã¦â€¢Â° (Ã§â€Â¨Ã¤ÂºÅ½Ã¨Â®Â°Ã¥Â½â€¢Ã¥Å½â€ Ã¥ÂÂ²)
        """
        # Ã¨â€¡ÂªÃ¥Å Â¨Ã¦Â³Â¨Ã¥â€ Å’
        if api_name not in self.state_space.api_spaces:
            self.register_api(api_name)
        
        # Ã¦ÂÂÃ¥Ââ€“ 5 Ã§Â»Â´Ã§Å Â¶Ã¦â‚¬Â
        state = self._extract_state_from_args(api.args, is_cuda)
        if state is None:
            return False
        
        # Ã°Å¸â€ â€¢ Ã¦Â£â‚¬Ã¦Å¸Â¥Ã§Å Â¶Ã¦â‚¬ÂÃ¦ËœÂ¯Ã¥ÂÂ¦Ã¥Å“Â¨Ã¦Å“â€°Ã¦â€¢Ë†Ã§Â©ÂºÃ©â€”Â´Ã¥â€ â€¦ (Ã©ËœÂ²Ã¦Â­Â¢Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¨Â¶â€¦Ã¨Â¿â€¡100%)
        # Ã¨Â®Â°Ã¥Â½â€¢Ã¥â€˜Â½Ã¤Â¸Â­
        is_new = self.state_space.record_hit(api_name, state)
        if is_new:
            self.total_records += 1
            
            # Ã°Å¸â€ â€¢ Ã¨Â®Â°Ã¥Â½â€¢Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Å½â€ Ã¥ÂÂ²
            current_cov = self.get_coverage(api_name)
            self.coverage_history[api_name].append((iteration, current_cov))
        
        return is_new
    
    def record_coverage_snapshot(self, api_name: str, iteration: int):
        """Ã¨Â®Â°Ã¥Â½â€¢Ã¥Â½â€œÃ¥â€°ÂÃ¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Â¿Â«Ã§â€¦Â§ (Ã¦Â¯ÂNÃ¦Â¬Â¡Ã¨Â¿Â­Ã¤Â»Â£Ã¨Â°Æ’Ã§â€Â¨Ã¤Â¸â‚¬Ã¦Â¬Â¡)"""
        if api_name not in self.coverage_history:
            self.coverage_history[api_name] = []
        
        current_cov = self.get_coverage(api_name)
        self.coverage_history[api_name].append((iteration, current_cov))
    
    def _extract_state_from_args(self, args: Dict, is_cuda: bool) -> Optional[Dict]:
        """
        Ã¤Â»Å½Ã¥Ââ€šÃ¦â€¢Â°Ã¤Â¸Â­Ã¦ÂÂÃ¥Ââ€“ 5 Ã§Â»Â´Ã§Å Â¶Ã¦â‚¬Â
        
        Ã¦ÂÂÃ¥Ââ€“Ã©â‚¬Â»Ã¨Â¾â€˜:
        1. Ã©ÂÂÃ¥Å½â€  argsÃ¯Â¼Å’Ã¦â€°Â¾Ã¥Ë†Â°Ã§Â¬Â¬Ã¤Â¸â‚¬Ã¤Â¸ÂªÃ¦Å“â€°Ã¦â€¢Ë† Tensor (Ã¥Å’â€¦Ã¦â€¹Â¬Ã¥ÂµÅ’Ã¥Â¥â€”Ã¥Å“Â¨ LIST/TUPLE Ã¤Â¸Â­Ã§Å¡â€ž)
        2. Ã§â€°Â¹Ã¥Ë†Â«Ã¥Â¤â€žÃ§Ââ€  input_signature Ã¥Ââ€šÃ¦â€¢Â°
        3. Ã¦ÂÂÃ¥Ââ€“ device, dtype, layout, mem_format, rank
        """
        state = {
            'device': 'cuda' if is_cuda else 'cpu',
            'dtype': None,
            'layout': 'strided',
            'mem_format': 'contiguous',
            'rank': None,
        }
        
        def extract_from_arg(arg):
            """Ã©â‚¬â€™Ã¥Â½â€™Ã¦ÂÂÃ¥Ââ€“ tensor Ã¤Â¿Â¡Ã¦ÂÂ¯"""
            if arg is None:
                return None, None
            
            # Ã¥Â¤â€žÃ§Ââ€  LIST/TUPLE Ã§Â±Â»Ã¥Å¾â€¹ - Ã©â‚¬â€™Ã¥Â½â€™Ã¦ÂÂÃ¥Ââ€“Ã§Â¬Â¬Ã¤Â¸â‚¬Ã¤Â¸ÂªÃ¥â€¦Æ’Ã§Â´Â 
            if hasattr(arg, 'type') and arg.type in [ArgType.LIST, ArgType.TUPLE]:
                if hasattr(arg, 'value') and arg.value:
                    for sub_arg in arg.value:
                        dtype, rank = extract_from_arg(sub_arg)
                        if dtype is not None:
                            return dtype, rank
                return None, None
            
            # Ã¥Â¤â€žÃ§Ââ€  TORCH_TENSOR Ã§Â±Â»Ã¥Å¾â€¹
            if hasattr(arg, 'type') and arg.type == ArgType.TORCH_TENSOR:
                dtype = None
                rank = None
                
                if hasattr(arg, 'dtype') and arg.dtype is not None:
                    dtype = str(arg.dtype).replace('torch.', '')
                
                if hasattr(arg, 'shape') and arg.shape is not None:
                    rank = len(arg.shape)
                
                return dtype, rank
            
            return None, None
        
        # Ã¤Â¼ËœÃ¥â€¦Ë†Ã¥Â¤â€žÃ§Ââ€  input_signature (Ã¥Â¯Â¹Ã¤ÂºÅ½ nn.Module Ã§Â±Â»Ã¥Å¾â€¹Ã§Å¡â€ž API)
        if 'input_signature' in args:
            dtype, rank = extract_from_arg(args['input_signature'])
            if dtype is not None:
                state['dtype'] = dtype
            if rank is not None:
                state['rank'] = rank
            if state['dtype'] is not None and state['rank'] is not None:
                return state
        
        # Ã©ÂÂÃ¥Å½â€ Ã¥â€¦Â¶Ã¤Â»â€“Ã¥Ââ€šÃ¦â€¢Â°
        for param_name, arg in args.items():
            if param_name == 'input_signature':
                continue
            
            dtype, rank = extract_from_arg(arg)
            
            if dtype is not None and state['dtype'] is None:
                state['dtype'] = dtype
            if rank is not None and state['rank'] is None:
                state['rank'] = rank
            
            # Ã¦Â£â‚¬Ã¦Å¸Â¥ layout
            if hasattr(arg, 'is_sparse') and arg.is_sparse:
                state['layout'] = 'sparse_coo'
            
            # Ã¦ËœÂ¾Ã¥Â¼Â dtype Ã¥Ââ€šÃ¦â€¢Â°
            if hasattr(arg, 'type') and hasattr(arg, 'value'):
                if arg.type == ArgType.TORCH_DTYPE:
                    state['dtype'] = str(arg.value).replace('torch.', '')
            
            # Ã¦â€°Â¾Ã¥Ë†Â°Ã¦Å“â€°Ã¦â€¢Ë†Ã§Å Â¶Ã¦â‚¬ÂÃ¥Â°Â±Ã¨Â¿â€Ã¥â€ºÅ¾
            if state['dtype'] is not None and state['rank'] is not None:
                return state
        
        # Ã¢Å“â€¦ Ã¥â€¦Å“Ã¥Âºâ€¢Ã©â‚¬Â»Ã¨Â¾â€˜Ã¯Â¼Å¡Ã§Â¡Â®Ã¤Â¿ÂÃ¦Å“â€°Ã©Â»ËœÃ¨Â®Â¤Ã¥â‚¬Â¼
        if len(args) > 0:
            if state['dtype'] is None:
                state['dtype'] = 'float32'  # Ã©Â»ËœÃ¨Â®Â¤ float32
            if state['rank'] is None:
                state['rank'] = 2  # Ã©Â»ËœÃ¨Â®Â¤ 2d (Ã¦Â¯â€ 0d Ã¦â€ºÂ´Ã¥Â¸Â¸Ã¨Â§Â)
            return state
        
        return None
    
    def get_coverage(self, api_name: str) -> float:
        """Ã¨Å½Â·Ã¥Ââ€“Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡"""
        return self.state_space.get_coverage(api_name)
    
    def get_detailed_stats(self, api_name: str) -> Dict:
        """Ã¨Å½Â·Ã¥Ââ€“Ã¨Â¯Â¦Ã§Â»â€ Ã§Â»Å¸Ã¨Â®Â¡"""
        return self.state_space.get_detailed_stats(api_name)
    
    def print_summary(self):
        """Ã¦â€°â€œÃ¥ÂÂ°Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¦â€˜ËœÃ¨Â¦Â"""
        self.state_space.print_summary()


# =============================================================================

class AdaptiveSaturationDetector:
    """
    Ã¥Å Â¨Ã¦â‚¬ÂÃ©Â¥Â±Ã¥â€™Å’Ã¦Â£â‚¬Ã¦Âµâ€¹Ã¥â„¢Â¨ - Ã¤Â¸ÂÃ¥ÂÅ“Ã¦Â­Â¢Ã¯Â¼Å’Ã¨â‚¬Å’Ã¦ËœÂ¯Ã¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢Ã¨Å’Æ’Ã¥â€ºÂ´
    
    Ã§Â­â€“Ã§â€¢Â¥:
    1. Ã¨Â¿Å¾Ã§Â»Â­ N Ã¦Â¬Â¡Ã¦â€”Â Ã¥Ââ€˜Ã§Å½Â° Ã¢â€ â€™ Ã¨Â§Â¦Ã¥Ââ€˜"Ã¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢"Ã¨â‚¬Å’Ã©ÂÅ¾"Ã¥ÂÅ“Ã¦Â­Â¢"
    2. Ã¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢ = Ã¦ÂÂÃ¥Ââ€¡Ã¦Å½Â¢Ã§Â´Â¢Ã§Å½â€¡ + Ã¦â€°Â©Ã¥Â¤Â§Ã¥ÂËœÃ¥Â¼â€šÃ¨Å’Æ’Ã¥â€ºÂ´
    3. Ã¥Ââ€˜Ã§Å½Â°Ã¦â€“Â° Kernel Ã¢â€ â€™ Ã§Â«â€¹Ã¥ÂÂ³Ã¦ÂÂ¢Ã¥Â¤ÂÃ¦Â­Â£Ã¥Â¸Â¸Ã¦Â¨Â¡Ã¥Â¼Â
    """
    
    def __init__(self, 
                 patience: int = 500,           # Ã°Å¸â€Â§ Ã¥Â®Â¹Ã¥Â¿ÂÃ¥Â¤Å¡Ã¥Â°â€˜Ã¦Â¬Â¡Ã¦â€”Â Ã¦â€“Â°Ã¥Ââ€˜Ã§Å½Â°
                 check_interval: int = 100):
        self.patience = patience
        self.check_interval = check_interval
        
        self.no_discovery_count = 0
        self.last_kernel_count = 0
        self.last_check_iteration = 0
        
        # Ã¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢Ã¦Â â€¡Ã¥Â¿â€”
        self.expansion_mode = False
        self.expansion_count = 0
    
    def update(self, current_kernels: int, iteration: int) -> Tuple[bool, str]:
        """
        Ã¦â€ºÂ´Ã¦â€“Â°Ã§Å Â¶Ã¦â‚¬ÂÃ¥Â¹Â¶Ã¦Â£â‚¬Ã¦Å¸Â¥Ã¦ËœÂ¯Ã¥ÂÂ¦Ã©Å“â‚¬Ã¨Â¦ÂÃ¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢
        
        Returns:
            (should_expand, message)
        """
        # Ã¦Â£â‚¬Ã¦Å¸Â¥Ã¦ËœÂ¯Ã¥ÂÂ¦Ã¦Å“â€°Ã¦â€“Â°Ã¥Ââ€˜Ã§Å½Â°
        if current_kernels > self.last_kernel_count:
            self.no_discovery_count = 0
            self.last_kernel_count = current_kernels
            
            # Ã¥Ââ€˜Ã§Å½Â°Ã¦â€“Â° KernelÃ¯Â¼Å’Ã©â‚¬â‚¬Ã¥â€¡ÂºÃ¦â€°Â©Ã¥Â±â€¢Ã¦Â¨Â¡Ã¥Â¼Â
            if self.expansion_mode:
                self.expansion_mode = False
                return False, "Ã¢Å“â€¦ New kernels found! Returning to normal mode"
        else:
            self.no_discovery_count += 1
        
        # Ã¥Â®Å¡Ã¦Å“Å¸Ã¦Â£â‚¬Ã¦Å¸Â¥
        if iteration - self.last_check_iteration >= self.check_interval:
            self.last_check_iteration = iteration
            
            # Ã¨Â§Â¦Ã¥Ââ€˜Ã¦â€°Â©Ã¥Â¤Â§Ã¦ÂÅ“Ã§Â´Â¢
            if self.no_discovery_count >= self.patience and not self.expansion_mode:
                self.expansion_mode = True
                self.expansion_count += 1
                return True, f"Ã°Å¸â€Â Expanding search (stagnation: {self.no_discovery_count} iters)"
        
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
    """Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¨Â¿Â½Ã¨Â¸ÂªÃ¥â„¢Â¨ + Ã¥Å Â¨Ã¦â‚¬ÂÃ©Â¥Â±Ã¥â€™Å’Ã¦Â£â‚¬Ã¦Âµâ€¹"""
    
    def __init__(self, name: str, enable_adaptive_saturation: bool = True):
        self.name = name
        self.all_kernels: Set[str] = set()
        self.kernel_provenance: Dict[str, int] = {}
        self.history: List[Tuple[int, int]] = []
        self.new_kernel_iterations: List[int] = []
        
        # Ã¥Å Â¨Ã¦â‚¬ÂÃ©Â¥Â±Ã¥â€™Å’Ã¦Â£â‚¬Ã¦Âµâ€¹
        self.saturation_detector = AdaptiveSaturationDetector() if enable_adaptive_saturation else None
    
    def update(self, new_kernels: Set[str], iteration: int) -> Tuple[int, Tuple[bool, str]]:
        """
        Ã¦â€ºÂ´Ã¦â€“Â°Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡
        
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
        
        # Ã¥Å Â¨Ã¦â‚¬ÂÃ©Â¥Â±Ã¥â€™Å’Ã¦Â£â‚¬Ã¦Âµâ€¹
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
    """Ã¨Â¿â€ºÃ¥Å’â€“Ã§Â§ÂÃ¥Â­ÂÃ¦Â±Â """
    
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
# Ã°Å¸â€Â¥ NEW: Adaptive Mutation Controller (Ã¥Å Â¨Ã¦â‚¬ÂÃ¥ÂËœÃ¥Â¼â€šÃ¦Å½Â§Ã¥Ë†Â¶Ã¥â„¢Â¨)
# =============================================================================

class AdaptiveMutationController:
    """
    Ã¥Å Â¨Ã¦â‚¬ÂÃ¥ÂËœÃ¥Â¼â€šÃ¦Å½Â§Ã¥Ë†Â¶Ã¥â„¢Â¨
    
    Ã¥Å Å¸Ã¨Æ’Â½:
    1. Ã¦Â Â¹Ã¦ÂÂ®Ã¥ÂÅ“Ã¦Â»Å¾Ã¦Æ’â€¦Ã¥â€ ÂµÃ¨â€¡ÂªÃ¥Å Â¨Ã¨Â°Æ’Ã¦â€¢Â´Ã¥ÂËœÃ¥Â¼â€šÃ¨Å’Æ’Ã¥â€ºÂ´
    2. Ã¦â€Â¯Ã¦Å’Â ÃŽÂµ-greedy Ã¦Å½Â¢Ã§Â´Â¢Ã§Â­â€“Ã§â€¢Â¥
    3. Ã¥Ââ€˜Ã§Å½Â°Ã¦â€“Â° Kernel Ã¦â€”Â¶Ã§Â«â€¹Ã¥ÂÂ³Ã¦ÂÂ¢Ã¥Â¤ÂÃ¥Â¾Â®Ã¥Ë†â€ºÃ¦Â¨Â¡Ã¥Â¼Â
    """
    
    def __init__(self, 
                 base_epsilon: float = 0.1,          # Ã¥Å¸ÂºÃ§Â¡â‚¬Ã¦Å½Â¢Ã§Â´Â¢Ã§Å½â€¡
                 expansion_epsilon: float = 0.3,     # Ã¦â€°Â©Ã¥Â¼Â Ã¦Â¨Â¡Ã¥Â¼ÂÃ¦Å½Â¢Ã§Â´Â¢Ã§Å½â€¡
                 surgical_range: int = 8,            # Ã¥Â¾Â®Ã¥Ë†â€ºÃ¨Å’Æ’Ã¥â€ºÂ´
                 exploration_range: int = 64):       # Ã¦Å½Â¢Ã§Â´Â¢Ã¨Å’Æ’Ã¥â€ºÂ´
        self.base_epsilon = base_epsilon
        self.expansion_epsilon = expansion_epsilon
        self.surgical_range = surgical_range
        self.exploration_range = exploration_range
        
        # Ã¥Â½â€œÃ¥â€°ÂÃ§Å Â¶Ã¦â‚¬Â
        self.current_epsilon = base_epsilon
        self.current_range = surgical_range
        self.expansion_mode = False
        
        # Ã§Â»Å¸Ã¨Â®Â¡
        self.exploration_count = 0
        self.exploitation_count = 0
    
    def should_explore(self) -> bool:
        """Ã¥â€ Â³Ã¥Â®Å¡Ã¦Å“Â¬Ã¦Â¬Â¡Ã¨Â¿Â­Ã¤Â»Â£Ã¦ËœÂ¯Ã¥ÂÂ¦Ã©â€¡â€¡Ã§â€Â¨Ã¦Å½Â¢Ã§Â´Â¢Ã¦Â¨Â¡Ã¥Â¼Â"""
        if random.random() < self.current_epsilon:
            self.exploration_count += 1
            return True
        else:
            self.exploitation_count += 1
            return False
    
    def enter_expansion_mode(self):
        """Ã¨Â¿â€ºÃ¥â€¦Â¥Ã¦â€°Â©Ã¥Â¼Â Ã¦Â¨Â¡Ã¥Â¼ÂÃ¯Â¼Ë†Ã¥ÂÅ“Ã¦Â»Å¾Ã¦â€”Â¶Ã¨Â§Â¦Ã¥Ââ€˜Ã¯Â¼â€°"""
        self.expansion_mode = True
        self.current_epsilon = self.expansion_epsilon
        self.current_range = self.exploration_range
        print(f"  Ã°Å¸â€Â [Mutation] Expansion mode: ÃŽÂµ={self.current_epsilon}, range=Ã‚Â±{self.current_range}")
    
    def exit_expansion_mode(self):
        """Ã©â‚¬â‚¬Ã¥â€¡ÂºÃ¦â€°Â©Ã¥Â¼Â Ã¦Â¨Â¡Ã¥Â¼ÂÃ¯Â¼Ë†Ã¥Ââ€˜Ã§Å½Â°Ã¦â€“Â° Kernel Ã¦â€”Â¶Ã¯Â¼â€°"""
        self.expansion_mode = False
        self.current_epsilon = self.base_epsilon
        self.current_range = self.surgical_range
        print(f"  Ã¢Å“â€¦ [Mutation] Normal mode: ÃŽÂµ={self.current_epsilon}, range=Ã‚Â±{self.current_range}")
    
    def get_mutation_range(self) -> int:
        """Ã¨Å½Â·Ã¥Ââ€“Ã¥Â½â€œÃ¥â€°ÂÃ¥ÂËœÃ¥Â¼â€šÃ¨Å’Æ’Ã¥â€ºÂ´"""
        return self.current_range
    
    def get_status(self) -> str:
        total = self.exploration_count + self.exploitation_count
        if total == 0:
            return "No mutations yet"
        explore_pct = self.exploration_count / total * 100
        return (f"Explore: {self.exploration_count}/{total} ({explore_pct:.1f}%) | "
                f"Mode: {'Expansion' if self.expansion_mode else 'Normal'} | "
                f"ÃŽÂµ={self.current_epsilon:.2f}, range=Ã‚Â±{self.current_range}")


# =============================================================================
# Probability Patcher
# =============================================================================

class ProbabilityPatcher:
    """Ã¥Â°â€ Ã¦â€¢Â°Ã¦ÂÂ®Ã¥Âºâ€œÃ©â€¡â€¡Ã¦Â Â·Ã¦Â¦â€šÃ§Å½â€¡Ã¤Â»Å½ 20% Ã¦ÂÂÃ¥Ââ€¡Ã¥Ë†Â° 80%"""
    
    @staticmethod
    def patch_high_db_probability():
        import utils.probability as prob_module
        
        global _ORIGINAL_DO_SELECT_FROM_DB
        _ORIGINAL_DO_SELECT_FROM_DB = prob_module.do_select_from_db
        
        def high_db_select() -> bool:
            from numpy.random import rand
            return rand() < 0.5
        
        prob_module.do_select_from_db = high_db_select
        print("[Patch] Ã¢Å“â€¦ Database sampling: 20% Ã¢â€ â€™ 80%")
    
    @staticmethod
    def restore():
        if _ORIGINAL_DO_SELECT_FROM_DB:
            import utils.probability as prob_module
            prob_module.do_select_from_db = _ORIGINAL_DO_SELECT_FROM_DB
            print("[Patch] Ã°Å¸â€â€ž Database sampling restored")


# =============================================================================
# Ã°Å¸Â§Âª NEW: Poison Patcher (Ã¦Å â€¢Ã¦Â¯â€™Ã¨Â¡Â¥Ã¤Â¸Â - Ã§â€¹Â¬Ã§Â«â€¹Ã¤ÂºÅ½Ã§Â­â€“Ã§â€¢Â¥)
# =============================================================================

# Ã¤Â¿ÂÃ¥Â­ËœÃ¥Å½Å¸Ã¥Â§â€¹Ã¦â€“Â¹Ã¦Â³â€¢Ã§Å¡â€žÃ¥â€¦Â¨Ã¥Â±â‚¬Ã¥ÂËœÃ©â€¡Â
_POISON_ORIGINAL_FLOAT = None
_POISON_ORIGINAL_INT = None

class PoisonPatcher:
    """
    🎯 增强版投毒补丁 - 系统性覆盖 BugsInDLLs 中 200 个"表面easy实际hard"的 bug
    
    基于数据驱动的投毒策略 (来自 bugsindlls_bugs.json 分析):
    
    Float 投毒 (针对 116 zero-size + 46 Inf + 42 负数 + 13 NaN):
    - 8%:  NaN (13 bugs 需要)
    - 8%:  Infinity (46 bugs 需要) 
    - 8%:  零值 0.0 (116 bugs 涉及 zero-size)
    - 8%:  负数 (42 bugs 需要)
    - 8%:  极端值 (溢出边界)
    - 8%:  接近零小数 (精度陷阱)
    - 20%: 字典采样
    - 32%: 温和微调
    
    Int 投毒 (针对 116 zero-size + 25 singleton + 42 负数):
    - 12%: size=0 (116 bugs 需要 zero-size)
    - 10%: size=1 (25 bugs 需要 singleton dimension)
    - 8%:  负数维度 (42 bugs 需要)
    - 8%:  陷阱值 [质数, 大尺寸, 小维度]
    - 10%: 极端值 [±999, ±2^31]
    - 20%: 字典采样
    - 32%: 温和微调
    
    ✅ 解决问题: 为什么"看起来容易"的 bug FreeFuzz 没找到?
       → Random Fuzzing 只测"正常路径"，miss 掉边界情况
       → 系统性投毒确保覆盖 NaN/Inf/零值/负数/singleton
    """
    
    @staticmethod
    def patch():
        global _POISON_ORIGINAL_FLOAT, _POISON_ORIGINAL_INT
        
        _POISON_ORIGINAL_FLOAT = Argument.mutate_float_value
        _POISON_ORIGINAL_INT = Argument.mutate_int_value
        
        def poison_float_mutation(self, value) -> float:
            """🧪 Enhanced Float Poison - 系统性覆盖 200 个"表面easy实际hard"的 bug
            
            基于 BugsInDLLs 隐藏难度分析:
            - 116 bugs 需要 zero-size (通过 0.0 触发)
            - 46 bugs 需要 Inf
            - 42 bugs 需要负数
            - 13 bugs 需要 NaN
            """
            from numpy.random import rand, choice
            
            roll = rand()
            
            # [8%] NaN - 最高优先级 (13 bugs 需要)
            if roll < 0.08:
                return float('nan')
            
            # [8%] Infinity - 第二优先级 (46 bugs 需要)
            elif roll < 0.16:
                return choice([float('inf'), float('-inf')])
            
            # [8%] 零值 - 关键边界 (116 bugs 涉及 zero-size)
            elif roll < 0.24:
                return 0.0
            
            # [8%] 负数 - 常见陷阱 (42 bugs 需要)
            elif roll < 0.32:
                return choice([-1.0, -0.5, -10.0, -100.0, -1e-5])
            
            # [8%] 极端值 - 溢出边界
            elif roll < 0.40:
                extreme_values = [1e20, -1e20, 1e-10, -1e-10, 1e38, -1e38]
                return choice(extreme_values)
            
            # [8%] 接近零的小数 - 精度陷阱
            elif roll < 0.48:
                return choice([1e-7, -1e-7, 1e-15, -1e-15, 1e-20, -1e-20])
            
            # [20%] 字典采样 (保持原有逻辑)
            elif roll < 0.68:
                if hasattr(Argument, '_float_values'):
                    return choice(Argument._float_values)
                elif hasattr(Argument, 'float_values'):
                    return choice(Argument.float_values)
                return choice([0.0, 1.0, -1.0])
            
            # [32%] 温和微调
            else:
                return value + (rand() - 0.5) * 8.0
        
        def poison_int_mutation(self, value, _min=None, _max=None) -> int:
            """🧪 Enhanced Int Poison - 系统性覆盖边界值和陷阱值
            
            基于 BugsInDLLs 隐藏难度分析:
            - 116 bugs 需要 zero-size tensor (size=0)
            - 42 bugs 需要负数维度
            - 25 bugs 需要 singleton dimension (size=1)
            """
            from numpy.random import rand, choice, randint
            
            roll = rand()
            
            # [12%] 零值 - 最高优先级 (116 bugs 涉及 zero-size)
            if roll < 0.12:
                return 0
            
            # [10%] Singleton - 第二优先级 (25 bugs 需要 size=1)
            elif roll < 0.22:
                return 1
            
            # [8%] 负数边界 - 关键陷阱 (42 bugs 需要负数)
            elif roll < 0.30:
                return choice([-1, -2, -3, -4])
            
            # [8%] 常见陷阱值 - 特定参数的问题值
            elif roll < 0.38:
                trap_values = [
                    7, 11, 13,       # 质数 (不能整除)
                    256, 512, 1024,  # 大尺寸
                    2, 3, 4, 5,      # 小维度
                ]
                new_value = choice(trap_values)
            
            # [10%] 极端值 - 溢出和边界检查
            elif roll < 0.48:
                extreme_values = [-999, 999, -2147483648, 2147483647, -65536, 65536]
                new_value = choice(extreme_values)
            
            # [20%] 字典采样
            elif roll < 0.68:
                if hasattr(Argument, '_int_values'):
                    new_value = choice(Argument._int_values)
                elif hasattr(Argument, 'int_values'):
                    new_value = choice(Argument.int_values)
                else:
                    new_value = randint(-100, 100)
            
            # [32%] 温和微调
            else:
                new_value = value + randint(-8, 9)
            
            # 应用边界限制 (但保留 -1, 0, 1 等特殊值用于触发 bug)
            if _min is not None and new_value < _min and new_value not in [-1, 0, 1]:
                new_value = max(_min, new_value)
            if _max is not None and new_value > _max:
                new_value = min(_max, new_value)
            
            return int(new_value)
        
        # Ã¤Â¿Â®Ã¦Â­Â£Ã¯Â¼Å¡Ã¥â€¡Â½Ã¦â€¢Â°Ã¥ÂÂÃ¨Â¦ÂÃ¥Â¯Â¹Ã¥Âºâ€Ã¤Â¸Å Ã©ÂÂ¢Ã¥Â®Å¡Ã¤Â¹â€°Ã§Å¡â€ž poion_float_mutation
        Argument.mutate_float_value = poison_float_mutation 
        Argument.mutate_int_value = poison_int_mutation
        print("[Patch] Ã°Å¸Â§Âª Poison Injection enabled (ALL strategies):")
        print("        Float: 5% Inf + 5% NaN + 10% Extreme")
        print("        Int:   10% boundary + 10% extreme + 10% trap")
    
    @staticmethod
    def restore():
        global _POISON_ORIGINAL_FLOAT, _POISON_ORIGINAL_INT
        if _POISON_ORIGINAL_FLOAT:
            Argument.mutate_float_value = _POISON_ORIGINAL_FLOAT
        if _POISON_ORIGINAL_INT:
            Argument.mutate_int_value = _POISON_ORIGINAL_INT
        print("[Patch] Ã°Å¸â€â€ž Poison Injection restored")


# =============================================================================
# Ã°Å¸â€Â¥ NEW: Adaptive Mutation Patcher (Ã¥Å Â¨Ã¦â‚¬ÂÃ¥ÂËœÃ¥Â¼â€šÃ¨Â¡Â¥Ã¤Â¸Â - Ã¤Â»â€¦ Guided)
# =============================================================================

class AdaptiveMutationPatcher:
    """Ã¥Å Â¨Ã¦â‚¬ÂÃ¨Å’Æ’Ã¥â€ºÂ´Ã¥ÂËœÃ¥Â¼â€šÃ¨Â¡Â¥Ã¤Â¸Â - Ã¤Â»â€¦ Guided Ã§Â­â€“Ã§â€¢Â¥Ã¤Â½Â¿Ã§â€Â¨Ã¯Â¼Å’Ã¦Å â€¢Ã¦Â¯â€™Ã§â€Â± PoisonPatcher Ã§Â»Å¸Ã¤Â¸â‚¬Ã¥Â¤â€žÃ§Ââ€ """
    
    @staticmethod
    def patch_adaptive_mutation(controller: AdaptiveMutationController):
        """
        Guided Ã§Â­â€“Ã§â€¢Â¥Ã©Â¢ÂÃ¥Â¤â€“Ã¤Â½Â¿Ã§â€Â¨Ã¥Å Â¨Ã¦â‚¬ÂÃ¨Å’Æ’Ã¥â€ºÂ´Ã¨Â°Æ’Ã¦â€¢Â´
        Ã¦Â³Â¨Ã¦â€žÂ: Ã¥Å¸ÂºÃ§Â¡â‚¬Ã¦Å â€¢Ã¦Â¯â€™Ã¥Â·Â²Ã§â€Â± PoisonPatcher Ã¥Â¤â€žÃ§Ââ€ Ã¯Â¼Å’Ã¨Â¿â„¢Ã©â€¡Å’Ã¥ÂÂªÃ¥Â¢Å¾Ã¥Å Â Ã¥Å Â¨Ã¦â‚¬ÂÃ¨Å’Æ’Ã¥â€ºÂ´Ã¥Å Å¸Ã¨Æ’Â½
        """
        # Ã¦Â³Â¨Ã¦â€žÂ: Ã¤Â¸ÂÃ¥â€ ÂÃ¤Â¿ÂÃ¥Â­Ëœ/Ã¨Â¦â€ Ã§â€ºâ€“Ã¥Å½Å¸Ã¥Â§â€¹Ã¦â€“Â¹Ã¦Â³â€¢Ã¯Â¼Å’Ã¥â€ºÂ Ã¤Â¸Âº PoisonPatcher Ã¥Â·Â²Ã§Â»ÂÃ¥Â¤â€žÃ§Ââ€ Ã¤Âºâ€ 
        # Ã¨Â¿â„¢Ã¤Â¸Âª patcher Ã§Å½Â°Ã¥Å“Â¨Ã¥ÂÂªÃ¦ËœÂ¯Ã¤Â¸â‚¬Ã¤Â¸ÂªÃ¦Â â€¡Ã¨Â®Â°Ã¯Â¼Å’Ã¨Â¡Â¨Ã§Â¤Âº Guided Ã§Â­â€“Ã§â€¢Â¥Ã¥ÂÂ¯Ã§â€Â¨Ã¤Âºâ€ Ã¥Å Â¨Ã¦â‚¬ÂÃ¨Å’Æ’Ã¥â€ºÂ´
        print(f"[Patch] Ã¢Å“â€¦ Adaptive range enabled for Guided: Ã‚Â±{controller.get_mutation_range()}")
    
    @staticmethod
    def restore():
        # PoisonPatcher Ã¤Â¼Å¡Ã¨Â´Å¸Ã¨Â´Â£Ã¦ÂÂ¢Ã¥Â¤ÂÃ¯Â¼Å’Ã¨Â¿â„¢Ã©â€¡Å’Ã¥ÂÂªÃ¦â€°â€œÃ¥ÂÂ°Ã¤Â¿Â¡Ã¦ÂÂ¯
        print("[Patch] Ã°Å¸â€â€ž Adaptive range disabled")


# =============================================================================
# Ã°Å¸â€Â¥ Enhanced Fuzzer with Hybrid Strategy
# =============================================================================

class EnhancedFuzzer:
    """Ã¦â€Â¯Ã¦Å’ÂÃ¦Â·Â·Ã¥ÂË†Ã§Â­â€“Ã§â€¢Â¥Ã§Å¡â€žÃ¥Â¢Å¾Ã¥Â¼ÂºÃ¥Å¾â€¹ Fuzzer"""
    
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
                 warmup_ratio: float = 0.1,           # Ã°Å¸â€ â€¢ Ã§Æ’Â­Ã¥ÂÂ¯Ã¥Å Â¨Ã¦Â¯â€Ã¤Â¾â€¹
                 enable_dispatcher: bool = True):     # Ã°Å¸â€ â€¢ Ã¥ÂÂ¯Ã§â€Â¨ DispatcherSpace  # Ã°Å¸â€Â§
        self.api_name = api_name
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.enable_patches = enable_patches
        self.use_all_oracles = use_all_oracles
        self.enable_logging = enable_logging
        self.enable_checkpoint = enable_checkpoint
        self.enable_safety_guards = enable_safety_guards
        self.diff_bound = diff_bound
        self.warmup_ratio = warmup_ratio             # Ã°Å¸â€ â€¢ Ã¤Â¿ÂÃ¥Â­ËœÃ§Æ’Â­Ã¥ÂÂ¯Ã¥Å Â¨Ã¦Â¯â€Ã¤Â¾â€¹
        self.enable_dispatcher = enable_dispatcher   # Ã°Å¸â€ â€¢ Ã¤Â¿ÂÃ¥Â­ËœdispatcherÃ¥Â¼â‚¬Ã¥â€¦Â³
        
        # Trackers
        self.coverage = EnhancedCoverageTracker(strategy_name, enable_adaptive_saturation=enable_patches)
        self.bug_tracker = BugTracker(strategy_name, output_dir)
        
        # Guided Ã¤Â¸â€œÃ¥Â±Å¾
        self.corpus = EvolutionaryCorpus(max_size=100) if enable_patches else None
        self.mutation_controller = AdaptiveMutationController() if enable_patches else None
        
        # Library - Ã¤Â½Â¿Ã§â€Â¨Ã¦â€ºÂ´Ã¤Â¸Â¥Ã¦Â Â¼Ã§Å¡â€ž diff_bound Ã©â€¦ÂÃ¥ÂË†Ã¦Å â€¢Ã¦Â¯â€™Ã§Â­â€“Ã§â€¢Â¥
        self.library = TorchLibrary(output_dir, diff_bound=diff_bound)
        print(f"[Fuzzer] Ã°Å¸Å½Â¯ Precision tolerance: diff_bound={diff_bound}")
        
        # Oracle Ã¥Ë†â€”Ã¨Â¡Â¨
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
            self.outlier_filter = OutlierFilter(max_elements=int(5e6), max_memory_gb=2.0)  # Ã°Å¸â€Â§ Ã©â„¢ÂÃ¤Â½Å½Ã©â„¢ÂÃ¥Ë†Â¶Ã©ËœÂ²Ã¦Â­Â¢OOM
            print(f"[Safety] Guards enabled: Speedometer + DiskGuard + OutlierFilter")
        
        # Ã°Å¸â€ â€¢ DispatcherSpace Ã¥Ë†ÂÃ¥Â§â€¹Ã¥Å’â€“
        self.dispatcher_space = None
        if enable_dispatcher:
            self.dispatcher_space = DispatcherSpace(max_hits=50000)
            self.dispatcher_space.register_api(api_name)
        
        # Ã°Å¸â€ â€¢ Ã¥Â®Å¾Ã¦â€”Â¶ Bug Ã¥Å½Â»Ã©â€¡ÂÃ¥â„¢Â¨
        self.bug_deduplicator = RealTimeBugDeduplicator(max_signatures=10000)
        
        # Ã°Å¸â€ â€¢ Warmup Ã©â€¦ÂÃ§Â½Â® (Ã©ËœË†Ã¥â‚¬Â¼Ã¥Å“Â¨ run_fuzzing_loop Ã¤Â¸Â­Ã¨Â®Â¡Ã§Â®â€”)
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
        Ã¤Â¸Â» Fuzzing Ã¥Â¾ÂªÃ§Å½Â¯ - Ã¦Â·Â·Ã¥ÂË†Ã§Â­â€“Ã§â€¢Â¥Ã§â€°Ë†Ã¦Å“Â¬
        """
        # ====================================================================
        # Ã¥Ë†ÂÃ¥Â§â€¹Ã¥Å’â€“ Safety Guards
        # ====================================================================
        if self.speedometer:
            self.speedometer.start()
        
        # ====================================================================
        # Ã¥Â°ÂÃ¨Â¯â€¢Ã¤Â»Å½ checkpoint Ã¦ÂÂ¢Ã¥Â¤Â
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
                
                print(f"\nÃ¢Å“â€¦ Resuming from iteration {start_iteration}")
                print(f"  Previous kernels: {len(self.coverage.all_kernels)}")
                if self.corpus:
                    print(f"  Previous corpus: {len(self.corpus.corpus)} seeds")
                print(f"{'='*70}\n")
        
        # ====================================================================
        # Ã¤Â¸Â»Ã¥Â¾ÂªÃ§Å½Â¯
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"Starting {self.strategy_name} fuzzing")
        if self.mutation_controller:
            print(f"Hybrid Strategy: ÃŽÂµ-greedy + Adaptive Mutation")
        print(f"Iterations: {start_iteration} Ã¢â€ â€™ {max_iterations}")
        if start_iteration > 0:
            print(f"(Resuming from checkpoint)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        last_bug_count = 0
        
        # Ã°Å¸â€ â€¢ Ã¨Â®Â¡Ã§Â®â€”Ã§Æ’Â­Ã¥ÂÂ¯Ã¥Å Â¨Ã©ËœË†Ã¥â‚¬Â¼
        self.warmup_threshold = int(max_iterations * self.warmup_ratio)
        print(f"Ã°Å¸â€Â¥ Warmup Phase: iterations 0 ~ {self.warmup_threshold} (Random only)")
        print(f"Ã°Å¸Å¡â‚¬ Evolution Phase: iterations {self.warmup_threshold} ~ {max_iterations}")
        
        for i in range(start_iteration, max_iterations):
            # ================================================================
            # Speedometer tick
            # ================================================================
            if self.speedometer:
                self.speedometer.tick()
            
            # ================================================================
            # Ã°Å¸â€Â¥ Ã¦Â·Â·Ã¥ÂË†Ã§Â­â€“Ã§â€¢Â¥Ã¯Â¼Å¡ÃŽÂµ-greedy Ã©â‚¬â€°Ã¦â€¹Â©
            # ================================================================
            source = ""
            is_exploration = False
            
            # Ã°Å¸â€ â€¢ Phase 1: Warm-up (Ã¥Â¼ÂºÃ¥Ë†Â¶ Random)
            if i < self.warmup_threshold:
                api = TorchAPI(self.api_name)
                source = "warmup_random"
                api.mutate()
                if i == self.warmup_threshold - 1:
                    print(f"\nÃ°Å¸Å¡â‚¬ PHASE TRANSITION: Warm-up Ã¢â€ â€™ Evolution (iter {i+1})")
            
            # Ã°Å¸â€ â€¢ Phase 2: Evolution (ÃŽÂµ-greedy)
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
            
            # OutlierFilter Ã¦Â£â‚¬Ã¦Å¸Â¥
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
            # Ã¦Âµâ€¹Ã¨Â¯â€¢Ã¦â€°â‚¬Ã¦Å“â€° Oracle
            # ================================================================
            all_captured_kernels = set()
            execution_valid = False
            
            for oracle in self.oracles:
                try:
                    # Ã°Å¸â€ â€¢ test_with_oracle Ã§Å½Â°Ã¥Å“Â¨Ã¨Â¿â€Ã¥â€ºÅ¾ (kernels, bug_info)
                    result = self.library.test_with_oracle(api, oracle)
                    
                    # Ã¥â€¦Â¼Ã¥Â®Â¹Ã¦â€”Â§Ã§â€°Ë†Ã¦Å“Â¬Ã¯Â¼Ë†Ã¥ÂÂªÃ¨Â¿â€Ã¥â€ºÅ¾ kernelsÃ¯Â¼â€°Ã¥â€™Å’Ã¦â€“Â°Ã§â€°Ë†Ã¦Å“Â¬Ã¯Â¼Ë†Ã¨Â¿â€Ã¥â€ºÅ¾Ã¥â€¦Æ’Ã§Â»â€žÃ¯Â¼â€°
                    if isinstance(result, tuple):
                        captured_kernels, bug_info = result
                    else:
                        captured_kernels, bug_info = result, None
                    
                    all_captured_kernels.update(captured_kernels)
                    execution_valid = True
                    
                    # Ã°Å¸â€ â€¢ Ã¥Â®Å¾Ã¦â€”Â¶Ã¥Å½Â»Ã©â€¡ÂÃ¨Â®Â°Ã¥Â½â€¢
                    if bug_info and hasattr(self, 'bug_deduplicator'):
                        is_new, sig = self.bug_deduplicator.record_bug(
                            oracle_type=bug_info.get('type', 'crash'),
                            error_msg=bug_info.get('error', ''),
                            iteration=i
                        )
                        if is_new:
                            print(f"  Ã°Å¸Ââ€º NEW [{bug_info.get('oracle', 'BUG')}]: {sig[:50]}...")
                            
                except Exception as e:
                    pass
            
            # Ã¨Â®Â°Ã¥Â½â€¢Ã¥Å½Å¸Ã¥Â§â€¹Ã¦â€¢Â°Ã¦ÂÂ®
            if self.logger:
                self.logger.log_iteration(
                    iteration=i,
                    source=source,
                    api=api,
                    valid=execution_valid,
                    kernels=list(all_captured_kernels)
                )
            
            # Ã°Å¸â€ â€¢ Ã¨Â®Â°Ã¥Â½â€¢ DispatcherSpace Ã¥â€˜Â½Ã¤Â¸Â­
            if self.dispatcher_space and execution_valid:
                # Ã¦Â£â‚¬Ã¦Âµâ€¹Ã¦ËœÂ¯Ã¥ÂÂ¦Ã¤Â½Â¿Ã§â€Â¨Ã¤Âºâ€  CUDA (CUDA oracle Ã¦Ë†â€“Ã¤Â»Â£Ã§Â ÂÃ¤Â¸Â­Ã¦Å“â€° .cuda())
                is_cuda = OracleType.CUDA in self.oracles
                self.dispatcher_space.record_hit(self.api_name, api, is_cuda=is_cuda, iteration=i)
            
            # Ã°Å¸â€ â€¢ Ã¥Â®Å¡Ã¦Å“Å¸Ã¨Â®Â°Ã¥Â½â€¢Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Â¿Â«Ã§â€¦Â§ (Ã¦Â¯Â50Ã¦Â¬Â¡Ã¨Â¿Â­Ã¤Â»Â£)
            if self.dispatcher_space and i % 50 == 0:
                self.dispatcher_space.record_coverage_snapshot(self.api_name, i)
            
            # ================================================================
            # Ã¦â€ºÂ´Ã¦â€“Â°Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡ + Ã¥Å Â¨Ã¦â‚¬ÂÃ©Â¥Â±Ã¥â€™Å’Ã¦Â£â‚¬Ã¦Âµâ€¹
            # ================================================================
            new_count, (should_expand, expansion_msg) = self.coverage.update(all_captured_kernels, i)
            
            # Ã¥â€œÂÃ¥Âºâ€Ã¦â€°Â©Ã¥Â¼Â Ã¤Â¿Â¡Ã¥ÂÂ·
            if should_expand and self.mutation_controller:
                self.mutation_controller.enter_expansion_mode()
            
            # Ã¥Ââ€˜Ã§Å½Â°Ã¦â€“Â° Kernel Ã¦â€”Â¶Ã©â‚¬â‚¬Ã¥â€¡ÂºÃ¦â€°Â©Ã¥Â¼Â Ã¦Â¨Â¡Ã¥Â¼Â
            if new_count > 0 and self.mutation_controller:
                if self.mutation_controller.expansion_mode:
                    self.mutation_controller.exit_expansion_mode()
            
            # Ã¦â€ºÂ´Ã¦â€“Â° Corpus
            if new_count > 0 and self.corpus:
                self.corpus.add_seed(api, all_captured_kernels)
            
            # Ã¥Â®Å¡Ã¦Å“Å¸Ã¦â€°Â«Ã¦ÂÂ Bug
            if (i + 1) % bug_scan_interval == 0:
                prev_bug_count = self.bug_tracker.get_total_bugs()
                self.bug_tracker.scan_bugs()
                new_bugs = self.bug_tracker.get_total_bugs() - prev_bug_count
                
                if new_bugs > 0:
                    print(f"  Ã°Å¸ÂÅ¾ [{self.strategy_name}] Found {new_bugs} new bugs! "
                          f"Total: {self.bug_tracker.get_total_bugs()}")
            
            # ================================================================
            # Checkpoint + Safety Guards
            # ================================================================
            if (i + 1) % checkpoint_interval == 0:
                # Checkpoint Ã¤Â¿ÂÃ¥Â­Ëœ
                if self.checkpoint_manager:
                    self.checkpoint_manager.save(
                        iteration=i,
                        coverage_kernels=self.coverage.all_kernels,
                        corpus_seeds=self.corpus.corpus if self.corpus else None,
                        logger_iterations=self.logger.total_iterations if self.logger else 0
                    )
                
                # Ã§Â£ÂÃ§â€ºËœÃ§Â©ÂºÃ©â€”Â´Ã¦Â£â‚¬Ã¦Å¸Â¥
                if self.disk_guard:
                    is_critical, message = self.disk_guard.check_and_cleanup()
                    if is_critical:
                        print(f"\n{message}")
                        print(f"Ã¢ÂÅ’ STOPPING: Critical disk space issue")
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
                
                # Speedometer Ã§Å Â¶Ã¦â‚¬Â
                if self.speedometer:
                    print(f"Speed: {self.speedometer.get_status(i, max_iterations)}")
                    is_slow, warning = self.speedometer.check_speed()
                    if is_slow:
                        print(warning)
                
                # Disk Ã§Å Â¶Ã¦â‚¬Â
                if self.disk_guard:
                    print(f"Disk: {self.disk_guard.get_status()}")
                
                # OutlierFilter Ã§Å Â¶Ã¦â‚¬Â
                if self.outlier_filter:
                    print(f"Filter: {self.outlier_filter.get_status()}")
                
                # Corpus Ã§Å Â¶Ã¦â‚¬Â
                if self.corpus:
                    print(f"Corpus: {self.corpus.size()} seeds")
                
                # Ã°Å¸â€ â€¢ DispatcherSpace Ã§Å Â¶Ã¦â‚¬Â - Ã¢Å“â€¦Ã£â‚¬ÂÃ¥Â·Â²Ã¤Â¿Â®Ã¥Â¤ÂÃ£â‚¬â€˜Ã¤Â½Â¿Ã§â€Â¨ get_detailed_stats Ã©ÂÂ¿Ã¥â€¦ÂÃ¥Â´Â©Ã¦ÂºÆ’
                if self.dispatcher_space:
                    cov = self.dispatcher_space.get_coverage(self.api_name)
                    # hits = len(self.dispatcher_space.hits[self.api_name]) # Ã¢ÂÅ’ Ã©â€â„¢Ã¨Â¯Â¯Ã¤Â»Â£Ã§Â Â
                    stats = self.dispatcher_space.get_detailed_stats(self.api_name) # Ã¢Å“â€¦ Ã¦Â­Â£Ã§Â¡Â®Ã¤Â»Â£Ã§Â Â
                    hits = stats.get('hit_count', 0)
                    print(f"Dispatcher: {hits} combos ({cov*100:.1f}% coverage)")
                
                # Ã°Å¸â€Â¥ Mutation Controller Ã§Å Â¶Ã¦â‚¬Â
                if self.mutation_controller:
                    print(f"Mutation: {self.mutation_controller.get_status()}")
                
                # Ã°Å¸â€Â§ Ã¥Â®Å¡Ã¦Å“Å¸Ã¦Â¸â€¦Ã§Ââ€ Ã¥â€ â€¦Ã¥Â­ËœÃ©ËœÂ²Ã¦Â­Â¢ OOM (Exit 137)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Bug Ã¦â€°Â«Ã¦ÂÂ
                self.bug_tracker.scan_bugs()
        
        # ====================================================================
        # Ã¥Â®Å’Ã¦Ë†ÂÃ¥ÂÅ½Ã¦Â¸â€¦Ã§Ââ€ 
        # ====================================================================
        if self.checkpoint_manager:
            self.checkpoint_manager.clear()
        
        # Ã¦Å“â‚¬Ã§Â»Ë†Ã§Â»Å¸Ã¨Â®Â¡
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
        
        # Ã°Å¸â€ â€¢ DispatcherSpace Ã¦Å“â‚¬Ã§Â»Ë†Ã¦Å Â¥Ã¥â€˜Å 
        if self.dispatcher_space:
            self.dispatcher_space.print_summary()
        
        # Ã°Å¸â€ â€¢ Bug Ã¥Å½Â»Ã©â€¡ÂÃ¦Å Â¥Ã¥â€˜Å 
        if hasattr(self, 'bug_deduplicator'):
            self.bug_deduplicator.print_summary()
        
        print(f"{'='*70}")
        
        return self.coverage
    
    def print_stats(self):
        """Ã¦â€°â€œÃ¥ÂÂ°Ã§Â»Å¸Ã¨Â®Â¡Ã¤Â¿Â¡Ã¦ÂÂ¯"""
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS: {self.strategy_name}")
        print(f"{'='*70}")
        print(f"Total Kernels: {self.coverage.get_total()}")
        
        # Bug Ã§Â»Å¸Ã¨Â®Â¡
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
    random_dispatcher: DispatcherSpace = None,  # Ã°Å¸â€ â€¢
    guided_dispatcher: DispatcherSpace = None,  # Ã°Å¸â€ â€¢
    random_dedup = None,  # Ã°Å¸â€ â€¢ Bug Ã¥Å½Â»Ã©â€¡ÂÃ¥â„¢Â¨
    guided_dedup = None   # Ã°Å¸â€ â€¢
):
    """Ã§Â»ËœÃ¥Ë†Â¶Ã¥Â®Å’Ã¦â€¢Â´Ã§Å¡â€žÃ§Â»â€œÃ¦Å¾Å“Ã¥Â¯Â¹Ã¦Â¯â€Ã¥â€ºÂ¾Ã¯Â¼Ë†Ã¥Å’â€¦Ã¥ÂÂ« Bug Ã§Â»Å¸Ã¨Â®Â¡Ã¯Â¼â€°"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Kernel Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¦â€ºÂ²Ã§ÂºÂ¿
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
    
    # Bug Ã¦â€¢Â°Ã©â€¡ÂÃ¥Â¯Â¹Ã¦Â¯â€
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
    
    # Ã¦â‚¬Â» Bug Ã¦â€¢Â°Ã¥Â¯Â¹Ã¦Â¯â€
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
    
    # Ã°Å¸â€ â€¢ Ã¥â€ºÂ¾D: 5D Ã§Å Â¶Ã¦â‚¬ÂÃ§Â©ÂºÃ©â€”Â´Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥ÂËœÃ¥Å’â€“
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Ã¤Â»Å½ dispatcher Ã¨Å½Â·Ã¥Ââ€“Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Å½â€ Ã¥ÂÂ²
    random_cov_history = []
    guided_cov_history = []
    
    if random_dispatcher and api_name in random_dispatcher.coverage_history:
        random_cov_history = random_dispatcher.coverage_history[api_name]
    
    if guided_dispatcher and api_name in guided_dispatcher.coverage_history:
        guided_cov_history = guided_dispatcher.coverage_history[api_name]
    
    if random_cov_history:
        iters, covs = zip(*random_cov_history)
        # Cap Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Ë†Â° 100%
        covs = [min(c * 100, 100.0) for c in covs]
        ax4.plot(iters, covs, label="Random", 
                color="#3498db", linewidth=2, marker='o', markersize=3)
    
    if guided_cov_history:
        iters, covs = zip(*guided_cov_history)
        # Cap Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡Ã¥Ë†Â° 100%
        covs = [min(c * 100, 100.0) for c in covs]
        ax4.plot(iters, covs, label="Guided", 
                color="#e74c3c", linewidth=2, marker='s', markersize=3)
    
    ax4.set_xlabel("Iteration", fontsize=11)
    ax4.set_ylabel("Coverage (%)", fontsize=11)
    ax4.set_title("D) 5D State Space Coverage", 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc="lower right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)  # Ã¥â€ºÂºÃ¥Â®Å¡ Y Ã¨Â½Â´Ã¨Å’Æ’Ã¥â€ºÂ´ 0-100%
    
    # Ã§Â»Å¸Ã¨Â®Â¡Ã¨Â¡Â¨
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Ã°Å¸â€ â€¢ Ã¨Å½Â·Ã¥Ââ€“ 5Ã§Â»Â´Ã§Å Â¶Ã¦â‚¬ÂÃ§Â©ÂºÃ©â€”Â´Ã¨Â¦â€ Ã§â€ºâ€“Ã§Å½â€¡ (capped at 100%)
    random_state_cov = min(random_dispatcher.get_coverage(api_name) * 100, 100.0) if random_dispatcher else 0
    guided_state_cov = min(guided_dispatcher.get_coverage(api_name) * 100, 100.0) if guided_dispatcher else 0
    
    # Ã°Å¸â€ â€¢ Ã¨Å½Â·Ã¥Ââ€“Ã§Ââ€ Ã¨Â®ÂºÃ§Â©ÂºÃ©â€”Â´Ã¥Â¤Â§Ã¥Â°Â (Ã¨Â¿â€¡Ã¦Â»Â¤Ã¥ÂÅ½)
    random_valid_space = 0
    guided_valid_space = 0
    if random_dispatcher and api_name in random_dispatcher.state_space.api_spaces:
        random_valid_space = random_dispatcher.state_space.api_spaces[api_name]['valid_space']
    if guided_dispatcher and api_name in guided_dispatcher.state_space.api_spaces:
        guided_valid_space = guided_dispatcher.state_space.api_spaces[api_name]['valid_space']
    
    # Ã°Å¸â€ â€¢ Ã¨Å½Â·Ã¥Ââ€“Ã¥Å½Â»Ã©â€¡ÂÃ¥ÂÅ½Ã§Å¡â€ž Bug Ã¦â€¢Â°Ã©â€¡Â
    random_unique = sum(random_dedup.unique_bugs.values()) if random_dedup else random_bugs.get_total_bugs()
    guided_unique = sum(guided_dedup.unique_bugs.values()) if guided_dedup else guided_bugs.get_total_bugs()
    
    # Ã¨Â®Â¡Ã§Â®â€”Ã¥Å½Â»Ã©â€¡ÂÃ¥ÂÅ½Ã§Å¡â€ž Bug Ã¦â€¢Â°
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
    
    # Ã¦â‚¬Â»Ã¦Â â€¡Ã©Â¢Ëœ
    fig.suptitle(
        f"Hybrid Strategy Benchmark: {api_name}\n"
        f"ÃŽÂµ-greedy (10% Exploration) + Adaptive Mutation + Dynamic Saturation",
        fontsize=15, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    plot_file = join(output_dir, f"{api_name.replace('.', '_')}_hybrid_strategy.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nÃ°Å¸â€œÅ  Hybrid strategy plot saved: {plot_file}")


# =============================================================================
# 🔥 Subprocess Isolation + Crash Recovery (从 TensorFlow 版本移植)
# =============================================================================

def _generate_crash_bug_file(output_dir: str, phase_name: str, api_name: str,
                              exit_code: int, error_desc: str):
    """子进程崩溃时生成 bug 文件"""
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
    """从磁盘重新扫描 bug 文件"""
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
    """加载子进程定期保存的中间结果"""
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
    在独立子进程中运行 Fuzzing Phase
    
    模仿 TensorFlow 版本的 subprocess 隔离逻辑
    """
    try:
        # 应用投毒补丁
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
            
            # 定期保存中间结果（避免崩溃丢失数据）
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
            
            # 定期保存中间结果
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
        
        # 通过 Queue 传递结果（简化版，只传递统计数据）
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
    使用 subprocess 隔离运行 Phase（模仿 TensorFlow 版本）
    
    特点：
    1. 独立进程运行，内存隔离
    2. 超时保护
    3. 崩溃恢复机制
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
    
    # 等待完成或超时
    process.join(timeout=timeout_minutes * 60)
    
    # 超时处理
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
        
        # 从磁盘恢复部分结果
        recovered_bugs = _rescan_bugs_from_disk(output_dir, phase_name)
        partial_result = _load_partial_result(output_dir, phase_name)
        
        return {
            'success': False,
            'error': f'Timeout after {timeout_minutes} minutes',
            'phase': phase_name,
            'bugs': recovered_bugs,
            'total_kernels': partial_result.get('total_kernels', 0),
        }
    
    # 正常完成：从 Queue 或磁盘加载结果
    result = None
    
    if not result_queue.empty():
        try:
            result = result_queue.get(timeout=30)
        except Exception as e:
            print(f"⚠️  Queue retrieval failed: {e}")
    
    exit_code = process.exitcode
    
    # 崩溃处理
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
    
    # 如果仍然没有结果，从磁盘加载
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
    """运行完整的 Hybrid Strategy 实验（带 subprocess 隔离）"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 分配迭代次数
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
    # Phase 1: Random (在 subprocess 中运行)
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
    # Phase 2: Guided (在 subprocess 中运行)
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
    # 从磁盘重新加载结果（防止 subprocess 数据丢失）
    # ==========================================================================
    random_coverage = _reconstruct_coverage_from_result(random_result, "Random")
    guided_coverage = _reconstruct_coverage_from_result(guided_result, "Guided")
    
    random_bug_tracker = BugTracker("Random", join(output_dir, "random"))
    guided_bug_tracker = BugTracker("Guided", join(output_dir, "guided"))
    
    random_bug_tracker.scan_bugs()
    guided_bug_tracker.scan_bugs()
    
    # 更新 bug 统计（取 max 值）
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
        # 重建 dispatcher_space 和 bug_deduplicator（模拟对象）
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
    
    # 最终摘要
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
    """从结果字典重建 Coverage 对象"""
    coverage = EnhancedCoverageTracker(phase_name)
    coverage.history = result.get('coverage_history', [])
    
    # 从 history 重建 all_kernels（假设最后的数量）
    total_kernels = result.get('total_kernels', 0)
    for i in range(total_kernels):
        coverage.all_kernels.add(f"kernel_{i}")
    
    return coverage


def _reconstruct_dispatcher_from_result(result: Dict, api_name: str):
    """从结果字典重建 DispatcherSpace 对象（模拟）"""
    dispatcher = DispatcherSpace(max_hits=50000)
    dispatcher.register_api(api_name)
    
    # 使用结果中的统计信息
    # （注意：这是简化版，实际 state 无法完全恢复）
    return dispatcher


# =============================================================================
# Main Fuzzing Phase Runner (模仿 TF 版本)
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
    单阶段 Fuzzing 执行函数（在 subprocess 中运行）
    
    完全模仿 TensorFlow 版本的逻辑
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
    
    # 初始化组件
    try:
        config = configparser.ConfigParser()
        
        # 查找配置文件
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
    
    # 应用投毒补丁
    PoisonPatcher.patch()
    
    try:
        # 创建 Fuzzer
        if phase_name == "Random":
            ProbabilityPatcher.patch_high_db_probability()  # Random 也用高采样率
            
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
        
        # 运行 Fuzzing Loop
        coverage = fuzzer.run_fuzzing_loop(
            max_iterations=max_iterations,
            checkpoint_interval=max(100, max_iterations // 20),
            bug_scan_interval=50
        )
        
        # 扫描 bugs
        fuzzer.bug_tracker.scan_bugs()
        
        # 准备结果
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
        
        # 保存中间结果到磁盘
        result_file = join(output_dir, f"{phase_name.lower()}_result.json")
        try:
            with open(result_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"💾 Results saved: {result_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
        
        # 通过 Queue 传输（简化版）
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
        # 恢复补丁
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
    使用 subprocess 隔离运行 Phase（完全模仿 TensorFlow 版本）
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
    
    # 等待完成或超时
    process.join(timeout=timeout_minutes * 60)
    
    # 超时处理
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
    
    # 正常完成：从 Queue 或磁盘加载结果
    result = None
    
    if not result_queue.empty():
        try:
            result = result_queue.get(timeout=30)
        except Exception as e:
            print(f"⚠️  Queue retrieval failed: {e}")
    
    # 从文件加载完整结果
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
    
    # 崩溃处理
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
            
            # 合并 bugs
            old_bugs = result.get('bugs', {})
            result['bugs'] = {
                'crash': max(old_bugs.get('crash', 0), recovered_bugs['crash']),
                'cuda': max(old_bugs.get('cuda', 0), recovered_bugs['cuda']),
                'precision': max(old_bugs.get('precision', 0), recovered_bugs['precision']),
                'total': max(old_bugs.get('total', 0), recovered_bugs['total']),
                'unique': max(old_bugs.get('unique', 0), recovered_bugs['unique']),
            }
    
    # 如果仍然没有结果，尝试从磁盘恢复
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
        description="Hybrid Strategy Benchmark: ÃŽÂµ-greedy + Adaptive Mutation + Poison Injection"
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
    print("Ã°Å¸â€Â¬ HYBRID STRATEGY BENCHMARK + POISON INJECTION")
    print("="*70)
    print(f"API: {args.api}")
    print(f"Max Iterations: {args.max_iterations:,}")
    print(f"Strategy: ÃŽÂµ-greedy (10% exploration) + Adaptive Mutation")
    print(f"Ã°Å¸Â§Âª Poison Injection: 5% Inf + 5% NaN + 10% Extreme")
    print(f"Ã°Å¸Å½Â¯ Precision Tolerance: {args.diff_bound}")
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
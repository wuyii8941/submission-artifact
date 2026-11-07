"""Route A: TorchFuzz-backed graph generation for DeepOpFuzz.

This package is part of submission-artifact.  It keeps the generator,
condition binding, artifact observation, policies, and event schema in the
same implementation tree as the original fuzzer.
"""

from .model import (
    ConditionRequest,
    GenerationResult,
    GraphRequest,
    Observation,
    TestCase,
)

__all__ = [
    "ConditionRequest",
    "GenerationResult",
    "GraphRequest",
    "Observation",
    "TestCase",
]

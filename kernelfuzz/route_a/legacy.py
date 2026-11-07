"""Replay adapter for the original DeepOpFuzz programs (A5).

The original generator remains unchanged.  This adapter only wraps a saved
Python program in the Route A ``TestCase`` schema and sends it through the
same executor, observer, evidence store, and JSONL event format used by
A0--A4.  It is intentionally replay-only: it does not reinterpret legacy
coverage counters as Route A target hits.
"""

import hashlib
from pathlib import Path
from typing import Iterable, Optional

from .model import TestCase


def testcase_from_legacy_program(
    program_path: str,
    *,
    source_metadata: Optional[dict] = None,
) -> TestCase:
    path = Path(program_path).resolve()
    code = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    metadata = dict(source_metadata or {})
    metadata.update({"source": "legacy_deepopfuzz", "program_path": str(path)})
    return TestCase(
        test_id="legacy-" + digest[:24],
        code=code,
        graph_manifest={
            "source": "legacy_deepopfuzz",
            "structure_signature": None,
            "full_signature": digest,
            "legacy_program_sha256": digest,
        },
        conditions=[],
        execution={"runtime_device": "cuda", "source": "legacy"},
        provenance=metadata,
    )


def target_tokens_from_metadata(metadata: Optional[dict]) -> Iterable[str]:
    """Read optional target tokens without treating legacy labels as hits."""
    if not metadata:
        return ()
    tokens = metadata.get("observer_target_tokens", ())
    if isinstance(tokens, str):
        return (tokens,)
    return tuple(str(token) for token in tokens)

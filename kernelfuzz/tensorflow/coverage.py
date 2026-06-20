"""HLO / Graph coverage signal and bug basin tracking for TF fuzzer.

Coverage signals:
  - HLO ops: XLA operation inventory (post-fusion)
  - HLO signatures: (op_name, count) tuples from HLO text
  - Fusion patterns: (kind, inner_op_set) from optimized HLO
  - Graph signatures: content hash from GraphDef
  - HLO hashes: content hash for fine-grained dedup

Bug basin dedup:
  - Jaccard similarity at 0.7 threshold
  - Subset matching for small known basins
  - Weighted: 60% op_fingerprint + 40% hlo_ops
"""
import re
import hashlib
import tensorflow as tf
from collections import defaultdict


# ---- HLO extraction (XLA path) ----

def get_hlo_text(fn, args, stage='optimized_hlo'):
    """Extract HLO text for an XLA-compiled tf.function."""
    try:
        fn_xla = tf.function(fn, jit_compile=True)
        concrete = fn_xla.get_concrete_function(*args)
        try:
            ir_callable = fn_xla.experimental_get_compiler_ir(*args)
            hlo = ir_callable(stage=stage)
            return hlo
        except Exception:
            return None
    except Exception:
        return None


def get_graph_def_hash(fn, args):
    """Get GraphDef hash for non-XLA tf.function."""
    try:
        fn_tfn = tf.function(fn)
        concrete = fn_tfn.get_concrete_function(*args)
        gdef = concrete.graph.as_graph_def()
        op_counts = defaultdict(int)
        for node in gdef.node:
            op_counts[node.op] += 1
        sig_str = str(sorted(op_counts.items()))
        h = hashlib.md5(sig_str.encode()).hexdigest()[:16]
        return h, dict(op_counts)
    except Exception:
        return None, {}


def extract_hlo_ops(hlo_text):
    """Extract HLO op names from HLO text."""
    if not hlo_text:
        return frozenset()
    ops = set()
    for line in hlo_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('HloModule'):
            continue
        m = re.match(r'(?:ROOT\s+)?%\S+\s*=\s*\S+\s+(\w+)\(', line)
        if m:
            ops.add(m.group(1))
    return frozenset(ops)


def extract_hlo_signature(hlo_text):
    if not hlo_text:
        return ()
    ops = defaultdict(int)
    for line in hlo_text.split('\n'):
        line = line.strip()
        m = re.match(r'(?:ROOT\s+)?%\S+\s*=\s*\S+\s+(\w+)\(', line)
        if m:
            ops[m.group(1)] += 1
    return tuple(sorted(ops.items()))


def extract_hlo_fusions(hlo_text):
    """Extract XLA fusion patterns from optimized HLO text.

    Returns a set of (fusion_kind, frozenset_of_inner_ops) tuples.
    """
    if not hlo_text:
        return set()

    lines = hlo_text.split('\n')

    fused_bodies = {}
    current_name = None
    current_ops = set()
    brace_depth = 0
    in_fused = False

    for line in lines:
        stripped = line.strip()
        header = re.match(r'^%(fused[\w.]*)\s*\(', stripped)
        if header:
            current_name = header.group(1)
            current_ops = set()
            in_fused = True
            brace_depth = 0

        if in_fused:
            brace_depth += stripped.count('{') - stripped.count('}')
            op_match = re.match(r'(?:ROOT\s+)?%\S+\s*=\s*\S+\s+(\w+)\(', stripped)
            if op_match:
                op_name = op_match.group(1)
                if op_name not in ('parameter', 'get-tuple-element', 'tuple',
                                   'constant', 'broadcast'):
                    current_ops.add(op_name)

            if brace_depth <= 0 and '}' in stripped and current_name is not None:
                fused_bodies[current_name] = frozenset(current_ops) if current_ops else frozenset()
                in_fused = False
                current_name = None

    fusions = set()
    for line in lines:
        stripped = line.strip()
        call_match = re.search(
            r'fusion\([^)]*\),\s*kind=(\w+),\s*calls=%(fused[\w.]*)', stripped)
        if call_match:
            kind = call_match.group(1)
            body_name = call_match.group(2)
            inner_ops = fused_bodies.get(body_name, frozenset())
            if inner_ops:
                fusions.add((kind, inner_ops))

    if not fusions and fused_bodies:
        for name, ops in fused_bodies.items():
            fusions.add(('unknown', ops))

    return fusions


def extract_hlo_fusion_hash(hlo_text):
    """Get a content hash of the full HLO module for fine-grained dedup."""
    if not hlo_text:
        return None
    cleaned = re.sub(r'%\S+', '%VAR', hlo_text)
    cleaned = re.sub(r'metadata=\{[^}]*\}', '', cleaned)
    return hashlib.md5(cleaned.encode()).hexdigest()[:16]


class CoverageTracker:
    def __init__(self):
        self.seen_hlo_ops = set()
        self.seen_hlo_signatures = set()
        self.seen_graph_sigs = set()
        self.seen_fusion_patterns = set()
        self.seen_hlo_hashes = set()
        self.total = 0
        self.novel = 0
        self.coverage_history = []

    def check_novelty(self, hlo_text=None, graph_sig=None):
        self.total += 1
        is_novel = False

        new_ops = set()
        new_fusions = set()
        if hlo_text:
            ops = extract_hlo_ops(hlo_text)
            sig = extract_hlo_signature(hlo_text)
            new_ops = ops - self.seen_hlo_ops
            if new_ops or sig not in self.seen_hlo_signatures:
                is_novel = True
            self.seen_hlo_ops.update(ops)
            self.seen_hlo_signatures.add(sig)

            fusions = extract_hlo_fusions(hlo_text)
            new_fusions = fusions - self.seen_fusion_patterns
            if new_fusions:
                is_novel = True
            self.seen_fusion_patterns.update(fusions)

            hlo_hash = extract_hlo_fusion_hash(hlo_text)
            if hlo_hash and hlo_hash not in self.seen_hlo_hashes:
                is_novel = True
                self.seen_hlo_hashes.add(hlo_hash)

        if graph_sig and graph_sig not in self.seen_graph_sigs:
            is_novel = True
            self.seen_graph_sigs.add(graph_sig)

        if is_novel:
            self.novel += 1

        self.coverage_history.append({
            "iter": self.total,
            "hlo_ops": len(self.seen_hlo_ops),
            "hlo_sigs": len(self.seen_hlo_signatures),
            "fusion_patterns": len(self.seen_fusion_patterns),
            "hlo_hashes": len(self.seen_hlo_hashes),
        })

        return is_novel, {
            "new_ops": sorted(new_ops),
            "new_fusions": len(new_fusions),
            "total_hlo_ops": len(self.seen_hlo_ops),
            "total_hlo_sigs": len(self.seen_hlo_signatures),
            "total_fusion_patterns": len(self.seen_fusion_patterns),
            "total_hlo_hashes": len(self.seen_hlo_hashes),
            "total_graph_sigs": len(self.seen_graph_sigs),
        }

    def stats(self):
        return {
            "total_programs": self.total,
            "novel_programs": self.novel,
            "hlo_ops": len(self.seen_hlo_ops),
            "hlo_signatures": len(self.seen_hlo_signatures),
            "fusion_patterns": len(self.seen_fusion_patterns),
            "hlo_hashes": len(self.seen_hlo_hashes),
            "graph_signatures": len(self.seen_graph_sigs),
            "saturation_ratio": round(self.novel / max(self.total, 1), 3),
        }


class BugBasinTracker:
    """Dedup known bug patterns by op fingerprint similarity."""

    def __init__(self, similarity_threshold=0.7):
        self.known_basins = []
        self.threshold = similarity_threshold

    def _jaccard(self, a, b):
        a, b = set(a), set(b)
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def is_known(self, op_fingerprint, hlo_ops=None):
        op_set = set(op_fingerprint)
        for known_fp, known_hlo, _ in self.known_basins:
            if len(known_fp) <= 2 and known_fp.issubset(op_set):
                return True
            fp_sim = self._jaccard(op_fingerprint, known_fp)
            if hlo_ops and known_hlo:
                hlo_sim = self._jaccard(hlo_ops, known_hlo)
                combined = 0.6 * fp_sim + 0.4 * hlo_sim
            else:
                combined = fp_sim
            if combined >= self.threshold:
                return True
        return False

    def add_basin(self, op_fingerprint, hlo_ops=None, description=""):
        self.known_basins.append((set(op_fingerprint), set(hlo_ops or []), description))

    def stats(self):
        return {"known_basins": len(self.known_basins)}

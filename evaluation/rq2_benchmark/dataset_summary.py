#!/usr/bin/env python3
"""
RQ2 Ground-Truth Bug Dataset Summary
=====================================

Dataset: 48 confirmed torch.compile correctness bugs reproducible on PyTorch 2.2.0+cu121

Data Sources:
  1. Demystifying Dataset (35 bugs)
     - From: "Demystifying and Detecting Correctness Bugs in torch.compile" (under review)
     - 116 confirmed torch.compile correctness bugs with GitHub issue links
     - We wrote faithful reproducers for all 116, tested on PyTorch 2.2.0
     - 35 bugs confirmed reproducible (BUG), rest were PASS/ERROR on 2.2.0

  2. GitHub-Wide Search (13 bugs)
     - Systematic search of pytorch/pytorch GitHub issues for torch.compile
       silent correctness bugs NOT in the Demystifying dataset
     - Search queries: "torch.compile wrong result", "inductor miscompile",
       "inductor silent correctness", "wrong output", "different result", etc.
     - ~90 candidate issues identified, reproducers written and tested
     - 13 bugs confirmed reproducible on PyTorch 2.2.0
     - Many early PT2 bugs (93xxx-100xxx series) already fixed by 2.2.0

Verification Method:
  - Each bug tested with original GitHub issue reproducer (or faithful adaptation)
  - Differential oracle: eager mode vs torch.compile output comparison
  - Cross-validated on 3 PyTorch versions:
      * PyTorch 2.2.0+cu121: 48 BUG (target version)
      * PyTorch 2.4.0+cu121: fewer (bugs progressively fixed)
      * PyTorch nightly: fewest (most bugs fixed)

Bug Type Distribution (48 bugs):
  Low-level Code Generation:  18 (37.5%)  -- codegen, fusion, buffer aliasing
  In-Place Operation Handling:  9 (18.8%)  -- mutation, view, alias tracking
  Operator Transformation:      8 (16.7%)  -- op lowering, fusion rules
  Graph Semantic Capturing:     7 (14.6%)  -- dynamo capture, guard, cache
  Memory Layout Conflict:       3  (6.3%)  -- stride, layout mismatch
  Graph Caching:                3  (6.3%)  -- stale compiled graph

Device Distribution (48 bugs):
  CUDA-only: 18 (37.5%)
  CPU-only:  30 (62.5%)

Source Distribution (48 bugs):
  Demystifying dataset: 35 (72.9%)
  GitHub-wide search:   13 (27.1%)

Notable Bug Clusters:
  - Output-order-dependent miscompiles (#114221-#114229, #115159, #115260, #114215):
    8 bugs where inductor buffer aliasing causes computation results to change
    based on unrelated graph structure (return tuple order, extra output nodes).
    Root cause: fusion scheduling reuses buffer storage incorrectly.
"""
import csv
import os
from collections import Counter

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'ground_truth_bugs.csv')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        bugs = list(reader)

    print(f"RQ2 Ground-Truth Bug Dataset: {len(bugs)} bugs on PyTorch 2.2.0")
    print("=" * 70)

    print(f"\nBy Source:")
    for src, count in Counter(b['source'] for b in bugs).most_common():
        print(f"  {src}: {count}")

    print(f"\nBy Bug Type:")
    for bt, count in Counter(b['bug_type'] for b in bugs).most_common():
        print(f"  {bt}: {count}")

    print(f"\nBy Device:")
    for dev, count in Counter(b['device'] for b in bugs).most_common():
        print(f"  {dev}: {count}")

    print(f"\nAll {len(bugs)} bugs:")
    print(f"  {'Issue':>8s}  {'Type':<30s}  {'Device':<6s}  {'Source':<16s}  Title")
    print(f"  {'-'*8}  {'-'*30}  {'-'*6}  {'-'*16}  {'-'*40}")
    for b in sorted(bugs, key=lambda x: int(x['issue_id'])):
        print(f"  #{b['issue_id']:>7s}  {b['bug_type']:<30s}  {b['device']:<6s}  {b['source']:<16s}  {b['title']}")


if __name__ == '__main__':
    main()

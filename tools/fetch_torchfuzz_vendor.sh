#!/usr/bin/env bash
set -euo pipefail

# Fetches only the upstream TorchFuzz source used by Route A.
# The revision is pinned in third_party/torchfuzz/SOURCE.json after fetching.
proxy_url="${TORCHFUZZ_PROXY:-http://127.0.0.1:27890}"
revision="1aa1528f66debd0fd1cd893a3ac73fa49cbfa52f"
base_url="https://raw.githubusercontent.com/pytorch/pytorch/${revision}/tools/experimental/torchfuzz"
dest_root="$(cd "$(dirname "$0")/.." && pwd)/third_party/torchfuzz"

paths=(
  "__init__.py" "checks.py" "codegen.py" "fuzzer.py" "multi_process_fuzzer.py"
  "ops_fuzzer.py" "runner.py" "tensor_descriptor.py" "tensor_fuzzer.py"
  "type_promotion.py" "visualize_graph.py"
  "cuda/__init__.py" "cuda/_checks.py" "cuda/_codegen.py"
  "operators/__init__.py" "operators/_dtypes.py" "operators/activations.py"
  "operators/arg.py" "operators/argsort.py" "operators/base.py"
  "operators/bitwise.py" "operators/comparison.py" "operators/constant.py"
  "operators/elementwise_math.py" "operators/gather.py"
  "operators/index_select.py" "operators/item.py" "operators/layout.py"
  "operators/logical.py" "operators/loss_functions.py"
  "operators/manipulation_indexing.py" "operators/masked_select.py"
  "operators/matrix_multiply.py" "operators/nn_functional.py"
  "operators/nonzero.py" "operators/reduction.py" "operators/registry.py"
  "operators/scalar_pointwise.py" "operators/special_functions.py"
  "operators/tensor_creation.py" "operators/tensor_pointwise.py"
  "operators/unique.py"
)

for rel_path in "${paths[@]}"; do
  target="${dest_root}/${rel_path}"
  mkdir -p "$(dirname "$target")"
  curl -fL --proxy "$proxy_url" --max-time 90 "${base_url}/${rel_path}" -o "$target"
done

printf '%s\n' "${revision}" > "${dest_root}/REVISION"

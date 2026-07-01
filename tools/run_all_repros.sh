#!/usr/bin/env bash
# Run every confirmed-bug reproducer via tools/run_repro.sh and emit a
# per-bug pass/fail summary CSV. Skips the _uncounted/ stub dirs.
#
# Usage:
#   tools/run_all_repros.sh [stack-filter]    # stack-filter: e.g. pytorch
#
# Output:
#   results/repro_results.csv   columns: stack, bug, status, exit_code, log
#
# Each reproducer is given a 5-minute soft cap (override with DOF_REPRO_TIMEOUT).
# Set DOF_*_PYTHON / DOF_TVM_HOME as documented in run_repro.sh before invoking.

set -u
cd "$(dirname "$0")/.."

filter="${1:-}"
timeout_s="${DOF_REPRO_TIMEOUT:-300}"
out_dir="results/repro_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out_dir"
csv="$out_dir/repro_results.csv"
echo "stack,bug,status,exit_code,log" > "$csv"

total=0; ok=0; bug=0; err=0
for bug_dir in data/confirmed_bugs/*/bug_* data/confirmed_bugs/*/tf_* data/confirmed_bugs/*/iree_*; do
    [ -d "$bug_dir" ] || continue
    # Skip _uncounted/ stubs
    case "$bug_dir" in *_uncounted/*) continue ;; esac
    stack=$(echo "$bug_dir" | awk -F/ '{print $3}')
    [ -n "$filter" ] && [ "$stack" != "$filter" ] && continue
    bug=$(basename "$bug_dir")
    log="$out_dir/${stack}_${bug}.log"
    total=$((total+1))

    timeout "$timeout_s" tools/run_repro.sh "$bug_dir" >"$log" 2>&1
    ec=$?

    case "$ec" in
        0)   status="ok";    ok=$((ok+1)) ;;
        124) status="timeout"; err=$((err+1)) ;;
        *)   status="bug";   bug=$((bug+1)) ;;
    esac
    echo "$stack,$bug,$status,$ec,$log" >> "$csv"
    printf "%-12s %-50s %-8s ec=%d\n" "$stack" "$bug" "$status" "$ec"
done

echo
echo "=== summary ==="
echo "total: $total"
echo "ok (exit 0 — bug did not manifest in this env): $ok"
echo "bug (non-zero exit — bug manifested as crash/error): $bug"
echo "timeout / dispatcher error: $err"
echo "CSV: $csv"

#!/usr/bin/env bash
# Run a single confirmed-bug reproducer with stack-appropriate environment.
#
# Usage:
#   tools/run_repro.sh data/confirmed_bugs/pytorch/bug_003_slice_scatter_backward
#   tools/run_repro.sh data/confirmed_bugs/tvm/bug_001_gemv_keyerror
#
# Picks one .py from the bug directory (prefer minimal_repro.py > repro.py >
# reproducer.py > reproduce.py > the only .py present), then runs it with:
#   - the right interpreter, selected from these env vars (set them once):
#       DOF_PT_PYTHON     # PyTorch nightly venv python   (pytorch/)
#       DOF_TF_PYTHON     # TensorFlow nightly venv python (tensorflow/)
#       DOF_TVM_HOME      # TVM install root (TVM_HOME/bin/python + lib/)
#       DOF_IREE_PYTHON   # IREE+ORT venv python          (iree/)
#     If a variable is unset the dispatcher falls back to `python3` on $PATH.
#   - stack-specific environment:
#       TVM:  prepends $DOF_TVM_HOME/lib to LD_LIBRARY_PATH
#       TF:   TF_CPP_MIN_LOG_LEVEL=2 to silence verbose logs
#   - TMPDIR pinned to $DOF_TMPDIR or /tmp/deepopfuzz (never $HOME)
#
# Exit code = the reproducer's exit code (0 == bug did not manifest;
# non-zero == bug manifested as a crash/error). Some reproducers print
# "BUG CONFIRMED" / "MISMATCH" on stdout and exit 0; check stdout too.

set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <bug_dir> [extra-args-to-python...]" >&2
    exit 2
fi

bug_dir="$1"; shift
if [ ! -d "$bug_dir" ]; then
    echo "error: '$bug_dir' is not a directory" >&2
    exit 2
fi

# Pick the canonical reproducer file in the bug dir.
script=""
for candidate in minimal_repro.py repro.py reproducer.py reproduce.py; do
    if [ -f "$bug_dir/$candidate" ]; then
        script="$bug_dir/$candidate"
        break
    fi
done
if [ -z "$script" ]; then
    # Fall back to the single .py present, if any
    n=$(find "$bug_dir" -maxdepth 1 -name '*.py' | wc -l)
    if [ "$n" = "1" ]; then
        script=$(find "$bug_dir" -maxdepth 1 -name '*.py')
    else
        echo "error: no canonical reproducer in $bug_dir (found $n .py files)" >&2
        exit 2
    fi
fi

# Detect stack from path component under data/confirmed_bugs/
stack=$(echo "$bug_dir" | sed -nE 's|.*/data/confirmed_bugs/([^/]+)/.*|\1|p')
if [ -z "$stack" ]; then
    echo "error: cannot detect stack from path '$bug_dir'" >&2
    exit 2
fi

# Default TMPDIR (never $HOME, never $TMP)
export TMPDIR="${DOF_TMPDIR:-/tmp/deepopfuzz}"
mkdir -p "$TMPDIR"

# Select interpreter and stack-specific environment.
case "$stack" in
    pytorch)
        PY="${DOF_PT_PYTHON:-python3}"
        ;;
    tensorflow)
        PY="${DOF_TF_PYTHON:-python3}"
        export TF_CPP_MIN_LOG_LEVEL=2
        ;;
    xla)
        # xla/ no longer exists; the erf shape-dependent bug was moved to
        # tensorflow/bug_001_erf_shape_dependent_lowering — run that path instead
        PY="${DOF_TF_PYTHON:-python3}"
        export TF_CPP_MIN_LOG_LEVEL=2
        ;;
    tvm)
        if [ -n "${DOF_TVM_HOME:-}" ]; then
            PY="$DOF_TVM_HOME/bin/python"
            export LD_LIBRARY_PATH="$DOF_TVM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        else
            PY="python3"
        fi
        ;;
    iree)
        PY="${DOF_IREE_PYTHON:-python3}"
        # IREE reproducers may shell out to iree-compile / iree-import-onnx;
        # those must already be on $PATH (typically via the IREE venv).
        ;;
    *)
        echo "warning: unrecognized stack '$stack', using python3" >&2
        PY="python3"
        ;;
esac

if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
    echo "error: interpreter '$PY' not executable; set DOF_${stack^^}_PYTHON" >&2
    exit 2
fi

echo "[run_repro] stack=$stack  script=$script  python=$PY" >&2
exec "$PY" "$script" "$@"

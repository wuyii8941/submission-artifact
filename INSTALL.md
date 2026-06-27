# Installation

## Hardware

- Linux x86_64
- NVIDIA GPU with CUDA support (the main campaigns and the historical
  benchmark are validated on NVIDIA T4 and RTX A6000)
- ~50 GB free disk for caches and intermediate results

## Software

- Python 3.11
- CUDA 12.x

## Python environment

We recommend a dedicated venv for the historical benchmark (which is pinned
to PyTorch 2.2.0) and a separate venv for the latest-PyTorch campaigns.

### Historical benchmark (RQ2/RQ3): PyTorch 2.2.0

```bash
python3 -m venv .venv-pt220
source .venv-pt220/bin/activate
pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy
pip install nnsmith                  # for the NNSmith baseline (RQ2)
```

### Primary campaign (RQ1): PyTorch nightly

```bash
python3 -m venv .venv-ptnightly
source .venv-ptnightly/bin/activate
pip install --upgrade pip
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
pip install numpy scipy
```

### TensorFlow (RQ1 TF/XLA campaign)

```bash
python3 -m venv .venv-tf
source .venv-tf/bin/activate
pip install tf-nightly numpy
```

## External runtime dependency: FreeFuzz oracle classes

The PyTorch fuzzer (`kernelfuzz/fuzzer.py`) reuses the test-execution
oracle from the FreeFuzz baseline. Clone FreeFuzz and point an
environment variable at its `classes/` package:

```bash
git clone <FreeFuzz repo, anonymized for review> /path/to/FreeFuzz
export KERNELFUZZ_FREEFUZZ_SRC=/path/to/FreeFuzz/kernel-guided/src
```

If `KERNELFUZZ_FREEFUZZ_SRC` is unset the fuzzer falls back to
`${ARTIFACT_ROOT}/../FreeFuzz-baseline/kernel-guided/src/`, so placing
the FreeFuzz checkout as a sibling of this artifact root also works.

## External datasets

### WhiteFox pre-generated tests (baseline replay)

The WhiteFox baseline replays 61k pre-generated programs from the
WhiteFox project. Clone the WhiteFox repository (cited in the paper) and
set:

```bash
export WHITEFOX_TEST_DIR=/path/to/WhiteFox/starcoder-generated/torch-inductor-generated
```

### BugsInDLLs benchmark (RQ3 external)

Obtain the BugsInDLLs dataset from the original authors (cited in the
paper). Set the path used by the corresponding evaluation scripts under
`evaluation/rq3_ablation/bugsindlls/`.

## Working directory

Scripts assume the artifact root is the current working directory:

```bash
cd /path/to/this/repo
export ARTIFACT_ROOT=$(pwd)
export TMPDIR=${TMPDIR:-/tmp/deepopfuzz}
mkdir -p $TMPDIR
```

## Smoke test

After installation, verify the environment by running any confirmed-bug
reproducer:

```bash
cd data/confirmed_bugs/pytorch
python bug_003_slice_scatter_backward/minimal_repro.py
```

A successful run prints an eager-vs-compiled gradient diff, demonstrating
that the PyTorch/Inductor stack is working.

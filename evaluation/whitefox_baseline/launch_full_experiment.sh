#!/bin/bash
# Launch full WhiteFox baseline experiment across 3 GPUs
# GPU 0: cuda, forward-only (default WhiteFox)
# GPU 1: cuda, with backward testing (test_ad=True, fairness enhancement)
# GPU 2: cpu, forward-only (for comparison)

PYTHON="python3"
SCRIPT="${ARTIFACT_ROOT}/paper/evaluation/whitefox_baseline/batch_runner.py"
LOGDIR="${ARTIFACT_ROOT}/paper/evaluation/whitefox_baseline"

echo "=== Launching WhiteFox Baseline Experiment ==="
echo "Start time: $(date)"
echo ""

# GPU 0: Default WhiteFox (cuda, forward-only)
echo "[1/3] GPU 0: cuda, forward-only (default WhiteFox config)"
nohup ${PYTHON} ${SCRIPT} \
    --gpu 0 --device cuda --timeout 10 --batch-size 50 \
    > "${LOGDIR}/stdout_gpu0_cuda.log" 2>&1 &
PID1=$!
echo "  PID: ${PID1}"

# GPU 1: Enhanced WhiteFox (cuda, with backward testing)
echo "[2/3] GPU 1: cuda, with backward testing (fairness enhancement)"
nohup ${PYTHON} ${SCRIPT} \
    --gpu 1 --device cuda --timeout 15 --batch-size 30 --test-ad \
    > "${LOGDIR}/stdout_gpu1_cuda_ad.log" 2>&1 &
PID2=$!
echo "  PID: ${PID2}"

# GPU 2: CPU baseline
echo "[3/3] GPU 2: cpu, forward-only"
nohup ${PYTHON} ${SCRIPT} \
    --gpu 2 --device cpu --timeout 10 --batch-size 50 \
    > "${LOGDIR}/stdout_gpu2_cpu.log" 2>&1 &
PID3=$!
echo "  PID: ${PID3}"

echo ""
echo "All experiments launched."
echo "PIDs: ${PID1} ${PID2} ${PID3}"
echo "Monitor with:"
echo "  tail -f ${LOGDIR}/stdout_gpu0_cuda.log"
echo "  tail -f ${LOGDIR}/stdout_gpu1_cuda_ad.log"
echo "  tail -f ${LOGDIR}/stdout_gpu2_cpu.log"
echo ""
echo "Expected completion: ~8-12 hours"

# Save PIDs
echo "${PID1} ${PID2} ${PID3}" > "${LOGDIR}/experiment_pids.txt"

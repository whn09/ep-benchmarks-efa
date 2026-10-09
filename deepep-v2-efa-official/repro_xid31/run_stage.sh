#!/usr/bin/env bash
# Drive one Xid31-repro stage on N nodes from the laptop (foreground ssh, workers first).
#
#   NODES="B200-1 B200-2 B200-3 B200-4" STAGE=pressure|stress TAG=<name> ./run_stage.sh [extra args]
#
# STAGE=pressure  upstream tests/elastic/test_ep.py --do-pressure-test (bit-exact checks of
#                 dispatch / cached dispatch / combine per seed) at the target MoE shape
# STAGE=stress    repro_xid31/stress_1f1b.py (handles in flight, cached dispatch, skew, hot rank)
#
# Every run arms a GPU coredump on exception into repro_xid31/dumps/ on each host, so a
# reproduced Xid31 names the faulting kernel and PC (cuda-gdb <core>).
# Logs: $LOGDIR/<TAG>.node<i>.log on each host.
set -euo pipefail
NODES="${NODES:?NODES=\"<leader> <worker> ...\"}"
STAGE="${STAGE:?STAGE=pressure|stress}"
TAG="${TAG:?TAG=<log name>}"
IMAGE="${IMAGE:-deepep-v2-efa-official:sm100-874779c}"
PORT="${PORT:-8800}"
LOGDIR="${LOGDIR:-\$HOME/xid31_runs}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
N=${#NODE_ARR[@]}
SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
MASTER_IP=$($SSH -n "${NODE_ARR[0]}" 'hostname -I | awk "{print \$1}"' 2>/dev/null | tr -d ' \r')
# Absolute, not $HOME: the path also travels inside EXTRA_ENV (CUDA_COREDUMP_FILE), which
# run_test_ep.sh forwards verbatim, so an unexpanded $HOME would reach the container literally.
REPO=$($SSH -n "${NODE_ARR[0]}" 'echo $HOME/work/ep-benchmarks-efa/deepep-v2-efa-official' 2>/dev/null | tr -d ' \r')

# Shape from shape.env (placeholders) / shape.local.env (gitignored); 8192 tokens/rank, 28 SM.
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/shape.env"; [ ! -f "$HERE/shape.local.env" ] || . "$HERE/shape.local.env"
SHAPE="--hidden=$HIDDEN --num-topk=$TOPK --num-experts=$NUM_EXPERTS"
case "$STAGE" in
  pressure) SCRIPT=""; ARGS="$SHAPE --do-pressure-test --reuse-elastic-buffer --skip-perf-test" ;;
  stress)   SCRIPT="$REPO/repro_xid31/stress_1f1b.py"; ARGS="$SHAPE" ;;
  *) echo "STAGE must be pressure|stress" >&2; exit 1 ;;
esac
DUMP_FLAGS="skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory"
ENVS="CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_GENERATION_FLAGS=$DUMP_FLAGS \
CUDA_COREDUMP_FILE=$REPO/repro_xid31/dumps/core_${TAG}_%h_%p ${EXTRA_ENV:-}"

echo "=== $TAG: $STAGE on $N nodes ($NODES), leader $MASTER_IP:$PORT, image $IMAGE"
echo "    args: $ARGS $*"
pids=()
for ((i = N - 1; i >= 0; i--)); do
  $SSH -n "${NODE_ARR[$i]}" "mkdir -p $LOGDIR $REPO/repro_xid31/dumps && cd $REPO && \
    IMAGE=$IMAGE WORLD_SIZE=$N NUM_PROCESSES=8 TOKENS=8192 NUM_SMS=28 MASTER_PORT=$PORT \
    TEST_FIRST_ONLY=0 RUN_TIMEOUT=${RUN_TIMEOUT:-10800} EXTRA_MOUNT=$REPO/repro_xid31 \
    ${SCRIPT:+TEST_SCRIPT=$SCRIPT} EXTRA_ENV='$ENVS' \
    bash run_test_ep.sh $i $MASTER_IP $ARGS $* > $LOGDIR/$TAG.node$((i + 1)).log 2>&1" &
  pids+=($!)
  sleep 1
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== $TAG rc=$rc"
exit $rc

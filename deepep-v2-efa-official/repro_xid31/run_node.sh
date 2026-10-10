#!/usr/bin/env bash
# On-host launcher: start THIS node's share of the repro training. Run it on every node with the
# same TAG / IMAGE / PORT / NNODES / MASTER_ADDR and that node's NODE_RANK (0 = the master).
#
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=<node 0 private ip> TAG=<name> \
#     IMAGE=deepep-v2-efa-official:sm100-b90a617-mlm1edcc0c ./run_node.sh [extra megatron args]
#
# Same container, arguments and coredump setup as run_megatron.sh (the laptop driver); the argument
# list itself comes from megatron_args.sh. The container is detached and named after TAG, writes
# its log to $LOGDIR/<TAG>.node<NODE_RANK+1>.log, and this script then `docker wait`s for it, so a
# dropped terminal does not stop the training (re-attach with `docker wait <TAG>` or tail the log).
# Exit code = the container's.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NNODES="${NNODES:?NNODES=<number of nodes>}"
NODE_RANK="${NODE_RANK:?NODE_RANK=<0..NNODES-1, 0 = master>}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR=<private ip of node 0>}"
TAG="${TAG:?TAG=<run name, identical on every node>}"
IMAGE="${IMAGE:-deepep-v2-efa-official:sm100-874779c-mlm1edcc0c}"
PORT="${PORT:-29500}"
NUM_LAYERS="${NUM_LAYERS:-12}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
RUN_TIMEOUT="${RUN_TIMEOUT:-10800}"
OVERLAP="${OVERLAP:-1}"
LOGDIR="${LOGDIR:-$HOME/xid31_runs}"
DUMPS="$HERE/dumps"
N=$NNODES
# shellcheck disable=SC1091
. "$HERE/megatron_args.sh"

mkdir -p "$LOGDIR" "$DUMPS"
LOG="$LOGDIR/$TAG.node$((NODE_RANK + 1)).log"
HCA=""; ls /sys/class/infiniband | grep -qv '^rdmap' && HCA=rdmap
DUMP_FLAGS="skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory"
echo "=== $TAG node_rank $NODE_RANK/$NNODES master $MASTER_ADDR:$PORT image $IMAGE layers $NUM_LAYERS overlap=$OVERLAP"
echo "    shape: hidden $HIDDEN, experts $NUM_EXPERTS, top-$TOPK ($( [ -f "$HERE/shape.local.env" ] && echo shape.local.env || echo 'shape.env PLACEHOLDERS'))"
echo "    log:   $LOG"

docker rm -f "$TAG" >/dev/null 2>&1 || true
docker run -d --name "$TAG" --init --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband --device=/dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro -v "$DUMPS:$DUMPS" -v "$LOGDIR:$LOGDIR" \
  -e NCCL_GIN_TYPE=5 -e NCCL_SYM_GIN_KERNELS_ENABLE=0 ${HCA:+-e NCCL_IB_HCA=$HCA} \
  -e NCCL_DEBUG=WARN -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
  -e CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 -e CUDA_COREDUMP_GENERATION_FLAGS=$DUMP_FLAGS \
  -e "CUDA_COREDUMP_FILE=$DUMPS/core_${TAG}_%h_%p" ${EXTRA_DOCKER_ENV:-} \
  "$IMAGE" bash -lc "exec > $LOG 2>&1; cat /opt/Megatron-LM/BUILD_REF /opt/DeepEP/BUILD_REF; \
    timeout -s ABRT --kill-after=60 $RUN_TIMEOUT \
    torchrun --nnodes $NNODES --node-rank $NODE_RANK --nproc-per-node 8 --master-addr $MASTER_ADDR --master-port $PORT \
      /opt/Megatron-LM/pretrain_gpt.py $MLM_ARGS $*" >/dev/null
rc=$(docker wait "$TAG")
docker rm "$TAG" >/dev/null 2>&1 || true
echo "=== $TAG node_rank $NODE_RANK exited rc=$rc"
exit "$rc"

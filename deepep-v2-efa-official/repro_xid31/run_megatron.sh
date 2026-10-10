#!/usr/bin/env bash
# MoE-shaped Megatron-LM dev pretraining on N nodes from the laptop, DeepEP v2 (`deepepv2`
# flex backend) over EFA GIN type 5, mock data. The point is not loss but the REAL call
# pattern of the target workload: combined-1F1B EP overlap (--overlap-moe-expert-parallel-comm),
# async dispatch/combine on the comm stream, and combine-backward = CACHED dispatch.
#
#   NODES="B200-1 B200-2 B200-3 B200-4" TAG=<name> ./run_megatron.sh [extra megatron args]
#
# The model shape comes from shape.env (public DeepSeek-R1-like placeholders) overridden by
# shape.local.env (gitignored) when present. Fixed recipe: MBS 1, GA 2, EP = all GPUs, TP1/PP1,
# 28 DeepEP SMs, sigmoid router with expert bias, permute + router fusion, grouped GEMM, MTP 1.
# NUM_LAYERS (default 12, first NUM_DENSE_LAYERS dense) is cut from the full depth because one EP
# group on 32 GPUs has expert DP = 1, so the full optimizer state does not fit; every
# dispatch/combine call keeps the full shape regardless of depth.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NODES="${NODES:?NODES=\"<leader> <worker> ...\"}"
TAG="${TAG:?TAG=<log name>}"
IMAGE="${IMAGE:-deepep-v2-efa-official:sm100-874779c-mlm1edcc0c}"
PORT="${PORT:-29500}"
NUM_LAYERS="${NUM_LAYERS:-12}"
TRAIN_ITERS="${TRAIN_ITERS:-100000}"
LOGDIR="${LOGDIR:-\$HOME/xid31_runs}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
N=${#NODE_ARR[@]}
SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
MASTER_IP=$($SSH -n "${NODE_ARR[0]}" 'hostname -I | awk "{print \$1}"' 2>/dev/null | tr -d ' \r')
REPO=$($SSH -n "${NODE_ARR[0]}" 'echo $HOME/work/ep-benchmarks-efa/deepep-v2-efa-official' 2>/dev/null | tr -d ' \r')
DUMPS="$REPO/repro_xid31/dumps"
OVERLAP="${OVERLAP:-1}"
# shellcheck disable=SC1091
. "$HERE/megatron_args.sh"

DUMP_FLAGS="skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory"
echo "=== $TAG: Megatron dev on $N nodes ($NODES), master $MASTER_IP:$PORT, image $IMAGE, layers $NUM_LAYERS, overlap=$OVERLAP"
# Detached containers + polling. A foreground `ssh ... docker run` dies with the laptop's ssh
# session, the remote container keeps running, and a driver that trusted the ssh rc started the
# next arm on top of it (2026-10-09 07:08: two trainings on the same GPUs). The container now
# writes its own log into a mounted dir, is named after the tag, and only `docker inspect` -- with
# retries -- decides when the run is over.
for ((i = N - 1; i >= 0; i--)); do
  $SSH -n "${NODE_ARR[$i]}" "mkdir -p $LOGDIR $DUMPS && L=\$(eval echo $LOGDIR) && \
    HCA=\$(ls /sys/class/infiniband | grep -qv '^rdmap' && echo rdmap); \
    docker rm -f $TAG >/dev/null 2>&1; \
    docker run -d --name $TAG --init --gpus all --privileged --network=host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband --device=/dev/gdrdrv \
      -v /sys/class/infiniband:/sys/class/infiniband:ro -v $DUMPS:$DUMPS -v \$L:\$L \
      -e NCCL_GIN_TYPE=5 -e NCCL_SYM_GIN_KERNELS_ENABLE=0 \${HCA:+-e NCCL_IB_HCA=\$HCA} \
      -e NCCL_DEBUG=WARN -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
      -e CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 -e CUDA_COREDUMP_GENERATION_FLAGS=$DUMP_FLAGS \
      -e CUDA_COREDUMP_FILE=$DUMPS/core_${TAG}_%h_%p ${EXTRA_DOCKER_ENV:-} \
      $IMAGE bash -lc \"exec > \$L/$TAG.node$((i + 1)).log 2>&1; cat /opt/Megatron-LM/BUILD_REF /opt/DeepEP/BUILD_REF; \
        timeout -s ABRT --kill-after=60 ${RUN_TIMEOUT:-10800} \
        torchrun --nnodes $N --node-rank $i --nproc-per-node 8 --master-addr $MASTER_IP --master-port $PORT \
          /opt/Megatron-LM/pretrain_gpt.py $MLM_ARGS $*\" >/dev/null" || { echo "!! launch failed on ${NODE_ARR[$i]}"; exit 1; }
  sleep 1
done
# Done when no node reports the container running; an ssh failure counts as "still running".
while :; do
  sleep 30
  running=0
  for n in "${NODE_ARR[@]}"; do
    st=$($SSH -n "$n" "docker inspect -f '{{.State.Running}}' $TAG 2>/dev/null || echo gone" 2>/dev/null | tr -d ' \r')
    case "$st" in false|gone) ;; *) running=1 ;; esac
  done
  [ "$running" = 1 ] || break
done
for n in "${NODE_ARR[@]}"; do $SSH -n "$n" "docker rm $TAG >/dev/null 2>&1" 2>/dev/null; done
echo "=== $TAG done"
exit 0

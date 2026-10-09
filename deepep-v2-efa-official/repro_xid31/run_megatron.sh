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
# shellcheck disable=SC1091
. "$HERE/shape.env"; [ ! -f "$HERE/shape.local.env" ] || . "$HERE/shape.local.env"
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
NDENSE=$NUM_DENSE_LAYERS; NMOE=$((NUM_LAYERS - NDENSE))
GBS=$((N * 8 * 2))   # DP = all GPUs (TP1/PP1), MBS 1, GA 2 -- the target GA
# OVERLAP=0 drops --overlap-moe-expert-parallel-comm (combined-1F1B EP A2A overlap): the A/B that
# separates "the overlap schedule / comm-stream lifetimes" from "DeepEP on its own".
OVERLAP="${OVERLAP:-1}"
OVERLAP_ARG=""; [ "$OVERLAP" = 0 ] || OVERLAP_ARG="--overlap-moe-expert-parallel-comm"

MLM_ARGS="--num-layers $NUM_LAYERS --hidden-size $HIDDEN --ffn-hidden-size $FFN_HIDDEN --num-attention-heads $NUM_HEADS \
--seq-length $SEQ_LEN --max-position-embeddings $SEQ_LEN --micro-batch-size 1 --global-batch-size $GBS \
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size $((N * 8)) \
--moe-layer-freq '([0]*$NDENSE+[1]*$NMOE)' --num-experts $NUM_EXPERTS --moe-router-topk $TOPK --moe-ffn-hidden-size $MOE_FFN_HIDDEN \
--moe-shared-expert-intermediate-size $SHARED_EXPERT_FFN --moe-router-score-function sigmoid \
--moe-router-topk-scaling-factor $TOPK_SCALING --moe-router-enable-expert-bias --moe-router-bias-update-rate 0.001 \
--moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 0.0001 --moe-router-dtype fp32 \
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepepv2 --moe-deepep-num-sms 28 \
--moe-grouped-gemm --moe-permute-fusion --moe-router-fusion $OVERLAP_ARG \
--mtp-num-layers 1 \
--swiglu --normalization RMSNorm --position-embedding-type rope --untie-embeddings-and-output-weights \
--disable-bias-linear --hidden-dropout 0.0 --attention-dropout 0.0 --transformer-impl transformer_engine \
--bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
--lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 10 --weight-decay 0.1 --clip-grad 1.0 \
--train-iters $TRAIN_ITERS --mock-data --tokenizer-type NullTokenizer --vocab-size $VOCAB \
--log-interval 1 --eval-iters 0 --eval-interval 100000000 --seed 1234"

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

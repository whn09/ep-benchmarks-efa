#!/usr/bin/env bash
# Launch pplx-garden bench_all_to_all.py inside pplx-garden-efa:dev.
# Args:
#   $1  NODE_RANK   (0 = leader, 1 = worker)
#   $2  MASTER_IP   (private IP of leader)
#   $3+ extra args forwarded to bench_all_to_all
#
# Default config: 2 nodes × 8 GPU = 16 ranks, 2 nets/GPU,
# decode shape (128 tokens, 256 experts, top-8, hidden 7168, FP8).
set -eux

NODE_RANK="${1:?node rank}"
MASTER_IP="${2:?master ip}"
shift 2

NUM_NODES="${NUM_NODES:-2}"
WORLD_SIZE=$((NUM_NODES * 8))
MASTER_PORT="${MASTER_PORT:-29500}"
NETS_PER_GPU="${NETS_PER_GPU:-2}"

VERBS=()
for u in /dev/infiniband/uverbs*; do
  [[ -e "$u" ]] && VERBS+=(--device="$u")
done

docker run --rm \
  --gpus=all --network=host \
  --shm-size=32g --ulimit=memlock=-1 --ulimit=stack=67108864 \
  "${VERBS[@]}" \
  --device=/dev/gdrdrv \
  --cap-add=IPC_LOCK --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
  --security-opt=seccomp=unconfined \
  pplx-garden-efa:dev \
  bash -lc "cd /app && python3 -m benchmarks.bench_all_to_all \
    --world-size ${WORLD_SIZE} \
    --nets-per-gpu ${NETS_PER_GPU} \
    --init-method=tcp://${MASTER_IP}:${MASTER_PORT} \
    --node-rank=${NODE_RANK} \
    --nvlink=8 $*"

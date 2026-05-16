#!/usr/bin/env bash
# Launch UCCL-EP bench/test_low_latency_pplx.py — same UCCL LL workload
# but measured pplx-garden-style (warmup + repeats, p50/p99 statistics)
# so the numbers are directly comparable to pplx-garden's bench output.
#
# Args:
#   $1 NODE_RANK
#   $2 MASTER_IP
#   $3+ extra args forwarded to test_low_latency_pplx.py
set -eux

NODE_RANK="${1:?node rank}"
MASTER_IP="${2:?master ip}"
shift 2

NNODES=2
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"

docker run --rm \
  --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband --device=/dev/gdrdrv \
  -e FI_PROVIDER=efa \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e NCCL_DEBUG=WARN \
  uccl-ep-efa:dev \
  bash -lc "cd /opt/uccl/ep/bench && \
    torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
             --node_rank=${NODE_RANK} \
             --master_addr=${MASTER_IP} --master_port=${MASTER_PORT} \
    test_low_latency_pplx.py --num-tokens=128 --hidden=7168 \
                             --num-topk=8 --num-experts=288 $*"

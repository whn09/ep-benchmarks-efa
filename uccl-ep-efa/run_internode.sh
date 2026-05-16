#!/usr/bin/env bash
# Launch UCCL-EP tests/test_internode.py via torchrun across 2 nodes.
# Args:
#   $1 NODE_RANK   (0 = leader, 1 = worker)
#   $2 MASTER_IP   (private IP of leader)
#   $3+ extra args forwarded to test_internode.py
set -eux

NODE_RANK="${1:?node rank}"
MASTER_IP="${2:?master ip}"
shift 2

NNODES=2
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

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
    test_internode.py --num-tokens=4096 --hidden=7168 --num-topk=8 \
                      --num-experts=288 --test-ll-compatibility $*"

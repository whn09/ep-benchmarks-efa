#!/usr/bin/env bash
# Launch DeepEP V2 tests/elastic/test_ep.py inside deepep-v2-efa:dev.
# Args:
#   $1  NODE_RANK   (0 = leader, 1 = worker)
#   $2  MASTER_IP   (private IP of leader)
#   $3+ extra args forwarded to test_ep.py
set -eux

NODE_RANK="${1:?node rank}"
MASTER_IP="${2:?master ip}"
shift 2

WORLD_SIZE="${WORLD_SIZE:-2}"     # number of NODES, per init_dist convention
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

docker run --rm \
  --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband --device=/dev/gdrdrv \
  -e MASTER_ADDR="${MASTER_IP}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e RANK="${NODE_RANK}" \
  -e FI_PROVIDER=efa \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
  ${EP_NIC_NAME:+-e EP_NIC_NAME=$EP_NIC_NAME} \
  deepep-v2-efa:dev \
  bash -lc "cd /opt/deepep && python tests/elastic/test_ep.py \
    --num-processes ${NUM_PROCESSES} $*"

#!/usr/bin/env bash
# Launch DeepEP V1 tests/test_low_latency.py inside the deepep-v1-efa:dev image.
# Args:
#   $1  NODE_RANK  (0 = leader, 1 = worker)
#   $2  MASTER_IP  (private IP of leader)
#   $3+ extra args forwarded to test_low_latency.py
set -eux

NODE_RANK="${1:?node rank}"
MASTER_IP="${2:?master ip}"
shift 2

WORLD_SIZE=2
MASTER_PORT="${MASTER_PORT:-29501}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

docker run --rm \
  --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband --device=/dev/gdrdrv \
  -e MASTER_ADDR="${MASTER_IP}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e RANK="${NODE_RANK}" \
  -e NVSHMEM_REMOTE_TRANSPORT=libfabric \
  -e NVSHMEM_LIBFABRIC_PROVIDER=efa \
  -e FI_PROVIDER=efa \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e NCCL_DEBUG=WARN \
  -e NVSHMEM_DEBUG=WARN \
  ${NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE:+-e NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE=$NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE} \
  deepep-v1-efa:dev \
  bash -lc "cd /opt/deepep && python tests/test_low_latency.py --num-processes ${NUM_PROCESSES} $*"

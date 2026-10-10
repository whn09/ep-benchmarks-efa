#!/usr/bin/env bash
# Run tests/ep/test_ep.py from the V2.5 port image on THIS host. Same command on every node,
# only NODE_RANK differs (test_ep.py spawns 8 ranks itself; RANK = node index, WORLD_SIZE = nodes).
#
#   NODE_RANK=0 MASTER_ADDR=<node 0 private ip> WORLD_SIZE=4 [IMAGE=...] [MASTER_PORT=8371] \
#     ./run_test_ep.sh [extra test_ep.py args]
#
# Defaults reproduce the matched comparison in results/: 8192 tokens, hidden 7168, top-8,
# 256 experts, 24 SMs, first case only, --defer-epilogue=0 (=> async_with_compute_stream=0,
# the same first case as amazon-contributing main's tests/elastic/test_ep.py).
set -euo pipefail
NODE_RANK="${NODE_RANK:?}"; MASTER_ADDR="${MASTER_ADDR:?}"; WORLD_SIZE="${WORLD_SIZE:?number of nodes}"
IMAGE="${IMAGE:-deepep-v25-efa:sm100-f501cd1}"
MASTER_PORT="${MASTER_PORT:-8371}"
NIC=$(ls /sys/class/infiniband | grep -m1 rdmap)
HCA=""; ls /sys/class/infiniband | grep -qv '^rdmap' && HCA=rdmap
docker run --rm --init --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband --device=/dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT" -e WORLD_SIZE="$WORLD_SIZE" -e RANK="$NODE_RANK" \
  -e EP_NIC_NAME="$NIC" ${HCA:+-e NCCL_IB_HCA=$HCA} -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e PYTHONUNBUFFERED=1 \
  "$IMAGE" bash -c "cat /opt/DeepEP/BUILD_REF /opt/aon/BUILD_REF; \
    python3 -u /opt/DeepEP/tests/ep/test_ep.py --num-processes=8 \
      --num-tokens=8192 --hidden=7168 --num-topk=8 --num-experts=256 --num-sms=24 --allow-hybrid-mode=1 \
      --test-first-only --defer-epilogue=0 $*"

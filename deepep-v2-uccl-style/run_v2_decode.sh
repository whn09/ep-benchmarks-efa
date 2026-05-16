#!/usr/bin/env bash
# DeepEP V2 decode / low-latency-shape benchmark with UCCL-EP-comparable
# params (num_experts=288, num_topk=8, num_tokens=128).
#
# DeepEP V2 has no separate LL kernel; its `ElasticBuffer.dispatch/combine`
# is a unified path. We pick knobs that resemble a low-latency profile:
#   --prefer-overlap-with-compute 0
#   --allow-multiple-reduction 0
#   --do-cpu-sync 1
# Numbers are NOT directly apples-to-apples with V1/UCCL `test_low_latency.py`,
# but they're the closest V2 can get and mirror the test config used by
# the deepep-efa-coexistence repo.
set -eux

NODE_RANK="${1:?usage: $0 NODE_RANK MASTER_IP [NUM_NODES]}"
MASTER_IP="${2:?usage: $0 NODE_RANK MASTER_IP [NUM_NODES]}"
NUM_NODES="${3:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_SMS="${NUM_SMS:-0}"
NUM_QPS="${NUM_QPS:-0}"
EP_NIC_NAME="${EP_NIC_NAME:-rdmap85s0}"

docker run --rm \
  --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband --device=/dev/gdrdrv \
  -e MASTER_ADDR="${MASTER_IP}" -e MASTER_PORT="${MASTER_PORT}" \
  -e WORLD_SIZE="${NUM_NODES}" -e RANK="${NODE_RANK}" \
  -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
  -e EP_NIC_NAME="${EP_NIC_NAME}" \
  ${EP_RDMA_GBS:+-e EP_RDMA_GBS=$EP_RDMA_GBS} \
  deepep-v2-efa:dev \
  bash -lc "cd /opt/deepep && python tests/elastic/test_ep.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 \
    --prefer-overlap-with-compute 0 --allow-hybrid-mode 1 \
    --allow-multiple-reduction 0 --do-cpu-sync 1 \
    --num-sms ${NUM_SMS} --num-qps ${NUM_QPS} \
    --test-first-only --skip-check"

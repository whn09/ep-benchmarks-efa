#!/bin/bash
# usage: run_det.sh <tag> <port> <extra env>
TAG=$1; PORT=$2; XE="$3"
I=${4:-deepep-v2-efa-official:sm103-stackc105b59x3c737dc-nb}
cd ~/work/ep-benchmarks-efa/deepep-v2-efa-official
COMMON="IMAGE=$I WORLD_SIZE=2 NUM_PROCESSES=8 TOKENS=128 NUM_SMS=8 MASTER_PORT=$PORT NCCL_DEBUG=WARN TEST_FIRST_ONLY=0 EXTRA_MOUNT=/home/ubuntu/detscript TEST_SCRIPT=/home/ubuntu/detscript/ep_determinism.py RUN_TIMEOUT=${6:-1500}"
ARGS="--num-topk=8 --duration=${5:-300} --deterministic=0 --seed=1"
ssh -n B300-2 "mkdir -p ~/detscript && cp ~/ep_determinism.py ~/detscript/ && cd ~/work/ep-benchmarks-efa/deepep-v2-efa-official && $COMMON EXTRA_ENV=\"$XE\" bash run_test_ep.sh 1 172.31.14.167 $ARGS > ~/det_$TAG.node2.log 2>&1" &
sleep 2; mkdir -p ~/detscript && cp ~/ep_determinism.py ~/detscript/
env $COMMON EXTRA_ENV="$XE" bash run_test_ep.sh 0 172.31.14.167 $ARGS > ~/det_$TAG.node1.log 2>&1
wait
echo "== $TAG"; cat ~/det_$TAG.node1.log | grep -E "^EP_DET " | sed "s/ sha .*//" ; ssh B300-2 "grep -E \"^EP_DET \" ~/det_$TAG.node2.log | sed \"s/ sha .*//\""

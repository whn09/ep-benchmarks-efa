#!/usr/bin/env bash
# Does the customer's combine register-reallocation deadlock reproduce on p5en / sm_90?
#
#   NODES="<leader> <worker>" [IMAGE=...] ./probe_combine_deadlock.sh
#
# WHY. The reported deadlock (8 x p6-b300, BF16 training) is in
# hybrid_combine_unordered.cuh's setmaxnreg admission: the unordered kernel adds a
# 17th (proxy) warp, so the CTA is 544 threads, and upstream's forward-warp target of
# 256 - 40 = 216 registers/thread silently encodes 128 static registers/thread, which
# 544 threads cannot have (65536 / 544 = 120.5). The feasibility condition is
# T <= 2S - F  (T = forward target, S = compiled static regs/thread, F = the scale-up
# floor, 40), and it is INDEPENDENT of the channel count -- the proxy warp's holding
# cancels exactly. So the bug is not b300-specific in principle: sm_90 has the same
# 65536-register file per SM. What decides whether a given build hits it is (a) how
# many channels per SM the buffer lands on, because the gate is
# `kAdjustRegisters = (kNumChannelsPerSM == 4 or == 8) and not kUseExpandedLayout`
# and C = 5/6/7 bypasses the mechanism entirely, and (b) what ptxas actually compiled
# S to on this arch.
#
# Both are observable without a training run, which is what the DeepEP maintainers
# asked for. Neither is observable from the campaign logs, because EP_BUFFER_DEBUG
# printf()s from inside dispatch's host polling loop -- i.e. inside the timed region
# -- so run_campaign.sh never sets it. That is why this is a separate script writing
# to a separate log dir: nothing it produces may pool with a timed cell.
#
# DO NOT RUN THIS WHILE A CAMPAIGN IS LIVE. It puts a second tenant on the GPUs and
# on the EFA devices; a probe has broken a running benchmark here before. The script
# refuses if any GPU on either node already has a compute process.
#
# WHAT EACH CELL IS FOR.
#   A  ovlp=0, 12 SM -- the campaign's working point. Prints the channels/SM every
#      published number was measured at, i.e. whether our own numbers were ever in
#      the gated shape at all.
#   B  ovlp=1, 12 SM -- ovlp=1 is what the customer's training used (it is the
#      library default; the benchmark's default is 0, which is the divergence that
#      kept this out of our campaigns). At ovlp=0 buffer.hpp clamps channels/SM to
#      <= 4; only =1 can reach 8.
#   C  ovlp=1, 4 SM -- 4 SM is DeepEP's OWN automatic floor (elastic.py:854,
#      `max(4, ...)`), which is where the customer landed, and the shape that
#      reportedly deadlocks. This is the cell expected to hang.
#   D  ovlp=1, 4 SM, with EP_JIT_PTXAS_VERBOSE -- the static register count S for
#      this arch's instantiation, which is what turns T <= 2S - F from a model into
#      a number. Run even if C hangs: the compile happens before the launch.
#
# A hang self-terminates: test_ep.py's --num-gpu-timeout-secs / --num-cpu-timeout-secs
# default to 100, so a deadlocked cell aborts in ~100 s rather than wedging the node.
set -euo pipefail
cd "$(dirname "$0")"

NODES="${NODES:?NODES=\"<leader> <worker>\" (2 nodes is enough: the guard is num_scaleout_ranks > 1)}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
NNODES=${#NODE_ARR[@]}
LEADER=${NODE_ARR[0]}
IMAGE="${IMAGE:-deepep-v2-efa-official:sm90-54fffef}"
LOGDIR="${LOGDIR:-\$HOME/probe_deadlock_$(date -u +%Y%m%d)}"
REPO_DIR="${REPO_DIR:-\$HOME/deepep-v2-efa-official}"
PORT_BASE="${PORT_BASE:-8990}"
SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

[ "$NNODES" -ge 2 ] || { echo "need >= 2 nodes: the combine path under test is" \
  "guarded by num_scaleout_ranks > 1 (combine.hpp:174)" >&2; exit 1; }

MASTER_IP="${MASTER_IP:-$($SSH -n "$LEADER" \
  "ip -4 -o addr show | awk '/172\.31\./{split(\$4,a,\"/\"); print a[1]; exit}'")}"
echo "=== MASTER_IP=$MASTER_IP  IMAGE=$IMAGE  nodes=$NNODES"

# The probe must not be the second tenant on a busy GPU.
for h in "${NODE_ARR[@]}"; do
  n=$($SSH -n "$h" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l')
  n=$(echo "$n" | tr -d ' \r')
  [ "$n" = 0 ] || { echo "$h has $n compute process(es) -- refusing. A probe has" \
    "broken a live benchmark here before." >&2; exit 1; }
  ref=$($SSH -n "$h" "docker run --rm --entrypoint cat $IMAGE /opt/DeepEP/BUILD_REF")
  echo "    $h  idle, $IMAGE = $ref"
done

port=$PORT_BASE
run() {   # name  ovlp  sms  extra_env
  local name=$1 ovlp=$2 sms=$3 extra=$4
  port=$((port + 1))
  local tag="probe_${name}_${NNODES}N_${sms}sm_ovlp${ovlp}_dbg"
  echo
  echo "=== $tag  (port $port)  EXTRA_ENV='$extra'"
  local env="IMAGE=$IMAGE WORLD_SIZE=$NNODES NUM_PROCESSES=8 TOKENS=128 \
NUM_SMS=$sms MASTER_PORT=$port NCCL_DEBUG=WARN TEST_FIRST_ONLY=1 \
PREFER_OVERLAP=$ovlp EXTRA_ENV='$extra'"
  local pids=()
  for ((i = NNODES - 1; i >= 0; i--)); do
    $SSH -n "${NODE_ARR[$i]}" \
      "mkdir -p $LOGDIR && cd $REPO_DIR && $env bash run_test_ep.sh $i $MASTER_IP \
       > $LOGDIR/$tag.node$((i + 1)).log 2>&1" &
    pids+=($!)
    sleep 1
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "    rc=$rc  (a nonzero rc here is a RESULT, not a failure of the probe)"
  # The three lines that answer the question, pulled straight out of the leader log.
  $SSH -n "$LEADER" "grep -m3 -E 'channels per SM|registers|Deadlock|deadlock|timeout|Timeout' \
    $LOGDIR/$tag.node1.log 2>/dev/null | sed 's/^/    /'" || true
  sleep 20
}

run chan_ovlp0_12sm 0 12 "EP_BUFFER_DEBUG=1"
run chan_ovlp1_12sm 1 12 "EP_BUFFER_DEBUG=1"
run chan_ovlp1_4sm  1 4  "EP_BUFFER_DEBUG=1"
run ptxas_ovlp1_4sm 1 4  "EP_BUFFER_DEBUG=1 EP_JIT_PTXAS_VERBOSE=1"

echo
echo "=== fetch:"
for ((i = 0; i < NNODES; i++)); do
  echo "    scp '${NODE_ARR[$i]}:${LOGDIR}/*.node$((i + 1)).log' ./probe_logs/"
done
echo "=== read off: 'Elastic buffer uses N channels per SM' (buffer.hpp:956) and"
echo "    ptxas' 'Used N registers' for the hybrid_unordered_combine instantiation."

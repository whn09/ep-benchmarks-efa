#!/usr/bin/env bash
# Launch tests/elastic/test_ep.py from amazon-contributing/DeepEP inside
# deepep-v2-efa-official:dev on bare EC2 (no Slurm / enroot).
#
# Args:
#   $1  NODE_RANK   0 = leader, 1 = worker.  test_ep.py is NOT torchrun:
#                   it mp.spawn()s NUM_PROCESSES local ranks itself, so
#                   RANK is the NODE index and WORLD_SIZE the NODE count.
#   $2  MASTER_IP   private IP of the leader
#   $3+ extra args forwarded to test_ep.py
#
# Env:
#   TOKENS=8192      8192 = prefill (report bandwidth), 128 = decode (report latency).
#   NUM_SMS=12       MUST be explicit on EFA: --num-sms 0 routes through ibstat,
#                    which cannot see EFA devices, and dies with ZeroDivisionError
#                    in get_theoretical_num_sms(). Also NOT freely tunable -- it
#                    changes the allocated QP count non-monotonically and a value
#                    landing on num_qps < num_ranks hangs. Known-good: 12 on 2
#                    nodes, 6 on 4 nodes.  See docs/runbook_zh.md 4.2.
#   MASTER_PORT      use a DIFFERENT port per repetition: a killed run leaves a
#                    TCPStore listener behind and the next one wedges in rendezvous.
#   EP_MIN_TOKENS_PER_PART / EP_NUM_SUB_PARTS
#                    decode part-geometry knobs. Need amazon-contributing/DeepEP
#                    PR #1 + #2; without them decode dispatch is ~2.2x slower.
#   IGNORE_LOCAL=1   pass --ignore-local-traffic, so the reported scale-out GB/s
#                    is a wire rate. WITHOUT it, SO includes intra-node traffic
#                    and can exceed p5en's 50 GB/s per-GPU wire.
#   EP_BUFFER_DEBUG=1
#                    prints the GIN layout line -- but it ALSO printf()s a
#                    per-call "CPU side received count" from inside dispatch's
#                    host polling loop (`csrc/elastic/buffer.hpp:1151`), i.e.
#                    inside the timed region. Its cost is not separable from
#                    run-to-run spread (2x p5en, 12 SM, 128 tok, all-rank means:
#                    off 359.2 / 371.1 us, on 327.2 / 373.5 us), so no number is
#                    claimed -- but it is asymmetric across launchers, so an arm
#                    that sets it is not comparable to one that does not. Use it
#                    to confirm the layout, then turn it OFF for anything you publish.
#   IMAGE, WORLD_SIZE, NUM_PROCESSES, EP_NIC_NAME, NCCL_DEBUG, EXTRA_ENV
set -euo pipefail

NODE_RANK="${1:?node rank (0=leader, 1=worker)}"
MASTER_IP="${2:?master ip (leader private ip)}"
shift 2

IMAGE="${IMAGE:-deepep-v2-efa-official:dev}"
WORLD_SIZE="${WORLD_SIZE:-2}"          # number of NODES, per init_dist convention
MASTER_PORT="${MASTER_PORT:-8371}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_SMS="${NUM_SMS:-12}"
TOKENS="${TOKENS:-8192}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# --- preflight -------------------------------------------------------------
# Ranks leaked by a previous run pin ~48 GB/GPU, and the NEXT run then reports
# silently ~2x-inflated latency while still exiting rc=0 with complete output.
# rc is not a health check; refuse to start rather than publish a bogus number.
if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  if [ "${used:-0}" -gt 1024 ]; then
    echo "REFUSING TO START: ${used} MiB already in use on the busiest GPU." >&2
    echo "  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv" >&2
    echo "Set ALLOW_BUSY_GPU=1 to override." >&2
    [ "${ALLOW_BUSY_GPU:-0}" = "1" ] || exit 1
  fi
fi

for dev in /dev/infiniband /dev/gdrdrv; do
  [ -e "$dev" ] || { echo "MISSING $dev -- see README 'Host prerequisites'." >&2; exit 1; }
done

# EP_NIC_NAME is PCI-derived so it differs per instance type (rdmap85s0 on p5en).
# Pick the fastest rdmap* on THIS host instead of trusting the name in the image.
if [ -z "${EP_NIC_NAME:-}" ]; then
  best=""; best_rate=0
  for n in /sys/class/infiniband/rdmap*; do
    [ -d "$n" ] || continue
    for p in "$n"/ports/*/rate; do
      [ -r "$p" ] || continue
      r=$(awk '{print $1}' "$p")
      if [ "${r:-0}" -gt "$best_rate" ]; then best_rate=$r; best=$(basename "$n"); fi
    done
  done
  [ -n "$best" ] && EP_NIC_NAME="$best" && echo "=== EP_NIC_NAME=$best (auto, ${best_rate} Gb/s) ==="
fi

# EXTRA_ENV="NAME=VALUE NAME=VALUE" -- one-off env for A/B arms (e.g. the pre-1.50.0
# route-B vars OFI_NCCL_GIN_STRONG_SIGNAL=1 / NCCL_GIN_TYPE=5). Deliberately NOT a
# default: with the packaged 1.50.0 stack GDAKI loads without any of them.
EXTRA_ENV_ARGS=""
for kv in ${EXTRA_ENV:-}; do EXTRA_ENV_ARGS="$EXTRA_ENV_ARGS -e $kv"; done

# Stamp what code is actually in the image into every log. The image tag is a
# name someone chose; BUILD_REF is what `git rev-parse HEAD` said at build time.
# Without this, a rebuilt-but-same-tag image produces numbers you cannot attribute.
echo "=== IMAGE=${IMAGE}  DeepEP=$(docker run --rm --entrypoint cat "${IMAGE}" \
  /opt/DeepEP/BUILD_REF 2>/dev/null || echo 'BUILD_REF absent -- image predates SHA stamping') ==="

set -x
docker run --rm \
  --gpus all --privileged --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband --device=/dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -e MASTER_ADDR="${MASTER_IP}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e RANK="${NODE_RANK}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  ${EP_NIC_NAME:+-e EP_NIC_NAME=$EP_NIC_NAME} \
  ${EP_BUFFER_DEBUG:+-e EP_BUFFER_DEBUG=$EP_BUFFER_DEBUG} \
  ${EP_MIN_TOKENS_PER_PART:+-e EP_MIN_TOKENS_PER_PART=$EP_MIN_TOKENS_PER_PART} \
  ${EP_NUM_SUB_PARTS:+-e EP_NUM_SUB_PARTS=$EP_NUM_SUB_PARTS} \
  ${EP_JIT_CACHE_DIR:+-e EP_JIT_CACHE_DIR=$EP_JIT_CACHE_DIR} \
  ${EXTRA_ENV_ARGS} \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONFAULTHANDLER=1 \
  "${IMAGE}" \
  bash -lc "timeout -s ABRT --kill-after=60 ${RUN_TIMEOUT:-1800} \
    python3 -u /opt/DeepEP/tests/elastic/test_ep.py \
    --num-processes=${NUM_PROCESSES} --num-tokens=${TOKENS} \
    --hidden=7168 --num-topk=8 --num-experts=256 \
    --num-sms=${NUM_SMS} --allow-hybrid-mode=1 \
    --prefer-overlap-with-compute=0 --test-first-only \
    ${IGNORE_LOCAL:+--ignore-local-traffic} $*"
# python3 -u / PYTHONUNBUFFERED: without them stdout is block-buffered into the
# redirected log, so a run that hangs in NCCL/GIN init shows an EMPTY log.
#
# ABRT, not INT: a rank parked inside ncclCommInitRank in C never regains the
# interpreter loop, so it cannot act on SIGINT. --kill-after=60 guarantees a
# wedged rank stops pinning GPU memory instead of tripping the preflight above.

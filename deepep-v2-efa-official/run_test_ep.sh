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
#   NUM_SMS=12       Pass it explicitly. At the image's pinned DeepEP (8e7b42e)
#                    --num-sms 0 does NOT crash -- get_rdma_gbs reads
#                    /sys/class/infiniband/<nic>/ports/*/rate first and only falls
#                    back to ibstat (the ZeroDivisionError in docs/runbook_zh.md
#                    4.2 is the older ec623f3 pin). But it reports ONE device's
#                    rate: correct on p5en (1 EFA per GPU = 50 GB/s), half the
#                    truth on p6-b300 (2 per GPU = 100 GB/s). It is also NOT freely
#                    tunable -- it changes the allocated QP count non-monotonically
#                    and a value landing on num_qps < num_ranks hangs. Known-good:
#                    12 on 2 p5en nodes, 6 on 4; 24 on 2 p6-b300 nodes.
#   MASTER_PORT      use a DIFFERENT port per repetition: a killed run leaves a
#                    TCPStore listener behind and the next one wedges in rendezvous.
#   EP_MIN_TOKENS_PER_PART / EP_NUM_SUB_PARTS / EP_MIN_SUB_TOKENS /
#   EP_SM100_MIN_SUB_TOKENS
#                    decode part-geometry knobs, all four forwarded to the JIT by
#                    amazon-contributing/DeepEP PR #1. Without #1 they are silently
#                    inert; without #2 decode dispatch is ~1.5x (p5en) to ~1.6x
#                    (b300) slower. Reaching them by editing the header instead of
#                    the env is unsafe: the JIT cache key hashes `flags` but NOT
#                    included header content, so an env-free header patch can serve
#                    a stale cubin.  See README 'Rules that decide ...' rule 1.
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
#   NCCL_IB_HCA      auto-set to `rdmap` when this host also has non-EFA ibverbs
#                    devices (p6-b300 does); without it NCCL creates only 2 GIN
#                    GDAKI NICs and rank 4+ dies. See the block below.
#   TEST_FIRST_ONLY=1  0 runs the whole ep-mode product (hours), not just BF16.
#   IMAGE, WORLD_SIZE, NUM_PROCESSES, EP_NIC_NAME, NCCL_DEBUG, EXTRA_ENV
#
# Hardware note: the image must be built for THIS GPU's arch, and on p6-b300 that
# also means a newer CUDA base -- `--build-arg TORCH_CUDA_ARCH_LIST=10.3
# --build-arg CUDA_VERSION=13.3.1`. See the Dockerfile header.
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
# --test-first-only stops after the FIRST entry of enumerate_ep_modes(), which at
# this pin (test_ep.py:33-41) is do_handle_copy=1, expert_alignment=128,
# use_fp8_dispatch=1, num_bias=0, with_previous_event=0, async_with_compute_stream=0,
# allocate_on_comm_stream=0. So every number this repo publishes is *FP8 dispatch at
# alignment 128* -- there is no flag to select BF16 dispatch on its own, and
# TEST_FIRST_ONLY=0 runs the whole 2x2x2x3x... product (hours), not just BF16.
TEST_FIRST_ONLY="${TEST_FIRST_ONLY:-1}"
# "0" must disable it, so test the value -- ${VAR:+...} would keep passing the flag.
# `if`, not `[ ] && VAR=`: under `set -e` a false test as the last command of the
# line exits the script.
FIRST_ONLY_ARG=""
if [ "$TEST_FIRST_ONLY" != "0" ]; then FIRST_ONLY_ARG="--test-first-only"; fi
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

# The image must be built for THIS GPU. A Hopper cubin on Blackwell (or the
# reverse) does not fail here, it fails inside the test far from its cause, and on
# sm_103 a CUDA-13.0.2-based image fails later still -- at the first dispatch,
# in ptxas. Both are stamped into the image, so compare them up front.
# ALLOW_ARCH_MISMATCH=1 to override (e.g. an image predating the stamping).
if command -v nvidia-smi >/dev/null 2>&1; then
  host_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  img_arch=$(docker run --rm --entrypoint printenv "${IMAGE}" EP_BUILD_ARCH 2>/dev/null || true)
  img_cuda=$(docker run --rm --entrypoint printenv "${IMAGE}" EP_BUILD_CUDA 2>/dev/null || true)
  if [ -z "$img_arch" ]; then
    echo "WARNING: ${IMAGE} has no EP_BUILD_ARCH stamp (built before it existed);" \
         "arch is unchecked." >&2
  elif [ "$img_arch" != "$host_cc" ]; then
    echo "REFUSING TO START: image built for arch '$img_arch', this host is '$host_cc'." >&2
    echo "  Rebuild: ./build_image.sh   (it derives both build args from the GPU)" >&2
    [ "${ALLOW_ARCH_MISMATCH:-0}" = "1" ] || exit 1
  fi
  # 13.0.x ptxas cannot assemble the >= sm_100 PTX in ptx.cuh. Only the major.minor
  # matters, and the JIT uses the base image's nvcc, not torch's.
  case "$host_cc" in
    10.*) case "$img_cuda" in
            13.0*|12.*|"") echo "REFUSING TO START: sm_10x needs a CUDA >= 13.3.x base," \
                                "image has '${img_cuda:-unknown}'. It would build fine and" \
                                "die in ptxas at the first dispatch." >&2
                           [ "${ALLOW_ARCH_MISMATCH:-0}" = "1" ] || exit 1 ;;
          esac ;;
  esac
fi

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
  # Not `[ -n "$best" ] && ...`: under `set -e` that exits the script with no
  # message on a host where no rdmap* device is visible.
  if [ -n "$best" ]; then
    EP_NIC_NAME="$best"
    echo "=== EP_NIC_NAME=$best (auto, ${best_rate} Gb/s) ==="
  else
    echo "WARNING: no readable /sys/class/infiniband/rdmap*/ports/*/rate;" \
         "falling back to the image's EP_NIC_NAME." >&2
  fi
fi

# NCCL picks the GIN GDAKI NICs itself, and on a host whose ibverbs device list is
# MIXED it under-creates them: p6-b300 shows 18 devices (16 rdmap* EFA + the two
# non-EFA ibp198s0f0 / ibp199s0f0, which fail ce_probe with errno 95), and NCCL
# built only 2 GDAKI NICs -- so rank 4 asking for its own NIC dies with
#   NET/IB : Requested properties for GIN GDAKI NIC 4, only 2 GIN GDAKI NICs
#   ...  RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:185): 5
# One GDAKI NIC per local rank (8/node) is what ElasticBuffer needs. Restricting
# NCCL to the EFA devices by prefix fixes it; the log then reads
#   NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0..7 'rdmap*'
# p5en has 16 pure rdmap* devices, so it never trips this and needs no default.
# Only inject when a non-rdmap device is actually present -- a blanket default
# would silently mask a host where the EFA devices are named something else.
if [ -z "${NCCL_IB_HCA:-}" ]; then
  for n in /sys/class/infiniband/*; do
    [ -d "$n" ] || continue
    case "$(basename "$n")" in
      rdmap*) ;;
      *) NCCL_IB_HCA=rdmap
         echo "=== NCCL_IB_HCA=rdmap (auto: non-EFA ibverbs device $(basename "$n") present) ==="
         break;;
    esac
  done
fi

# EXTRA_ENV="NAME=VALUE NAME=VALUE" -- one-off env for A/B arms. It is how the
#   NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0
# pair is passed, and you almost always want that pair: 1.50.0 LOADS the GDAKI
# plugin either way (`Loaded gin plugin Libfabric_GDAKI (v14)` prints in both arms)
# but NCCL RUNS the type-2 Libfabric proxy backend unless type 5 is forced --
# ~9% less prefill and 2.2-5.4x the decode latency. Set NCCL_GIN_TYPE=5 alone and
# it crashes; both, or neither. Not baked in as a default so that the default arm
# stays a measurable control. run_campaign.sh passes it and stamps `_gin5` into
# the log name; a run without it gets `_type2`, so the two can never be pooled.
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
  ${NCCL_IB_HCA:+-e NCCL_IB_HCA=$NCCL_IB_HCA} \
  ${EP_BUFFER_DEBUG:+-e EP_BUFFER_DEBUG=$EP_BUFFER_DEBUG} \
  ${EP_MIN_TOKENS_PER_PART:+-e EP_MIN_TOKENS_PER_PART=$EP_MIN_TOKENS_PER_PART} \
  ${EP_NUM_SUB_PARTS:+-e EP_NUM_SUB_PARTS=$EP_NUM_SUB_PARTS} \
  ${EP_MIN_SUB_TOKENS:+-e EP_MIN_SUB_TOKENS=$EP_MIN_SUB_TOKENS} \
  ${EP_SM100_MIN_SUB_TOKENS:+-e EP_SM100_MIN_SUB_TOKENS=$EP_SM100_MIN_SUB_TOKENS} \
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
    --prefer-overlap-with-compute=0 ${FIRST_ONLY_ARG} \
    ${IGNORE_LOCAL:+--ignore-local-traffic} $*"
# python3 -u / PYTHONUNBUFFERED: without them stdout is block-buffered into the
# redirected log, so a run that hangs in NCCL/GIN init shows an EMPTY log.
#
# ABRT, not INT: a rank parked inside ncclCommInitRank in C never regains the
# interpreter loop, so it cannot act on SIGINT. --kill-after=60 guarantees a
# wedged rank stops pinning GPU memory instead of tripping the preflight above.

#!/usr/bin/env bash
# Acceptance checks on test_ep.py logs. `rc=0` is not a health check here -- a run
# can exit 0 with half its ranks missing, or with the type-2 proxy backend, or with
# leaked memory from a previous run inflating every number by ~2x.
#
#   ./verify_run.sh logs/*.log
#   ./verify_run.sh logs/official_2N_24sm_128tok_qpdefault_nodbg_gin5_rep1.node*.log
#
# Per file: FAIL = the number is not usable. WARN = usable but not comparable to
# an arm that differs on that axis. Exits nonzero if anything FAILed.
#
# Rank completeness is audited TWICE, because each node's log carries only its own
# LOCAL ranks (8 of 16 at 2 nodes) -- a full world in one file is not the
# expectation. Per file: local ranks present vs world/nodes. Pooled per tag across
# the files you passed: total distinct ranks vs world. A table built from one node
# is wrong -- combine is layered by node and the slow node flips between runs.
set -uo pipefail

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fails=0
for f in "$@"; do
  [ -f "$f" ] || { echo "$f: not a file"; fails=$((fails + 1)); continue; }
  echo "=== $f"
  b=$(basename "$f")
  tag=${b%.node*.log}

  # --- did it produce data at all -------------------------------------------
  # Count MATCHES, not lines: concurrent ranks regularly glue two per-rank lines
  # onto one physical line, and a per-line count silently drops the second.
  grep -o 'EP: *[0-9]*/[0-9]* | dispatch:' "$f" \
    | sed 's/EP: *//; s/ | dispatch://' > "$TMP/pairs" || true
  n_disp=$(wc -l < "$TMP/pairs" | tr -d ' ')
  world=$(cut -d/ -f2 "$TMP/pairs" | sort -rn | head -1)
  cut -d/ -f1 "$TMP/pairs" | sort -u >> "$TMP/ranks.$tag"
  nodes=$(printf '%s' "$tag" | sed -n 's/.*_\([0-9]\{1,\}\)N_.*/\1/p')
  if [ "${n_disp:-0}" = 0 ]; then
    echo "  FAIL no dispatch lines -- the run produced no data"
    fails=$((fails + 1))
  else
    local_exp=""
    if [ -n "${nodes:-}" ] && [ -n "${world:-}" ] && [ "$nodes" -gt 0 ]; then
      local_exp=$((world / nodes))
    fi
    n_uniq=$(cut -d/ -f1 "$TMP/pairs" | sort -u | wc -l | tr -d ' ')
    if [ -n "$local_exp" ] && [ "$n_uniq" -lt "$local_exp" ]; then
      echo "  FAIL $n_uniq distinct ranks, expected $local_exp local (world $world / ${nodes}N)"
      fails=$((fails + 1))
    else
      echo "  ok   dispatch lines $n_disp, $n_uniq distinct ranks (world ${world:-?}${local_exp:+, $local_exp local expected})"
    fi
    printf '%s\n' "${world:-0}" > "$TMP/world.$tag"
  fi

  # --- the two b300 blockers ------------------------------------------------
  if grep -q 'only [0-9]* GIN GDAKI NICs' "$f"; then
    echo "  FAIL NCCL under-created GDAKI NICs -- needs NCCL_IB_HCA=rdmap"
    echo "       (auto-injected by run_test_ep.sh when a non-rdmap ibverbs device exists)"
    fails=$((fails + 1))
  fi
  if grep -qE 'ptxas fatal|compiler\.hpp:239' "$f"; then
    echo "  FAIL JIT/ptxas failure -- on sm_10x the CUDA base must be >= 13.3.x"
    fails=$((fails + 1))
  fi

  # --- which GIN backend actually ran --------------------------------------
  # Both arms print `Loaded gin plugin Libfabric_GDAKI (v14)`, so that line does
  # NOT prove GDAKI ran. The env is the reliable tell; the skip line confirms it
  # but only appears with NCCL_DEBUG=INFO.
  if grep -q 'NCCL_GIN_TYPE=5' "$f"; then
    if grep -q 'GIN/Plugin: Skipping plugin Libfabric.*NCCL_GIN_TYPE=5 requested' "$f"; then
      echo "  ok   GIN type 5 requested and type 2 skipped (INFO-confirmed)"
    else
      echo "  ok   GIN type 5 requested (set NCCL_DEBUG=INFO to see the skip line)"
    fi
  else
    echo "  WARN no NCCL_GIN_TYPE=5 -- this is the type-2 proxy backend"
    echo "       (~9% less prefill, 2.2-5.4x the decode latency). Do not pool with a _gin5 arm."
  fi
  case "$b" in
    *_gin5_*|*_gin5.*) grep -q 'NCCL_GIN_TYPE=5' "$f" || {
        echo "  FAIL name says _gin5 but the env does not -- mislabelled log"; fails=$((fails + 1)); } ;;
    *_type2_*|*_type2.*) if grep -q 'NCCL_GIN_TYPE=5' "$f"; then
        echo "  FAIL name says _type2 but NCCL_GIN_TYPE=5 was set -- mislabelled log"; fails=$((fails + 1)); fi ;;
  esac

  # --- which --prefer-overlap-with-compute actually ran ---------------------
  # It is a CLI arg, not an env var, so no `-e` line carries it. Read it from the
  # CLI form rather than from run_test_ep.sh's banner: the banner only exists since
  # 2026-08-31, the `set -x` trace of the docker command has always had it, and
  # both spell it the same way. Absent = a log old enough to predate the flag,
  # which was the =0 default. =1 is a different kernel configuration (double
  # buffering, warp counts) and on PR #8+#9 it also disables the channel-clamp
  # removal and the forward-warp pairing, so the two must never pool.
  ovlp=$(grep -o -- '--prefer-overlap-with-compute=[0-9]*' "$f" | head -1 | cut -d= -f2)
  case "$b" in
    *_ovlp0_*|*_ovlp0.*) [ "${ovlp:-0}" = 0 ] || {
        echo "  FAIL name says _ovlp0 but the run passed =$ovlp -- mislabelled log"
        fails=$((fails + 1)); } ;;
    *_ovlp1_*|*_ovlp1.*) [ "${ovlp:-}" = 1 ] || {
        echo "  FAIL name says _ovlp1 but the run passed =${ovlp:-<absent>} -- mislabelled log"
        fails=$((fails + 1)); } ;;
    *) if [ "${ovlp:-0}" != 0 ]; then
         echo "  FAIL --prefer-overlap-with-compute=$ovlp but the name carries no _ovlp1"
         echo "       segment -- it would pool with the =0 arm"
         fails=$((fails + 1))
       fi ;;
  esac

  # --- things that make a number incomparable rather than wrong ------------
  if grep -q 'EP_BUFFER_DEBUG=' "$f"; then
    echo "  WARN EP_BUFFER_DEBUG was set -- it printf()s from inside dispatch's timed"
    echo "       host polling loop. Not comparable to an arm without it."
  fi
  if grep -q -- '--num-sms=0' "$f"; then
    echo "  WARN --num-sms=0 -- the auto path reads ONE NIC's rate (half the truth on"
    echo "       b300, 2 NICs/GPU). Pass NUM_SMS explicitly for a reproducible number."
  fi
  if ! grep -q -- '--ignore-local-traffic' "$f"; then
    echo "  WARN no --ignore-local-traffic: the SO column includes intra-node traffic"
    echo "       and can exceed the per-GPU wire rate. It is not a wire rate."
  fi
  # An out-of-range num_allocated_qps is CLAMPED into [2, 17], not rejected, since
  # upstream 9c1f2511 (it used to trip EP_HOST_ASSERT). It is reachable straight from
  # this launcher -- test_ep.py has --num-qps / --num-allocated-qps and allocates
  # max() of the two -- so a cell that asks for 768 quietly measures 17, and the log
  # is the only place that says so. Same shape as the kMaxParts clamp that once made
  # a whole sweep a no-op.
  if grep -q 'clamped num_allocated_qps' "$f"; then
    echo "  WARN $(grep -o 'clamped num_allocated_qps from [0-9]* to [0-9]*' "$f" | head -1)"
    echo "       -- the effective QP count is the clamped one, not the one you asked"
    echo "       for. Do not label this cell with the requested value."
  fi
  if grep -q -- '--test-first-only' "$f"; then
    echo "  note --test-first-only: FP8 dispatch at expert_alignment=128 (the first"
    echo "       entry of enumerate_ep_modes), not BF16."
  fi

  # --- provenance -----------------------------------------------------------
  ref=$(grep -o 'DeepEP=[0-9a-f]\{7,40\}' "$f" | head -1 | cut -d= -f2)
  qps=$(grep -o '#QPs: *[0-9]*/[0-9]*' "$f" | head -1)
  echo "  info DeepEP=${ref:-<unstamped image>}${qps:+  $qps}"
  # `#QPs: A/B` is A used / B allocated (test_ep.py:82). A > B means the kernels were
  # told to use more QPs than the buffer holds: the 9c1f2511 clamp bounds only the
  # ALLOCATION, while --num-qps passes through, so `--num-qps 768` prints `768/17`.
  # Every log under results/ has A == B (11/11 or 5/5), so a mismatch is the log
  # telling you the cell is not the configuration its name claims.
  if [ -n "${qps:-}" ]; then
    used=$(printf '%s' "$qps" | sed 's|.*: *||; s|/.*||')
    alloc=$(printf '%s' "$qps" | sed 's|.*/||')
    if [ "${used:-0}" -gt "${alloc:-0}" ]; then
      echo "  WARN #QPs $used/$alloc -- more QPs requested than allocated (the allocation"
      echo "       was clamped). Fix the request; do not publish this as $used QPs."
    fi
  fi
  if [ -z "${ref:-}" ]; then
    echo "  WARN no DeepEP BUILD_REF in the log -- the image tag alone cannot attribute"
    echo "       a number; we have measured ~1.8x decode swings between two commits."
  fi
done

echo
# --- pooled audit: this is the one that decides whether a tag is publishable ---
for rf in "$TMP"/ranks.*; do
  [ -e "$rf" ] || continue
  tag=$(basename "$rf"); tag=${tag#ranks.}
  pooled=$(sort -u "$rf" | wc -l | tr -d ' ')
  world=$(cat "$TMP/world.$tag" 2>/dev/null || echo 0)
  if [ "$world" -gt 0 ] && [ "$pooled" -lt "$world" ]; then
    echo "POOLED $tag: $pooled/$world ranks -- either a node's log was not passed to"
    echo "       this script, or ranks are genuinely missing. Do not publish until it is $world."
    fails=$((fails + 1))
  else
    echo "POOLED $tag: $pooled/$world ranks"
  fi
done

echo
if [ "$fails" -gt 0 ]; then echo "=== $fails FAIL(s)"; exit 1; fi
echo "=== no FAILs"

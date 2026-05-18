#!/usr/bin/env bash
# Sweep NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE for LL bench (and optionally HT).
# Run from your LOCAL machine; both nodes are reached via ssh aliases.
#
# Usage:
#   bash sweep_max_nic.sh <leader_ssh> <worker_ssh> <leader_priv_ip> [ll|ht|both]
# Example:
#   bash sweep_max_nic.sh P5EN-1 P5EN-2 172.31.45.156 ll
set -eu

LEADER_SSH="${1:?leader ssh alias}"
WORKER_SSH="${2:?worker ssh alias}"
LEADER_IP="${3:?leader private ip}"
MODE="${4:-ll}"

VALUES=(0 1 2 4 8 16 32)   # 0 = unset (NVSHMEM default)
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-./maxnic_sweep_${TS}}"
mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"

run_one() {
  local kind="$1"
  local v="$2"
  local script
  if [[ "$kind" == "ll" ]]; then
    script="run_low_latency.sh"
  else
    script="run_internode.sh"
  fi

  local label env_export
  if [[ "$v" == "0" ]]; then
    label="default"
    env_export=""
  else
    label="$v"
    env_export="export NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE=$v;"
  fi

  echo
  echo "==========================================================="
  echo "[$kind] MAX_NIC_PER_PE=$label"
  echo "==========================================================="

  local lf="$OUT_DIR/${kind}_max${label}_leader.log"
  local wf="$OUT_DIR/${kind}_max${label}_worker.log"

  ssh "$WORKER_SSH" "cd ~/work/deepep-v1-efa && $env_export bash $script 1 $LEADER_IP" > "$wf" 2>&1 &
  local wpid=$!
  ssh "$LEADER_SSH" "cd ~/work/deepep-v1-efa && $env_export bash $script 0 $LEADER_IP" > "$lf" 2>&1 || true
  wait "$wpid" 2>/dev/null || true

  if [[ "$kind" == "ll" ]]; then
    echo "--- summary (rank 0 only) ---"
    grep -E "\[rank 0\] (Dispatch|Combine).*bandwidth" "$lf" \
      | tee "$OUT_DIR/${kind}_max${label}.summary"
  else
    echo "--- summary ---"
    grep -E "Best dispatch|Best combine" "$lf" \
      | tee "$OUT_DIR/${kind}_max${label}.summary"
  fi
}

case "$MODE" in
  ll)   for v in "${VALUES[@]}"; do run_one ll "$v"; done ;;
  ht)   for v in "${VALUES[@]}"; do run_one ht "$v"; done ;;
  both) for v in "${VALUES[@]}"; do run_one ll "$v"; done
        for v in "${VALUES[@]}"; do run_one ht "$v"; done ;;
  *) echo "Unknown mode: $MODE  (use ll|ht|both)"; exit 1 ;;
esac

echo
echo "All results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -+2

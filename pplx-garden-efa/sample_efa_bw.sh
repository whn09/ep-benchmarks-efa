#!/bin/bash
# Sample per-NIC EFA bandwidth N times at INTERVAL seconds, append to OUT.
# Usage: sample_efa_bw.sh OUT N INTERVAL
OUT="${1:-/tmp/efa_bw.log}"
N="${2:-30}"
INTERVAL="${3:-2}"

DEVS=$(ls /sys/class/infiniband/ | grep rdmap)
declare -A PREV_TX PREV_RX
read_stats() {
  local dev="$1"
  rdma statistic show | grep "^link $dev"
}
for dev in $DEVS; do
  stats=$(read_stats "$dev")
  PREV_TX[$dev]=$(echo "$stats" | grep -oP 'tx_bytes \K[0-9]+')
  PREV_RX[$dev]=$(echo "$stats" | grep -oP 'rx_bytes \K[0-9]+')
done

: > "$OUT"
for ((i=1; i<=N; i++)); do
  sleep "$INTERVAL"
  ts=$(date +%H:%M:%S)
  total_tx=0; total_rx=0; nics_active=0
  echo "=== sample $i @ $ts (interval ${INTERVAL}s) ===" >> "$OUT"
  printf "%-12s %12s %12s\n" "nic" "tx_Gbps" "rx_Gbps" >> "$OUT"
  for dev in $DEVS; do
    stats=$(read_stats "$dev")
    curr_tx=$(echo "$stats" | grep -oP 'tx_bytes \K[0-9]+')
    curr_rx=$(echo "$stats" | grep -oP 'rx_bytes \K[0-9]+')
    tx_diff=$((curr_tx - PREV_TX[$dev]))
    rx_diff=$((curr_rx - PREV_RX[$dev]))
    PREV_TX[$dev]=$curr_tx
    PREV_RX[$dev]=$curr_rx
    tx_gbps=$(awk "BEGIN {printf \"%.2f\", $tx_diff*8/$INTERVAL/1e9}")
    rx_gbps=$(awk "BEGIN {printf \"%.2f\", $rx_diff*8/$INTERVAL/1e9}")
    printf "%-12s %12s %12s\n" "$dev" "$tx_gbps" "$rx_gbps" >> "$OUT"
    total_tx=$(awk "BEGIN {printf \"%.2f\", $total_tx + $tx_gbps}")
    total_rx=$(awk "BEGIN {printf \"%.2f\", $total_rx + $rx_gbps}")
    if awk -v g="$tx_gbps" 'BEGIN{exit !(g+0>1.0)}'; then nics_active=$((nics_active+1)); fi
  done
  echo "TOTAL  tx=${total_tx} Gbps  rx=${total_rx} Gbps  active(>=1Gbps tx)=${nics_active}/32" >> "$OUT"
done

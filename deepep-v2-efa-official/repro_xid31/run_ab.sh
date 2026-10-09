#!/usr/bin/env bash
# Rotate Megatron arms until a deadline, one run per slot, and record per-run outcome.
#
#   NODES="..." DEADLINE_UTC=10:55 SLOT_SECS=2700 ./run_ab.sh arm1 arm2 ...
#   arm := base | pr5 | ovl0 | probe | min
#
# A failure is rare and timing-dependent (same mock data failed at step 72 once and ran 290+
# steps clean the next time), so arms are ROTATED rather than run in blocks and compared as
# failures per GPU-hour, never as "it failed / it did not" from a single slot.
# Outcome per run, appended to $SUMMARY: arm, tag, start, secs, last step, Xid delta on each node,
# and which node/GPU logged the first Xid.
set -uo pipefail
cd "$(dirname "$0")"
NODES="${NODES:?}"
DEADLINE_UTC="${DEADLINE_UTC:?HH:MM}"
SLOT_SECS="${SLOT_SECS:-2700}"
SUMMARY="${SUMMARY:-$HOME/xid31_ab_summary.tsv}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
SSH="ssh -n -o ConnectTimeout=10"
xids () { for n in "${NODE_ARR[@]}"; do $SSH "$n" 'sudo dmesg | grep -c "Xid"' 2>/dev/null | tr -d ' \r'; done | paste -sd, -; }
deadline=$(date -u -j -f "%H:%M" "$DEADLINE_UTC" +%s 2>/dev/null || date -u -d "$DEADLINE_UTC" +%s)
[ -s "$SUMMARY" ] || printf 'arm\ttag\tstart_utc\tsecs\tlast_step\txid_before\txid_after\n' > "$SUMMARY"
i=0
while :; do
  now=$(date -u +%s); left=$((deadline - now))
  [ "$left" -gt 600 ] || { echo "=== deadline reached"; break; }
  arm=${@:$((i % $# + 1)):1}; i=$((i + 1))
  case "$arm" in
    base) IMG=deepep-v2-efa-official:sm100-874779c-mlm1edcc0c; OV=1 ;;
    pr5)  IMG=deepep-v2-efa-official:sm100-874779c-pr5-mlm1edcc0c; OV=1 ;;
    ovl0) IMG=deepep-v2-efa-official:sm100-874779c-mlm1edcc0c; OV=0 ;;
    probe) IMG=deepep-v2-efa-official:sm100-874779c-probe-mlm1edcc0c; OV=1 ;;   # EP_PROBE + slot/dst guard
    min)   IMG=deepep-v2-efa-official:sm100-874779c-min-mlm1edcc0c; OV=1 ;;     # cached data channels above QP 0
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  secs=$(( left - 300 < SLOT_SECS ? left - 300 : SLOT_SECS ))
  tag="ab_${arm}_$(date -u +%H%M)"
  before=$(xids); t0=$(date -u +%s)
  echo "=== $tag image=$IMG overlap=$OV secs=$secs xid_before=$before"
  NODES="$NODES" TAG="$tag" NUM_LAYERS=24 TRAIN_ITERS=100000 RUN_TIMEOUT="$secs" PORT=$((29600 + i)) \
    IMAGE="$IMG" OVERLAP="$OV" ./run_megatron.sh || true
  after=$(xids); dt=$(( $(date -u +%s) - t0 ))
  last=$($SSH "${NODE_ARR[$((${#NODE_ARR[@]} - 1))]}" \
    "grep -oE ' iteration +[0-9]+/' ~/xid31_runs/$tag.node${#NODE_ARR[@]}.log | tail -1 | tr -dc 0-9" 2>/dev/null)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$tag" "$(date -u -r "$t0" +%H:%M 2>/dev/null || date -u -d @"$t0" +%H:%M)" \
    "$dt" "${last:-?}" "$before" "$after" | tee -a "$SUMMARY"
  sleep 30
done

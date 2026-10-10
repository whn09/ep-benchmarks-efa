#!/usr/bin/env bash
# On-host arm rotation: run the SAME command on every node (only NODE_RANK differs) and the nodes
# stay in lockstep with no ssh between them.
#
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0 private ip> RUN_ID=<same on all nodes> \
#     DEADLINE_UTC=HH:MM [SLOT_SECS=2700] [MLM_TAG=mlm1edcc0c] ./run_ab_node.sh arm1 arm2 ...
#   arm := main_pre | main_fixed | base | pr5 | ovl0 | probe | min   (images as in run_ab.sh)
#
# Slot i always runs arm[i % #arms] with TAG ab_<RUN_ID>_<i>_<arm> on port 29600+i, so the nodes
# agree on every slot without talking to each other. A slot ends when this node's container exits:
# a fault on any GPU takes every rank down within ~100 s (DeepEP barrier timeout), and RUN_TIMEOUT
# ends a clean slot at the same moment everywhere, so the nodes leave a slot within a couple of
# minutes of each other and meet again at the next slot's torchrun rendezvous.
# Start all nodes within a minute or two of each other. Each node appends one line per slot to
# $SUMMARY (local Xid count before/after, last logged step on the node that prints iterations);
# merge the 4 files afterwards -- a slot "failed" if ANY node's Xid count rose.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NNODES="${NNODES:?}"; NODE_RANK="${NODE_RANK:?}"; MASTER_ADDR="${MASTER_ADDR:?}"
RUN_ID="${RUN_ID:?RUN_ID=<identical on every node, e.g. 1010a>}"
DEADLINE_UTC="${DEADLINE_UTC:?HH:MM}"
SLOT_SECS="${SLOT_SECS:-2700}"
MLM_TAG="${MLM_TAG:-mlm1edcc0c}"
NUM_LAYERS="${NUM_LAYERS:-24}"
LOGDIR="${LOGDIR:-$HOME/xid31_runs}"
SUMMARY="${SUMMARY:-$HOME/xid31_ab_${RUN_ID}.node$((NODE_RANK + 1)).tsv}"
[ $# -gt 0 ] || { echo "give at least one arm" >&2; exit 1; }
deadline=$(date -u -d "$DEADLINE_UTC" +%s)
xid () { sudo dmesg | grep -c Xid; }
[ -s "$SUMMARY" ] || printf 'arm\ttag\tnode\tstart_utc\tsecs\tlast_step\txid_before\txid_after\trc\n' > "$SUMMARY"
i=0
while :; do
  left=$(( deadline - $(date -u +%s) ))
  [ "$left" -gt 600 ] || { echo "=== deadline reached"; break; }
  arm=${@:$((i % $# + 1)):1}
  case "$arm" in
    main_pre)   IMG=deepep-v2-efa-official:sm100-b90a617-$MLM_TAG; OV=1 ;;
    main_fixed) IMG=deepep-v2-efa-official:sm100-6de427b-$MLM_TAG; OV=1 ;;
    base)       IMG=deepep-v2-efa-official:sm100-874779c-$MLM_TAG; OV=1 ;;
    pr5)        IMG=deepep-v2-efa-official:sm100-874779c-pr5-$MLM_TAG; OV=1 ;;
    ovl0)       IMG=deepep-v2-efa-official:sm100-874779c-$MLM_TAG; OV=0 ;;
    probe)      IMG=deepep-v2-efa-official:sm100-874779c-probe-$MLM_TAG; OV=1 ;;
    min)        IMG=deepep-v2-efa-official:sm100-874779c-min-$MLM_TAG; OV=1 ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  docker image inspect "$IMG" >/dev/null 2>&1 || { echo "!! image $IMG missing on this node" >&2; exit 1; }
  secs=$(( left - 300 < SLOT_SECS ? left - 300 : SLOT_SECS ))
  tag="ab_${RUN_ID}_${i}_${arm}"
  before=$(xid); t0=$(date -u +%s)
  echo "=== slot $i: $tag image=$IMG overlap=$OV secs=$secs xid_before=$before"
  NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR TAG=$tag IMAGE=$IMG \
    PORT=$((29600 + i)) NUM_LAYERS=$NUM_LAYERS TRAIN_ITERS=100000 RUN_TIMEOUT=$secs OVERLAP=$OV \
    LOGDIR=$LOGDIR "$HERE/run_node.sh"
  rc=$?
  after=$(xid)
  last=$(grep -oE ' iteration +[0-9]+/' "$LOGDIR/$tag.node$((NODE_RANK + 1)).log" 2>/dev/null | tail -1 | tr -dc 0-9)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$tag" "$((NODE_RANK + 1))" \
    "$(date -u -d @"$t0" +%H:%M)" "$(( $(date -u +%s) - t0 ))" "${last:--}" "$before" "$after" "$rc" | tee -a "$SUMMARY"
  i=$((i + 1))
  sleep 30
done

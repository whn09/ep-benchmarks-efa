#!/usr/bin/env bash
# 2-node b300 campaign: main vs PR #1+#2 vs PR #8+#9 vs the merged stack.
# Runs from the laptop; it only builds the CELLS list and hands it to
# run_campaign.sh, so every log-naming and rotation rule there still applies.
#
#   NODES="B300-1 B300-2" ./run_b300_stack_campaign.sh
#
# The matrix is deliberately the same as results/p5en_stack_20260831 (2 nodes,
# 12 SM, 128 and 8192 tokens/rank, qpdefault and subparts1) so that b300 and
# p5en rows are comparable cell by cell. It is wider only in that all four arms
# carry both token counts.
#
# Env:
#   REPS=3          passed through
#   ARMS            override the arm list (see ARM_TAGS below)
#   DRY=1           print the cells and the resolved image tags, run nothing
#
# Preconditions (both checked here, because each one silently produces a wrong
# number rather than an error):
#   1. efa.ko >= 3.3.0 on BOTH hosts. Without it NCCL_GIN_TYPE=5 falls back to
#      the type-2 proxy while the log tag still says _gin5 -- run
#      prepare_host_efa150.sh first.
#   2. Idle GPUs. test_ep.py allocates a large slab per rank.
set -euo pipefail
cd "$(dirname "$0")"

NODES="${NODES:?NODES=\"B300-1 B300-2\"}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
LEADER=${NODE_ARR[0]}
SSH="ssh -n -o ConnectTimeout=10"
ARCH=sm103

MAIN_REF=54fffeff810723f574c574b1790dff189f3c6ffb   # upstream main we measured on p5en
PR12_REF=bfbdd15ff448783f877cb2210cb3246c8452b05e   # amazon-contributing/DeepEP #1 + #2
PR89_REF=3c737dcf0da5889ba7efd26e05b4808307cc38af   # #8 + #9
STACK_TAG="${ARCH}-stack${PR89_REF:0:7}x${PR12_REF:0:7}"

img_main="deepep-v2-efa-official:${ARCH}-${MAIN_REF:0:7}"
img_pr12="deepep-v2-efa-official:${ARCH}-${PR12_REF:0:7}"
img_pr89="deepep-v2-efa-official:${ARCH}-${PR89_REF:0:7}"
img_stack="deepep-v2-efa-official:${STACK_TAG}"

# No associative arrays: the laptop that drives this is macOS, i.e. bash 3.2.
NNODES=${#NODE_ARR[@]}
have () {   # $1 image -> echoes the number of nodes that have it
  local img=$1 n c=0
  for n in "${NODE_ARR[@]}"; do
    $SSH "$n" "docker image inspect $img >/dev/null 2>&1" && c=$((c + 1))
  done
  echo "$c"
}
echo "=== images expected on every node:"
have_main=$(have "$img_main")
have_pr12=$(have "$img_pr12")
have_pr89=$(have "$img_pr89")
have_stack=$(have "$img_stack")
printf '  %-52s on %s/%s nodes\n' "$img_main" "$have_main" "$NNODES"
printf '  %-52s on %s/%s nodes\n' "$img_pr12" "$have_pr12" "$NNODES"
printf '  %-52s on %s/%s nodes\n' "$img_pr89" "$have_pr89" "$NNODES"
printf '  %-52s on %s/%s nodes\n' "$img_stack" "$have_stack" "$NNODES"

# The stack image's arm label is its MERGE sha, not the tag: two hosts must have
# produced the same merge (Dockerfile.stack pins GIT_AUTHOR_DATE/GIT_COMMITTER_DATE
# for exactly this reason). If they differ, the two nodes are running different
# code and the cell is meaningless.
stack_arm=""
if [ "$have_stack" = "$NNODES" ]; then
  refs=""
  for n in "${NODE_ARR[@]}"; do
    r=$($SSH "$n" "docker run --rm --entrypoint cat $img_stack /opt/DeepEP/BUILD_REF" | tr -d ' \r')
    echo "  $n stack BUILD_REF=$r"
    refs="$refs $r"
  done
  uniq_n=$(printf '%s\n' $refs | sort -u | wc -l | tr -d ' ')
  [ "$uniq_n" = 1 ] || { echo "!! the stack merge sha differs across nodes:$refs"; exit 2; }
  stack_arm="stack$(printf '%s' $refs | head -c 7)"
  echo "  stack arm label: $stack_arm"
fi

step_ok=1
echo "=== preconditions:"
for n in "${NODE_ARR[@]}"; do
  kv=$($SSH "$n" 'modinfo efa 2>/dev/null | awk "/^version:/{print \$2}"' | tr -d ' \r')
  apps=$($SSH "$n" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' | tr -d ' \r')
  printf '  %-8s efa.ko=%-8s compute_procs=%s\n' "$n" "$kv" "$apps"
  case "$kv" in 3.3.*) ;; *) echo "     !! needs efa.ko >= 3.3.0 (prepare_host_efa150.sh)"; step_ok=0 ;; esac
  [ "$apps" = 0 ] || { echo "     !! GPUs busy"; step_ok=0; }
done

# arm|image|tokens|num_sms|knobtag|extra env|prefer_overlap
# subparts1 = EP_NUM_SUB_PARTS=1, which only the #1+#2 JIT reads, so it is carried
# on the two arms that contain it. One main+subparts1 cell is kept as the no-op
# control: if it moves, the knob is reaching something it should not.
CELLS=""
add () { CELLS="${CELLS}$1
"; }
[ "$have_main" = "$NNODES" ] && {
  add "main${MAIN_REF:0:7}|$img_main|8192|12|qpdefault|"
  add "main${MAIN_REF:0:7}|$img_main|128|12|qpdefault|"
  add "main${MAIN_REF:0:7}|$img_main|128|12|subparts1|EP_NUM_SUB_PARTS=1"
}
[ "$have_pr12" = "$NNODES" ] && {
  add "pr12${PR12_REF:0:7}|$img_pr12|8192|12|qpdefault|"
  add "pr12${PR12_REF:0:7}|$img_pr12|128|12|qpdefault|"
  add "pr12${PR12_REF:0:7}|$img_pr12|128|12|subparts1|EP_NUM_SUB_PARTS=1"
  add "pr12${PR12_REF:0:7}|$img_pr12|8192|12|subparts1|EP_NUM_SUB_PARTS=1"
}
[ "$have_pr89" = "$NNODES" ] && {
  add "pr89${PR89_REF:0:7}|$img_pr89|8192|12|qpdefault|"
  add "pr89${PR89_REF:0:7}|$img_pr89|128|12|qpdefault|"
}
[ -n "$stack_arm" ] && {
  add "$stack_arm|$img_stack|8192|12|qpdefault|"
  add "$stack_arm|$img_stack|128|12|qpdefault|"
  add "$stack_arm|$img_stack|128|12|subparts1|EP_NUM_SUB_PARTS=1"
  add "$stack_arm|$img_stack|8192|12|subparts1|EP_NUM_SUB_PARTS=1"
}

echo "=== cells ($(printf '%s' "$CELLS" | grep -c . ) per rep, REPS=${REPS:-3}):"
printf '%s' "$CELLS" | sed 's/^/  /'

# DRY first, so the matrix can be reviewed while the hosts are still busy and
# still on the old kmod -- which is exactly when you want to review it.
[ "${DRY:-}" != 1 ] || { echo; echo "DRY=1, nothing run."; exit 0; }
[ "$step_ok" = 1 ] || { echo; echo "REFUSING: preconditions above are not met."; exit 3; }

CELLS="$CELLS" REPS="${REPS:-3}" NODES="$NODES" ./run_campaign.sh "$ARCH"

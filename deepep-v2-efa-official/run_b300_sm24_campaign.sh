#!/usr/bin/env bash
# 2-node b300 campaign at 24 SM: does the PR stack's combine win survive the SM
# count the reference row uses?
#
#   NODES="B300-1 B300-2" ./run_b300_sm24_campaign.sh
#
# WHY THIS EXISTS. results/b300_stack_20260903 measured all four arms at 12 SM,
# and the repo README's throughput table compares its combine (83.21 GB/s /
# 2816.0 us) against a 2026-08 GDAKI row taken at 24 SM (131 / 1788.1). That
# comparison is not decidable from either campaign: route B's own sweep
# (deepep-v2-efa-gdaki-b200/results/b300_20260813/b300_pfsm_p1_*) says 12 -> 24 SM
# is worth -36.1% on combine all by itself, and at matched 12 SM the two arms are
# at parity. So the open question is not "why is 12 SM slower" -- it is whether
# PR #8+#9's combine win is still there once combine has the SMs it wants, or
# whether it was only recovering SM-starvation. Same for the decode side: the
# 12 SM stack decode is 286.4 us and b300's decode used to be SM-invariant
# (project_deepep_b300_decode_floor), but that was measured BEFORE the PRs
# removed the floor, so it has to be re-checked rather than assumed.
#
# THE MATRIX. All four arms x both token counts at 24 SM, so 24 SM gets its own
# complete additivity table rather than a single cell hanging off the 12 SM one:
#
#   8192 tok  main / #1+#2 / #8+#9 / stack   @ 24 SM     the combine question
#    128 tok  main / #1+#2 / #8+#9 / stack   @ 24 SM     is decode still SM-flat?
#   8192 tok  stack                          @ 12 SM     drift anchor
#    128 tok  stack                          @ 12 SM     drift anchor
#
# The two 12 SM anchor cells are the point of the whole thing being one campaign:
# they re-measure a cell results/b300_stack_20260903 already has, so a 24-vs-12
# delta computed inside this campaign can be checked against the same delta
# across campaigns. If the anchor has drifted, the cross-campaign comparison in
# the README is what is wrong, not this table. Everything runs at the DEFAULT
# part geometry (no EP_NUM_SUB_PARTS): on b300 that knob is inert on decode and
# costs the stack +102 us on prefill dispatch, so it is not the deployed arm.
#
# Env:
#   REPS=3          passed through (rotated: every cell once per rep)
#   SMS="24"        SM counts to sweep at, space separated. `SMS="24 48"` adds
#                   route B's dispatch optimum without editing anything; the 12 SM
#                   anchor is added on top regardless.
#   ANCHOR=0        drop the two 12 SM anchor cells (do not: see above)
#   DRY=1           print the matrix and the image inventory, run nothing
#
# Preconditions, both checked below because each produces a WRONG NUMBER rather
# than an error: efa.ko >= 3.3.0 on both hosts (otherwise NCCL_GIN_TYPE=5 falls
# back to the type-2 proxy while the tag still says _gin5) and idle GPUs.
#
# After the run, from this directory:
#   mkdir -p results/b300_sm24_20260903/logs
#   scp 'B300-1:~/epruns/*_2N_24sm_*.node1.log' results/b300_sm24_20260903/logs/
#   scp 'B300-2:~/epruns/*_2N_24sm_*.node2.log' results/b300_sm24_20260903/logs/
#   ... and the same two for *_2N_12sm_* (both nodes: combine is node-layered)
#   ./verify_run.sh results/b300_sm24_20260903/logs/*.log
#   python3 results/b300_sm24_20260903/make_sm24_tables.py | tee \
#           results/b300_sm24_20260903/tables.txt
set -euo pipefail
cd "$(dirname "$0")"

NODES="${NODES:?NODES=\"B300-1 B300-2\"}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
SSH="ssh -n -o ConnectTimeout=10"
ARCH=sm103
SMS="${SMS:-24}"
ANCHOR="${ANCHOR:-1}"

MAIN_REF=54fffeff810723f574c574b1790dff189f3c6ffb   # the upstream main both campaigns use
PR12_REF=bfbdd15ff448783f877cb2210cb3246c8452b05e   # amazon-contributing/DeepEP #1 + #2
PR89_REF=3c737dcf0da5889ba7efd26e05b4808307cc38af   # #8 + #9
STACK_TAG="${ARCH}-stack${PR89_REF:0:7}x${PR12_REF:0:7}"

img_main="deepep-v2-efa-official:${ARCH}-${MAIN_REF:0:7}"
img_pr12="deepep-v2-efa-official:${ARCH}-${PR12_REF:0:7}"
img_pr89="deepep-v2-efa-official:${ARCH}-${PR89_REF:0:7}"
img_stack="deepep-v2-efa-official:${STACK_TAG}"

NNODES=${#NODE_ARR[@]}
have () {   # $1 image -> how many nodes have it (macOS bash 3.2: no associative arrays)
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
printf '  %-52s on %s/%s nodes\n' "$img_main"  "$have_main"  "$NNODES"
printf '  %-52s on %s/%s nodes\n' "$img_pr12"  "$have_pr12"  "$NNODES"
printf '  %-52s on %s/%s nodes\n' "$img_pr89"  "$have_pr89"  "$NNODES"
printf '  %-52s on %s/%s nodes\n' "$img_stack" "$have_stack" "$NNODES"

# The stack arm is labelled by its MERGE sha, and this campaign's whole purpose is
# to be subtractable from results/b300_stack_20260903 -- which only holds if the
# merge is the same tree. Dockerfile.stack pins the git dates so it is; assert it
# instead of trusting it, and assert it equals the sha that campaign measured.
STACK_SHA_20260903=a35285f
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
  [ "$stack_arm" = "stack$STACK_SHA_20260903" ] || {
    echo "!! this stack image is $stack_arm, but results/b300_stack_20260903 measured"
    echo "   stack$STACK_SHA_20260903. The 24-vs-12 SM comparison would cross two trees."
    exit 2; }
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
CELLS=""
add () { CELLS="${CELLS}$1
"; }
for sm in $SMS; do
  for tok in 8192 128; do
    [ "$have_main"  = "$NNODES" ] && add "main${MAIN_REF:0:7}|$img_main|$tok|$sm|qpdefault|"
    [ "$have_pr12"  = "$NNODES" ] && add "pr12${PR12_REF:0:7}|$img_pr12|$tok|$sm|qpdefault|"
    [ "$have_pr89"  = "$NNODES" ] && add "pr89${PR89_REF:0:7}|$img_pr89|$tok|$sm|qpdefault|"
    [ -n "$stack_arm" ]           && add "$stack_arm|$img_stack|$tok|$sm|qpdefault|"
  done
done
if [ "$ANCHOR" = 1 ] && [ -n "$stack_arm" ]; then
  add "$stack_arm|$img_stack|8192|12|qpdefault|"
  add "$stack_arm|$img_stack|128|12|qpdefault|"
fi

ncells=$(printf '%s' "$CELLS" | grep -c .)
echo "=== cells ($ncells per rep, REPS=${REPS:-3}, SMS='$SMS', anchor=$ANCHOR):"
printf '%s' "$CELLS" | sed 's/^/  /'
echo "=== ~$(( (ncells * ${REPS:-3} * 100) / 60 )) min at ~100 s/cell including the 20 s settle"

# DRY first, so the matrix can be reviewed while the hosts are still busy -- which
# is exactly when you want to review it.
[ "${DRY:-}" != 1 ] || { echo; echo "DRY=1, nothing run."; exit 0; }
[ "$step_ok" = 1 ] || { echo; echo "REFUSING: preconditions above are not met."; exit 3; }

# PORT_BASE moved off run_campaign.sh's 8500 default: a killed cell from the 12 SM
# campaign can leave a TCPStore listener behind on those ports.
CELLS="$CELLS" REPS="${REPS:-3}" NODES="$NODES" PORT_BASE="${PORT_BASE:-8700}" \
  ./run_campaign.sh "$ARCH"

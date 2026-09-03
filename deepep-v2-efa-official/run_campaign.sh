#!/usr/bin/env bash
# Multi-node campaign driver. Runs from your laptop; drives run_test_ep.sh on every
# node over ssh, one cell at a time, and names every log so that
# results/*/make_tables.py can pool it.
#
#   NODES="<leader> <worker> [...]" ./run_campaign.sh [ARCH]
#
# ARCH   sm90 | sm103   (default: probed from the leader's compute_cap). It only
#        selects the default cell list -- there is no arch-specific code here.
#
# Env:
#   NODES          REQUIRED. ssh aliases/hosts, space separated, FIRST = leader.
#                  WORLD_SIZE is the count of these, per test_ep.py's convention
#                  (it is not torchrun: WORLD_SIZE = nodes, RANK = node index).
#   MASTER_IP      leader's PRIVATE ip. Default: asked of the leader over ssh.
#                  A public ip works for ssh and not for the rendezvous.
#   IMAGE_BASE     image for the `official` arm. Default deepep-v2-efa-official:<arch>
#   IMAGE_PRS      image for the `prs` arm (amazon-contributing/DeepEP #1 + #2).
#                  Default deepep-v2-efa-official:<arch>-bfbdd15 (PR #2's head); if
#                  that image does not exist the prs cells are skipped with a message
#                  rather than failing 9 runs one at a time.
#   REPS=3         reps are ROTATED (every cell once per rep), not blocked per arm,
#                  so a slow drift cannot be mistaken for an arm effect.
#   NUM_PROCESSES=8   local ranks per node
#   GIN_ENV        default "NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0" (the
#                  same default run_test_ep.sh applies on its own). Set to "" to
#                  measure the type-2 proxy backend on purpose; the tag then says
#                  _type2 instead of _gin5 so the two never pool.
#   PORT_BASE=8500 a killed run leaves a TCPStore listener behind, so every cell
#                  gets its own port.
#   IGNORE_LOCAL   unset (default) or 1. Passes --ignore-local-traffic, which changes
#                  ONLY the GB/s denominator, never the timed calls. Default off
#                  because every published number was measured that way. On, the tag
#                  gains _igl1 so the two denominators cannot pool.
#   PREFER_OVERLAP 0 (default) or 1 -- --prefer-overlap-with-compute, and the
#                  default for cells that leave the 7th CELLS field empty. Stamped
#                  in BOTH states as _ovlp0 / _ovlp1, because it selects a different
#                  kernel configuration rather than a different denominator. Logs
#                  written before 2026-08-31 carry no segment and are all =0; a
#                  generator written against those names will not see new logs, so
#                  give a new axis a new results/ directory.
#   LOGDIR         remote log dir. Default $HOME/epruns
#   REPO_DIR       remote checkout. Default $HOME/work/ep-benchmarks-efa/deepep-v2-efa-official
#   CELLS          override the matrix. One cell per line:
#                    arm|image|tokens|num_sms|knobtag|extra env|prefer_overlap
#                  `arm` and `knobtag` only name the log; `extra env` is passed
#                  through EXTRA_ENV on top of GIN_ENV. The 7th field is optional
#                  and defaults to PREFER_OVERLAP; it is a field rather than a
#                  campaign-wide variable so that an overlap A/B rotates inside
#                  each rep instead of running as two blocks (two blocks make a
#                  slow drift indistinguishable from the effect).
#
# Everything a number depends on is in the log name -- arm, nodes, SM count,
# tokens, knob, debug state, GIN backend, rep, node. A missing axis silently
# overwrites the other arm's log, which has cost us data before.
set -euo pipefail
cd "$(dirname "$0")"

NODES="${NODES:?NODES=\"<leader> <worker> ...\" (ssh hosts, first = leader)}"
# shellcheck disable=SC2206
NODE_ARR=($NODES)
NNODES=${#NODE_ARR[@]}
LEADER=${NODE_ARR[0]}

SSH="ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
REPS="${REPS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PORT_BASE="${PORT_BASE:-8500}"
LOGDIR="${LOGDIR:-\$HOME/epruns}"
REPO_DIR="${REPO_DIR:-\$HOME/work/ep-benchmarks-efa/deepep-v2-efa-official}"
GIN_ENV="${GIN_ENV-NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0}"

case "$GIN_ENV" in
  *NCCL_GIN_TYPE=5*) GIN_TAG="_gin5" ;;
  *)                 GIN_TAG="_type2" ;;
esac

# --ignore-local-traffic changes ONLY the byte accounting, never the buffer calls
# (test_ep.py:254-345 adjusts num_scaleout_send_tokens / num_scaleup_recv_tokens and
# the combine byte counts; dispatch_args and combine_args are untouched). So the us
# column is identical either way, and SO/SU are NOT: with it, SO counts only bytes
# leaving this scale-out group, i.e. true cross-node traffic, which at 2 nodes is
# about half of what the default prints. Default OFF because every number published
# under results/ and in the runbook was measured that way, and mixing the two
# denominators in one document is how a scaling conclusion gets inverted. It goes in
# the tag whenever it is on, so the two can never pool.
IGNORE_LOCAL="${IGNORE_LOCAL-}"
IGL_TAG=""
[ -z "$IGNORE_LOCAL" ] || IGL_TAG="_igl1"

# --prefer-overlap-with-compute. 0 is the working point and every number under
# results/ predating 2026-08-31 was measured there. Unlike IGNORE_LOCAL above this
# one is stamped in BOTH states (`_ovlp0` / `_ovlp1`), because it is not a
# reporting flag -- it selects a different kernel configuration (double-buffering
# and warp counts), and on PR #8+#9 it additionally gates the channel-clamp removal
# and the forward-warp pairing, which is what makes a main-vs-#9 pair at =1 a
# bracket on the other two changes without a rebuild. Two configurations that must
# never be pooled both deserve a visible segment, not one silent default.
# It is per-CELL (the 7th CELLS field) so that an overlap A/B rotates inside each
# rep like every other axis; this is only the default for cells that omit it.
PREFER_OVERLAP="${PREFER_OVERLAP:-0}"

ARCH="${1:-}"
if [ -z "$ARCH" ]; then
  cc=$($SSH -n "$LEADER" 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1' | tr -d ' \r')
  case "$cc" in
    9.0)  ARCH=sm90  ;;
    10.0) ARCH=sm100 ;;
    10.3) ARCH=sm103 ;;
    *) echo "cannot infer arch from leader compute_cap='$cc' -- pass sm90|sm103" >&2; exit 1 ;;
  esac
  echo "=== ARCH=$ARCH (probed from $LEADER, compute_cap $cc) ==="
fi

IMAGE_BASE="${IMAGE_BASE:-deepep-v2-efa-official:${ARCH}}"
IMAGE_PRS="${IMAGE_PRS:-deepep-v2-efa-official:${ARCH}-bfbdd15}"

if [ -z "${MASTER_IP:-}" ]; then
  MASTER_IP=$($SSH -n "$LEADER" 'hostname -I | awk "{print \$1}"' | tr -d ' \r')
  echo "=== MASTER_IP=$MASTER_IP (private ip of $LEADER) ==="
fi

# Default matrices. 12 SM is the OPERATING POINT on every arch here: it is
# run_test_ep.sh's default and AWS's published point, so the `prs` arm is measured
# there and nowhere else. 24 SM is carried on the `official` arm only, as an axis:
# it buys reduced-combine time (2N layer total -14.7%) at +2.2% dispatch, and with
# the PRs applied 12 SM wins decode outright, so it is a trade to measure rather
# than a default to adopt. Keeping the two arches' cell lists identical is
# deliberate -- a b300-vs-p5en comparison at different SM counts is not one.
if [ -z "${CELLS:-}" ]; then
  case "$ARCH" in
    sm90|sm100|sm103) CELLS="
official|$IMAGE_BASE|8192|12|qpdefault|
official|$IMAGE_BASE|128|12|qpdefault|
official|$IMAGE_BASE|8192|24|qpdefault|
official|$IMAGE_BASE|128|24|qpdefault|
prs|$IMAGE_PRS|8192|12|prsdflt|
prs|$IMAGE_PRS|128|12|prsdflt|
prs|$IMAGE_PRS|128|12|prsmtpp1|EP_MIN_TOKENS_PER_PART=1
" ;;
  esac
fi

# `prsmtpp1` is the clamp-OFF control: PR #2 short-circuits to the pre-patch
# geometry when EP_MIN_TOKENS_PER_PART=1, so it is the old behaviour inside the
# NEW binary. If it does not land on the `official` arm, the difference came from
# the build or the environment and not from the clamp.

have_prs=1
if ! $SSH -n "$LEADER" "docker image inspect $IMAGE_PRS >/dev/null 2>&1"; then
  have_prs=0
  echo "=== $IMAGE_PRS absent on $LEADER -- skipping the prs arm."
  echo "    Build it with: ./build_image.sh $ARCH bfbdd15ff448783f877cb2210cb3246c8452b05e"
  echo "    (that is PR #2's head as of 2026-08-31; a rebase moves it --"
  echo "     gh pr view 2 --repo amazon-contributing/DeepEP --json headRefOid --jq .headRefOid)"
fi

port=$((PORT_BASE))
ok=0; bad=0; skipped=0
declare -a TAGS=()

cell () {   # $1 arm  $2 image  $3 tokens  $4 sms  $5 knobtag  $6 extra env  $7 rep  $8 ovlp
  local arm=$1 img=$2 tok=$3 sms=$4 knob=$5 extra=$6 rep=$7 ovlp=${8:-$PREFER_OVERLAP}
  local tag env rc pids i
  if [ "$arm" = prs ] && [ "$have_prs" = 0 ]; then skipped=$((skipped + 1)); return 0; fi
  case "$ovlp" in 0|1) ;; *) echo "ovlp must be 0 or 1, got '$ovlp'" >&2; exit 1 ;; esac
  # PREFER_OVERLAP is read by run_test_ep.sh from its OWN environment (it becomes a
  # CLI arg, not a container env var), so putting it in the EXTRA_ENV column only
  # forwards a docker -e that nothing reads -- a silent no-op that would label two
  # identical runs as different arms. Refuse it there.
  case " $extra " in
    *" PREFER_OVERLAP="*)
      echo "PREFER_OVERLAP in the EXTRA_ENV column is a no-op (it is a CLI arg, not" >&2
      echo "container env). Use the 7th CELLS field or the campaign-level variable." >&2
      exit 1 ;;
  esac
  tag="${arm}_${NNODES}N_${sms}sm_${tok}tok_${knob}_nodbg${GIN_TAG}${IGL_TAG}_ovlp${ovlp}_rep${rep}"
  port=$((port + 1))
  # EP_BUFFER_DEBUG is never set here: it printf()s from inside dispatch's host
  # polling loop, i.e. inside the timed region, and only some launchers forward it.
  # GIN_ENV goes through as GIN_ENV, not folded into EXTRA_ENV: run_test_ep.sh
  # DEFAULTS it to the type-5 pair, so an empty GIN_ENV folded into EXTRA_ENV
  # would silently come back as type 5 while the tag still said _type2.
  env="IMAGE=$img WORLD_SIZE=$NNODES NUM_PROCESSES=$NUM_PROCESSES TOKENS=$tok \
NUM_SMS=$sms MASTER_PORT=$port IGNORE_LOCAL='$IGNORE_LOCAL' NCCL_DEBUG=WARN TEST_FIRST_ONLY=1 \
PREFER_OVERLAP=$ovlp GIN_ENV='$GIN_ENV' EXTRA_ENV='$extra'"

  echo "=== $tag  (port $port)"
  pids=()
  # Workers first, leader last: the leader creates the TCPStore, workers retry.
  # Foreground ssh, never nohup: a detached launch fails asymmetrically on a
  # broken pipe, and the survivor's retry can overwrite a published log.
  for ((i = NNODES - 1; i >= 0; i--)); do
    $SSH -n "${NODE_ARR[$i]}" \
      "mkdir -p $LOGDIR && cd $REPO_DIR && $env bash run_test_ep.sh $i $MASTER_IP \
       > $LOGDIR/$tag.node$((i + 1)).log 2>&1" &
    pids+=($!)
    sleep 1
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  if [ "$rc" = 0 ]; then ok=$((ok + 1)); echo "    rc 0"
  else bad=$((bad + 1)); echo "    !! nonzero rc -- inspect $tag.node*.log"; fi
  TAGS+=("$tag")
  sleep 20   # let GPU memory drop before the next cell's busy-GPU preflight
}

echo "=== $NNODES nodes ($NODES), leader $LEADER, $REPS reps, GIN='${GIN_ENV:-<none, type 2>}'"
for rep in $(seq 1 "$REPS"); do
  echo "########## rep $rep ##########"
  while IFS='|' read -r arm img tok sms knob extra ovlp; do
    [ -n "${arm:-}" ] || continue
    cell "$arm" "$img" "$tok" "$sms" "$knob" "${extra:-}" "$rep" "${ovlp:-$PREFER_OVERLAP}"
  done <<< "$CELLS"
done

echo
echo "=== cells ok=$ok bad=$bad skipped=$skipped"
echo "=== fetch and check (every node writes its OWN logs -- combine is layered by"
echo "    node, so a table built from the leader alone is wrong):"
# LOGDIR is kept unexpanded ($HOME/...) because it is evaluated by the remote
# shell in the run commands above -- but scp since OpenSSH 9 speaks sftp, which
# does NOT run a remote shell, so a `$HOME` in an scp path fails with
# `remote readdir("$HOME/..."): No such file or directory`. Resolve it once here so
# the printed command is one that actually works when pasted.
LOGDIR_ABS=$($SSH -n "$LEADER" "eval echo \"$LOGDIR\"" | tr -d ' \r')
for ((i = 0; i < NNODES; i++)); do
  echo "    scp '${NODE_ARR[$i]}:${LOGDIR_ABS:-$LOGDIR}/*.node$((i + 1)).log' ./logs/"
done
echo "    ./verify_run.sh logs/*.log        # acceptance checks, per log"
echo "    EPRUNS=./logs python3 results/*/parse_ep.py <tag>   # per-tag, all ranks pooled"
printf '%s\n' "${TAGS[@]}" | sort -u

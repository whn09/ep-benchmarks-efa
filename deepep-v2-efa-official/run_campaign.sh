#!/usr/bin/env bash
# The ONLY campaign driver in this repo. Runs from your laptop; checks the
# preconditions that produce a wrong number rather than an error, drives
# run_test_ep.sh on every node over ssh one cell at a time, and names every log so
# that results/*/make_*_tables.py can pool it.
#
#   NODES="<leader> <worker> [...]" [PRESET=...] ./run_campaign.sh [ARCH]
#
# ARCH   sm90 | sm100 | sm103   (default: probed from the leader's compute_cap). It
#        only selects image tags and the default cell list -- no arch-specific code.
#
# PRESET selects the matrix. This is what used to be a per-campaign wrapper script;
#        a campaign is a CELLS list plus the assertions that make it subtractable
#        from the last one, and both belong next to the driver that runs them.
#
#   default   the general-purpose 2-arm list: `official` x {8192,128} tok x
#             {12,24} SM, plus the `prs` arm at 12 SM and its
#             EP_MIN_TOKENS_PER_PART=1 control. Not tied to a published campaign.
#   stack     4 arms -- upstream main, amazon-contributing/DeepEP #1+#2, #8+#9, and
#             the merged stack -- at 12 SM, both token counts, plus `subparts1`
#             (EP_NUM_SUB_PARTS=1) on the arms that read it. Reproduces
#             results/p5en_stack_20260831 and results/b300_stack_20260903.
#   smsweep   the same 4 arms x both token counts at every SM count in $SMS
#             (default 24), default part geometry, plus two 12 SM stack cells as a
#             drift anchor. Reproduces results/b300_sm24_20260903 with
#             SMS=24 PORT_BASE=8700.
#
# Setting CELLS yourself overrides PRESET entirely; every check below still runs.
#
# Env:
#   NODES          REQUIRED. ssh aliases/hosts, space separated, FIRST = leader.
#                  WORLD_SIZE is the count of these, per test_ep.py's convention
#                  (it is not torchrun: WORLD_SIZE = nodes, RANK = node index).
#   MASTER_IP      leader's PRIVATE ip. Default: asked of the leader over ssh.
#                  A public ip works for ssh and not for the rendezvous.
#   SMS="24"       smsweep only: SM counts to sweep, space separated. `SMS="24 48"`
#                  adds route B's dispatch optimum without editing anything; the
#                  12 SM anchor is added on top regardless.
#   ANCHOR=1       smsweep only. 0 drops the two 12 SM anchor cells (do not: they
#                  are what makes a 24-vs-12 delta checkable inside one campaign).
#   STACK_SHA_EXPECT  smsweep only, default a35285f -- the merge sha
#                  results/b300_stack_20260903 measured. The 24-vs-12 comparison
#                  crosses two trees if the stack image is a different merge, so
#                  this is asserted rather than trusted. Clear it (STACK_SHA_EXPECT=)
#                  when you deliberately measure a new stack, and then re-run the
#                  anchor cells instead of subtracting the old campaign.
#   IMAGE_BASE     image for the `official` arm. Default deepep-v2-efa-official:<arch>
#   IMAGE_PRS      image for the `prs` arm (amazon-contributing/DeepEP #1 + #2).
#                  Default deepep-v2-efa-official:<arch>-bfbdd15 (PR #2's head).
#   REPS=3         reps are ROTATED (every cell once per rep), not blocked per arm,
#                  so a slow drift cannot be mistaken for an arm effect.
#   NUM_PROCESSES=8   local ranks per node
#   GIN_ENV        default "NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0" (the
#                  same default run_test_ep.sh applies on its own). Set to "" to
#                  measure the type-2 proxy backend on purpose; the tag then says
#                  _type2 instead of _gin5 so the two never pool.
#   PORT_BASE=8500 a killed run leaves a TCPStore listener behind, so every cell
#                  gets its own port. Move it off 8500 when a previous campaign on
#                  these hosts was killed mid-cell.
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
#   DRY=1          print the resolved matrix and the image inventory, run nothing.
#   FORCE=1        run even if the preconditions below are not met. Every number
#                  produced that way is suspect; there is no tag for it.
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
SSH_N="ssh -n -o ConnectTimeout=10"
PRESET="${PRESET:-default}"
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

# The pinned trees the 4-arm presets measure. Two campaigns are only subtractable
# if their arm labels came from the same shas, so these are constants here and not
# something a caller passes in.
MAIN_REF=54fffeff810723f574c574b1790dff189f3c6ffb   # upstream main
PR12_REF=bfbdd15ff448783f877cb2210cb3246c8452b05e   # amazon-contributing/DeepEP #1 + #2
PR89_REF=3c737dcf0da5889ba7efd26e05b4808307cc38af   # #8 + #9
IMG_MAIN="deepep-v2-efa-official:${ARCH}-${MAIN_REF:0:7}"
IMG_PR12="deepep-v2-efa-official:${ARCH}-${PR12_REF:0:7}"
IMG_PR89="deepep-v2-efa-official:${ARCH}-${PR89_REF:0:7}"
IMG_STACK="deepep-v2-efa-official:${ARCH}-stack${PR89_REF:0:7}x${PR12_REF:0:7}"

if [ -z "${MASTER_IP:-}" ]; then
  MASTER_IP=$($SSH -n "$LEADER" 'hostname -I | awk "{print \$1}"' | tr -d ' \r')
  echo "=== MASTER_IP=$MASTER_IP (private ip of $LEADER) ==="
fi

# ---------------------------------------------------------------- inventory ----
# Which images exist on EVERY node, and what code is actually inside them. Both
# halves matter: an image present only on the leader fails the cell N times one at
# a time, and an image whose BUILD_REF differs per node runs two different trees
# under one arm label -- which is not a failure at all, just a wrong number.
# macOS drives this (bash 3.2), so the table is a newline-delimited string rather
# than an associative array.
INV=""       # <image>|<node count>|<comma-joined unique BUILD_REFs>
inv_probe () {
  local img n c refs r u
  for img in $(printf '%s\n' "$@" | sort -u); do
    [ -n "$img" ] || continue
    c=0; refs=""
    for n in "${NODE_ARR[@]}"; do
      if $SSH_N "$n" "docker image inspect $img >/dev/null 2>&1"; then
        c=$((c + 1))
        r=$($SSH_N "$n" "docker run --rm --entrypoint cat $img /opt/DeepEP/BUILD_REF 2>/dev/null" | tr -d ' \r')
        refs="$refs ${r:-noref}"
      fi
    done
    # shellcheck disable=SC2086
    u=$(printf '%s\n' $refs | sort -u | paste -sd, -)
    INV="$INV$img|$c|$u
"
    printf '  %-56s %s/%s nodes  %s\n' "$img" "$c" "$NNODES" "${u:-<absent>}"
    case "$u" in
      *,*) echo "!! $img has a different BUILD_REF per node ($u): the nodes would" >&2
           echo "   run different code under one arm label. Rebuild or re-pull." >&2
           exit 2 ;;
    esac
  done
}
inv_field () { printf '%s' "$INV" | awk -F'|' -v i="$1" -v f="$2" '$1==i{print $f; exit}'; }
have_all  () { [ "$(inv_field "$1" 2)" = "$NNODES" ]; }
inv_ref   () { inv_field "$1" 3; }

# The tag says which build you asked for; BUILD_REF says which one you got. A
# retagged or rebuilt-from-a-moved-branch image is the one failure mode that
# survives every other check in this file.
ref_is () {   # $1 image  $2 expected sha (any length)
  local got; got=$(inv_ref "$1")
  case "$got" in
    "$2"*) return 0 ;;
    noref) echo "  WARN $1 has no /opt/DeepEP/BUILD_REF -- cannot verify it is $2" ;;
    *) echo "!! $1 should be $2 but its BUILD_REF is $got" >&2; exit 2 ;;
  esac
}

# ------------------------------------------------------------------ presets ----
CELLS_IN="${CELLS:-}"
PRESET_SHOWN="$PRESET"
if [ -n "$CELLS_IN" ]; then
  PRESET_SHOWN="<explicit CELLS>"
  echo "=== CELLS given explicitly; PRESET=$PRESET ignored"
  echo "=== images referenced by CELLS:"
  # shellcheck disable=SC2046
  inv_probe $(printf '%s\n' "$CELLS_IN" | awk -F'|' '$1 != "" && NF > 1 {print $2}')
else
  add () { CELLS_IN="${CELLS_IN}$1
"; }
  case "$PRESET" in
    default)
      # 12 SM is the OPERATING POINT on every arch here: it is run_test_ep.sh's
      # default and AWS's published point, so the `prs` arm is measured there and
      # nowhere else. 24 SM is carried on the `official` arm only, as an axis: it
      # buys reduced-combine time (2N layer total -14.7%) at +2.2% dispatch, and
      # with the PRs applied 12 SM wins decode outright, so it is a trade to
      # measure rather than a default to adopt. Keeping the two arches' cell lists
      # identical is deliberate -- a b300-vs-p5en comparison at different SM counts
      # is not one.
      echo "=== images (PRESET=default):"
      inv_probe "$IMAGE_BASE" "$IMAGE_PRS"
      add "official|$IMAGE_BASE|8192|12|qpdefault|"
      add "official|$IMAGE_BASE|128|12|qpdefault|"
      add "official|$IMAGE_BASE|8192|24|qpdefault|"
      add "official|$IMAGE_BASE|128|24|qpdefault|"
      add "prs|$IMAGE_PRS|8192|12|prsdflt|"
      add "prs|$IMAGE_PRS|128|12|prsdflt|"
      # `prsmtpp1` is the clamp-OFF control: PR #2 short-circuits to the pre-patch
      # geometry when EP_MIN_TOKENS_PER_PART=1, so it is the old behaviour inside
      # the NEW binary. If it does not land on the `official` arm, the difference
      # came from the build or the environment and not from the clamp.
      add "prs|$IMAGE_PRS|128|12|prsmtpp1|EP_MIN_TOKENS_PER_PART=1"
      ;;
    stack|smsweep)
      echo "=== images (PRESET=$PRESET):"
      inv_probe "$IMG_MAIN" "$IMG_PR12" "$IMG_PR89" "$IMG_STACK"
      if have_all "$IMG_MAIN"; then ref_is "$IMG_MAIN" "$MAIN_REF"; fi
      if have_all "$IMG_PR12"; then ref_is "$IMG_PR12" "$PR12_REF"; fi
      if have_all "$IMG_PR89"; then ref_is "$IMG_PR89" "$PR89_REF"; fi
      # The stack arm is labelled by its MERGE sha, not by its tag: the merge is
      # made inside the image (Dockerfile.stack pins GIT_AUTHOR_DATE/
      # GIT_COMMITTER_DATE so it is reproducible), so the label has to be read out
      # of the image rather than computed from the two parents.
      A_MAIN="main${MAIN_REF:0:7}"; A_PR12="pr12${PR12_REF:0:7}"
      A_PR89="pr89${PR89_REF:0:7}"; A_STACK=""
      if have_all "$IMG_STACK"; then
        A_STACK="stack$(inv_ref "$IMG_STACK" | cut -c1-7)"
        echo "  stack arm label: $A_STACK"
      fi
      ;;
    *) echo "unknown PRESET='$PRESET' (default|stack|smsweep), or set CELLS" >&2; exit 1 ;;
  esac

  case "$PRESET" in
    stack)
      # The matrix is deliberately the same as results/p5en_stack_20260831, so
      # b300 and p5en rows are comparable cell by cell.
      # subparts1 = EP_NUM_SUB_PARTS=1, which only the #1+#2 JIT reads, so it rides
      # only the two arms that contain #1+#2. One main+subparts1 cell is kept as the
      # no-op control: if it moves, the knob is reaching something it should not.
      # This is why the knob axis is NOT a cross product -- pr89 x subparts1 would
      # be a cell that measures nothing.
      add "$A_MAIN|$IMG_MAIN|8192|12|qpdefault|"
      add "$A_MAIN|$IMG_MAIN|128|12|qpdefault|"
      add "$A_MAIN|$IMG_MAIN|128|12|subparts1|EP_NUM_SUB_PARTS=1"
      add "$A_PR12|$IMG_PR12|8192|12|qpdefault|"
      add "$A_PR12|$IMG_PR12|128|12|qpdefault|"
      add "$A_PR12|$IMG_PR12|128|12|subparts1|EP_NUM_SUB_PARTS=1"
      add "$A_PR12|$IMG_PR12|8192|12|subparts1|EP_NUM_SUB_PARTS=1"
      add "$A_PR89|$IMG_PR89|8192|12|qpdefault|"
      add "$A_PR89|$IMG_PR89|128|12|qpdefault|"
      [ -z "$A_STACK" ] || {
        add "$A_STACK|$IMG_STACK|8192|12|qpdefault|"
        add "$A_STACK|$IMG_STACK|128|12|qpdefault|"
        add "$A_STACK|$IMG_STACK|128|12|subparts1|EP_NUM_SUB_PARTS=1"
        add "$A_STACK|$IMG_STACK|8192|12|subparts1|EP_NUM_SUB_PARTS=1"
      }
      ;;
    smsweep)
      # Does #8+#9's combine win survive once combine has the SMs? At 12 SM the
      # question is undecidable: route B's own sweep says 12 -> 24 SM is worth
      # -36.1% on combine by itself, so a win at 12 SM may only be recovering SM
      # starvation. Everything here is the DEFAULT part geometry -- on b300
      # EP_NUM_SUB_PARTS=1 is inert on decode and costs the stack +102 us on
      # prefill dispatch, so it is not the deployed arm.
      SMS="${SMS:-24}"
      ANCHOR="${ANCHOR:-1}"
      STACK_SHA_EXPECT="${STACK_SHA_EXPECT-a35285f}"
      if [ -n "$A_STACK" ] && [ -n "$STACK_SHA_EXPECT" ] \
         && [ "$A_STACK" != "stack$STACK_SHA_EXPECT" ]; then
        echo "!! this stack image is $A_STACK, but results/b300_stack_20260903" >&2
        echo "   measured stack$STACK_SHA_EXPECT -- a 24-vs-12 SM delta taken across" >&2
        echo "   the two campaigns would cross two trees. STACK_SHA_EXPECT= to" >&2
        echo "   override, and then take the delta inside this campaign only." >&2
        exit 2
      fi
      for sm in $SMS; do
        for tok in 8192 128; do
          add "$A_MAIN|$IMG_MAIN|$tok|$sm|qpdefault|"
          add "$A_PR12|$IMG_PR12|$tok|$sm|qpdefault|"
          add "$A_PR89|$IMG_PR89|$tok|$sm|qpdefault|"
          [ -z "$A_STACK" ] || add "$A_STACK|$IMG_STACK|$tok|$sm|qpdefault|"
        done
      done
      # The 12 SM anchor is the point of this being one campaign: it re-measures a
      # cell results/b300_stack_20260903 already holds, so a 24-vs-12 delta taken
      # inside this campaign can be checked against the same delta taken across
      # campaigns. If the anchor has drifted, the cross-campaign number is what is
      # wrong, not this table.
      if [ "$ANCHOR" = 1 ] && [ -n "$A_STACK" ]; then
        add "$A_STACK|$IMG_STACK|8192|12|qpdefault|"
        add "$A_STACK|$IMG_STACK|128|12|qpdefault|"
      fi
      ;;
  esac
fi

# Drop the cells whose image is not on every node, once and with a reason, rather
# than failing them one at a time over the next hour.
CELLS=""
skipped=0
while IFS='|' read -r arm img tok sms knob extra ovlp; do
  [ -n "${arm:-}" ] || continue
  if have_all "$img"; then
    CELLS="$CELLS$arm|$img|$tok|$sms|$knob|${extra:-}|${ovlp:-}
"
  else
    skipped=$((skipped + 1))
    echo "=== skip $arm ${tok}tok ${sms}sm $knob -- $img on $(inv_field "$img" 2)/$NNODES nodes"
    if [ "$img" = "$IMAGE_PRS" ]; then
      echo "    build it with: ./build_image.sh $ARCH $PR12_REF"
      echo "    (that is PR #2's head as of 2026-08-31; a rebase moves it --"
      echo "     gh pr view 2 --repo amazon-contributing/DeepEP --json headRefOid --jq .headRefOid)"
    fi
  fi
done <<< "$CELLS_IN"

# ----------------------------------------------------------- preconditions ----
# Both of these produce a WRONG NUMBER rather than an error, which is why they are
# checked here and not left to the run:
#   1. efa.ko >= 3.3.0 (EFA installer 1.50.0). Without it NCCL_GIN_TYPE=5 falls
#      back to the type-2 CPU proxy while the log tag still says _gin5 -- run
#      prepare_host_efa150.sh. Only checked when type 5 is what we asked for.
#   2. Idle GPUs. test_ep.py allocates a large slab per rank, and leaked ranks from
#      a previous run make the next one report ~2x the latency at rc=0.
step_ok=1
echo "=== preconditions:"
for n in "${NODE_ARR[@]}"; do
  kv=$($SSH_N "$n" 'modinfo efa 2>/dev/null | awk "/^version:/{print \$2}"' | tr -d ' \r')
  apps=$($SSH_N "$n" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' | tr -d ' \r')
  printf '  %-10s efa.ko=%-8s compute_procs=%s\n' "$n" "${kv:-?}" "$apps"
  if [ "$GIN_TAG" = _gin5 ]; then
    ge=$(awk -v v="$kv" 'BEGIN{n=split(v,a,"."); print (n>=2 && (a[1]+0>3 || (a[1]+0==3 && a[2]+0>=3))) ? 1 : 0}')
    [ "$ge" = 1 ] || { echo "     !! GIN type 5 needs efa.ko >= 3.3.0 (prepare_host_efa150.sh)"; step_ok=0; }
  fi
  [ "$apps" = 0 ] || { echo "     !! GPUs busy"; step_ok=0; }
done

ncells=$(printf '%s' "$CELLS" | grep -c . || true)
echo "=== cells ($ncells per rep, REPS=$REPS, PRESET=$PRESET_SHOWN, skipped=$skipped):"
printf '%s' "$CELLS" | sed 's/^/  /'
echo "=== ~$(( (ncells * REPS * 100) / 60 )) min at ~100 s/cell including the 20 s settle"
[ "$ncells" != 0 ] || { echo "no cells to run."; exit 1; }

# DRY first, so the matrix can be reviewed while the hosts are still busy and still
# on the old kmod -- which is exactly when you want to review it.
[ "${DRY:-}" != 1 ] || { echo; echo "DRY=1, nothing run."; exit 0; }
if [ "$step_ok" != 1 ]; then
  [ "${FORCE:-}" = 1 ] || { echo; echo "REFUSING: preconditions above are not met (FORCE=1 to override)."; exit 3; }
  echo; echo "!! FORCE=1: running with the preconditions above unmet. Nothing in the log"
  echo "   name records that, so treat every number from this campaign as suspect."
fi

# ---------------------------------------------------------------- the runs ----
port=$((PORT_BASE))
ok=0; bad=0
declare -a TAGS=()

cell () {   # $1 arm  $2 image  $3 tokens  $4 sms  $5 knobtag  $6 extra env  $7 rep  $8 ovlp
  local arm=$1 img=$2 tok=$3 sms=$4 knob=$5 extra=$6 rep=$7 ovlp=${8:-$PREFER_OVERLAP}
  local tag env rc pids i
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

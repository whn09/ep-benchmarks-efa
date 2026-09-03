#!/usr/bin/env bash
# Build the "两组 PR 叠在一起" image from Dockerfile.stack, on THIS host.
#
#   usage: ./build_stack_image.sh [ARCH] [REF_A] [REF_B] [TAG]
#     ARCH   sm90 | sm100 | sm103  (default: probed from nvidia-smi)
#     REF_A  branch checked out first  (default: PR #8+#9 head, 3c737dc...)
#     REF_B  branch merged onto it     (default: PR #1+#2 head, bfbdd15...)
#     TAG    default deepep-v2-efa-official:<arch>-stack<A7>x<B7>
#
# It layers on the single-arm image for REF_A rather than rebuilding the CUDA base,
# so the cost is one `setup.py install` instead of a full build -- but that also
# means the base image must already exist for THIS arch, and the script says so
# instead of starting a 40-minute build that ends in "no such image".
#
# The tag records BOTH shas, because "stack" alone is not a configuration: which two
# things were stacked is the whole content of the experiment.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-}"
REF_A="${2:-3c737dcf0da5889ba7efd26e05b4808307cc38af}"   # PR #8+#9 head, 2026-08-31
REF_B="${3:-bfbdd15ff448783f877cb2210cb3246c8452b05e}"   # PR #1+#2 head, 2026-08-31
TAG="${4:-}"

if [ -z "$ARCH" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  case "$cc" in
    9.0) ARCH=sm90 ;; 10.0) ARCH=sm100 ;; 10.3) ARCH=sm103 ;;
    *) echo "cannot infer arch from compute_cap='$cc' -- pass sm90|sm100|sm103" >&2; exit 1 ;;
  esac
  echo "=== ARCH=$ARCH (probed, compute_cap $cc) ==="
fi

# A PR head moves when the branch is rebased, and a 7-char prefix in a tag is not
# enough to notice. Resolve both to full shas up front and fail on a typo here
# rather than inside the ADD layer as `invalid response status 422`.
for r in "$REF_A" "$REF_B"; do
  case "$r" in
    *[!0-9a-f]*|"") echo "REF must be a full 40-char sha (a branch name would float): '$r'" >&2
                    exit 1 ;;
  esac
  [ "${#r}" = 40 ] || { echo "REF must be 40 chars: '$r' (${#r})" >&2; exit 1; }
done

BASE="deepep-v2-efa-official:${ARCH}-${REF_A:0:7}"
if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  echo "base image $BASE does not exist on this host." >&2
  echo "  ./build_image.sh $ARCH $REF_A" >&2
  exit 1
fi
TAG="${TAG:-deepep-v2-efa-official:${ARCH}-stack${REF_A:0:7}x${REF_B:0:7}}"

echo "=== $TAG = $BASE + merge($REF_B)"
docker build -f Dockerfile.stack -t "$TAG" \
  --build-arg "BASE=$BASE" --build-arg "REF_A=$REF_A" --build-arg "REF_B=$REF_B" .

# What actually got installed, printed once so it lands in the build log next to the
# tag. The four content assertions are inside the Dockerfile (a failing build is a
# louder signal than a line in a log nobody reads).
docker run --rm --entrypoint cat "$TAG" /opt/DeepEP/BUILD_REF
docker run --rm --entrypoint cat "$TAG" /opt/DeepEP/BUILD_REF_PARENTS
echo "=== built $TAG"

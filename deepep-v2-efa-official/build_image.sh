#!/usr/bin/env bash
# Build the image for THIS host's GPU arch.
#
# It wraps the two build args that decide whether the image can run at all, so
# that "I forgot --build-arg CUDA_VERSION" cannot happen: on sm_103 that omission
# does not fail the build, it fails at the FIRST dispatch with a ptxas error (see
# the Dockerfile header).
#
# usage: ./build_image.sh [ARCH] [DEEPEP_REF] [TAG]
#   ARCH        sm90 | sm100 | sm103   (default: probed from nvidia-smi)
#   DEEPEP_REF  branch / tag / bare sha (default: the Dockerfile's default, `main`)
#   TAG         image tag (default: deepep-v2-efa-official:<arch>-<sha7>, plus the
#               plain :<arch> alias when no ref was asked for)
#
# The tag carries the arch on purpose. A Hopper cubin does not run on Blackwell
# and vice versa, so two arches sharing one tag is the single most expensive
# mistake available here -- and `docker run` gives no hint, the failure surfaces
# as a kernel launch error deep in the test.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-}"
DEEPEP_REF="${2:-}"
TAG="${3:-}"

if [ -z "$ARCH" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  case "$cc" in
    9.0)  ARCH=sm90  ;;
    10.0) ARCH=sm100 ;;
    10.3) ARCH=sm103 ;;
    *) echo "cannot infer arch from compute_cap='$cc' -- pass it: $0 sm90|sm100|sm103" >&2
       exit 1 ;;
  esac
  echo "=== ARCH=$ARCH (probed, compute_cap $cc) ==="
fi

# sm_103 is NOT a backward-compatible target of sm_100, and its ptxas floor is
# higher than sm_90's: 13.0.2 rejects the `#if __CUDA_ARCH__ >= 1000` PTX in
# deep_ep/include/deep_ep/common/ptx.cuh. One arch per image -- `9.0;10.3` would
# need two different CUDA bases anyway.
case "$ARCH" in
  sm90)  CUDA_VERSION=13.0.2; TORCH_CUDA_ARCH_LIST=9.0  ;;
  sm100) CUDA_VERSION=13.3.1; TORCH_CUDA_ARCH_LIST=10.0 ;;   # not run here
  sm103) CUDA_VERSION=13.3.1; TORCH_CUDA_ARCH_LIST=10.3 ;;
  *) echo "unknown ARCH=$ARCH" >&2; exit 1 ;;
esac

# Resolve whatever was asked for -- branch, tag, or sha -- to a 40-char sha BEFORE
# building, and name the image after the sha. Three things depend on doing it here:
#   - The Dockerfile's default FLOATS (`main`). Without resolution, a plain
#     `./build_image.sh sm90` would put differently-built code under the same tag
#     every time upstream pushes, and nothing in `docker images` would say so.
#   - A wrong sha does not fail the build early; it fails inside the `ADD
#     .../commits/${DEEPEP_REF}` layer as `invalid response status 422`, several
#     cached layers in. One API call up front turns that into a one-line error.
#   - Passing the sha (not `main`) makes the ADD's cache key constant, so rebuilding
#     the same commit reuses the layer instead of re-resolving it.
REF_SPEC="${DEEPEP_REF:-main}"
API="https://api.github.com/repos/amazon-contributing/DeepEP/commits/${REF_SPEC}"
# The `.sha` media type returns the bare sha as the body, so this needs no jq/gh.
SHA=$(curl -fsSL -H 'Accept: application/vnd.github.sha' "$API" 2>/dev/null | tr -dc 'a-f0-9' || true)
if [ "${#SHA}" != 40 ]; then
  echo "could not resolve DeepEP ref '$REF_SPEC' to a sha." >&2
  echo "  curl -sSL -H 'Accept: application/vnd.github.sha' $API" >&2
  echo "A branch/tag typo, a sha that does not exist, or the unauthenticated GitHub" >&2
  echo "rate limit (60/hour) all land here. For a PR head:" >&2
  echo "  gh pr view <n> --repo amazon-contributing/DeepEP --json headRefOid --jq .headRefOid" >&2
  exit 1
fi
echo "=== DeepEP ref '$REF_SPEC' -> $SHA ==="
REF_ARGS=(--build-arg "DEEPEP_REF=$SHA")
TAG_SHA="deepep-v2-efa-official:${ARCH}-${SHA:0:7}"
TAG="${TAG:-$TAG_SHA}"

# Extra names for the same image id. The sha tag is what you cite and archive; the
# plain :<arch> alias exists because run_campaign.sh's IMAGE_BASE defaults to it, and
# it is only attached when no ref was asked for (i.e. this IS the default build).
ALIASES=()
[ "$TAG" = "$TAG_SHA" ] || ALIASES+=("$TAG_SHA")
[ -n "$DEEPEP_REF" ] || ALIASES+=("deepep-v2-efa-official:${ARCH}")

# The installer tarball (~650 MB) is fetched here, not committed. The public bucket
# serves it under a VERSIONED name, so this needs no rename and no dev bucket:
#   https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
# 1.50.0 was dev-bucket-only (`-latest`) when this kit was written and is now GA.
# The filename still carries the version because that is what makes the image
# self-describing: the version is checked here AND asserted inside the image
# against the tarball's own ChangeLog, so a truncated or swapped file cannot
# quietly become "the stack we measured".
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.50.0}"
TARBALL="aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
if [ ! -f "$TARBALL" ]; then
  URL="https://efa-installer.amazonaws.com/$TARBALL"
  echo "=== fetching $TARBALL (~650 MB) from $URL ==="
  # -f so a 404 is an error instead of an HTML page saved as a tarball; download to
  # a temp name so an interrupted transfer is not mistaken for a complete one next run.
  if ! curl -fSL --retry 3 -o "$TARBALL.part" "$URL"; then
    rm -f "$TARBALL.part"
    echo >&2
    echo "Could not fetch $URL." >&2
    echo "If that version is not GA yet it only exists in the dev bucket under a" >&2
    echo "floating name, which has to be renamed to the version you verified:" >&2
    echo "  curl -O https://aws-efa-installer-dev.s3.amazonaws.com/aws-efa-installer-latest.tar.gz" >&2
    echo "  tar xzOf aws-efa-installer-latest.tar.gz aws-efa-installer/ChangeLog.md | head -4" >&2
    echo "  mv aws-efa-installer-latest.tar.gz $TARBALL   # only if the ChangeLog says it" >&2
    exit 1
  fi
  mv "$TARBALL.part" "$TARBALL"
fi
# Fail here rather than 15 minutes into the build.
got=$(tar xzOf "$TARBALL" aws-efa-installer/ChangeLog.md 2>/dev/null \
      | grep -m1 -o '^## \[[0-9.]*\]' || true)
if [ "$got" != "## [$EFA_INSTALLER_VERSION]" ]; then
  echo "$TARBALL is not installer $EFA_INSTALLER_VERSION (ChangeLog says '${got:-<unreadable>}')." >&2
  echo "Delete it and re-run to re-download." >&2
  exit 1
fi

echo "=== building $TAG  (CUDA $CUDA_VERSION, TORCH_CUDA_ARCH_LIST $TORCH_CUDA_ARCH_LIST, EFA $EFA_INSTALLER_VERSION, DeepEP $REF_SPEC @ ${SHA:0:7}) ==="
set -x
docker build -t "$TAG" \
  --build-arg "CUDA_VERSION=$CUDA_VERSION" \
  --build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
  --build-arg "EFA_INSTALLER_VERSION=$EFA_INSTALLER_VERSION" \
  "${REF_ARGS[@]}" .
set +x

if [ "${#ALIASES[@]}" -gt 0 ]; then
  for a in "${ALIASES[@]}"; do docker tag "$TAG" "$a"; echo "=== also tagged $a"; done
fi

# Never trust the tag for what code is inside: BUILD_REF is what `git rev-parse
# HEAD` said at build time. The `ADD` of the GitHub commit API is what makes a
# moving ref invalidate the layer cache, but a re-tagged image defeats reading.
built_ref=$(docker run --rm --entrypoint cat "$TAG" /opt/DeepEP/BUILD_REF)
echo "=== $TAG"
echo "    DeepEP     $built_ref"
echo "    build arch $(docker run --rm --entrypoint printenv "$TAG" EP_BUILD_ARCH)"
echo "    build CUDA $(docker run --rm --entrypoint printenv "$TAG" EP_BUILD_CUDA)"
echo "    EFA inst.  $(docker run --rm --entrypoint printenv "$TAG" EP_EFA_INSTALLER)"
# The sha was resolved on THIS host before the build; the image's BUILD_REF is what
# the builder's `git fetch` actually landed on. They can only differ if the ADD layer
# was served from a cache built against a different ref -- exactly the failure the ADD
# exists to prevent -- so say so instead of publishing numbers from unknown code.
if [ "$built_ref" != "$SHA" ]; then
  echo >&2
  echo "WARNING: asked for $SHA but the image contains $built_ref." >&2
  echo "  Rebuild with --no-cache before measuring anything from it." >&2
fi
echo
echo "run it with:  IMAGE=$TAG ./run_test_ep.sh <node_rank> <leader_ip>"

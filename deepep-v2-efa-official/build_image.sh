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
#   DEEPEP_REF  branch / tag / bare sha (default: the Dockerfile's pin)
#   TAG         image tag (default: deepep-v2-efa-official:<arch>[-<ref7>])
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

REF_ARGS=()
NICK=""
if [ -n "$DEEPEP_REF" ]; then
  REF_ARGS=(--build-arg "DEEPEP_REF=$DEEPEP_REF")
  NICK="-$(printf '%s' "$DEEPEP_REF" | tr -c 'a-zA-Z0-9' '-' | cut -c1-7)"
fi
TAG="${TAG:-deepep-v2-efa-official:${ARCH}${NICK}}"

# The build context wants a VERSION-named tarball, not `-latest`: the dev bucket's
# `-latest` object will become 1.51.0 one day and the Dockerfile would then build a
# different stack with no record of it. The version is asserted inside the image
# against the tarball's ChangeLog.
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.50.0}"
TARBALL="aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
if [ ! -f "$TARBALL" ]; then
  echo "MISSING $TARBALL in the build context (~650 MB, deliberately not in git)." >&2
  if [ -f aws-efa-installer-latest.tar.gz ]; then
    got=$(tar xzOf aws-efa-installer-latest.tar.gz aws-efa-installer/ChangeLog.md 2>/dev/null \
          | grep -m1 -o '^## \[[0-9.]*\]' || true)
    echo "  Found aws-efa-installer-latest.tar.gz, ChangeLog says ${got:-<unreadable>}." >&2
    echo "  If that is [$EFA_INSTALLER_VERSION]:  mv aws-efa-installer-latest.tar.gz $TARBALL" >&2
  else
    echo "  curl -O https://aws-efa-installer-dev.s3.amazonaws.com/aws-efa-installer-latest.tar.gz" >&2
    echo "  head -6 <(tar xzOf aws-efa-installer-latest.tar.gz aws-efa-installer/ChangeLog.md)" >&2
    echo "  mv aws-efa-installer-latest.tar.gz $TARBALL      # only after checking the ChangeLog" >&2
  fi
  exit 1
fi

echo "=== building $TAG  (CUDA $CUDA_VERSION, TORCH_CUDA_ARCH_LIST $TORCH_CUDA_ARCH_LIST, EFA $EFA_INSTALLER_VERSION${DEEPEP_REF:+, DeepEP $DEEPEP_REF}) ==="
set -x
docker build -t "$TAG" \
  --build-arg "CUDA_VERSION=$CUDA_VERSION" \
  --build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
  --build-arg "EFA_INSTALLER_VERSION=$EFA_INSTALLER_VERSION" \
  "${REF_ARGS[@]}" .
set +x

# Never trust the tag for what code is inside: BUILD_REF is what `git rev-parse
# HEAD` said at build time. The `ADD` of the GitHub commit API is what makes a
# moving ref invalidate the layer cache, but a re-tagged image defeats reading.
echo "=== $TAG"
echo "    DeepEP     $(docker run --rm --entrypoint cat "$TAG" /opt/DeepEP/BUILD_REF)"
echo "    build arch $(docker run --rm --entrypoint printenv "$TAG" EP_BUILD_ARCH)"
echo "    build CUDA $(docker run --rm --entrypoint printenv "$TAG" EP_BUILD_CUDA)"
echo "    EFA inst.  $(docker run --rm --entrypoint printenv "$TAG" EP_EFA_INSTALLER)"
echo
echo "run it with:  IMAGE=$TAG ./run_test_ep.sh <node_rank> <leader_ip>"

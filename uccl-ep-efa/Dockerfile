# syntax=docker/dockerfile:1.10
# =============================================================================
# uccl-ep-efa  -- UCCL-EP V0.x on AWS EFA (cu13)
# =============================================================================
#
# UCCL-EP is a fork of DeepEP that replaces the NVSHMEM transport with UCCL's
# own RDMA stack (ibverbs + libnl + numa). On AWS EFA it talks to the EFA
# rdma-core verbs path directly -- no patched NVSHMEM required.
#
# Stack:
#   - Base:    nvcr.io/nvidia/pytorch:26.04-py3  (CUDA 13.2.1, torch 2.12, sm90)
#   - UCCL:    uccl-project/uccl @ main  (built from source)
#   - EFA:     aws-efa-installer 1.48.0 (provides libfabric + rdma-core libs)
#   - GDRCopy: 2.5.2  (libgdrapi only; kernel module gdrdrv lives on host)
#
# Run example:
#   docker run --rm -it --gpus all --privileged --network=host --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     --device=/dev/infiniband --device=/dev/gdrdrv \
#     uccl-ep-efa:dev bash
# =============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

ARG EFA_INSTALLER_VERSION=1.48.0
ARG GDRCOPY_VERSION=2.5.2
ARG UCCL_REPO=https://github.com/uccl-project/uccl.git
ARG UCCL_REF=main

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. System deps + EFA installer (for libfabric + rdma-core / ibverbs headers)
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git ca-certificates curl wget gnupg \
      build-essential cmake ninja-build \
      pkg-config autoconf libtool patchelf \
      rdma-core libibverbs-dev libnl-3-dev libnl-route-3-dev \
      libnuma-dev libhwloc-dev \
      python3-dev python3-pip; \
    cd /tmp; \
    curl -fSL "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" -o efa.tgz; \
    tar -xzf efa.tgz; \
    cd aws-efa-installer; \
    ./efa_installer.sh -y --skip-kmod -g --no-verify; \
    cd /; rm -rf /tmp/efa.tgz /tmp/aws-efa-installer; \
    ls /opt/amazon/efa/lib/libfabric* /opt/amazon/efa/include/rdma/fabric.h; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 1b. GDRCopy userspace library (libgdrapi). Kernel module gdrdrv lives on the
#     host -- container only needs the userspace .so.
# -----------------------------------------------------------------------------
RUN set -eux; \
    cd /tmp; \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz; \
    tar xf v${GDRCOPY_VERSION}.tar.gz; \
    cd gdrcopy-${GDRCOPY_VERSION}; \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local; \
    ldconfig; \
    test -f /usr/local/lib/libgdrapi.so; \
    cd /; rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# -----------------------------------------------------------------------------
# 2. Python build tools required by uccl/ep/setup.py
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir nanobind pybind11

# -----------------------------------------------------------------------------
# 3. Clone + build uccl/ep against EFA, then install uccl wheel and .so
# -----------------------------------------------------------------------------
# CCCL header path: NGC pytorch:26.04 ships CCCL under
# /usr/local/cuda/targets/x86_64-linux/include/cccl, not /usr/local/cuda/include.
# uccl/ep's CUDA extension transitively includes <cuda/std/...>, so put it on
# the include path for the host C++ compiler. (Same fix as deepep-v1-efa.)
ENV CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH}

RUN set -eux; \
    cd /opt; \
    git clone --recursive ${UCCL_REPO} uccl; \
    cd uccl; \
    git checkout ${UCCL_REF}; \
    git submodule update --init --recursive; \
    cd ep; \
    EFA_HOME=/opt/amazon/efa \
    TORCH_CUDA_ARCH_LIST="9.0+PTX" \
    python setup.py build_ext --inplace; \
    cd /opt/uccl; \
    pip install --no-cache-dir .; \
    # Resolve site-packages uccl from /tmp so import doesn't pick up the in-tree
    # /opt/uccl/uccl source dir.
    UCCL_DIR=$(cd /tmp && python3 -c "import uccl, os; print(os.path.dirname(uccl.__file__))"); \
    echo "UCCL_DIR=${UCCL_DIR}"; \
    find /opt/uccl/ep -name "ep*.so" | head -1 | xargs -I{} cp -v {} "${UCCL_DIR}/"; \
    ls -la "${UCCL_DIR}"/ep*.so; \
    cd /opt/uccl/ep/deep_ep_wrapper; \
    pip install --no-cache-dir --no-deps .; \
    # Verify import surface (libcuda.so.1 not present at build time so just
    # check the .so was placed in the SITE-PACKAGES uccl, not the source tree).
    cd /tmp && python3 -c "import uccl, os; d=os.path.dirname(uccl.__file__); \
so=[f for f in os.listdir(d) if f.startswith('ep') and f.endswith('.so')]; \
assert so, f'ep*.so missing in {d}'; \
print('site-packages uccl dir:', d); print('uccl ep .so:', so)"

# -----------------------------------------------------------------------------
# 4. Default env for AWS EFA
# -----------------------------------------------------------------------------
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1

WORKDIR /opt/uccl/ep/bench
CMD ["/bin/bash"]

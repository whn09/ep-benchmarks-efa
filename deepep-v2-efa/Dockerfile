# syntax=docker/dockerfile:1.10
# =============================================================================
# DeepEP V2 (NCCL GIN backend) on AWS EFA
# =============================================================================
#
# DeepEP V2 (deepseek-ai/DeepEP main, post-2026 refactor) replaces the
# NVSHMEM EP transport with NCCL's "Gin" interface. On AWS EFA, Gin is
# implemented by aws-ofi-nccl (`ncclGinPlugin_v11+`), so the V2 EP path
# runs on EFA *via NCCL + ofi-nccl* with no NVSHMEM involvement at runtime.
#
# Stack:
#   - Base:         nvcr.io/nvidia/pytorch:26.04-py3  (CUDA 13.2.1, torch 2.12, sm90)
#   - DeepEP:       deepseek-ai/DeepEP@main          (V2, JIT)
#   - NCCL:         pip nvidia-nccl-cu13 >=2.30.4    (Gin-aware build)
#   - NVSHMEM:      pip nvidia-nvshmem-cu13           (build-time only; not used
#                                                     at runtime by V2's EP path)
#   - aws-ofi-nccl: built from master                 (provides ncclGinPlugin_v11+)
#   - EFA:          aws-efa-installer 1.48.0          (libfabric)
#   - GDRCopy:      2.5.2                             (libgdrapi)
#
# At runtime, NCCL_NET_PLUGIN must point at the source-built ofi plugin
# (the EFA installer ships its own ofi plugin too, but it lacks GIN).
#
# Run example:
#   docker run --rm -it --gpus all --privileged --network=host --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     --device=/dev/infiniband --device=/dev/gdrdrv \
#     -e NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so \
#     deepep-v2-efa:dev bash
# =============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

ARG EFA_INSTALLER_VERSION=1.48.0
ARG GDRCOPY_VERSION=2.5.2
ARG DEEPEP_REPO=https://github.com/deepseek-ai/DeepEP.git
ARG DEEPEP_REF=main
ARG OFI_NCCL_REPO=https://github.com/aws/aws-ofi-nccl.git
ARG OFI_NCCL_REF=master
ARG OFI_NCCL_PREFIX=/opt/aws-ofi-nccl

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. System deps + EFA installer (libfabric + rdma-core)
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git ca-certificates curl wget gnupg \
      build-essential cmake ninja-build \
      pkg-config autoconf automake libtool \
      libnuma-dev libhwloc-dev \
      python3-dev python3-pip; \
    cd /tmp; \
    curl -fSL "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" -o efa.tgz; \
    tar -xzf efa.tgz; \
    cd aws-efa-installer; \
    ./efa_installer.sh -y --skip-kmod -g --no-verify; \
    cd /; rm -rf /tmp/efa.tgz /tmp/aws-efa-installer; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 1b. GDRCopy userspace (libgdrapi). Kernel module gdrdrv lives on the host.
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
# 2. NCCL (cu13) + NVSHMEM (cu13) Python wheels — DeepEP's setup.py finds
#    these via importlib.metadata and uses their headers/libs at JIT/build time.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --no-deps \
      "nvidia-nccl-cu13>=2.30.4" \
      "nvidia-nvshmem-cu13"

# -----------------------------------------------------------------------------
# 3. Build aws-ofi-nccl from master (GIN-capable). The plugin built here
#    lives at $OFI_NCCL_PREFIX/lib/libnccl-net-ofi.so. Override
#    NCCL_NET_PLUGIN at runtime so NCCL loads THIS plugin, not the EFA
#    installer's older one at /opt/amazon/ofi-nccl/lib/.
# -----------------------------------------------------------------------------
RUN set -eux; \
    cd /tmp; \
    git clone --recursive ${OFI_NCCL_REPO} aws-ofi-nccl; \
    cd aws-ofi-nccl; \
    git checkout ${OFI_NCCL_REF}; \
    NCCL_INC=$(python3 -c "import importlib.metadata as m, os; \
files=m.distribution('nvidia-nccl-cu13').files; \
inc=[str(f.locate()) for f in files if str(f).endswith('include/nccl.h')][0]; \
print(os.path.dirname(inc))"); \
    NCCL_LIB=$(python3 -c "import importlib.metadata as m, os; \
files=m.distribution('nvidia-nccl-cu13').files; \
lib=[str(f.locate()) for f in files if 'libnccl.so' in str(f) and not str(f).endswith('.json')][0]; \
print(os.path.dirname(lib))"); \
    echo "NCCL_INC=${NCCL_INC}  NCCL_LIB=${NCCL_LIB}"; \
    ./autogen.sh; \
    ./configure \
      --prefix=${OFI_NCCL_PREFIX} \
      --with-libfabric=/opt/amazon/efa \
      --with-cuda=/usr/local/cuda \
      --with-nccl=${NCCL_INC}/.. \
      --enable-platform-aws \
      --disable-tests; \
    make -j$(nproc); \
    make install; \
    test -f ${OFI_NCCL_PREFIX}/lib/libnccl-net-ofi.so; \
    cd /; rm -rf /tmp/aws-ofi-nccl

# -----------------------------------------------------------------------------
# 4. Build + install DeepEP V2 (JIT — the heavy nvcc work happens at runtime
#    on first import; build here just registers the wheel and copies headers).
# -----------------------------------------------------------------------------
# NGC pytorch:26.04 ships CCCL under
# /usr/local/cuda/targets/x86_64-linux/include/cccl, not /usr/local/cuda/include.
# DeepEP's setup includes <cuda/...> so put it on CPATH.
ENV CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:${CPATH}
ENV TORCH_CUDA_ARCH_LIST="9.0"

RUN set -eux; \
    cd /opt; \
    git clone ${DEEPEP_REPO} deepep; \
    cd deepep; \
    git checkout ${DEEPEP_REF}; \
    # setup.py refuses to build with a dirty git tree -- record current SHA
    # so DeepEP doesn't try to call `git rev-parse` on a missing repo later.
    git config --global --add safe.directory /opt/deepep; \
    pip install --no-cache-dir --no-deps --no-build-isolation -v .; \
    # Verify import metadata (libcuda.so.1 not present at build time)
    cd /tmp && python3 -c "import importlib.metadata as m; \
print('deep_ep version:', m.version('deep_ep'))"

# Patch deep_ep/utils/envs.py: get_rdma_gbs() upstream uses `ibstat`, which
# doesn't list EFA NICs (EFA isn't IB-class). Without this patch, V2 hits
# ZeroDivisionError in get_theoretical_num_sms() at runtime. The patch
# tries /sys/class/infiniband/<nic>/ports/1/rate first (works on EFA), then
# falls back to ibstat. Also adds EP_RDMA_GBS env override.
COPY patch_envs.py /tmp/patch_envs.py
RUN python3 /tmp/patch_envs.py && rm /tmp/patch_envs.py

# -----------------------------------------------------------------------------
# 5. Default runtime env tuned for AWS EFA + DeepEP V2 GIN
# -----------------------------------------------------------------------------
# IMPORTANT ordering: pip nvidia-nccl-cu13 (>=2.30.4) MUST come before system
# /usr/lib (which has libnccl.so.2.29.7 from NGC's libnccl2 deb). DeepEP V2's
# check_nccl_so() asserts the loaded NCCL matches what DeepEP linked against
# (the pip wheel), so the system one would cause an AssertionError at import.
RUN PIP_NCCL_LIB=$(python3 -c "import importlib.metadata as m, os; \
files=m.distribution('nvidia-nccl-cu13').files; \
lib=[str(f.locate()) for f in files if 'libnccl.so' in str(f)][0]; \
print(os.path.dirname(lib))") && \
    echo "${PIP_NCCL_LIB}" > /etc/ld.so.conf.d/0-pip-nccl.conf && \
    ldconfig
ENV LD_LIBRARY_PATH=${OFI_NCCL_PREFIX}/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}
ENV NCCL_NET_PLUGIN=${OFI_NCCL_PREFIX}/lib/libnccl-net-ofi.so
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
# DeepEP V2 looks at EP_NIC_NAME to identify the RDMA device. Default is
# `mlx5_0` (Mellanox), which doesn't exist on EFA. Setting * tells
# ofi-nccl to use any matching NIC; setting a specific rdmap*s0 picks one.
ENV EP_NIC_NAME=rdmap79s0

WORKDIR /opt/deepep
CMD ["/bin/bash"]

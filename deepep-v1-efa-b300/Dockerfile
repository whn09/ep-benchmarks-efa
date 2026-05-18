# syntax=docker/dockerfile:1.10
# =============================================================================
# DeepEP V1 + amazon-contributing NVSHMEM (libfabric/EFA) + AWS p5.48xlarge
# =============================================================================
#
# Stack:
#   - Base:    nvcr.io/nvidia/pytorch:26.04-py3  (CUDA 13.2.1, torch 2.12, sm90)
#   - NVSHMEM: amazon-contributing/upstream-to-nvshmem @ devel_enriched
#              (libfabric remote transport for EFA)
#   - DeepEP:  rauteric/DeepEP @ remove-fence
#              (upstream-NVSHMEM-compatible internode kernels, fence removed
#               for unordered transports like EFA)
#
# Mapping of DeepEP's third-party/README.md instructions to this image:
#   - "Install NVSHMEM 3.3.9+ binaries"     -> replaced by source build of
#       amazon-contributing devel_enriched (has libfabric EFA transport +
#       multi-NIC round-robin which the upstream binary lacks).
#   - "Configure nvidia regkeys for IBGDA"  -> SKIPPED. EFA does not implement
#       IB verbs, so IBGDA is N/A; we use the libfabric remote transport.
#   - "Install GDRCopy + gdrdrv module"     -> KEPT. On EFA, GDRCopy is used
#       by NVSHMEM to register GPU memory as MRs to libfabric (NOT for
#       CPU-assisted IBGDA). Host needs /dev/gdrdrv loaded; container needs
#       libgdrapi.so present for the build to detect GDRCopy.
#   - Post-install env: NVSHMEM_DIR / PATH / LD_LIBRARY_PATH -> set in ENV.
#
# Runtime expectations on host:
#   - NVIDIA driver 595+ (CUDA 13.2-capable)
#   - /dev/gdrdrv exists (gdrdrv kernel module loaded; see
#     feedback_gdrdrv_after_kernel_upgrade.md)
#   - EFA installer present at /opt/amazon/efa on host (only used at runtime
#     for /dev/infiniband; libfabric headers/libs come from the installer
#     baked in below for build reproducibility)
#
# Run example (on each node):
#   docker run --rm -it --gpus all --privileged --network=host --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     --device=/dev/infiniband --device=/dev/gdrdrv \
#     -v /opt/amazon/efa:/opt/amazon/efa:ro \
#     deepep-v1-efa:dev bash
# =============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

ARG NVSHMEM_REPO=https://github.com/amazon-contributing/upstream-to-nvshmem.git
ARG NVSHMEM_REF=devel_enriched
ARG DEEPEP_REPO=https://github.com/rauteric/DeepEP.git
ARG DEEPEP_REF=remove-fence
ARG EFA_INSTALLER_VERSION=1.48.0
ARG GDRCOPY_VERSION=2.5.2
ARG NVSHMEM_PREFIX=/opt/nvshmem

ENV DEBIAN_FRONTEND=noninteractive
ENV NVSHMEM_DIR=${NVSHMEM_PREFIX}
ENV PATH=${NVSHMEM_PREFIX}/bin:${PATH}
ENV LD_LIBRARY_PATH=${NVSHMEM_PREFIX}/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}

# -----------------------------------------------------------------------------
# 1. System deps + EFA installer (gives libfabric headers + libs for build)
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git ca-certificates curl wget gnupg \
      build-essential cmake ninja-build \
      pkg-config autoconf libtool \
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
#     host -- container only needs the userspace .so for NVSHMEM detection.
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
# 2. Build amazon-contributing NVSHMEM with libfabric (EFA) transport
# -----------------------------------------------------------------------------
# Key flags:
#   NVSHMEM_LIBFABRIC_SUPPORT=1   -> the EFA path
#   NVSHMEM_USE_GDRCOPY=1         -> CPU-assisted GDR (we have /dev/gdrdrv)
#   NVSHMEM_IBGDA_SUPPORT=0       -> EFA does not implement IBGDA verbs
#   NVSHMEM_IBRC_SUPPORT=0        -> not used; saves build time
#   NVSHMEM_BUILD_TESTS=0         -> not needed for DeepEP
#   NVSHMEM_BUILD_EXAMPLES=0
#   NVSHMEM_BUILD_PYTHON_LIB=0    -> DeepEP links the C/CUDA libs only
RUN set -eux; \
    cd /opt; \
    git clone --depth 1 -b ${NVSHMEM_REF} ${NVSHMEM_REPO} nvshmem-src; \
    cd nvshmem-src; \
    NVSHMEM_LIBFABRIC_SUPPORT=1 \
    NVSHMEM_USE_GDRCOPY=1 \
    NVSHMEM_IBGDA_SUPPORT=0 \
    NVSHMEM_IBRC_SUPPORT=0 \
    NVSHMEM_IBDEVX_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_BUILD_TESTS=0 \
    NVSHMEM_BUILD_EXAMPLES=0 \
    NVSHMEM_BUILD_PYTHON_LIB=0 \
    NVSHMEM_BUILD_PYTHON_DEVICE_LIB=0 \
    NVSHMEM_USE_NCCL=0 \
    cmake -S . -B build -G Ninja \
      -DCMAKE_INSTALL_PREFIX=${NVSHMEM_PREFIX} \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="90;100" \
      -DLIBFABRIC_HOME=/opt/amazon/efa \
      -DGDRCOPY_HOME=/usr/local; \
    cmake --build build --parallel; \
    cmake --install build; \
    ls ${NVSHMEM_PREFIX}/lib/libnvshmem_host.so* ${NVSHMEM_PREFIX}/lib/libnvshmem_device.a; \
    # nvshmem-info -a prints transport+version; libfabric line should be present
    LD_LIBRARY_PATH=${NVSHMEM_PREFIX}/lib:${LD_LIBRARY_PATH} \
      ${NVSHMEM_PREFIX}/bin/nvshmem-info -a | tee /tmp/nvshmem-info.log; \
    grep -qi "libfabric" /tmp/nvshmem-info.log; \
    rm -rf /opt/nvshmem-src /tmp/nvshmem-info.log

# -----------------------------------------------------------------------------
# 3. Build rauteric/DeepEP @ remove-fence against the patched NVSHMEM
# -----------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="9.0;10.0"
# NGC pytorch:26.04 ships CCCL headers under /usr/local/cuda/targets/x86_64-linux/include/cccl
# rather than the default /usr/local/cuda/include. NVSHMEM headers transitively include
# cuda/std/tuple, so we must put cccl on the include path for the host C++ compiler.
ENV CCCL_INCLUDE=/usr/local/cuda/targets/x86_64-linux/include/cccl
ENV CPATH=${CCCL_INCLUDE}:${CPATH}
RUN set -eux; \
    test -f ${CCCL_INCLUDE}/cuda/std/tuple; \
    cd /opt; \
    git clone -b ${DEEPEP_REF} ${DEEPEP_REPO} deepep; \
    cd deepep; \
    NVSHMEM_DIR=${NVSHMEM_PREFIX} python setup.py bdist_wheel; \
    pip install --no-deps dist/*.whl; \
    # Skip `import deep_ep` here: it transitively dlopens libcuda.so.1 which is
    # only present at runtime (mounted by the nvidia container runtime).
    # Verify by static metadata + .so existence instead.
    python -c "import importlib.metadata as m; print('deep_ep wheel:', m.version('deep_ep'))"; \
    find /usr/local/lib/python3.12/dist-packages -name "deep_ep_cpp*.so" -ls

# -----------------------------------------------------------------------------
# 4. Default env tuned for AWS EFA / DeepEP
# -----------------------------------------------------------------------------
ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1

WORKDIR /opt/deepep
CMD ["/bin/bash"]

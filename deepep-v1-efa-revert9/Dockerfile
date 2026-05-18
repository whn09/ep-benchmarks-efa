# syntax=docker/dockerfile:1.10
# =============================================================================
# deepep-v1-efa with PR #9 reverted
# =============================================================================
#
# Layered on top of deepep-v1-efa:dev. Re-clones amazon-contributing
# upstream-to-nvshmem, reverts commit 0ec7be33 ("libfabric: Disable
# unsolicited write for EFA provider"), rebuilds NVSHMEM, and reinstalls
# libnvshmem_host.so* into /opt/nvshmem/lib so DeepEP picks it up at
# runtime via LD_LIBRARY_PATH.
#
# Reference: PR https://github.com/amazon-contributing/upstream-to-nvshmem/pull/9
# Commit:    https://github.com/amazon-contributing/upstream-to-nvshmem/commit/0ec7be33
#
# Effect on AWS EFA: enables write-with-immediate without re-posting
# recv buffers per completion, removing ~50 us per dispatch from the
# critical path. Trade-off: theoretical CQ-overflow risk on
# arbitrary communication patterns (DeepEP V1 LL is bounded so the
# pattern doesn't trigger overflow in practice).
# =============================================================================

ARG BASE_IMAGE=deepep-v1-efa:dev
FROM ${BASE_IMAGE}

ARG NVSHMEM_REF=devel_enriched
ARG REVERT_COMMIT=0ec7be33

RUN set -eux; \
    cd /opt; \
    git clone https://github.com/amazon-contributing/upstream-to-nvshmem.git nvshmem-src; \
    cd nvshmem-src; \
    git checkout ${NVSHMEM_REF}; \
    git config user.email "build@local"; \
    git config user.name "build"; \
    # Revert ONLY the FI_RX_CQ_DATA / unsolicited-write commit
    git revert --no-edit ${REVERT_COMMIT}; \
    git log -1 --oneline

RUN set -eux; \
    cd /opt/nvshmem-src; \
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
      -DCMAKE_INSTALL_PREFIX=/opt/nvshmem \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DLIBFABRIC_HOME=/opt/amazon/efa \
      -DGDRCOPY_HOME=/usr/local; \
    cmake --build build --parallel; \
    cmake --install build; \
    rm -rf /opt/nvshmem-src

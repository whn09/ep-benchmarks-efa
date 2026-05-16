# syntax=docker/dockerfile:1.10
# =============================================================================
# pplx-garden-efa  --  Perplexity AI's pplx-garden P2P all-to-all on AWS EFA
# =============================================================================
#
# Builds on top of upstream `docker/dev.Dockerfile` (CUDA 12.9 +
# torch 2.9.0+cu129 + EFA + GDRCopy + Rust). After the dev image is in
# place, the wheel is built and installed inside the same image so the
# benchmark `python3 -m benchmarks.bench_all_to_all` runs out of the box.
#
# Stack:
#   - Base CUDA: nvidia/cuda:12.9.1-devel-ubuntu24.04
#   - PyTorch: 2.9.0+cu129
#   - EFA: aws-efa-installer 1.44.0
#   - GDRCopy: 2.5.1 (libgdrapi, kernel module from host)
#   - Rust: 1.91.0
#   - pplx-garden: built from the source tree at /app
#
# Run example (per node):
#   docker run --rm -it \
#     --shm-size=32g --ulimit=memlock=-1 --ulimit=stack=67108864 \
#     --gpus=all --network=host \
#     $(for u in /dev/infiniband/uverbs*; do echo --device=$u; done) \
#     --device=/dev/gdrdrv \
#     --cap-add=IPC_LOCK --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
#     --security-opt=seccomp=unconfined \
#     pplx-garden-efa:dev bash
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: GDRCopy deb builder (matches upstream dev.Dockerfile)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS gdrcopy-builder
RUN apt-get update && apt-get install -y \
    build-essential devscripts debhelper fakeroot pkg-config wget
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.5.1.tar.gz && \
    tar -xf v2.5.1.tar.gz && \
    cd gdrcopy-2.5.1/packages/ && \
    CUDA=/usr/local/cuda ./build-deb-packages.sh -t -k

# -----------------------------------------------------------------------------
# Stage 2: dev base (mirrors upstream dev.Dockerfile final stage)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      patchelf libclang-dev clang-18 clang-format-18 \
      git build-essential cmake libssl-dev wget curl ninja-build \
      pkg-config python3-dev python3-setuptools python3-pip \
      python3-build python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV TORCH_CUDA_ARCH_LIST="9.0a;10.0a+PTX"
RUN python3 -m pip install torch==2.9.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# EFA installer: provides libfabric + libibverbs + plugin
ARG EFA_INSTALLER_VERSION=1.44.0
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    apt-get update && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    rm -rf /var/lib/apt/lists/* /tmp/aws-efa-installer* && \
    ldconfig
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent

COPY --from=gdrcopy-builder \
    /tmp/gdrcopy-2.5.1/packages/libgdrapi_2.5.1-1_amd64.Ubuntu24_04.deb \
    /tmp/libgdrapi.deb
RUN dpkg -i /tmp/libgdrapi.deb && rm /tmp/libgdrapi.deb

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.91.0 --component llvm-tools-preview
ENV PATH="/root/.cargo/bin:$PATH" \
    CARGO_HOME="/root/.cargo" \
    RUSTUP_HOME="/root/.rustup"

RUN python3 -m pip install numpy ninja maturin pytest coverage mypy pylint ruff

# -----------------------------------------------------------------------------
# Stage 3: build + install pplx-garden wheel
# -----------------------------------------------------------------------------
FROM dev AS final

# Clone the source straight from upstream (could also COPY from the host
# build context, but cloning makes the image self-contained).
ARG PPLX_REPO=https://github.com/perplexityai/pplx-garden.git
ARG PPLX_REF=main
RUN git clone ${PPLX_REPO} /app && cd /app && git checkout ${PPLX_REF}

WORKDIR /app
ENV TORCH_CMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch
RUN export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    python3 -m build --wheel && \
    python3 -m pip install /app/dist/*.whl

# `import pplx_garden` transitively dlopens libcuda.so.1, which is only
# mounted by the nvidia container runtime at `docker run`, not at
# `docker build`. Verify by metadata + module path instead.
RUN python3 -c "import importlib.metadata as m; \
print('pplx_garden version:', m.version('pplx-garden'))" && \
    test -f /app/benchmarks/bench_all_to_all.py

CMD ["/bin/bash"]

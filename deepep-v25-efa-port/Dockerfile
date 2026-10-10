# DeepEP V2.5 (deepseek-ai/DeepEP main) + the amazon-contributing EFA port, on AWS EFA
#
# Layered on the EFA 1.50.0 / CUDA 13.3.1 / torch 2.13 image from ./Dockerfile (built by
# build_image.sh). Three things V2.5 needs that that image does not have:
#   1) NCCL >= 2.32.3 (V2.5 uses 2.32-only API and checks the runtime version)
#   2) an aws-ofi-nccl with EFA-GDA backendVersion 2. NCCL 2.32 asks for v2; the 1.21.1
#      in installer 1.50.0 supports 0..1 and GIN type 5 fails with
#      "unsupported backendVersion 2 (plugin supports 0..1)". Support landed on master
#      (452f302, 2026-09-11) and is in no release yet, so it is built from a pinned sha
#      into /opt/aon and selected by NCCL_NET_PLUGIN.
#   3) libdw-dev (elfutils) for the DeepJIT submodule, whose URL is ssh (rewritten to https)
ARG BASE=deepep-v2-efa-official:sm100
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libdw-dev autoconf automake libtool libhwloc-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages --no-cache-dir --no-deps "nvidia-nccl-cu13==2.32.3"

ARG AON_REF=cb0c85e87376113dedd17a8f212878ed12dbf51a
RUN git clone -q https://github.com/aws/aws-ofi-nccl.git /tmp/aon && cd /tmp/aon \
    && git checkout -q "$AON_REF" && git submodule update -q --init --recursive \
    && ./autogen.sh >/dev/null \
    && ./configure -q --prefix=/opt/aon --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda \
         --enable-platform-aws --disable-tests \
    && make -j"$(nproc)" >/dev/null && make install >/dev/null \
    && nm -D /opt/aon/lib/libnccl-net-ofi.so | grep -qw ncclGinPlugin_v14 \
    && echo "$AON_REF" > /opt/aon/BUILD_REF && cd / && rm -rf /tmp/aon
ENV NCCL_NET_PLUGIN=/opt/aon/lib/libnccl-net-ofi.so
ENV LD_LIBRARY_PATH=/opt/aon/lib:${LD_LIBRARY_PATH}

# DEEPEP_REF must be a full 40-char sha (a short sha cannot be fetched)
ARG DEEPEP_REPO=https://github.com/whn09/DeepEP.git
ARG DEEPEP_REF
RUN test "${#DEEPEP_REF}" = 40 \
    && SP=$(python3 -c 'import site;print(site.getsitepackages()[0])') \
    && rm -rf /opt/DeepEP $SP/deep_ep* && (sed -i '/deep_ep/d' $SP/easy-install.pth 2>/dev/null || true) \
    && mkdir -p /opt/DeepEP && cd /opt/DeepEP && git init -q \
    && git remote add origin "$DEEPEP_REPO" \
    && git fetch -q --depth 1 origin "$DEEPEP_REF" && git checkout -q FETCH_HEAD \
    && git rev-parse HEAD > /opt/DeepEP/BUILD_REF \
    && git config --global url."https://github.com/".insteadOf "git@github.com:" \
    && git submodule update -q --init --recursive --depth 1 \
    && NCCL_LIB="$(python3 -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')" \
    && LIBRARY_PATH="$NCCL_LIB:${LIBRARY_PATH:-}" python3 setup.py install \
    && rm -rf build \
    && cd / && python3 -c "import importlib.util as u; assert u.find_spec('deep_ep')"

# GIN type 5 (EFA GDA) also selects the EFA-GDA device backend in DeepEP's JIT.
# NCCL_CUMEM_HOST_ENABLE=0: with 2.32.3 the torch NCCL all_to_all in test_ep's reference
# path fails `ncclCuMemHostAlloc ... out of memory` on GPUs 4-7 otherwise
ENV NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
WORKDIR /workspace

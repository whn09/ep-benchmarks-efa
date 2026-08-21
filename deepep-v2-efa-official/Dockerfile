# DeepEP v2 on AWS EFA
#
# 容器只装用户态（libfabric / rdma-core / libnccl-ofi / gdrcopy lib）。
# 内核态（efa.ko 3.3.0、efa_nv_peermem、gdrdrv）必须由 host 提供，容器装不了。
#
# 依赖链（每一层都必须够新，缺一层 GIN 就退不出来）：
#   DeepEP v2 unordered kernels (counting signals)
#     -> NCCL GIN            : nvidia-nccl-cu13 >= 2.30.4（建议 2.31+）
#     -> ncclGinPlugin_v14   : libnccl-ofi 1.21.1      [EFA installer 1.50.0]
#     -> FI_EFA_GDA_OPS      : libfabric 2.6.0 efa-direct
#     -> ibv_*_comp_cntr     : rdma-core 64.0
#     -> COMP_CNTR cap       : efa.ko 3.3.0            [host]

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git pkg-config \
        python3 python3-dev python3-pip python3-venv \
        libhwloc-dev pciutils environment-modules kmod check libsubunit-dev

# ---------------------------------------------------------------------------
# 1) EFA 用户态 1.50.0
#    --skip-kmod        : 容器里不能装内核模块
#    --skip-limit-conf  : memlock 由 docker --ulimit 给
#    --no-verify        : 跳过 GPG（dev 包没签名）
#    这里刻意不加 --skip-rdma-core：容器需要 rdma-core 64.0 的 libibverbs
# ---------------------------------------------------------------------------
COPY aws-efa-installer-latest.tar.gz /tmp/
RUN apt-get update && cd /tmp && tar xzf aws-efa-installer-latest.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && cd / && rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer-latest.tar.gz \
    && rm -rf /var/lib/apt/lists/*

# installer 的 NGC 分支会顺带装 apt 的 libnccl2/libnccl-dev 2.28.3（并置成 hold）。
# 2.28.3 < 2.30.4 => 没有 GIN。DeepEP 走 pip 那份 2.31.2 的 rpath 所以不受影响，
# 但 `ldconfig` 把 libnccl.so.2 解析到 2.28.3、/usr/include/nccl.h 也是它，
# 任何人动一下 LD_LIBRARY_PATH/LD_PRELOAD 就会静默掉到没有 GIN 的那份上。
# 删掉是安全的：libnccl-ofi-ngc-v3 这个 deb 没有任何 Depends 字段，
# libnccl-net-ofi.so 只链 libfabric.so.1、根本不链 libnccl（它是被 NCCL 加载的插件）。
# 但删掉之后必须给下面第 4 步补一个 -L（见那里的 LIBRARY_PATH）：setup.py 只发
# `-l:libnccl.so.2` + rpath 而不发 -L，所以链接期本来是靠 apt 这份 .so 解析的。
RUN apt-mark unhold libnccl2 libnccl-dev \
    && apt-get remove -y libnccl2 libnccl-dev \
    && ldconfig

ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# 2) GDRCopy 用户态（GIN 的 gdrcopy worker 需要 2.5+；gdrdrv 内核模块在 host）
# ---------------------------------------------------------------------------
RUN git clone -b v2.5 --depth 1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/usr/local lib_install \
    && ldconfig && cd / && rm -rf /tmp/gdrcopy

# ---------------------------------------------------------------------------
# 3) PyTorch + NCCL + NVSHMEM
#    NCCL 用 2.31+：低于 2.31 时 DeepEP 要求编译期与运行期 NCCL 版本严格一致。
#    NVSHMEM 目前是 setup.py 的硬依赖（legacy 后端），即便只跑 v2 也要装。
# ---------------------------------------------------------------------------
RUN pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130 \
 && pip3 install --break-system-packages --no-cache-dir \
        "nvidia-nccl-cu13==2.31.2" "nvidia-nvshmem-cu13==3.7.2" --no-deps \
 && pip3 install --break-system-packages --no-cache-dir \
        ninja numpy nvidia-ml-py

# ---------------------------------------------------------------------------
# 4) DeepEP（amazon-contributing fork，含 EFA 所需的 unordered kernels）
# ---------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV EP_JIT_CACHE_DIR=/root/.deep_ep
# 钉死 commit：runbook 里的性能数字是在这个 tree 上测的，跟着 main 漂会静默失配。
# 想跟最新就改成 main（注意此时数字不再可比）。
# `git clone --depth 1 --branch <sha>` 不接受裸 sha，只能 init + fetch。
# third-party/fmt 是 submodule，不拉的话 setup.py 和 JIT 都缺 fmt 头文件。
ARG DEEPEP_COMMIT=ec623f31b605b27d67c9b224d69378137f77bbe3
RUN mkdir -p /opt/DeepEP && cd /opt/DeepEP && git init -q \
    && git remote add origin https://github.com/amazon-contributing/DeepEP.git \
    && git fetch -q --depth 1 origin "$DEEPEP_COMMIT" \
    && git checkout -q FETCH_HEAD \
    && git submodule update -q --init --recursive --depth 1 \
    && NCCL_LIB="$(python3 -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')" \
    && test -e "$NCCL_LIB/libnccl.so.2" \
    && LIBRARY_PATH="$NCCL_LIB:${LIBRARY_PATH:-}" python3 setup.py install \
    && rm -rf build

# ---------------------------------------------------------------------------
# 5) 运行时默认值
# ---------------------------------------------------------------------------
# EFA 上网卡名不是 mlx5_0，DeepEP 用它查网卡属性。
# 实际名字用 `ibv_devinfo -l` 查，启动容器时按需覆盖。
# 注意：它只喂给 get_rdma_gbs()，而那条路在 EFA 上必失败（走 ibstat，
# ibstat 看不到 EFA 设备）。所以跑多机时 test_ep.py 必须显式给 --num-sms，
# 否则 get_theoretical_num_sms() 会 ZeroDivisionError。
ENV EP_NIC_NAME=rdmap85s0
# installer 在容器里会走 NGC 分支，插件落成 libnccl-net-ofi.so，
# 所以这里用 NCCL 的名字解析（`ofi` -> libnccl-net-ofi.so）而不是写死全路径。
ENV NCCL_NET_PLUGIN=ofi
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1

# 不要把 WORKDIR 设成 /opt/DeepEP：那样 python 会优先 import 源码目录里的
# deep_ep/ 包（不含编译好的 _C.so），报 ModuleNotFoundError: deep_ep._C
WORKDIR /workspace

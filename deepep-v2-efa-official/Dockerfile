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

# CUDA 版本只决定 **ptxas / nvcc**（torch 那份 wheel 是 cu130，CUDA 次版本兼容，
# 换 base 不用换 wheel）。JIT 用的是 base 里的 nvcc，不是 wheel 里的。
#
# H200 / sm_90 用 13.0.2 就够。**B300 / sm_103 必须 >= 13.3.x**：
# `deep_ep/include/deep_ep/common/ptx.cuh` 有多处 `#if __CUDA_ARCH__ >= 1000`
# （`:106` 的 `st.bulk`、`:229` 起的 `.v4.s64` LD/ST，形如
# `ld.L1::no_allocate...global.nc.v4.s64 {4 个 64 位寄存器}`）；sm_103 = 1030 会进这些
# 分支，而 13.0.2 的 ptxas 在汇编生成出来的 PTX 时报
#   `line NNN; error : Arguments mismatch for instruction 'mov'` -> `ptxas fatal`
# -> `Assertion (csrc/jit/compiler.hpp:239): "NVCC compilation failed"`，
# 在第一次 dispatch（JIT 触发点）时才炸，build 阶段看不出来。
# 13.3.1 的 ptxas 接受这些形式。实测：b300 上换 base 后 dispatch/combine 正常出数。
#
# 这个坑**没有宏可以绕**：`DISABLE_AGGRESSIVE_PTX_INSTRS` 在 V2 的 impls/*.cuh 里
# 零处引用（只有 `csrc/kernels/legacy/utils.cuh` 读它，那是 V1 的路径），而且
# `compiler.hpp` 构造 flags 时只处理 EP_JIT_CPP_STANDARD / EP_NUM_TOPK_IDX_BITS 那几个，
# `EP_JIT_EXTRA_FLAGS` 在源码里还是 TODO —— 没有给 JIT 传 flag 的入口。
# setup.py 对 arch != 9.0 加的那个 -D 只作用于 AOT 的 _C.so，JIT 拿不到。
ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

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
#    The tarball is named by VERSION, and `build_image.sh` fetches it from the public
#    bucket's versioned URL (https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz)
#    -- nothing to rename. The name is versioned so that the image is self-describing:
#    a `-latest` filename here would build a different stack the day that object
#    becomes 1.51.0, with nothing in the image saying so. The grep below is the real
#    guard; the filename alone would only move the mistake.
ARG EFA_INSTALLER_VERSION=1.50.0
ENV EP_EFA_INSTALLER=${EFA_INSTALLER_VERSION}
COPY aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz /tmp/aws-efa-installer.tar.gz
RUN apt-get update && cd /tmp && tar xzf aws-efa-installer.tar.gz \
    && { grep -q "^## \[${EFA_INSTALLER_VERSION}\]" aws-efa-installer/ChangeLog.md \
         || { echo "tarball is not EFA installer ${EFA_INSTALLER_VERSION}:" >&2; \
              head -8 aws-efa-installer/ChangeLog.md >&2; exit 1; }; } \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && cd / && rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer.tar.gz \
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
# 9.0 = Hopper（p5en / H200，本目录所有数字都是这一档）；
# 10.0 = B200 sm_100；10.3 = B300 sm_103（p6-b300）。
# 必须是 ARG 而不是写死的 ENV：sm_103 不是 sm_100 的向下兼容目标，Hopper 的
# cubin 和 PTX 在 Blackwell 上都跑不了，所以换硬件必须重编。b300 上用
# `--build-arg TORCH_CUDA_ARCH_LIST=10.3`。
# 建议一个镜像一档，别写 `9.0;10.3`：sm_103 要求 CUDA_VERSION >= 13.3.x（见文件头），
# 所以"一个镜像通吃两代"本来就得连 base 一起换。另外 `setup.py:148` 在这个字符串不
# **恰好等于** `9.0` 时会强制 `DISABLE_AGGRESSIVE_PTX_INSTRS=1` —— 那个宏只有
# `csrc/kernels/legacy/utils.cuh`（V1 路径）读，V2 的 impls 一处都不读，所以对 v2
# 的性能无影响，只影响同一个 .so 里的 legacy kernel。
ARG TORCH_CUDA_ARCH_LIST=9.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# Stamp both build args into the image. A Hopper cubin does not run on Blackwell,
# and the failure surfaces far from its cause, so run_test_ep.sh compares
# EP_BUILD_ARCH against the host's compute_cap before it starts anything.
# CUDA_VERSION is declared before FROM, so it has to be re-declared to be read here.
ARG CUDA_VERSION
ENV EP_BUILD_ARCH=${TORCH_CUDA_ARCH_LIST}
ENV EP_BUILD_CUDA=${CUDA_VERSION}
ENV EP_JIT_CACHE_DIR=/root/.deep_ep
# DEEPEP_REF 接受 **branch / tag / 裸 sha** 三种写法。
#
# 默认钉在一个 sha 上，而不是跟 main 漂 —— 理由不是"怕新代码"，是三条：
#   1) runbook §9 的每个数字都对应一个确定的 tree。跟着 main 漂，重建出来的数字
#      和文档不符时，你分不清是自己环境搞错了还是代码变了。
#   2) upstream 会**改写历史**：我们钉过的一个 sha 现在已经不在 main 上了（被重写成
#      另一个 sha，内容也变了 4 个文件 +41/−19）。钉着的时候这件事是可见的，
#      漂着的时候完全隐形。
#   3) `ordered` kernel 在 EFA GDAKI 上是**不正确**的（见 runbook §9.5）。默认 kernel
#      选择将来一旦变化，漂的镜像会静默跑到错的 kernel 上。
# 想跟最新就 `--build-arg DEEPEP_REF=main`，下面的 ADD 会自动让 layer cache 失效。
# 没有这个 ADD 的话，`RUN git fetch origin main` 会命中旧 layer —— 你以为拉到了
# 最新，实际拿的是上次的代码。这比明着钉死更危险。
#   注意：ADD 走 GitHub API，未认证限速 60 次/小时；只是用来算 cache key。
#
# `git clone --depth 1 --branch <sha>` 不接受裸 sha，只能 init + fetch。
# third-party/fmt 是 submodule，不拉的话 setup.py 和 JIT 都缺 fmt 头文件。
# 当前值 = 写这行时的 `main`（比上一版 pin 8e7b42e 快进 1 个 commit）。那个 commit
# 只把**显式**越界的 `num_allocated_qps` 从 `EP_HOST_ASSERT` 改成 clamp 进 [2,17] + 打
# 一行 `[WARN] DeepEP clamped num_allocated_qps from A to B`。本目录所有臂用的是 auto(0)
# 或显式 5，两者都在范围内、逐行走同一条路，所以 §9 的数字不受影响（没有重测；
# `verify_run.sh` 会 grep 那行 WARN，免得将来有人 sweep 到范围外而不自知）。
ARG DEEPEP_REF=9c1f251118873e5dee9652cdfaf9848753cda7f1
ADD https://api.github.com/repos/amazon-contributing/DeepEP/commits/${DEEPEP_REF} /tmp/deepep-ref.json
RUN mkdir -p /opt/DeepEP && cd /opt/DeepEP && git init -q \
    && git remote add origin https://github.com/amazon-contributing/DeepEP.git \
    && git fetch -q --depth 1 origin "$DEEPEP_REF" \
    && git checkout -q FETCH_HEAD \
    && git rev-parse HEAD > /opt/DeepEP/BUILD_REF \
    && echo "DeepEP build ref '$DEEPEP_REF' resolved to $(cat /opt/DeepEP/BUILD_REF)" \
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
# 注意：它只喂给 get_rdma_gbs()。**这条路在本 Dockerfile 默认钉的 9c1f2511 上是通的**
# ——`envs.py:306` 先读 `/sys/class/infiniband/<nic>/ports/*/rate`，`ibstat`（EFA 上必
# 失败，它走 libibumad 而 EFA 没有 ib_umad）只是 fallback，全失败时抛的是带修复建议
# 的 RuntimeError 而不是 ZeroDivisionError。
# 但它返回的是**一块**网卡的速率：p5en 每 GPU 1 张 EFA（=50 GB/s，正好），
# p6-b300 每 GPU 2 张（真值 100 GB/s，自动探测只看到一半）。所以 `--num-sms` 仍然
# 建议显式给 —— 理由是可复现和 b300 上探测值偏低，不是会崩。
# 前提：容器要能看到 sysfs（run_test_ep.sh 挂了 /sys/class/infiniband:ro）。
ENV EP_NIC_NAME=rdmap85s0
# installer 在容器里会走 NGC 分支，插件落成 libnccl-net-ofi.so，
# 所以这里用 NCCL 的名字解析（`ofi` -> libnccl-net-ofi.so）而不是写死全路径。
ENV NCCL_NET_PLUGIN=ofi
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1

# 不要把 WORKDIR 设成 /opt/DeepEP：那样 python 会优先 import 源码目录里的
# deep_ep/ 包（不含编译好的 _C.so），报 ModuleNotFoundError: deep_ep._C
WORKDIR /workspace

# DeepEP v2 on AWS EFA — 安装、构建、跑分

**两种机型都支持**，同一份 Dockerfile、同一套脚本：

| 机型 | GPU | arch | 每 GPU EFA | 每 GPU 线速 | 本文数字 |
|---|---|---|---|---|---|
| `p5en.48xlarge` | 8×H200 | `sm_90` | 1 张（共 16） | 50 GB/s | §5 全部（×2 和 ×4 节点） |
| `p6-b300.48xlarge` | 8×B300 | `sm_103` | 2 张（共 16） | 100 GB/s | 只有 §3.4 的抽查，campaign 待跑（§8 待办 8） |

差别集中在**两个 build 参数 + 一个运行时变量**，`build_image.sh` / `run_test_ep.sh` 会自己
处理（机制和现象见 §3.4）；host 那一侧两种机型完全一样。B200 / `sm_100` 参数已就位但没跑过。

全部用已发布的包，不用编 NCCL、不用编 aws-ofi-nccl、不用换内核模块。

四步：**host 装依赖 → build 镜像 → 跑 prefill 看带宽 → 跑 decode 看延迟**；成体系地扫一遍
是第五步，`run_campaign.sh` 一条命令（§4.5，两种机型共用）。

所有版本判据和数字都是在真机上跑出来的（2026-08-25，4 台 p5en.48xlarge，installer 1.50.0 /
`deep_ep 2.1.0+ec623f3`；每个数字都是**全 rank 均值** —— 2 节点 16 rank、4 节点 32 rank，
原始日志和聚合脚本在 [`results/p5en_2n4n_20260825/`](../results/p5en_2n4n_20260825/)）。
**§5 那些数字是 p5en 的，没有一行可以拿去当 b300 的读数** —— 反过来也一样。
背景与"为什么"放在最后的附录，正文只留操作。

**开跑之前先记住一件事：容器里必须多传两个环境变量**
`NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0`，否则 NCCL 走的是代理式后端，
prefill 慢 ~9%、decode 慢 2.2–5.4×。理由和数据见 §6。

---

## 0. 依赖链与版本门槛

**内核态装 host，用户态装容器，两边都要。**

| 层 | 需要的版本 | 判据 | 装在哪 |
|---|---|---|---|
| DeepEP v2 unordered kernels | `amazon-contributing/DeepEP` | — | 容器 |
| NCCL GIN | `nvidia-nccl-cu13` ≥ 2.30.4（用 2.31.2） | — | 容器 |
| aws-ofi-nccl | **1.21.1** | **`ncclGinPlugin_v14`** 符号 | 容器 |
| libfabric | **2.6.0amzn1.0** | 16 个 `fabric: efa-direct` | 容器 |
| rdma-core | **64.0amzn0** | libibverbs 里 20 个 `comp_cntr` 符号 | 容器 |
| **efa 内核驱动** | **3.3.0** | `EFA_QUERY_DEVICE_CAPS_COMP_CNTR` (1<<8) | **host** |
| gdrcopy | ≥ 2.5 | `/dev/gdrdrv` | kmod host / lib 容器 |

加粗的四个**全部由 EFA installer 1.50.0 提供** —— 这就是必须用 1.50.0 的原因（1.49.0 为什么不够见附录 A）。

**两种机型的门槛完全一样，但 b300 出厂更远。** p6-b300 的出厂/DLAMI 镜像带的是 EFA
1.47.0（efa.ko 3.0.0，`GinPlugin` 符号一个都没有），在 host 升到 1.50.0 之前 GDAKI 根本
起不来 —— 而且 §3.4 那两个坑要等 GDAKI 起来之后才现形，所以顺序不能颠倒：**先升 host，
再谈 arch**。

GIN = GPU-Initiated Networking（NCCL Device API）；EFA 上的实现叫 `Libfabric_GDAKI`，走 `efa-direct` fabric 的 GDA ops，用硬件 completion counter（CE）承载 counting signal。

---

## 1. 实例前置条件（四条，缺一条都白干）

1. **ENI 必须在创建实例时就 `InterfaceType=efa`** —— 事后打不开，只能重建。用 `NetworkCardIndex=0..15` 逐个指定；两种机型正常状态都是 **16 个 EFA 设备**，区别在每 GPU 摊到几张：p5en 1 张、b300 2 张。
   若 `lsmod | grep efa` 有模块但 `ibv_devinfo` 报 `No IB devices found`、`/dev/infiniband` 不存在 —— 就是这一条没做到。
2. **安全组自引用放通全部流量**（入站 + 出站，Source/Destination = 该安全组自身）。EFA 不走 TCP 端口。
3. **两台同 AZ + 同一个 cluster placement group。** 跨 AZ 会报 `ibv_create_ah failed with EINVAL ... Remote GID is in a different availability zone`。
4. **设备名不是 `mlx5_0`、也不连号**，而且两种机型 `ibv_devinfo -l` 的**总数不一样**：
   - p5en：**16** 个，全是 EFA —— `rdmap85s0 86s0 87s0 88s0 / 110s0 111s0 112s0 113s0 / 135s0 136s0 137s0 138s0 / 160s0 161s0 162s0 163s0`。
   - p6-b300：**18** 个 —— 16 个 `rdmap*`（EFA）**加** `ibp198s0f0` / `ibp199s0f0`，后两个**不是 EFA**（`ce_probe` 对它们是 CE FAIL / errno 95）。多出来这两个会让 NCCL 少建 GDAKI NIC，必须 `NCCL_IB_HCA=rdmap` 排掉（§3.4 坑 1）。所以"`ibv_devinfo -l` 数出 16"这条自检在 b300 上要读成"16 个 `rdmap*`"，不是"一共 16 个"。

最省事的做法是 capacity block + 官方 DLAMI + cluster placement group，让模板把 16 张 EFA ENI 配好。

---

## 2. Host 侧安装（每台机器都做）

### 2.1 下载 installer

1.50.0 已经 GA，公共桶就有带版本号的 URL，**不需要改名**：

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
tar xzf aws-efa-installer-1.50.0.tar.gz          # 约 650 MB（含所有发行版的 RPM/DEB）
head -6 aws-efa-installer/ChangeLog.md           # ## [1.50.0] - Aug 2026
#   - Upgrade to rdma-core 64.0amzn0 / efa driver 3.3.0
#   - Upgrade to Libfabric 2.6.0amzn1.0 / OFI NCCL Plugin 1.21.1
```

build context 那一份不用你操心：`build_image.sh` 发现本地没有
`aws-efa-installer-1.50.0.tar.gz` 就自己去同一个 URL 下（所以每个节点各自从 S3 拉，比从
你笔记本 scp 过去快得多，`rsync` 记得 `--exclude '*.tar.gz'`）。

**文件名带版本号不是洁癖，是为了让镜像自己说得清用的是哪套栈。** 如果 Dockerfile 的
`COPY` 写的是 `-latest`，那个对象哪天变成 1.51.0，同一份 Dockerfile 就会构出另一套栈，
而镜像里没有任何东西记下这件事。版本会被核两遍（build 前在 `build_image.sh` 里、build 中
在镜像里 `grep` tarball 自己的 `ChangeLog.md`），通过的版本记成 `EP_EFA_INSTALLER`。

> 只有**还没 GA** 的版本才需要走 dev 桶那条老路 —— 那里只有浮动名字
> `aws-efa-installer-latest.tar.gz`（这个 key 不是固定版本，今天是 1.50.0 下周可能是
> 1.51.0），先看 `ChangeLog.md` 再改成你核过的版本号。dev 桶的包没 GPG 签名，所以
> `efa_installer.sh` 要带 `--no-verify`。

### 2.2 安装（host 上**不要** `--skip-kmod`，我们要的就是 efa.ko 3.3.0）

```bash
cd aws-efa-installer
sudo ./efa_installer.sh -y --no-verify        # 跳过 GPG 校验（dev 桶的包没签名；GA 的加着也无害）
sudo reboot                                   # 换内核模块必须重启
```

DLAMI 通常已自带 gdrcopy 2.5.x 和 `efa_nv_peermem`，一般只需要升 installer 本身。

### 2.3 重启后验证 host

```bash
export PATH=/opt/amazon/efa/bin:$PATH          # fi_info 默认不在 PATH 上

cat /sys/module/efa/version                    # 3.3.0g
lsmod | grep efa_nv_peermem                    # GDAKI 必需
modinfo gdrdrv | grep ^version ; ls -l /dev/gdrdrv
ibv_devinfo -l | grep -c rdmap                 # 16（b300 总行数是 18，多两个非 EFA 的 ibp*）
fi_info | grep -c "fabric: efa-direct"         # 16
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1   # 9.0 / 10.3，决定 build 参数

# COMP_CNTR capability 在 ABI 头里 —— 这才是 1.50.0 的真判据。
# 头文件在 DKMS 源码树的 src/ 下面（不是 /usr/src/efa-*/ 的第一层）：
grep COMP_CNTR /usr/src/efa-*/src/efa-abi.h        # 期望 3 行，含 = 1 << 8

# 交叉验证：直接看装上去的模块。Ubuntu 的 DKMS 模块在 updates/dkms/ 下且是
# zstd 压缩的，所以路径要用 modinfo -n 问出来，不能写死：
zstd -dc "$(modinfo -n efa)" | strings | grep -c comp_cntr   # 期望 12，不是 0
```

> ⚠️ **两个会把健康节点误判成坏节点的坑。**
> 1. **别 grep `/usr/include/rdma/efa-abi.h`** —— 那是 host 上 distro rdma-core 带的
>    用户态头，在装了 1.50.0 的机器上它照样 **0 个 COMP_CNTR**，因为 DKMS 只换内核模块
>    不换 host 的用户态包。容器里自带 rdma-core 64.0，所以这不影响跑，但拿它当判据必错。
> 2. **`strings` 的路径写死会静默返回 0。** `strings: No such file` 走 stderr，
>    `grep -c` 拿到空输入就打印 `0`，看起来像"驱动不支持"，实际是文件没找到。
>    上面两条命令的正确输出是 **3 行** 和 **12**；任何一条打出 `0` 先确认路径存在。
>
> 最终判据仍然是**在一次性容器里**跑附录 A 的 `ibv_create_comp_cntr` 探针（`CE OK` × 16）——
> 它同时覆盖 host 内核模块和容器里那份 rdma-core，正好是依赖链最底下两层。

**`fi_info -p efa-direct` 永远失败，别拿它当判据。** 它固定返回 `-61 (No data available)`，哪怕节点完全健康 —— `-p` 过滤的是 **provider** 名（就是 `efa`），只有 **fabric** 名叫 `efa-direct`。用 `fi_info | grep fabric`。
同样，"有 efa-direct" 也**不**代表版本够（1.49.0 就有 16 个），见附录 A。

---

## 3. 构建容器镜像

Dockerfile 见同目录 `Dockerfile`（已逐行核对过，见 §3.2）。

```bash
rsync -avz --exclude '*.tar.gz' deepep-v2-efa-official/ <node>:~/work/deepep-v2-efa-official/
ssh <node> "cd ~/work/deepep-v2-efa-official && ./build_image.sh"
```

`build_image.sh` 会 probe GPU 的 `compute_cap` 并据此推出 build 参数，tag 里带 arch
（`deepep-v2-efa-official:sm90` / `:sm103`）。**别直接 `docker build`** —— 下表那两个参数
不是可选的，而且其中一个是**晚炸**的。显式写法：`./build_image.sh sm103 [DEEPEP_REF] [TAG]`。
installer tarball 不在 build context 里的话它会自己下（§2.1）。

约 **21.4 GB**（压缩后 7.7 GB），首次十几分钟。

| 目标 | build 参数（`build_image.sh` 自动设） |
|---|---|
| p5en / H200，`sm_90` | `TORCH_CUDA_ARCH_LIST=9.0  CUDA_VERSION=13.0.2` —— §5 所有数字 |
| p6-b300 / B300，`sm_103` | `TORCH_CUDA_ARCH_LIST=10.3  CUDA_VERSION=13.3.1` —— 运行时还要 `NCCL_IB_HCA=rdmap`（自动注入），§3.4 |
| B200，`sm_100` | `TORCH_CUDA_ARCH_LIST=10.0  CUDA_VERSION=13.3.1` —— 没跑过 |

**一个镜像只编一档。** Hopper 的 cubin 在 Blackwell 上跑不了，`sm_103` 也不是 `sm_100`
的向下兼容目标，而且 `9.0;10.3` 本来就要连 CUDA base 一起换（§3.4）。两个参数都会写进
镜像的 `EP_BUILD_ARCH` / `EP_BUILD_CUDA`，`run_test_ep.sh` 启动前会和 host 对一遍，不
一致直接拒绝跑 —— 否则这两种错都在离现场很远的地方才炸。

第二条臂（`amazon-contributing/DeepEP` #1 + #2，#1 是 #2 head 的祖先，一个 ref 就是叠加）：

```bash
./build_image.sh sm103 5a594a5db2d1b7c45c60c82b0cf026e9440886a4
# -> deepep-v2-efa-official:sm103-5a594a5   （run_campaign.sh 默认找的就是这个 tag）
```

### 3.1 启动容器

```bash
docker run --rm -it \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1 \
  --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  deepep-v2-efa-official:sm90 bash            # b300 上是 :sm103
```

- `--ulimit memlock=-1` 必须有（容器里跳过了 `limits.conf`），否则 RDMA 注册内存失败。
- `--device /dev/infiniband` 透 EFA 设备，`--device /dev/gdrdrv` 给 gdrcopy。
- `--network host` 让多机直接互通；`--privileged` 跑通后可收紧成 `--cap-add IPC_LOCK`。

### 3.2 Dockerfile 里已经处理掉的坑（按构建顺序）

1. **不能在装 EFA 之前清 apt 缓存。** installer 自己会 `apt-get install tcl libnl-3-200 ...`，`/var/lib/apt/lists` 被清空又没重新 `apt-get update` 就直接 `Unable to locate package tcl`。`rm -rf /var/lib/apt/lists/*` 要放在 EFA 那一层末尾，并在该层开头再 `apt-get update`。
2. 容器里 installer 用 `--skip-kmod --skip-limit-conf --no-verify`，但**不要** `--skip-rdma-core` —— 容器需要 rdma-core 64.0 的 libibverbs。
3. **`third-party/fmt` 是 submodule**，必须拉（`--recurse-submodules`，或钉 commit 后用 `git submodule update --init --recursive`），否则 `setup.py` 和 JIT 都缺 fmt 头文件。
4. **必须装 `ninja`**，否则 `AssertionError: With dlink=True, ninja is required to build cuda extension deep_ep._C`。
5. **必须装 `numpy` + `nvidia-ml-py`**（`pynvml`），`deep_ep/` 和 `tests/` 都 import，缺了运行时才炸。
6. **NVSHMEM 是 `setup.py` 的硬依赖**（legacy 后端还在 source list 里），只跑 v2 也要装 `nvidia-nvshmem-cu13`。
7. **NCCL 用 2.31+**：低于 2.31 时 DeepEP 断言编译期与运行期 NCCL 版本严格相等。torch 2.13.0+cu130 自带 nccl 2.29.7 / nvshmem 3.4.5，被后面的 `--no-deps` 安装覆盖成 2.31.2 / 3.7.2。
8. **`WORKDIR` 不能是 `/opt/DeepEP`**，否则 python 优先 import 源码目录里的 `deep_ep/`（不含 `_C.so`），报 `ModuleNotFoundError: No module named 'deep_ep._C'`。设成 `/workspace`，用绝对路径跑 test。
9. **插件文件名不是 `libnccl-net.so`。** installer 在容器里走 NGC 分支，装 `libnccl-ofi-ngc-v3`，落成 `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`。用 `NCCL_NET_PLUGIN=ofi` 让 NCCL 做名字解析，别写死路径。

### 3.3 本目录 Dockerfile 相对"能跑就行"版本的三处加固

前两处是复现性问题，第三处是它们暴露出来的。三处都**重建镜像验证过**，不是推理。

**1) DeepEP 版本用 `DEEPEP_REF`，默认钉一个 sha，但"跟最新"是一等公民。**
`DEEPEP_REF` 接受 branch / tag / 裸 sha 三种写法；`git clone --depth 1 --branch <sha>`
**不接受裸 sha**，所以只能 init + fetch：

```dockerfile
ARG DEEPEP_REF=8e7b42e9b22de4bf70d1de6858db3725c341b628      # 或 main
ADD https://api.github.com/repos/amazon-contributing/DeepEP/commits/${DEEPEP_REF} /tmp/deepep-ref.json
RUN mkdir -p /opt/DeepEP && cd /opt/DeepEP && git init -q \
    && git remote add origin https://github.com/amazon-contributing/DeepEP.git \
    && git fetch -q --depth 1 origin "$DEEPEP_REF" \
    && git checkout -q FETCH_HEAD \
    && git rev-parse HEAD > /opt/DeepEP/BUILD_REF \
    && git submodule update -q --init --recursive --depth 1 && ...
```

**为什么默认是 sha 而不是 `main`** —— 不是怕新代码，是三条具体的坑：

1. **§5 每个数字都对应一个确定的 tree。** 漂的话，重建出来数字对不上时，你分不清是
   自己环境不对还是代码变了。
2. **upstream 会改写历史。** 我们原来钉的 `ec623f3` 现在**已经不在 `main` 上**了 ——
   它被重写成 `cc55cce`（而且内容也动了：`buffer.hpp` / `nccl.cu` / `combine.hpp` /
   `elastic.py` 共 +41/−19），`main` 又往前走了 4 个 commit 到 `8e7b42e`。
   钉着的时候这件事是**可见的**（fetch 得到一个不在任何分支上的 sha）；漂着的时候
   它对你完全隐形。
3. **`ordered` kernel 在 EFA GDAKI 上不正确**（§7）。默认 kernel 选择哪天变了，
   漂的镜像会静默跑到错的 kernel 上。

**那个 `ADD` 不是装饰。** 没有它，`RUN git fetch origin main` 是一条固定的命令字符串，
Docker 会**命中旧 layer** —— 你以为拉到了最新，实际拿的是上次的代码。**这比明着钉死
更危险**：钉死是诚实地旧，缓存命中是假装地新。`ADD` 一个 URL 每次都会重取，内容变了
layer 才失效；钉 sha 时该 URL 内容恒定，缓存照样有效。（走 GitHub API，未认证限速
60 次/小时，仅用于算 cache key。）

**每次跑都把实际 sha 打进日志。** 镜像 tag 是人起的名字，`BUILD_REF` 是构建时
`git rev-parse HEAD` 的结果。`run_test_ep.sh` 会在开跑前打印：

```
=== IMAGE=deepep-v2-efa-official:dev  DeepEP=8e7b42e9b22de4bf70d1de6858db3725c341b628 ===
```

同 tag 重建过的镜像因此不会产生无法归属的数字。

**怎么知道 Xuan 推了新代码**（一条命令，不用建镜像）：

```bash
PINNED=$(grep -oE '^ARG DEEPEP_REF=\S+' Dockerfile | cut -d= -f2)
TIP=$(git ls-remote https://github.com/amazon-contributing/DeepEP.git refs/heads/main | cut -f1)
[ "$PINNED" = "$TIP" ] && echo "up to date" || echo "main 已前进：$PINNED -> $TIP，去重测后再 bump"
```

流程是**先量后 bump**，不是自动跟随：`main` 前进 → 跑 §5 的对照组 → 确认在噪声内 →
改默认值。`8e7b42e` 就是这么定的（对照见 summary.txt TABLE 8 的 8 组配对，全部落在 0.7% 以内）。

**2) 删掉 apt 的 `libnccl2` / `libnccl-dev` 2.28.3。** installer 的 NGC 分支会顺带装上它们并置成 `hold`。2.28.3 < 2.30.4 ⇒ **没有 GIN**，而 `ldconfig` 恰好把 `libnccl.so.2` 解析到它、`/usr/include/nccl.h` 也是它。留着的话任何人动一下 `LD_LIBRARY_PATH` / `LD_PRELOAD` 就会静默掉到没 GIN 的那份上。删是安全的：`libnccl-ofi-ngc-v3` 这个 deb **没有任何 `Depends` 字段**，`libnccl-net-ofi.so` **只链 `libfabric.so.1`、根本不链 libnccl**（它是被 NCCL 加载的插件），`apt-get -s remove` 也只动这两个包。

```dockerfile
RUN apt-mark unhold libnccl2 libnccl-dev \
    && apt-get remove -y libnccl2 libnccl-dev && ldconfig
```

**3) 补一个 `-L`，否则第 2 步会把构建搞坏。** 这是删掉 2.28.3 之后重建才暴露出来的：

```
/usr/bin/ld: cannot find -l:libnccl.so.2: No such file or directory
```

`setup.py` 的 link flags 是 `-l:libnccl.so.2` + `-Wl,-rpath,<pip>/nvidia/nccl/lib`，**只有 rpath，没有 `-L`**。也就是说原版镜像的 DeepEP 是**链接期解析到 apt 那份 2.28.3、运行期才由 rpath 切到 pip 那份 2.31.2**。头文件（`-I <pip>/nvidia/nccl/include`）一直是 2.31.2 的，所以编出来是对的，但依赖关系是隐式的。补上：

```dockerfile
NCCL_LIB="$(python3 -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')" \
  && test -e "$NCCL_LIB/libnccl.so.2" \
  && LIBRARY_PATH="$NCCL_LIB:${LIBRARY_PATH:-}" python3 setup.py install
```

顺带说明为什么不用改 `EP_HYBRID_KERNEL`：默认就是 EFA 需要的 `unordered`，不用显式钉（真要钉也无害，见 §6）。

### 3.4 B300 / `sm_103` 比 p5en 多出来的两处（脚本已自动处理，但必须知道）

在 2 × p6-b300 上照这份 runbook 走会撞到两个 p5en 复现不出来的失败。两个都不用改源码：
一个是 build 参数（`build_image.sh` 按 `compute_cap` 自动给），一个是运行时 env
（`run_test_ep.sh` 按设备列表自动注入）。**现象和修复是实测的，机制是在钉的那个
DeepEP tree 上读出来的。** 之所以还要在这里写清楚：这两个失败都在离原因很远的地方才炸，
你迟早会在别的镜像、别的 launcher 上再见到它们。

**坑 1：NCCL 只建了 2 个 GIN GDAKI NIC，rank 4 直接崩。**

```
transport/net_ib/gin.cc:262 (ncclGinIbGdakiGetProperties)
NCCL WARN NET/IB : Requested properties for GIN GDAKI NIC 4, only 2 GIN GDAKI NICs have been created
RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:185): 5
```

p6-b300 上 `ibv_devinfo -l` 返回 **18** 个设备：16 个 `rdmap*`（EFA）+ `ibp198s0f0` /
`ibp199s0f0`（**不是 EFA**，附录 A 的 `ce_probe` 对这两个是 CE FAIL / errno 95）。NCCL 默认的
网卡选择在这种混合列表上只建了 2 个 GDAKI NIC，而 `ElasticBuffer` 要的是**每个本地 rank 一个**
（每节点 8 个）。设 `NCCL_IB_HCA=rdmap`（前缀匹配，排掉那两个 `ibp`）即可，日志会变成
`NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0..7 'rdmap*'`，`ElasticBuffer` 初始化
成功并选中 `unordered` kernel。p5en 是 16 个纯 `rdmap*`，所以默认就能建满、碰不到。
`run_test_ep.sh` 现在会**在检测到非 `rdmap` 的 ibverbs 设备时**自动注入这个变量 —— 不无条件设，
否则会掩盖掉"这台机器上 EFA 设备根本不叫 rdmap"这种情况。

**坑 2：`sm_103` 上 elastic kernel JIT 编译失败（CUDA 13.0.2 的 ptxas）。**

```
NVCC compilation failed: ptxas ..._kernel.ptx, line 400; error : Arguments mismatch for instruction 'mov'
ptxas fatal   : Ptx assembly aborted due to errors
RuntimeError: Assertion (csrc/jit/compiler.hpp:239): "NVCC compilation failed"
```

**在第一次 dispatch 时才炸，build 阶段看不出来** —— 这些 kernel 是运行时用 base 镜像里的
`nvcc` JIT 出来的。`deep_ep/include/deep_ep/common/ptx.cuh` 里有多处
`#if __CUDA_ARCH__ >= 1000`（`:106` 的 `st.bulk`、`:229` 起的 `.v4.s64` LD/ST，形如
`ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s64 {4 个 64 位寄存器}`）；`sm_103` = 1030
会进这些分支，13.0.2 的 ptxas 不认。**修复：`--build-arg CUDA_VERSION=13.3.1`**，b300 上实测
换完 dispatch/combine 正常出数。pip 的 torch 仍然是 `cu130`（CUDA 次版本兼容），只动 base。

**这个坑没有宏可以绕。** `DISABLE_AGGRESSIVE_PTX_INSTRS` 在这个 pin 上只有
`csrc/kernels/legacy/utils.cuh`（V1 路径）引用，`deep_ep/include/deep_ep/impls/` 里**零处**；
而 JIT 自己在 `csrc/jit/compiler.hpp` 里拼 flags，只处理 `EP_JIT_CPP_STANDARD` /
`EP_NUM_TOPK_IDX_BITS` 那几个，`EP_JIT_EXTRA_FLAGS` 在源码里还是 TODO —— 根本没有往 JIT 传
`-D` 的入口。`setup.py` 对 `arch != 9.0` 加的那个 `-D` 只作用于 AOT 的 `_C.so`。把那个 TODO
实现掉，是值得往上游提的一条。

**所以 B300 的完整必要条件是：** host `efa.ko` 3.3.0（和 p5en 一样）+
`--build-arg CUDA_VERSION=13.3.1` + `--build-arg TORCH_CUDA_ARCH_LIST=10.3` + 运行时
`NCCL_IB_HCA=rdmap` + 多机显式 `--num-sms`。最后一条在 b300 上的理由和 p5en 不同：自动探测
**不是崩，而是偏低** —— `get_rdma_gbs` 只返回**一块**网卡的速率，而 b300 每 GPU 两张 EFA，
真值 100 GB/s 它只看到 50。

**这两处之外，b300 和 p5en 的流程一模一样** —— host 装法、GIN 那对环境变量、SM 要显式给、
campaign 怎么扫（§4.5，`sm103` 的默认 cell 表已经在里面），全都不变。

修完之后的抽查（2 节点 16 rank，`--num-tokens 8192`）：dispatch 12 SM ≈ **1025 µs**、
combine 24 SM ≈ **1800 µs**。这是两个不同 SM 档上各一次读数，**不是 campaign** ——
它只说明这套栈跑通了，不说明 B300 有多快。b300 上成体系的数字来自**另一个镜像**
（AWS `awsome-distributed-ai#1234` 那份 recipe：CUDA 13.1.2 / torch 2.11，DeepEP pin 相同），
在 `../adai-ep-comparison-b300/RESULTS_b300.md`。那批数字里和本文直接相关的两条：PR #2 的
clamp 在 b300 上复现且**更大**（decode dispatch −37.6%，p5en 是 −33.5%），但
**PR #1 的 `EP_NUM_SUB_PARTS=1` 在 `sm_103` 上是零到微负**（p5en 上是能叠加的），所以 §5.5
"再叠一个 `EP_NUM_SUB_PARTS=1`"那条建议是 Hopper 专属；另外 b300 上 clamp 是把 SM 曲线
**压平**而不是移动最优点（打了补丁之后 12 SM 和 24 SM 打平）。

---

## 4. 跑测试

`test_ep.py` **不是 torchrun 语义**：它自己 `torch.multiprocessing.spawn` 起 8 个 local rank。所以

- `WORLD_SIZE` = **节点数**（2），不是全局 rank 数
- `RANK` = **节点序号**（0 / 1）
- `--num-processes` = 每节点 local rank 数（8）

### 4.1 单机自检（不碰 EFA）

```bash
python3 /opt/DeepEP/tests/elastic/test_ep.py \
  --num-processes 8 --allow-hybrid-mode 0 --test-first-only
```

确认 DeepEP、NCCL、JIT 链路是好的。首次会 JIT 编译，慢一点正常（缓存在 `$EP_JIT_CACHE_DIR`）。

### 4.2 两机跑分（prefill 带宽 / decode 延迟只差一个 `--num-tokens`）

两个节点都跑，**只有 `RANK` 不同**：

```bash
NODE_RANK=0          # 另一台改成 1
MASTER=<节点0私网IP>
TOKENS=8192          # prefill 看带宽；decode 看延迟就改成 128

docker run --rm --name ep \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1 \
  --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -e EP_BUFFER_DEBUG=1 \
  -e MASTER_ADDR=$MASTER -e MASTER_PORT=8371 \
  -e WORLD_SIZE=2 -e RANK=$NODE_RANK \
  deepep-v2-efa-official:sm90 \
  bash -lc "python3 -u /opt/DeepEP/tests/elastic/test_ep.py \
      --num-processes=8 --num-tokens=$TOKENS --hidden=7168 --num-topk=8 \
      --num-experts=256 --num-sms=12 --allow-hybrid-mode=1 \
      --prefer-overlap-with-compute=0 --test-first-only"
```

上面这条裸 `docker run` 是 p5en 的。**b300 上还要加 `-e NCCL_IB_HCA=rdmap`**，否则 NCCL 只建
2 个 GDAKI NIC、rank 4 起不来（§3.4）；用 `run_test_ep.sh` 的话它会自动检测注入。

**`--num-sms` 一律显式给。** 在本文跑 campaign 的那个 pin（`ec623f3`）上不给会直接崩；在
Dockerfile 现在钉的 `8e7b42e` 上不崩，但探测值偏低（b300 上是真值的一半，§3.4）。崩的样子：

```
File ".../deep_ep/buffers/elastic.py", line 824, in get_theoretical_num_sms
ZeroDivisionError: float division by zero
```

`--num-sms 0`（默认=自动）→ `get_theoretical_num_sms()` → `get_rdma_gbs()` → `subprocess.run(['ibstat'])` 抓 `CA '<nic>' ... Rate: N`。而 `ibstat` 走 libibumad，EFA 没有 `ib_umad`：`ibstat rdmap85s0` → `ibpanic: ... IB device can't be found`。于是打一行 `Failed to get RDMA connection speed:` 返回 0；第 824 行的分母保护写的是 `num_rdma_ranks > 1`、判断却用 `num_scaleout_ranks > 1`，两者不一致 ⇒ 除零。**单机不触发**，只在上多机那一刻出现。

> **这条只对本文钉的 `ec623f3` 成立，别照搬到 `main`。** 修复是
> `a4f923c envs: probe RDMA link rate via sysfs, survive probe failure`（在 `main` 上是
> `e0f110a`）：先读 `/sys/class/infiniband/<nic>/ports/*/rate`，`ibstat` 只作 fallback，且
> `EP_NIC_NAME` 未设时自己挑最快的设备。该 commit **不在 `ec623f3` 里**（`git log
> ec623f3..7a6059a3` 可见），所以我们这个 pin 仍然必须显式给 `--num-sms`；换到 `main` 之后
> 自动探测就能用了。

**在这个 pin 上 `--num-sms` 不改 QP 数，是一根纯性能轴。** `ec623f3` 删掉了
`kMinGinContextSharingFactor`，auto path 变成常量 `kDefaultGinContextCnt = 11`，**与 SM 数无关**
（§5.4 有源码推导）。`results/p5en_2n4n_20260825/logs/` 里每一份日志都打 `#QPs: 11/11` —— 6 / 12 /
24 / 32 SM、2 节点和 4 节点，全都是 11。随 SM 变的那套（`ceil_div(num_sms × 4, 10)`，12 SM → 5、
24 SM → 10、上限 clamp 到 17）是 **`7a6059a3` pin** 的公式，pin 的日志在 12 SM 打 `#QPs: 5/5`。
`num_qps < num_ranks` 在这个 pin 上也不致命：4 节点 32 rank 传 `--num-allocated-qps 5` 拿到
`#QPs: 5/5`，照样跑完出数（§5.4）。所以 SM 可以放心扫。

**推荐工作点（p5en）：2 节点和 4 节点都用 24 SM。** 下表 8192 tok、GIN type 5、`--test-first-only`、
未开 `EP_BUFFER_DEBUG`，全 rank 均值（完整数据见
[`results/p5en_2n4n_20260825/summary.txt`](../results/p5en_2n4n_20260825/summary.txt) TABLE 5/6）：

| SM | 2 节点 dispatch | 2 节点 reduced combine | 2 节点合计 | 4 节点 dispatch | 4 节点 reduced combine | 4 节点合计 |
|---|---|---|---|---|---|---|
| 6 | 2290.5 µs | 7377.7 µs | 9668.2 µs | 4030.7 µs | 9518.8 µs | 13549.4 µs |
| 12 | 1502.9 µs | 4237.9 µs | 5740.8 µs | 3955.3 µs | 7943.2 µs | 11898.5 µs |
| 16 | 1510.5 µs | 3584.1 µs | 5094.6 µs | — | — | — |
| **24** | 1535.7 µs | **3362.6 µs** | **4898.3 µs** | 3972.7 µs | **7728.3 µs** | **11701.0 µs** |
| 32 | 1576.5 µs | 3486.4 µs | 5062.9 µs | — | — | — |

- **dispatch 从 12 到 24 SM 基本是平的**（2 节点 +2.2%、4 节点 +0.4%），付钱的是
  reduced combine。所以拿这点 dispatch 换 combine 总是划算：2 节点用 32.8 µs 的 dispatch
  换回 875.3 µs 的 reduced combine（层总时间 −14.7%），4 节点 −1.7%。
- **6 SM 在任何规模上都是错的选择**（相对 24 SM，两个口径都给出来）：2 节点 dispatch
  +49.2%、dispatch+redComb +97.4%；4 节点 dispatch 只 +1.5%，但 dispatch+redComb +15.8%。
  两个口径差这么远，正是因为付钱的是 reduced combine 而不是 dispatch —— 引用时必须说清
  是哪一个。
- **128 tok（decode）同样选 24 SM**：2 节点 dispatch 169.4 → 147.3 µs（−13.0%）、
  reduced combine 179.0 → 160.1 µs（−10.6%）；4 节点 dispatch 在 6/12/24 SM 上是平的
  （181.2–184.7 µs，散布 1.9%），只有 reduced combine 动，24 SM 比 12 SM 好 5.4%
  （253.3 → 239.6 µs）。这条适用于未打 §5.5 那两个 PR 的代码；打了之后 2 节点 decode
  反而 12 SM 更好。
- **这张表别搬到 b300 上。** SM 轴在本镜像的 `sm_103` 上还没扫过（§8 待办 8，`run_campaign.sh`
  的 `sm103` cell 表就是为它准备的）；另一个镜像上的观测是打了 §5.5 的 clamp 之后 b300 的
  SM 曲线是被**压平**（12 SM 和 24 SM 打平）而不是最优点移动，和 p5en 的形状不一样。
  b300 默认起点取 24 SM，理由只是"和 p5en 的工作点对齐好比较"，不是量出来的最优点。

### 4.3 日志里必须出现的 GIN 证据

```
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)      <-- 决定性的一行
NCCL INFO NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0 'rdmap86s0'   (HCA 0..7 各一行)
NCCL INFO Using network Libfabric
```

`EP_BUFFER_DEBUG=1` 每个 rank 另打一组：

```
DeepEP initialized with NCCL version: 2.31.2 (loaded library)
EP NCCL device communicator has 0 allocated QPs        <-- 正常，不是失败
GIN layout: gin_context_cnt=11, gin_indexed_signals_cnt=21, num_qp=11
```

- 只要 `Loaded gin plugin Libfabric_GDAKI (v14)` 出现，§0 那条依赖链就是全线打通的 —— 这一行同时证明了 v14 符号、GDA ops、CE verb、内核 capability 位。
- **`has 0 allocated QPs` 看着像出错，其实正常**：那是显式分配的 QP 数（没显式给就是 0），真正在用的是下一行 `num_qp=11`。很多人读到这里就以为没起来。
- 判 NCCL 版本用上面这行或 `NCCL_DEBUG=INFO` 的 `NCCL version 2.31.2+cuda13.3`，**别用 `torch.cuda.nccl.version()`**（它报 torch 编译期 header，永远是 2.29.7），见附录 B。

### 4.4 其他可选测试

```bash
python3 /opt/DeepEP/tests/elastic/test_barrier.py     # 最小连通性
python3 /opt/DeepEP/tests/elastic/test_pp.py          # pipeline 并行
python3 /opt/DeepEP/tests/legacy/test_internode.py    # 旧 NVSHMEM 后端（对照组）
/opt/amazon/efa/bin/fi_pingpong -p efa [<对端私网IP>]  # 纯 libfabric 连通性，排除 DeepEP
```

### 4.5 一条命令跑完整个 campaign（p5en 和 B300 同一套）

`run_test_ep.sh` 是"一台机器上的一个 cell"。`run_campaign.sh` 从你的笔记本 ssh 驱动所有
节点、按矩阵逐个 cell 跑、并把每个日志命名成 `results/*/make_tables.py` 能直接 pool 的
形式。**两个架构用同一个脚本**，arch 只决定默认的 cell 列表。

```bash
# 1. 先降到单机 8 rank（reductive first）。B300 那两个坑都在这一步现形，2 分钟成本，
#    而且能把"镜像编错了"和"网络起不来"分开。
IMAGE=deepep-v2-efa-official:sm103 WORLD_SIZE=1 NUM_PROCESSES=8 \
TOKENS=128 NUM_SMS=24 MASTER_PORT=8499 NCCL_DEBUG=INFO \
  ./run_test_ep.sh 0 127.0.0.1 2>&1 | tee /tmp/smoke.node1.log
./verify_run.sh /tmp/smoke.node1.log

# 2. campaign（arch 从 leader 探；也可以位置参数显式给 sm90 / sm103）
NODES="<leader> <worker>" ./run_campaign.sh

# 3. 把**每台**机器的日志都拉回来，先验收再看数
for n in 1 2; do scp "<node$n>:~/epruns/*.node$n.log" ./logs/; done
./verify_run.sh logs/*.log
EPRUNS=./logs python3 results/p5en_2n4n_20260825/make_tables.py
```

默认 cell（3 个 rep，**rep 内轮换**：每个 cell 每个 rep 各跑一次，不是按臂分块跑完再换，
这样漂移不会被读成臂效应）：

| arch | cells |
|---|---|
| `sm90` | `official` × {8192, 128} tok × {12, 24} SM；`prs` × {8192, 128} tok @ 12 SM；`prs` + `EP_MIN_TOKENS_PER_PART=1` @ 128 tok |
| `sm103` | `official` × {8192 @ 24, 128 @ 24, 8192 @ 12}；`prs` × {8192, 128} @ 24 SM；`prs` + `EP_MIN_TOKENS_PER_PART=1` @ 128 tok |

要改就 `CELLS="arm|image|tokens|sms|knobtag|额外 env"`，一行一个 cell。`prs` 那条臂的镜像
不存在时会**整条跳过并打一行提示**，而不是让 9 个 run 一个一个失败。`sm103` 里那个 12 SM
的 prefill cell 是为了和 §3.4 里那两个先于本 campaign 的点读对齐。

**`prsmtpp1` 是负控，不是一个变体。** PR #2 在 `EP_MIN_TOKENS_PER_PART=1` 时短路回打
patch 前的几何，所以那个 cell 是**新 binary 里的旧行为**。它要是没落回 `official` 臂，
说明差异来自构建或环境，而不是 clamp。

驱动脚本替你守住的东西，每一条都是踩过的：

- **每个 cell 都带 `EXTRA_ENV="NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0"`**，并按有无
  它在 tag 里写 `_gin5` / `_type2` —— 两种后端永远不可能被 pool 到一起。§6.1。
- **每个 cell 换一个 `MASTER_PORT`**：被 kill 的 run 会留下 TCPStore listener，下一个卡在
  rendezvous。
- **每一个轴都进文件名**：arm、节点数、SM、tokens、knob、debug、后端、rep、node。少一个轴
  就是静默覆盖掉另一条臂的日志。
- **`EP_BUFFER_DEBUG` 全程不设**、`--ignore-local-traffic` 全程带、`--num-sms` 全程显式，
  `NUM_SMS` 不用 0。
- **前台 ssh，不用 `nohup`**：detached launch 在断管时是**不对称**失败的，活下来那边重启会
  覆盖已发布的日志。
- **cell 之间等 20 s**：`run_test_ep.sh` 会拒绝在忙 GPU 上启动 —— 上一轮残留的 rank 会让
  下一轮以 `rc=0` 报出约 2 倍的延迟。
- **不给 JIT cache 挂 host 目录**：每个 `--rm` 容器重新编（1–3 min，发生在 `test_ep.py`
  自己的 warmup 里，不在计时区）。这个成本换来的是对 §7 规矩 1 那个坑的免疫。真要挂，就
  **一个镜像 tag 一个 host 目录**。

`verify_run.sh` 是验收闸门：丢 rank（先按**本机 local rank** 逐文件查，再按 tag 汇总对
world）、B300 那两个坑的签名、tag 名和实际 env 不符 —— 这些是 FAIL；type-2 后端、
`EP_BUFFER_DEBUG`、`--num-sms=0`、缺 `--ignore-local-traffic`、镜像没有 `BUILD_REF` 戳 ——
这些是 WARN。**run 自己 `rc=0` 一条都不能证明。**

---

## 5. 实测结果（`p5en.48xlarge` × 2 和 × 4，2026-08-25）

**本节全部是 p5en / `sm_90`。** b300 在本镜像上目前只有 §3.4 的抽查（两个 SM 档各一次读数），
成体系的 b300 数字来自另一个镜像、在 `../adai-ep-comparison-b300/RESULTS_b300.md`，
**和本节不同口径不能混排**；本镜像的 b300 campaign 是 §8 待办 8，跑法见 §4.5。

环境：Ubuntu 24.04，driver 595.91.07，各 8×H200 + 16 EFA，同 AZ 同 cluster placement group，
installer 1.50.0（production tarball，md5 `e5a5178944b1f1112f3b2eb3b15ca5a7`，efa.ko 3.3.0g）。
容器 `deepep-efa:1.50.0`（torch 2.13.0+cu130 / nccl 2.31.2+cuda13.3 / `deep_ep 2.1.0+ec623f3`）。
16 卡和 32 卡 `test_ep.py` 都 **exit 0，正确性检查全过**；日志两侧都报
`Selected provider is efa, fabric is efa-direct`。

**§5.1–5.3 都在 GIN type 5 上**（`NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0`，见 §6），
`--test-first-only`，未开 `EP_BUFFER_DEBUG`，SM = **24**（§4.2 的工作点）。§5.4 / §5.5 的对照实验
在 **12 SM** 上做，因为要和手编栈以及 PR 分支对齐；每张表自己标了 SM 数。

**口径**（每次报速率都要一起报，只报 GB/s 曾经把结论弄反过）：

- **时间是全 rank 均值** —— 2 节点 16 个 rank、4 节点 32 个 rank，从每台机器的日志里汇总。
  单台机器的均值**不能**代替它：combine 是按节点分层的（见本节末），单节点均值能偏 12%。
- **`SO` / `SU` 是跨 rank 的 min–max，时间是同一批样本的均值** —— 两列的统计口径不同，
  详见 §5.1 末。
- `SU` = **日志里打印的那个 `bytes` 字段 ÷ 时间**。那个字段是 `num_scaleup_bytes`
  （`tests/elastic/test_ep.py:271`），即本 rank 收到的**全部** token 字节；
  日志里**不打印** scale-out 字节，只能用 `SO × 时间` 反推。
- `SO` **不是线速** —— `test_ep.py:253` 的循环在不加 `--ignore-local-traffic` 时会把
  本机也算作一个 scale-out 目的地，所以机内流量也计入。
  **线速占比 = `SO × (N−1)/N ÷ 每 GPU 线速`**。分母**按机型换**：p5en 是 **50 GB/s**
  （16×200 Gb/s ÷ 8 = 每 GPU 一张 EFA），p6-b300 是 **100 GB/s**（每 GPU 两张）。
  拿 50 去除 b300 的 SO 会得到约两倍的"线速占比"，这是最容易犯又最难发现的一个错。
  p5en 上 N=2 时线速占比数值上等于 SO，N=4 时是 SO × 1.5（这个巧合来自分母恰好是 50）。
  这个修正假设 N 个目的节点均分，
  是近似；要精确值就加 `--ignore-local-traffic` 重跑，`SO` 会直接变成真跨机字节 ÷ 时间。
- 每 rank **scale-up**（= 打印的 `bytes`）字节，8192 tok：dispatch 2 节点 395.9–402.4 MB /
  4 节点 441.4–447.7 MB，combine 759.7–772.1 / 847.0–859.0 MB。128 tok：dispatch
  5.6–6.3 / 5.8–6.4 MB，combine 10.8–12.1 / 11.1–12.3 MB。
- 每 rank **scale-out** 字节（`SO × 时间`，含机内），8192 tok：dispatch 2 节点
  121.3–123.0 MB / 4 节点 219.1–223.6 MB，combine 232.7–236.2 / 420.9–428.5 MB。

### 5.1 Prefill（`--num-tokens=8192`）— 带宽

时间是**均值**，`SO` / `SU` 是**跨 rank 的 min–max**（下面「区间从哪来」解释为什么）。

| op | 2 节点 时间 | SO | SU | 4 节点 时间 | SO | SU |
|---|---|---|---|---|---|---|
| dispatch | **1535.7 µs** | 79–80 | 257–263 | **3972.7 µs** | 55–56 | 111–113 |
| expanded dispatch | 1537.3 µs | 79–80 | 258–262 | 3971.3 µs | 55–56 | 111–113 |
| cached dispatch | 1620.7 µs | 75–76 | 244–249 | 4256.6 µs | 51–52 | 103–106 |
| combine | 3534.0 µs | 62–73 | 202–239 | 7710.7 µs | 51–57 | 103–115 |
| reduced combine | 3362.6 µs | 64–79 | 209–259 | 7728.3 µs | 51–57 | 103–115 |

（2 节点 3 轮 × 16 rank = 48 个 rank 观测，4 节点 2 轮 × 32 rank = 64 个，全部汇总；
`results/p5en_2n4n_20260825/make_tables.py`，跑一次即可重现整张表。）

**dispatch 在两个规模上都跑在线速的 80–84%**：2 节点 79.6%（`SO` 均值 79.6，N=2 时线速%
数值上等于 `SO`），4 节点 83.7%（`SO` 均值 55.8 × 1.5）。4 节点利用率反而更高的原因：
每 rank 计入的 scale-out 字节从 122.1 涨到 221.3 MB（+81%），其中真正跨机的比例又从
1/2 升到 3/4，于是**真跨机**字节是 61.0 → 165.9 MB（×2.72），而时间只涨到 2.59×。

#### 区间从哪来 —— 它不是重复性方差

`SO` / `SU` 那两列是**同一时刻 48（或 64）个 rank 之间的 min–max**，不是同一个数字重复跑
若干次的离散。两者是不同的东西，混起来看会以为方差大得离谱。拆开之后：

| 2 节点 / 24 SM / 8192 tok | 时间跨 rank 离散 | 打印字节跨 rank 离散 | `SO` 离散 |
|---|---|---|---|
| dispatch | **0.5%** | 1.6% | 2.1% |
| expanded dispatch | 0.3% | 1.6% | 1.7% |
| cached dispatch | 0.4% | 1.6% | 1.9% |
| combine | **16.1%** | 1.6% | 16.8% |
| reduced combine | **20.7%** | 1.6% | 21.8% |

（4 节点：dispatch 时间 0.9% / 字节 1.4% / `SO` 2.3%；combine 时间 11.1%。）

1. **dispatch 的区间来自分母，不是速度。** 时间跨 rank 只差 0.5%（4 节点 0.9%），
   但每个 rank 路由到的 token 数不同（`get_unbalanced_scores` 制造的不均衡），
   字节数差 1.4–1.6%，而 `GB/s = 本 rank 字节 ÷ 本 rank 时间`，于是把字节的离散继承了过来。
2. **再叠一层整数打印。** `test_ep.py` 用 `:.0f` 打印 GB/s。`SO ≈ 79.6` 时一个整数单位
   就是 **1.26%**，4 节点 `SO ≈ 55.8` 时是 **1.79%**。所以 `79–80` 和 `55–56` 是
   **两个相邻整数 —— 这个打印格式能表达的最窄区间**。dispatch 上根本没有可测的方差问题。
3. **真正宽的只有 combine / reduced combine**，而那里的驱动是**时间**不是字节：字节仍然只差
   1.6%，时间却差 16–21%。这就是下面 §5.3 的按节点分层，不是网络抖动 —— 2 节点上
   node1 3300.3 µs vs node2 3767.7 µs（13.2%），4 节点上看起来只有 4.2%
   （7862.5 / 7641.1 / 7541.1 / 7798.1 µs）—— 那是因为两轮里慢的机器不是同一台，汇总时
   互相抵消了。dispatch 完全不分层（两节点差 0.1%）。
4. **重复性本身极好，连 combine 也一样。** 逐轮的全 rank 均值：

   | 24 SM / 8192 tok | rep1 | rep2 | rep3 | 轮间离散 |
   |---|---|---|---|---|
   | 2 节点 dispatch | 1535.5 | 1537.0 | 1534.5 µs | **0.16%** |
   | 2 节点 combine | 3533.2 | 3535.1 | 3533.6 µs | **0.05%** |
   | 2 节点 reduced combine | 3366.3 | 3356.1 | 3365.5 µs | **0.30%** |
   | 4 节点 dispatch | 3971.6 | 3973.8 | — | **0.06%** |
   | 4 节点 combine | 7713.8 | 7707.6 | — | **0.08%** |

   所以 combine 那 16–21% 是**结构性的、可重复的 rank 间分布**，不是不稳定 ——
   全 rank 均值在不同轮次之间只差 0.05–0.30%。

**一句话结论**：dispatch 是紧的（48–64 个 rank 观测、2–3 轮，时间跨 rank 离散 0.5–0.9%、
轮间 0.06–0.16%），打印出来的 1 GB/s 量化本身就占 1.3–1.8%，所以 `79–80` 已经是最窄区间；
唯一有真实离散的是 combine，它是**按机器分层**的（可重复，轮间 0.05%），
而且慢的那台**在同一批连续 rep 里是固定的、只在不同批次之间翻转**（见 §5.3）。

### 5.2 Decode（`--num-tokens=128`）— 延迟

这个尺寸每 rank 只有 5.6–6.4 MB，是**消息率受限**不是带宽受限，只看时间：

| op | 2 节点 | 4 节点 |
|---|---|---|
| dispatch | **147.3 µs** | **184.7 µs** |
| expanded dispatch | 146.6 µs | 184.0 µs |
| cached dispatch | 135.8 µs | 184.3 µs |
| combine | 151.9 µs | 236.9 µs |
| reduced combine | 160.1 µs | 239.6 µs |

从 2 节点到 4 节点 dispatch 只涨 25%，**这是 type 5 的特性**：type-2 代理后端在同一组机器上
是 365.1 → 1003.2 µs（2.75×）。§5.5 的两个 PR 能把 2 节点 dispatch 再压到 106 µs。

### 5.3 combine 按节点分层 —— 这决定了怎么汇总

dispatch 在 rank 之间是均匀的（2 节点 24 SM 8192 tok：16 个 rank 全在 1531–1538 µs），
combine 和 reduced combine **不是** —— 它们按机器整齐地分成两层：

| 2 节点 / 24 SM / 8192 tok，3 轮汇总，各取本机 8 个 rank 的均值 | node1 | node2 |
|---|---|---|
| dispatch | 1535.0 µs | 1536.3 µs（差 0.1%） |
| combine | 3300.3 µs | 3767.7 µs（差 13.2%） |
| reduced combine | 3074.9 µs | 3650.3 µs（差 17.1%） |

**慢的那台机器在同一批连续 rep 里是固定的，只在不同批次之间翻转** —— 它是 per-launch 的
性质，不是 per-iteration 的噪声。官方镜像 24 SM 那三轮全是 node2 慢（combine
3764 / 3771 / 3769 µs 对 3303 / 3299 / 3299），而 `main` 的两轮和 pin 的两轮都是另一台慢；
幅度在所有批次里都稳定在 14–18%。所以汇总多轮**不会**摊平 2 节点的分层，单节点的 combine
均值会落在 ~3100 或 ~3770 µs，取决于你读的是哪份日志。汇总全部 16 个 rank 之后，24 SM
那三轮彼此只差 0.30%（reduced combine 3366.3 / 3356.1 / 3365.5 µs）。**报全 rank。**
4 节点上两轮里慢的机器不同（rep1 是 node4 的 8087 µs，rep2 是 node1 的 8117 µs），
所以汇总后的 4 节点分层只剩 4.2%。

### 5.4 和从源码手编整套栈的对比

`deepep-v2-efa-gdaki-h200:dev-7a6059a3` 是手编 NCCL `2.30.7-1` + 手编 `--enable-gdaki`
aws-ofi-nccl + DeepEP `7a6059a3` 的栈，多带一个 `--skip-check`。同一批机器、同一天、
12 SM、两侧都在 type 5 上：

| arm | dispatch | cached dispatch | combine | reduced combine |
|---|---|---|---|---|
| 2 节点 8192 tok 手编 pin | 1517.0 µs | 1749.1 µs | 3621.2 µs | 4166.2 µs |
| 2 节点 8192 tok 正式包 | 1502.9 µs | **1591.0 µs** | 3602.5 µs | 4237.9 µs |
| 2 节点 128 tok 手编 pin | 301.2 µs | 269.5 µs | 175.9 µs | 188.3 µs |
| 2 节点 128 tok 正式包 | **169.4 µs** | 165.9 µs | 162.7 µs | 179.0 µs |
| 4 节点 8192 tok 手编 pin | 3962.5 µs | 4709.2 µs | 7993.7 µs | 8067.7 µs |
| 4 节点 8192 tok 正式包 | 3955.3 µs | **4239.7 µs** | 7842.6 µs | 7943.2 µs |
| 4 节点 128 tok 手编 pin | 320.9 µs | 287.3 µs | 277.4 µs | 286.1 µs |
| 4 节点 128 tok 正式包 | **184.3 µs** | 179.5 µs | 244.2 µs | 253.3 µs |

prefill dispatch 是平手（差 0.2–0.9%），正式包在 **cached dispatch 上快 9.9–11.1%、
decode dispatch 上快 1.74–1.78×**。**没有任何性能理由去从源码编这套栈。**

**两边跑的是同一个 kernel** —— 结论按 blob 比出来的，不是读 commit message 猜的；以下全部
可从源码核实，无需机器。`ec623f3` 的标题是 `feat: add EP_HYBRID_KERNEL toggle between
unordered and ordered kernels`，`csrc/kernels/elastic/kernel_select.hpp:37-52` 写明 **unordered
是 default**（env 为空即 unordered），于是这个 commit 里 hybrid dispatch kernel 有两份。比一下
blob 就知道谁是谁：

| `ec623f3` 里的文件 | vs `af9a040:hybrid_dispatch.cuh`（fork 之前的 upstream） | vs `7a6059a3:hybrid_dispatch.cuh`（= 手编 pin 跑的） |
|---|---|---|
| `hybrid_dispatch.cuh` —— 即 `ordered` 变体 | **+6 / −1** | +56 / −468 |
| `hybrid_dispatch_unordered.cuh` —— **default** | +485 / −51 | **+45 / −28**（其中 11 行是 license header） |

所以 `unordered` **就是** `7a6059a3` 那套 kernel（AWS 的 EFA GDA 移植版）改了个名字，而 `ordered`
是这个分支 fork 之前的 upstream kernel、几乎原样带着走。combine 同理（`+12/−6` vs `+24/−7`）。
那 +45/−28 的全部实质内容是：多一个 `num_unaligned_recv_tokens_per_expert` 输出指针（每个 expert
一次 store）、两个 lambda 从两个 warp 分支里提到外面、删掉恒等 lambda `phys_token_slot`，加注释。

⇒ **`EP_HYBRID_KERNEL=ordered` 不是一个有效的 A/B。** 那个开关选的是 *upstream* kernel：
发端 publish 一个 trailing tail signal、收端假定 tail 之前的全部已落地 —— 实测它在
**EFA GDAKI 上结果不正确**（`NCCL_GIN_TYPE=5`，signal 可能超过数据），只在有序的 proxy
路径上成立。而且它向 NCCL 要的是另一套 fabric（`csrc/kernels/backend/nccl.cu:111-127`：
129 个 **exclusive** context、`ginSignalCount = num_ranks + 4`、`ginVaSignalsRequired`、
`ginStrongSignalsRequired`），就算正确也不是单变量对照。

**26 vs 9 那个 commit 分叉，大部分只是 SHA 层面的。** `merge-base(7a6059a3, ec623f3) = af9a040`，
`7a6059a3` 独有 26 个 commit、`ec623f3` 独有 9 个；但按**内容**看那多半是同一批活 rebase 过：
`qp_mapping.cuh` —— 也就是 pin 自己的 tip commit `7a6059a` "perf(gin): Balanced contiguous
QP-channel mapping" 落地的地方 —— 两版只差 **+10 / −0，全是 license header**。commit 数只能定
分支，不能定行为。

**这样只剩下 dispatch 路径上唯一一处实质差异：`gin_resource_alloc.cuh`（+80 / −50），即 context
数的策略。** 两边在 12 SM / 每 SM 4 channel / 16 rank 下都能从源码算出来，且和各自日志逐位吻合：

| | context 数（= QP 数） | signal/ctx | data QP（= ctx − 1 个 notify） | channel/ctx | `kNumParts` | channel/SM | 日志 |
|---|---|---|---|---|---|---|---|
| `7a6059a3`（手编 pin） | `ceil_div(12×4, kMinGinContextSharingFactor=10)` = **5**，**随 SM 变** | `(256 − 2·5)/5` = 49 | 4 | `ceil(48/4)` = 12 | `49/12` → **4** | 4 | `5 / 49 / 5` ✅ |
| `ec623f3`（正式包） | `kMinGinContextSharingFactor` 已删除，auto path 改成固定常量 `kDefaultGinContextCnt` = **11**，**与 SM 无关** | `(256 − 2·11)/11` = 21 | 10 | `ceil(48/10)` = 5 | `21/5` → **4** | 4 | `11 / 21 / 11` ✅ |

**注意没变的东西：两边都是 `kNumParts = 4`、48 个 channel**，也就是 kernel 的模板实参完全一样。
唯一活着的差别是这 48 个 channel 摊在多少个 QP 上 —— **手编 pin 是 4 QP × 12 channel，
正式包是 10 QP × 5 channel** —— 而下面的表把这个差别测出来是：对 plain dispatch 一文不值。

算术已经排掉两个候选机制：

- **`kNumParts` 不是机制。** 上表按 `compute_part_allocation()`（`gin_resource_alloc.cuh:122-152`）
  算出来**两边都是 4**；`ec623f3` 自己的注释也这么说（11 的等价选项 `{5, 6, 7, 8, 9, 14}`）。
- **正式包不是"少了调优"。** front-loading（`kMidTotal`）和 forward double-buffer
  （`kNumDispatchFwdBuffers`）都在 `hybrid_dispatch_unordered.cuh` 里 —— 这是必然的，因为那个文件
  **就是** `7a6059a3` 的 kernel。

**两个 commit 上这个 flag 的语义不同**，想在手编 pin 上试要注意。两边
`tests/elastic/test_ep.py` 都有 `--num-allocated-qps` / `--num-qps`，但 `7a6059a3` 上请求值会被按
SM 算出来的 context 数**从上面截断**（`nccl.cu:172-179` 打个 warning 然后压回去），所以 12 SM 下
5 是上限、11 走不到；`ec623f3` 上请求值**直接替换**默认值，只受
`[kMinGinContextCnt=2, kMaxGinContextCnt=17]` 约束（`nccl.cu:129-137`）。

**把 2 节点 prefill 的所有 arm 放在一根轴上**（16 rank / 12 SM / 8192 tok，全 rank 均值）：

| arm | dispatch | SO GB/s | 线速% |
|---|---|---|---|
| 正式包 `ec623f3`，默认 env（type 2） | 1644.0 µs | 74.0 | 74.0 |
| 正式包 + `--num-allocated-qps 5`（type 2） | 1631.1 µs | 75.0 | 75.0 |
| 手编 pin `7a6059a3`，`--skip-check` | 1517.0 µs | 80.6 | 80.6 |
| 正式包 + 完整 5 个 route-B 变量 | 1504.4 µs | 81.0 | 81.0 |
| **正式包 + `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`** | **1502.9 µs** | 81.2 | 81.2 |
| `main` `8e7b42e` + 那一对 | 1502.0 µs | 81.0 | 81.0 |

`线速% = SO × (N−1)/N ÷ 50 GB/s`，N=2 时数值上等于 SO。**这一整段差距全部由那两个环境变量
解释** —— 不是 QP fan-out，也不是 `--skip-check`：另外三个 route-B 变量叠上去毫无增益，
`main` 和 `ec623f3` 无法区分。QP flag 在 type 5 之上对 plain dispatch 是零影响
（1502.9 → 1508.5 µs），它真正管的是 **cached dispatch，而且 5 个 context 更差 9.6%**
（1591.0 → 1743.9 µs）；4 节点 decode 上更差 19.3%（1003.2 → 1197.0 µs）。**保持默认 11。**

`OFI_NCCL_GIN_STRONG_SIGNAL=1` 单独跑在 128 tok 上是 **750.4 µs**（16 个 rank 散在
371.0–1130.0 µs），即主动变坏 —— 和 unordered kernel 只要 weak signal 一致。

### 5.5 两个待合的 PR：decode dispatch 再减 33%

decode 上有两个互相独立的杠杆：GIN 后端（只改 env，§6）和 dispatch 的 part 几何（两个待合的
PR）。两者**可以叠加**。镜像用 PR #2 的 head `b097b03`，PR #1 是它的祖先，所以这一个 SHA
就是"两个 PR 叠加"这个 arm。

这个 head 与 `main` 的 merge-base 是 `cc55cce`，所以打了补丁的镜像和未打补丁的 `ec623f3`
镜像**不共享 base**。两点让对比仍然站得住：clamp-off 对照组（`EP_MIN_TOKENS_PER_PART=1`）
是**在同一个打了补丁的镜像里**跑的，base 恒定，所以 clamp 的效果是干净的
（2 节点 171.5 → 112.7 µs，−34.3%）；而 `main` `8e7b42e` 包含补丁 base 的全部内容还多 4 个
commit，它和 `ec623f3` 的 8 组配对全部落在 0.7% 以内（summary.txt TABLE 8）。

| PR（基于 `main`） | 内容 |
|---|---|
| [amazon-contributing/DeepEP#1](https://github.com/amazon-contributing/DeepEP/pull/1) | 把 `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` / `EP_MIN_TOKENS_PER_PART` 转发给 JIT，不改任何默认值 |
| [amazon-contributing/DeepEP#2](https://github.com/amazon-contributing/DeepEP/pull/2) | 加 `kMinTokensPerPart`（默认 15，`EP_MIN_TOKENS_PER_PART` 可覆盖），`kNumParts = min(预算, 每 channel token 数 / 15)` |

**2 节点 / 16 rank / 12 SM / 128 tok**（全 rank 均值）：

| arm | dispatch | 相对同后端 | combine | reduced combine |
|---|---|---|---|---|
| 未打补丁，type 2 | 365.1 µs | 1.00× | 175.1 µs | 192.7 µs |
| #1 + #2，type 2 | 237.1 µs | **−35.1%** | 176.3 | 193.6 |
| 未打补丁，type 5 | 169.4 µs | 1.00× | 162.7 | 179.0 |
| #1 + #2，type 5 | 112.7 µs | **−33.5%** | 162.2 | 178.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`，type 5 | **106.4 µs** | **−37.2%** | 162.1 | 178.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`，type 5 | 171.5 µs | 关掉 clamp 的对照 | 162.6 | 178.8 |

**4 节点 / 32 rank / 12 SM / 128 tok：**

| arm | dispatch | 相对同后端 | combine | reduced combine |
|---|---|---|---|---|
| 未打补丁，type 2 | 1003.2 µs | 1.00× | 344.2 µs | 357.0 µs |
| #1 + #2，type 2 | 627.4 µs | **−37.5%** | 336.9 | 346.8 |
| 未打补丁，type 5 | 184.3 µs | 1.00× | 244.2 | 253.3 |
| #1 + #2，type 5 | 169.5 µs | **−8.0%** | 243.5 | 252.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`，type 5 | **155.9 µs** | **−15.4%** | 243.6 | 251.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`，type 5 | 184.1 µs | 关掉 clamp 的对照 | 243.5 | 252.9 |

三条结论：

1. **两个杠杆互相独立、可以叠加。** 叠起来 2 节点 decode dispatch 从 365.1 → 106.4 µs，
   **3.43×** —— 一对环境变量加两个 commit。
2. **combine 和 reduced combine 在每一行里都动不到 1%。** 这就是"机制在 dispatch 内部的
   part 几何、不在网络"的证据：EFA 两边干的活一样多。expanded / cached dispatch 跟着
   plain dispatch 走。
3. **clamp 的收益在 4 节点 type 5 上崩掉了** —— −8.0% 而不是 −33.5%，而且关掉 clamp 的对照
   （184.1 µs）落在未打补丁（184.3 µs）的噪声里，说明是 clamp 本身不再付钱、而不是 PR 里的
   别的东西。type-5 4 节点 decode dispatch 有一个 **156–185 µs 的地板**，part 几何摸不到；
   `EP_NUM_SUB_PARTS=1` 走得最远，也就到这里。**原因未知 —— 不要把 2 节点那个比例外推到
   更大的集群。**

**为什么 part 几何会有代价**（combine 完全不动就是证据）：
`kNumParts`（一个 channel 的 token 分几次 `flush_part` put 发出去）只由 `compute_part_allocation()` 决定，而它只在 GIN indexed-signal 预算紧时**从上面**压；预算恰恰在 channel 里 token 最少时最松，所以 decode 必然落到 `kMaxParts` —— 轴的最坏一端。128 token / 12 SM 下每 channel 只有 3 个 token 却被描述成 4 part × 1 token：最后一个 part 永远是空的，3 个 token 发成 3 次单 token put 而不是 1 次 3-token put。sub-part 早有这两道保护（clamp 到 `kBatchSize`、`EP_SM100_MIN_SUB_TOKENS`），part 一道都没有。

**取舍**：只关心 prefill —— `main` 直接用，Dockerfile 不用改。这两个 PR 在 prefill 上是噪声
（2 节点 24 SM 8192 tok：未打补丁 1535.7 vs 1536.0 µs；4 节点 12 SM 8192 tok：3955.3 vs 3955.3 µs）。
**要发 decode / 小 token 的数字 —— 合并前自己 cherry-pick 这两个**，同时无论如何都要设 type-5
那一对。#1 不改任何默认值，所以必须和 #2 叠加才有收益。合并后 #2 的默认值 15 自动生效，不需要设
环境变量；要回未 clamp 的几何做对照给 `EP_MIN_TOKENS_PER_PART=1`（它是**短路**回旧值 ——
除以 1 是第三种几何，不是对照组）。

**打了 PR 之后 2 节点 decode dispatch 反而 12 SM 更好**（112.7 vs 145.3 µs），所以 §4.2 那条
"2 节点 decode 用 24 SM"适用于未打补丁的代码；4 节点上两档在 dispatch + reduced combine 上打平
（422.4 vs 422.2 µs）。

---

## 6. 环境变量速查

| 变量 | 用的值 | 说明 |
|---|---|---|
| `EP_HYBRID_KERNEL` | `unordered`（默认） | **EFA 上必须保持 unordered。** 设成 `ordered` 走上游 kernel，要求 VA signal + strong signal，EFA 不支持 |
| `EP_NIC_NAME` | `rdmap85s0` | 上游默认 `mlx5_0` 是 IB 的。用 `ibv_devinfo -l` 查实际名。它只喂 `get_rdma_gbs()`；在钉的 `8e7b42e` 上那条路**是通的**（先读 sysfs），但只返回**一块**网卡的速率 —— p5en 每 GPU 一张（50 GB/s，正好），b300 每 GPU 两张（真值 100，探测到 50）。所以还是直接给 `--num-sms` |
| `NCCL_IB_HCA` | b300 上 `rdmap` | 只在这台机器还有非 EFA 的 ibverbs 设备时才需要（b300 有 2 个 `ibp*`），不设 NCCL 只会建 2 个 GDAKI NIC 然后崩。`run_test_ep.sh` 自动检测注入（§3.4） |
| `EP_MIN_TOKENS_PER_PART` / `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` | 见 §5.5 | decode 的 part 几何。四个都要靠 PR #1 才能进 JIT，没有 #1 时它们**静默无效** |
| `NCCL_NET_PLUGIN` | `ofi` | 用名字解析，别写死路径 |
| `FI_PROVIDER` | `efa` | 是 `efa` 不是 `efa-direct`，fabric 由插件自己选 |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | |
| `NCCL_GIN_TYPE` | **`5`** | **必须显式设，否则默认走 type 2 代理后端**，prefill 慢 ~9%、decode 慢 2.2–5.4×。必须和下一行成对设，单独设会崩。详见本节末 |
| `NCCL_SYM_GIN_KERNELS_ENABLE` | **`0`** | 同上。sym GIN kernel 要 strong signal，GDAKI 没有 |
| `EP_BUFFER_DEBUG` | **只在调试时** `1` | 打 NCCL 版本 / QP 数 / GIN layout。**但它在计时区间里 printf** —— `csrc/elastic/buffer.hpp:1151` 在 dispatch 的 host 轮询循环里拼 stringstream 打 "CPU side received count"。而且 `deepep-v2-efa-gdaki-b200` 的启动脚本不转发它、`run_test_ep.sh` 转发，所以**只有一侧设了的跨镜像对比是被污染的**（它的代价我们没能从 type-2 那一侧自身的轮间散布里分离出来，所以不给数字）。确认完 layout 就关掉 |
| `EXTRA_ENV` | `"NAME=VALUE …"` | `run_test_ep.sh` 的一次性 env 钩子，用来做单变量 A/B（上面那对 GIN 变量就是这么传的） |
| `EP_JIT_CACHE_DIR` | `/root/.deep_ep` | 做 A/B 时**每个变体必须用不同目录**（见 §7） |
| `EP_DISABLE_GIN` | `0` | 设 1 关掉 GIN，用来隔离问题 |

镜像里还烧了三个只读的出处戳（`build_image.sh` 写入、`run_test_ep.sh` 读来做启动前检查）：
`EP_BUILD_ARCH`（`TORCH_CUDA_ARCH_LIST`）、`EP_BUILD_CUDA`、`EP_EFA_INSTALLER`。
和 host 的 `compute_cap` 不符时 launcher **直接拒绝启动**（`ALLOW_ARCH_MISMATCH=1` 可越过）——
Hopper 的 cubin 在 Blackwell 上失败的位置离原因很远。

launcher / driver 侧的旋钮（不是容器内的环境变量）：

| 变量 | 用的值 | 说明 |
|---|---|---|
| `TOKENS` / `NUM_SMS` | `8192` / `128`；`12`、`24` | prefill 和 decode 只差 `TOKENS`。`NUM_SMS` **别用 0** —— auto 路径读**一块**网卡的速率（b300 上是真值的一半） |
| `IGNORE_LOCAL` | `1` | 传 `--ignore-local-traffic`。不带的话 SO 分母含同机流量，会超过单卡线速，那就不是线速率 |
| `TEST_FIRST_ONLY` | `1` | `--test-first-only` = FP8 dispatch @ `expert_alignment=128`（`enumerate_ep_modes()` 的第一项）。设 `0` 是跑整个模式笛卡尔积，几小时 |
| `NODES` | `"<leader> <worker>"` | `run_campaign.sh` 的节点列表（ssh 别名或 IP），顺序即 `NODE_RANK` |
| `IMAGE_BASE` / `IMAGE_PRS` | `:<arch>` / `:<arch>-5a594a5` | 两条臂的镜像 tag。`IMAGE_PRS` 不存在时 `prs` 那几个 cell **整条跳过**并打提示 |
| `GIN_ENV` | 那对 GIN 变量（默认） | 置空即测 type-2 对照组；tag 里会相应写 `_type2` 而不是 `_gin5` |
| `REPS` / `CELLS` / `PORT_BASE` / `LOGDIR` | `3` / arch 默认表 / `8500` / `~/epruns` | rep 在 cell 之间轮换；`CELLS` 一行一个 cell（`arm\|image\|tokens\|sms\|knobtag\|额外 env`） |

`EP_HYBRID_KERNEL` 是这个 fork 的核心：Amazon 的 unordered kernel 用 in-band token header + counting signal 同步，不依赖投递顺序；上游 ordered kernel 依赖有序 RDMA write，在 EFA 的 SRD 乱序投递下不成立。选择发生在 JIT 生成期、变体名进生成源码的文件名，所以两种模式的 JIT 缓存天然互不干扰。（"EFA" 这个词在这个 fork 的代码和 README 里一次都没出现过 —— 对外只叫 "unordered delivery"。）

**1.50.0 之后不再需要的东西**（一个都没设，`Libfabric_GDAKI (v14)` 照样加载）：`FI_EFA_USE_HW_CNTR=1`、`OFI_NCCL_GIN_STRONG_SIGNAL=1`、`NCCL_RMA_DISABLE=1`、NCCL `sym_kernels.cc` 的 GIN waiver 补丁、源码编 aws-ofi-nccl（`--enable-gdaki`）、`insmod` 自编 CE-capable `efa.ko`、`LD_PRELOAD` 顶掉 torch 的 NCCL。

### 6.1 必须设的那两个环境变量

1.50.0 的 `libnccl-net-ofi.so` 注册了**两个** GIN plugin：一个 Libfabric 代理式
（**type 2**）、一个 `Libfabric_GDAKI`（**type 5**）。两个都会加载，
`Loaded gin plugin Libfabric_GDAKI (v14)` 两种情况下都打 —— 但**默认 NCCL 选 type 2**。
所以这两个变量不影响 GDAKI 能不能**加载**，影响的是加载之后**用不用它**：

```
NCCL_GIN_TYPE=5  NCCL_SYM_GIN_KERNELS_ENABLE=0
```

**实测收益**（4 × p5en，16/32 rank，12 SM，`--test-first-only`，未开 `EP_BUFFER_DEBUG`，
全 rank 均值；完整数据见
[`results/p5en_2n4n_20260825/summary.txt`](../results/p5en_2n4n_20260825/summary.txt)）：

| | 默认（type 2） | 加这一对（type 5） | |
|---|---|---|---|
| 2 节点 8192 tok dispatch | 1644.0 µs / 74.0 GB/s (SO) | **1502.9 µs / 81.2** | −8.6% |
| 2 节点 128 tok dispatch | 365.1 µs | **169.4 µs** | **2.16×** |
| 4 节点 8192 tok dispatch | 4315.0 µs / 51.5 | **3955.3 µs / 56.0** | −8.3% |
| 4 节点 128 tok dispatch | 1003.2 µs | **184.3 µs** | **5.44×** |

**差距随规模变大**：type-2 的 decode dispatch 从 2 节点到 4 节点是 365.1 → 1003.2 µs
（2.75×），type 5 只是 169.4 → 184.3 µs（1.09×）。type-2 的 4 节点 decode 还很不稳定，
五轮散在 833.1 / 986.7 / 1045.4 / 1067.5 / 1083.2 µs；type 5 的 32 个 rank 全在
183.4–185.6 µs 之内。

**必须成对设，单独设 `NCCL_GIN_TYPE=5` 会直接崩**，日志把原因写清楚了：

```
gin/gin_host.cc:229 (ncclGinValidateSignalRequest)
  NCCL WARN GIN strong signals are required, but the GIN plugin does not support them.
gin/gin_host.cc:440 (ncclGinDevCommSetup)
  NCCL WARN GIN: DevComm setup failed on all available backends
→ RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:217): 3
```

symmetric-memory GIN kernel（`NCCL_SYM_GIN_KERNELS_ENABLE`，默认 1）要求 strong signal，
GDAKI 没实现。type 2 还在候选列表里时 NCCL 静默回退到它 —— 这既是默认 env 能跑起来的原因，
也是没人注意到这件事的原因；把 type 2 排除掉又不放宽 strong signal 要求，就一个后端都不剩了。

**这一对是最小且充分的组合。** 逐个单变量测过（2 节点 12 SM 128 tok，全 rank dispatch）：
默认 359.2 / 371.1 µs、`FI_EFA_USE_HW_CNTR=1` 375.8 µs、`NCCL_RMA_DISABLE=1` 359.5 µs、
`NCCL_SYM_GIN_KERNELS_ENABLE=0` 单独 359.4 µs、`OFI_NCCL_GIN_STRONG_SIGNAL=1`
**750.4 µs**（16 个 rank 散在 371.0–1130.0 µs，主动变坏）、这一对 169.4 µs、完整 5 个 route-B 变量
170.0 µs（不比这一对好）。

**怎么确认真的跑在 type 5 上**（`NCCL_DEBUG=INFO`）：

```
只有 type 5 打：  GIN/Plugin: Skipping plugin Libfabric index 3 type 2:
                    NCCL_GIN_TYPE=5 requested
                  devCommCreate: creating 11 contexts / creating 4 contexts
两侧都打：        GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)
                  NET/OFI Selected provider is efa, fabric is efa-direct
                  [Proxy Progress] Device N CPU core M      <-- 两侧各 16 条
```

唯一可靠的判据是 `Skipping plugin ... type 2` 这一行。`Loaded gin plugin Libfabric_GDAKI`
只说明插件注册了、**不说明它被选中**；`[Proxy Progress]` 也**不是**判据 —— NCCL 给普通集合
通信也建代理线程，两侧各 16 条。两份 grep 结果在
[`results/p5en_2n4n_20260825/logs/`](../results/p5en_2n4n_20260825/logs/) 里
（`gin_plugin_selection_default.grep.txt` / `_gin5symgin0.grep.txt`）。

传这一对的办法：`run_test_ep.sh` 的 `EXTRA_ENV="NAME=VALUE …"` 钩子。故意**不**设成默认值，
这样默认那一侧仍然是个可测的对照组。

---

## 7. 测性能的四条硬规矩

1. **每个变体独立 `EP_JIT_CACHE_DIR`** —— 而且原因和直觉相反。cache key 是
   `name$$compiler$$flags$$code`（`csrc/jit/compiler.hpp:123`），`compiler` 只是 `"NVCC13.1"`，
   `code` 是**生成出来的 wrapper**；`deep_ep/include/deep_ep/impls/` 下的实现头文件只是被
   `#include`，**内容根本不进 key**。所以两个只差一个头文件补丁的镜像 key 完全相同，共享 cache
   目录会把没打补丁的 cubin 喂给打了补丁的镜像 —— 补丁测出来是个完美的 no-op。`flags` **在** key
   里，所以同一个镜像内用 env 开关（`EP_NUM_SUB_PARTS` / `EP_MIN_TOKENS_PER_PART` …）做 A/B 是
   安全的，不需要分目录。"A/B 完全没差别"最常见的原因就是这个。
2. **交错跑 rep**（`A B A B ...`），绝不能先跑完所有 A 再跑 B —— 否则热漂移和集群漂移全算到其中一个变体头上。
3. **`rc=0` 不是健康检查。** 上一轮崩掉的 rank 可能活着、每张 GPU 占着 ~48 GB。下一轮如果泄漏不大、GDAKI init 还能过，会跑完、`rc=0`、输出完整，然后**延迟虚高约 2×**，日志里没有任何提示（4 节点上见过 combine 连续四轮 7.7→12.5→16.5→19.6 ms 而每轮都报成功，显存同步爬 0→8.9→29.7→43 GB）。所以每轮之间必须断言 `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 全是 0 MiB，并且**每轮换一个 `MASTER_PORT`**（`TIME_WAIT` 表现为 rendezvous 卡死）。用 docker 跑省事 —— `docker rm -f` 会带走整棵进程树。
4. **报数字要报全 rank，并且带上口径。** rank 之间是系统性不同的，而且差异往往**按节点分层**（§5.3 里 combine 就是一整台机器慢 14%）。单个 rank、单台机器的区间都不是这一轮的数字。同时永远把时间（µs）和带宽（GB/s）一起报，并说明分母 —— 只报 GB/s 曾经把结论弄反过。

---

## 8. 还没测的（待办实验）

按「读者最想要 / 代价最小」排。全都不需要重新 build 镜像。

1. **DeepEP V1 跑一次 `--num-tokens 8192` + FP8 dispatch**（1 次 run，用现成的 `deepep-v1-efa:dev`）。仓库顶层的吞吐表里 V1 是 4096/BF16，DeepEP V2 是 8192/FP8，**两个方向都不可比**：V1 从来没跑 8192，GDAKI 也从来没跑 4096。V1 自己已经有 4096 的 FP8 dispatch 数（p5 48.17 / p5en 54.98 GB/s），所以缺的那一格在 V1 那边，不用重跑本目录的 campaign。
2. **4 节点的 `--num-sms` 轴上界没封**（2 次 run：4N/16 SM、4N/32 SM）。2 节点已经封死，而且最优点在**内部**（24 SM 的 dispatch+redComb 4898.3 µs，32 SM 反而 5062.9）。4 节点只有 6/12/24，曲线还在单调变好（13549.4 → 11898.5 → 11701.0 µs），所以「4 节点也用 24 SM」这条建议站在一条没封的轴上。
3. **任何 scale 都没跑过 `--ignore-local-traffic`**（2 次 run：2N、4N）。launcher 里钩子是现成的（`IGNORE_LOCAL=1`，`run_test_ep.sh:124`）。这里所有 GB/s 都是 **SO** 分母，会把同机目标算进去；`wire% = SO × (N−1)/N ÷ 50` 只在**2 节点**上对着实测核过一次，4 节点的 ×0.75 至今是纯算术，没人验过。
4. **没有单机基线**（1 次 run）。本目录全部是 ≥ 2 节点，所以 DeepEP 自身的 kernel 开销和跨机 EFA 开销从来没分开过。
5. **combine 慢的那台机器到底跟机器还是跟角色**（1 次 run，把 `NODE_RANK` 对调）。combine / reduced combine 按机器分层（2 节点 13~17%），慢的那台在一批连续 rep 内固定、跨批会翻转；把 leader 角色换到另一台就能判定。
6. **4 节点那几个臂太薄**（给 4N 的行补第 3 个 rep）。这里多数 4 节点臂只有 1~2 个 rep，2 节点是 3 个；rep 间离散度实测 ≤ 0.31%，但那主要是 2 节点上测的。
7. **BF16 dispatch，任何 scale 都没测过**（要改 harness，不是加跑一次）。`run_test_ep.sh` 传的是 `--test-first-only`，而 `enumerate_ep_modes()`（`test_ep.py:33-41`）的第一项是 `use_fp8_dispatch=1, expert_alignment=128`。所以**本文每一个数字都是 FP8 dispatch @ alignment 128**；`test_ep.py` 没有单独选 BF16 的开关，`TEST_FIRST_ONLY=0` 是把整个模式笛卡尔积跑一遍（几小时）。目前唯一一个 BF16 读数来自 b300 那个 kit，显示 BF16 dispatch 是**更慢**的那一侧，所以这里的数字没有因为这个选择被美化 —— 但它仍然是一根没测的轴。
8. **这个镜像在 `sm_103` 上的正式 campaign**（2 次 run：prefill + decode，3 个 rep，带 SM 轴）。b300 目前只有 §3.4 那个抽查，而成体系的 b300 数字来自另一个镜像，所以本文没有任何一行是可以跨架构比的。build 和 launcher 现在都已经支持 b300，剩下的就是每台机器 `./build_image.sh`（想同时跑 `prs` 臂就跑两次）然后 `NODES="<leader> <worker>" ./run_campaign.sh`（§4.5）—— `sm103` 的默认 cell 表里留了一个 12 SM 的 prefill cell，专门用来和 §3.4 的抽查对齐。

---

## 附录 A：为什么必须 1.50.0（判据要挑对）

1.49.0 和 1.50.0 的差别**不在 `efa-direct` 这一层**。1.49.0 已经报 16 个 `efa-direct` domain，它的 libnccl-ofi 1.20.0 也已经导出 `ncclGinPlugin_v11` / `v13`（12 个 Gin 符号）、二进制里就有 GDAKI 字符串。**所以拿"有没有 efa-direct"或"有没有 Gin 符号"判断版本，会得出错误结论。**

真正卡住 1.49.0 的是最底下两层：

- `EFA_QUERY_DEVICE_CAPS_COMP_CNTR` 在 efa 3.1.0 的 `efa-abi.h` 里**不存在**（diff 过 3.0.0 / 3.3.0：`comp_mask` 改名 `supported_caps`，`COMP_CNTR = 1<<8` 是新加的）；
- rdma-core 63.0 的 libibverbs 里 **0 个 `comp_cntr` 符号**，64.0 才有 20 个。

`ncclGinPlugin_v14` 才是 1.21.1 新加的那一版。完全没有 Gin 符号的是 1.18.0（installer 1.47.0，很多 DLAMI 的默认版本）。

| installer | efa 驱动 | libfabric | rdma-core | ofi-nccl | GinPlugin 符号 |
|---|---|---|---|---|---|
| 1.47.0 | 3.0.0 | 2.4.0amzn1.0 | 61.0 | 1.18.0 | **无** |
| 1.48.0 | 3.0.0 | 2.4.0amzn1.0 | 61.0 | 1.19.0 | — |
| 1.49.0 | 3.1.0 | 2.4.0amzn5.0 | 63.0 | 1.20.0 | v11 / v13 |
| **1.50.0** | **3.3.0** | **2.6.0amzn1.0** | **64.0amzn0** | **1.21.1** | **v11 / v13 / v14** |

> 只有 1.50.0 那一行是在这两台机器上现场查的。机器已升到 1.50.0，旧包不在机上了，1.47/1.48/1.49 三行属于历史记录（与我们在 1.47.0 出厂栈和 1.49.0 上的独立观测一致：出厂 1.47.0 跑不了 GDAKI、1.49.0 只有 v11+v13），**本次未重新验证**。要拿它做采购/升级决策请在目标机上自己查一遍。

### 容器内版本自查

```bash
fi_info --version | head -3                       # 2.6.0amzn1.0
dpkg -l | grep -E "libfabric|nccl-ofi|ibverbs"    # ofi-nccl 1.21.1 / rdma-core 64.0amzn0
nm -D --defined-only /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep GinPlugin   # 必须有 v14
nm -D --defined-only /lib/x86_64-linux-gnu/libibverbs.so.1 | grep -c comp_cntr      # 20
ibv_devinfo -l | grep -c rdmap ; fi_info | grep -c "fabric: efa-direct"             # 16 / 16
python3 -c "import deep_ep, torch; print(deep_ep.__version__, torch.__version__)"
printenv EP_BUILD_ARCH EP_BUILD_CUDA EP_EFA_INSTALLER                               # 这个镜像是给哪档编的
```

`grep GinPlugin` 里没有 v14 = 插件是 1.20.0 或更老，后面全白跑。`comp_cntr` 为 0 = rdma-core 是 63.0。

**数 `rdmap` 而不是数总行数**：b300 上 `ibv_devinfo -l` 一共 18 行（多两个非 EFA 的 `ibp*`，
§1 第 4 条），照 "= 16" 去核会误判成坏节点。p5en 上两种数法结果相同。

### 一锤定音的 CE 探针

GDAKI 的成败最终取决于 `ibv_create_comp_cntr` 这一个 verb。**在容器里**跑最有意义 —— 它同时验证 host 内核模块和容器里那份 rdma-core，正好是依赖链最底下两层。健康节点上 16 个 `rdmap*` 全部 `CE OK`；**b300 上另外那两个 `ibp*` 是 `CE FAIL` / errno 95，这是对的**（它们不是 EFA），别当成故障 —— 要处理的是让 NCCL 别选中它们（`NCCL_IB_HCA=rdmap`，§3.4）。驱动状态是**每节点**的：同一批实例里刚重启的机器可能回到旧模块，某天忽然挂了先跑这个。

```c
/* ce_probe.c — gcc -o ce_probe ce_probe.c -libverbs */
#include <stdio.h>
#include <errno.h>
#include <infiniband/verbs.h>
int main(void) {
	int n = 0;
	struct ibv_device **d = ibv_get_device_list(&n);
	for (int i = 0; i < n; i++) {
		struct ibv_context *c = ibv_open_device(d[i]);
		if (!c) continue;
		struct ibv_comp_cntr_init_attr a = {0};
		errno = 0;
		struct ibv_comp_cntr *cc = ibv_create_comp_cntr(c, &a);
		printf("%-14s %s (errno=%d)\n", ibv_get_device_name(d[i]),
		       cc ? "CE OK" : "CE FAIL", errno);
		if (cc) ibv_destroy_comp_cntr(cc);
		ibv_close_device(c);
	}
	return 0;
}
```

---

## 附录 B：镜像里有三个 NCCL 版本

会让人在第一步就误判"`--no-deps` 那步没生效"：

| 来源 | 版本 | 谁会用到它 |
|---|---|---|
| pip `nvidia-nccl-cu13` → `dist-packages/nvidia/nccl/lib/libnccl.so.2` | **2.31.2** | **DeepEP 编译期和运行期用的就是这个** |
| apt `libnccl2`（EFA installer NGC 分支拖进来的），`/usr/include/nccl.h` 也是它 | **2.28.3** | `ldconfig` 把 `libnccl.so.2` 解析到这个 |
| torch 自己编译期的 header | **2.29.7** | `torch.cuda.nccl.version()` 返回 `(2, 29, 7)` |

运行期 DeepEP 不受 2.28.3 影响：`setup.py:95` 用 `find_pkgs.find_nccl_root()` 拿 Python 环境里的路径，再 `-I {root}/include` + `-Wl,-rpath,{root}/lib`（`setup.py:122-124`），所以**头文件和运行期加载的都是 pip 那份 2.31.2**（用 `/proc/self/maps` 确认过活进程映射的就是它）。

**但链接期不是。** link flags 里只有 `-l:libnccl.so.2` 和那个 rpath，**没有 `-L`**，所以链接器是靠 `/lib/x86_64-linux-gnu/libnccl.so`（apt 的 2.28.3）解析这个 `-l` 的。soname 一致所以链得过、跑起来也对，但这是隐式依赖 —— 一删 apt 包构建就断（§3.3 第 3 条，实测复现）。

```bash
ldconfig -p | grep libnccl.so      # -> /lib/x86_64-linux-gnu/libnccl.so.2，也就是 2.28.3
grep -E "NCCL_(MAJOR|MINOR|PATCH)" /usr/include/nccl.h        # 2 / 28 / 3
find / -name 'libnccl.so.2' -not -path '/proc/*' 2>/dev/null  # 两份，认清哪份是 pip 的
```

**2.28.3 < 2.30.4，没有 GIN。** 运行期没事只是因为走 rpath。收紧办法见 §3.3 第 2、3 条 —— 删 apt 包**必须**同时补 `-L`，两个改动是一对。

---

## 附录 C：故障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `ZeroDivisionError` in `get_theoretical_num_sms` | **只在旧 pin `ec623f3` 上**：`--num-sms 0` 走 `ibstat` 自动探测，EFA 上必失败（§4.2）。Dockerfile 现在钉的 `8e7b42e` 先读 sysfs，不会崩 | 显式给 `--num-sms 12`（§4.2） |
| `Failed to get RDMA connection speed:` | 同上，`ibstat` 看不到 EFA 设备 | 单机无害；多机必须给 `--num-sms` |
| `num_sms` 自动探测出来偏小（b300 上 `rdma_gbs=50.0`） | `get_rdma_gbs` 只返回**一块**网卡的速率；b300 每 GPU 两张 EFA，真值 100 GB/s | 显式给 `--num-sms`（§3.4） |
| `only 2 GIN GDAKI NICs have been created` + `NCCL exception (nccl.cu:185): 5` | 混合 ibverbs 设备列表（b300 有 2 个非 EFA 的 `ibp*`）让 NCCL 少建了 GDAKI NIC | `NCCL_IB_HCA=rdmap`；`run_test_ep.sh` 已自动注入（§3.4） |
| `Arguments mismatch for instruction 'mov'` → `ptxas fatal` → `compiler.hpp:239` | `sm_103` 命中 `ptx.cuh` 的 `__CUDA_ARCH__ >= 1000` 分支（`.v4.s64`），CUDA 13.0.2 的 ptxas 不认。第一次 dispatch 才炸 | 重建镜像：`--build-arg CUDA_VERSION=13.3.1`（§3.4）；没有宏能绕 |
| 改了 `--num-sms` 后整轮**无输出挂死** | 不是 QP 数的问题：`ec623f3` 上 `#QPs` 恒为 11、与 SM 无关，本仓库 6/12/16/24/32 SM 全部跑通（§4.2）。先查上一轮有没有漏进程 —— rc=0 也不代表干净 | `nvidia-smi` 确认显存全 0 MiB，每轮换 `MASTER_PORT`（§7） |
| `run_test_ep.sh` 报 `REFUSING TO START`（arch / CUDA 不匹配） | 镜像的 `EP_BUILD_ARCH` 和 host `compute_cap` 不符，或 `sm_10x` 上配了 < 13.3 的 CUDA base | 按 §3 表格重建（`./build_image.sh` 自己 probe）。确实要跨档跑再 `ALLOW_ARCH_MISMATCH=1` |
| `ibv_devinfo -l` 数出 18 个设备（b300） | **正常**：16 个 EFA `rdmap*` + 2 个非 EFA `ibp*`（§1 第 4 条） | 自检改成 `grep -c rdmap`；运行时靠 `NCCL_IB_HCA=rdmap` 排掉 |
| `ce_probe` 对 `ibp198s0f0` / `ibp199s0f0` 报 CE FAIL / errno 95 | **正常**，它们不是 EFA 设备 | 只看 16 个 `rdmap*` 是否全 `CE OK` |
| b300 的"线速占比"算出来约 200% | 分母用了 p5en 的 50 GB/s；b300 每 GPU 两张 EFA = 100 GB/s | 换分母（§5 口径） |
| `ibv_devinfo`: No IB devices found，但 `lsmod` 有 efa | 启动时 ENI 没开 EFA | 重建实例，`InterfaceType=efa` |
| `fi_info -p efa-direct` → `-61 (No data available)` | **正常现象**，`-p` 匹配 provider（`efa`）不是 fabric | 用 `fi_info \| grep fabric` |
| `fi_info: command not found` | 不在默认 PATH | `export PATH=/opt/amazon/efa/bin:$PATH` |
| `grep GinPlugin` 里没有 **v14** | libnccl-ofi 是 1.20.0 或更老 | 升到 installer 1.50.0（1.20.0 有 v11/v13，别被误导） |
| `libibverbs` 里 0 个 `comp_cntr` | rdma-core 是 63.0 | 同上 |
| `cntr_open_ext failed: Operation not supported` at GDAKI `createContext` | CE 三要素缺一（内核模块 / libfabric / rdma-core ABI） | 跑附录 A 的 ce_probe；再 `FI_LOG_LEVEL=warn` 看 provider-init 警告 |
| `cntr_open_ext failed: Cannot allocate memory`（注意不是 not supported） | CE **counter 耗尽**，不是能力问题 —— 上一轮泄漏的 rank 还占着 | 清干净残留进程（§7 规矩 3） |
| `NCCL GIN is unavailable` assert | GIN 没起来 | 先过附录 A 的自查；日志里找 `Loaded gin plugin` |
| `Cannot get gin type: ... net device type (5) is not a gin type` | 插件不支持 GDAKI，或旧 libnccl 赢了加载顺序 | 确认 `NCCL_NET_PLUGIN=ofi` 且 `dpkg -l` 是 1.21.1 |
| installer 在 docker build 里 `Unable to locate package tcl` | apt 缓存被提前清掉 | 同一层里先 `apt-get update`（§3.2 坑 1） |
| `ModuleNotFoundError: deep_ep._C` | `WORKDIR` 是源码目录 | 换到 `/workspace`（§3.2 坑 8） |
| `AssertionError: ... ninja is required` | 缺 ninja | `pip install ninja` |
| `ModuleNotFoundError: numpy` / `pynvml` | 缺运行时依赖 | `pip install numpy nvidia-ml-py` |
| `third-party/fmt` 头文件找不到 | submodule 没拉 | `--recurse-submodules` |
| DeepEP 断言 NCCL 编译期/运行期版本不一致 | NCCL < 2.31 | 用 2.31+ |
| `Cannot find package: nvshmem` | 没装 NVSHMEM | `pip install nvidia-nvshmem-cu13`，v2 也需要 |
| `ibv_create_ah failed with EINVAL ... different availability zone` | 跨 AZ | 同 AZ + cluster placement group |
| RDMA 内存注册失败 | memlock 限制 | `docker run --ulimit memlock=-1` |
| 延迟虚高 ~2× 但 `rc=0`、输出完整 | 上一轮泄漏的 rank 在抢显存 | `rc` 查不出来，必须查 `nvidia-smi`（§7 规矩 3） |
| SO GB/s 每 rank 超过 50 | 不是线速 —— 默认口径把机内流量也算进去了 | 加 `--ignore-local-traffic`（§5 口径） |
| GDAKI 昨天好今天坏 | 某个节点重启后回到旧驱动 | 跑附录 A 的 ce_probe |

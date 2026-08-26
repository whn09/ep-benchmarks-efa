# DeepEP v2 on AWS EFA — 从零跑到可复现的数字

这份文档是一条**从空实例走到一张表**的路径：装 host → 建镜像 → 跑通一次 → 跑完整个
campaign → 验收 → 生成和 §9 一样的表。按顺序走，每一步都给了"应该看到什么"，看不到就去
§12 的故障表对。全程只用已发布的包：**不编 NCCL、不编 aws-ofi-nccl、不换内核模块。**

**支持两种机型，同一份 Dockerfile、同一套脚本：**

| 机型 | GPU | arch | 每 GPU EFA | 每 GPU 线速 | 本文数字 |
|---|---|---|---|---|---|
| `p5en.48xlarge` | 8×H200 | `sm_90` | 1 张（共 16 张） | 50 GB/s | §9 全部（2 节点和 4 节点） |
| `p6-b300.48xlarge` | 8×B300 | `sm_103` | 2 张（共 16 张） | 100 GB/s | §9.7：抽查通了，campaign 待跑 |

机型差别只有三处，脚本会自己处理，但你需要知道它们存在（§5.3 讲机制）：CUDA base、
`TORCH_CUDA_ARCH_LIST`、运行时 `NCCL_IB_HCA=rdmap`。B200 / `sm_100` 的参数已就位，没跑过。

**你会得到什么**（p5en × 2 节点，16 rank，24 SM，`--test-first-only`，全 rank 均值）：

| | 时间 | SO GB/s | 线速占比 |
|---|---|---|---|
| prefill dispatch（8192 tok） | 1535.7 µs | 79–80 | 79.6% |
| prefill reduced combine | 3362.6 µs | 64–79 | — |
| decode dispatch（128 tok） | 147.3 µs | — | — |

**一件必须先知道的事**：容器里必须有
`NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0`。这两个变量决定 NCCL 用不用 GDAKI ——
没有的话 prefill 慢 ~9%、decode 慢 2.2–5.4×，而日志看起来一切正常（§10.1 给了收益数据和
判据）。`run_test_ep.sh` 和 `run_campaign.sh` **默认就传**（`GIN_ENV`，置空即 type-2 对照臂），
所以按本文走不会漏；但**裸 `docker run` 必须自己加**。

**上游还有一套 kit，两者分工不同**：
[`awslabs/awsome-distributed-ai` → `micro-benchmarks/expert-parallelism/deepep-v2-benchmark`](https://github.com/awslabs/awsome-distributed-ai/tree/main/micro-benchmarks/expert-parallelism/deepep-v2-benchmark)
是**安装**参考：一个 `setup_deepep_gin.sh` 把 DeepEP v2 装进*你已有的*容器（比如 vLLM 镜像）、
一份从裸 CUDA 基础镜像起的参考 Dockerfile、一对 Slurm + enroot/pyxis 的验证 sbatch。要往现有
推理镜像里塞 DeepEP v2、或者你的集群是 Slurm，用它。本目录是**测量**用的：pin 死 commit、
一份镜像一个 arch（含 b300 的 `sm_103`）、多机 campaign 驱动、验收脚本、裸 EC2 ssh 起进程、
每个 cell 的日志命名带齐所有变量。要复现 §9 的表，用本目录。两边的依赖结论是独立得出的且
一致：installer ≥ 1.50 装容器、host efa ≥ 3.3.0 + `gdrdrv`、NCCL ≥ 2.31、镜像里只留一份 NCCL、
以及 `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0` 这一对（他们的 sbatch 也是这两个）。

---

## 1. 依赖链：为什么必须是 EFA installer 1.50.0

**内核态装 host，用户态装容器，两边都要。**

| 层 | 需要的版本 | 判据 | 装在哪 |
|---|---|---|---|
| DeepEP v2 unordered kernels | `amazon-contributing/DeepEP` | — | 容器 |
| NCCL GIN | `nvidia-nccl-cu13` ≥ 2.30.4（用 2.31.2） | — | 容器 |
| aws-ofi-nccl | **1.21.1** | `ncclGinPlugin_v14` 符号 | 容器 |
| libfabric | **2.6.0amzn1.0** | 16 个 `fabric: efa-direct` | 容器 |
| rdma-core | **64.0amzn0** | libibverbs 里 20 个 `comp_cntr` 符号 | 容器 |
| **efa 内核驱动** | **3.3.0** | `EFA_QUERY_DEVICE_CAPS_COMP_CNTR` (1<<8) | **host** |
| gdrcopy | ≥ 2.5 | `/dev/gdrdrv` | kmod 在 host / lib 在容器 |

加粗的四层**全部由 installer 1.50.0 一次给齐**，这就是版本门槛的全部原因。各版本对照：

| installer | efa 驱动 | libfabric | rdma-core | ofi-nccl | GinPlugin 符号 |
|---|---|---|---|---|---|
| 1.47.0 | 3.0.0 | 2.4.0amzn1.0 | 61.0 | 1.18.0 | **无** |
| 1.48.0 | 3.0.0 | 2.4.0amzn1.0 | 61.0 | 1.19.0 | — |
| 1.49.0 | 3.1.0 | 2.4.0amzn5.0 | 63.0 | 1.20.0 | v11 / v13 |
| **1.50.0** | **3.3.0** | **2.6.0amzn1.0** | **64.0amzn0** | **1.21.1** | **v11 / v13 / v14** |

> 只有 1.50.0 那一行是现场查的；1.47/1.48/1.49 三行是历史记录（与我们在出厂 1.47.0 栈和
> 1.49.0 上的独立观测一致）。要拿它做采购/升级决策请在目标机上自己查。

**判据要挑对，否则会把不够的版本判成够。** 1.49.0 已经有 16 个 `efa-direct` domain，
它的 libnccl-ofi 1.20.0 也已经导出 `ncclGinPlugin_v11` / `v13`（12 个 Gin 符号）、二进制里
就有 GDAKI 字符串。所以"有没有 efa-direct"和"有没有 Gin 符号"都**不是**判据。真判据两个：
`nm` 里有没有 **v14**，以及 `ibv_create_comp_cntr` 这个 verb 能不能成功（§4.4 的探针）。

**tarball 的版本号也不是判据**，所以 Dockerfile 里那一层装完 installer 立刻
`nm -D /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep -qw ncclGinPlugin_v14`，不过则
构建当场失败：存在过 ChangeLog 同样以 `## [1.50.0]` 开头、里面却是 ofi-nccl **1.20.0**（只有
v11/v13）的 `aws-efa-installer-1.50.0.tar.gz`。GA 那份是 1.21.1、带 v14。少了 v14 的镜像照样
建得出来、照样跑得通，只是静默走 type-2 CPU proxy —— 正好是本目录要对比的那条对照臂。

**p6-b300 出厂带的是 1.47.0**（efa.ko 3.0.0，`GinPlugin` 符号一个都没有），所以 b300 上
第 3 步不是可选的。顺序也不能颠倒：GDAKI 起不来的时候，§5.3 那两个 b300 专有失败根本还
没轮到现形。

GIN = GPU-Initiated Networking（NCCL Device API）。EFA 上的实现叫 `Libfabric_GDAKI`，
走 `efa-direct` fabric 的 GDA ops，用硬件 completion counter（CE）承载 counting signal。

---

## 2. 准备实例（四条，缺一条都白干）

1. **EFA 必须在创建实例时就打开**（`InterfaceType=efa`）—— 事后打不开，只能重建。
   用 `NetworkCardIndex=0..15` 逐个指定；两种机型都是 **16 张 EFA**，区别只在每 GPU 摊几张
   （p5en 1 张、b300 2 张）。
   若 `lsmod | grep efa` 有模块，但 `ibv_devinfo` 报 `No IB devices found`、`/dev/infiniband`
   不存在 —— 就是这一条没做到。
2. **安全组自引用放通全部流量**（入站 + 出站，Source/Destination = 该安全组自身）。
   EFA 不走 TCP 端口，只放几个端口是不够的。
3. **同 AZ + 同一个 cluster placement group。** 跨 AZ 会在建立连接时报
   `ibv_create_ah failed with EINVAL ... Remote GID is in a different availability zone`。
4. **设备名不是 `mlx5_*`，也不连号**，而且两种机型 `ibv_devinfo -l` 的**总数不同**：
   - p5en：**16** 个，全是 EFA ——
     `rdmap85s0 86s0 87s0 88s0 / 110s0 111s0 112s0 113s0 / 135s0 136s0 137s0 138s0 / 160s0 161s0 162s0 163s0`
   - p6-b300：**18** 个 —— 16 个 `rdmap*`（EFA）**加** `ibp198s0f0` / `ibp199s0f0`，
     后两个**不是 EFA**。多出来这两个会让 NCCL 少建 GDAKI NIC（§5.3）。

所以自检一律**数 `rdmap` 而不是数总行数**：`ibv_devinfo -l | grep -c rdmap` 应为 16。
两种机型都适用，b300 上按"总数 = 16"去核会把健康机器判成坏的。

最省事的做法：capacity block + 官方 DLAMI + cluster placement group，让模板把 16 张
EFA ENI 配好。

---

## 3. Host 安装（每台机器做一次，需要重启）

### 3.1 装

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
tar xzf aws-efa-installer-1.50.0.tar.gz          # 约 650 MB（含所有发行版的 RPM/DEB）
head -6 aws-efa-installer/ChangeLog.md           # ## [1.50.0] - Aug 2026
#   - Upgrade to rdma-core 64.0amzn0 / efa driver 3.3.0
#   - Upgrade to Libfabric 2.6.0amzn1.0 / OFI NCCL Plugin 1.21.1

cd aws-efa-installer
sudo ./efa_installer.sh -y --no-verify           # --no-verify 跳过 GPG 校验
sudo reboot                                      # 见下：不是每次都必须，但最省事
```

**要不要重启，installer 自己会在最后一行说。** 它先 `modprobe -r efa` 卸掉旧模块
（`efa_installer.sh:415-425`）：卸成功就打 `Please logout/login to complete the
installation.`，只有卸失败才 `NEED_REBOOT=1` 并改打 `Please reboot`。空闲机器上通常卸得掉
（`/sys/module/efa/refcnt` = 0、`/sys/module/efa/holders/` 为空），所以这一步经常不需要重启。
反过来，只要还有进程占着 EFA（跑着的任务、没清掉的容器）就一定要重启。**不确定就重启，
两分钟的事**；真正的验收是下面 §3.2 那几行，不是"我重启过了"。

> 装之前先看一眼 `find /lib/modules/$(uname -r) -name 'ib_uverbs.*'`。**Ubuntu AMI 缺
> `ib_uverbs` 时 `-y` 不会替你升内核**：它打出一段 `apt-get upgrade` 命令然后直接
> **exit 1**（`efa_installer.sh:258-284`，只有交互模式才会 prompt 并自动 reboot）。
> 那种机器要先 `apt-get upgrade` + 重启，再回来跑 installer。DLAMI 不会遇到。

host 上**不要**加 `--skip-kmod` —— 要的就是 efa.ko 3.3.0。DLAMI 通常已自带 gdrcopy 2.5.x 和
`efa_nv_peermem`，一般只需要升 installer 本身。

> **想用还没 GA 的版本**：那种版本只在 dev 桶、只有浮动名字
> `aws-efa-installer-latest.tar.gz`（这个 key 不是固定版本，今天是 1.50.0 下周可能是
> 1.51.0）。先 `tar xzOf ... ChangeLog.md | head -4` 看清版本，再改名成你核过的版本号。
> dev 桶的包没签名，所以 `--no-verify` 是必须的。

### 3.2 验证 host（装完必做，重启过就重启后做）

```bash
export PATH=/opt/amazon/efa/bin:$PATH          # fi_info 默认不在 PATH 上

cat /sys/module/efa/version                    # 3.3.0g
lsmod | grep efa_nv_peermem                    # GDAKI 必需
modinfo gdrdrv | grep ^version ; ls -l /dev/gdrdrv
ibv_devinfo -l | grep -c rdmap                 # 16
fi_info | grep -c "fabric: efa-direct"         # 16
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1   # 9.0 或 10.3
```

最后一行决定第 4 步的 build 参数。`compute_cap` 是 `10.3` 就是 b300，`9.0` 是 p5en / H200。

再核一次 COMP_CNTR capability —— 它在内核 ABI 头里，是 1.50.0 的真判据。头文件在 DKMS
源码树的 `src/` 下面（不是 `/usr/src/efa-*/` 的第一层）：

```bash
grep COMP_CNTR /usr/src/efa-*/src/efa-abi.h                  # 期望 3 行，含 = 1 << 8

# 交叉验证装上去的模块本身。Ubuntu 的 DKMS 模块在 updates/dkms/ 下、zstd 压缩，
# 路径必须用 modinfo -n 问出来：写死路径的话 strings 报错走 stderr、grep -c 打印 0，
# 看起来像"驱动不支持"，实际是文件没找到。
zstd -dc "$(modinfo -n efa)" | strings | grep -c comp_cntr    # 期望 12
```

**别 grep `/usr/include/rdma/efa-abi.h`** —— 那是 host 上 distro rdma-core 带的用户态头，
在装了 1.50.0 的机器上它照样是 **0 个 COMP_CNTR**（DKMS 只换内核模块，不换 host 的用户态
包）。容器里自带 rdma-core 64.0，所以这不影响跑，但拿它当判据必错。

**`fi_info -p efa-direct` 永远失败，别拿它当判据。** 它在完全健康的节点上也固定返回
`-61 (No data available)`：`-p` 过滤的是 **provider** 名（就是 `efa`），只有 **fabric** 名叫
`efa-direct`。用 `fi_info | grep fabric`。

---

## 4. 构建镜像（每台机器一条命令）

### 4.1 建

每台机器上各 clone 一份、各建一次（镜像是本地的，没有 registry）：

```bash
git clone https://github.com/whn09/ep-benchmarks-efa.git ~/work/ep-benchmarks-efa
cd ~/work/ep-benchmarks-efa/deepep-v2-efa-official
./build_image.sh
```

**clone 到这个路径**：`~/work/ep-benchmarks-efa/deepep-v2-efa-official` 正是 §6 那个
campaign 驱动脚本的 `REPO_DIR` 默认值。放别处也能跑，但 §6 要显式带上
`REPO_DIR=<你的路径>`，否则它 ssh 进去 `cd` 一个不存在的目录。

`build_image.sh` 会 probe `compute_cap`、据此设 build 参数、把 arch 写进 tag
（`deepep-v2-efa-official:sm90` / `:sm103`），并在 build context 里没有 installer tarball 时
自己从 §3.1 那个 URL 下一份（tarball 有 650 MB，不进 git；每台机器各自从 S3 拉，比从
你笔记本推过去快得多）。显式写法：`./build_image.sh sm103 [DEEPEP_REF] [TAG]`。

> 只有在你本地改了这套脚本、想先试一把再提交时，才需要绕过 git 同步一次：
> `rsync -avz --exclude '*.tar.gz' deepep-v2-efa-official/ <node>:~/work/ep-benchmarks-efa/deepep-v2-efa-official/`。
> 别把它当常规路径 —— rsync 过去的目录 `git status` 是脏的，出了数字之后没法回答
> "这是哪个 commit 跑出来的"。

约 **21.4 GB**（压缩后 7.7 GB），冷启十几分钟。

**别直接 `docker build`** —— 下面两个参数不是可选的，而且其中一个是**晚炸**的：

| 目标 | build 参数（`build_image.sh` 自动设） |
|---|---|
| p5en / H200，`sm_90` | `TORCH_CUDA_ARCH_LIST=9.0  CUDA_VERSION=13.0.2` |
| p6-b300 / B300，`sm_103` | `TORCH_CUDA_ARCH_LIST=10.3  CUDA_VERSION=13.3.1` |
| B200，`sm_100` | `TORCH_CUDA_ARCH_LIST=10.0  CUDA_VERSION=13.3.1`（没跑过） |

**一个镜像只编一档。** Hopper 的 cubin 在 Blackwell 上跑不了，`sm_103` 也不是 `sm_100` 的
向下兼容目标，而 `9.0;10.3` 本来就要连 CUDA base 一起换（§5.3）。两个参数都会写进镜像的
`EP_BUILD_ARCH` / `EP_BUILD_CUDA`，启动前 `run_test_ep.sh` 会和 host 对一遍，不一致直接
拒绝跑（要跨档硬跑：`ALLOW_ARCH_MISMATCH=1`）。

### 4.2 第二条臂：两个待合的 PR

decode 的 part 几何有两个待合 PR（数据在 §9.6）。#1 是 #2 head 的祖先，所以**一个 ref 就是
"两个 PR 叠加"**这条臂：

```bash
gh pr view 2 --repo amazon-contributing/DeepEP --json headRefOid --jq .headRefOid
./build_image.sh sm103 5a594a5db2d1b7c45c60c82b0cf026e9440886a4
# -> deepep-v2-efa-official:sm103-5a594a5   （run_campaign.sh 默认找这个 tag）
```

PR head 会随 rebase 变，所以先用 `gh pr view` 取当前值再建。只关心 prefill 的话这条臂可以
跳过：两个 PR 在 prefill 上是噪声（§9.6）。

### 4.3 镜像里装了什么（进容器自查）

```bash
docker run --rm -it --gpus all --network host --ipc host --privileged --ulimit memlock=-1 \
  --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v "$PWD":/workspace \
  deepep-v2-efa-official:sm90 bash            # b300 上是 :sm103
```

- `-v "$PWD":/workspace` 从 §4.1 那个 clone 目录里跑，容器的 `WORKDIR` 正是 `/workspace`，
  §4.4 的 `ce_probe.c` 才能在容器里编。跑测试的 `run_test_ep.sh` **不挂**这个卷（它只需要
  镜像里已经装好的 DeepEP）。
- `--ulimit memlock=-1` **必须有**（容器里跳过了 `limits.conf`），否则 RDMA 注册内存失败。
- `--device /dev/infiniband` 透 EFA 设备，`--device /dev/gdrdrv` 给 gdrcopy。
- `--network host` 让多机直接互通；`--privileged` 跑通后可以收紧成 `--cap-add IPC_LOCK`。
- 挂 `/sys/class/infiniband:ro` 是为了让容器能读网卡速率（`EP_NIC_NAME` 那条路）。

```bash
fi_info --version | head -3                       # 2.6.0amzn1.0
dpkg -l | grep -E "libfabric|nccl-ofi|ibverbs"    # ofi-nccl 1.21.1 / rdma-core 64.0amzn0
nm -D --defined-only /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep GinPlugin   # 必须有 v14
nm -D --defined-only /lib/x86_64-linux-gnu/libibverbs.so.1 | grep -c comp_cntr      # 20
ibv_devinfo -l | grep -c rdmap                                                      # 16
python3 -c "import deep_ep, torch; print(deep_ep.__version__, torch.__version__)"
printenv EP_BUILD_ARCH EP_BUILD_CUDA EP_EFA_INSTALLER   # 这个镜像是给哪档编的
cat /opt/DeepEP/BUILD_REF                              # DeepEP 的实际 sha
```

`grep GinPlugin` 里没有 v14 = 插件是 1.20.0 或更老，后面全白跑。`comp_cntr` 为 0 =
rdma-core 是 63.0。镜像里有**三个** NCCL 版本，读日志前先看附录 A，否则会误判。

### 4.4 一锤定音的 CE 探针

GDAKI 的成败最终取决于 `ibv_create_comp_cntr` 这一个 verb。**在容器里**跑最有意义 ——
它同时验证 host 内核模块和容器里那份 rdma-core，正好是依赖链最底下两层。源码见附录 C：

```bash
gcc -o /tmp/ce_probe /workspace/ce_probe.c -libverbs && /tmp/ce_probe
```

健康节点上 16 个 `rdmap*` 全部 `CE OK`。**b300 上另外那两个 `ibp*` 是 `CE FAIL` /
errno 95，这是对的**（它们不是 EFA 设备），要做的是让 NCCL 别选中它们（§5.3）。
驱动状态是**每节点**的：同一批实例里刚重启的机器可能回到旧模块，某天忽然挂了先跑这个。

---

## 5. 第一次跑通（约 10 分钟）

`test_ep.py` **不是 torchrun 语义**：它自己 `torch.multiprocessing.spawn` 起 8 个 local rank。

- `WORLD_SIZE` = **节点数**（不是全局 rank 数）
- `RANK` = **节点序号**
- `--num-processes` = 每节点 local rank 数（8）

### 5.1 单机 8 rank：先把镜像和 JIT 排除掉

```bash
IMAGE=deepep-v2-efa-official:sm90 WORLD_SIZE=1 NUM_PROCESSES=8 \
TOKENS=128 NUM_SMS=24 MASTER_PORT=8499 NCCL_DEBUG=INFO \
  ./run_test_ep.sh 0 127.0.0.1 2>&1 | tee /tmp/smoke.node1.log
./verify_run.sh /tmp/smoke.node1.log
```

首次会 JIT 编译（1–3 min，发生在 `test_ep.py` 自己的 warmup 里，不在计时区）。这一步能把
"镜像编错了"和"网络起不来"分开 —— **b300 那两个失败都在这里现形**（§5.3）。

### 5.2 两机一个 cell

两台都跑，只有 node rank 不同（先起 worker，最后起 leader）：

```bash
# node 1（worker）：./run_test_ep.sh 1 <leader 私网IP>
# node 0（leader）：./run_test_ep.sh 0 <leader 私网IP>
IMAGE=deepep-v2-efa-official:sm90 WORLD_SIZE=2 NUM_PROCESSES=8 \
TOKENS=8192 NUM_SMS=24 MASTER_PORT=8500 IGNORE_LOCAL=1 TEST_FIRST_ONLY=1 \
  ./run_test_ep.sh <node_rank> <leader_ip>
```

GDAKI 那两个变量**不用写**：`run_test_ep.sh` 的 `GIN_ENV` 默认就是它们，开跑前会打印
`=== GIN: NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0 ===`。要跑 type-2 对照臂就
`GIN_ENV= ./run_test_ep.sh ...`（显式置空，脚本会警告一次）。

`TOKENS=8192` 是 prefill（看带宽），`TOKENS=128` 是 decode（看延迟）—— **两者只差这一个
参数**。`run_test_ep.sh` 干的事：检查 GPU 空闲、检查镜像 arch 和 host 一致、探测
`EP_NIC_NAME`、b300 上注入 `NCCL_IB_HCA=rdmap`、打印 `IMAGE=` 和 `DeepEP=<sha>`、然后
`docker run` 起 `test_ep.py`。它展开出来的命令等价于：

```bash
python3 -u /opt/DeepEP/tests/elastic/test_ep.py \
  --num-processes=8 --num-tokens=$TOKENS --hidden=7168 --num-topk=8 \
  --num-experts=256 --num-sms=24 --allow-hybrid-mode=1 \
  --prefer-overlap-with-compute=0 --test-first-only --ignore-local-traffic
```

**`--num-sms` 一律显式给**（`NUM_SMS` 别用 0）。自动探测走 `get_rdma_gbs()`，它只返回
**一块**网卡的速率：p5en 每 GPU 一张（50 GB/s，正好），b300 每 GPU 两张（真值 100，
只看到 50），于是 b300 上自动挡会系统性偏低。SM 怎么选见 §10.2。

### 5.3 b300 和 p5en 不同的两处

两处都不用改源码，脚本会自动处理；写在这里是因为它们**都在离原因很远的地方才炸**，
你迟早会在别的镜像、别的 launcher 上再见到。

**一、NCCL 只建了 2 个 GIN GDAKI NIC，rank 4 直接崩。**

```
transport/net_ib/gin.cc:262 (ncclGinIbGdakiGetProperties)
NCCL WARN NET/IB : Requested properties for GIN GDAKI NIC 4, only 2 GIN GDAKI NICs have been created
RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:185): 5
```

b300 的 ibverbs 设备列表是混合的（16 个 EFA `rdmap*` + 2 个非 EFA `ibp*`，§2 第 4 条），
NCCL 默认的网卡选择在这种列表上只建了 2 个 GDAKI NIC，而 `ElasticBuffer` 要的是**每个本地
rank 一个**（每节点 8 个）。设 `NCCL_IB_HCA=rdmap`（前缀匹配，排掉那两个 `ibp`）即可，
日志会变成 `NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0..7 'rdmap*'`。
`run_test_ep.sh` **只在检测到非 `rdmap` 的 ibverbs 设备时**注入这个变量 —— 不无条件设，
否则会掩盖掉"这台机器上 EFA 设备根本不叫 rdmap"这种情况。p5en 是 16 个纯 `rdmap*`，
默认就能建满。

**二、`sm_103` 上 elastic kernel JIT 编译失败。**

```
NVCC compilation failed: ptxas ..._kernel.ptx, line 400; error : Arguments mismatch for instruction 'mov'
ptxas fatal   : Ptx assembly aborted due to errors
RuntimeError: Assertion (csrc/jit/compiler.hpp:239): "NVCC compilation failed"
```

**在第一次 dispatch 时才炸，build 阶段看不出来** —— 这些 kernel 是运行时用 base 镜像里的
`nvcc` JIT 出来的。`deep_ep/include/deep_ep/common/ptx.cuh` 有多处
`#if __CUDA_ARCH__ >= 1000`（`:106` 的 `st.bulk`、`:229` 起的 `.v4.s64` LD/ST，形如
`ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s64 {4 个 64 位寄存器}`）；`sm_103` = 1030
会进这些分支，CUDA 13.0.2 的 ptxas 不认，13.3.1 的接受。所以 `sm_103` 的 CUDA base 必须
≥ 13.3.x（`build_image.sh` 自动给）。pip 的 torch 仍是 `cu130`（CUDA 次版本兼容），只动 base。

**这个坑没有宏可以绕。** `DISABLE_AGGRESSIVE_PTX_INSTRS` 只有 `csrc/kernels/legacy/utils.cuh`
（V1 路径）引用，`deep_ep/include/deep_ep/impls/` 里**零处**；JIT 自己在
`csrc/jit/compiler.hpp` 拼 flags，只处理 `EP_JIT_CPP_STANDARD` / `EP_NUM_TOPK_IDX_BITS` 那几个，
`EP_JIT_EXTRA_FLAGS` 在源码里还是 TODO —— 根本没有往 JIT 传 `-D` 的入口。`setup.py` 对
`arch != 9.0` 加的那个 `-D` 只作用于 AOT 的 `_C.so`。（把那个 TODO 实现掉值得提给上游。）

### 5.4 日志里必须出现的东西

```
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)
NCCL INFO GIN/Plugin: Skipping plugin Libfabric index 3 type 2: NCCL_GIN_TYPE=5 requested
NCCL INFO NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0 'rdmap86s0'   (HCA 0..7 各一行)
NCCL INFO Using network Libfabric
EP: 0/16 | dispatch: 1535.7 us | ...                                (每个本地 rank 一行)
```

- `Loaded gin plugin Libfabric_GDAKI (v14)` 证明 §1 那条依赖链全线打通（这一行同时证明了
  v14 符号、GDA ops、CE verb、内核 capability 位）。
- **但它不证明 GDAKI 被用上了。** 插件注册 ≠ 被选中，两种后端都打这一行。唯一可靠的判据是
  `Skipping plugin ... type 2`（需要 `NCCL_DEBUG=INFO`）。`[Proxy Progress]` 也**不是**判据
  —— NCCL 给普通集合通信也建代理线程，两种后端各 16 条。
- **每份日志只有本机的 8 个 rank**（2 节点共 16）。这不是丢 rank，汇总时必须把每台机器的
  日志都拿到（§7）。
- 判 NCCL 版本用 `NCCL version 2.31.2+cuda13.3`，**别用 `torch.cuda.nccl.version()`**
  （它报 torch 编译期 header，永远是 2.29.7，见附录 A）。

其他可选自检：

```bash
python3 /opt/DeepEP/tests/elastic/test_barrier.py     # 最小连通性
python3 /opt/DeepEP/tests/elastic/test_pp.py          # pipeline 并行
python3 /opt/DeepEP/tests/legacy/test_internode.py    # 旧 NVSHMEM 后端（对照组）
/opt/amazon/efa/bin/fi_pingpong -p efa [<对端私网IP>]  # 纯 libfabric 连通性，排除 DeepEP
```

---

## 6. 正式测量：一条命令跑完 campaign

`run_test_ep.sh` 是"一台机器上的一个 cell"；`run_campaign.sh` 是驱动器：它 ssh 到每个节点，
按矩阵逐个 cell 跑，并把日志命名成 `make_tables.py` 能直接汇总的形式。
**两种机型同一个脚本**，arch 只决定默认 cell 列表。

在**能 ssh 到所有节点**的那台机器上跑它（通常是你的笔记本，因为 `NODES` 用的是你
`~/.ssh/config` 里的 alias；那台也需要一份 clone，但不需要 GPU、也不建镜像）。它在节点上
只用 `REPO_DIR`（默认 `$HOME/work/ep-benchmarks-efa/deepep-v2-efa-official`，即 §4.1 那个
clone）里的 `run_test_ep.sh`，不会往节点推代码。

```bash
# arch 从 leader 探；也可以位置参数显式给 sm90 / sm103
NODES="<leader> <worker>" ./run_campaign.sh
```

默认 cell（3 个 rep，**rep 内轮换**：每个 cell 每个 rep 各跑一次，不是把一条臂跑完再换下一条，
否则热漂移和集群漂移会被读成臂效应）：

| arch | cells |
|---|---|
| `sm90` | `official` × {8192, 128} tok × {12, 24} SM；`prs` × {8192, 128} tok @ 12 SM；`prs` + `EP_MIN_TOKENS_PER_PART=1` @ 128 tok |
| `sm103` | `official` × {8192 @ 24, 128 @ 24, 8192 @ 12}；`prs` × {8192, 128} @ 24 SM；`prs` + `EP_MIN_TOKENS_PER_PART=1` @ 128 tok |

要改就 `CELLS="arm|image|tokens|sms|knobtag|额外 env"`，一行一个 cell。`prs` 臂的镜像
（§4.2）不存在时那几个 cell **整条跳过并打一行提示**，不会一个一个失败。

**`prsmtpp1` 是负控，不是一个变体。** PR #2 在 `EP_MIN_TOKENS_PER_PART=1` 时短路回打补丁
前的几何，所以那个 cell 是**新 binary 里的旧行为**。它没落回 `official` 臂，就说明差异来自
构建或环境，而不是 clamp。

驱动脚本替你守住的东西，每一条都是踩过的：

- **每个 cell 都带那两个 GIN 变量**（`run_test_ep.sh` 的默认值，campaign 按 `GIN_ENV`
  原样透传），并按有无它在 tag 里写 `_gin5` / `_type2`，`verify_run.sh` 再拿 tag 和日志里的
  实际 env 对一遍 —— 两种后端永远不可能被汇总到一起（§10.1）。
- **每个 cell 换一个 `MASTER_PORT`**：被 kill 的 run 会留下 TCPStore listener，下一个卡在
  rendezvous。
- **每一个轴都进文件名**：arm、节点数、SM、tokens、knob、debug、后端、rep、node。少一个轴
  就是静默覆盖掉另一条臂的日志。
- **`EP_BUFFER_DEBUG` 全程不设**、`--ignore-local-traffic` 全程带、`--num-sms` 全程显式。
- **前台 ssh，不用 `nohup`**：detached launch 在断管时是**不对称**失败的，活下来那边重启会
  覆盖已发布的日志。
- **cell 之间等 20 s**：`run_test_ep.sh` 会拒绝在忙 GPU 上启动。
- **不给 JIT cache 挂 host 目录**：每个 `--rm` 容器重新编。这个成本换来的是对 §8 规矩 1 的
  免疫；真要挂，就**一个镜像 tag 一个 host 目录**。

---

## 7. 验收和出表

```bash
# 把每台机器的日志都拉回来 —— 每份只有本机 8 个 rank
for n in 1 2; do scp "<node$n>:~/epruns/*.node$n.log" ./logs/; done

./verify_run.sh logs/*.log                                    # 先验收
EPRUNS=./logs python3 results/p5en_2n4n_20260825/make_tables.py   # 再出表
```

`verify_run.sh` 是闸门，**因为 `rc=0` 什么都不能证明**（§8 规矩 3）。

- **FAIL**（数字不可用）：没有 dispatch 行；本机 rank 数少于 `world/节点数`；按 tag 汇总后
  rank 数少于 world（= 你没把某台机器的日志传进来，或者真丢了 rank）；
  `only N GIN GDAKI NICs`；`ptxas fatal` / `compiler.hpp:239`；tag 名和实际 env 不符。
- **WARN**（数字可用但不可比）：跑在 type-2 后端上；设了 `EP_BUFFER_DEBUG`；`--num-sms=0`；
  缺 `--ignore-local-traffic`；日志里没有 `DeepEP=<sha>` 出处戳；
  `clamped num_allocated_qps from A to B`（你显式给的 QP 数越界了，`9c1f2511` 起是**静默
  clamp** 而不是报错 —— 实际跑的是 B，别拿 A 给这个 cell 命名。和 §8 规矩 1 那个
  `kMaxParts` clamp 是同一种坑）。

`make_tables.py` **生成** §9 的每一张表，`parse_ep.py <tag>` 看单个 tag。别手抄表格 ——
手抄的数字能通过所有 review。两个脚本都用 `finditer` 解析：并发 rank 会把两条记录粘在同一
物理行上，按行 `search` 会静默丢掉一半。

---

## 8. 测量的四条硬规矩

1. **每个变体独立 `EP_JIT_CACHE_DIR`** —— 原因和直觉相反。JIT cache key 是
   `name$$compiler$$flags$$code`（`csrc/jit/compiler.hpp:123`），`compiler` 只是 `"NVCC13.1"`，
   `code` 是**生成出来的 wrapper**；`deep_ep/include/deep_ep/impls/` 下的实现头文件只是被
   `#include`，**内容根本不进 key**。所以两个只差一个头文件补丁的镜像 key 完全相同，共享
   cache 目录会把没打补丁的 cubin 喂给打了补丁的镜像 —— 补丁测出来是个完美的 no-op。
   `flags` **在** key 里，所以同一个镜像内用 env 开关（`EP_NUM_SUB_PARTS` /
   `EP_MIN_TOKENS_PER_PART` …）做 A/B 是安全的。"A/B 完全没差别"最常见的原因就是这个。
2. **交错跑 rep**（`A B A B ...`），绝不能先跑完所有 A 再跑 B —— 否则热漂移和集群漂移全算
   到其中一条臂头上。`run_campaign.sh` 默认就是轮换的。
3. **`rc=0` 不是健康检查。** 上一轮崩掉的 rank 可能还活着、每张 GPU 占着 ~48 GB。下一轮如果
   泄漏不大、GDAKI init 还能过，会跑完、`rc=0`、输出完整，然后**延迟虚高约 2×**，日志里
   没有任何提示（4 节点上见过 combine 连续四轮 7.7 → 12.5 → 16.5 → 19.6 ms 而每轮都报成功，
   显存同步爬 0 → 8.9 → 29.7 → 43 GB）。所以每轮之间必须断言
   `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 全是 0 MiB，并且**每轮换一个
   `MASTER_PORT`**（`TIME_WAIT` 表现为 rendezvous 卡死）。用 docker 跑省事 ——
   `docker rm -f` 会带走整棵进程树。
4. **报数字要报全 rank，并且带上口径。** rank 之间是系统性不同的，而且差异往往**按节点
   分层**（§9.4：combine 是一整台机器慢 13–17%）。单个 rank、单台机器的区间都不是这一轮的
   数字。同时永远把时间（µs）和带宽（GB/s）一起报，并说明分母 —— 只报 GB/s 曾经把结论弄反过。

---

## 9. 实测结果

### 9.0 环境与口径

**p5en.48xlarge × 2 和 × 4，2026-08-25。** Ubuntu 24.04，driver 595.91.07，各 8×H200 +
16 EFA，同 AZ 同 cluster placement group，installer 1.50.0（efa.ko 3.3.0g，tarball md5
`e5a5178944b1f1112f3b2eb3b15ca5a7`）。容器 torch 2.13.0+cu130 / nccl 2.31.2+cuda13.3 /
`deep_ep 2.1.0+ec623f3`。16 卡和 32 卡 `test_ep.py` 都 **exit 0、正确性检查全过**。

**§9.1–9.4 的条件**：GIN type 5、`--test-first-only`、未开 `EP_BUFFER_DEBUG`、SM = **24**。
§9.5 / §9.6 的对照实验在 **12 SM** 上做（要和对照臂对齐），每张表自己标了 SM 数。

**出处**：这些数字的镜像 `BUILD_REF` 是 `ec623f3`，并在 `8e7b42e` 上复核过一遍，8 组配对
全部落在 **0.7%** 以内（`summary.txt` TABLE 8）。Dockerfile 现在默认钉 `9c1f2511`
（= `8e7b42e` 快进 1 个 commit，那个 commit 只把**显式**越界的 `num_allocated_qps` 从
assert 改成 clamp；本文所有臂用的是 auto 或显式 5，都在范围内，走同一条路 —— 这是读
patch 得出的，**没有重测**）。所以按本文重建应当复现这些表。原始日志和生成脚本在
[`results/p5en_2n4n_20260825/`](../results/p5en_2n4n_20260825/)。

**口径**（每次报速率都要一起报）：

- **时间是全 rank 均值** —— 2 节点 16 个 rank、4 节点 32 个，从每台机器的日志汇总。单台
  机器的均值**不能**代替它：combine 按节点分层（§9.4），单节点均值能偏 12%。
- **`SO` / `SU` 是跨 rank 的 min–max，时间是同一批样本的均值** —— 两列统计口径不同，
  §9.3 解释这个区间是什么。
- `SU` = 日志里那个 `bytes` 字段 ÷ 时间。该字段是 `num_scaleup_bytes`
  （`tests/elastic/test_ep.py:271`），即本 rank 收到的**全部** token 字节。日志**不打印**
  scale-out 字节，只能用 `SO × 时间` 反推。
- `SO` **不是线速**：`test_ep.py:253` 的循环在不加 `--ignore-local-traffic` 时会把本机也算作
  一个 scale-out 目的地，机内流量也计入。
  **线速占比 = `SO × (N−1)/N ÷ 每 GPU 线速`**，分母**按机型换**：p5en **50 GB/s**
  （16×200 Gb/s ÷ 8），b300 **100 GB/s**（每 GPU 两张 EFA）。拿 50 去除 b300 的 SO 会得到约
  两倍的"线速占比"，而输出看不出任何异常 —— 这是最容易犯又最难发现的错。
  p5en 上 N=2 时线速占比数值上等于 `SO`、N=4 时是 `SO × 1.5`（这个巧合来自分母恰好是 50）。
  该修正假设 N 个目的节点均分，是近似；要精确值就加 `--ignore-local-traffic`，`SO` 会直接
  变成真跨机字节 ÷ 时间。
- 每 rank **scale-up**（= 打印的 `bytes`）字节，8192 tok：dispatch 2 节点 395.9–402.4 MB /
  4 节点 441.4–447.7 MB，combine 759.7–772.1 / 847.0–859.0 MB。128 tok：dispatch
  5.6–6.3 / 5.8–6.4 MB，combine 10.8–12.1 / 11.1–12.3 MB。
- 每 rank **scale-out** 字节（`SO × 时间`，含机内），8192 tok：dispatch 2 节点
  121.3–123.0 MB / 4 节点 219.1–223.6 MB，combine 232.7–236.2 / 420.9–428.5 MB。

### 9.1 Prefill（`--num-tokens=8192`）— 带宽

时间是**均值**，`SO` / `SU` 是**跨 rank 的 min–max**。

| op | 2 节点 时间 | SO | SU | 4 节点 时间 | SO | SU |
|---|---|---|---|---|---|---|
| dispatch | **1535.7 µs** | 79–80 | 257–263 | **3972.7 µs** | 55–56 | 111–113 |
| expanded dispatch | 1537.3 µs | 79–80 | 258–262 | 3971.3 µs | 55–56 | 111–113 |
| cached dispatch | 1620.7 µs | 75–76 | 244–249 | 4256.6 µs | 51–52 | 103–106 |
| combine | 3534.0 µs | 62–73 | 202–239 | 7710.7 µs | 51–57 | 103–115 |
| reduced combine | 3362.6 µs | 64–79 | 209–259 | 7728.3 µs | 51–57 | 103–115 |

（2 节点 3 轮 × 16 rank = 48 个 rank 观测，4 节点 2 轮 × 32 rank = 64 个，全部汇总。）

**dispatch 在两个规模上都跑在线速的 80–84%**：2 节点 79.6%，4 节点 83.7%
（`SO` 均值 55.8 × 1.5）。4 节点利用率反而更高的原因：每 rank 计入的 scale-out 字节从
122.1 涨到 221.3 MB（+81%），其中真正跨机的比例又从 1/2 升到 3/4，于是**真跨机**字节是
61.0 → 165.9 MB（×2.72），而时间只涨到 2.59×。

### 9.2 Decode（`--num-tokens=128`）— 延迟

这个尺寸每 rank 只有 5.6–6.4 MB，是**消息率受限**不是带宽受限，只看时间：

| op | 2 节点 | 4 节点 |
|---|---|---|
| dispatch | **147.3 µs** | **184.7 µs** |
| expanded dispatch | 146.6 µs | 184.0 µs |
| cached dispatch | 135.8 µs | 184.3 µs |
| combine | 151.9 µs | 236.9 µs |
| reduced combine | 160.1 µs | 239.6 µs |

从 2 节点到 4 节点 dispatch 只涨 25%，**这是 type 5 的特性**：type-2 代理后端在同一组机器上
是 365.1 → 1003.2 µs（2.75×，§10.1）。§9.6 的两个 PR 能把 2 节点 dispatch 再压到 106 µs。

### 9.3 那个 `SO` 区间是什么 —— 它不是重复性方差

`SO` / `SU` 两列是**同一时刻 48（或 64）个 rank 之间的 min–max**，不是同一个数字重复跑若干
次的离散。混起来看会以为方差大得离谱。拆开：

| 2 节点 / 24 SM / 8192 tok | 时间跨 rank 离散 | 打印字节跨 rank 离散 | `SO` 离散 |
|---|---|---|---|
| dispatch | **0.5%** | 1.6% | 2.1% |
| expanded dispatch | 0.3% | 1.6% | 1.7% |
| cached dispatch | 0.4% | 1.6% | 1.9% |
| combine | **16.1%** | 1.6% | 16.8% |
| reduced combine | **20.7%** | 1.6% | 21.8% |

（4 节点：dispatch 时间 0.9% / 字节 1.4% / `SO` 2.3%；combine 时间 11.1%。）

1. **dispatch 的区间来自分母，不是速度。** 时间跨 rank 只差 0.5%（4 节点 0.9%），但每个
   rank 路由到的 token 数不同（`get_unbalanced_scores` 制造的不均衡），字节数差 1.4–1.6%，
   而 `GB/s = 本 rank 字节 ÷ 本 rank 时间`，于是继承了字节的离散。
2. **再叠一层整数打印。** `test_ep.py` 用 `:.0f` 打印 GB/s。`SO ≈ 79.6` 时一个整数单位就是
   **1.26%**，4 节点 `SO ≈ 55.8` 时是 **1.79%**。所以 `79–80` 和 `55–56` 是**两个相邻整数
   —— 这个打印格式能表达的最窄区间**。dispatch 上根本没有可测的方差问题。
3. **真正宽的只有 combine / reduced combine**，而那里的驱动是**时间**不是字节：字节仍然只差
   1.6%，时间却差 16–21%。那是 §9.4 的按节点分层，不是网络抖动。
4. **重复性极好，连 combine 也一样。** 逐轮的全 rank 均值：

   | 24 SM / 8192 tok | rep1 | rep2 | rep3 | 轮间离散 |
   |---|---|---|---|---|
   | 2 节点 dispatch | 1535.5 | 1537.0 | 1534.5 µs | **0.16%** |
   | 2 节点 combine | 3533.2 | 3535.1 | 3533.6 µs | **0.05%** |
   | 2 节点 reduced combine | 3366.3 | 3356.1 | 3365.5 µs | **0.30%** |
   | 4 节点 dispatch | 3971.6 | 3973.8 | — | **0.06%** |
   | 4 节点 combine | 7713.8 | 7707.6 | — | **0.08%** |

所以 combine 那 16–21% 是**结构性的、可重复的 rank 间分布**，不是不稳定：全 rank 均值在
不同轮次之间只差 0.05–0.30%。

### 9.4 combine 按节点分层 —— 这决定了怎么汇总

dispatch 在 rank 之间是均匀的（2 节点 24 SM 8192 tok：16 个 rank 全在 1531–1538 µs），
combine 和 reduced combine **不是** —— 它们按机器整齐地分成两层：

| 2 节点 / 24 SM / 8192 tok，3 轮汇总，各取本机 8 个 rank 均值 | node1 | node2 |
|---|---|---|
| dispatch | 1535.0 µs | 1536.3 µs（差 0.1%） |
| combine | 3300.3 µs | 3767.7 µs（差 13.2%） |
| reduced combine | 3074.9 µs | 3650.3 µs（差 17.1%） |

**慢的那台机器在同一批连续 rep 里是固定的，只在不同批次之间翻转** —— 它是 per-launch 的
性质，不是 per-iteration 的噪声。24 SM 那三轮全是 node2 慢（combine 3764 / 3771 / 3769 µs
对 3303 / 3299 / 3299），另外两批则是另一台慢；幅度在所有批次里都稳定在 14–18%。所以
**汇总多轮不会摊平 2 节点的分层**，单节点的 combine 均值会落在 ~3100 或 ~3770 µs，取决于
你读的是哪份日志。汇总全部 16 个 rank 之后，那三轮彼此只差 0.30%。**报全 rank。**
4 节点上两轮里慢的机器不同（rep1 是 node4 的 8087 µs，rep2 是 node1 的 8117 µs），
所以汇总后的 4 节点分层只剩 4.2%。

### 9.5 不要从源码手编这套栈

对照臂 `dev-7a6059a3` 是手编 NCCL `2.30.7-1` + 手编 `--enable-gdaki` aws-ofi-nccl +
DeepEP `7a6059a3`，多带一个 `--skip-check`。同一批机器、同一天、12 SM、两侧都在 type 5 上：

| arm | dispatch | cached dispatch | combine | reduced combine |
|---|---|---|---|---|
| 2 节点 8192 tok 手编 | 1517.0 µs | 1749.1 µs | 3621.2 µs | 4166.2 µs |
| 2 节点 8192 tok 正式包 | 1502.9 µs | **1591.0 µs** | 3602.5 µs | 4237.9 µs |
| 2 节点 128 tok 手编 | 301.2 µs | 269.5 µs | 175.9 µs | 188.3 µs |
| 2 节点 128 tok 正式包 | **169.4 µs** | 165.9 µs | 162.7 µs | 179.0 µs |
| 4 节点 8192 tok 手编 | 3962.5 µs | 4709.2 µs | 7993.7 µs | 8067.7 µs |
| 4 节点 8192 tok 正式包 | 3955.3 µs | **4239.7 µs** | 7842.6 µs | 7943.2 µs |
| 4 节点 128 tok 手编 | 320.9 µs | 287.3 µs | 277.4 µs | 286.1 µs |
| 4 节点 128 tok 正式包 | **184.3 µs** | 179.5 µs | 244.2 µs | 253.3 µs |

prefill dispatch 平手（差 0.2–0.9%），正式包在 **cached dispatch 上快 9.9–11.1%、
decode dispatch 上快 1.74–1.78×**。**没有任何性能理由去手编。**

**两边跑的是同一个 kernel** —— 按 blob 比出来的，不是读 commit message 猜的，以下全部可从
源码核实。`csrc/kernels/elastic/kernel_select.hpp:37-52` 写明 **unordered 是 default**
（env 为空即 unordered），而 hybrid dispatch kernel 在正式包这棵树里有两份：

| 正式包里的文件 | vs `af9a040`（fork 之前的 upstream） | vs `7a6059a3`（手编臂跑的） |
|---|---|---|
| `hybrid_dispatch.cuh` —— 即 `ordered` 变体 | **+6 / −1** | +56 / −468 |
| `hybrid_dispatch_unordered.cuh` —— **default** | +485 / −51 | **+45 / −28**（其中 11 行是 license header） |

所以 `unordered` **就是** `7a6059a3` 那套 kernel（AWS 的 EFA GDA 移植版）换了个名字，
`ordered` 是 fork 之前的 upstream kernel 几乎原样带着走。combine 同理（`+12/−6` vs
`+24/−7`）。那 +45/−28 的全部实质内容是：多一个
`num_unaligned_recv_tokens_per_expert` 输出指针（每个 expert 一次 store）、两个 lambda 从
warp 分支里提到外面、删掉恒等 lambda `phys_token_slot`，加注释。

⇒ **`EP_HYBRID_KERNEL=ordered` 不是一个有效的 A/B。** 那个开关选的是 *upstream* kernel：
发端 publish 一个 trailing tail signal、收端假定 tail 之前的全部已落地 —— 实测它在
**EFA GDAKI 上结果不正确**（`NCCL_GIN_TYPE=5`，signal 可能超过数据），只在有序的 proxy
路径上成立。而且它向 NCCL 要的是另一套 fabric（`csrc/kernels/backend/nccl.cu:111-127`：
129 个 **exclusive** context、`ginSignalCount = num_ranks + 4`、`ginVaSignalsRequired`、
`ginStrongSignalsRequired`），就算正确也不是单变量对照。**EFA 上保持默认 unordered。**

**dispatch 路径上唯一一处实质差异是 context 数策略**（`gin_resource_alloc.cuh`，+80/−50）。
两边在 12 SM / 每 SM 4 channel / 16 rank 下都能从源码算出来，且和各自日志逐位吻合：

| | context 数（= QP 数） | signal/ctx | data QP | channel/ctx | `kNumParts` | 日志 |
|---|---|---|---|---|---|---|
| 手编 `7a6059a3` | `ceil_div(12×4, 10)` = **5**，随 SM 变 | 49 | 4 | 12 | **4** | `5 / 49 / 5` ✅ |
| 正式包 | 固定常量 `kDefaultGinContextCnt` = **11**，与 SM 无关 | 21 | 10 | 5 | **4** | `11 / 21 / 11` ✅ |

**注意没变的东西：两边都是 `kNumParts = 4`、48 个 channel**，kernel 的模板实参完全一样。
唯一活着的差别是这 48 个 channel 摊在多少个 QP 上（手编 4 QP × 12 channel，正式包
10 QP × 5 channel）—— 下表把这个差别测出来是：对 plain dispatch 一文不值。

**2 节点 prefill 所有臂放在一根轴上**（16 rank / 12 SM / 8192 tok，全 rank 均值）：

| arm | dispatch | SO GB/s | 线速% |
|---|---|---|---|
| 正式包，默认 env（type 2） | 1644.0 µs | 74.0 | 74.0 |
| 正式包 + `--num-allocated-qps 5`（type 2） | 1631.1 µs | 75.0 | 75.0 |
| 手编 `7a6059a3`，`--skip-check` | 1517.0 µs | 80.6 | 80.6 |
| 正式包 + 5 个 route-B 变量全开 | 1504.4 µs | 81.0 | 81.0 |
| **正式包 + `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`** | **1502.9 µs** | 81.2 | 81.2 |
| `main` `8e7b42e` + 那一对 | 1502.0 µs | 81.0 | 81.0 |

**这一整段差距全部由那两个环境变量解释** —— 不是 QP fan-out、也不是 `--skip-check`：另外
三个 route-B 变量叠上去毫无增益，`main` 和正式包无法区分。**`--num-sms` 在正式包上不改 QP
数，是一根纯性能轴**（auto path 是常量 11，`results/.../logs/` 里每份日志都打 `#QPs: 11/11`
—— 6/12/16/24/32 SM、2 节点和 4 节点全都是 11），所以 SM 可以放心扫。QP flag 在 type 5 之上
对 plain dispatch 是零影响（1502.9 → 1508.5 µs），它真正管的是 **cached dispatch，而且 5 个
context 更差 9.6%**（1591.0 → 1743.9 µs）；4 节点 decode 上更差 19.3%
（1003.2 → 1197.0 µs）。**保持默认 11。**

### 9.6 两个待合的 PR：decode dispatch 再减 33%

decode 上有两个互相独立的杠杆：GIN 后端（只改 env，§10.1）和 dispatch 的 part 几何
（两个 PR，§4.2 的第二条臂）。两者**可以叠加**。

| PR（基于 `main`） | 内容 |
|---|---|
| [#1](https://github.com/amazon-contributing/DeepEP/pull/1) | 把 `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` / `EP_MIN_TOKENS_PER_PART` 转发给 JIT，不改任何默认值 |
| [#2](https://github.com/amazon-contributing/DeepEP/pull/2) | 加 `kMinTokensPerPart`（默认 15，`EP_MIN_TOKENS_PER_PART` 可覆盖），`kNumParts = min(预算, 每 channel token 数 / 15)` |

**2 节点 / 16 rank / 12 SM / 128 tok**（全 rank 均值）：

| arm | dispatch | 相对同后端 | combine | reduced combine |
|---|---|---|---|---|
| 未打补丁，type 2 | 365.1 µs | 1.00× | 175.1 µs | 192.7 µs |
| #1 + #2，type 2 | 237.1 µs | **−35.1%** | 176.3 | 193.6 |
| 未打补丁，type 5 | 169.4 µs | 1.00× | 162.7 | 179.0 |
| #1 + #2，type 5 | 112.7 µs | **−33.5%** | 162.2 | 178.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`，type 5 | **106.4 µs** | **−37.2%** | 162.1 | 178.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`，type 5 | 171.5 µs | 关掉 clamp 的负控 | 162.6 | 178.8 |

**4 节点 / 32 rank / 12 SM / 128 tok：**

| arm | dispatch | 相对同后端 | combine | reduced combine |
|---|---|---|---|---|
| 未打补丁，type 2 | 1003.2 µs | 1.00× | 344.2 µs | 357.0 µs |
| #1 + #2，type 2 | 627.4 µs | **−37.5%** | 336.9 | 346.8 |
| 未打补丁，type 5 | 184.3 µs | 1.00× | 244.2 | 253.3 |
| #1 + #2，type 5 | 169.5 µs | **−8.0%** | 243.5 | 252.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`，type 5 | **155.9 µs** | **−15.4%** | 243.6 | 251.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`，type 5 | 184.1 µs | 关掉 clamp 的负控 | 243.5 | 252.9 |

> 出处：`#1 + #2` 这条臂的镜像 `BUILD_REF` 是 `b097b03`（当时 #2 的 head，内容即这两个
> commit）；现在重建用 §4.2 查到的当前 head。负控（`EP_MIN_TOKENS_PER_PART=1`）是在**同一个
> 打了补丁的镜像里**跑的，所以 clamp 的效果是干净的、不含 base 差异。

三条结论：

1. **两个杠杆独立、可叠加。** 叠起来 2 节点 decode dispatch 从 365.1 → 106.4 µs，**3.43×**
   —— 一对环境变量加两个 commit。
2. **combine 和 reduced combine 在每一行里都动不到 1%。** 这就是"机制在 dispatch 内部的
   part 几何、不在网络"的证据：EFA 两边干的活一样多。expanded / cached dispatch 跟着
   plain dispatch 走。
3. **clamp 的收益在 4 节点 type 5 上崩掉了** —— −8.0% 而不是 −33.5%，而且负控（184.1 µs）
   落在未打补丁（184.3 µs）的噪声里，说明是 clamp 本身不再付钱、而不是 PR 里别的东西。
   type-5 4 节点 decode dispatch 有一个 **156–185 µs 的地板**，part 几何摸不到。
   **原因未知 —— 不要把 2 节点那个比例外推到更大的集群。**

**为什么 part 几何会有代价**（combine 完全不动就是证据）：`kNumParts`（一个 channel 的
token 分几次 `flush_part` put 发出去）只由 `compute_part_allocation()` 决定，而它只在 GIN
indexed-signal 预算紧时**从上面**压；预算恰恰在 channel 里 token 最少时最松，所以 decode
必然落到 `kMaxParts` —— 轴的最坏一端。128 token / 12 SM 下每 channel 只有 3 个 token 却被
描述成 4 part × 1 token：最后一个 part 永远是空的，3 个 token 发成 3 次单 token put 而不是
1 次 3-token put。sub-part 早有两道保护（clamp 到 `kBatchSize`、`EP_SM100_MIN_SUB_TOKENS`），
part 一道都没有。

**取舍**：只关心 prefill 就不用管这两个 PR（2 节点 24 SM 8192 tok：1535.7 vs 1536.0 µs；
4 节点 12 SM 8192 tok：3955.3 vs 3955.3 µs，都是噪声）。要发 decode / 小 token 的数字就
自己 cherry-pick 这两个（#1 不改默认值，必须和 #2 叠加才有收益），并且无论如何都要设
type-5 那一对。合并后 #2 的默认值 15 自动生效，不需要设环境变量。

### 9.7 b300 现状

本镜像在 b300 上目前只有抽查（2 节点 16 rank，`--num-tokens 8192`）：dispatch 12 SM
≈ **1025 µs**、combine 24 SM ≈ **1800 µs**。这是两个不同 SM 档上各一次读数，**不是
campaign** —— 它只说明这套栈在 b300 上跑通了，不说明 B300 有多快。跑法见 §6，
`sm103` 的默认 cell 表里留了一个 12 SM 的 prefill cell 用来和这两个点对齐。

b300 上成体系的数字来自**另一个镜像**（AWS `awsome-distributed-ai#1234` 的 recipe：
CUDA 13.1.2 / torch 2.11，DeepEP pin 相同），在
`../adai-ep-comparison-b300/RESULTS_b300.md`，**口径不同，不能和 §9.1–9.6 混排**。那批数字
里和本文直接相关的两条：PR #2 的 clamp 在 b300 上复现且**更大**（decode dispatch −37.6%，
p5en 是 −33.5%），但 **PR #1 的 `EP_NUM_SUB_PARTS=1` 在 `sm_103` 上是零到微负**，所以
§9.6 里"再叠一个 `EP_NUM_SUB_PARTS=1`"是 Hopper 专属；另外 b300 上 clamp 是把 SM 曲线
**压平**（打了补丁后 12 SM 和 24 SM 打平）而不是移动最优点。

---

## 10. 参数速查

### 10.1 必须设的那两个环境变量

1.50.0 的 `libnccl-net-ofi.so` 注册了**两个** GIN plugin：一个 Libfabric 代理式
（**type 2**）、一个 `Libfabric_GDAKI`（**type 5**）。两个都会加载，
`Loaded gin plugin Libfabric_GDAKI (v14)` 两种情况下都打 —— 但**默认 NCCL 选 type 2**。
所以这两个变量不影响 GDAKI 能不能**加载**，影响的是加载之后**用不用它**：

```
NCCL_GIN_TYPE=5  NCCL_SYM_GIN_KERNELS_ENABLE=0
```

**实测收益**（4 × p5en，16/32 rank，12 SM，`--test-first-only`，全 rank 均值）：

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

**这一对是最小且充分的组合**，其余 GIN 相关变量逐个单变量测过（2 节点 12 SM 128 tok，
全 rank dispatch）：默认 359.2 / 371.1 µs、`FI_EFA_USE_HW_CNTR=1` 375.8 µs、
`NCCL_RMA_DISABLE=1` 359.5 µs、`NCCL_SYM_GIN_KERNELS_ENABLE=0` 单独 359.4 µs、
`OFI_NCCL_GIN_STRONG_SIGNAL=1` **750.4 µs**（16 个 rank 散在 371.0–1130.0 µs，主动变坏 ——
和 unordered kernel 只要 weak signal 一致）、这一对 169.4 µs、完整 5 个 route-B 变量
170.0 µs（不比这一对好）。

判据见 §5.4：唯一可靠的是 `Skipping plugin ... type 2` 那一行。两份 grep 结果在
[`results/p5en_2n4n_20260825/logs/`](../results/p5en_2n4n_20260825/logs/)
（`gin_plugin_selection_default.grep.txt` / `_gin5symgin0.grep.txt`）。

**这一对是 `run_test_ep.sh` 的默认值**（`GIN_ENV`），`run_campaign.sh` 原样透传并在 tag 里写
`_gin5`；`GIN_ENV=` 置空就是 type-2 对照臂，tag 写 `_type2`，`verify_run.sh` 会拿 tag 和日志里
的实际 env 对一遍（不符直接 FAIL）。**默认在 launcher 而不是烧进镜像**：烧进镜像的话对照臂就
不可测了，而放在 launcher 里对照臂只是 `GIN_ENV=` 一个字。

### 10.2 SM 怎么选

**p5en 的工作点：2 节点和 4 节点都用 24 SM。** 8192 tok、type 5、`--test-first-only`、
全 rank 均值：

| SM | 2 节点 dispatch | 2 节点 reduced combine | 2 节点合计 | 4 节点 dispatch | 4 节点 reduced combine | 4 节点合计 |
|---|---|---|---|---|---|---|
| 6 | 2290.5 µs | 7377.7 µs | 9668.2 µs | 4030.7 µs | 9518.8 µs | 13549.4 µs |
| 12 | 1502.9 µs | 4237.9 µs | 5740.8 µs | 3955.3 µs | 7943.2 µs | 11898.5 µs |
| 16 | 1510.5 µs | 3584.1 µs | 5094.6 µs | — | — | — |
| **24** | 1535.7 µs | **3362.6 µs** | **4898.3 µs** | 3972.7 µs | **7728.3 µs** | **11701.0 µs** |
| 32 | 1576.5 µs | 3486.4 µs | 5062.9 µs | — | — | — |

- **dispatch 从 12 到 24 SM 基本是平的**（2 节点 +2.2%、4 节点 +0.4%），付钱的是 reduced
  combine。所以拿这点 dispatch 换 combine 总是划算：2 节点用 32.8 µs 的 dispatch 换回
  875.3 µs 的 reduced combine（层总时间 −14.7%），4 节点 −1.7%。
- **6 SM 在任何规模上都是错的选择**（相对 24 SM，两个口径都给）：2 节点 dispatch +49.2%、
  dispatch+redComb +97.4%；4 节点 dispatch 只 +1.5%，但 dispatch+redComb +15.8%。两个口径
  差这么远，正是因为付钱的是 reduced combine —— 引用时必须说清是哪一个。
- **decode（128 tok）也选 24 SM**：2 节点 dispatch 169.4 → 147.3 µs（−13.0%）、reduced
  combine 179.0 → 160.1 µs（−10.6%）；4 节点 dispatch 在 6/12/24 SM 上是平的
  （181.2–184.7 µs，散布 1.9%），只有 reduced combine 动，24 SM 比 12 SM 好 5.4%
  （253.3 → 239.6 µs）。
- **打了 §9.6 那两个 PR 之后，2 节点 decode 反而 12 SM 更好**（112.7 vs 145.3 µs）；
  4 节点两档在 dispatch + reduced combine 上打平（422.4 vs 422.2 µs）。所以
  `run_campaign.sh` 的 `prs` 臂默认跑 12 SM。
- **这张表别搬到 b300**：SM 轴在本镜像的 `sm_103` 上还没扫过（§11 第 8 条）。另一个镜像上的
  观测是 clamp 把 b300 的 SM 曲线压平而不是移动最优点，形状和 p5en 不同。b300 默认起点取
  24 SM，理由只是"和 p5en 的工作点对齐好比较"，不是量出来的最优点。

### 10.3 环境变量

容器内：

| 变量 | 用的值 | 说明 |
|---|---|---|
| `NCCL_GIN_TYPE` | **`5`**（launcher 默认已设） | 见 §10.1。必须和下一行成对，单独设会崩 |
| `NCCL_SYM_GIN_KERNELS_ENABLE` | **`0`**（launcher 默认已设） | 同上。sym GIN kernel 要 strong signal，GDAKI 没有 |
| `EP_HYBRID_KERNEL` | `unordered`（默认，不用设） | **EFA 上必须是 unordered**；`ordered` 在 GDAKI 上结果不正确（§9.5） |
| `EP_NIC_NAME` | `rdmap85s0`（`run_test_ep.sh` 自动探） | 上游默认 `mlx5_0` 是 IB 的。只喂 `get_rdma_gbs()`，而它只返回**一块**网卡的速率，所以 `--num-sms` 仍要显式给 |
| `NCCL_IB_HCA` | b300 上 `rdmap`（自动注入） | 只在这台机器还有非 EFA 的 ibverbs 设备时需要（§5.3） |
| `EP_MIN_TOKENS_PER_PART` / `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` | 见 §9.6 | decode 的 part 几何。**要 PR #1 才能进 JIT**，没有 #1 时它们静默无效 |
| `NCCL_NET_PLUGIN` | `ofi` | 用名字解析，别写死路径（插件叫 `libnccl-net-ofi.so`） |
| `FI_PROVIDER` / `FI_EFA_USE_DEVICE_RDMA` | `efa` / `1` | provider 是 `efa` 不是 `efa-direct`，fabric 由插件自己选 |
| `EP_JIT_CACHE_DIR` | `/root/.deep_ep` | 做 A/B 时**每个镜像一个目录**（§8 规矩 1） |
| `EP_BUFFER_DEBUG` | **只在调试时** `1` | 打 NCCL 版本 / QP 数 / GIN layout，但它**在计时区间里 printf**（`csrc/elastic/buffer.hpp:1151` 在 dispatch 的 host 轮询循环里拼 stringstream）。确认完 layout 就关掉，不要用它跑要发布的数字 |
| `EP_DISABLE_GIN` | `0` | 设 1 关掉 GIN，用来隔离问题 |

镜像里还有三个只读出处戳，`run_test_ep.sh` 启动前会读：`EP_BUILD_ARCH`、`EP_BUILD_CUDA`、
`EP_EFA_INSTALLER`。和 host 的 `compute_cap` 不符就拒绝启动（`ALLOW_ARCH_MISMATCH=1` 越过）。

launcher / driver 侧（不是容器内变量）：

| 变量 | 用的值 | 说明 |
|---|---|---|
| `TOKENS` / `NUM_SMS` | `8192` / `128`；`12`、`24` | prefill 和 decode 只差 `TOKENS`；`NUM_SMS` 别用 0（§5.2） |
| `IGNORE_LOCAL` | `1` | 传 `--ignore-local-traffic`。不带的话 SO 分母含机内流量（§9.0 口径） |
| `TEST_FIRST_ONLY` | `1` | `--test-first-only` = FP8 dispatch @ `expert_alignment=128`（`enumerate_ep_modes()` 第一项）。设 `0` 是跑整个模式笛卡尔积，几小时 |
| `EXTRA_ENV` | `"NAME=VALUE …"` | 一次性 env 钩子，用来做单变量 A/B。同名变量会**顶掉** `GIN_ENV` 的默认值（脚本显式丢弃重复项并打一行提示，不依赖 docker 怎么处理重复的 `-e`） |
| `NODES` | `"<leader> <worker>"` | `run_campaign.sh` 的节点列表（ssh 别名或 IP），顺序即 node rank |
| `IMAGE_BASE` / `IMAGE_PRS` | `:<arch>` / `:<arch>-5a594a5` | 两条臂的镜像 tag；`IMAGE_PRS` 不存在时那几个 cell 整条跳过 |
| `GIN_ENV` | 那对 GIN 变量（**两个脚本共同的默认值**） | 置空即 type-2 对照臂，tag 相应写 `_type2`。`run_campaign.sh` 按 `GIN_ENV` 透传而不折进 `EXTRA_ENV`，否则置空会被子脚本的默认值改回 type 5 而 tag 仍写 `_type2` |
| `REPS` / `CELLS` / `PORT_BASE` / `LOGDIR` | `3` / arch 默认表 / `8500` / `~/epruns` | `CELLS` 一行一个 cell |

---

## 11. 还没测的

按「读者最想要 / 代价最小」排。全都不需要重新 build 镜像。

1. **DeepEP V1 跑一次 `--num-tokens 8192` + FP8 dispatch**（1 次 run，用现成的
   `deepep-v1-efa:dev`）。仓库顶层的吞吐表里 V1 是 4096/BF16、V2 是 8192/FP8，**两个方向都
   不可比**。V1 自己已经有 4096 的 FP8 dispatch 数（p5 48.17 / p5en 54.98 GB/s），所以缺的
   那一格在 V1 那边，不用重跑本目录的 campaign。
2. **4 节点的 `--num-sms` 轴上界没封**（2 次 run：4N/16 SM、4N/32 SM）。2 节点已经封死且
   最优点在**内部**（24 SM 4898.3 µs，32 SM 反而 5062.9）；4 节点只有 6/12/24，曲线还在
   单调变好，所以"4 节点也用 24 SM"站在一条没封的轴上。
3. **没有单机基线**（1 次 run）。本目录全部是 ≥ 2 节点，所以 DeepEP 自身的 kernel 开销和
   跨机 EFA 开销从来没分开过。
4. **combine 慢的那台机器跟机器还是跟角色**（1 次 run，把 node rank 对调）。§9.4 的分层在
   一批连续 rep 内固定、跨批翻转；把 leader 角色换到另一台就能判定。
5. **4 节点那几条臂太薄**（给 4N 的行补第 3 个 rep）。多数 4 节点臂只有 1–2 个 rep，
   2 节点是 3 个；轮间离散实测 ≤ 0.31%，但那主要是 2 节点上测的。
6. **BF16 dispatch，任何 scale 都没测过**（要改 harness，不是加跑一次）。`--test-first-only`
   固定选中 `use_fp8_dispatch=1, expert_alignment=128`（`test_ep.py:33-41`），所以**本文每一个
   数字都是 FP8 dispatch @ alignment 128**；`test_ep.py` 没有单独选 BF16 的开关。目前唯一一个
   BF16 读数来自 b300 那个 kit，显示 BF16 dispatch 是**更慢**的那一侧，所以这里的数字没有
   因为这个选择被美化 —— 但它仍然是一根没测的轴。
7. **`--ignore-local-traffic` 的精确校验**：`wire% = SO × (N−1)/N ÷ 每 GPU 线速` 只在
   **2 节点**上对着实测核过一次，4 节点的 ×0.75 至今是纯算术。
8. **本镜像在 `sm_103` 上的正式 campaign**（§9.7）。build 和 launcher 都已支持 b300，剩下的
   就是每台机器 `./build_image.sh`（要 `prs` 臂就跑两次）然后
   `NODES="<leader> <worker>" ./run_campaign.sh`。

---

## 12. 故障对照表

| 现象 | 原因 | 处理 |
|---|---|---|
| `run_test_ep.sh` 报 `REFUSING TO START`（arch / CUDA 不匹配） | 镜像的 `EP_BUILD_ARCH` 和 host `compute_cap` 不符，或 `sm_10x` 配了 < 13.3 的 CUDA base | 按 §4.1 重建（`./build_image.sh` 自己 probe）。确实要跨档跑再 `ALLOW_ARCH_MISMATCH=1` |
| `only 2 GIN GDAKI NICs have been created` + `NCCL exception (nccl.cu:185): 5` | 混合 ibverbs 设备列表（b300 有 2 个非 EFA 的 `ibp*`）让 NCCL 少建了 GDAKI NIC | `NCCL_IB_HCA=rdmap`，`run_test_ep.sh` 已自动注入（§5.3） |
| `Arguments mismatch for instruction 'mov'` → `ptxas fatal` → `compiler.hpp:239` | `sm_103` 命中 `ptx.cuh` 的 `__CUDA_ARCH__ >= 1000` 分支，CUDA 13.0.2 的 ptxas 不认。第一次 dispatch 才炸 | 重建镜像用 CUDA 13.3.1（§5.3）；没有宏能绕 |
| `ibv_devinfo -l` 数出 18 个设备（b300） | **正常**：16 个 EFA `rdmap*` + 2 个非 EFA `ibp*` | 自检用 `grep -c rdmap`（§2 第 4 条） |
| `ce_probe` 对 `ibp198s0f0` / `ibp199s0f0` 报 CE FAIL / errno 95 | **正常**，它们不是 EFA 设备 | 只看 16 个 `rdmap*` 是否全 `CE OK` |
| b300 算出来的"线速占比"约 200% | 分母用了 p5en 的 50 GB/s；b300 每 GPU 两张 EFA = 100 GB/s | 换分母（§9.0 口径） |
| `num_sms` 自动探测偏小（b300 上 `rdma_gbs=50.0`） | `get_rdma_gbs` 只返回**一块**网卡的速率 | 显式给 `--num-sms`（§5.2） |
| `ZeroDivisionError` in `get_theoretical_num_sms` / `Failed to get RDMA connection speed:` | 旧 DeepEP tree 上 `--num-sms 0` 走 `ibstat` 自动探测，而 `ibstat` 走 libibumad，EFA 没有 `ib_umad`。当前 pin 先读 sysfs，不会崩 | 显式给 `--num-sms`；单机不触发，只在上多机那一刻出现 |
| 改了 `--num-sms` 后整轮**无输出挂死** | 不是 QP 数的问题（`#QPs` 恒为 11、与 SM 无关，6/12/16/24/32 SM 全部跑通） | 先查上一轮有没有漏进程：`nvidia-smi` 确认显存全 0 MiB，每轮换 `MASTER_PORT`（§8 规矩 3） |
| 延迟虚高 ~2× 但 `rc=0`、输出完整 | 上一轮泄漏的 rank 在抢显存 | `rc` 查不出来，必须查 `nvidia-smi`（§8 规矩 3） |
| A/B 完全没有差别 | 两个镜像共享了 JIT cache 目录 —— 实现头文件的内容不进 cache key | 一个镜像一个 `EP_JIT_CACHE_DIR`（§8 规矩 1） |
| installer 装完，`/sys/module/efa/version` 还是旧版本（例如 3.0.0） | 旧模块当时卸不掉（有进程占着 EFA），installer 结尾打的是 `Please reboot` | 清掉占用进程/容器后重启，再核这一行（§3.1） |
| `efa_installer.sh -y` 直接 exit 1，打了一段 `apt-get upgrade` | Ubuntu AMI 的内核不带 `ib_uverbs`；`-y` 模式不会替你升内核 | 按它打的命令 `apt-get upgrade` + 重启，再回来装（§3.1） |
| `ibv_devinfo`: No IB devices found，但 `lsmod` 有 efa | 启动时 ENI 没开 EFA | 重建实例，`InterfaceType=efa`（§2 第 1 条） |
| `ibv_create_ah failed with EINVAL ... different availability zone` | 跨 AZ | 同 AZ + cluster placement group |
| `fi_info -p efa-direct` → `-61 (No data available)` | **正常现象**，`-p` 匹配 provider（`efa`）不是 fabric | 用 `fi_info \| grep fabric`（§3.2） |
| `fi_info: command not found` | 不在默认 PATH | `export PATH=/opt/amazon/efa/bin:$PATH` |
| `grep GinPlugin` 里没有 **v14** | libnccl-ofi 是 1.20.0 或更老 | 升到 installer 1.50.0（1.20.0 有 v11/v13，别被误导，§1） |
| `libibverbs` 里 0 个 `comp_cntr` | rdma-core 是 63.0 | 同上 |
| `grep COMP_CNTR` 打印 0 | grep 的是 host 的 `/usr/include/rdma/efa-abi.h`（用户态头），或 `strings` 路径写死了没找到文件 | 用 `/usr/src/efa-*/src/efa-abi.h` 和 `modinfo -n efa`（§3.2） |
| `cntr_open_ext failed: Operation not supported` at GDAKI `createContext` | CE 三要素缺一（内核模块 / libfabric / rdma-core ABI） | 跑 §4.4 的 ce_probe；再 `FI_LOG_LEVEL=warn` 看 provider-init 警告 |
| `cntr_open_ext failed: Cannot allocate memory`（不是 not supported） | CE **counter 耗尽**，不是能力问题 —— 上一轮泄漏的 rank 还占着 | 清干净残留进程（§8 规矩 3） |
| `NCCL GIN is unavailable` assert | GIN 没起来 | 先过 §4.3 的自查；日志里找 `Loaded gin plugin` |
| `GIN strong signals are required` → `DevComm setup failed on all available backends` | 只设了 `NCCL_GIN_TYPE=5`，没设 `NCCL_SYM_GIN_KERNELS_ENABLE=0` | 两个必须成对（§10.1） |
| `Cannot get gin type: ... net device type (5) is not a gin type` | 插件不支持 GDAKI，或旧 libnccl 赢了加载顺序 | 确认 `NCCL_NET_PLUGIN=ofi` 且 `dpkg -l` 是 1.21.1 |
| `has 0 allocated QPs` | **正常**，那是显式分配的 QP 数（没显式给就是 0） | 真正在用的是同组日志里的 `num_qp=11` |
| installer 在 docker build 里 `Unable to locate package tcl` | apt 缓存被提前清掉 | 同一层里先 `apt-get update`（附录 B） |
| `ModuleNotFoundError: deep_ep._C` | `WORKDIR` 是源码目录 | 换到 `/workspace`（附录 B） |
| `AssertionError: ... ninja is required` / `ModuleNotFoundError: numpy` / `pynvml` | 缺构建或运行时依赖 | `pip install ninja numpy nvidia-ml-py` |
| `third-party/fmt` 头文件找不到 | submodule 没拉 | `--recurse-submodules` |
| DeepEP 断言 NCCL 编译期/运行期版本不一致 | NCCL < 2.31 | 用 2.31+ |
| `Cannot find package: nvshmem` | 没装 NVSHMEM | `pip install nvidia-nvshmem-cu13`，只跑 v2 也需要 |
| RDMA 内存注册失败 | memlock 限制 | `docker run --ulimit memlock=-1`（§4.3） |
| `SO` GB/s 每 rank 超过 50 | 不是线速 —— 默认口径把机内流量也算进去了 | 加 `--ignore-local-traffic`（§9.0 口径） |
| GDAKI 昨天好今天坏 | 某个节点重启后回到旧驱动 | 跑 §4.4 的 ce_probe |

---

## 附录 A：镜像里有三个 NCCL 版本

会让人在第一步就误判"`--no-deps` 那步没生效"：

| 来源 | 版本 | 谁会用到它 |
|---|---|---|
| pip `nvidia-nccl-cu13` → `dist-packages/nvidia/nccl/lib/libnccl.so.2` | **2.31.2** | **DeepEP 编译期和运行期用的就是这个** |
| apt `libnccl2`（EFA installer NGC 分支拖进来的），`/usr/include/nccl.h` 也是它 | **2.28.3** | `ldconfig` 把 `libnccl.so.2` 解析到这个 |
| torch 自己编译期的 header | **2.29.7** | `torch.cuda.nccl.version()` 返回 `(2, 29, 7)` |

运行期 DeepEP 不受 2.28.3 影响：`setup.py:95` 用 `find_pkgs.find_nccl_root()` 拿 Python 环境
里的路径，再 `-I {root}/include` + `-Wl,-rpath,{root}/lib`（`setup.py:122-124`），所以
**头文件和运行期加载的都是 pip 那份 2.31.2**（用 `/proc/self/maps` 确认过活进程映射的就是它）。

**但链接期不是。** link flags 里只有 `-l:libnccl.so.2` 和那个 rpath，**没有 `-L`**，所以
链接器是靠 `/lib/x86_64-linux-gnu/libnccl.so`（apt 的 2.28.3）解析这个 `-l` 的。soname 一致
所以链得过、跑起来也对，但这是隐式依赖 —— 本目录的 Dockerfile 删掉了 apt 那两个包并补了
`-L`（附录 B）。

```bash
ldconfig -p | grep libnccl.so      # 未处理时指向 /lib/x86_64-linux-gnu/libnccl.so.2
grep -E "NCCL_(MAJOR|MINOR|PATCH)" /usr/include/nccl.h
find / -name 'libnccl.so.2' -not -path '/proc/*' 2>/dev/null   # 认清哪份是 pip 的
```

**2.28.3 < 2.30.4，没有 GIN。** 判版本用日志里的 `NCCL version 2.31.2+cuda13.3`。

---

## 附录 B：Dockerfile 的设计取舍

按构建顺序，每一条都是重建验证过的，不是推理。

1. **不能在装 EFA 之前清 apt 缓存。** installer 自己会 `apt-get install tcl libnl-3-200 ...`，
   `/var/lib/apt/lists` 被清空又没重新 `apt-get update` 就直接 `Unable to locate package tcl`。
   `rm -rf /var/lib/apt/lists/*` 要放在 EFA 那一层末尾，并在该层开头再 `apt-get update`。
2. **容器里 installer 用 `--skip-kmod --skip-limit-conf --no-verify`，但不要
   `--skip-rdma-core`** —— 容器需要 rdma-core 64.0 的 libibverbs。
3. **installer tarball 按版本号命名**，`build_image.sh` 从公共桶的版本化 URL 拉。名字带版本
   是为了让镜像自己说清用的哪套栈：`COPY` 里写 `-latest` 的话，那个对象哪天变成 1.51.0，
   同一份 Dockerfile 就会构出另一套栈而镜像里毫无记录。版本核两遍（build 前 + 镜像内 grep
   tarball 自己的 `ChangeLog.md`），通过的记成 `EP_EFA_INSTALLER`。
4. **删掉 apt 的 `libnccl2` / `libnccl-dev` 2.28.3。** installer 的 NGC 分支会顺带装上它们并
   置成 `hold`。2.28.3 < 2.30.4 ⇒ **没有 GIN**，而 `ldconfig` 恰好把 `libnccl.so.2` 解析到它。
   留着的话任何人动一下 `LD_LIBRARY_PATH` / `LD_PRELOAD` 就会静默掉到没 GIN 的那份上。
   删是安全的：`libnccl-ofi-ngc-v3` 这个 deb **没有任何 `Depends` 字段**，
   `libnccl-net-ofi.so` **只链 `libfabric.so.1`、根本不链 libnccl**（它是被 NCCL 加载的插件）。
5. **删了就必须补一个 `-L`**（两个改动是一对）。`setup.py` 的 link flags 是
   `-l:libnccl.so.2` + rpath、**没有 `-L`**，删掉 apt 包会直接
   `cannot find -l:libnccl.so.2`。补法：
   ```dockerfile
   NCCL_LIB="$(python3 -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')" \
     && test -e "$NCCL_LIB/libnccl.so.2" \
     && LIBRARY_PATH="$NCCL_LIB:${LIBRARY_PATH:-}" python3 setup.py install
   ```
6. **DeepEP 版本用 `DEEPEP_REF`（branch / tag / 裸 sha 都接受），默认钉一个 sha。**
   `git clone --depth 1 --branch <sha>` 不接受裸 sha，所以只能 init + fetch。钉 sha 的三个
   理由：§9 每个数字都对应一个确定的 tree，重建对不上时才分得清是环境问题还是代码变了；
   upstream 会**改写历史**（我们钉过的一个 sha 现在已经不在 `main` 上，钉着的时候这件事是
   可见的、漂着的时候完全隐形）；`ordered` kernel 在 EFA GDAKI 上不正确（§9.5），默认 kernel
   选择哪天变了，漂的镜像会静默跑到错的 kernel 上。想跟最新就
   `--build-arg DEEPEP_REF=main`。
7. **那个 `ADD` 不是装饰。**
   ```dockerfile
   ADD https://api.github.com/repos/amazon-contributing/DeepEP/commits/${DEEPEP_REF} /tmp/deepep-ref.json
   ```
   没有它，`RUN git fetch origin main` 是一条固定的命令字符串，Docker 会**命中旧 layer** ——
   你以为拉到了最新，实际拿的是上次的代码。**这比明着钉死更危险**：钉死是诚实地旧，缓存
   命中是假装地新。钉 sha 时该 URL 内容恒定，缓存照样有效。（走 GitHub API，未认证限速
   60 次/小时，仅用于算 cache key。）
8. **每次跑都把实际 sha 打进日志。** 镜像 tag 是人起的名字，`BUILD_REF` 是构建时
   `git rev-parse HEAD` 的结果，`run_test_ep.sh` 开跑前打印
   `=== IMAGE=... DeepEP=9c1f2511... ===`。同 tag 重建过的镜像因此不会产生无法归属的数字。
9. **`WORKDIR` 不能是 `/opt/DeepEP`**，否则 python 优先 import 源码目录里的 `deep_ep/`
   （不含 `_C.so`），报 `ModuleNotFoundError: deep_ep._C`。设成 `/workspace`，用绝对路径跑 test。
10. **其他硬依赖**：`third-party/fmt` 是 submodule（缺了 `setup.py` 和 JIT 都少 fmt 头文件）；
    必须装 `ninja`（否则 `dlink=True` 断言）、`numpy` + `nvidia-ml-py`（运行时才炸）；
    NVSHMEM 是 `setup.py` 的硬依赖，只跑 v2 也要装；NCCL 用 2.31+（低于 2.31 时 DeepEP 断言
    编译期与运行期版本严格相等）。

**怎么知道上游推了新代码**（一条命令，不用建镜像）：

```bash
PINNED=$(grep -oE '^ARG DEEPEP_REF=\S+' Dockerfile | cut -d= -f2)
TIP=$(git ls-remote https://github.com/amazon-contributing/DeepEP.git refs/heads/main | cut -f1)
[ "$PINNED" = "$TIP" ] && echo "up to date" || echo "main 已前进：$PINNED -> $TIP，重测后再 bump"
```

流程是**先量后 bump**，不是自动跟随：`main` 前进 → 跑 §9 的对照臂 → 确认在噪声内 → 改默认值。

---

## 附录 C：`ce_probe.c`

GDAKI 的成败最终取决于 `ibv_create_comp_cntr` 这一个 verb（用法见 §4.4）。

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

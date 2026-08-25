# DeepEP v2 on AWS EFA — 安装、构建、跑分

面向 `p5en.48xlarge`（8×H200，`sm_90`）×2 和 ×4。全部用已发布的包，不用编 NCCL、不用编 aws-ofi-nccl、不用换内核模块。

四步：**host 装依赖 → build 镜像 → 跑 prefill 看带宽 → 跑 decode 看延迟**。

所有版本判据和数字都是在真机上跑出来的（2026-08-25，4 台 p5en.48xlarge，installer 1.50.0 /
`deep_ep 2.1.0+ec623f3`；每个数字都是**全 rank 均值** —— 2 节点 16 rank、4 节点 32 rank，
原始日志和聚合脚本在 [`results/p5en_2n4n_20260825/`](../results/p5en_2n4n_20260825/)）。
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

GIN = GPU-Initiated Networking（NCCL Device API）；EFA 上的实现叫 `Libfabric_GDAKI`，走 `efa-direct` fabric 的 GDA ops，用硬件 completion counter（CE）承载 counting signal。

---

## 1. 实例前置条件（四条，缺一条都白干）

1. **ENI 必须在创建实例时就 `InterfaceType=efa`** —— 事后打不开，只能重建。p5en 每张卡一个 EFA ENI，用 `NetworkCardIndex=0..15` 逐个指定；正常状态是 **16 个 EFA 设备**。
   若 `lsmod | grep efa` 有模块但 `ibv_devinfo` 报 `No IB devices found`、`/dev/infiniband` 不存在 —— 就是这一条没做到。
2. **安全组自引用放通全部流量**（入站 + 出站，Source/Destination = 该安全组自身）。EFA 不走 TCP 端口。
3. **两台同 AZ + 同一个 cluster placement group。** 跨 AZ 会报 `ibv_create_ah failed with EINVAL ... Remote GID is in a different availability zone`。
4. p5en 上设备名不是 `mlx5_0`、也不连号：`rdmap85s0 86s0 87s0 88s0 / 110s0 111s0 112s0 113s0 / 135s0 136s0 137s0 138s0 / 160s0 161s0 162s0 163s0`。

最省事的做法是 capacity block + 官方 DLAMI + cluster placement group，让模板把 16 张 EFA ENI 配好。

---

## 2. Host 侧安装（每台机器都做）

### 2.1 下载 installer

`aws s3 cp` 在 CLI v2 上是坏的（HeadObject 拿 `301 Moved Permanently`，带 `--region` 也一样）。用：

```bash
curl -O https://aws-efa-installer-dev.s3.amazonaws.com/aws-efa-installer-latest.tar.gz
# 或：aws s3api get-object --bucket aws-efa-installer-dev \
#       --key aws-efa-installer-latest.tar.gz --no-sign-request --region us-east-1 \
#       aws-efa-installer-latest.tar.gz

tar xzf aws-efa-installer-latest.tar.gz          # 约 650 MB（含所有发行版的 RPM/DEB）
head -12 aws-efa-installer/ChangeLog.md          # 必须看到 ## [1.50.0]
#   - Upgrade to rdma-core 64.0amzn0 / efa driver 3.3.0
#   - Upgrade to Libfabric 2.6.0amzn1.0 / OFI NCCL Plugin 1.21.1
```

> ⚠️ **`-latest` 这个 key 不是固定版本。** 今天下到的是 1.50.0，下周可能是 1.51.0。
> 要可复现就把下到的 tarball 连同它的 `ChangeLog.md` 版本号一起存档，别只存文件名。

### 2.2 安装（host 上**不要** `--skip-kmod`，我们要的就是 efa.ko 3.3.0）

```bash
cd aws-efa-installer
sudo ./efa_installer.sh -y --no-verify        # dev 桶的包没 GPG 签名，所以要 --no-verify
sudo reboot                                   # 换内核模块必须重启
```

DLAMI 通常已自带 gdrcopy 2.5.x 和 `efa_nv_peermem`，一般只需要升 installer 本身。

### 2.3 重启后验证 host

```bash
export PATH=/opt/amazon/efa/bin:$PATH          # fi_info 默认不在 PATH 上

cat /sys/module/efa/version                    # 3.3.0g
lsmod | grep efa_nv_peermem                    # GDAKI 必需
modinfo gdrdrv | grep ^version ; ls -l /dev/gdrdrv
ibv_devinfo -l                                 # 16 个 rdmap*
fi_info | grep -c "fabric: efa-direct"         # 16

# COMP_CNTR capability 在 ABI 头里 —— 这才是 1.50.0 的真判据
grep COMP_CNTR /usr/src/efa-*/efa-abi.h 2>/dev/null || \
  strings /lib/modules/$(uname -r)/updates/efa.ko | grep -c comp_cntr   # 期望几十个，不是 0
```

**`fi_info -p efa-direct` 永远失败，别拿它当判据。** 它固定返回 `-61 (No data available)`，哪怕节点完全健康 —— `-p` 过滤的是 **provider** 名（就是 `efa`），只有 **fabric** 名叫 `efa-direct`。用 `fi_info | grep fabric`。
同样，"有 efa-direct" 也**不**代表版本够（1.49.0 就有 16 个），见附录 A。

---

## 3. 构建容器镜像

Dockerfile 见同目录 `Dockerfile`（已逐行核对过，见 §3.2）。

```bash
mkdir -p ~/deepep-docker && cd ~/deepep-docker
cp /path/to/aws-efa-installer-latest.tar.gz .     # 放进 build context，别在镜像层里重复下载
cp /path/to/Dockerfile .
docker build --progress=plain -t deepep-efa:1.50.0 .
```

约 **21.4 GB**（压缩后 7.7 GB），首次十几分钟。

### 3.1 启动容器

```bash
docker run --rm -it \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1 \
  --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  deepep-efa:1.50.0 bash
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

**1) 钉死 DeepEP commit。** 不钉的话就是 `git clone --depth 1` 拉 `main`、不带任何 pin（`git rev-parse --is-shallow-repository` = true）。今天建出来是 `2.1.0+ec623f3`，明天重建就是别的 commit，而 §5 的数字全是在 `ec623f3` 上测的 —— 会静默失配。
`git clone --depth 1 --branch <sha>` **不接受裸 sha**，只能 init + fetch：

```dockerfile
ARG DEEPEP_COMMIT=ec623f31b605b27d67c9b224d69378137f77bbe3
RUN mkdir -p /opt/DeepEP && cd /opt/DeepEP && git init -q \
    && git remote add origin https://github.com/amazon-contributing/DeepEP.git \
    && git fetch -q --depth 1 origin "$DEEPEP_COMMIT" \
    && git checkout -q FETCH_HEAD \
    && git submodule update -q --init --recursive --depth 1 && ...
```

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
  deepep-efa:1.50.0 \
  bash -lc "python3 -u /opt/DeepEP/tests/elastic/test_ep.py \
      --num-processes=8 --num-tokens=$TOKENS --hidden=7168 --num-topk=8 \
      --num-experts=256 --num-sms=12 --allow-hybrid-mode=1 \
      --prefer-overlap-with-compute=0 --test-first-only"
```

**`--num-sms` 在 EFA 上必须显式给**，不给会直接崩：

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

**而且 `--num-sms` 不是自由参数。** 它会连带改实际分配的 QP 数，且**非单调**（实测 0→17 QP、12→5、24→10），落到 `num_qps < num_ranks` 的档位会**直接挂死**（GIN auto-tuner 打完那几行就再无输出，600 s 超时）。16 rank / 12 SM（`num_qp=11`）和 24 SM 这两档都是好的。

**推荐工作点：2 节点和 4 节点都用 24 SM。** 下表 8192 tok、GIN type 5、`--test-first-only`、
未开 `EP_BUFFER_DEBUG`，全 rank 均值（完整数据见
[`results/p5en_2n4n_20260825/summary.txt`](../results/p5en_2n4n_20260825/summary.txt) TABLE 5/6）：

| SM | 2 节点 dispatch | 2 节点 reduced combine | 2 节点合计 | 4 节点 dispatch | 4 节点 reduced combine | 4 节点合计 |
|---|---|---|---|---|---|---|
| 6 | 2290.5 µs | 7371.6 µs | 9662.1 µs | 4031.7 µs | 9494.3 µs | 13526.0 µs |
| 12 | 1502.9 µs | 4226.0 µs | 5728.9 µs | 3955.3 µs | 7963.6 µs | 11918.9 µs |
| 16 | 1510.6 µs | 3568.1 µs | 5078.7 µs | — | — | — |
| **24** | 1535.7 µs | **3364.3 µs** | **4900.0 µs** | 3972.6 µs | **7709.8 µs** | **11682.4 µs** |
| 32 | 1576.5 µs | 3486.4 µs | 5062.9 µs | — | — | — |

- **dispatch 从 12 到 24 SM 基本是平的**（2 节点 +2.2%、4 节点 +0.4%），付钱的是
  reduced combine。所以拿这点 dispatch 换 combine 总是划算：2 节点层总时间 −14.5%，
  4 节点 −2.0%。
- **6 SM 在任何规模上都是错的选择**：2 节点差 52%、4 节点差 13.5%。
- **128 tok（decode）同样选 24 SM**：2 节点 dispatch 169.4 → 147.3 µs（−13.0%）、
  reduced combine 179.0 → 160.4 µs（−10.4%）；4 节点 dispatch 在 6/12/24 SM 上是平的
  （181–185 µs，散布 2.4%），只有 reduced combine 动，24 SM 比 12 SM 好 5.7%
  （253.5 → 239.1 µs）。这条适用于未打 §5.5 那两个 PR 的代码；打了之后 2 节点 decode
  反而 12 SM 更好。

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

---

## 5. 实测结果（`p5en.48xlarge` × 2 和 × 4，2026-08-25）

环境：Ubuntu 24.04，driver 595.91.07，各 8×H200 + 16 EFA，同 AZ 同 cluster placement group，
installer 1.50.0（production tarball，md5 `e5a5178944b1f1112f3b2eb3b15ca5a7`，efa.ko 3.3.0g）。
容器 `deepep-efa:1.50.0`（torch 2.13.0+cu130 / nccl 2.31.2+cuda13.3 / `deep_ep 2.1.0+ec623f3`）。
16 卡和 32 卡 `test_ep.py` 都 **exit 0，正确性检查全过**；日志两侧都报
`Selected provider is efa, fabric is efa-direct`。

**下面所有数字都在 GIN type 5 上**（`NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0`，见 §6），
`--test-first-only`，未开 `EP_BUFFER_DEBUG`，SM = 24（§4.2 的工作点）。

**口径**（每次报速率都要一起报，只报 GB/s 曾经把结论弄反过）：

- **时间是全 rank 均值** —— 2 节点 16 个 rank、4 节点 32 个 rank，从每台机器的日志里汇总。
  单台机器的均值**不能**代替它：combine 是按节点分层的（见本节末），单节点均值能偏 12%。
- `SU` = 打印的每 rank 字节 ÷ 时间。
- `SO` **不是线速** —— 没加 `--ignore-local-traffic` 时机内流量也算在内。
  **线速占比 = `SO × (N−1)/N ÷ 50 GB/s`**（p5en 每 GPU 50 GB/s = 16×200 Gb/s ÷ 8）；
  N=2 时线速占比数值上等于 SO，N=4 时是 SO × 1.5。
- 每 rank scale-out 字节，8192 tok：dispatch 2 节点 395.9–402.4 MB / 4 节点 441.4–447.7 MB，
  combine 759.7–772.1 / 847.0–859.0 MB。128 tok：dispatch 5.6–6.3 / 5.8–6.4 MB，
  combine 10.8–12.1 / 11.1–12.3 MB。

### 5.1 Prefill（`--num-tokens=8192`）— 带宽

| op | 2 节点 时间 | SO | SU | 4 节点 时间 | SO | SU |
|---|---|---|---|---|---|---|
| dispatch | **1535.6 µs** | 79–80 | 257–263 | **3972.6 µs** | 55–56 | 111–113 |
| expanded dispatch | 1537.3 µs | 79–80 | 257–263 | 3970.7 µs | 55–56 | 111–113 |
| cached dispatch | 1620.7 µs | 75–76 | 244–249 | 4256.6 µs | 51–52 | 103–106 |
| combine | 3529.1 µs | 62–73 | 202–239 | 7702.1 µs | 51–57 | 103–115 |
| reduced combine | 3364.3 µs | 64–79 | 209–259 | 7710.1 µs | 51–57 | 103–115 |

（2 节点 3 轮、4 节点 2 轮，先跨 rank 再跨 rep 取均值。）

**dispatch 在两个规模上都跑在线速的 81–84%**：2 节点 81%（SO 79–80 ≡ 线速%），
4 节点 84%（56 × 1.5）。跨节点比例从 1/2 升到 3/4，所以 4 节点每 rank 的 scale-out
字节多 11%、时间是 2 节点的 2.59×，而线速利用率反而更高。

### 5.2 Decode（`--num-tokens=128`）— 延迟

这个尺寸每 rank 只有 5.6–6.4 MB，是**消息率受限**不是带宽受限，只看时间：

| op | 2 节点 | 4 节点 |
|---|---|---|
| dispatch | **147.3 µs** | **184.7 µs** |
| expanded dispatch | 146.6 µs | 184.0 µs |
| cached dispatch | 135.8 µs | 184.2 µs |
| combine | 151.9 µs | 236.5 µs |
| reduced combine | 160.4 µs | 239.1 µs |

从 2 节点到 4 节点 dispatch 只涨 25%，**这是 type 5 的特性**：type-2 代理后端在同一组机器上
是 365 → 1001 µs（2.7×）。§5.5 的两个 PR 能把 2 节点 dispatch 再压到 106 µs。

### 5.3 combine 按节点分层 —— 这决定了怎么汇总

dispatch 在 rank 之间是均匀的（2 节点 24 SM 8192 tok：16 个 rank 全在 1531–1538 µs），
combine 和 reduced combine **不是** —— 它们按机器整齐地分成两层：

| 2 节点 / 24 SM / 8192 tok，单轮，各取本机 8 个 rank 的均值 | node1 | node2 |
|---|---|---|
| dispatch | 1536.0 µs | 1535.0 µs（差 0.1%） |
| combine | 3302.9 µs | 3763.9 µs（差 14%） |
| reduced combine | 3091.7 µs | 3653.9 µs（差 18%） |

**哪台机器是慢的那台在不同次运行里会翻转。** 所以单节点的 combine 均值会落在 ~3100 或
~3670 µs，取决于你读的是哪份日志 —— 看起来像双峰行为，其实不是：汇总全部 16 个 rank
之后，24 SM 那三轮彼此只差 1.7%（3391.5 / 3335.7 / 3365.5 µs）。**报全 rank。**

### 5.4 和从源码手编整套栈的对比

`deepep-v2-efa-gdaki-h200:dev-7a6059a3` 是手编 NCCL `2.30.7-1` + 手编 `--enable-gdaki`
aws-ofi-nccl + DeepEP `7a6059a3` 的栈，多带一个 `--skip-check`。同一批机器、同一天、
12 SM、两侧都在 type 5 上：

| arm | dispatch | cached dispatch | combine | reduced combine |
|---|---|---|---|---|
| 2 节点 8192 tok 手编 pin | 1515.0 µs | 1749.0 µs | 3643.4 µs | 4164.6 µs |
| 2 节点 8192 tok 正式包 | 1502.9 µs | **1591.0 µs** | 3629.9 µs | 4226.0 µs |
| 2 节点 128 tok 手编 pin | 303.9 µs | 271.6 µs | 176.4 µs | 188.8 µs |
| 2 节点 128 tok 正式包 | **169.4 µs** | 165.9 µs | 162.7 µs | 179.0 µs |
| 4 节点 8192 tok 手编 pin | 3961.0 µs | 4710.3 µs | 8004.0 µs | 8079.6 µs |
| 4 节点 8192 tok 正式包 | 3955.3 µs | **4239.4 µs** | 7850.0 µs | 7963.6 µs |
| 4 节点 128 tok 手编 pin | 320.8 µs | 284.3 µs | 276.1 µs | 280.4 µs |
| 4 节点 128 tok 正式包 | **184.5 µs** | 180.0 µs | 243.6 µs | 253.5 µs |

prefill dispatch 是平手（差 0.1–0.8%），正式包在 **cached dispatch 上快 9–10%、
decode dispatch 上快 1.74–1.79×**。**没有任何性能理由去从源码编这套栈。**

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
| 手编 pin `7a6059a3`，`--skip-check` | 1515.0 µs | 80.6 | 80.6 |
| 正式包 + 完整 5 个 route-B 变量 | 1505.4 µs | 81.0 | 81.0 |
| **正式包 + `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`** | **1502.9 µs** | 81.2 | 81.2 |
| `main` `8e7b42e` + 那一对 | 1502.0 µs | 81.0 | 81.0 |

`线速% = SO × (N−1)/N ÷ 50 GB/s`，N=2 时数值上等于 SO。**这一整段差距全部由那两个环境变量
解释** —— 不是 QP fan-out，也不是 `--skip-check`：另外三个 route-B 变量叠上去毫无增益，
`main` 和 `ec623f3` 无法区分。QP flag 在 type 5 之上对 plain dispatch 是零影响
（1502.9 → 1508.5 µs），它真正管的是 **cached dispatch，而且 5 个 context 更差 9.6%**
（1591.0 → 1743.9 µs）；4 节点 decode 上更差 19.6%（1001.2 → 1197.5 µs）。**保持默认 11。**

`OFI_NCCL_GIN_STRONG_SIGNAL=1` 单独跑在 128 tok 上是 **750.4 µs**（32 个 rank 散在
371–1130 µs），即主动变坏 —— 和 unordered kernel 只要 weak signal 一致。

### 5.5 两个待合的 PR：decode dispatch 再减 33%

decode 上有两个互相独立的杠杆：GIN 后端（只改 env，§6）和 dispatch 的 part 几何（两个待合的
PR）。两者**可以叠加**。镜像用 PR #2 的 head `b097b03`，PR #1 是它的祖先，所以这一个 SHA
就是"两个 PR 叠加"这个 arm。

| PR（基于 `main`） | 内容 |
|---|---|
| [amazon-contributing/DeepEP#1](https://github.com/amazon-contributing/DeepEP/pull/1) | 把 `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` / `EP_MIN_TOKENS_PER_PART` 转发给 JIT，不改任何默认值 |
| [amazon-contributing/DeepEP#2](https://github.com/amazon-contributing/DeepEP/pull/2) | 加 `kMinTokensPerPart`（默认 15，`EP_MIN_TOKENS_PER_PART` 可覆盖），`kNumParts = min(预算, 每 channel token 数 / 15)` |

**2 节点 / 16 rank / 12 SM / 128 tok**（全 rank 均值）：

| arm | dispatch | 相对同后端 | combine | reduced combine |
|---|---|---|---|---|
| 未打补丁，type 2 | 365.1 µs | 1.00× | 175.0 µs | 192.4 µs |
| #1 + #2，type 2 | 237.1 µs | **−35.1%** | 176.3 | 193.6 |
| 未打补丁，type 5 | 169.4 µs | 1.00× | 162.7 | 179.0 |
| #1 + #2，type 5 | 112.7 µs | **−33.5%** | 162.4 | 178.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`，type 5 | **106.4 µs** | **−37.2%** | 162.2 | 178.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`，type 5 | 171.5 µs | 关掉 clamp 的对照 | 162.6 | 179.0 |

**4 节点 / 32 rank / 12 SM / 128 tok：**

| arm | dispatch | 相对同后端 | combine | reduced combine |
|---|---|---|---|---|
| 未打补丁，type 2 | 1001.2 µs | 1.00× | 343.3 µs | 350.7 µs |
| #1 + #2，type 2 | 627.2 µs | **−37.4%** | 335.1 | 347.8 |
| 未打补丁，type 5 | 184.5 µs | 1.00× | 243.6 | 253.5 |
| #1 + #2，type 5 | 169.5 µs | **−8.1%** | 243.7 | 253.1 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`，type 5 | **155.9 µs** | **−15.5%** | 243.9 | 251.7 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`，type 5 | 184.1 µs | 关掉 clamp 的对照 | 243.3 | 253.1 |

三条结论：

1. **两个杠杆互相独立、可以叠加。** 叠起来 2 节点 decode dispatch 从 365.1 → 106.4 µs，
   **3.43×** —— 一对环境变量加两个 commit。
2. **combine 和 reduced combine 在每一行里都动不到 1%。** 这就是"机制在 dispatch 内部的
   part 几何、不在网络"的证据：EFA 两边干的活一样多。expanded / cached dispatch 跟着
   plain dispatch 走。
3. **clamp 的收益在 4 节点 type 5 上崩掉了** —— −8.1% 而不是 −33.5%，而且关掉 clamp 的对照
   （184.1 µs）落在未打补丁（184.5 µs）的噪声里，说明是 clamp 本身不再付钱、而不是 PR 里的
   别的东西。type-5 4 节点 decode dispatch 有一个 **156–185 µs 的地板**，part 几何摸不到；
   `EP_NUM_SUB_PARTS=1` 走得最远，也就到这里。**原因未知 —— 不要把 2 节点那个比例外推到
   更大的集群。**

**为什么 part 几何会有代价**（combine 完全不动就是证据）：
`kNumParts`（一个 channel 的 token 分几次 `flush_part` put 发出去）只由 `compute_part_allocation()` 决定，而它只在 GIN indexed-signal 预算紧时**从上面**压；预算恰恰在 channel 里 token 最少时最松，所以 decode 必然落到 `kMaxParts` —— 轴的最坏一端。128 token / 12 SM 下每 channel 只有 3 个 token 却被描述成 4 part × 1 token：最后一个 part 永远是空的，3 个 token 发成 3 次单 token put 而不是 1 次 3-token put。sub-part 早有这两道保护（clamp 到 `kBatchSize`、`EP_SM100_MIN_SUB_TOKENS`），part 一道都没有。

**取舍**：只关心 prefill —— `main` 直接用，Dockerfile 不用改。这两个 PR 在 prefill 上是噪声
（2 节点 24 SM 8192 tok：未打补丁 1535.7 vs 1536.0 µs；4 节点 12 SM 8192 tok：3955.3 vs 3955.0 µs）。
**要发 decode / 小 token 的数字 —— 合并前自己 cherry-pick 这两个**，同时无论如何都要设 type-5
那一对。#1 不改任何默认值，所以必须和 #2 叠加才有收益。合并后 #2 的默认值 15 自动生效，不需要设
环境变量；要回未 clamp 的几何做对照给 `EP_MIN_TOKENS_PER_PART=1`（它是**短路**回旧值 ——
除以 1 是第三种几何，不是对照组）。

**打了 PR 之后 2 节点 decode dispatch 反而 12 SM 更好**（112.7 vs 145.3 µs），所以 §4.2 那条
"2 节点 decode 用 24 SM"适用于未打补丁的代码；4 节点上两档在 dispatch + reduced combine 上打平
（422.6 vs 422.4 µs）。

---

## 6. 环境变量速查

| 变量 | 用的值 | 说明 |
|---|---|---|
| `EP_HYBRID_KERNEL` | `unordered`（默认） | **EFA 上必须保持 unordered。** 设成 `ordered` 走上游 kernel，要求 VA signal + strong signal，EFA 不支持 |
| `EP_NIC_NAME` | `rdmap85s0` | 上游默认 `mlx5_0` 是 IB 的。用 `ibv_devinfo -l` 查实际名。它只喂 `get_rdma_gbs()`，而那条路在 EFA 上本来就坏，所以给对给错都不如直接给 `--num-sms` |
| `NCCL_NET_PLUGIN` | `ofi` | 用名字解析，别写死路径 |
| `FI_PROVIDER` | `efa` | 是 `efa` 不是 `efa-direct`，fabric 由插件自己选 |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | |
| `NCCL_GIN_TYPE` | **`5`** | **必须显式设，否则默认走 type 2 代理后端**，prefill 慢 ~9%、decode 慢 2.2–5.4×。必须和下一行成对设，单独设会崩。详见本节末 |
| `NCCL_SYM_GIN_KERNELS_ENABLE` | **`0`** | 同上。sym GIN kernel 要 strong signal，GDAKI 没有 |
| `EP_BUFFER_DEBUG` | **只在调试时** `1` | 打 NCCL 版本 / QP 数 / GIN layout。**但它在计时区间里 printf** —— `csrc/elastic/buffer.hpp:1151` 在 dispatch 的 host 轮询循环里拼 stringstream 打 "CPU side received count"。而且 `deepep-v2-efa-gdaki-b200` 的启动脚本不转发它、`run_test_ep.sh` 转发，所以**只有一侧设了的跨镜像对比是被污染的**（它的代价我们没能从 type-2 那一侧自身的轮间散布里分离出来，所以不给数字）。确认完 layout 就关掉 |
| `EXTRA_ENV` | `"NAME=VALUE …"` | `run_test_ep.sh` 的一次性 env 钩子，用来做单变量 A/B（上面那对 GIN 变量就是这么传的） |
| `EP_JIT_CACHE_DIR` | `/root/.deep_ep` | 做 A/B 时**每个变体必须用不同目录**（见 §7） |
| `EP_DISABLE_GIN` | `0` | 设 1 关掉 GIN，用来隔离问题 |

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
| 4 节点 8192 tok dispatch | 4316.2 µs / 51.5 | **3955.3 µs / 56.0** | −8.4% |
| 4 节点 128 tok dispatch | 1001.2 µs | **184.5 µs** | **5.43×** |

**差距随规模变大**：type-2 的 decode dispatch 从 2 节点到 4 节点是 365 → 1001 µs（2.7×），
type 5 只是 169.4 → 184.5 µs（1.09×）。type-2 的 4 节点 decode 还很不稳定，五轮散在
833 / 987 / 1016 / 1045 / 1083 µs；type 5 三轮 184.3 / 184.5 / 184.8，32 个 rank 全在
183.4–185.9 µs 之内。

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
**750.4 µs**（32 rank 散在 371–1130 µs，主动变坏）、这一对 169.4 µs、完整 5 个 route-B 变量
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

1. **每个变体独立 `EP_JIT_CACHE_DIR`。** GIN 的 device 侧头文件会编进 DeepEP 的 JIT kernel，共享缓存会悄悄复用对方的 kernel。"A/B 完全没差别"最常见的原因就是这个。
2. **交错跑 rep**（`A B A B ...`），绝不能先跑完所有 A 再跑 B —— 否则热漂移和集群漂移全算到其中一个变体头上。
3. **`rc=0` 不是健康检查。** 上一轮崩掉的 rank 可能活着、每张 GPU 占着 ~48 GB。下一轮如果泄漏不大、GDAKI init 还能过，会跑完、`rc=0`、输出完整，然后**延迟虚高约 2×**，日志里没有任何提示（4 节点上见过 combine 连续四轮 7.7→12.5→16.5→19.6 ms 而每轮都报成功，显存同步爬 0→8.9→29.7→43 GB）。所以每轮之间必须断言 `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 全是 0 MiB，并且**每轮换一个 `MASTER_PORT`**（`TIME_WAIT` 表现为 rendezvous 卡死）。用 docker 跑省事 —— `docker rm -f` 会带走整棵进程树。
4. **报数字要报全 rank，并且带上口径。** rank 之间是系统性不同的，而且差异往往**按节点分层**（§5.3 里 combine 就是一整台机器慢 14%）。单个 rank、单台机器的区间都不是这一轮的数字。同时永远把时间（µs）和带宽（GB/s）一起报，并说明分母 —— 只报 GB/s 曾经把结论弄反过。

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
ibv_devinfo -l ; fi_info | grep -c "fabric: efa-direct"                             # 16 / 16
python3 -c "import deep_ep, torch; print(deep_ep.__version__, torch.__version__)"
```

`grep GinPlugin` 里没有 v14 = 插件是 1.20.0 或更老，后面全白跑。`comp_cntr` 为 0 = rdma-core 是 63.0。

### 一锤定音的 CE 探针

GDAKI 的成败最终取决于 `ibv_create_comp_cntr` 这一个 verb。**在容器里**跑最有意义 —— 它同时验证 host 内核模块和容器里那份 rdma-core，正好是依赖链最底下两层。健康节点上 16 个 `rdmap*` 全部 `CE OK`。驱动状态是**每节点**的：同一批实例里刚重启的机器可能回到旧模块，某天忽然挂了先跑这个。

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
| `ZeroDivisionError` in `get_theoretical_num_sms` | `--num-sms 0` 走 `ibstat` 自动探测，EFA 上必失败 | 显式给 `--num-sms 12`（§4.2） |
| `Failed to get RDMA connection speed:` | 同上，`ibstat` 看不到 EFA 设备 | 单机无害；多机必须给 `--num-sms` |
| 改了 `--num-sms` 后整轮**无输出挂死** | `num_sms` 连带改 `num_allocated_qps` 且**非单调**（0→17、12→5、24→10），落到 `num_qps < num_ranks` 就挂 | 看日志里 `num_qp=` 是否 ≥ rank 数；固定在官方三个工作点（§4.2） |
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

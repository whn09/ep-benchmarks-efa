---
marp: true
theme: default
paginate: true
size: 16:9
header: "DeepEP V2 on AWS EFA"
footer: "p5en.48xlarge × 2 / 4 · 2026-08"
style: |
  section { font-size: 22px; }
  h1 { color: #1f4e79; }
  h2 { color: #2e75b6; }
  table { font-size: 20px; }
  code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
---

# DeepEP V2 on AWS EFA
## 原理 · 安装 · 性能

- MoE 训推的 all-to-all（dispatch + combine）
- 用**发布包**跑，不编 NCCL、不编 aws-ofi-nccl、不换内核模块
- p5en × 2，decode dispatch **106 µs**（AWS baseline 365 µs → **3.4 ×**）

---

# 1 · DeepEP V2 是什么

- **MoE 的 GPU all-to-all 内核**：dispatch 把 token 路由到 expert，combine 收回来
- V1 → V2 换了内网后端：**NVSHMEM → NCCL GIN**（GPU-Initiated Networking）
  - GIN 让 CUDA kernel 直接发 RDMA，不用 CPU proxy
- V2 SM 占用从 24 降到 **4–6 SM**，同带宽下更省算力
- 统一 `ElasticBuffer` API，SM 数解析式算出、不再 auto-tune

**为什么关心 EFA**：p5en / p6-b300 都是 EFA 网络，不是 IB / RoCE，需要专门的 GDA 通路

---

# 2 · EFA-GDA 依赖链

**GDA = GPU Direct Async**：NIC WQ 映射到 GPU 显存，kernel 自己 ring doorbell，CPU 全程不介入。

| 层 | 版本 | 判据 |
|---|---|---|
| DeepEP V2 unordered kernels | `amazon-contributing/DeepEP` | — |
| NCCL GIN | ≥ **2.30.4**（用 2.31.2） | 有 `nccl_device.h` |
| aws-ofi-nccl | **1.21.1** | `nm` 里有 `ncclGinPlugin_v14` |
| libfabric | 2.6.0amzn1.0 | 16 个 `fabric: efa-direct` |
| rdma-core | 64.0amzn0 | 20 个 `comp_cntr` 符号 |
| **efa 内核驱动** | **3.3.0** | `EFA_QUERY_DEVICE_CAPS_COMP_CNTR` |

**加粗四层由 EFA installer 1.50.0 一次给齐** —— 这就是版本门槛的全部原因。

---

# 3 · 两个必须的环境变量

installer 1.50.0 装完，`libnccl-net-ofi.so` 注册**两个** GIN backend：type 2（CPU proxy）和 type 5（GDAKI）。**默认走 type 2**，且日志两边都会打 `Loaded gin plugin Libfabric_GDAKI (v14)`（这行不能作为判据）。

```bash
export NCCL_GIN_TYPE=5
export NCCL_SYM_GIN_KERNELS_ENABLE=0   # 缺这个会 crash
```

| dispatch | type 2（默认） | **type 5** | 加速比 |
|---|---:|---:|---:|
| 2N / 16 rank / 8192 tok | 1644 µs / 74 GB/s | **1503 µs / 81** | 1.09× |
| 2N / 128 tok | 365 µs | **169 µs** | **2.16×** |
| 4N / 32 rank / 8192 tok | 4315 µs | **3955 µs / 84% wire** | 1.09× |
| 4N / 128 tok | 1003 µs | **184 µs** | **5.44×** |

**规模越大差距越大** —— 决定要不要设这两个变量的最重要理由。

---

# 4 · 为什么 GDA 比 CPU proxy 快得多

**数据链路差**：

| 栈 | 路径 | 每 rank CPU 代价 |
|---|---|---|
| DeepEP V1 + NVSHMEM (EFA) | `kernel` → NVSHMEM proxy 线程 → libfabric → NIC | CPU 唤醒 + syscall + verbs |
| UCCL-EP | `kernel` → 4 个 Rust proxy 线程/rank → ibverbs → NIC | **4 线程/rank × 8 = 32 线程/节点** |
| **DeepEP V2 + GDAKI (type 5)** | `kernel` 直接写 **64 B WQE** + ring MMIO doorbell (`FI_EFA_GDA_OPS`) | **0 线程** |

**收益的两个来源**：
1. **延迟**：省掉 CPU 唤醒 + syscall + verbs 三段
2. **规模弹性**：CPU proxy 走每条链需 handoff 序列化；GPU 把 N 个 flush 分成 N 个 thread 并行发

**代价的量化（同一 128-tok decode，type-2 proxy vs type-5 GDA）**：
- 2N：dispatch 365 → 169 µs（**2.16×**），4N：1003 → 184 µs（**5.44×**）
- **差距随规模放大** —— GPU 侧的 fan-out 是常数代价，CPU proxy 侧是线性代价

---

# 5 · Host 安装（每台机器做一次）

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
tar xzf aws-efa-installer-1.50.0.tar.gz
sudo ./aws-efa-installer/efa_installer.sh -y --no-verify
```

**验证**：
- `modinfo efa | grep ^version` → 3.3.0
- `ibv_devinfo -l` → 16 张 EFA 设备（p5en 每 GPU 2 张）
- 我们提供的 CE 探针 `ce_probe.c`：`ibv_create_comp_cntr` 成功 = GDAKI 能起

**坑**：p6-b300 出厂带 installer 1.47.0（`efa.ko` 3.0.0，无 `GinPlugin` 符号），GDAKI 直接起不来，必须先升 1.50.0。

---

# 6 · 构建镜像（每台机器一条命令）

```bash
git clone https://github.com/whn09/ep-benchmarks-efa.git ~/work/ep-benchmarks-efa
cd ~/work/ep-benchmarks-efa/deepep-v2-efa-official
./build_image.sh sm90                    # p5en / H200
./build_image.sh sm103                   # p6-b300 / B300（需要 CUDA ≥ 13.3）
```

镜像里干了什么：
1. 装 EFA installer 1.50.0 用户态（`--skip-kmod`；内核态在 host）
2. **构建期 gate**：`nm libnccl-net-ofi.so | grep ncclGinPlugin_v14`（一份 pre-GA tarball 里 aws-ofi-nccl 是 1.20.0，没有 v14；ChangeLog 判不出来）
3. 删 apt 的 libnccl 2.28.3，装 pip 的 2.31.2 + nvshmem 3.7.2
4. build DeepEP，默认跟 `main`，镜像 tag 自带 sha（`sm90-<sha7>`）；`sm90-bfbdd15` 是 PR #1 + #2 那条臂

---

# 7 · 跑第一个 cell

```bash
# leader（在 P5EN-1 上）
MASTER_IP=$(hostname -I | awk '{print $1}')   # ← 用私网 v4，别用 hostname
./run_test_ep.sh 0 "$MASTER_IP"

# worker（在 P5EN-2 上，几秒内启动）
./run_test_ep.sh 1 172.31.14.20
```

**坑**：`ip-172-31-14-20` 这种 hostname 在 EC2 上会同时解析出 IPv4 和 **link-local IPv6 `fe80::…`**；libuv 尝试 v6 时 `errno 22 (EINVAL)`。

日志里应该看到：
```
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)      # <- 决定性一行
> #SM: 12, #QPs: 11/11
```

---

# 8 · Campaign 出正式表

```bash
NODES="P5EN-1 P5EN-2" ./run_campaign.sh
```

- 7 cell × 3 rep × 2 node = **42 份日志**（`prs` 镜像不在时只跑 4 个 official cell = 24 份）
- Reps 是**轮转**跑（一 rep 跑完所有 cell 再进下一 rep），不是分块 —— 慢漂移不会被误认成 arm 效应
- 每个 cell 独立 `MASTER_PORT`（防止上一次残留 socket 占端口）
- 每个 log 名字自带所有变量：`arm_nodes_sms_tokens_knob_dbg_gin_rep_node.log`
- **验收**：`./verify_run.sh logs/*.log` → `=== no FAILs`

**规矩**：单节点的均值**不能**代替全 rank 均值 —— combine 按机器分层，偏差可 12–18%。

---

# 9 · 性能：Prefill (12 SM = 默认, 8192 tok, GIN type 5)

| op | 2N 时间 | SO GB/s | 4N 时间 | SO GB/s |
|---|---:|---:|---:|---:|
| dispatch | **1502.9 µs** | 81–82 | **3955.3 µs** | 56 |
| combine | 3602.5 µs | 60–73 | 7842.6 µs | 49–58 |
| reduced combine | 4237.9 µs | 53–60 | 7943.2 µs | 46–58 |

- **dispatch 在两个规模都跑到线速的 81–84%**（2N 81.2%，4N 84.0%）
- **4N 利用率反而更高**：每 rank 真跨机字节从 61 MB → 166 MB（×2.72），时间只涨到 2.63×
- combine 时间跨 rank 分布 12–23% —— 那是 **机器分层**（node-layered），不是网络抖动；4N 三轮全 rank 均值只差 0.6%（redComb 7905.6 / 7972.2 / 7951.9 µs）
- **24 SM 是一根轴，不是默认**：2N 用 +2.2% 的 dispatch 换 −20.7% 的 reduced combine（一层总时间 −14.7%），但 SM 预算翻倍，且打了第 10 页那两个 PR 后 decode 反而 12 SM 更好

---

# 10 · 性能：Decode + PR #1 + #2（12 SM, 128 tok, type 5）

**decode 的两个杠杆可以叠加：**
1. `NCCL_GIN_TYPE=5`（后端）
2. Part 几何（DeepEP `amazon-contributing` PR **#1** + **#2**）

| arm | 2N dispatch | Δ | 4N dispatch | Δ |
|---|---:|---:|---:|---:|
| 未打补丁 type 2 | 365 µs | 1.00× | 1003 µs | 1.00× |
| 未打补丁 type 5 | 169 µs | −54% | 184 µs | **−82%** |
| +PR #1 #2 | 113 µs | −69% | 170 µs | −83% |
| + `EP_NUM_SUB_PARTS=1` | **106 µs** | **−71%** | **156 µs** | **−84%** |

- 2N：一对 env var + 两个 commit 把 decode dispatch 从 365 µs 干到 **106 µs（3.43×）**
- 4N：clamp 单独 −8%，但绝对时间几乎与 2N 持平（**GDAKI 规模弹性极好**）

---

# 11 · 跨栈对比：V1 / UCCL-EP / V2 —— 同 EFA 硬件

**Prefill 吞吐（每 rank 有效带宽，GB/s，大=好）**

| 栈 | 后端 | p5en dispatch | p5en combine | b300 dispatch | b300 combine |
|---|---|---:|---:|---:|---:|
| DeepEP V1 + Amazon NVSHMEM | CPU proxy | 62.54 | 58.48 | 109.84 | 101.72 |
| UCCL-EP | CPU proxy（32 线程/节点） | 60.64 | 17.11 | 90.03 | 58.99 |
| **DeepEP V2 + EFA GDAKI** | **GPU-initiated** | **81.25** | **65.75** | **125** | **131** |

**Decode 延迟（p5en dispatch + combine, 128 tok, µs, 小=好）**

| 栈 | p5en dispatch | p5en combine | b300 dispatch | b300 combine |
|---|---:|---:|---:|---:|
| DeepEP V1 + NVSHMEM (EFA) | 602 | 561 | 691 | 416 |
| UCCL-EP（pplx-style p50） | 212 | 324 | 103–136 | 229–367 |
| **DeepEP V2 + GDAKI** | **151.6** | **189.6** | 200.4 | 160.1 |

**一句话**：**V2 + GDAKI 是唯一同时打赢 V1（+30% 吞吐）和 UCCL-EP（−28% 延迟）的栈**，而且 CPU 占用是 **0 vs 32 线程/节点**。数据来源：`ep-benchmarks-efa/README.md § Side-by-side results`。

---

# 12 · 三个坑

**1. 版本号判据不够** —— 两份都叫 `aws-efa-installer-1.50.0.tar.gz` 的 tarball 里，aws-ofi-nccl 一份是 1.21.1（有 v14），一份是 1.20.0（没有）。判据用 **`nm` 上的 `ncclGinPlugin_v14` 符号**。

**2. hostname 会解析出 link-local v6** —— rendezvous 用 hostname 会 `errno 22`。用**私网 IPv4** 直连。

**3. `torch.multiprocessing.spawn` 的 signal handling 差** —— 容器 PID 1 是 python 时，SIGTERM 到达前 worker 可能已经在 rendezvous 上把 `MASTER_PORT` 挂住。给 `docker run` 加 **`--init`**（tini 当 PID 1 正确 reap + 转发 signal）。

---

## 参考

- 完整方法 + 数字：`docs/runbook_zh.md`（本目录）
- 复现脚本：`run_test_ep.sh` / `run_campaign.sh`
- 待合 PR：[`amazon-contributing/DeepEP#1`](https://github.com/amazon-contributing/DeepEP/pull/1) / [`#2`](https://github.com/amazon-contributing/DeepEP/pull/2)
- 上游 install 参考：`awslabs/awsome-distributed-ai` `micro-benchmarks/expert-parallelism/deepep-v2-benchmark`

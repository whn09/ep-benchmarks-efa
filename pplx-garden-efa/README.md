# pplx-garden-efa — Perplexity AI's pplx-garden P2P all-to-all on AWS EFA

`pplx-garden` is Perplexity AI's open-source P2P MoE dispatch/combine
library, built on top of their custom Rust RDMA fabric library
(`fabric-lib`). Unlike DeepEP (NVSHMEM/NCCL backends) and UCCL-EP
(ibverbs-direct), pplx-garden talks to libfabric directly and aggregates
multiple NICs per GPU. The upstream paper:
[RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656).

This image wraps upstream's `docker/dev.Dockerfile` and adds a wheel
build step so `python3 -m benchmarks.bench_all_to_all` runs out of the
box. Validated on:
- 2× **p5.48xlarge** (H100 80GB × 8, **32 EFA v1 NICs × 100 Gbps**), us-east-2, 2026-05-16
- 2× **p5en.48xlarge** (H200 80GB × 8, **16 EFA v2 NICs × 200 Gbps**), us-east-2, 2026-05-16

**TL;DR perf**:
- **p5en.48xlarge** decode EP=16: dispatch **224 µs**, combine **246 µs**
  — matches upstream README's p5en numbers (215 / 242) within 5 %.
- **p5.48xlarge** decode EP=16: dispatch 402 µs, combine 517 µs (p50).
- prefill (4096 tokens) on p5en: dispatch **3128 µs**, combine **5386 µs**
  — also matches upstream (3197 / 5379).

See [§4 Validated numbers](#4-validated-numbers).

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `pplx-garden-efa:dev` |
| `run_a2a.sh` | 2-node launcher for `benchmarks.bench_all_to_all` |
| `monitor_efa.sh` | Per-NIC EFA bandwidth snapshot |
| `sample_efa_bw.sh` | Time-series per-NIC sampler |

## Host prerequisites

Same as the other images in this repo: NVIDIA driver 595+, EFA hardware,
`/dev/gdrdrv` present, intra-VPC SG with self-reference rule. See
`../deepep-v1-efa/README.md` for the gdrdrv recovery snippet.

Additionally, pplx-garden uses `pidfd_getfd` for cross-process FD
sharing, which needs `SYS_PTRACE` and `SYS_ADMIN` capabilities. These
are added by `run_a2a.sh`.

---

## 1. Build the image (per node)

```bash
rsync -avz /Users/henanwan/Documents/workspace/moonshot/pplx-garden-efa/ \
  P5EN-1:~/work/pplx-garden-efa/
rsync -avz /Users/henanwan/Documents/workspace/moonshot/pplx-garden-efa/ \
  P5EN-2:~/work/pplx-garden-efa/

ssh P5EN-1 'cd ~/work/pplx-garden-efa && docker build -t pplx-garden-efa:dev .'
ssh P5EN-2 'cd ~/work/pplx-garden-efa && docker build -t pplx-garden-efa:dev .'
```

Build time ≈ 15-20 min on a cold cache (Rust toolchain, EFA installer,
GDRCopy deb build, then maturin/cargo wheel pack). On a warm cache the
final stage takes ~80 s.

### Build-time component pins

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:12.9.1-devel-ubuntu24.04` |
| PyTorch | 2.9.0+cu129 |
| EFA installer | 1.44.0 (matches upstream `dev.Dockerfile`) |
| GDRCopy | 2.5.1 (libgdrapi from a deb built in-stage) |
| Rust | 1.91.0 |
| pplx-garden | `perplexityai/pplx-garden@main` |

This image deliberately mirrors upstream's `dev.Dockerfile` for the dev
stage and adds a single `final` stage that:
1. clones the repo into `/app`
2. runs `python3 -m build --wheel`
3. `pip install`s the wheel
4. verifies the package is importable by metadata (skipping the runtime
   import — `libcuda.so.1` is only mounted by the nvidia container
   runtime at `docker run`, not at `docker build`).

### Build gotcha worth remembering

**`import pplx_garden` cannot succeed at build time.** The package's
`fabric_lib.py` does `from pplx_garden._rust import (...)`, and the Rust
extension dlopens `libcuda.so.1`, which is *not* present at `docker
build` time. Verify with `importlib.metadata.version("pplx-garden")` and
the existence of `/app/benchmarks/bench_all_to_all.py` instead.

---

## 2. Run the benchmark

```bash
LEADER_IP=$(ssh P5EN-1 'hostname -I | awk "{print \$1}"')

# Worker first
ssh P5EN-2 "cd ~/work/pplx-garden-efa && bash run_a2a.sh 1 $LEADER_IP" &
# Leader
ssh P5EN-1 "cd ~/work/pplx-garden-efa && bash run_a2a.sh 0 $LEADER_IP"
```

`run_a2a.sh <node_rank> <master_ip> [extra args]`. The launcher passes
the standard `--world-size $((NUM_NODES * 8)) --nets-per-gpu 2
--init-method=tcp://$MASTER_IP:29500 --node-rank=$NODE_RANK --nvlink=8`
combo from upstream's README; defaults match decode shape (128 tokens,
256 experts, top-8, hidden 7168, FP8 → BF16).

Useful args (all forwarded after the launcher's required flags):

| Arg | Default | Notes |
|---|---|---|
| `--max-num-tokens` | 128 | bench shape; 4096 = prefill |
| `--num-experts` | 256 | |
| `--num-experts-per-token` | 8 | top-k |
| `--hidden-dim` | 7168 | |
| `--in-dtype` | fp8_e4m3 | dispatch dtype |
| `--out-dtype` | bf16 | combine dtype |
| `--num-warmup` / `--num-repeats` | 10000 / 10000 | total = 20000 iterations (~3-5 min wall time) |
| `--nvlink` | 8 (in launcher) | drop the flag in the bench cmd to force pure-RDMA |

### Customising the launcher

Override via env:

```bash
NETS_PER_GPU=4 bash run_a2a.sh 0 $LEADER_IP    # use more NICs
NUM_NODES=4 bash run_a2a.sh 0 $LEADER_IP       # if you have 4 nodes (world-size becomes 32)
```

`--nets-per-gpu 2` is upstream's recommended default for this instance
class (32 EFA NICs / 8 GPU = 4 NICs/GPU available; using 2 picks the
NUMA-closest pair). Try 4 if you want to push more aggregate bandwidth.

---

## 3. Monitor per-NIC EFA bandwidth (optional)

Same helpers as other images; see
`../deepep-v1-efa/README.md#monitoring-per-nic-efa-bandwidth` for the
full recipe.

```bash
rsync -avz ./monitor_efa.sh ./sample_efa_bw.sh P5EN-1:~/
ssh P5EN-1 'chmod +x ~/monitor_efa.sh ~/sample_efa_bw.sh'
ssh P5EN-1 'sleep 30 && bash ~/sample_efa_bw.sh /tmp/pplx_efa_bw.log 30 2'
```

---

## 4. Validated numbers

Test config: 16 ranks (2 nodes × 8 GPU), nets-per-gpu=2, nvlink=8.

### Decode shape (128 tokens, 256 experts, top-8, hidden 7168, FP8/BF16)

20000 iterations on p5; 20000 on p5en. End-to-end latency:

| Op | p5.48xlarge mean | p5.48xlarge p50 | p5.48xlarge p99 | p5en.48xlarge mean | p5en.48xlarge p50 | p5en.48xlarge p99 |
|---|---|---|---|---|---|---|
| Dispatch | 402 µs ± 98 | 397 µs | 440 µs | **224 µs ± 31** | **222 µs** | **259 µs** |
| Combine  | 699 µs ± 723 | 517 µs | 3847 µs | **246 µs ± 27** | **245 µs** | **278 µs** |

Bandwidth (decode):

| | p5.48xlarge | p5en.48xlarge |
|---|---|---|
| Dispatch BW | 19.0 GB/s | **34.0 GB/s** |
| Combine BW | 28.4 GB/s | **59.8 GB/s** |

On p5en the long tail is gone — combine p99 is only 278 µs vs p5's
3847 µs. The p5 tail was caused by EFA v1 SRD retry/ACK overhead under
the all-to-all pattern; v2 hardware fixes it.

### Prefill shape (4096 tokens, same other params), p5en only

2000 iterations (1000 warmup + 1000 repeats):

| Op | mean | p50 | p99 | BW |
|---|---|---|---|---|
| Dispatch (both) | 3128 µs ± 118 | 3122 µs | 3540 µs | 77.6 GB/s |
| Combine (both)  | 5386 µs ± 98 | 5365 µs | 5878 µs | 87.6 GB/s |

Matches upstream README's published p5en prefill numbers (3197 / 5379 µs)
within 2 %. p5 prefill was not measured.

### Kernel send/recv times (decode, p5en)

| Phase | Dispatch send | Dispatch recv | Combine send | Combine recv |
|---|---|---|---|---|
| mean | 28.7 µs | 18.0 µs | 37.0 µs | 13.7 µs |
| p99 | 30.5 µs | 19.4 µs | 39.2 µs | 14.5 µs |

Send/recv kernel times are tight on p5en — variance is in the
wait/sync between ranks, not the kernel itself.

---

## 5. Side-by-side comparisons

### 5a. pplx-garden — p5 vs p5en vs upstream

Same decode shape (128 tokens, EP=16):

| Source | Dispatch | Combine |
|---|---|---|
| **pplx-EFA p5en (this image)** | **224 µs** | **246 µs** |
| pplx-EFA p5en (upstream README) | 214.8 µs | 241.5 µs |
| **pplx-EFA p5 (this image)** | 402 µs | 517 µs (p50) |
| pplx ratio (p5 / p5en) | 1.79× | 2.10× |

Our p5en numbers match upstream within 5 %. The p5 → p5en gap (~2×) is
hardware: EFA v1 (100 Gbps × 32) vs v2 (200 Gbps × 16, newer SRD HW).

### 5b. LL/decode on EFA: pplx-garden vs DeepEP V1 vs UCCL-EP

**On p5en.48xlarge (decode shape, EP=16):**

| Stack | Dispatch | Combine | Notes |
|---|---|---|---|
| **UCCL-EP** | **207 µs** | 301 µs | per-rank avg kernel-time (different reporting from pplx) |
| **pplx-garden (this image)** | **224 µs** | **246 µs** (p50) | end-to-end |
| DeepEP V1 + amazon NVSHMEM | 602 µs | 561 µs | per-rank avg |

**On p5.48xlarge (decode shape, EP=16):**

| Stack | Dispatch | Combine |
|---|---|---|
| pplx-garden | **402 µs** | 517 µs (p50) |
| DeepEP V1 + amazon NVSHMEM (default MAX_NIC_PER_PE) | ~759 µs | ~651 µs |
| DeepEP V1 + amazon NVSHMEM (MAX_NIC_PER_PE=8) | ~627 µs | ~769 µs |
| UCCL-EP | ~3200 µs | ~830 µs |

**On p5en, UCCL-EP slightly edges out pplx-garden on dispatch (207 vs
224 µs).** Both are far ahead of DeepEP V1 (602 µs). The reporting
metric differs (UCCL reports per-rank kernel time, pplx reports
end-to-end including any wait), so for a strict apples-to-apples LL
race they're effectively tied at ~200-250 µs.

**On p5, pplx-garden was the clear leader** because UCCL-EP scaled
poorly with 32 EFA v1 NICs — the p5 → p5en jump for UCCL is enormous
(~15× lower dispatch latency).

pplx-garden was specifically engineered for decode (the upstream paper
"RDMA Point-to-Point Communication for LLM Systems"), aggregates
multiple NICs per GPU at the application layer, and sidesteps
NVSHMEM/NCCL overhead by talking to libfabric directly via a Rust
runtime. UCCL-EP achieves similar latency through a different path
(direct ibverbs + custom EP kernels).

### 5c. Recommended stack for EFA workloads

| Workload | Best on p5en | Best on p5 |
|---|---|---|
| Production decode / LL inference (per-token A2A) | **UCCL-EP or pplx-garden** (~200-250 µs) | **pplx-garden** (~400 µs) |
| Production MoE training (HT / large batch) | **DeepEP V1 + amazon NVSHMEM** (62 GB/s BW) | **DeepEP V1 + amazon NVSHMEM** (60 GB/s BW) |
| Provider-portable RDMA (Nvidia + AMD + EFA + CX7) | UCCL-EP | UCCL-EP |
| Very large EP (>EP128) | Watch DeepEP V2 + ofi-nccl GIN | Watch DeepEP V2 + ofi-nccl GIN |

---

## Caveats / known issues

- **Combine p99 tail is 3.8 ms, mean 700 µs.** The bench runs at ~10000
  iter/s without any warmup-style grace, so a few outliers from across-
  rank waiting drag the mean up. Use **p50 (517 µs)** as the steady-
  state number.
- **`docker build` skips runtime import.** The build only validates the
  wheel exists and the bench module is on disk. Real `import
  pplx_garden` requires `--gpus=all` and `libcuda.so.1`, which only
  exist at `docker run` time.
- **`SYS_PTRACE` + `SYS_ADMIN` are required.** The launcher adds them
  via `--cap-add`. Without them pplx-garden fails to use `pidfd_getfd`
  for cross-process FD sharing, breaking the all-to-all setup.
- **EFA installer pinned to 1.44.0** (matches upstream
  `dev.Dockerfile`). The other images in this repo use 1.48.0; both
  versions work, but we keep 1.44.0 here to stay close to upstream's
  validated config.
- **CUDA 12.9 + torch 2.9.0+cu129**, not the CUDA 13.2 + torch 2.12 used
  by the V1/V2 DeepEP images. Driver 595.58.03 supports both, so the
  two image families coexist on the same host without conflict.

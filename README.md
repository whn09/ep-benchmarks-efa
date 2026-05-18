# ep-benchmarks-efa

Reproducible benchmarks of MoE expert-parallel (EP) all-to-all libraries
on AWS EFA. Each subdirectory contains a self-contained Dockerfile +
launcher scripts + monitoring helpers + a detailed README with build
gotchas, run commands, and validated numbers on both **p5.48xlarge** and
**p5en.48xlarge**.

## Why

DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP) and its
ecosystem (UCCL-EP, pplx-garden) are the de-facto MoE all-to-all
libraries for current-gen GPUs. They were built primarily for InfiniBand
(IBGDA + NVSHMEM); running on AWS EFA needs hardware-specific patches,
plugins, and tuning. This repo collects:

1. Working Docker images for each library on EFA.
2. Apples-to-apples performance numbers on p5.48xlarge (H100, EFA v1)
   and p5en.48xlarge (H200, EFA v2) — the same launcher and config across
   stacks, so comparisons are meaningful.
3. Build gotchas and runtime caveats for each library on EFA.

## What's in this repo

| Directory | Library | Backend | EFA support path |
|---|---|---|---|
| [`deepep-v1-efa/`](deepep-v1-efa/) | DeepEP V1 (`rauteric/DeepEP@remove-fence`) | NVSHMEM libfabric | amazon-contributing/upstream-to-nvshmem `devel_enriched` (libfabric remote transport, multi-NIC RR) |
| [`uccl-ep-efa/`](uccl-ep-efa/) | UCCL-EP (`uccl-project/uccl@main`) | UCCL Rust RDMA stack | direct ibverbs / libfabric, multi-NIC at app layer |
| [`deepep-v2-efa/`](deepep-v2-efa/) | DeepEP V2 (`deepseek-ai/DeepEP@main`) | NCCL Gin | aws-ofi-nccl `master` (`ncclGinPlugin_v11+`) |
| [`deepep-v2-uccl-style/`](deepep-v2-uccl-style/) | DeepEP V2 (same image as above) | NCCL Gin | reuses `deepep-v2-efa:dev`; runs prefill / decode with **UCCL-EP-comparable params** so V2 numbers are directly comparable to UCCL's published p5en results |
| [`pplx-garden-efa/`](pplx-garden-efa/) | pplx-garden (`perplexityai/pplx-garden@main`) | custom Rust libfabric | direct libfabric + multi-NIC aggregation (`fabric-lib`) |

Each directory is self-contained. To reproduce any one of them:

```bash
rsync -avz <dir>/ <node>:~/work/<dir>/
ssh <node> "cd ~/work/<dir> && docker build -t <dir>:dev ."
ssh <worker> "cd ~/work/<dir> && bash <run-script> 1 <leader-private-ip>" &
ssh <leader> "cd ~/work/<dir> && bash <run-script> 0 <leader-private-ip>"
```

See each subdirectory's `README.md` for exact commands, env vars, and
component-version pins.

## Hardware tested

| Instance | GPU | EFA NICs | Per-NIC BW | Aggregate | Generation |
|---|---|---|---|---|---|
| p5.48xlarge | H100 80GB × 8 | 32 | 100 Gbps | 3.2 Tbps | v1 |
| p5en.48xlarge | H200 80GB × 8 | 16 | 200 Gbps | 3.2 Tbps | **v2** (newer SRD) |

All numbers below: 2 nodes × 8 GPU = **16 ranks**, us-east-2, 2026-05-16.
Same Dockerfiles and launchers work on both instances; only `EP_NIC_NAME`
differs (`rdmap79s0` on p5, `rdmap85s0` on p5en — both auto-detectable).

## Side-by-side results

### Normal mode / "throughput" — RDMA bandwidth (per-rank effective)

Test config: 4096 tokens, hidden 7168, top-k 8, ~256 experts, FP8 →
BF16. Larger numbers = better. *DeepEP V2 row uses the
`deepep-v2-uccl-style/` launcher so it shares params with the others
(288 experts, top-8).*

| Stack | p5 Dispatch BF16 | p5 Combine | p5en Dispatch BF16 | p5en Combine |
|---|---|---|---|---|
| **DeepEP V1 + amazon NVSHMEM** | **59.94 GB/s** | **53.92 GB/s** | **62.54 GB/s** | **58.48 GB/s** |
| UCCL-EP | 48.72 GB/s | 13.92 GB/s | 60.64 GB/s | 17.11 GB/s |
| DeepEP V2 + aws-ofi-nccl GIN | ~2 GB/s | ~12 GB/s | 5 GB/s | 20 GB/s |

Reference: upstream DeepEP README on H800 + CX7 IB reports 43 GB/s for
both dispatch and combine at EP=16. EFA matches or exceeds that on the
mature stack (V1) thanks to wider per-rank NIC fan-out.

UCCL-EP's published p5en numbers (`uccl-project/uccl/ep`) are
**50 GB/s prefill dispatch and 18 GB/s combine** — within ~10 % of our
reproduction (60.64 / 17.11), confirming the bench is consistent.

### Low-latency / "decode" — end-to-end dispatch + combine latency

Test config: 128 tokens, hidden 7168, top-k 8, 288 experts (256 for V1
since its bench defaults differ). Lower = better.

| Stack | p5 Dispatch | p5 Combine | p5en Dispatch | p5en Combine |
|---|---|---|---|---|
| pplx-garden (decode shape) | 402 µs | 517 µs (p50) | **222 µs** (p50) | **245 µs** (p50) |
| UCCL-EP (`run_ll_pplx_style.sh`, pplx-style measurement) | 1281 µs (p50) | 1428 µs (p50) | **212 µs** (p50) | 324 µs (p50) |
| UCCL-EP (`run_low_latency.sh`, UCCL self-report) | ~3200 µs | ~830 µs | 207 µs | 301 µs |
| DeepEP V1 + amazon NVSHMEM | ~700 µs | ~720 µs | 602 µs | 561 µs |
| DeepEP V2 (`deepep-v2-uccl-style/` decode) | 2700 µs | 2100 µs (avg) | 1690 µs | 1700 µs |

UCCL's two bench modes report consistent numbers on p5en (~210 µs
dispatch, ~310 µs combine) — the difference is just statistic format.
On p5 the two methods diverge more (3200 vs 1281 µs dispatch) because
EFA v1 has a longer tail; the pplx-style p50 strips outliers.

The **`run_ll_pplx_style.sh` row uses pplx-garden's own measurement
methodology** (warmup + repeats, p50 over end-to-end), which lets you
compare UCCL vs pplx-garden directly:

- **p5en**: dispatch is essentially tied (212 vs 222 µs p50) — UCCL
  slightly edges pplx by 4.5 %; combine: pplx wins by 32 %.
- **p5**: pplx-garden is clearly faster (402 vs 1281 µs dispatch — 3.2×;
  517 vs 1428 µs combine — 2.8×). UCCL's RDMA stack scales much worse
  on EFA v1 / 32 NICs than the libfabric stack pplx uses.

**On p5en, UCCL-EP and pplx-garden are tied for fastest LL dispatch
(207-224 µs)**, ~2.7-2.9× faster than DeepEP V1. **On p5, pplx-garden
is the clear leader** because UCCL's RDMA stack scales much worse with
EFA v1 (32 × 100 Gbps NICs) than with EFA v2 (16 × 200 Gbps NICs).

### pplx-garden prefill (4096 tokens, p5en only)

| Op | mean | p50 | BW |
|---|---|---|---|
| Dispatch | 3128 µs | 3122 µs | 77.6 GB/s |
| Combine | 5386 µs | 5365 µs | 87.6 GB/s |

Matches upstream README's published p5en numbers (3197 / 5379) within 2 %.

## Recommendations

| Workload | p5.48xlarge | p5en.48xlarge |
|---|---|---|
| **MoE training** (HT all-to-all, large batches) | DeepEP V1 + amazon NVSHMEM | DeepEP V1 + amazon NVSHMEM |
| **MoE inference, decode** (per-token A2A) | pplx-garden | UCCL-EP **or** pplx-garden (≈ tied) |
| **MoE inference, prefill** (large batches) | DeepEP V1 (HT mode) or pplx-garden | DeepEP V1 (HT mode) or pplx-garden |
| **Provider-portable** (also AMD / CX7 / etc) | UCCL-EP | UCCL-EP |
| **Very large EP (>EP128, low SM budget)** | watch DeepEP V2 + ofi-nccl GIN (still maturing) | watch DeepEP V2 + ofi-nccl GIN |

## Cross-stack hardware sensitivity

The p5 → p5en jump (EFA v1 → v2 + 16 ranks NICs at 200 Gbps each)
hits each stack very differently. This is the most striking result in
this repo:

| Stack | LL dispatch p5 → p5en | LL combine p5 → p5en | HT dispatch p5 → p5en |
|---|---|---|---|
| DeepEP V1 | 700 → 602 µs (-14 %) | 720 → 561 µs (-22 %) | 60 → 62.5 GB/s (+4 %) |
| pplx-garden | 402 → 224 µs (-44 %) | 517 → 246 µs (-52 %) | (not tested HT) |
| **UCCL-EP** | **3200 → 207 µs** (**-94 %, 15× lower**) | 830 → 301 µs (-64 %) | 49 → 61 GB/s (+24 %) |
| DeepEP V2 GIN | (slow on both) | (slow on both) | 2 → 5 GB/s (2.5×) |

UCCL is the most sensitive — designed for higher-speed, fewer-NIC
topologies; EFA v1 with 32 NICs starves it. DeepEP V1 is the least
sensitive because its NVSHMEM-libfabric path was already mature on EFA
v1. pplx-garden sits in between.

If you only have p5: DeepEP V1 dominates LL. If you have p5en: it's a
much closer race, and the right answer depends on your workload shape.

## Software pins

| Component | deepep-v1-efa | uccl-ep-efa | deepep-v2-efa | pplx-garden-efa |
|---|---|---|---|---|
| NGC base image | pytorch:26.04-py3 (CUDA 13.2.1, torch 2.12) | same | same | nvidia/cuda:12.9.1 + torch 2.9.0+cu129 |
| EFA installer | 1.48.0 | 1.48.0 | 1.48.0 | 1.44.0 (matches upstream) |
| GDRCopy | 2.5.2 | 2.5.2 | 2.5.2 | 2.5.1 |
| NCCL | (NGC's built-in) | n/a | pip nvidia-nccl-cu13 ≥ 2.30.4 | n/a |
| NVSHMEM | amazon-contributing devel_enriched (libfabric+EFA) | n/a | pip nvidia-nvshmem-cu13 (build-time only) | n/a |
| aws-ofi-nccl | n/a | n/a | master (GIN-capable) | n/a |
| EP library | rauteric/DeepEP@remove-fence | uccl-project/uccl@main | deepseek-ai/DeepEP@main | perplexityai/pplx-garden@main |

## What's NOT in this repo

- IB / RoCE numbers — those would require different hardware. We cite
  upstream DeepEP README's H800+CX7 numbers and Amazon's internal
  RoCE-vs-EFA perf table inside individual READMEs as references.
- DeepEP V2 LL benches — currently V2 has limited LL value-add on EFA
  due to ofi-nccl GIN plugin maturity; HT-only smoke is included.
- Exhaustive sweeps — only DeepEP V1 has a `MAX_NIC_PER_PE` sweep
  (`deepep-v1-efa/sweep_max_nic.sh`). Tuning is per-stack:
  `nets-per-gpu` for pplx-garden, `NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE`
  for DeepEP V1, etc.

## Pre-built images on Amazon ECR

To skip the 5-20 min docker build on each node, pull pre-built images
from the private ECR registry (account `579019700964`, region
`us-east-2`):

```bash
ACCOUNT=579019700964
REGION=us-east-2
REGISTRY=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# One-time login (per node, on instances with the right IAM role)
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $REGISTRY

# Pull whichever image(s) you need
docker pull $REGISTRY/ep-benchmarks-efa/deepep-v1-efa:dev
docker pull $REGISTRY/ep-benchmarks-efa/uccl-ep-efa:dev
docker pull $REGISTRY/ep-benchmarks-efa/deepep-v2-efa:dev
docker pull $REGISTRY/ep-benchmarks-efa/pplx-garden-efa:dev

# Re-tag locally so the launcher scripts (which use the short name) work
for name in deepep-v1-efa uccl-ep-efa deepep-v2-efa pplx-garden-efa; do
  docker tag $REGISTRY/ep-benchmarks-efa/$name:dev $name:dev
done
```

Available tags:
- `:dev` — current build for each image. **Note for
  `deepep-v1-efa`**: as of 2026-05-17, `:dev` points to the
  PR-#9-reverted variant (faster LL on EFA, see
  [`deepep-v1-efa/INVESTIGATION_pr9_revert.md`](deepep-v1-efa/INVESTIGATION_pr9_revert.md)).
- `:2026-05-17` — frozen snapshot of `:dev` for reproducibility.
- `:revert-pr9` and `:revert-pr9-2026-05-17` (V1 only) — alias for
  the current `:dev`, makes the variant explicit.
- `:pr9-baseline` (V1 only) — V1 image **with** PR #9 in NVSHMEM
  (the previous `:dev` pre-2026-05-17). Use this if you need the
  unsolicited-write CQ-overflow protection enabled (e.g. workloads
  with arbitrary communication patterns; DeepEP V1 LL has bounded
  patterns and works fine without it).

## Reproducing on your own EFA cluster

Each subdirectory has a step-by-step recipe. Common requirements:

- 2 nodes in the same VPC subnet with a self-referencing security group
  (all-traffic ingress/egress within the SG).
- NVIDIA driver 595+ on each node.
- `/dev/gdrdrv` present (gdrdrv kernel module loaded) — see any of the
  sub-READMEs for the recovery snippet if your DLAMI rebooted into a
  newer kernel and `gdrdrv` DKMS didn't follow.
- For pplx-garden: `SYS_PTRACE` and `SYS_ADMIN` Linux caps (handled by
  its launcher).

## License

Each subdirectory's source library has its own license — DeepSeek
DeepEP (MIT, NVSHMEM SLA for kernels referencing NVSHMEM), UCCL
(Apache-2), pplx-garden (BSD-3-clause), aws-ofi-nccl (Apache-2),
amazon-contributing/upstream-to-nvshmem (NVIDIA NVSHMEM SLA).

The Dockerfiles, launcher scripts, and READMEs in this repo are
released under MIT.

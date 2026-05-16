# deepep-v2-efa — DeepEP V2 (NCCL GIN) on AWS EFA

DeepEP V2 (`deepseek-ai/DeepEP@main`, post-2026 refactor) replaces V1's
NVSHMEM EP transport with NCCL's "Gin" interface. On AWS EFA, Gin is
implemented by `aws-ofi-nccl` (`ncclGinPlugin_v11+`), so the V2 EP path
runs on EFA *via NCCL + ofi-nccl* with no NVSHMEM involvement at runtime.
NVSHMEM is still pulled in at build time because `setup.py` references it
for the legacy V1 paths bundled in V2.

Validated on:
- 2× **p5.48xlarge** (H100 80GB × 8, **32 EFA v1 NICs × 100 Gbps**), us-east-2, 2026-05-16
- 2× **p5en.48xlarge** (H200 80GB × 8, **16 EFA v2 NICs × 200 Gbps**), us-east-2, 2026-05-16

**Performance is currently far below V1 on EFA on both instance types** —
see [§4 Validated numbers](#4-validated-numbers) below. This image is
useful as a working reference for DeepEP V2 + EFA plumbing; treat the
perf numbers as a baseline for ongoing ofi-nccl GIN optimisation, not a
replacement for V1.

**Note on EP_NIC_NAME**: the default `rdmap79s0` only exists on p5; on
p5en the NIC names start at `rdmap85s0`. Override with
`EP_NIC_NAME=rdmap85s0 bash run_test_ep.sh ...` on p5en, or any other
NIC listed by `ls /sys/class/infiniband/`.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `deepep-v2-efa:dev` |
| `patch_envs.py` | Build-time patch making `get_rdma_gbs()` work on EFA |
| `run_test_ep.sh` | 2-node launcher for `tests/elastic/test_ep.py` |
| `monitor_efa.sh` | Per-NIC EFA bandwidth snapshot |
| `sample_efa_bw.sh` | Time-series per-NIC sampler |

## Host prerequisites

Same as `deepep-v1-efa`: NVIDIA driver 595+, EFA hardware, `/dev/gdrdrv`
present, intra-VPC SG with self-reference rule. See
`../deepep-v1-efa/README.md` for the gdrdrv recovery snippet.

---

## 1. Build the image (per node)

```bash
rsync -avz /Users/henanwan/Documents/workspace/moonshot/deepep-v2-efa/ \
  P5EN-1:~/work/deepep-v2-efa/
rsync -avz /Users/henanwan/Documents/workspace/moonshot/deepep-v2-efa/ \
  P5EN-2:~/work/deepep-v2-efa/

ssh P5EN-1 'cd ~/work/deepep-v2-efa && docker build -t deepep-v2-efa:dev .'
ssh P5EN-2 'cd ~/work/deepep-v2-efa && docker build -t deepep-v2-efa:dev .'
```

Build is fast (~3 min on a warm cache) because DeepEP V2 is JIT — no nvcc
work happens at build time. The heavy bits are aws-ofi-nccl autotools
(~80 s) and DeepEP wheel pack (~60 s).

### Build-time component pins

| Component | Version |
|---|---|
| Base image | `nvcr.io/nvidia/pytorch:26.04-py3` (CUDA 13.2.1, torch 2.12, sm90) |
| EFA installer | 1.48.0 |
| GDRCopy | 2.5.2 (libgdrapi only) |
| NCCL | pip `nvidia-nccl-cu13>=2.30.4` |
| NVSHMEM | pip `nvidia-nvshmem-cu13` (build-time only) |
| aws-ofi-nccl | `aws/aws-ofi-nccl@master` (provides `ncclGinPlugin_v11+`) |
| DeepEP | `deepseek-ai/DeepEP@main` (V2, JIT) |

### Build gotchas worth remembering

1. **`pip install` must use `--no-build-isolation`.** PEP 517 isolated
   builds spin up a fresh venv without torch, but DeepEP's `setup.py`
   does `from torch.utils.cpp_extension import …` at import time. Fix:
   `pip install --no-cache-dir --no-deps --no-build-isolation -v .`.
2. **Pin the pip-installed NCCL ahead of system NCCL via ldconfig.** NGC
   `pytorch:26.04-py3` ships `libnccl2-2.29.7` from a deb in
   `/usr/lib/x86_64-linux-gnu/`. DeepEP V2 links against the pip wheel
   (>=2.30.4) and runs `check_nccl_so()` at import time, asserting that
   the loaded NCCL matches what it linked against. Without an
   `/etc/ld.so.conf.d/0-pip-nccl.conf` entry, the system one wins and
   import fails with `AssertionError: Invalid NCCL versions`.
3. **CCCL include path.** Same NGC quirk as the V1 image — CCCL lives
   under `/usr/local/cuda/targets/x86_64-linux/include/cccl`. Added to
   `CPATH` in the Dockerfile.
4. **`get_rdma_gbs()` doesn't work on EFA.** Upstream uses `ibstat
   <nic>` to parse a `Rate:` line, but EFA NICs don't show up in
   `ibstat` (they're not IB-class). Without a fix, V2 hits
   `ZeroDivisionError` deep inside `get_theoretical_num_sms()` at
   runtime. The Dockerfile applies `patch_envs.py`, which:
     - reads `/sys/class/infiniband/<nic>/ports/1/rate` first (works on EFA)
     - falls back to `ibstat` (works on Mellanox/IB/RoCE)
     - honours `EP_RDMA_GBS=<float>` as an override
5. **`from deep_ep import …` from `/opt/deepep` fails.** When cwd is the
   source tree, Python imports the source `deep_ep/` package (no
   `_C.so`) instead of the installed wheel. Run benches from `/tmp` or
   anywhere outside `/opt/deepep`. The launcher script uses `cd
   /opt/deepep && python tests/elastic/test_ep.py` which works because
   the test script imports from `deep_ep` *after* the package's `_C` is
   resolvable via the wheel install — the trick is making sure the
   launcher uses `python <full-path-to-test>` not `python -m`.

### NCCL GIN plugin selection

The image ships **two** ofi plugins:
- `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so` — from EFA installer
  1.48.0, **no GIN support** (older API).
- `/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` — built from `aws/aws-ofi-nccl@master`,
  **provides `ncclGinPlugin_v11+`**.

`NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` is set as a
default ENV in the image and re-exported in the launcher. Both lockings
are needed because some NCCL versions read it before the launcher runs.

---

## 2. Run the benchmark

```bash
LEADER_IP=$(ssh P5EN-1 'hostname -I | awk "{print \$1}"')

ssh P5EN-2 "cd ~/work/deepep-v2-efa && bash run_test_ep.sh 1 $LEADER_IP --test-first-only" &
ssh P5EN-1 "cd ~/work/deepep-v2-efa && bash run_test_ep.sh 0 $LEADER_IP --test-first-only"
```

`run_test_ep.sh <node_rank> <master_ip> [extra args]`. `--test-first-only`
runs a single shape (4096 tokens, hidden 7168, top-6, 256 experts) and
exits — useful for smoke. Drop the flag to run the full sweep (the EP
config defaults are documented in `tests/elastic/test_ep.py`).

`init_dist` convention is the same as V1: `WORLD_SIZE=#nodes`,
`RANK=#node_rank`. Inside the container, `torch.multiprocessing.spawn`
fans out to the 8 local GPUs.

Useful args: `--num-tokens`, `--hidden`, `--num-topk`, `--num-experts`,
`--num-sms`, `--num-qps`, `--allow-hybrid-mode 0/1`, `--skip-perf-test`,
`--unbalanced-ratio`, `--dump-profile-traces <dir>`.

### Image-baked runtime env

| Env | Value | Why |
|---|---|---|
| `NCCL_NET_PLUGIN` | `/opt/aws-ofi-nccl/lib/libnccl-net-ofi.so` | Force the GIN-capable plugin |
| `LD_LIBRARY_PATH` | prepends `/opt/aws-ofi-nccl/lib`, `/opt/amazon/efa/lib` | |
| `FI_PROVIDER` | `efa` | |
| `FI_EFA_USE_DEVICE_RDMA` | 1 | |
| `EP_NIC_NAME` | `rdmap79s0` | DeepEP V2's default `mlx5_0` doesn't exist on EFA. Override per node if needed. |

`EP_RDMA_GBS=<float>` (set via `-e` in the launcher) overrides the sysfs
reading from the patched `get_rdma_gbs()` if needed.

---

## 3. Monitor per-NIC EFA bandwidth (optional)

Same helpers as the V1 image; see the instructions in
`../deepep-v1-efa/README.md#monitoring-per-nic-efa-bandwidth`.

```bash
rsync -avz ./monitor_efa.sh ./sample_efa_bw.sh P5EN-1:~/
ssh P5EN-1 'chmod +x ~/monitor_efa.sh ~/sample_efa_bw.sh'
ssh P5EN-1 'sleep 30 && bash ~/sample_efa_bw.sh /tmp/v2_efa_bw.log 30 2'
```

---

## 4. Validated numbers

Test config: 16 ranks (2 nodes × 8 GPU), 4096 tokens, hidden 7168,
top-6, 256 experts, FP8 dispatching + BF16 combining,
`--test-first-only`. DeepEP V2 reports two bandwidth components per op:

- **SO** (scaleout) = cross-node EFA traffic
- **SU** (scaleup)  = intra-node NVLink traffic

| Op | p5.48xlarge SO | p5en.48xlarge SO | p5.48xlarge SU | p5en.48xlarge SU |
|---|---|---|---|---|
| Dispatch (raw)        | 2 GB/s | **5 GB/s** (+150 %) | 6 GB/s | 13 GB/s |
| Combine               | 11-13 GB/s | **16-28 GB/s** | 29-33 GB/s | 41-73 GB/s |
| Reduced combine       | 11-12 GB/s | **23-24 GB/s** | 29-32 GB/s | 60-62 GB/s |
| Reduce epilogue (NVLink) | — | — | 2640-2693 GB/s | **3500-3580 GB/s** |

All test cases ran to completion on both instance types without
correctness errors.

---

## 5. Side-by-side comparisons

### 5a. DeepEP V2 on EFA: this image (both instances) vs coworker p5en

A coworker measured V2 GIN on **p5en.48xlarge** with the same
`aws-ofi-nccl@master` baseline plus an experimental "shadow hack":

| Stack / topology | 16 ranks Dispatch | 16 ranks Combine | 32 ranks Dispatch | 32 ranks Combine |
|---|---|---|---|---|
| IB H800 + CX7 (DeepEP github) | 90 GB/s | 81 GB/s | 61 GB/s | 61 GB/s |
| ofi-nccl master baseline (p5en, coworker) | 7 GB/s | 17 GB/s | 5 GB/s | 14 GB/s |
| ofi-nccl master + shadow hack (p5en, coworker) | 8 GB/s | 18 GB/s | — | — |
| **This image, p5en** (master, --test-first-only) | **5 GB/s** | **22-28 GB/s** | (untested) | (untested) |
| **This image, p5** (master, --test-first-only) | **2 GB/s** | **12 GB/s** | (untested) | (untested) |

Our p5en numbers (5 / 22-28) sit close to the coworker's master
baseline (7 / 17) — within ofi-nccl HEAD drift — confirming the GIN
path itself is working, just slow. The p5 → p5en jump (2 → 5 GB/s
dispatch, 2.5×) is the expected EFA v1 → v2 hardware difference.

### 5b. V2 vs V1 vs UCCL-EP cross-stack

**On p5en.48xlarge:**

| Stack | Dispatch SO | Combine SO | Notes |
|---|---|---|---|
| DeepEP V1 + amazon NVSHMEM | **62.5 GB/s** | **58.5 GB/s** | Mature stack |
| UCCL-EP | 60.6 GB/s | 17.1 GB/s | dispatch matches V1, combine weak |
| **DeepEP V2 + aws-ofi-nccl GIN** | **5 GB/s** | 22-28 GB/s | GIN plugin still maturing |

**On p5.48xlarge:**

| Stack | Dispatch SO | Combine SO |
|---|---|---|
| DeepEP V1 + amazon NVSHMEM | ~60 GB/s | ~54 GB/s |
| UCCL-EP | ~49 GB/s | ~14 GB/s |
| **DeepEP V2 + aws-ofi-nccl GIN** | ~2 GB/s | ~12 GB/s |

V2 is **12× behind V1 on p5en, 30× behind V1 on p5** for normal-mode
dispatch. The reasons are well understood:

1. **GIN plugin is new.** `aws/aws-ofi-nccl@master` has had GIN-related
   commits multiple times per week (e.g. *"gin: Implement iget for GIN
   v13 API"*, *"perf(gin): coalesce writedata and metadata send"*) at
   the time of writing. Multi-NIC striping and per-message batching are
   actively being optimised.
2. **`num_allocated_qps` is capped to 2** in V2's EFA path to avoid
   overflowing GIN's 128-slot ring, which limits per-PE concurrency.
3. **V2 is designed for very large EP** (EP2048) where its SM-saving
   architecture wins. For EP=16 on EFA v1, V1 is still the right choice.

### 5c. Recommended stack today (2 nodes)

| Workload | Recommendation |
|---|---|
| Production MoE training, normal kernels (any p5/p5en) | **DeepEP V1 + amazon NVSHMEM** (`../deepep-v1-efa/`) |
| Production MoE inference, low-latency kernels (p5en) | **UCCL-EP** (`../uccl-ep-efa/`, dispatch 207 µs) or **pplx-garden** (224 µs decode) |
| Production MoE inference, low-latency kernels (p5) | DeepEP V1 with `MAX_NIC_PER_PE=8` (UCCL is much slower on p5) |
| Experimentation with very large EP (>EP128) | Watch DeepEP V2 + ofi-nccl GIN, but expect ongoing improvement |
| Provider-portable RDMA (Nvidia + AMD + EFA + CX7) | UCCL-EP (`../uccl-ep-efa/`) |

---

## Caveats / known issues

- **`destroy()` is not called before DeepEP elastic buffer destruction,
  which can leak resources.`** — printed once per rank during
  `--test-first-only`. Cosmetic; tests pass.
- **`Failed to get RDMA connection speed`** — this should NOT appear
  thanks to `patch_envs.py`. If it does, the patch may have failed to
  apply (upstream changed the function); the env var override
  `EP_RDMA_GBS=12.5` is the manual escape hatch.
- **Build size**: image is ~12-13 GB compressed (CCCL + NCCL wheels +
  ofi-nccl source build). About 1 GB larger than the V1 image.
- **`UndefinedVar: Usage of undefined variable '$CPATH'`** during build —
  harmless docker buildkit warning, same as V1 image.
- **PR `shadow hack`** is *not* applied. Coworker's numbers show it adds
  ~1 GB/s on dispatch and ~1 GB/s on combine at 16 ranks on p5en. If
  you have access to the patch and want to try it on p5, drop it into
  this directory as `shadow_hack.patch` and `git apply` inside the
  Dockerfile right before the DeepEP build step.

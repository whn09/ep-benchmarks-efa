# deepep-v1-efa — DeepEP V1 on AWS EFA via patched NVSHMEM

DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP) (V1) does not run
on AWS EFA out of the box because upstream NVSHMEM has no libfabric remote
transport and DeepEP's internode kernels assume IBGDA. This image runs DeepEP
V1 on EFA by combining:

- `amazon-contributing/upstream-to-nvshmem` @ `devel_enriched` —
  adds the libfabric remote transport with EFA support and multi-NIC
  round-robin (`NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE`).
- `rauteric/DeepEP` @ `remove-fence` — internode kernels updated to use
  upstream NVSHMEM API (no IBGDA dependency) and drops fences that aren't
  required when the transport is unordered (EFA SRD).

Validated on:
- 2× **p5.48xlarge** (H100 80GB × 8, **32 EFA v1 NICs × 100 Gbps**), us-east-2, 2026-05-16
- 2× **p5en.48xlarge** (H200 80GB × 8, **16 EFA v2 NICs × 200 Gbps**), us-east-2, 2026-05-16

Both runs share the same Dockerfile and launcher scripts; only the EFA
NIC name changes (auto-detected by `EP_NIC_NAME`). See section 4 for
per-instance numbers and how the hardware difference shows up.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `deepep-v1-efa:dev` from `nvcr.io/nvidia/pytorch:26.04-py3` |
| `run_internode.sh` | 2-node launcher for `tests/test_internode.py` (normal mode) |
| `run_low_latency.sh` | 2-node launcher for `tests/test_low_latency.py` |
| `sweep_max_nic.sh` | Driver script to sweep `NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE` across {1,2,4,8,16,32,default} |
| `monitor_efa.sh` | Per-NIC EFA bandwidth snapshot (`-s`, `-b`, `-d`, `-l`) |
| `sample_efa_bw.sh` | Time-series per-NIC sampler (writes a log file) |

---

## Host prerequisites

Each node must have:

- **NVIDIA driver 595+** (CUDA 13.2-capable). NGC `pytorch:26.04-py3` ships
  CUDA 13.2.1 and refuses to start with older drivers.
- **EFA hardware + kernel modules.** `fi_info -p efa` should list multiple
  domains (e.g. `rdmap79s0-rdm` … `rdmap98s0-rdm` on p5.48xlarge).
- **`/dev/gdrdrv` present and world-readable.** After AWS DLAMI kernel
  upgrades, `gdrdrv` DKMS sometimes does not rebuild and the device node is
  missing. Fix:

  ```bash
  sudo dkms install gdrdrv/2.5.2 -k $(uname -r)
  sudo modprobe gdrdrv
  MAJOR=$(awk '$2=="gdrdrv"{print $1}' /proc/devices)
  sudo mknod /dev/gdrdrv c $MAJOR 0
  sudo chmod 666 /dev/gdrdrv
  gdrcopy_copybw   # expect ~11 MB/s
  ```

  Required because `modprobe gdrdrv` does not install a udev rule and so
  does not create `/dev/gdrdrv` on its own. Otherwise NVSHMEM falls back
  off the GPU memory MR registration path.

- **VPC / Security Group:** both nodes in the same VPC, the EC2 SG must
  allow all-traffic ingress/egress within itself. EFA SRD over plain SGs
  works for sparse flows but can deadlock at higher QP counts.

---

## Build the image (per node)

```bash
cd ~/work/deepep-v1-efa
docker build -t deepep-v1-efa:dev .
```

Build time ≈ 12 min on p5.48xlarge: ~1 min for system deps + EFA installer +
GDRCopy, ~10 min for NVSHMEM (libfabric only, sm90), ~1 min for DeepEP wheel.

### Build-time component pins

| Component | Version |
|---|---|
| Base image | `nvcr.io/nvidia/pytorch:26.04-py3` (CUDA 13.2.1, torch 2.12, sm90) |
| EFA installer | 1.48.0 |
| GDRCopy | 2.5.2 |
| NVSHMEM | `amazon-contributing/upstream-to-nvshmem@devel_enriched` (3.6.5) |
| DeepEP | `rauteric/DeepEP@remove-fence` (1.2.1+cd500bf) |

NVSHMEM CMake flags (key ones):

| Flag | Value | Why |
|---|---|---|
| `NVSHMEM_LIBFABRIC_SUPPORT` | ON | The EFA transport |
| `NVSHMEM_USE_GDRCOPY` | ON | Lets NVSHMEM register GPU memory as MRs to libfabric |
| `NVSHMEM_IBGDA_SUPPORT` | OFF | EFA does not implement IB verbs |
| `NVSHMEM_IBRC_SUPPORT` | OFF | Not used; saves build time |
| `NVSHMEM_USE_MLX5DV` | ON (default) | Compiled in but unused on EFA |
| `LIBFABRIC_HOME` | `/opt/amazon/efa` | EFA-installer-provided libfabric |
| `GDRCOPY_HOME` | `/usr/local` | Where the gdrcopy `make lib_install` puts headers |
| `CMAKE_CUDA_ARCHITECTURES` | 90 | H100. Add 100 for B300. |

### Build gotchas worth remembering

- **`apt-get update` and EFA installer must be in the same RUN layer.** The
  EFA installer apt-installs `environment-modules` and `tcl`. If the previous
  RUN clears `/var/lib/apt/lists/*`, the EFA installer fails with
  `Unable to locate package environment-modules`.
- **NGC `pytorch:26.04-py3` puts CCCL headers under
  `/usr/local/cuda/targets/x86_64-linux/include/cccl`**, not
  `/usr/local/cuda/include`. NVSHMEM's `nvshmem_tensor.h` includes
  `cuda/std/tuple`, which DeepEP's host C++ pass (`g++` on `deep_ep.cpp`)
  must be able to resolve. The Dockerfile prepends that path to `CPATH`.
- **`import deep_ep` cannot be done at build time.** It transitively dlopens
  `libcuda.so.1`, which is only mounted by the nvidia container runtime at
  `docker run`, not during `docker build`. The Dockerfile only verifies
  the wheel + the compiled `.so` exist; the import is exercised at runtime.

---

## Run

Get the leader node's private IP (the one that the worker rank dials):

```bash
ssh P5EN-1 'hostname -I | awk "{print \$1}"'   # e.g. 172.31.45.156
```

### Normal kernels (test_internode.py)

```bash
# Worker (P5EN-2) first — it dials in
ssh P5EN-2 'cd ~/work/deepep-v1-efa && bash run_internode.sh 1 <leader-ip>' &
# Leader (P5EN-1)
ssh P5EN-1 'cd ~/work/deepep-v1-efa && bash run_internode.sh 0 <leader-ip>'
```

`run_internode.sh <node_rank> <master_ip> [extra args]`. `node_rank` is 0
for leader, 1 for worker. Defaults to 8 GPUs/node, 4096 tokens, hidden 7168,
256 experts, top-k 8.

Runtime is ~5 min: 32 correctness checks (BF16/FP8 × top-k × async × previous)
followed by an autotuning sweep over SM count and chunk sizes.

### Low-latency kernels (test_low_latency.py)

```bash
ssh P5EN-2 'cd ~/work/deepep-v1-efa && bash run_low_latency.sh 1 <leader-ip>' &
ssh P5EN-1 'cd ~/work/deepep-v1-efa && bash run_low_latency.sh 0 <leader-ip>'
```

Defaults: 8 GPUs/node, 128 tokens, hidden 7168, 288 experts, top-k 8.
Different `MASTER_PORT` (29501) than normal so they don't collide.

### Runtime env baked into the image

| Env | Value |
|---|---|
| `NVSHMEM_DIR` | `/opt/nvshmem` |
| `NVSHMEM_REMOTE_TRANSPORT` | `libfabric` |
| `NVSHMEM_LIBFABRIC_PROVIDER` | `efa` |
| `FI_PROVIDER` | `efa` |
| `FI_EFA_USE_DEVICE_RDMA` | 1 |
| `LD_LIBRARY_PATH` | `/opt/nvshmem/lib:/opt/amazon/efa/lib:…` |

The launcher scripts also set `NCCL_DEBUG=WARN` and `NVSHMEM_DEBUG=WARN`
which keep stdout readable.

---

## Validated numbers (2 nodes × 8 GPU = 16 ranks, 2026-05-16)

### Normal mode

All 32 correctness tests passed on both instance types. Best autotuned configs:

| Op | p5.48xlarge (H100, 32 v1 NICs) | p5en.48xlarge (H200, 16 v2 NICs) | Δ |
|---|---|---|---|
| Dispatch (BF16) RDMA BW | 59.94 GB/s | **62.54 GB/s** | +4 % |
| Dispatch (FP8)  RDMA BW | 48.17 GB/s | **54.98 GB/s** | +14 % |
| Combine RDMA BW         | 53.92 GB/s | **58.48 GB/s** | +8 % |
| NVL BW (typical)        | ~195 GB/s   | ~200 GB/s        | small (NVLink, instance-independent) |

Normal mode is BW-saturated by DeepEP's chunk pipeline ceiling; the per-NIC
generation jump (v1 → v2) helps only a little because the chunks already
amortise per-NIC overhead well.

### Low-latency mode

| Per-rank metric | p5.48xlarge | p5en.48xlarge | Δ |
|---|---|---|---|
| Dispatch + Combine BW | 16.5 GB/s | **20.4 GB/s** | +24 % |
| Dispatch + Combine avg latency | ~1333 µs | **1083 µs** | -19 % |
| Dispatch latency | ~700 µs (10-12 GB/s) | **602 µs** (12.5 GB/s) | -14 % |
| Combine latency | ~720 µs (18-22 GB/s) | **561 µs** (25.9 GB/s) | -22 % |

LL is small-message-latency-bound, so it picks up most of the EFA v2
SRD-floor improvement. Our build sits between the perf table's "3.6"
and "3.7-track" rows (see § "Reference comparison" below) — rebuilding
against a newer `devel_enriched` HEAD should close more of the remaining
gap to the table's 3.7 column.

Theoretical aggregate EFA capacity is the same on both instances
(~3.2 Tbps = ~400 GB/s unidirectional), but the per-NIC ceiling
matters at small messages.

### Per-NIC distribution during normal mode (p5.48xlarge, 32 NICs)

| Metric | Value |
|---|---|
| Active NICs | **32 / 32** |
| Per-NIC tx range (peak) | 80 - 83 Gbps |
| Spread across NICs | ~3 Gbps (≈4 %) |
| Total tx (peak sample) | **2776 Gbps** ≈ 347 GB/s |

Per-NIC utilisation ~83 % of the ~100 Gbps single-NIC ceiling. Sampling
recipe is in section [Monitoring per-NIC EFA bandwidth](#monitoring-per-nic-efa-bandwidth)
below.

---

## Monitoring per-NIC EFA bandwidth

`monitor_efa.sh` reads counters from `rdma statistic show` and prints
per-NIC tx/rx Gbps. For benchmark runs we need a *log-to-file* version
that samples N times, not the interactive `-b` mode — that's what
`sample_efa_bw.sh` does.

### One-shot snapshot

```bash
ssh P5EN-1 'bash ~/monitor_efa.sh -s'      # printable table
ssh P5EN-1 'bash ~/monitor_efa.sh -b 2'    # 2-second instantaneous bandwidth
ssh P5EN-1 'bash ~/monitor_efa.sh -l'      # list all 32 EFA NICs
```

### Time-series sampling during a bench run

```bash
# sync the helpers once (both live in this directory)
rsync -avz ./monitor_efa.sh ./sample_efa_bw.sh P5EN-1:~/
ssh P5EN-1 'chmod +x ~/monitor_efa.sh ~/sample_efa_bw.sh'
```

```bash
# Usage: sample_efa_bw.sh OUT N INTERVAL  ->  N samples × INTERVAL seconds
ssh P5EN-1 'bash ~/sample_efa_bw.sh /tmp/deepep_efa_bw.log 30 2'
```

**Timing matters:** the bench takes ~2-3 min total but spends the first
~30-45 s on init/correctness tests (negligible NIC traffic) and the last
~10 s tearing down. Sleep 30 s after launching the bench, then start
sampling for ~60 s, to land squarely in the autotune sweep.

End-to-end recipe:

```bash
LEADER_IP=$(ssh P5EN-1 'hostname -I | awk "{print \$1}"')

# 1. launch bench in background on both nodes
ssh P5EN-2 "cd ~/work/deepep-v1-efa && bash run_internode.sh 1 $LEADER_IP \
  > /tmp/deepep_normal_worker.log 2>&1" &
ssh P5EN-1 "cd ~/work/deepep-v1-efa && bash run_internode.sh 0 $LEADER_IP \
  > /tmp/deepep_normal_leader.log 2>&1" &

# 2. wait for steady state, then sample on the leader
ssh P5EN-1 'sleep 30 && bash ~/sample_efa_bw.sh /tmp/deepep_efa_bw.log 30 2'

# 3. inspect aggregate per-sample
ssh P5EN-1 'grep "^TOTAL" /tmp/deepep_efa_bw.log'

# 4. inspect per-NIC distribution at a peak sample
ssh P5EN-1 'awk "/=== sample 5 @/ {flag=1} /=== sample 6 @/ {flag=0} flag" \
  /tmp/deepep_efa_bw.log'
```

A side-by-side comparison with UCCL-EP using the exact same recipe is in
`../uccl-ep-efa/README.md`.

---

## DeepEP V1: EFA (this image) vs RoCE / IB / NVSHMEM-version reference

Two reference baselines:

1. **Upstream DeepEP README** — H800 + CX7 InfiniBand, older software
   stack. Published 16-EP numbers: 43 GB/s dispatch BW, 43 GB/s combine BW,
   118 µs LL dispatch latency (63 GB/s), 195 µs LL combine latency
   (74 GB/s). Useful as a public reference, but a different generation of
   HW + NVSHMEM than what we run.
2. **Amazon internal perf table** (referenced 2026-05) — same DeepEP V1
   workload measured across **RoCE** (NVSHMEM IBRC transport),
   **NVSHMEM 3.6** on EFA, and **NVSHMEM 3.7** on EFA, at 2 / 4 / 8 nodes.
   Closer apples-to-apples for our measurement.

   On the version labels: at the time of writing, the latest tagged
   NVSHMEM release is **3.6.5** — what the table calls "NVSHMEM 3.6" is
   the 3.6 release line, and "NVSHMEM 3.7" is the in-flight branch (the
   amazon `devel_enriched` head) that will become 3.7 once cut. Our image
   builds against `devel_enriched`, so our row is effectively the **3.7-track**
   build, just at an earlier point on that branch than what the table's
   3.7 column was measured at — explaining why our numbers sit between
   the table's 3.6 and 3.7 rows rather than matching 3.7 exactly.

Comparing to **2-node** column (matches our 16-rank run):

### Normal-mode RDMA bandwidth (per-rank effective, 2 nodes)

| Stack | HT Dispatch | HT Combine |
|---|---|---|
| RoCE (NVSHMEM IBRC) | **79.7 GB/s** | **66.4 GB/s** |
| EFA NVSHMEM 3.7-track (table) | 61.4 GB/s | 60.7 GB/s |
| EFA NVSHMEM 3.6 (table) | 58.1 GB/s | 60.8 GB/s |
| **EFA NVSHMEM `devel_enriched` p5en (this image)** | **62.54 GB/s** (BF16) | **58.48 GB/s** |
| EFA NVSHMEM `devel_enriched` p5 (this image) | 59.94 GB/s (BF16) | 53.92 GB/s |
| IB H800+CX7 (DeepEP README) | 43 GB/s | 43 GB/s |

Our p5en measurement matches the table's 3.7-track row on dispatch
(62.5 vs 61.4 GB/s) and slightly under-runs it on combine (58.5 vs 60.7).
RoCE is still ~25 % faster on dispatch and ~14 % faster on combine —
RC verbs have lower per-WQE overhead than EFA SRD even after multi-NIC
striping.

### Low-latency mode latency (2 nodes)

| Stack | LL Dispatch | LL Combine | LL Dispatch / RoCE |
|---|---|---|---|
| RoCE (NVSHMEM IBRC) | **182 µs** | **320 µs** | 1.0× |
| **EFA NVSHMEM `devel_enriched` p5en (this image)** | **602 µs** | **561 µs** | **3.3×** |
| EFA NVSHMEM 3.7-track (table) | 405 µs | 446 µs | 2.2× |
| EFA NVSHMEM 3.6 (table) | 881 µs | 910 µs | 4.8× |
| EFA NVSHMEM `devel_enriched` p5 (this image) | ~700 µs | ~720 µs | 3.8× |
| IB H800+CX7 (DeepEP README) | 118 µs | 195 µs | 0.65× |

(The IB README row uses a different generation of HW + NVSHMEM than the
others; treat it as a public reference, not a direct apples-to-apples
comparison with the EFA rows.)

**Important:** the structurally interesting LL gap on EFA is ~**2× vs
RoCE**, not ~6× — the latter only shows up if you compare against upstream
README's IB numbers on a different stack. The 3.6 → 3.7-track jump alone
closes about half of the LL gap (881 → 405 µs on dispatch). Our p5en
measurement (602 µs) sits between the table's 3.6 and 3.7 rows, closer
to 3.6 — `devel_enriched` is on the 3.7 track but at an earlier commit
than the table's 3.7 column. Switching p5 → p5en already closed ~14 %
of the dispatch gap (700 → 602 µs); rebuilding against current
`devel_enriched` HEAD should close more.

### Why the LL gap is ~2× even on the 3.7 track

- **EFA SRD small-message latency floor: ~10-15 µs** vs IB RC's ~1-2 µs
  (no HW-level reliable delivery; libfabric does software ACK/retry).
  LL mode is dominated by per-token tiny messages (128 tokens × 16 ranks),
  so each RTT picks up the SRD-vs-RC difference.
- **No IBGDA on EFA.** On IB, IBGDA lets the GPU initiate RDMA ops with
  zero CPU on the critical path. On EFA, every op goes GPU → CPU
  proxy → libfabric → NIC. The 3.7-track work reduced this proxy overhead
  but the path is still there.

### Tuning: NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE sweep on p5

The amazon NVSHMEM libfabric transport round-robins across NICs. On p5
with 32 EFA NICs, default fan-out is wider than ideal for LL's small
per-token messages — limiting it lowers per-message setup overhead.
Sweep results (LL bench, 2× p5.48xlarge, 16 ranks, rank 0 numbers):

| MAX_NIC_PER_PE | Dispatch µs | Combine µs | Dispatch BW | Combine BW |
|---|---|---|---|---|
| 1 | 781 | 890 | 9.6 GB/s | 16.3 GB/s |
| 2 | 792 | 842 | 9.5 GB/s | 17.3 GB/s |
| 4 | 647 | 814 | 11.6 GB/s | 17.9 GB/s |
| **8** (best dispatch) | **627** | 769 | **12.0 GB/s** | 18.9 GB/s |
| 16 | 733 | **672** | 10.3 GB/s | **21.6 GB/s** |
| 32 | 650 | 764 | 11.6 GB/s | 19.0 GB/s |
| **default** (no setting) | 759 | 651 | 9.9 GB/s | **22.3 GB/s** |

- **`=1` and `=2` are negative** — single-NIC queuing stalls SRD.
- **`=8` is the best dispatch point** (627 µs), 17 % faster than default.
- **`=16` and the default give the best combine** — combine apparently
  benefits from wider fan-out than dispatch.
- Dispatch and combine prefer different fan-outs, so there's no single
  optimum. Pick based on which phase dominates your workload (PD-disagg
  decode is dispatch-heavy; RL rollouts may be combine-heavy).

Recommended starting point: **`-e NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE=8`**
in `run_low_latency.sh` if dispatch latency matters; leave default
(or `=16`) if combine BW matters.

Even at the best `=8` setting on p5, dispatch latency was 627 µs — still
3.4× of RoCE's 182 µs (perf table). The p5 → p5en jump (see next section)
delivers a similar improvement (700 → 602 µs without re-tuning), and the
two improvements are partly additive — sweeping `MAX_NIC_PER_PE` on p5en
is a follow-up.

### Confirmed: p5 vs p5en hardware difference

The Amazon perf table's EFA rows are measured on **p5en.48xlarge**.
We've now run the exact same Dockerfile + launcher on both instance
types. The two differ in ways that matter specifically for LL mode:

| Dimension | p5.48xlarge | p5en.48xlarge |
|---|---|---|
| GPU | H100 80GB × 8 | H200 80GB × 8 |
| EFA NIC count | **32** | **16** |
| Per-NIC bandwidth | 100 Gbps | **200 Gbps** |
| Aggregate | 3.2 Tbps | 3.2 Tbps |
| EFA generation | v1 | **v2** (newer SRD HW) |

Measured LL dispatch on this image:

| Instance | LL Dispatch | LL Combine |
|---|---|---|
| p5.48xlarge | ~700 µs | ~720 µs |
| **p5en.48xlarge** | **602 µs** (-14 %) | **561 µs** (-22 %) |

For HT mode it's nearly a wash (60 → 62.5 GB/s dispatch, 54 → 58.5 combine)
because chunks are large enough to amortise per-NIC startup. For LL mode
the EFA v2 SRD's lower per-message latency floor and the wider per-NIC
pipe both help.

This is broadly consistent with our hypothesis: switching to p5en gets us
much of the way from "~3.8× of RoCE" toward "~2.2× of RoCE", but not all
the way — the residual gap to the perf table's 405 µs is most likely
software (`devel_enriched` HEAD progress) plus possible NIC-fan-out tuning.

### Takeaway

- **Normal mode**: DeepEP V1 + amazon NVSHMEM on EFA gets ~75-80 % of
  RoCE BW and ~140 % of upstream README's IB BW. Solid for production
  use of the normal kernels. p5en delivers 4-14 % over p5 here.
- **Low-latency mode**: 602 µs on p5en, 700 µs on p5. The Amazon perf
  table reports 405 µs on the 3.7-track p5en build; the gap to that is
  most likely the `devel_enriched` commit age plus possible NIC fan-out
  tuning. Two follow-ups:
    - **Rebuild against current `devel_enriched` HEAD** — pure NVSHMEM
      rebuild, no DeepEP change needed.
    - **Sweep `NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE` on p5en** — the p5
      sweet spot was `=8`; on p5en (16 NICs total) `=4` or `=8` is
      worth trying.
  Beyond those, the SRD/IBGDA gap vs IB is structural — plan around it
  for latency-critical paths (PD-disagg, smaller cross-node EP groups).
- Future hardware: EFA v3 on p5e/p6 (200 Gbps × 32 = 6.4 Tbps) widens the
  normal-mode aggregate ceiling. The LL gap depends on whether EFA
  introduces HW-accelerated ACKs / an IBGDA equivalent.

---

## Known runtime warnings

- `WARN: PE has previously called connect_endpoints()` — printed once per
  PE on first DeepEP buffer setup, harmless (NVSHMEM libfabric transport
  re-entry path).
- `bootstrap_uid_*` lines in stderr — verbose UID bootstrap traces from
  the amazon NVSHMEM build. They are informational, not errors. To suppress
  set `NVSHMEM_BOOTSTRAP_UID_DEBUG=0` (already off in this image's defaults
  but transports may re-enable).

---

## Extending to B300 (sm100)

- Change `CMAKE_CUDA_ARCHITECTURES=90` to `90;100` in the Dockerfile's
  NVSHMEM build.
- Set `TORCH_CUDA_ARCH_LIST="9.0;10.0"` for the DeepEP step.
- Use a CUDA 13+ NGC base (already the case here).
- Confirm B300's EFA generation (likely v3 / 32× 200 Gbps); the libfabric
  transport doesn't care about generation but multi-NIC defaults may want
  retuning via `NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE`.

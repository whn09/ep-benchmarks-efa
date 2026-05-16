# uccl-ep-efa — UCCL-EP on AWS EFA

UCCL-EP is a fork of DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP)
maintained by `uccl-project/uccl` that replaces the NVSHMEM transport with
UCCL's own RDMA stack (ibverbs + libnl + numa). It runs on AWS EFA *without*
a patched NVSHMEM, talking directly to the EFA rdma-core verbs path.

This image builds UCCL-EP from source on `nvcr.io/nvidia/pytorch:26.04-py3`
(CUDA 13.2.1) and exposes both the native `uccl.ep` binding and the
`deep_ep_wrapper` shim, so DeepEP-style benchmarks
(`bench/test_internode.py`, `bench/test_low_latency.py`) work unchanged.

Validated on:
- 2× **p5.48xlarge** (H100 80GB × 8, **32 EFA v1 NICs × 100 Gbps**), us-east-2, 2026-05-16
- 2× **p5en.48xlarge** (H200 80GB × 8, **16 EFA v2 NICs × 200 Gbps**), us-east-2, 2026-05-16

UCCL's behaviour changes dramatically across the two — see section 4 for
numbers and section 5 for the cross-stack comparison. Same Dockerfile +
launcher work on both.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `uccl-ep-efa:dev` |
| `run_internode.sh` | 2-node launcher for `bench/test_internode.py` (normal) |
| `run_low_latency.sh` | 2-node launcher for `bench/test_low_latency.py` |
| `run_ll_pplx_style.sh` | 2-node launcher for `bench/test_low_latency_pplx.py` (same UCCL LL workload, but pplx-garden-style measurement: warmup+repeats, p50/p99 stats — directly comparable to pplx-garden bench output) |
| `monitor_efa.sh` | Per-NIC EFA bandwidth snapshot (`-s`, `-b`, `-d`, `-l`) |
| `sample_efa_bw.sh` | Time-series per-NIC sampler (writes a log file) |

## Host prerequisites

Same as `deepep-v1-efa`: NVIDIA driver 595+, EFA hardware, `/dev/gdrdrv`
present, intra-VPC SG with self-reference rule. See
`../deepep-v1-efa/README.md` for the gdrdrv recovery snippet.

---

## 1. Build the image (per node)

```bash
# from the host, sync this dir to each node first if not already there
rsync -avz /Users/henanwan/Documents/workspace/moonshot/uccl-ep-efa/ \
  P5EN-1:~/work/uccl-ep-efa/
rsync -avz /Users/henanwan/Documents/workspace/moonshot/uccl-ep-efa/ \
  P5EN-2:~/work/uccl-ep-efa/

# on each node:
ssh P5EN-1 'cd ~/work/uccl-ep-efa && docker build -t uccl-ep-efa:dev .'
ssh P5EN-2 'cd ~/work/uccl-ep-efa && docker build -t uccl-ep-efa:dev .'
```

`docker build` on p5.48xlarge takes ~6 min: ~80 s for the EFA installer +
GDRCopy, ~80 s for UCCL's CUDA extension (parallel nvcc), <30 s for system
deps, the rest is wheel build / install.

### Build-time component pins

| Component | Version |
|---|---|
| Base image | `nvcr.io/nvidia/pytorch:26.04-py3` (CUDA 13.2.1, torch 2.12, sm90) |
| EFA installer | 1.48.0 |
| GDRCopy | 2.5.2 (libgdrapi only; kernel module gdrdrv lives on host) |
| UCCL | `uccl-project/uccl@main` (recursive submodules) |

### UCCL build flags

| Var | Value | Notes |
|---|---|---|
| `EFA_HOME` | `/opt/amazon/efa` | UCCL's `setup.py` reads this |
| `TORCH_CUDA_ARCH_LIST` | `9.0+PTX` | H100 |
| `CPATH` | adds `/usr/local/cuda/targets/x86_64-linux/include/cccl` | NGC 26.04 quirk: CCCL headers aren't on the default CUDA include path |

UCCL links against `libibverbs`, `libnl-3`, `libnl-route-3`, `libnuma`, and
EFA's `libefa` — installed via `apt-get install rdma-core libibverbs-dev
libnl-3-dev libnl-route-3-dev libnuma-dev` plus the EFA installer.

### Build gotcha worth remembering

**Never resolve `import uccl` while cwd is `/opt/uccl`.** UCCL's `setup.py`
installs the wheel to site-packages, but the source tree at `/opt/uccl/uccl/`
is also a Python package. Resolving `python -c "import uccl;
print(uccl.__path__[0])"` from `/opt/uccl` returns the *source* path, so any
subsequent `cp ep.abi3.so $UCCL_DIR/` deposits the file into the source tree,
not site-packages, and `from uccl import ep` will then fail at runtime
(`ModuleNotFoundError: No module named 'uccl.ep'`).

Fix: `cd /tmp` (or anywhere outside the source tree) first:

```dockerfile
UCCL_DIR=$(cd /tmp && python3 -c "import uccl, os; print(os.path.dirname(uccl.__file__))")
```

---

## 2. Run the benchmarks

Get the leader node's private IP (the one the worker dials):

```bash
LEADER_IP=$(ssh P5EN-1 'hostname -I | awk "{print \$1}"')   # e.g. 172.31.45.156
echo "$LEADER_IP"
```

### Normal-mode benchmark (test_internode.py)

```bash
# Worker first (it dials in)
ssh P5EN-2 'cd ~/work/uccl-ep-efa && bash run_internode.sh 1 '"$LEADER_IP"'' &
# Leader
ssh P5EN-1 'cd ~/work/uccl-ep-efa && bash run_internode.sh 0 '"$LEADER_IP"''
```

`run_internode.sh <node_rank> <master_ip> [extra args]`. Defaults match
`bench/run_ep.sh` from upstream: 8 GPUs/node, 4096 tokens, hidden 7168,
288 experts, top-k 8. Total runtime ~2-3 min: 24 correctness checks then
an autotuning sweep.

### Low-latency benchmark (test_low_latency.py)

```bash
ssh P5EN-2 'cd ~/work/uccl-ep-efa && bash run_low_latency.sh 1 '"$LEADER_IP"'' &
ssh P5EN-1 'cd ~/work/uccl-ep-efa && bash run_low_latency.sh 0 '"$LEADER_IP"''
```

Defaults: 8 GPUs/node, 128 tokens, hidden 7168, 288 experts, top-k 8.
Different `MASTER_PORT` (29501) than normal so they don't collide.

### Runtime convention

UCCL tests use `torchrun` (not `WORLD_SIZE=#nodes` like DeepEP V1 — the
launchers handle this). Image-baked env defaults: `FI_PROVIDER=efa`,
`FI_EFA_USE_DEVICE_RDMA=1`, `LD_LIBRARY_PATH` includes `/opt/amazon/efa/lib`.

---

## 3. Monitor per-NIC EFA bandwidth during a run

`monitor_efa.sh` reads counters from `rdma statistic show` and prints per-NIC
tx/rx Gbps. For benchmark runs we need a *log-to-file* version that samples
N times, not the interactive `-b` mode — that's what `sample_efa_bw.sh` does.

### One-shot snapshot

```bash
ssh P5EN-1 'bash ~/monitor_efa.sh -s'      # printable table
ssh P5EN-1 'bash ~/monitor_efa.sh -b 2'    # 2-second instantaneous bandwidth
```

### Time-series sampling during a bench run

`sample_efa_bw.sh` (also in this dir) samples N intervals and writes a log:

```bash
# sync the helpers once (both live in this directory)
rsync -avz ./monitor_efa.sh ./sample_efa_bw.sh P5EN-1:~/
ssh P5EN-1 'chmod +x ~/monitor_efa.sh ~/sample_efa_bw.sh'
```

Then while the bench is running:

```bash
# Usage: sample_efa_bw.sh OUT N INTERVAL
# 30 samples × 2 s = 60 s window
ssh P5EN-1 'bash ~/sample_efa_bw.sh /tmp/uccl_efa_bw.log 30 2'
```

**Timing matters:** the bench takes ~2-3 min total but spends the first
~30-45 s on init/correctness tests (negligible NIC traffic) and the last
~10 s tearing down. Sleep 30 s after launching the bench, then start
sampling for ~60 s, to land squarely in the autotune sweep.

End-to-end recipe (used to produce the numbers below):

```bash
# 1. launch bench in background on both nodes
ssh P5EN-2 "cd ~/work/uccl-ep-efa && bash run_internode.sh 1 $LEADER_IP \
  > /tmp/uccl_normal_worker.log 2>&1" &
ssh P5EN-1 "cd ~/work/uccl-ep-efa && bash run_internode.sh 0 $LEADER_IP \
  > /tmp/uccl_normal_leader.log 2>&1" &

# 2. wait for steady state, then sample on the leader
ssh P5EN-1 'sleep 30 && bash ~/sample_efa_bw.sh /tmp/uccl_efa_bw.log 30 2'

# 3. inspect aggregate per-sample
ssh P5EN-1 'grep "^TOTAL" /tmp/uccl_efa_bw.log'

# 4. inspect per-NIC distribution at a peak sample
ssh P5EN-1 'awk "/=== sample 5 @/ {flag=1} /=== sample 6 @/ {flag=0} flag" \
  /tmp/uccl_efa_bw.log'
```

---

## 4. Validated numbers (2 nodes × 8 GPU = 16 ranks, 2026-05-16)

24 correctness checks (BF16/FP8 × top-k × async × previous, with
`dispatch_use_fp8=true`) pass on both instance types.

### Normal mode (best autotuned configs)

| Op | p5.48xlarge | p5en.48xlarge | Δ |
|---|---|---|---|
| Dispatch (BF16) RDMA BW | 48.72 GB/s | **60.64 GB/s** | **+24 %** |
| Dispatch (FP8)  RDMA BW | 31.96 GB/s | **46.50 GB/s** | **+45 %** |
| Combine RDMA BW         | 13.92 GB/s | **17.11 GB/s** | +23 % |

### Low-latency mode

| Per-rank metric | p5.48xlarge | p5en.48xlarge | Δ |
|---|---|---|---|
| Dispatch + Combine BW | 5.38 GB/s | **32.71 GB/s** | **+508 %** (6× higher) |
| Dispatch + Combine avg latency | ~4096 µs | **674 µs** | **6.1× lower** |
| Dispatch BW | ~2.3 GB/s | **36.21 GB/s** | **15.7× higher** |
| Combine BW | ~21 GB/s | **48.32 GB/s** | +130 % |
| Dispatch kernel time | ~170 µs avg | **207 µs avg** | similar (kernel-bound, not BW-bound) |
| Combine kernel time | ~190 µs avg | **301 µs avg** | similar |

**The p5 → p5en jump for UCCL-EP LL is enormous** — 15× lower dispatch
latency, 6× higher D+C bandwidth. UCCL's RDMA stack is much more
sensitive to per-NIC bandwidth and EFA SRD HW generation than DeepEP's
NVSHMEM-libfabric stack: where DeepEP V1 only gains ~14-22 % moving p5 →
p5en, UCCL gains an order of magnitude.

On p5en, UCCL-EP's LL dispatch latency (207 µs) is the *fastest* of all
stacks tested on this hardware, slightly beating pplx-garden (224 µs)
and 2.9× ahead of DeepEP V1 (602 µs).

### Per-NIC distribution during normal mode (32 NICs, leader)

| Window (sample 1-7) | UCCL-EP | DeepEP V1 (reference) |
|---|---|---|
| Active NICs | **32 / 32** | 32 / 32 |
| Per-NIC tx range (peak) | 56 - 70 Gbps | **80 - 83 Gbps** |
| Spread across NICs | ~14 Gbps (≈25 %) | ~3 Gbps (≈4 %) |
| Total tx (peak sample) | **1976 Gbps** ≈ 247 GB/s | **2776 Gbps** ≈ 347 GB/s |

Both stacks stripe across all 32 EFA NICs. UCCL is **not** single-NIC bound
— the gap is in *per-NIC utilisation* and *NIC-balance uniformity*, not
NIC count.

### pplx-style LL measurement on p5en (decode shape, EP=16)

UCCL ships a separate bench (`bench/test_low_latency_pplx.py`) that
runs the same UCCL LL workload but uses pplx-garden's measurement
methodology (warmup + N repeats, p50/p99 statistics over end-to-end
latency). This makes UCCL numbers directly comparable to pplx-garden's
bench output. Run via `run_ll_pplx_style.sh`.

| Metric (p5en, decode 128 tok, 288 experts, top-8, FP8) | Value |
|---|---|
| Dispatch p50 latency | **212 µs** |
| Dispatch BW | 35.78 GB/s |
| Dispatch send / recv (kernel) | 41 / 27 µs |
| Combine p50 latency | 324 µs |
| Combine BW | 45.30 GB/s |
| Combine send / recv (kernel) | 47 / 43 µs |

Side-by-side (same workload, same measurement):

| Op | UCCL pplx-style | pplx-garden | UCCL advantage |
|---|---|---|---|
| Dispatch p50 | **212 µs** | 222 µs | **-4.5 %** (UCCL faster) |
| Combine p50 | 324 µs | **245 µs** | +32 % (pplx-garden faster) |
| Dispatch BW | 35.78 GB/s | 34.0 GB/s | tied |
| Combine BW | 45.30 GB/s | **59.8 GB/s** | pplx-garden +32 % |

**Verdict on p5en LL/decode:** UCCL and pplx-garden are essentially
tied on dispatch (UCCL slightly faster) but pplx-garden is clearly
ahead on combine. The UCCL self-bench earlier reported dispatch 207 µs
/ combine 301 µs (different reporting metric, same workload) — those
match the pplx-style 212 / 324 numbers, confirming both UCCL bench
modes are consistent.

---

## 5. Side-by-side: UCCL-EP vs DeepEP V1 + amazon NVSHMEM

Same hardware, same launcher conventions, same test parameters
(`--num-tokens=4096 --hidden=7168 --num-topk=8 --num-experts=256/288`).

### Normal-mode RDMA bandwidth (per-rank effective)

**On p5en.48xlarge:**

| Op | UCCL-EP | DeepEP V1 | Gap |
|---|---|---|---|
| Dispatch BF16 | **60.64 GB/s** | 62.54 GB/s | -3 % |
| Dispatch FP8  | 46.50 GB/s | 54.98 GB/s | -15 % |
| Combine       | 17.11 GB/s | 58.48 GB/s | **-71 %** |

**On p5.48xlarge:**

| Op | UCCL-EP | DeepEP V1 | Gap |
|---|---|---|---|
| Dispatch BF16 | 48.72 GB/s | 59.94 GB/s | -19 % |
| Dispatch FP8  | 31.96 GB/s | 48.17 GB/s | -34 % |
| Combine       | 13.92 GB/s | 53.92 GB/s | **-74 %** |

UCCL closes the dispatch gap to DeepEP V1 dramatically on p5en (BF16:
-19 % → -3 %). Combine remains weak — that's a kernel-level issue, not
a transport one (see "Reading the gap" below).

### Aggregate NIC throughput (peak sample, p5.48xlarge)

| Stack | Total tx | Peak per-NIC | NIC utilisation (vs ~100 Gbps cap) |
|---|---|---|---|
| UCCL-EP | 1976 Gbps | 56-70 Gbps | ~70 % |
| DeepEP V1 | 2776 Gbps | 80-83 Gbps | **~83 %** |

(Per-NIC monitoring on p5en not re-collected; expect higher per-NIC
utilisation given UCCL's ~24 % BW jump.)

### Low-latency mode (per rank)

**On p5en.48xlarge:**

| Metric | UCCL-EP | DeepEP V1 | UCCL advantage |
|---|---|---|---|
| Dispatch+Combine BW | **32.71 GB/s** | 20.4 GB/s | **+60 %** |
| Dispatch+Combine avg latency | **674 µs** | 1083 µs | **38 % lower** |
| Dispatch latency | **207 µs** | 602 µs | **2.9× lower** |
| Combine latency | 301 µs | 561 µs | **1.9× lower** |

**On p5.48xlarge** (UCCL was much weaker):

| Metric | UCCL-EP | DeepEP V1 | DeepEP advantage |
|---|---|---|---|
| Dispatch+Combine BW | 5.38 GB/s | 16.50 GB/s | +207 % |
| Dispatch+Combine avg latency | 4096 µs | 1333 µs | 3.07× lower |
| Dispatch latency | ~3200 µs | ~700 µs | 4.5× lower |

**The p5 vs p5en story flips entirely**: on p5 (EFA v1) DeepEP V1
dominates UCCL on LL by 3-4×; on p5en (EFA v2) UCCL is **2.9× faster
than DeepEP V1** on LL dispatch. UCCL is much more sensitive to the
EFA SRD HW generation than DeepEP's libfabric path is.

### Reading the gap (p5)

Per-NIC monitoring (section 3) rules out the simple "no multi-NIC striping"
hypothesis: UCCL pushes traffic across all 32 NICs on p5. The remaining
contributors look like:

1. **Lower per-NIC utilisation.** UCCL peaks at ~70 Gbps/NIC vs DeepEP V1's
   ~83 Gbps/NIC. With 32 NICs that's an aggregate gap of ~800 Gbps before
   any kernel-side overhead. Likely candidates: fewer in-flight QPs per NIC,
   smaller WQE batching, or CPU-thread contention on the proxy.
2. **Uneven NIC balance.** UCCL's spread is ~14 Gbps across NICs (25 %),
   DeepEP V1's is ~3 Gbps (4 %). Suggests UCCL's PE→NIC mapping is less
   PCIe-distance-aware than NVSHMEM libfabric's round-robin.
3. **Combine kernel overhead (normal mode).** UCCL's normal-mode combine
   gets only ~14 GB/s/rank vs DeepEP's ~54, while UCCL's *low-latency*
   combine matches DeepEP (~21 vs ~20 GB/s/rank). The bottleneck is in the
   normal combine *kernel* (reduction path), not the transport.
4. **Synchronous-style dispatch in low-latency mode.** UCCL dispatch send
   ≈170 µs vs DeepEP V1's ≈58 µs. The 3× gap looks like extra CPU-side
   coordination per chunk (UCCL's proxy thread), not raw RDMA latency.
   Worth profiling with `nsys` if anyone wants to close it.

These are first-pass numbers with default args. Both stacks have tunables
(`NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE` for DeepEP, UCCL has equivalent
NIC-selection envs) that haven't been swept here.

---

## Caveats / known issues

- **`UndefinedVar: Usage of undefined variable '$CPATH'`** during build —
  harmless docker buildkit warning; `CPATH` is set in `ENV` immediately
  before being used.
- **PR #828 `patch_uccl_ep_empty_tensor.sh`** is *not* applied. It fixes a
  path used by sglang's two-batch overlap when `topk_idx_ptr==0`, which
  the bench scripts never trigger. Add the patch only if you intend to
  integrate with sglang.
- **`pip install --no-deps` for the `deep_ep_wrapper` shim** is intentional:
  the wrapper's `setup.py` lists `deep_ep` as a runtime dep, but the wrapper
  *is* `deep_ep`. Letting pip resolve deps would clobber the wrapper.
- **`monitor_efa.sh -b` only prints one sample.** Use the
  `sample_efa_bw.sh` helper in this directory if you need a time-series
  log during a benchmark run.

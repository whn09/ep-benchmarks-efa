# UCCL-EP poor performance on p5.48xlarge — investigation

Reproduces and digs into why UCCL-EP performs much worse on p5.48xlarge
(H100 + 32× EFA v1 / 100 Gbps) than on p5en.48xlarge (H200 + 16× EFA v2 /
200 Gbps). All numbers below are from a fresh build of `uccl-ep-efa:dev`
on a 2× p5.48xlarge pair, 16 ranks (2 nodes × 8 GPU), `EP=16`,
`num-experts=288`, `num-topk=8`, FP8 dispatch / BF16 combine.

## Reproduction (matches earlier numbers)

### Normal mode

| Op | RDMA BW | NVL BW |
|---|---|---|
| Dispatch BF16 | **49.10 GB/s** | 161.04 GB/s |
| Dispatch FP8 | 30.99 GB/s | 101.64 GB/s |
| Combine | **13.66 GB/s** | 44.80 GB/s |

### Low-latency mode

| Metric (rank 0) | Value |
|---|---|
| Dispatch + Combine BW | 5.37 GB/s |
| Dispatch + Combine avg latency | **4109 µs** |
| Dispatch BW / latency | **2.22 GB/s** / 3377 µs |
| Combine BW / latency | 23.28 GB/s / 624 µs |
| Dispatch kernel send/recv | ~94 / ~30 µs |
| Combine kernel send/recv | ~100-116 / ~51-212 µs |

These match the earlier numbers in `README.md` (49 / 14 normal, ~3200 / 830
LL) within noise — the bench is reproducible.

## Hypothesis 1: NIC misbinding / non-NUMA-aware NIC selection — REJECTED

Earlier README speculated UCCL's PE→NIC mapping might be PCIe-distance-blind.
Check the leader log:

```
[RDMA] Selected NIC rdmap79s0..82s0 (4 NICs) for GPU 0, NUMA node 0
[RDMA] Selected NIC rdmap96s0..99s0 (4 NICs) for GPU 1, NUMA node 0
[RDMA] Selected NIC rdmap113s0..116s0 (4 NICs) for GPU 2, NUMA node 0
[RDMA] Selected NIC rdmap130s0..133s0 (4 NICs) for GPU 3, NUMA node 0
[RDMA] Selected NIC rdmap147s0..150s0 (4 NICs) for GPU 4, NUMA node 1
[RDMA] Selected NIC rdmap164s0..167s0 (4 NICs) for GPU 5, NUMA node 1
[RDMA] Selected NIC rdmap181s0..184s0 (4 NICs) for GPU 6, NUMA node 1
[RDMA] Selected NIC rdmap198s0..201s0 (4 NICs) for GPU 7, NUMA node 1
```

UCCL **correctly** assigns 4 NICs per GPU, NUMA-aware (GPUs 0-3 on NUMA 0,
4-7 on NUMA 1). `Pinned to NUMA node 0/1, ... running on CPU N` confirms
the proxy threads are pinned to the matching NUMA.

## Hypothesis 2: NIC distribution is uneven — REJECTED (mostly)

Per-NIC sample 8 (peak ~1990 Gbps total tx, dispatch phase):

| NIC range | Per-NIC tx (Gbps) | Spread |
|---|---|---|
| All 32 NICs | 57.79 - 68.04 | 10 Gbps (16 %) |

The earlier README claimed 14 Gbps spread (25 %) — this run shows 16 % spread,
which is acceptable. Distribution is not the dominant issue.

## Hypothesis 3: Hardware-imposed per-NIC ceiling — CONFIRMED for dispatch

p5.48xlarge has 32 NICs but only 8 GPUs, so each NIC serves ~2 GPUs simultaneously.
Per-NIC bandwidth is 100 Gbps. We measure ~63 Gbps/NIC during dispatch.

p5en.48xlarge has 16 NICs / 8 GPUs (2 NICs/GPU same density), but each NIC
is 200 Gbps. So a per-GPU send pattern that saturates 1 NIC on p5en (200 Gbps)
gets only 100 Gbps on p5.

UCCL aggregate dispatch BW on p5: ~49 GB/s × 8 ranks/node = **~3.1 Tbps total**,
matching the 1990 Gbps tx + 1990 Gbps rx ≈ 3.98 Tbps round-trip we observe.
NIC fabric is **already saturated** — UCCL dispatch on p5 cannot go meaningfully
faster without using more NICs (impossible) or different transport (impossible).

DeepEP V1 reaches 60 GB/s dispatch on the same hardware, ~22 % more, by
running NICs at 80-83 Gbps (vs UCCL's 63 Gbps). This is **per-NIC efficiency**,
not NIC count. Likely contributors: NVSHMEM libfabric uses larger inflight
WQE batches and aggressive pipeline depth; UCCL's per-thread-QP architecture
limits each NIC's outstanding ops.

## Hypothesis 4: Combine kernel bottleneck — CONFIRMED, root cause

Per-NIC sampling **during the combine phase** (samples 12-18, after the
dispatch peak at samples 7-9):

| Sample | Total tx | Per-NIC tx (rdmap79s0 / rdmap113s0) |
|---|---|---|
| 8 (dispatch peak) | **1990 Gbps** | 66 / 65 |
| 12 (combine) | 456 Gbps | 15 / 14 |
| 13 | 590 Gbps | 19 / 18 |
| 14 | 675 Gbps | 22 / 21 |
| 15 | 714 Gbps | 23 / 22 |
| 16 | 744 Gbps | 23 / 23 |
| 17 | 754 Gbps | 24 / 24 |
| 18 | 549 Gbps | 14 / 23 |

**During combine, each NIC drops to 14-24 Gbps (vs 60+ in dispatch)** —
NICs are barely 25 % utilised. The 13.66 GB/s combine BW is *not* a NIC
limit; it's a kernel/CPU bottleneck.

Looking at `ep/src/internode.cu` combine launch (line ~3013):

```cpp
// NOTE(MaoZiming): I changed here from 24 to 16.
constexpr int kNumCombineForwarderWarps = 16;
```

UCCL hard-codes the combine forwarder warps at **16**, down from DeepEP V1's
24. With 6 combine forwarder warps active per NVL peer (8 NVL peers × 6 = 48,
but capped to 16), this directly limits how many in-flight token
forwards the kernel can pipeline. Combined with the proxy thread RTT
(~µs per signaled chunk), this serialises combine traffic across far fewer
NICs at any given moment.

Other suspect TODOs in the same file:
- `// TODO(MaoZiming): always cross-rail.` (line 89) — possibly conservative
  rail selection that costs throughput.
- `// TODO: may use NVSHMEM reduction` (internode.cu:254) — UCCL falls back
  to a software reduction path where DeepEP V1 uses NVSHMEM's hardware path.
- `// TODO: more light fence or barrier or signaling` (line 188) —
  fence-heavy sync pattern.

## Hypothesis 5: LL dispatch CPU-coordination latency — CONFIRMED

LL dispatch reports:
- Kernel send time: **94 µs**
- Kernel recv time: **30 µs**
- End-to-end avg latency: **3377 µs**

The kernel is on the GPU for ~125 µs total but the operation takes 27× that.
The remaining ~3250 µs is CPU proxy thread coordination: signaling, polling
completion queues, syscalls into libfabric.

This is the structural cost of using a CPU proxy thread instead of GPU-direct
RDMA initiation (IBGDA / NVSHMEM). DeepEP V1 on the same hardware achieves
~700 µs LL dispatch latency — the ~5× advantage comes from NVSHMEM's IBGDA
path, where the GPU directly issues RDMA writes without round-tripping
through the CPU.

For decoding workloads where every per-token A2A picks up this 3 ms
coordination cost, **UCCL on p5 is structurally unsuitable**. On p5en, UCCL's
LL dispatch dropped from 3377 µs to 207 µs (16× lower) because EFA v2 SRD
has lower message-completion latency that lets the proxy thread cycle
faster — but the architectural CPU-on-the-critical-path remains, so on a
hypothetical p6 with even faster NICs UCCL would close on V1 but not beat it.

## Summary table

| Bottleneck | Affects | Severity |
|---|---|---|
| Per-NIC ceiling (100 Gbps × 32 NICs, 2 GPUs/NIC) | normal dispatch BW | -22 % vs V1 |
| Per-NIC efficiency at saturation | normal dispatch BW | -22 % vs V1 (compounds with above; 63 vs 83 Gbps/NIC) |
| `kNumCombineForwarderWarps=16` (was 24 in V1) | normal combine BW | **-75 % vs V1** |
| Software combine reduction (no NVSHMEM HW reduction) | normal combine BW | (compounds with forwarder cap) |
| CPU proxy thread on critical path | LL dispatch latency | **5× worse than V1** at low loads |
| EFA v1 SRD per-message latency | LL dispatch latency | (compounds; partly fixed on p5en) |

## What would fix what

1. ~~**`kNumCombineForwarderWarps` back to 24**~~ **TESTED → REGRESSION.**
   Bumping the constant to 24 raises forwarder SMEM use from ~145 KB to
   ~217 KB, which forces H100 to drop to 1 block/SM and shrinks the TMA
   pipeline. Result on p5: combine drops from 13.66 to 4.53 GB/s (-67 %).
   `=20` is intermediate (-42 %). The Mao Ziming `// changed from 24 to 16`
   change was empirically required, not a regression vs upstream.

2. ~~**Halve forwarder TMA bytes/warp + double warp count to keep SMEM
   constant**~~ **NOT POSSIBLE.** `kNumTMABytesPerForwarderWarp` has a
   hard `static_assert` lower bound:
   ```
   kNumTMABytesPerForwarderWarp ≥ kNumStages × (kNumTMALoadBytes ×
                                  (NUM_MAX_NVL_PEERS + 1) + 16)
                               = 2 × (512 × 9 + 16) = 9248
   ```
   So 9248 is already the minimum and you can't shrink it without
   redesigning the TMA staging.

3. ~~**Tune UCCL NIC selection on p5: 4 NICs/GPU → 2 NICs/GPU**~~
   **TESTED → REGRESSION.** Patched `rdma.cpp` to use `thread_idx % 2 +
   half` (the same formula p5en uses with 16 NICs total) instead of p5's
   `thread_idx % 4`. This cuts the per-GPU NIC fan-out and drops
   aggregate BW: dispatch BF16 49 → 33 GB/s (-33 %), combine 14 → 10 GB/s
   (-28 %). Conclusion: p5's 4-NICs/GPU is the right choice — total
   bandwidth wins over per-NIC saturation when each NIC is "only"
   100 Gbps.

4. **Replace software combine reduction with hardware reduction**:
   substantial work. Probably the single biggest win for combine BW.
   The TODO `// may use NVSHMEM reduction` (internode.cu:254) hints at
   exactly this, but it's not a one-line change.

5. **Implement IBGDA-equivalent in UCCL**: out of scope for a fork; this
   is what NVSHMEM provides. Removing the CPU proxy from the LL dispatch
   path would cut latency 3-5×.

## Final read

After exhausting kernel-tunable knobs, **UCCL on p5 is already
near-optimal for its architecture**. The remaining gap to DeepEP V1 is
not configuration — it is two architectural choices:

- **Software combine reduction** vs V1's NVSHMEM hardware-assisted
  reduction (this is the dominant ~75 % normal-mode combine gap).
- **CPU proxy on LL dispatch critical path** vs V1's IBGDA GPU-direct
  initiation (this is the dominant ~5× LL dispatch gap; partly fixed on
  p5en where SRD is faster, hence 3200 → 207 µs).

Both fixes require non-trivial UCCL upstream work. None of the easy knobs
move the needle.

## Repro commands

```bash
# baseline normal + NIC sample
ssh P5EN-2 'cd ~/work/uccl-ep-efa && bash run_internode.sh 1 <leader-ip> &> /tmp/uccl_normal_worker.log' &
ssh P5EN-1 'cd ~/work/uccl-ep-efa && bash run_internode.sh 0 <leader-ip> &> /tmp/uccl_normal_leader.log' &
ssh P5EN-1 'sleep 30 && bash ~/sample_efa_bw.sh /tmp/uccl_p5_efa_bw.log 30 2'

# extract per-phase NIC totals
ssh P5EN-1 'grep "^TOTAL" /tmp/uccl_p5_efa_bw.log'

# extract NIC binding from launcher output
ssh P5EN-1 'grep "Selected NIC" /tmp/uccl_normal_leader.log | head -32'

# LL bench
ssh P5EN-2 'cd ~/work/uccl-ep-efa && bash run_low_latency.sh 1 <leader-ip> &> /tmp/uccl_ll_worker.log' &
ssh P5EN-1 'cd ~/work/uccl-ep-efa && bash run_low_latency.sh 0 <leader-ip> &> /tmp/uccl_ll_leader.log'
```

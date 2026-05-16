# deepep-v2-uccl-style — DeepEP V2 with UCCL-EP-comparable params

This directory re-runs **DeepEP V2** (from the `deepep-v2-efa:dev` image
in [`../deepep-v2-efa/`](../deepep-v2-efa/)) using the **same workload
parameters as UCCL-EP's published p5en numbers**, so the bandwidth /
latency results are directly comparable.

It mirrors the methodology of an earlier internal `benchmark-uccl-style`
exercise that targeted the `antonai-work/deepep-v2-efa-base:v0.2.5-sm90a`
image. Here we use the upstream `deepseek-ai/DeepEP@main` build instead
to confirm the perf is the same as that fork's PR #612 patches.

## Why a separate launcher

`../deepep-v2-efa/run_test_ep.sh` uses DeepEP V2's own defaults
(`num_experts=256`, `num_topk=6`, `num_tokens=4096`). UCCL-EP's bench
uses **`num_experts=288`, `num_topk=8`**, two different `num_tokens`
(4096 for prefill, 128 for decode), and a specific set of V2 knobs that
most closely resembles a low-latency profile:

```
--prefer-overlap-with-compute 0
--allow-multiple-reduction 0
--do-cpu-sync 1
```

DeepEP V2 has no separate `low_latency_dispatch` / `low_latency_combine`
kernels (they exist in V1 but were unified into `ElasticBuffer.dispatch`
in V2). The flags above pick the V2 dispatch path that's closest in
spirit to V1's LL path, but **the result is not strictly apples-to-apples**
with UCCL's `test_low_latency.py`.

## Files

| File | Purpose |
|---|---|
| `run_v2_prefill.sh` | 2-node prefill launcher (4096 tokens, V2 HT-style knobs) |
| `run_v2_decode.sh` | 2-node decode launcher (128 tokens, V2 LL-style knobs) |

Both reuse the `deepep-v2-efa:dev` image — **build that image first**
(see `../deepep-v2-efa/README.md`).

## Run

```bash
LEADER_IP=$(ssh P5EN-1 'hostname -I | awk "{print \$1}"')

# Prefill
ssh P5EN-2 "cd ~/work/deepep-v2-uccl-style && bash run_v2_prefill.sh 1 $LEADER_IP" &
ssh P5EN-1 "cd ~/work/deepep-v2-uccl-style && bash run_v2_prefill.sh 0 $LEADER_IP"

# Decode
ssh P5EN-2 "cd ~/work/deepep-v2-uccl-style && bash run_v2_decode.sh 1 $LEADER_IP" &
ssh P5EN-1 "cd ~/work/deepep-v2-uccl-style && bash run_v2_decode.sh 0 $LEADER_IP"
```

`EP_NIC_NAME=rdmap85s0` (p5en) is the default; override to `rdmap79s0`
on p5 or any other NIC name listed by `ls /sys/class/infiniband/`.

Knobs available via env: `NUM_SMS`, `NUM_QPS`, `EP_RDMA_GBS`,
`EP_NIC_NAME`. Default values let DeepEP V2 auto-pick.

## Validated numbers (2× p5en.48xlarge, 16 ranks, 2026-05-16)

### Prefill (4096 tokens, hidden 7168, top-k 8, 288 experts, FP8/BF16)

| Op | SO (cross-node EFA) | SU (intra-node NVLink) | Latency |
|---|---|---|---|
| Dispatch         | **5 GB/s** | 15 GB/s | ~13.4 ms |
| Expanded dispatch | 5 GB/s | 15 GB/s | ~13.4 ms |
| Cached dispatch  | 5 GB/s | 15 GB/s | ~13.4 ms |
| Combine (best)   | 20-21 GB/s | 67 GB/s | 5.7-5.9 ms |
| Combine (worst across ranks) | 15 GB/s | 49 GB/s | 7.8-7.9 ms |
| Reduced combine  | 20 GB/s | 66-67 GB/s | 5.8 ms |

### Decode (128 tokens, hidden 7168, top-k 8, 288 experts, FP8/BF16)

| Op | SO (cross-node EFA) | SU (intra-node NVLink) | Latency |
|---|---|---|---|
| Dispatch         | **1 GB/s** | 3-4 GB/s | ~1.69 ms |
| Expanded dispatch | 1 GB/s | 3-4 GB/s | ~1.66 ms |
| Combine          | 6-7 GB/s | 6-7 GB/s | 1.6-1.9 ms |
| Reduced combine  | 8-9 GB/s | 8-9 GB/s | 1.6-1.7 ms |

## Side-by-side: V2 vs UCCL-EP (both on p5en.48xlarge, 16 ranks)

UCCL-EP numbers from `moonshot/uccl/ep/README.md`:

### Prefill / Normal mode

| Stack | Dispatch BW (SO) | Dispatch latency | Combine BW (SO) | Combine latency |
|---|---|---|---|---|
| UCCL-EP | **50 GB/s** | **1196 µs** | 18 GB/s | 6379 µs |
| **DeepEP V2 (this image)** | 5 GB/s | 13400 µs | **~20 GB/s** | **5800 µs** |
| DeepEP V2 (`antonai-work/v0.2.5-sm90a`) | 5 GB/s | 12710 µs | 16 GB/s | 7274 µs |

- **Combine is essentially tied** (V2 ~20 GB/s vs UCCL 18 GB/s).
- **Dispatch is 10× off** (V2 5 GB/s vs UCCL 50 GB/s, latency 11× higher).
- Our self-built V2 image performs **on par with the antonai-work v0.2.5
  fork** that includes PR #612 patches. The gap is fundamental to V2's
  GIN-based dispatch path, not a build issue.

### Decode / Low-latency mode

| Stack | Dispatch BW (SO) | Dispatch latency | Combine BW (SO) | Combine latency |
|---|---|---|---|---|
| UCCL-EP | **36 GB/s** | **226 µs** | **48 GB/s** | **293 µs** |
| **DeepEP V2 (this image)** | 1 GB/s | 1690 µs | 6-7 GB/s | 1700 µs |
| DeepEP V2 (`antonai-work/v0.2.5-sm90a`) | 1 GB/s | 3620 µs | 5 GB/s | 2164 µs |

- **Decode is severely off**: V2 dispatch latency 1690 µs vs UCCL's 226 µs (**7.5× slower**).
- Our build is ~2× faster on decode latency than the antonai-work v0.2.5
  fork (1690 vs 3620 µs dispatch, 1700 vs 2164 µs combine), suggesting
  ofi-nccl GIN improvements since v0.2.5 help the small-message path.
- DeepEP V2 has no LL-specialised kernel; for decode-shape on EFA the
  recommended stack remains UCCL-EP or pplx-garden.

## Key observations

1. **V2 main + self-built ofi-nccl GIN ≈ antonai-work v0.2.5-sm90a**
   on prefill (5 / 17 vs 5 / 16 GB/s). PR #612 patches (which the
   antonai-work fork carries) are essentially "the EFA path that
   eventually landed on V2 main" — no major upstream improvement since.
2. **Decode is improved** (1.7 ms vs 3.6 ms for dispatch latency on
   our build) thanks to ofi-nccl `master` GIN commits since v0.2.5.
3. **Dispatch BW gap to UCCL is structural for V2 on EFA**:
   - `num_allocated_qps` is capped to 2 to avoid GIN ring overflow,
     limiting per-PE concurrency.
   - V2's NCCL Gin path inherits NCCL's per-channel scheduling, which
     batches less aggressively than UCCL's hand-rolled multi-NIC
     aggregation.
4. **V2 is best for very large EP (>EP128) with low SM budget**, not
   for EP=16 where V1 / UCCL / pplx already saturate hardware.

For the cross-stack comparison see [`../README.md`](../README.md).

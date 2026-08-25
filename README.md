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

⭐ = start here. 🔒 = **local-only, not committed** (bulk campaign artefacts); the row is
kept for the record and its paths are intentionally not links.

| Directory | Library | Backend | EFA support path |
|---|---|---|---|
| [`deepep-v1-efa/`](deepep-v1-efa/) | DeepEP V1 (`rauteric/DeepEP@remove-fence`) | NVSHMEM libfabric | amazon-contributing/upstream-to-nvshmem `devel_enriched` (libfabric remote transport, multi-NIC RR) |
| [`uccl-ep-efa/`](uccl-ep-efa/) | UCCL-EP (`uccl-project/uccl@main`) | UCCL Rust RDMA stack | direct ibverbs / libfabric, multi-NIC at app layer |
| [`deepep-v2-efa/`](deepep-v2-efa/) | DeepEP V2 (`deepseek-ai/DeepEP@main`) | NCCL Gin | aws-ofi-nccl `master` (`ncclGinPlugin_v11+`) |
| [`deepep-v2-uccl-style/`](deepep-v2-uccl-style/) | DeepEP V2 (same image as above) | NCCL Gin | reuses `deepep-v2-efa:dev`; runs prefill / decode with **UCCL-EP-comparable params** so V2 numbers are directly comparable to UCCL's published p5en results |
| [`pplx-garden-efa/`](pplx-garden-efa/) | pplx-garden (`perplexityai/pplx-garden@main`) | custom Rust libfabric | direct libfabric + multi-NIC aggregation (`fabric-lib`) |
| [`deepep-v1-efa-b300/`](deepep-v1-efa-b300/) | DeepEP V1 (B300 variant) | NVSHMEM libfabric | adds `sm_100` to `CMAKE_CUDA_ARCHITECTURES` and `TORCH_CUDA_ARCH_LIST` |
| [`uccl-ep-efa-b300/`](uccl-ep-efa-b300/) | UCCL-EP (B300 variant) | UCCL Rust RDMA stack | adds `10.0+PTX` to `TORCH_CUDA_ARCH_LIST`; **also requires the rdma.cpp fix from [uccl-project/uccl#950](https://github.com/uccl-project/uccl/pull/950)** which is applied at build-time on the patched `:dev`/`:b300` images |
| [`deepep-v2-efa-b300/`](deepep-v2-efa-b300/) | DeepEP V2 (B300 variant) | NCCL Gin | adds `10.0` to `TORCH_CUDA_ARCH_LIST`, sets `EP_NIC_NAME=rdmap101s0` |
| `deepep-v2-efa-gdaki-b200/` 🔒 | DeepEP V2 — **AWS EFA-team fork** (`Xuan-1998/DeepEP@dev`) | NCCL Gin (+ GDAKI) | EFA hybrid dispatch/combine kernels, GIN QP/context auto-tuner, dedicated proxy warp; source NCCL `v2.30.7-1` + aws-ofi-nccl `master --enable-gdaki`. Runs today on the stock EFA stack (non-GDAKI GIN); GDAKI additionally needs a **counting-event-capable host `efa.ko`**, which no container can supply |
| [`pplx-garden-efa-b300/`](pplx-garden-efa-b300/) | pplx-garden (B300 variant) | custom Rust libfabric | adds `10.3a+PTX` to `TORCH_CUDA_ARCH_LIST` and patches `p2p-all-to-all/a2a-kernels/build.rs` to emit `compute_103a/sm_103a` (upstream hardcodes `sm_100a` only, which fails at runtime on B300's `sm_103`) |
| [`deepep-v2-efa-official/`](deepep-v2-efa-official/) ⭐ | DeepEP V2 — **public release** ([`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP)) | NCCL Gin + GDAKI | **Published packages only.** EFA installer **1.50.0** supplies efa.ko 3.3.0 + libfabric 2.6.0amzn1.0 + rdma-core 64.0amzn0 + aws-ofi-nccl 1.21.1 in one shot, so GDAKI comes up with **no source-built NCCL, no source-built aws-ofi-nccl and no patched kernel module**. ⚠️ But it *loads* GDAKI without *using* it: 1.50.0 registers a second, type-2 proxy-assisted GIN plugin and picks that by default, so **`NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0` are required for performance** (without them, prefill is 9% slower and decode 2.2× at 2 nodes / 5.4× at 4 nodes). With them set, performance matches or beats the hand-built path |

**If you want DeepEP V2 on EFA today, start with
[`deepep-v2-efa-official/`](deepep-v2-efa-official/)** — it is the released stack on
published packages and needs no patched components. `deepep-v2-efa-gdaki-b200/` remains
the record of how the capability was brought up before 1.50.0 existed (and is still the
home of the B200/B300 campaigns), but it is **🔒 local-only — kept out of git** because it
is 673 MB of campaign artefacts; its paths below are deliberately not links.
`deepep-v2-efa/` builds upstream `deepseek-ai/DeepEP`
and predates GDAKI entirely. A full Chinese runbook — host install, image build, prefill
bandwidth and decode latency tests, troubleshooting table — is at
[`deepep-v2-efa-official/docs/runbook_zh.md`](deepep-v2-efa-official/docs/runbook_zh.md).

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
| p5en.48xlarge | H200 80GB × 8 | 16 | 200 Gbps | 3.2 Tbps | v2 (newer SRD) |
| **p6-b300.48xlarge** | **B300 SXM6 (sm_103) × 8** | 16 | **400 Gbps** | **6.4 Tbps** | **v3** |
| **p6-b200.48xlarge** | **B200 SXM6 (sm_100) × 8** | 8 | **400 Gbps** | 3.2 Tbps | **v3** |

The 2026-05 rows below are 2 nodes × 8 GPU = **16 ranks**, us-east-2, 2026-05-16;
the 2026-08 GDAKI rows are later campaigns on other clusters and carry their own
date, node count and config in the footnote under each table — read that before
quoting a cell. Same Dockerfiles and launchers work on both instances; only
`EP_NIC_NAME` differs (`rdmap79s0` on p5, `rdmap85s0` on p5en — both auto-detectable).

## Side-by-side results

**What bold means in the three results tables below:** exactly the best cell in that column,
compared **only within the same config group** — the 2026-05/BF16 rows and the
2026-08 GDAKI rows are different shapes and different clocks (see each table's
footnote), so each group is bolded on its own. Two bold cells in a column means a
genuine tie; no bold at all means no arm is a defensible winner there. Near-ties are
called out in the prose, not by bold.
**⭐ marks the fastest stack on that hardware among the rows the table's own config
statement admits.** That admits the GDAKI row in the decode table (which already
tolerates 256 experts) but **not** in the throughput table: no GDAKI campaign ever
ran 4096 tokens, so those 8192-token FP8 cells cannot be starred against a
4096/BF16 column.

### Normal mode / "throughput" — RDMA bandwidth (per-rank effective)

Test config: 4096 tokens, hidden 7168, top-k 8, ~256 experts, FP8 →
BF16. Larger numbers = better. *DeepEP V2 row uses the
`deepep-v2-uccl-style/` launcher so it shares params with the others
(288 experts, top-8).*

| Stack | p5 Dispatch BF16 | p5 Combine | p5en Dispatch BF16 | p5en Combine | **B300 Dispatch BF16** | **B300 Combine** |
|---|---|---|---|---|---|---|
| **DeepEP V1 + amazon NVSHMEM** | **59.94 GB/s** ⭐ | **53.92 GB/s** ⭐ | **62.54 GB/s** ⭐ | **58.48 GB/s** ⭐ | **109.84 GB/s** ⭐ | **101.72 GB/s** ⭐ |
| UCCL-EP | 48.72 GB/s | 13.92 GB/s | 60.64 GB/s | 17.11 GB/s | 90.03 GB/s | 58.99 GB/s |
| DeepEP V2 + aws-ofi-nccl GIN (CPU-proxy, 2026-05) | ~2 GB/s | ~12 GB/s | 5 GB/s | 20 GB/s | ~~4 GB/s~~ | ~~21-26 GB/s~~ |
| **DeepEP V2 + EFA GDAKI** (route B, 2026-08) † | — | — | **81.25 GB/s** | **65.75 GB/s** | **125 GB/s** | **131 GB/s** |

† **The `DeepEP V2 + GIN` row above is obsolete — do not quote it.** It is the
CPU-proxy path from 2026-05. The GDAKI row replaces it and is **~30× the b300
dispatch number** in the struck-through cell. Config differences you must carry:
**no cell in the GDAKI row is 4096/BF16** — the b300 cells are **8192 tokens, FP8
dispatch, 24 SM** and the p5en cells are **8192 tokens, FP8 dispatch, 12 SM**
(81.25 GB/s = 1504 µs dispatch, 65.75 = 3592 µs combine, 399.8 MB per rank,
`results/p5en_ours_20260813/summary.txt`). **The released EFA 1.50.0 package reaches the same
number without building anything from source, but only if you set two env vars**:
`libnccl-net-ofi.so` registers both a proxy-assisted GIN backend (type 2) and
`Libfabric_GDAKI` (type 5), and NCCL picks type 2 by default. With
`NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0` the released image measures
**81.2 GB/s / 1502.9 µs** on the same args versus 74.0 / 1644.0 on the default env —
all-rank means, 16 ranks, see
[`deepep-v2-efa-official/results/p5en_2n4n_20260825/summary.txt`](deepep-v2-efa-official/results/p5en_2n4n_20260825/summary.txt).
The QP layout is not a lever: `--num-allocated-qps 5` moves plain 2-node prefill dispatch
−0.8% on type 2 and +0.4% on type 5, i.e. the sign flips with the *backend*, not with node
count; where it does cost something is cached dispatch (+9.6% on type 5) and 4-node decode
dispatch, +19.3%: 1003.2 → 1197.0 µs. Finally,
the GB/s is `test_ep.py`'s **SO** denominator. **SO counts intra-node destinations
too, so halve it for a wire rate** — 125 GB/s SO = 62.5 GB/s of the 100 GB/s
per-GPU wire. UCCL's `GB/s (RDMA)` in the row above uses the **same** denominator
(verified: byte counts agree to 0.1–1.5% on both ops), so it needs halving too;
its own 8192-token FP8 numbers on the same nodes are 66.47 / 68.39 GB/s.
Absolute time at that matched config (24 SM, 8192 tok, both FP8): DeepEP V2+GDAKI
**978.8 µs dispatch / 1788.1 µs combine** vs UCCL-EP **1815 / 3421 µs** (a
best-of over its chunk sweep) ⇒ **1.85× and 1.91× in DeepEP's favour**. Full
methodology and per-row provenance for this table:
`deepep-v2-efa-gdaki-b200/results/b300_20260813/UCCL_H2H_README.md` and
`deepep-v2-efa-gdaki-b200/docs/b300_实测报告_zh.md` (🔒 local-only).
**The UCCL-EP row is a same-stack re-check and reproduces within 5%**
(4096 tok BF16: 94.42 vs 90.03 dispatch, 56.93 vs 58.99 combine).

**Where 4096 and 8192 each come from** — because it decides what this table can and
cannot conclude. **4096 is nobody's chosen shape**: it is both benches' argparse
default (V1 `tests/test_internode.py:374` and V2 `tests/elastic/test_ep.py:577` are
each `default=4096`), inherited by the 2026-05 launchers — `deepep-v1-efa/run_internode.sh`
forwards no token count at all, while `uccl-ep-efa/run_internode.sh` and
`deepep-v2-uccl-style/run_v2_prefill.sh` hard-code 4096. **8192/FP8 is a chosen
shape**: it is upstream DeepEP V2's own published config (its README — 8K tokens per
batch, 7168 hidden, top-8, FP8 dispatching, BF16 combining, "following V3's
configuration") and it is what AWS's reference campaign ran (19 of 19 runs at
`--num-tokens=8192`), which is what makes our GDAKI numbers comparable to both at
once. **No GDAKI campaign ever ran 4096** — every GDAKI prefill arm is 8192 and every
decode arm is 128, plus a decode-shape sweep at 1 / 8 / 32 / 512 / 1024. Since DeepEP
V1 in turn has no 8192-token run, **V2-vs-V1 throughput is unmeasured in either
direction on both p5en and b300**, which is why the GDAKI row is bolded only inside
its own group and carries no ⭐ even though 125 > 109.84. The cheap way to close that
is from the V1 side, not ours: V1 already publishes FP8 dispatch at 4096 (48.17 p5 /
54.98 p5en, `deepep-v1-efa/README.md`), so **one V1 run at `--num-tokens 8192` with
FP8 dispatch on the existing image** would make the two rows directly comparable.

**B300 highlights**: First time on EFA we see >100 GB/s per-rank in
both directions for V1 (≈ 2.5× upstream IB README's 43 GB/s). UCCL-EP
combine jumps from 17 → 59 GB/s — the per-NIC ceiling that bottlenecked
combine on p5/p5en is gone with v3 NICs at 400 Gbps each.

Reference: upstream DeepEP README on H800 + CX7 IB reports 43 GB/s for
both dispatch and combine at EP=16. EFA matches or exceeds that on the
mature stack (V1) thanks to wider per-rank NIC fan-out.

UCCL-EP's published p5en numbers (`uccl-project/uccl/ep`) are
**50 GB/s prefill dispatch and 18 GB/s combine**; our reproduction is
60.64 / 17.11, i.e. **+21.3 % on dispatch and −4.9 % on combine** — the
same ballpark, but only combine is inside 10 %.

### Low-latency / "decode" — end-to-end dispatch + combine latency

Test config: 128 tokens, hidden 7168, top-k 8, 288 experts (256 for V1
since its bench defaults differ). Lower = better.

| Stack | p5 Dispatch | p5 Combine | p5en Dispatch | p5en Combine | **B300 Dispatch** | **B300 Combine** |
|---|---|---|---|---|---|---|
| pplx-garden (decode shape) | **402 µs** ⭐ | **517 µs** (p50) ⭐ | 222 µs (p50) | **245 µs** (p50) | **140 µs** (p50) ⭐ | **149 µs** (p50) ⭐ |
| UCCL-EP (`run_ll_pplx_style.sh`, pplx-style measurement) | 1281 µs (p50) | 1428 µs (p50) | 212 µs (p50) | 324 µs (p50) | 171 µs (p50) | 219 µs (p50) |
| UCCL-EP (`run_low_latency.sh`, UCCL self-report) | ~3200 µs | ~830 µs | **207 µs** | 301 µs | 277 µs | **149 µs** ⭐ |
| DeepEP V1 + amazon NVSHMEM (PR#9 reverted) | 585 µs | 639 µs | (n/a) | (n/a) | 691 µs | 416 µs |
| DeepEP V1 + amazon NVSHMEM (PR#9 in) | 765 µs | 641 µs | 602 µs | 561 µs | (not run) | (not run) |
| DeepEP V2 (`deepep-v2-uccl-style/` decode, CPU-proxy, 2026-05) | 2700 µs | 2100 µs (avg) | 1690 µs | 1700 µs | ~~1925 µs~~ | ~~1700 µs~~ |
| **DeepEP V2 + EFA GDAKI** (route B, op-level, 2026-08) ‡ | — | — | **151.6 µs** ⭐ | **189.6 µs** ⭐ | 200.4 µs | **160.1 µs** |
| **UCCL-EP** re-run on the GDAKI stack, same nodes (2026-08) ‡ | — | — | 220.2 µs | 301.0 µs | 103–136 µs | 229–367 µs |

‡ **The `DeepEP V2` decode row above is obsolete — do not quote it.** GDAKI route B
is **~10× faster** than the struck-through b300 cells. Read these three rows
together, with three caveats: (1) the GDAKI cells are **op-level** (DeepEP V2 splits
each op into two kernels — `dispatch_impl` + `dispatch_copy_epilogue_impl` — and
both are counted; kernel-only, the tuned p5en pair is 135.1 + 185.1 = 320.2 µs, so the
epilogues are worth +16.5 on dispatch and +4.6 on combine and must not be dropped when
comparing against the single fused kernels that UCCL-EP and the v1 IB references time);
(2) **all four GDAKI cells are 256 experts, not this table's 288** — b300 at 55 SM with
`:parts1`, p5en at 12 SM with `kMaxParts=1` + `EP_NUM_SUB_PARTS=1`. Steps: p5en
**341.2 µs** (151.6+189.6), b300 **360.5 µs** (200.4+160.1). The p5en UCCL-EP cells are
likewise our own 256-expert run on the same two nodes so the shape matches; at UCCL's own
288-expert default we measure 211.4 + 298.8 = **510.2 µs** against their **published 519**
(dispatch 6.5% under, combine 2.0% over, step 1.7% under), so their published row is not
a luckier machine. The *released-package* image reaches this shape faster still — 12 SM,
`ec623f3` + [PR#1](https://github.com/amazon-contributing/DeepEP/pull/1) +
[#2](https://github.com/amazon-contributing/DeepEP/pull/2) + `EP_NUM_SUB_PARTS=1` on the
type-5 backend measures **dispatch 106.4 / combine 162.1 µs** (all-rank, 16 ranks,
committed data in [`deepep-v2-efa-official/`](deepep-v2-efa-official/)) — consistent with
that campaign's own pin-vs-package result (the packaged stack beats the hand-built pin by
1.74–1.78× on decode dispatch), but it times ops with `test_ep.py`'s own clock rather than
`bench_kineto`, so treat it as magnitude corroboration and do not mix it into these cells;
(3) **UCCL-EP's per-op LL timings are not reproducible on b300**
— two runs of the identical config gave dispatch 136.12 → 103.15 µs (−24%) and combine
228.55 → 367.21 µs (+61%) while its own back-to-back dispatch+combine loop was
stable to 0.1% (439.34 → 438.94 µs). **Quote 439 µs for UCCL b300 decode**; the
b300 per-op cells above are ranges, not points, which is why the b300 dispatch
column of this 2026-08 pair carries no bold — UCCL's 103–136 µs does beat our
200.4, but not reproducibly. This does *not* apply to the p5en UCCL
cells, which reproduce their published per-op values to 6.5%/2.0% as above. On that
reproducible metric DeepEP
V2+GDAKI is **1.22× faster** (360.5 vs 439 µs), but **UCCL's dispatch alone is
1.5–1.9× faster than ours** while **our combine is 1.4–2.3× faster than its** — and
UCCL spends 4 CPU proxy threads per rank (32/node) to get there, where GDAKI route B
spends **zero**. **On p5en the same head-to-head is 1.53× in DeepEP's favour** (341.2 vs
521.1 µs step, both at 256 experts on the same two nodes) and there we win *both* ops —
151.6 vs 220.2 dispatch, 189.6 vs 301.0 combine. So the b300 dispatch deficit is specific
to that arm, not architectural; note though that the b300 arm runs at 55 SM and p5en at
12, so this is two observations rather than one controlled sweep.
The old `run_low_latency.sh` UCCL b300 row (277/149 µs) is at 288
experts with a different launcher and a pre-uccl#950 rev, so it is not like-for-like
with these and is left as-is.

**B300 highlights**: pplx-garden hits **140 µs / 149 µs (p50)** — the
fastest LL on EFA across all generations. UCCL-EP combine drops to
149 µs (self-report) / 219 µs (pplx-style), again leveraging v3's
per-NIC headroom. DeepEP V1 LL doesn't track the same way: dispatch
is roughly flat (765 → 691 µs across hardware) because its CPU proxy
RTT is already short, while combine improves more (641 → 416 µs)
where NVLink + reduce kernel scaling is bandwidth-bound. **DeepEP V2 is the fastest of the three
on B300 once GDAKI is used**: on the GIN CPU-proxy backend it is the
bottleneck, and route B (GDAKI, GPU-initiated) removes it —
**360.5 µs step** at 128 tokens, i.e. **1.22× faster than UCCL-EP** and
within **1.25×** of pplx-garden's 289 µs (140+149), measured on the same
two nodes. pplx-garden has not been re-run on the GDAKI EFA stack, so
that last comparison still crosses a stack boundary; the DeepEP-vs-UCCL
one does not.

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

### pplx-garden prefill (4096 tokens)

| Instance | Dispatch p50 | Combine p50 | Dispatch BW | Combine BW |
|---|---|---|---|---|
| p5en | 3122 µs | 5365 µs | 77.6 GB/s | 87.6 GB/s |
| **B300** | **1725 µs** | **2672 µs** | **140 GB/s** | **177 GB/s** |

Matches upstream README's published p5en numbers (3197 / 5379) within 2 %.
B300 prefill is ~1.8× faster than p5en, tracking the per-NIC bandwidth
doubling.

## Recommendations

| Workload | p5.48xlarge | p5en.48xlarge | **p6-b300.48xlarge** |
|---|---|---|---|
| **MoE training** (HT all-to-all, large batches) | DeepEP V1 + amazon NVSHMEM | DeepEP V1 + amazon NVSHMEM | **DeepEP V1** (110 GB/s dispatch, 102 GB/s combine) |
| **MoE inference, decode** (per-token A2A) | pplx-garden | UCCL-EP **or** pplx-garden (≈ tied) | **pplx-garden** (140/149 µs, old stack) **or DeepEP V2 + GDAKI** (360.5 µs step, 0 CPU threads, and it beats UCCL-EP's 439 µs on the same nodes) |
| **MoE inference, prefill** (large batches) | DeepEP V1 (HT mode) or pplx-garden | DeepEP V1 (HT mode) or pplx-garden | **DeepEP V2 + GDAKI** (0.98/1.79 ms at **8192** tok, 24 SM) — pplx's 1.7/2.7 ms is at **4096** tok on the old stack, and DeepEP V2 beats UCCL-EP 1.9× at matched 8192/24 SM |
| **Provider-portable** (also AMD / CX7 / etc) | UCCL-EP | UCCL-EP | UCCL-EP (B300 NIC selection needs uccl#950 — **now merged**, present at rev `dc676e58`; builds natively for sm_103 with `TORCH_CUDA_ARCH_LIST=10.3`) |
| **Very large EP (>EP128, low SM budget)** | watch DeepEP V2 + ofi-nccl GIN (still maturing) | **DeepEP V2 + EFA GDAKI** (route B; 341 µs decode step, 12 SM) | **DeepEP V2 + EFA GDAKI** (route B; 360.5 µs decode step, 2766.9 µs prefill step @8192 tok) |

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

`deepep-v2-efa-official/` pins differently and deliberately: base
`nvidia/cuda:13.0.2-devel-ubuntu24.04`, torch `2.13.0+cu130`, pip
`nvidia-nccl-cu13==2.31.2` + `nvidia-nvshmem-cu13==3.7.2`, gdrcopy v2.5, **EFA
installer 1.50.0**, and DeepEP pinned to commit `ec623f3`. Two of those pins are
load-bearing and non-obvious: NCCL must be **≥ 2.31** (below that DeepEP asserts
compile-time and run-time NCCL versions are exactly equal), and the installer must be
**1.50.0** — 1.49.0 already reports `efa-direct` and already exports
`ncclGinPlugin_v11`/`v13`, so neither of those is a valid version check; what it lacks
is the `COMP_CNTR` capability in `efa-abi.h` and any `comp_cntr` symbol in
rdma-core 63.0's libibverbs.

## What's NOT in this repo

- IB / RoCE numbers — those would require different hardware. We cite
  upstream DeepEP README's H800+CX7 numbers and Amazon's internal
  RoCE-vs-EFA perf table inside individual READMEs as references.
- ~~DeepEP V2 LL benches — currently V2 has limited LL value-add on EFA
  due to ofi-nccl GIN plugin maturity; HT-only smoke is included.~~
  **Stale as of 2026-08.** V2 decode on EFA is measured in
  [`deepep-v2-efa-official/`](deepep-v2-efa-official/) (106–365 µs dispatch at
  128 tokens, 16 ranks, depending on GIN backend and part geometry) and in
  `deepep-v2-efa-gdaki-b200/`. The plugin-maturity caveat was about
  aws-ofi-nccl's CPU-proxy GIN; installer 1.50.0's 1.21.1 does GDAKI.
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
- `:dev` — current build for **p5/p5en** (sm_90).
- `:2026-05-17` — frozen snapshot of p5/p5en `:dev`.
- `:b300` — **B300 build** (sm_103). Use these on
  p6-b300.48xlarge instances. UCCL-EP `:b300` includes the
  [PR #950](https://github.com/uccl-project/uccl/pull/950) NIC-selection
  fix; pplx-garden `:b300` includes the `compute_103a/sm_103a` build.rs
  patch (upstream only emits `sm_100a`, which fails at runtime on B300).
- `:b300-2026-05-18` — frozen B300 snapshot.
- For `deepep-v1-efa` only:
  - `:revert-pr9` and `:revert-pr9-2026-05-17` — alias for the
    current p5/p5en `:dev` with NVSHMEM PR #9 reverted (faster LL on EFA,
    see [`deepep-v1-efa/INVESTIGATION_pr9_revert.md`](deepep-v1-efa/INVESTIGATION_pr9_revert.md)).
  - `:pr9-baseline` — image **with** PR #9 in NVSHMEM (the previous
    `:dev` pre-2026-05-17). Use this if you need the unsolicited-write
    CQ-overflow protection enabled (e.g. non-DeepEP-V1 workloads).

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

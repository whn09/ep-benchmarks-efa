# main vs PR #1+#2 vs PR #8+#9 — prefill and decode, 2 and 4 nodes

Three trees of `amazon-contributing/DeepEP`, one hardware setup, one measurement
path, 158 logs. Every number below comes out of `make_3arm_tables.py` in this
directory; its full output is checked in as `tables.txt` and the raw per-node logs
as `logs/`.

The tables below are Markdown, so they *were* placed by hand. `check_comparison.py`
re-derives all 139 of them from `logs/` as `(arm, nodes, tokens, knob, op) -> value`
and exits nonzero on any drift — grepping for a number cannot catch one pasted into
the wrong row, and it cannot catch `logs/` changing without the prose being updated:

```bash
EPRUNS=./logs python3 check_comparison.py    # -> 139 claims checked, 0 MISMATCHES
```

## The arms

| arm | image tag | `BUILD_REF` | what it is |
|---|---|---|---|
| `main` | `deepep-v2-efa-official:sm90-54fffef` | `54fffeff810723f574c574b1790dff189f3c6ffb` | `main` as of 2026-08-31, the baseline |
| `PR #1+#2` | `deepep-v2-efa-official:sm90-bfbdd15` | `bfbdd15ff448783f877cb2210cb3246c8452b05e` | PR #2's head. #1 is a **strict subset** of #2 (`compare` says `ahead_by 1`), so one image covers both |
| `PR #8+#9` | `deepep-v2-efa-official:sm90-3c737dc` | `3c737dcf0da5889ba7efd26e05b4808307cc38af` | PR #9's head. #8 is `cdec5214`, #9 is `cdec5214` + `3c737dcf`, so again one image covers both |

`BUILD_REF` is read out of every log, not assumed: the image tag is a name someone
chose, `BUILD_REF` is what `git rev-parse HEAD` said inside the build.

## Setup

4 × p5en.48xlarge (H200, sm_90, 8 GPU/node, 16 × 200 Gb/s EFA), EFA installer
1.50.0, `efa.ko` 3.3.0g, driver 595.91.07, `gdrdrv` loaded. 12 SM,
`NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0` (EFA GDA), 8 ranks/node,
`--prefer-overlap-with-compute=0`, `--test-first-only`, 3 rotated reps per cell,
all ranks pooled from every node's log.

Two knob settings per arm: each tree's shipped default part geometry, and
`EP_NUM_SUB_PARTS=1`. The second is not a side experiment — forwarding that env var
to the JIT is PR #1's entire contribution, so an arm measured only at its default
is not measured at its operating point. See *EP_NUM_SUB_PARTS=1* below.

Two things that scope every number here:

- `--test-first-only` pins the first entry of `enumerate_ep_modes()`, which is
  **FP8 dispatch at `expert_alignment=128`**. There is no flag that selects BF16
  dispatch alone, so nothing below is a BF16 result.
- `--ignore-local-traffic` is **OFF**, matching every number already published
  under `results/` and in `docs/runbook_zh.md`. So the `SO` column includes
  intra-node traffic and is not a wire rate — at 2N it reads 81 GB/s against a
  50 GB/s per-GPU scale-out ceiling. No wire-utilisation column is printed;
  comparing three arms on one machine does not need one. Time is the metric,
  `MB/rank` is printed as the byte denominator so a GB/s that moved can be
  attributed to bytes or to time.

Baseline sanity: `main` at 2N/12 SM/8192 tok lands at 1499.8 µs against the
1502.9 µs published in `results/p5en_2n4n_20260825/summary.txt` — 0.2% apart, so
the environment did not drift under the EFA installer update.

## The headline

**The two PR sets fix different ops and neither dominates.**

| | prefill dispatch | prefill combine | decode dispatch | decode combine |
|---|---|---|---|---|
| PR #1+#2 | flat | flat | **−33.3%** (−37.5% at `EP_NUM_SUB_PARTS=1`) | flat |
| PR #8+#9 | flat | **−13.4%** | −27.2% | **−11.6%** |

(2 nodes, `reduced combine` for the combine columns, Δ vs `main` at default
geometry. Each arm is at its own default unless the cell says otherwise.)

PR #1+#2 is a **decode-dispatch** patch: it moves that one op by a third and
touches nothing else. PR #8+#9 is mostly a **combine** patch: it moves combine at
both token counts, and gets a smaller decode-dispatch win as well.

On the metric a MoE layer actually pays — `dispatch + reduced combine`, the pair
of calls it issues — PR #8+#9 wins decode at 2 nodes, and the two arms are at
parity at 4 nodes once PR #1+#2 is given `EP_NUM_SUB_PARTS=1`:

| 128 tok, layer total | 2N | Δ | 4N | Δ |
|---|---:|---:|---:|---:|
| `main`, default | 348.8 µs | — | 437.6 µs | — |
| PR #1+#2, default | 292.0 µs | −16.3% | 423.6 µs | −3.2% |
| PR #1+#2, `EP_NUM_SUB_PARTS=1` | 284.8 µs | −18.4% | **409.7 µs** | **−6.4%** |
| PR #8+#9, default | **273.0 µs** | **−21.7%** | **408.2 µs** | **−6.7%** |

At 2 nodes PR #1+#2 has the faster dispatch (106.1 vs 123.5 µs at each arm's best
knob) and still the slower layer (284.8 vs 273.0 µs), because PR #8+#9's combine
win is worth more than PR #1+#2's extra dispatch win. Quoting dispatch alone picks
the wrong arm. At 4 nodes the layer totals are 409.7 vs 408.2 µs — 0.4% apart,
which three reps cannot separate.

**Both patches shrink at 4 nodes, but PR #1+#2 shrinks far less than its default
geometry suggests:**

| Δ vs `main` | 2N | 4N |
|---|---:|---:|
| PR #1+#2, decode dispatch, default | −33.3% | −7.5% |
| PR #1+#2, decode dispatch, `EP_NUM_SUB_PARTS=1` | **−37.5%** | **−15.0%** |
| PR #8+#9, decode dispatch | −27.2% | −7.1% |
| PR #8+#9, prefill redComb | −13.4% | −2.2% |
| PR #8+#9, decode redComb | −16.6% | −6.5% |
| PR #8+#9, prefill layer total | −9.9% | −1.5% |
| PR #8+#9, decode layer total | −21.7% | −6.7% |

Read at default geometry, both PRs converge to the same 4-node decode dispatch
(170.3 vs 171.0 µs) and the story is "both wins mostly evaporate". Read at PR
#1+#2's own operating point, its 4-node dispatch is 156.4 µs — a −15.0% win that
is more than twice PR #8+#9's, and the win-shrinkage from 2N to 4N is 2.5× rather
than 4.4×. The knob does not change which arm wins the layer; it changes by how
much the dispatch win decays with scale.

## Prefill (8192 tok)

**Prefill dispatch is untouched by both PRs, at both scales.** All three arms land
within 0.2% of each other — 1498.4–1499.8 µs at 2N and 3961.2–3969.2 µs at 4N —
against a rep-to-rep spread of about 0.2%, so these are the same number.

**PR #8+#9 moves prefill combine, and only at 2 nodes.**

| 2N, 8192 tok | `main` | PR #1+#2 | PR #8+#9 |
|---|---:|---:|---:|
| dispatch | 1499.8 µs | 1498.7 µs (−0.1%) | 1498.4 µs (−0.1%) |
| cached dispatch | 1588.8 µs | 1587.4 µs (−0.1%) | **1496.2 µs (−5.8%)** |
| combine | 3587.8 µs | 3575.8 µs (−0.3%) | **3172.5 µs (−11.6%)** |
| reduced combine | 4238.0 µs | 4239.6 µs (+0.0%) | **3670.5 µs (−13.4%)** |
| **layer total** | 5737.8 µs | 5738.3 µs (+0.0%) | **5168.8 µs (−9.9%)** |

At 4 nodes the same rows read −0.1% / −7.2% / −1.4% / −2.2% / −1.5%: the combine
win almost entirely evaporates, while the `cached dispatch` win does not.

`cached dispatch` is worth a separate note. On `main` it is *slower* than plain
dispatch (1588.8 vs 1499.8 µs at 2N, 4254.0 vs 3965.2 at 4N) — the cached path
carries a penalty. PR #8+#9 removes that penalty rather than adding a speed-up:
its `cached dispatch` (1496.2 / 3946.2 µs) is at parity with its own plain
dispatch. This is the one op where PR #8+#9 helps prefill dispatch at 4 nodes
(−7.2%), and it is the only prefill row at 4N with a delta above noise.

## Decode (128 tok)

| 2N, 128 tok | `main` | PR #1+#2 | PR #8+#9 |
|---|---:|---:|---:|
| dispatch | 169.6 µs | **113.1 µs (−33.3%)** | 123.5 µs (−27.2%) |
| cached dispatch | 166.2 µs | **107.1 µs (−35.6%)** | 120.2 µs (−27.7%) |
| combine | 162.5 µs | 162.5 µs (+0.0%) | **143.6 µs (−11.6%)** |
| reduced combine | 179.2 µs | 178.9 µs (−0.2%) | **149.5 µs (−16.6%)** |
| **layer total** | 348.8 µs | 292.0 µs (−16.3%) | **273.0 µs (−21.7%)** |

| 4N, 128 tok | `main` | PR #1+#2 | PR #8+#9 |
|---|---:|---:|---:|
| dispatch | 184.0 µs | 170.3 µs (−7.5%) | 171.0 µs (−7.1%) |
| cached dispatch | 178.9 µs | 172.2 µs (−3.8%) | **167.6 µs (−6.4%)** |
| combine | 244.8 µs | 244.6 µs (−0.1%) | **234.6 µs (−4.2%)** |
| reduced combine | 253.6 µs | 253.3 µs (−0.1%) | **237.2 µs (−6.5%)** |
| **layer total** | 437.6 µs | 423.6 µs (−3.2%) | **408.2 µs (−6.7%)** |

The 2N decode-dispatch row reproduces the earlier `bfbdd15` campaign: −33.3% here
against −33.5% in `results/p5en_2n4n_20260825/summary.txt` TABLE 7, on a different
EFA installer.

PR #1+#2's decode combine is flat to within 0.2% at both scales — expected, since
its diff is `csrc/jit/compiler.hpp` plus `hybrid_dispatch_unordered.cuh`. It is
not a combine patch and does not accidentally act as one.

## `EP_NUM_SUB_PARTS=1`

**This knob is a lever on exactly one of the three trees, and it is PR #1 that
makes it one.** Probed inside each image:

```bash
docker run --rm --entrypoint bash <img> -c \
  'grep -rl EP_NUM_SUB_PARTS /opt/DeepEP/csrc /opt/DeepEP/deep_ep'
```

| arm | files that read it | effect of setting it |
|---|---|---|
| `main` | `deep_ep/include/deep_ep/impls/hybrid_dispatch_unordered.cuh` only | **silently inert** |
| PR #1+#2 | `csrc/jit/compiler.hpp` **and** the `.cuh` | forwarded to the JIT, takes effect |
| PR #8+#9 | `deep_ep/include/deep_ep/impls/hybrid_dispatch_unordered.cuh` only | **silently inert** |

The `.cuh` reads the macro; `compiler.hpp` is what puts the env var *into* the JIT
flags. Without the `compiler.hpp` half the env var is read by nothing and the
default part geometry is compiled in regardless. So `main` and PR #8+#9 are run
with the knob as an **inertness control**: they must land on their own default, and
a delta there would mean the environment moved rather than the knob.

The control holds. At 2N/128 tok, setting the knob moves `main` by +0.1% on
dispatch and −0.1% on `reduced combine`, and PR #8+#9 by −0.5% and +0.1% — all
inside rep-to-rep spread:

| 2N, 128 tok, dispatch | default | `EP_NUM_SUB_PARTS=1` | Δ |
|---|---:|---:|---:|
| `main` | 169.6 µs | 169.8 µs | +0.1% (inert) |
| PR #1+#2 | 113.1 µs | **106.1 µs** | **−6.2%** |
| PR #8+#9 | 123.5 µs | 122.9 µs | −0.5% (inert) |

**On the PR #1+#2 arm the knob is worth more at 4 nodes than at 2.** Decode, PR
#1+#2 only (the other two arms are inert, so there is nothing to measure):

| PR #1+#2, 128 tok | 2N default | 2N knob | Δ | 4N default | 4N knob | Δ |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 113.1 | **106.1** | **−6.2%** | 170.3 | **156.4** | **−8.1%** |
| expanded dispatch | 113.2 | 105.7 | −6.7% | 170.2 | 155.4 | −8.7% |
| cached dispatch | 107.1 | 103.2 | −3.6% | 172.2 | 155.1 | −9.9% |
| combine | 162.5 | 162.3 | −0.2% | 244.6 | 244.6 | +0.0% |
| reduced combine | 178.9 | 178.7 | −0.1% | 253.3 | 253.3 | −0.0% |

Combine is untouched at both scales, which is what a dispatch-geometry knob should
do. The 2N `106.1 µs` reproduces the `106.4 µs` measured on the earlier EFA
installer in `results/p5en_2n4n_20260825/`.

**Prefill: the knob costs dispatch at 2 nodes and pays for itself in combine.**

| PR #1+#2, 8192 tok | 2N default | 2N knob | Δ | 4N default | 4N knob | Δ |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 1498.7 | 1565.0 | **+4.4%** | 3969.2 | 3974.8 | +0.1% |
| cached dispatch | 1587.4 | 1636.6 | +3.1% | 4253.5 | 4259.1 | +0.1% |
| combine | 3575.8 | 3515.7 | −1.7% | 7868.3 | 7554.4 | **−4.0%** |
| reduced combine | 4239.6 | 4175.9 | −1.5% | 7942.3 | **7695.1** | **−3.1%** |
| **layer total** | 5738.3 | 5740.9 | +0.0% | 11911.5 | **11669.8** | **−2.0%** |

At 2 nodes the +4.4% on prefill dispatch (66 µs) and the −1.5% on prefill
`reduced combine` (64 µs) cancel: the layer total moves by 2.6 µs, i.e. not at all.
At 4 nodes the dispatch cost disappears (+0.1%) and the combine win grows to −3.1%,
so the layer total drops 241.7 µs. The two `reduced combine` distributions do not
overlap — default 7930.4 / 7935.7 / 7960.7 µs against knob 7674.8 / 7685.8 /
7724.6 µs — so the 4-node prefill combine win is real and not rep noise.

A prefill combine that moves under a *dispatch* geometry knob is not explained
here. It is reported because it is reproducible across three reps at 4 nodes and
because it is what makes the knob a net win on prefill rather than a trade.

**Each arm at its own operating point.** Decode:

| 128 tok | knob | dispatch | Δ disp | redComb | layer total | Δ total |
|---|---|---:|---:|---:|---:|---:|
| `main`, 2N | default | 169.6 | — | 179.2 | 348.8 | — |
| PR #1+#2, 2N | `=1` | 106.1 | −37.5% | 178.7 | 284.8 | −18.4% |
| PR #8+#9, 2N | default | 123.5 | −27.2% | 149.5 | **273.0** | **−21.7%** |
| `main`, 4N | default | 184.0 | — | 253.6 | 437.6 | — |
| PR #1+#2, 4N | `=1` | **156.4** | **−15.0%** | 253.3 | 409.7 | −6.4% |
| PR #8+#9, 4N | default | 171.0 | −7.1% | 237.2 | **408.2** | **−6.7%** |

Prefill:

| 8192 tok | knob | dispatch | Δ disp | redComb | layer total | Δ total |
|---|---|---:|---:|---:|---:|---:|
| `main`, 2N | default | 1499.8 | — | 4238.0 | 5737.8 | — |
| PR #1+#2, 2N | `=1` | 1565.0 | +4.3% | 4175.9 | 5740.9 | +0.1% |
| PR #8+#9, 2N | default | 1498.4 | −0.1% | 3670.5 | **5168.8** | **−9.9%** |
| `main`, 4N | default | 3965.2 | — | 7947.3 | 11912.5 | — |
| PR #1+#2, 4N | `=1` | 3974.8 | +0.2% | 7695.1 | **11669.8** | **−2.0%** |
| PR #8+#9, 4N | default | 3961.2 | −0.1% | 7773.2 | 11734.5 | −1.5% |

So at 4 nodes PR #1+#2 with the knob is the **better prefill arm** (−2.0% vs
−1.5%) and ties PR #8+#9 on decode (409.7 vs 408.2 µs). At 2 nodes PR #8+#9 wins
both. Neither result is visible at default geometry.

Two operational consequences:

- **Set it.** On the PR #1+#2 tree there is no scale or token count at which the
  default geometry beats `EP_NUM_SUB_PARTS=1` on layer total. Its worst case is 2N
  prefill, where it is a 2.6 µs wash out of 5738 µs.
- **Merging PR #9 without PR #1 removes the lever.** #9's own tree does not carry
  the `compiler.hpp` forwarding, so on a #9-only build the knob is inert. This is
  concrete input to the open review question of whether PR #2 alone is enough: #2
  alone is what *uses* the geometry, #1 is what makes it reachable without editing
  a header, and editing the header instead is unsafe here — the JIT cache key
  hashes `flags` but not included header content, so a header-only patch can be
  served a stale cubin.

One outlier is excluded from the tables above and is recorded in `tables.txt`'s
AUDIT section: `pr893c737dc_2N_12sm_128tok_subparts1` rep3 came back ~4× slow on
all 16 ranks of both nodes (dispatch 495.5 µs against a 123.1 µs median) at a
byte-identical `docker run` — same image, same env, same `#SM`/`#QPs`, same `GIN`
line. It was replaced by a rep4, so that cell still rests on three reps. The
policy is mechanical: a rep more than 25% off its cell's **median** is dropped from
the mean and printed, and the log stays in `logs/`.

## Scaling, both axes

The time ratio alone cannot separate a stack that scales badly from one that is
simply moving more data, so both are printed.

| prefill, 8192 tok | 2N µs | 4N µs | 4N/2N time | 2N MB/rank | 4N MB/rank | 4N/2N bytes |
|---|---:|---:|---:|---:|---:|---:|
| dispatch, `main` | 1499.8 | 3965.2 | 2.64× | 399.8 | 444.9 | 1.11× |
| dispatch, PR #1+#2 | 1498.7 | 3969.2 | 2.65× | 399.8 | 444.9 | 1.11× |
| dispatch, PR #8+#9 | 1498.4 | 3961.2 | 2.64× | 399.8 | 444.9 | 1.11× |
| redComb, `main` | 4238.0 | 7947.3 | 1.88× | 767.1 | 853.7 | 1.11× |
| redComb, PR #1+#2 | 4239.6 | 7942.3 | 1.87× | 767.1 | 853.7 | 1.11× |
| redComb, PR #8+#9 | 3670.5 | 7773.2 | **2.12×** | 767.1 | 853.7 | 1.11× |

| decode, 128 tok | 2N µs | 4N µs | 4N/2N time | 2N MB/rank | 4N MB/rank | 4N/2N bytes |
|---|---:|---:|---:|---:|---:|---:|
| dispatch, `main` | 169.6 | 184.0 | 1.08× | 5.9 | 6.1 | 1.04× |
| dispatch, PR #1+#2 | 113.1 | 170.3 | **1.51×** | 5.9 | 6.1 | 1.04× |
| dispatch, PR #8+#9 | 123.5 | 171.0 | 1.38× | 5.9 | 6.1 | 1.04× |
| redComb, `main` | 179.2 | 253.6 | 1.42× | 11.3 | 11.7 | 1.04× |
| redComb, PR #1+#2 | 178.9 | 253.3 | 1.42× | 11.3 | 11.7 | 1.04× |
| redComb, PR #8+#9 | 149.5 | 237.2 | **1.59×** | 11.3 | 11.7 | 1.04× |

Read the ratio columns as a *cost* of being patched, not a defect. Both PRs make
2N faster without making 4N slower in absolute time, so their scaling ratio gets
worse precisely because their 2N denominator got smaller. `main`'s decode
dispatch scales at 1.08× only because it was already 169.6 µs at 2N — 57 µs of
which the patches show is removable.

The byte columns are `num_scaleup_bytes` (the log's `bytes` field), so they are
the SU denominator and not cross-node traffic. Cross-node bytes would have to be
recovered as `SO × time`.

## Configuration each arm actually ran

| arm | `BUILD_REF` | `#SM` | `#QPs` |
|---|---|---|---|
| `main` | `54fffeff…` | 12 | 11/11 |
| PR #1+#2 | `bfbdd15f…` | 12 | 11/11 |
| PR #8+#9 | `3c737dcf…` | 12 | **13/13** |

`#QPs` is the GIN context count. PR #9 raises `kDefaultGinContextCnt` 11 → 13, so
13/13 here is the patch landing, not a misconfiguration. It also invalidates the
runbook's standing claim that `#QPs` is always 11/11 — that was true of every tree
measured before #9.

**PR #8+#9's win is not attributed.** The `3c737dc` image bundles four changes and
this campaign measures their sum:

1. deletion of the `num_channels_per_sm = std::min<int>(num_channels_per_sm, 4)`
   clamp in `csrc/elastic/buffer.hpp` — 4 → 8 channels/SM at 12 SM, so the arm
   runs 96 channels where the other two run 48. This does not show up in `#SM`.
2. `kDefaultGinContextCnt` 11 → 13, i.e. the `#QPs` row above.
3. cooperative forward warp pairs (`kNumFwWarpsPerChannel`, `pair_free_seq` /
   `pair_half_done_seq`).
4. remote-first two-sweep scheduling — this is PR #8 on its own.

Changes 1 and 3 are gated on `not prefer_overlap_with_compute`, and this campaign
always passes `--prefer-overlap-with-compute=0`, so both are active on exactly the
measured path. Splitting them needs images, not env vars — with one exception: an
arm at `--prefer-overlap-with-compute=1` would disable 1 and 3 while leaving 2 and
4 in place, which brackets the channel-count contribution without a rebuild.

## Node layering

`combine` splits by machine and one node's log is a sample of one layer, not of
the run. Per-node means, max−min across nodes:

| | `main` | PR #1+#2 | PR #8+#9 |
|---|---:|---:|---:|
| 2N prefill redComb | 8.8–9.1% | 7.8–9.3% | **1.8–3.8%** |
| 4N decode redComb | 9.2–10.6% | 9.9–10.1% | 10.8–11.5% |
| 2N prefill dispatch | 0.3–0.5% | 0.0–0.3% | 0.1–0.2% |
| 4N decode dispatch | 0.4–1.9% | 1.7–2.6% | 1.3–1.8% |

At 2 nodes PR #8+#9 cuts the prefill combine layering from ~9% to ~2–4%, and which
node is slow stops flipping. At 4 nodes it does not: the spread stays at ~11% and
the ordering is stable and monotone in node index (n1 < n2 < n3 < n4, e.g. 225 /
230 / 241 / 251 µs), which is a different phenomenon from the 2-node flip and is
not something PR #8 addresses.

Dispatch layering is under 3% everywhere, on every arm, so dispatch numbers would
survive being read off one node. Combine numbers would not, by up to 11%.

## What this does not answer

- **The stack.** #1+#2 and #8+#9 touch different ops and mostly different files.
  Their wins look additive, but additivity is a hypothesis here, not a
  measurement: no image contains both. That is the one experiment worth running
  next, and it needs a merge branch, not an env var.
- **The other three part-geometry knobs.** PR #1 forwards four
  (`EP_MIN_TOKENS_PER_PART`, `EP_NUM_SUB_PARTS`, `EP_MIN_SUB_TOKENS`,
  `EP_SM100_MIN_SUB_TOKENS`); only `EP_NUM_SUB_PARTS=1` is measured here, and only
  at the value 1. Whether the knob is at its optimum, or whether the four interact,
  is unmeasured.
- **Why prefill combine moves.** `EP_NUM_SUB_PARTS=1` takes −3.1% off 4-node
  prefill `reduced combine` reproducibly. That is not a combine knob and no
  mechanism is offered above.
- **24 SM.** Everything is at 12 SM. Since PR #9 doubles channels per SM, the
  shape of the SM axis is likely different on that arm, and this campaign says
  nothing about it.
- **Skew.** All runs are balanced (`--unbalanced-ratio` / `--masked-ratio`
  untouched). PR #8's remote-first scheduling is a scheduling change, so an
  imbalanced shape is where it would be expected to matter most.

## Reproduce

```bash
NODES="P5EN-1 P5EN-2" REPS=3 PORT_BASE=8700 LOGDIR='$HOME/epruns_3arm' CELLS="
main54fffef|deepep-v2-efa-official:sm90-54fffef|8192|12|qpdefault|
main54fffef|deepep-v2-efa-official:sm90-54fffef|128|12|qpdefault|
pr12bfbdd15|deepep-v2-efa-official:sm90-bfbdd15|8192|12|qpdefault|
pr12bfbdd15|deepep-v2-efa-official:sm90-bfbdd15|128|12|qpdefault|
pr893c737dc|deepep-v2-efa-official:sm90-3c737dc|8192|12|qpdefault|
pr893c737dc|deepep-v2-efa-official:sm90-3c737dc|128|12|qpdefault|
" ./run_campaign.sh sm90            # then again with NODES="P5EN-1 P5EN-2 P5EN-3 P5EN-4"

# the EP_NUM_SUB_PARTS=1 arm. 2 nodes carries all three trees (the two without the
# compiler.hpp forwarding are the inertness control); 4 nodes carries only
# pr12bfbdd15, because inertness is established at 2N and by grep inside the images.
NODES="P5EN-1 P5EN-2" REPS=3 PORT_BASE=8800 LOGDIR='$HOME/epruns_3arm' CELLS="
main54fffef|deepep-v2-efa-official:sm90-54fffef|128|12|subparts1|EP_NUM_SUB_PARTS=1
pr12bfbdd15|deepep-v2-efa-official:sm90-bfbdd15|8192|12|subparts1|EP_NUM_SUB_PARTS=1
pr12bfbdd15|deepep-v2-efa-official:sm90-bfbdd15|128|12|subparts1|EP_NUM_SUB_PARTS=1
pr893c737dc|deepep-v2-efa-official:sm90-3c737dc|128|12|subparts1|EP_NUM_SUB_PARTS=1
" ./run_campaign.sh sm90
NODES="P5EN-1 P5EN-2 P5EN-3 P5EN-4" REPS=3 PORT_BASE=9000 LOGDIR='$HOME/epruns_3arm' CELLS="
pr12bfbdd15|deepep-v2-efa-official:sm90-bfbdd15|8192|12|subparts1|EP_NUM_SUB_PARTS=1
pr12bfbdd15|deepep-v2-efa-official:sm90-bfbdd15|128|12|subparts1|EP_NUM_SUB_PARTS=1
" ./run_campaign.sh sm90

# every node writes its OWN logs; combine is layered by node, so fetch all of them
i=1; for h in P5EN-1 P5EN-2 P5EN-3 P5EN-4; do
  scp "$h:~/epruns_3arm/*.node$i.log" results/p5en_3arm_20260831/logs/; i=$((i+1)); done

./verify_run.sh results/p5en_3arm_20260831/logs/*.log     # -> === no FAILs
cd results/p5en_3arm_20260831
EPRUNS=./logs python3 make_3arm_tables.py > tables.txt
EPRUNS=./logs python3 check_comparison.py         # -> 139 claims checked, 0 MISMATCHES
```

`run_campaign.sh`'s inter-cell `sleep 20` is not enough after a *previous*
campaign: the first cell above hit `REFUSING TO START: 96342 MiB already in use`
from the preceding 4-node run's ranks, and the other nodes then died in TCPStore
rendezvous rather than at the preflight. Wait for `nvidia-smi` to read idle on
every node before starting a new campaign; the two cells lost that way were re-run
under a fresh `PORT_BASE`.

Images are built per node with `./build_image.sh sm90 <sha>`; the JIT cache is
container-internal (`EP_JIT_CACHE_DIR=/root/.deep_ep`, no host bind-mount,
`docker run --rm`), so the three images cannot share cubins. That matters because
PR #9 is largely a `.cuh` change and the JIT cache key does not hash included
header content — a shared cache would have measured #9 as a no-op.

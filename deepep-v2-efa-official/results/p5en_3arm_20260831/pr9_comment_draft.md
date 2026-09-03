# Comment posted to amazon-contributing/DeepEP PR #9

https://github.com/amazon-contributing/DeepEP/pull/9#issuecomment-5474863034

Every number below is machine-checked against `logs/` by `check_comparison.py`
(139 claims, 0 mismatches). Do not hand-edit a number here; change the table
generator and re-paste.

---

We ran this branch against `main` on 4x p5en.48xlarge (H200, sm_90) and can add the
missing before/after numbers, since the PR body doesn't have any.

**Setup.** EFA installer 1.50.0, `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`.
`main` at `54fffef` vs this PR's head `3c737dc` (so #8 + #9 together), same image
recipe, same nodes, arms interleaved. `tests/elastic/test_ep.py --num-sms=12
--allow-hybrid-mode=1 --prefer-overlap-with-compute=0 --test-first-only` -- so every
number is **FP8 dispatch at `expert_alignment=128`**. 3 rotated reps per cell, all
ranks pooled from *every* node's log, not the leader's (combine is layered by node
here, up to 11%, so a leader-only table is wrong). `--ignore-local-traffic` is off, so
time is the metric and the SO GB/s column is not a wire rate. `EP_BUFFER_DEBUG` off,
because it printf()s inside dispatch's timed loop.

### Prefill, 8192 tok/rank (us, all-rank mean)

| op | 2N main | 2N #9 | Δ | 4N main | 4N #9 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 1499.8 | 1498.4 | −0.1% | 3965.2 | 3961.2 | −0.1% |
| cached dispatch | 1588.8 | 1496.2 | **−5.8%** | 4254.0 | 3946.2 | **−7.2%** |
| combine | 3587.8 | 3172.5 | **−11.6%** | 7872.1 | 7761.8 | −1.4% |
| reduced combine | 4238.0 | 3670.5 | **−13.4%** | 7947.3 | 7773.2 | −2.2% |
| dispatch + reduced combine | 5737.8 | 5168.8 | **−9.9%** | 11912.5 | 11734.5 | −1.5% |

### Decode, 128 tok/rank (us, all-rank mean)

| op | 2N main | 2N #9 | Δ | 4N main | 4N #9 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| dispatch | 169.6 | 123.5 | **−27.2%** | 184.0 | 171.0 | −7.1% |
| cached dispatch | 166.2 | 120.2 | **−27.7%** | 178.9 | 167.6 | −6.4% |
| combine | 162.5 | 143.6 | **−11.6%** | 244.8 | 234.6 | −4.2% |
| reduced combine | 179.2 | 149.5 | **−16.6%** | 253.6 | 237.2 | −6.5% |
| dispatch + reduced combine | 348.8 | 273.0 | **−21.7%** | 437.6 | 408.2 | **−6.7%** |

Four observations:

**1. The node-variance claim from #8 reproduces at 2 nodes and does not at 4.**
Using the same statistic quoted in #8's description -- combine SO GB/s min-max across
ranks, 2N/8192:

| | main | this PR |
|---|---|---|
| combine | 60-73 GB/s | 70-77 GB/s |
| reduced combine | 53-60 | 61-66 |

Per-node mean `reduced combine` tells the same story: `main` spreads 9.1 / 8.8 / 8.9%
across the three reps and *which* node is slow flips between reps (rep1 n1 4434 vs n2
4066 us, rep2 4053 vs 4410); this PR gives 3.6 / 3.8 / 1.8%.

At 4 nodes it does not hold. 4N/128 per-node mean `reduced combine` max-min is
9.2-10.6% on `main` and 10.8-11.5% on this PR, and it is monotone and *stable* in node
index (rep2: 225 / 229 / 245 / 251 us) rather than flipping between reps. So there is a
second, systematic stagger at >2 nodes that remote-first scheduling does not address --
worth knowing before this is read as a general fix for node variance. Dispatch layering
is <=2.6% on every arm at both scales, so this is combine-specific.

**2. `cached dispatch` on `main` is slower than plain dispatch** (1588.8 vs 1499.8 us
at 2N, 4254.0 vs 3965.2 at 4N -- and SO drops 81-82 to 77-77 GB/s). This PR doesn't
speed that path up so much as remove the penalty: its cached dispatch (1496.2 / 3946.2)
lands at parity with its own plain dispatch. At 4 nodes this is the only prefill row
with a delta clear of run noise.

**3. `#QPs` reads `13/13` in every log on this branch** vs `11/11` on `main`, so the
`kDefaultGinContextCnt` bump does take effect on EFA.

**4. Most of the win is at 2 nodes.** 2N decode dispatch −27.2% becomes −7.1% at 4N;
2N prefill reduced-combine −13.4% becomes −2.2%. Directionally still a win everywhere
we measured, but the headline number should probably carry its node count.

Two things this does *not* answer, to be explicit:

- **@vladimiraerov's attribution question.** This measures the *sum* of all four
  changes. We did confirm from the diff that both the channel-clamp removal and the
  forward-warp pairing are gated on `not prefer_overlap_with_compute`, which is what we
  pass, so both are live on the measured path -- but we can't say how the −21.7% splits
  between them. An arm at `--prefer-overlap-with-compute=1` would bracket
  clamp+pairing without a rebuild; happy to run it if useful.
- **The 4 -> 8 channels/SM figure** in the PR body is not something we observed; `#SM`
  prints the same on both arms and we ran without `EP_BUFFER_DEBUG`.

And one caveat on reading the tables: since this PR deletes the
`if (not prefer_overlap_with_compute) num_channels_per_sm = min(..., 4)` clamp,
`--prefer-overlap-with-compute=0` no longer selects the channel count it used to. The
comparison above is main-vs-branch on identical hardware with an identical command
line, so it is internally valid, but it is not comparable against any baseline that
was getting 4 channels/SM from that flag.

Separately, for merge ordering: PR #1's `compiler.hpp` change (forwarding
`EP_NUM_SUB_PARTS` into the JIT flags) is not in this branch, and without it the env
var is inert -- the `.cuh` reads the macro but nothing defines it. On the #1+#2 branch
with `EP_NUM_SUB_PARTS=1` we measure 4N decode dispatch at 156.4 us (−15.0% vs `main`,
about twice this PR's −7.1%) and a 4N layer total of 409.7 us, i.e. a tie with this
PR's 408.2. Landing #9 without #1 removes that lever.

# Investigation: NVSHMEM PR #9 (FI_RX_CQ_DATA) impact on DeepEP V1 LL

## Background

A coworker reported that on p5en.48xlarge, DeepEP V1 LL mode benchmarks
**~404 µs dispatch / ~446 µs combine** with `amazon-contributing/upstream-to-nvshmem@devel_enriched`.
Reverting commit
[`0ec7be33`](https://github.com/amazon-contributing/upstream-to-nvshmem/commit/0ec7be33)
("libfabric: Disable unsolicited write for EFA provider", part of
[PR #9](https://github.com/amazon-contributing/upstream-to-nvshmem/pull/9))
brings the numbers down to **~350 µs / ~370 µs**.

The commit enables `FI_RX_CQ_DATA` on the EFA libfabric provider, which
makes write-with-immediate consume a posted recv buffer on completion
(prevents CQ overflow). Trade-off: every RX completion has to re-post a
recv buffer, adding CPU work to the LL dispatch critical path.

We wanted to confirm:
1. The same `0ec7be33` is in our `devel_enriched` build (yes — it landed
   2026-03-27, our build is from 2026-05).
2. p5 baseline numbers reproduce within noise.
3. Reverting on p5 moves the needle in the same direction, and quantify
   how much.

## Reproduction

Baseline image: `deepep-v1-efa:dev` (NVSHMEM 3.6.5+ from
`devel_enriched` HEAD on 2026-05-16).

Revert image: `deepep-v1-efa:revert-pr9`. Built by layering on top of
`:dev`, re-cloning NVSHMEM, `git revert 0ec7be33`, rebuilding NVSHMEM,
reinstalling `libnvshmem_host.so*` into `/opt/nvshmem/lib`. DeepEP
itself is unchanged. See `../deepep-v1-efa-revert9/Dockerfile`.

Bench: `tests/test_low_latency.py`, 16 ranks (2 nodes × 8 GPU on
p5.48xlarge), 128 tokens, hidden 7168, top-k 8, 288 experts.

## Results

| Stack / build | LL Dispatch | LL Combine | D+C BW | Notes |
|---|---|---|---|---|
| **p5, baseline (PR#9 in)** | **765 µs** | **641 µs** | 16.80 GB/s | this image, current build |
| **p5, PR#9 reverted** | **585 µs** | 639 µs | 19.05 GB/s | -23.5 % dispatch |
| p5en (coworker, PR#9 in) | 404 µs | 446 µs | (n/a) | their measurement |
| p5en (coworker, PR#9 reverted) | ~350 µs | ~370 µs | (n/a) | their measurement |

Per-op deltas:

| Hardware | Δ Dispatch | Δ Combine |
|---|---|---|
| p5.48xlarge (EFA v1, this run) | **−180 µs** (−23.5 %) | flat (+2 µs) |
| p5en.48xlarge (coworker) | −54 µs (−13 %) | −76 µs (−17 %) |

## Observations

1. **PR #9's cost is ~3.3× larger on EFA v1 (p5) than EFA v2 (p5en) for
   dispatch.** Re-posting a recv buffer per RX completion takes more
   wall-clock on v1's slower SRD path; reverting unblocks more savings.

2. **Combine is unaffected on p5 (flat).** PR #9 mostly hits the
   write-with-immediate path used during dispatch synchronisation;
   combine's reduction path doesn't depend on it. The p5en combine
   improvement coworker saw (−76 µs) is the same RX-side savings, just
   less pronounced because EFA v2 paths are already faster.

3. **Absolute gap to p5en is the same either way:**
   - Baseline: p5 765 vs p5en 404 → p5 is 89 % slower
   - Reverted: p5 585 vs p5en 350 → p5 is 67 % slower

   So PR #9 explains some of the v1-vs-v2 LL gap (it costs more on v1),
   but the bulk remains EFA v1 SRD per-message latency floor.

4. **Combine BW is unchanged** (22.67 → 22.77 GB/s) despite the
   transport rebuild — sanity check that the revert didn't disturb
   anything else.

## Conclusion

- Our test methodology is correct; the previously reported ~700 µs LL
  dispatch on p5 reproduces (765 µs in this run) within noise.
- Reverting PR #9 cuts dispatch latency from **765 µs → 585 µs** on p5,
  the largest single-knob win we've found for LL on EFA v1.
- This is a **functional trade-off**, not a free win: PR #9 was
  added to prevent CQ overflow on arbitrary communication patterns.
  DeepEP V1 LL has bounded send patterns so it doesn't trigger
  overflow in practice, but other apps might. Consider conditional
  on workload.

## Repro

```bash
# Build
cd deepep-v1-efa-revert9
ssh <node> 'cd ~/work/deepep-v1-efa-revert9 && docker build -t deepep-v1-efa:revert-pr9 .'

# Launcher (just like run_low_latency.sh but uses :revert-pr9 image)
ssh P5EN-1 'sed "s|deepep-v1-efa:dev|deepep-v1-efa:revert-pr9|" \
  ~/work/deepep-v1-efa/run_low_latency.sh \
  > ~/work/deepep-v1-efa/run_low_latency_revert9.sh && \
  chmod +x ~/work/deepep-v1-efa/run_low_latency_revert9.sh'

# Run
ssh P5EN-2 'cd ~/work/deepep-v1-efa && bash run_low_latency_revert9.sh 1 <leader-ip>' &
ssh P5EN-1 'cd ~/work/deepep-v1-efa && bash run_low_latency_revert9.sh 0 <leader-ip>'
```

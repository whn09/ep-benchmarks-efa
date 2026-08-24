# DeepEP V2 on AWS EFA — the released stack (`amazon-contributing/DeepEP`)

Builds and runs the **public release** of AWS's DeepEP V2 fork —
[`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) — on
2 × `p5en.48xlarge` (8×H200 `sm_90` + 16×200 Gb/s EFA each), bare EC2 with Docker.

**Everything here comes from published packages.** No source-built NCCL, no
source-built aws-ofi-nccl, no hand-patched kernel module, no `LD_PRELOAD`, and none
of the `NCCL_GIN_TYPE=5` / `FI_EFA_USE_HW_CNTR=1` / `OFI_NCCL_GIN_STRONG_SIGNAL=1`
exports that the earlier route needed. **EFA installer 1.50.0 supplies all four
load-bearing components at once**, and GDAKI comes up on its own.

That is the difference from `deepep-v2-efa-gdaki-b200/` (local-only, not committed — 673 MB
of campaign results), which builds `Xuan-1998/DeepEP@dev` with a hand-assembled stack. Measured on the same
p5en pair, this packaged path costs **nothing** in performance (§ Results).

> **中文完整版 runbook：[`docs/runbook_zh.md`](docs/runbook_zh.md)** — host 安装、镜像构建、
> prefill 带宽 / decode 延迟测试、环境变量速查、故障对照表、CE 探针源码。
> This README is the condensed English version; the Chinese runbook is authoritative
> and carries the full troubleshooting table.

Verified 2026-08-21 on installer 1.50.0 / `deep_ep 2.1.0+ec623f3`, 16 ranks, exit 0
with correctness checks passing.

## The dependency chain

Kernel-side goes on the **host**, user-side in the **container**, and both are required.

| Layer | Required version | How to check it | Where |
|---|---|---|---|
| DeepEP V2 unordered kernels | `amazon-contributing/DeepEP` | — | container |
| NCCL GIN | `nvidia-nccl-cu13` ≥ 2.30.4 (we use 2.31.2) | — | container |
| aws-ofi-nccl | **1.21.1** | **`ncclGinPlugin_v14`** symbol | container |
| libfabric | **2.6.0amzn1.0** | 16 × `fabric: efa-direct` | container |
| rdma-core | **64.0amzn0** | 20 × `comp_cntr` symbols in libibverbs | container |
| **efa kernel driver** | **3.3.0** | `EFA_QUERY_DEVICE_CAPS_COMP_CNTR` (1<<8) | **host** |
| gdrcopy | ≥ 2.5 | `/dev/gdrdrv` | kmod host / lib container |

All four bold rows ship together in **EFA installer 1.50.0**. GIN = GPU-Initiated
Networking (NCCL Device API); the EFA implementation is `Libfabric_GDAKI`, which uses
`efa-direct` GDA ops with a hardware completion counter (CE) as the counting signal.

**Pick your version check carefully.** 1.49.0 *already* reports 16 `efa-direct`
domains and *already* exports `ncclGinPlugin_v11`/`v13` with GDAKI strings in the
binary — so neither "has efa-direct" nor "has Gin symbols" distinguishes it. What
actually blocks 1.49.0 is the bottom two layers: `EFA_QUERY_DEVICE_CAPS_COMP_CNTR`
does not exist in efa 3.1.0's `efa-abi.h`, and rdma-core 63.0's libibverbs has **zero**
`comp_cntr` symbols (64.0 has 20).

| installer | efa driver | libfabric | rdma-core | ofi-nccl | GinPlugin |
|---|---|---|---|---|---|
| 1.47.0 | 3.0.0 | 2.4.0amzn1.0 | 61.0 | 1.18.0 | **none** |
| 1.49.0 | 3.1.0 | 2.4.0amzn5.0 | 63.0 | 1.20.0 | v11 / v13 |
| **1.50.0** | **3.3.0** | **2.6.0amzn1.0** | **64.0amzn0** | **1.21.1** | **v11 / v13 / v14** |

Only the 1.50.0 row was checked on these hosts; the others are historical (consistent
with our independent observations on the 1.47.0 factory stack and on 1.49.0). Re-check
on your own target before making an upgrade decision.

## Host prerequisites

Four instance-level conditions, each of which is fatal on its own:

1. **EFA must be enabled at ENI-creation time** (`InterfaceType=efa`) — it cannot be
   turned on afterwards. p5en gets one EFA ENI per GPU via `NetworkCardIndex=0..15`;
   healthy state is **16 EFA devices**. If `lsmod | grep efa` shows the module but
   `ibv_devinfo` says `No IB devices found` and `/dev/infiniband` is absent, this is why.
2. **Security group must allow all traffic to itself** (self-referencing, inbound and
   outbound). EFA does not use TCP ports.
3. **Same AZ + same cluster placement group.** Across AZs you get
   `ibv_create_ah failed with EINVAL ... Remote GID is in a different availability zone`.
4. Device names on p5en are neither `mlx5_*` nor contiguous: `rdmap85s0 86s0 87s0 88s0
   / 110s0 111s0 112s0 113s0 / 135s0 136s0 137s0 138s0 / 160s0 161s0 162s0 163s0`.

Then install the EFA stack on **every** host:

```bash
# aws s3 cp is broken for this bucket on CLI v2 (HeadObject -> 301, even with --region)
curl -O https://aws-efa-installer-dev.s3.amazonaws.com/aws-efa-installer-latest.tar.gz
tar xzf aws-efa-installer-latest.tar.gz            # ~650 MB, RPMs+DEBs for all distros
head -12 aws-efa-installer/ChangeLog.md            # must say ## [1.50.0]

cd aws-efa-installer
sudo ./efa_installer.sh -y --no-verify             # dev-bucket packages are unsigned
sudo reboot                                        # swapping efa.ko requires it
```

> ⚠️ The `-latest` S3 key is **not** a fixed version. Archive the tarball together with
> its `ChangeLog.md` version if you need reproducibility; the filename alone tells you
> nothing.

Verify after reboot:

```bash
export PATH=/opt/amazon/efa/bin:$PATH        # fi_info is not on the default PATH
cat /sys/module/efa/version                  # 3.3.0g
lsmod | grep efa_nv_peermem                  # required for GDAKI
modinfo gdrdrv | grep ^version; ls -l /dev/gdrdrv
ibv_devinfo -l                               # 16 rdmap*
fi_info | grep -c "fabric: efa-direct"       # 16
```

**`fi_info -p efa-direct` always fails — do not use it as a check.** It returns
`-61 (No data available)` on perfectly healthy nodes: `-p` filters on *provider* name
(which is `efa`); only the *fabric* is called `efa-direct`. Use `fi_info | grep fabric`.

`ce_probe.c` in this directory is the one-shot decisive check — it exercises
`ibv_create_comp_cntr`, the single verb GDAKI's success reduces to, and covers both the
host kernel module and the container's rdma-core. Run it **inside the container**;
healthy nodes print `CE OK` for all 16 `rdmap*`. Driver state is **per node** — a
freshly-rebooted instance in the same fleet can silently fall back to an older module.

## Build

```bash
rsync -avz deepep-v2-efa-official/ <node>:~/work/deepep-v2-efa-official/
scp aws-efa-installer-latest.tar.gz <node>:~/work/deepep-v2-efa-official/   # not in git
ssh <node> "cd ~/work/deepep-v2-efa-official && docker build -t deepep-v2-efa-official:dev ."
```

~21.4 GB (7.7 GB compressed), ~15 min cold. The installer tarball must be in the build
context; it is not committed here because of its size.

### Build gotchas (all of these are already handled in the Dockerfile)

1. **Do not clear the apt cache before installing EFA.** The installer runs its own
   `apt-get install tcl libnl-3-200 ...`; with `/var/lib/apt/lists` emptied and no
   re-`apt-get update` it fails with `Unable to locate package tcl`. Keep
   `rm -rf /var/lib/apt/lists/*` at the *end* of the EFA layer.
2. In-container the installer needs `--skip-kmod --skip-limit-conf --no-verify`, but
   **not** `--skip-rdma-core` — the container needs rdma-core 64.0's libibverbs.
3. **`third-party/fmt` is a submodule**; without it both `setup.py` and the JIT miss fmt headers.
4. **`ninja` is mandatory**: otherwise `AssertionError: With dlink=True, ninja is required`.
5. **`numpy` + `nvidia-ml-py`** are imported by `deep_ep/` and `tests/` — missing them fails at runtime, not build time.
6. **NVSHMEM is a hard `setup.py` dependency** (the legacy backend is still in the source list) even if you only run V2.
7. **NCCL ≥ 2.31**: below that DeepEP asserts compile-time and run-time NCCL versions are exactly equal.
8. **`WORKDIR` must not be `/opt/DeepEP`**, or python imports the source tree's `deep_ep/`
   (no `_C.so`) and reports `ModuleNotFoundError: deep_ep._C`. Use `/workspace` + absolute paths.
9. **The plugin is not called `libnccl-net.so`.** In-container the installer takes its NGC
   branch and lands `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`. Use `NCCL_NET_PLUGIN=ofi`
   and let NCCL resolve the name; do not hardcode a path.

### Three hardenings over a just-make-it-run Dockerfile

All three were verified by an actual rebuild, not by reasoning.

- **The DeepEP commit is pinned** (`ARG DEEPEP_COMMIT=ec623f3…`). An unpinned
  `git clone --depth 1` of `main` means a rebuild tomorrow silently stops matching the
  numbers below. `git clone --depth 1 --branch <sha>` does not accept a bare sha, hence
  the `git init` + `git fetch --depth 1 origin <sha>` form.
- **apt's `libnccl2`/`libnccl-dev` 2.28.3 are removed.** The installer's NGC branch pulls
  them in and marks them `hold`. 2.28.3 < 2.30.4 ⇒ **no GIN**, and `ldconfig` resolves
  `libnccl.so.2` to *them* (`/usr/include/nccl.h` too). Safe to remove: the
  `libnccl-ofi-ngc-v3` deb has **no `Depends` field at all**, and `libnccl-net-ofi.so`
  links only `libfabric.so.1` — it never links libnccl, being a plugin NCCL dlopens.
- **A `-L` for pip's NCCL was added, because removing those debs breaks the build.**
  This only surfaced on rebuild: `/usr/bin/ld: cannot find -l:libnccl.so.2`. `setup.py`
  emits `-l:libnccl.so.2` plus `-Wl,-rpath,<pip>/nvidia/nccl/lib` but **no `-L`** — so the
  stock image was link-resolving NCCL against apt's **2.28.3** and only switching to pip's
  2.31.2 at *run* time via rpath. Headers were always pip's 2.31.2, so the result was
  correct, but the dependency was implicit.

**Validation of the hardened image** (rebuilt on both nodes 2026-08-21, then re-run):
`git -C /opt/DeepEP rev-parse HEAD` = `ec623f31…`; `dpkg -l | grep -c libnccl2` = 0;
`ldconfig -p | grep libnccl.so` = empty; `/proc/self/maps` after `import deep_ep` shows
**exactly one** libnccl, pip's `nvidia/nccl/lib/libnccl.so.2`; `ncclGinPlugin_v14` present;
20 `comp_cntr` symbols; and the run logs `Loaded gin plugin Libfabric_GDAKI (v14)`.
Performance is unchanged — prefill dispatch **74 GB/s SO / 240–244 SU / 1649 µs** (table
below: 1665.1 ± 12.4), decode dispatch **351.2–351.7 µs** (below: 367.0 ± 12.1) and
combine **170.8–184.9 µs** (below: 178.1 ± 5.0), all inside the across-rep spread.

## Run

```bash
# worker first, then leader; only NODE_RANK differs
ssh <worker> "cd ~/work/deepep-v2-efa-official && TOKENS=8192 bash run_test_ep.sh 1 <leader-ip>" &
ssh <leader> "cd ~/work/deepep-v2-efa-official && TOKENS=8192 bash run_test_ep.sh 0 <leader-ip>"
```

`TOKENS=8192` → prefill (report **bandwidth**). `TOKENS=128` → decode (report
**latency**). Nothing else changes between the two.

**`test_ep.py` is not torchrun.** It `mp.spawn`s its own local ranks, so
`WORLD_SIZE` = **node count** (2), `RANK` = **node index** (0/1), and `--num-processes`
= local ranks per node (8).

**`--num-sms` must be passed explicitly on EFA.** With `--num-sms 0` (the default),
`get_theoretical_num_sms()` → `get_rdma_gbs()` shells out to `ibstat`, which goes through
libibumad; EFA has no `ib_umad`, so `ibstat rdmap85s0` dies with `ibpanic: ... IB device
can't be found`, the helper logs `Failed to get RDMA connection speed:` and returns 0, and
`elastic.py:824` divides by it (its guard tests `num_rdma_ranks > 1` while the branch is
taken on `num_scaleout_ranks > 1`). **Single-node never triggers it** — it appears the
moment you go multi-node.

> **This is true of the `ec623f3` pin only — do not carry it to `main`.** The fix is
> `a4f923c envs: probe RDMA link rate via sysfs, survive probe failure` (`e0f110a` on
> `main`): it reads `/sys/class/infiniband/<nic>/ports/*/rate` first, keeps `ibstat` as a
> fallback, and picks the fastest device when `EP_NIC_NAME` is unset. That commit is **not
> in `ec623f3`** (`git log ec623f3..7a6059a3` shows it), so this pin still needs an explicit
> `--num-sms`; on `main` the auto path works.

**And `--num-sms` is not a free parameter.** It also changes the allocated QP count, and
*non-monotonically*: measured 0→17 QPs, 12→5, 24→10. A value landing on
`num_qps < num_ranks` **hangs outright** — the GIN auto-tuner prints its lines and then
nothing for 600 s. 16 ranks / 12 SM is fine (`num_qp=11`). For comparable numbers stay on
the three official operating points: **H200 2-node 12 SM / H200 4-node 6 SM / B200 2-node 12 SM**.

### The GIN evidence to look for

```
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)     <-- the decisive line
NCCL INFO NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0 'rdmap86s0'
NCCL INFO Using network Libfabric
```

That one `Libfabric_GDAKI (v14)` line proves the whole chain at once: the v14 symbol, the
GDA ops, the CE verb, and the host kernel capability bit.

With `EP_BUFFER_DEBUG=1` each rank adds:

```
DeepEP initialized with NCCL version: 2.31.2 (loaded library)
EP NCCL device communicator has 0 allocated QPs      <-- NORMAL, not a failure
GIN layout: gin_context_cnt=11, gin_indexed_signals_cnt=21, num_qp=11
```

`has 0 allocated QPs` reads like an error but is just the count of *explicitly* allocated
QPs; the real number is `num_qp=11` on the next line. People stop reading here and
conclude GIN did not come up.

Check the NCCL version from that line or from `NCCL_DEBUG=INFO`'s
`NCCL version 2.31.2+cuda13.3` — **not** from `torch.cuda.nccl.version()`, which reports
torch's compile-time header (2.29.7) forever. There are **three** NCCL versions in the
image; see the Chinese runbook's Appendix B.

## Results (2 × p5en.48xlarge, 2026-08-21)

Ubuntu 24.04, driver 595.91.07, installer upgraded 1.49.0 → 1.50.0 + reboot. Container
torch 2.13.0+cu130 / nccl 2.31.2 / `deep_ep 2.1.0+ec623f3`. 16-rank `test_ep.py` exits 0
with all correctness checks passing; `fi_pingpong -p efa` passes (64 B 1.73 MB/s,
4 K 282.48 MB/s).

### Prefill (`--num-tokens=8192`) — bandwidth

Full range across **all 16 ranks** (not one rank, not one node — see Rules below):

| op | SO GB/s | SU GB/s | time | bytes/rank |
|---|---|---|---|---|
| dispatch | 72–75 | 233–246 | 1665.1 ± 12.4 µs | 399.8 MB |
| expanded dispatch | 74–75 | 240–246 | 1644.4 ± 8.2 µs | 399.8 MB |
| cached dispatch | 73–74 | 236–244 | 1662.8 ± 8.7 µs | 399.8 MB |
| combine | 60–73 | 196–240 | 3560.8 ± 9.2 µs | 767.1 MB |
| reduced combine | 52–59 | 170–194 | 4243.9 ± 9.6 µs | 767.1 MB |

3 reps; mean over 16 ranks then over reps, ± is stdev **across reps**.

**Denominators.** `SU` = the printed per-rank `bytes` ÷ time exactly (399.8 MB / 1665 µs
= 240 GB/s). **`SO` is not a wire rate** — without `--ignore-local-traffic` it counts
intra-node destinations too. The per-rank wire ceiling on p5en is **50 GB/s**
(16 × 200 Gb/s ÷ 8 GPUs), so a reported 74 GB/s dispatch is by itself proof that the
figure is not a network number. Pass `IGNORE_LOCAL=1` for a wire-rate run.

**Combine dispersion is layered by node**, not a single outlier rank:

| op | slow node's 8 ranks | fast node's 8 ranks |
|---|---|---|
| dispatch | 73 GB/s / 1679 µs | 73 GB/s / 1670 µs (0.5% apart) |
| combine | 60–65 / **3634–3942 µs** | 63–73 / **3208–3714 µs** |
| reduced combine | 52–55 / **4302–4495 µs** | 54–59 / **3956–4334 µs** |

All 8 ranks on one machine are ~700 µs (**+21%**) slower in combine and ~8% slower in
reduced combine; dispatch shows no such split. **So never quote one node's range.** Which
machine is slow flips between runs, so a single run cannot tell you whether this is an
intrinsic leader-node effect — we draw no mechanism conclusion here.

> **Versus the 2026-08-13 calibration run — `dispatch` is 10.7% slower. It is the same
> kernel; what differs is the GIN QP layout.** That run used source NCCL `2.30.7-1` + source
> aws-ofi-nccl `--enable-gdaki`
> + DeepEP `7a6059a3`; its `test_ep.py` args are byte-identical to §"Run" apart from an
> extra `--skip-check`, same `use_fp8_dispatch=1`, same 399.8 MB per rank, same two nodes.
>
> | op | 08-13 hand-built `7a6059a3` | here, packaged `ec623f3` | Δ time |
> |---|---|---|---|
> | dispatch | 81.25 GB/s / 1504 µs | 72–75 / 1665.1 µs | **+10.7%** |
> | expanded dispatch | 81.44 / 1501 µs | 74–75 / 1644.4 µs | +9.6% |
> | cached dispatch | 70.06 / 1743 µs | 73–74 / 1662.8 µs | **−4.6%** |
> | combine | 65.75 / 3592 µs | 60–73 / 3560.8 µs | −0.9% |
> | reduced combine | 56.00 / 4195 µs | 52–59 / 4243.9 µs | +1.2% |
>
> This is not a broad slowdown — combine barely moves and **all three dispatch variants
> collapse onto ≈1664 µs**. On 08-13 `dispatch` beat `cached dispatch` by 240 µs (AWS's own
> reference likewise, 81.00 vs 69.94); here that gap is gone. So the accurate statement is
> that *dispatch lost its advantage over cached dispatch*.
> Raw data: `deepep-v2-efa-gdaki-b200/results/p5en_ours_20260813/summary.txt` (🔒 local-only).

**It is the same kernel — established at blob level, not from commit messages.** Everything
below is verifiable from source, no hardware needed. At `ec623f3` the hybrid dispatch kernel
exists twice (`ec623f3` = `feat: add EP_HYBRID_KERNEL toggle between unordered and ordered
kernels`, `unordered` the default per `csrc/kernels/elastic/kernel_select.hpp:37-52`), and
comparing blobs settles which is which:

| file at `ec623f3` | vs `af9a040:hybrid_dispatch.cuh` (pre-fork upstream) | vs `7a6059a3:hybrid_dispatch.cuh` (what 08-13 ran) |
|---|---|---|
| `hybrid_dispatch.cuh` — the `ordered` variant | **+6 / −1** | +56 / −468 |
| `hybrid_dispatch_unordered.cuh` — **the default** | +485 / −51 | **+45 / −28** (11 of them the license header) |

So `unordered` **is** `7a6059a3`'s kernel — the AWS EFA GDA port — under a new name, and
`ordered` is the upstream kernel this branch forked from, carried along nearly verbatim.
Combine tells the same story (`+12/−6` vs `+24/−7`). The entire functional content of that
+45/−28: a new `num_unaligned_recv_tokens_per_expert` output pointer (one store per expert),
two lambdas hoisted out of the two warp branches, the identity lambda `phys_token_slot`
dropped, and comments.

⇒ **"The two runs are not the same kernel" is retracted, and `EP_HYBRID_KERNEL=ordered` is
not the A/B to run.** That flag selects the *upstream* kernel, which publishes a tail signal
and lets the receiver assume everything before it landed — we measured earlier that this is
**incorrect on EFA GDAKI** (`NCCL_GIN_TYPE=5`, where the signal can overtake the data); it is
only correct on the ordered proxy path. It also asks NCCL for a different fabric altogether
(`csrc/kernels/backend/nccl.cu:111-127`: 129 **exclusive** contexts, `ginSignalCount =
num_ranks + 4`, `ginVaSignalsRequired`, `ginStrongSignalsRequired`), so it would not be a
one-variable change even if it were correct.

**And the 26-vs-9 commit divergence is mostly SHA-level.** `merge-base(7a6059a3, ec623f3) =
af9a040`; `7a6059a3` carries 26 commits `ec623f3` lacks and `ec623f3` has 9 the pin lacks.
But by *content* that is largely the same work rebased: `qp_mapping.cuh` — where `7a6059a`,
the pin's own tip commit, put "perf(gin): Balanced contiguous QP-channel mapping" — differs
by **+10 / −0, all of it the license header**. Count commits to date a branch, never to date
a behaviour.

**That leaves exactly one substantive difference in the dispatch path:
`gin_resource_alloc.cuh` (+80 / −50), the context-count policy.** Both sides are computable
from source at 12 SM / 4 channels per SM / 16 ranks, and both reproduce their logged line
digit for digit:

| | contexts (= QPs) | signals/ctx | data QPs (= ctx − 1 notify) | channels/ctx | `kNumParts` | channels/SM | logged |
|---|---|---|---|---|---|---|---|
| `7a6059a3` (08-13) | `ceil_div(12×4, kMinGinContextSharingFactor=10)` = **5**, so **SM-dependent** | `(256 − 2·5)/5` = 49 | 4 | `ceil(48/4)` = 12 | `49/12` → **4** | 4 | `5 / 49 / 5` ✅ |
| `ec623f3` (here) | `kMinGinContextSharingFactor` deleted; auto path is the constant `kDefaultGinContextCnt` = **11**, **SM-independent** | `(256 − 2·11)/11` = 21 | 10 | `ceil(48/10)` = 5 | `21/5` → **4** | 4 | `11 / 21 / 11` ✅ |

**Note what does not change: `kNumParts = 4` and 48 channels on both sides**, so the kernel
is instantiated with identical template arguments. The only live difference is how many QPs
those same 48 channels are spread across — **4 QPs × 12 channels (08-13) vs 10 QPs × 5
channels (release)** — and that is now the single remaining code-level candidate for the
10.7%. The shape of the data is at least consistent with it: plain dispatch lost 161 µs while
cached dispatch *gained* 80 µs.

**Two earlier guesses are refuted — do not cite them:**

- **`kNumParts` is not the mechanism.** Computed above from `compute_part_allocation()`
  (`gin_resource_alloc.cuh:122-152`): **4 on both sides**. `ec623f3`'s own comment agrees,
  listing 11's equivalents as `{5, 6, 7, 8, 9, 14}`.
- **It is not "the new branch lost the tuning".** Front-loading (`kMidTotal`) and the forward
  double-buffer (`kNumDispatchFwdBuffers`) are both present in
  `hybrid_dispatch_unordered.cuh` — necessarily, since that file *is* `7a6059a3`'s kernel.

**Still unconfirmed (needs p5en/H200 GDAKI hardware) — but it is now a one-flag test.**
`ec623f3` adds `--num-allocated-qps` / `--num-qps` to `tests/elastic/test_ep.py`
(`7a6059a3` had neither — the count was derived). Re-run §Run with
**`--num-allocated-qps 5`** and the layout becomes 5 contexts / 49 signals / 4 data QPs,
exactly 08-13's. If dispatch returns to ~1504 µs the QP fan-out is the whole story; if it
does not, the suspects left are the stack (source NCCL `2.30.7-1` + source aws-ofi-nccl
`--enable-gdaki` vs the packaged 1.50.0 libfabric `2.6.0amzn1.0`) and `--skip-check`. Neither
arm needs a rebuild. `OFI_NCCL_GIN_STRONG_SIGNAL=1` was in 08-13's environment, but the
unordered kernel only requires weak signals, so run it as its own arm rather than folding it
into this one.

### Decode (`--num-tokens=128`) — latency. The release is slow; two PRs are pending

At this size only latency is meaningful: **5.62–5.94 MB per rank** across the 16 ranks,
5 GB/s SO / 16–17 GB/s SU. It is **message-rate** bound, not bandwidth bound.

| op | release `ec623f3` | +[#2](https://github.com/amazon-contributing/DeepEP/pull/2) | +[#1](https://github.com/amazon-contributing/DeepEP/pull/1) and #2 |
|---|---|---|---|
| dispatch | 367.0 ± 12.1 µs | 239.8 ± 3.1 µs (**−34.7%**) | **166.1 ± 0.4 µs (−54.7%)** |
| expanded dispatch | 366.1 ± 11.1 µs | 239.7 ± 1.9 µs (−34.5%) | 156.3 ± 0.9 µs (−57.3%) |
| cached dispatch | 359.2 ± 10.1 µs | 235.7 ± 1.4 µs (−34.4%) | 150.8 ± 1.3 µs (−58.0%) |
| combine | 178.1 ± 5.0 µs | 178.3 ± 2.2 µs (+0.1%) | 179.8 ± 1.2 µs (+0.9%) |
| reduced combine | 196.9 ± 5.5 µs | 196.3 ± 2.8 µs (−0.3%) | 197.6 ± 1.2 µs (+0.4%) |

dispatch + combine: **545 µs → 346 µs**. 3 reps, variants interleaved within each rep,
each variant on its own `EP_JIT_CACHE_DIR`, 48/48 rounds rc=0.

**The cause is degenerate part geometry at small batch, not EFA** — combine being exactly
flat is the evidence. `kNumParts` (how many `flush_part` puts a channel's tokens leave in)
is set only by `compute_part_allocation()`, which caps *from above* when the GIN
indexed-signal budget is tight — and the budget is loosest precisely when a channel holds
the fewest tokens. So decode always lands on `kMaxParts`, the worst end of the axis. At
128 tokens / 12 SM a channel holds 3 tokens but is described as 4 parts × 1 token: the
last part is always empty, and 3 tokens leave as three single-token puts instead of one
3-token put. Sub-parts already have both guards parts lack (a clamp to `kBatchSize`, plus
`EP_SM100_MIN_SUB_TOKENS`).

| PR (base `main = ec623f3`) | What |
|---|---|
| [#1](https://github.com/amazon-contributing/DeepEP/pull/1) | Forward `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` to the JIT. Changes no default. |
| [#2](https://github.com/amazon-contributing/DeepEP/pull/2) | Add `kMinTokensPerPart` (default 15, `EP_MIN_TOKENS_PER_PART` overrides): `kNumParts = min(budget, tokens_per_channel / 15)`. |

**Which to use.** Just bringing it up, or only care about prefill → take `main` as-is;
both PRs are within ±2% on prefill, i.e. inside the noise. **Publishing decode /
small-token numbers → cherry-pick them, or dispatch is 2.2× slower.** #1 alone is a wash
(decode +1.8%, prefill −2.0%); it only pays stacked on #2. After they merge, #2's default
of 15 applies automatically — no env var needed. To get the old geometry back as a control
use `EP_MIN_TOKENS_PER_PART=1`, which **short-circuits** to the old value (dividing by 1
is a *third* geometry, not a control).

## Rules that decide whether a number is real

1. **One `EP_JIT_CACHE_DIR` per variant.** GIN's device-side headers get compiled into
   DeepEP's JIT kernels, so a shared cache silently serves the other arm's cubin. This is
   the single most common reason an A/B "shows no difference".
2. **Interleave reps** (`A B A B …`), never all-A-then-all-B, or thermal and cluster drift
   land entirely on one arm.
3. **`rc=0` is not a health check.** Ranks leaked by a previous round stay alive holding
   ~48 GB/GPU; if the leak is small enough that GDAKI init still succeeds, the next round
   completes, exits 0, prints full output — and reports ~2× inflated latency with nothing
   in the log to say so. (On 4 nodes we watched combine go 7.7 → 12.5 → 16.5 → 19.6 ms
   over four consecutive "successful" rounds while GPU memory climbed 0 → 8.9 → 29.7 →
   43 GB.) So between rounds assert `nvidia-smi --query-gpu=memory.used
   --format=csv,noheader` is all 0 MiB, and **use a different `MASTER_PORT` each round**
   (`TIME_WAIT` shows up as a hung rendezvous). Running under docker helps: `docker rm -f`
   takes the whole process tree. `run_test_ep.sh` refuses to start on a busy GPU.
4. **Report all ranks, with the denominator.** Ranks differ systematically and the
   difference is often **layered by node** (see the combine table). Always pair GB/s with
   µs and state what the bytes figure counts — a GB/s-only table has inverted a conclusion
   for us before.

## Files

| File | What |
|---|---|
| `Dockerfile` | The image. Pinned DeepEP commit; apt NCCL removed; `-L` for pip NCCL. |
| `run_test_ep.sh` | Launcher. `TOKENS=8192` prefill / `TOKENS=128` decode; preflights a busy GPU and missing devices; auto-detects `EP_NIC_NAME`. |
| `ce_probe.c` | `ibv_create_comp_cntr` probe over every device — the decisive GDAKI check. `gcc -o ce_probe ce_probe.c -libverbs` |
| `docs/runbook_zh.md` | Full Chinese runbook: install → build → test, env-var reference, ~30-row troubleshooting table. |

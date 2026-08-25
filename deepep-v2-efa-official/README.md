# DeepEP V2 on AWS EFA — the released stack (`amazon-contributing/DeepEP`)

Builds and runs the **public release** of AWS's DeepEP V2 fork —
[`amazon-contributing/DeepEP`](https://github.com/amazon-contributing/DeepEP) — on
2 × `p5en.48xlarge` (8×H200 `sm_90` + 16×200 Gb/s EFA each), bare EC2 with Docker.

**Everything here comes from published packages.** No source-built NCCL, no
source-built aws-ofi-nccl, no hand-patched kernel module, no `LD_PRELOAD`.
**EFA installer 1.50.0 supplies all four load-bearing components at once.**

That is the difference from `deepep-v2-efa-gdaki-b200/` (local-only, not committed — 673 MB
of campaign results), which builds `Xuan-1998/DeepEP@dev` with a hand-assembled stack. Measured on the same
p5en pair, this packaged path costs **nothing** in performance (§ Results).

## The two environment variables you must set

1.50.0's `libnccl-net-ofi.so` registers **two** GIN plugins — a Libfabric
proxy-assisted one (**type 2**) and `Libfabric_GDAKI` (**type 5**) — and NCCL
selects **type 2** by default. `Loaded gin plugin Libfabric_GDAKI (v14)` prints
either way, so that line does **not** mean GDAKI ran. Force type 5:

```
NCCL_GIN_TYPE=5  NCCL_SYM_GIN_KERNELS_ENABLE=0     # both, or it crashes
```

12 SM, `--test-first-only`, no `EP_BUFFER_DEBUG`, mean over **all** ranks; `SO` =
per-rank scale-out bytes ÷ time as printed, which includes intra-node traffic and
so is not a wire rate
([full data](results/p5en_2n4n_20260825/summary.txt)):

| dispatch | type 2 (default) | **type 5** |
|---|---|---|
| 2 nodes / 16 ranks, 8192 tok | 1644.0 µs / 74.0 GB/s | **1502.9 µs / 81.2** |
| 2 nodes / 16 ranks, 128 tok | 365.1 µs | **169.4 µs (2.16×)** |
| 4 nodes / 32 ranks, 8192 tok | 4315.0 µs / 51.5 GB/s | **3955.3 µs / 56.0** (84% of wire) |
| 4 nodes / 32 ranks, 128 tok | 1003.2 µs, reps 833.1–1083.2 | **184.3 µs (5.44×)**, all 32 ranks in 183.4–185.6 |

**The gap widens with scale**, which is what makes this worth chasing: from 2 to 4
nodes type-2 decode dispatch goes 365.1 → 1003.2 µs (2.75×) while type 5 goes
169.4 → 184.3 µs (1.09×).

**Both are needed because the crash is informative.** `NCCL_GIN_TYPE=5` alone gives
`ncclGinValidateSignalRequest: GIN strong signals are required, but the GIN plugin
does not support them` → `GIN: DevComm setup failed on all available backends` →
`RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:217): 3`. The
symmetric-memory GIN kernels (on by default) require strong signals, which GDAKI does
not implement. While type 2 is still a candidate NCCL silently falls back to it; once
type 2 is excluded and the signal requirement stands, no backend is left.

**How to confirm which backend ran** (`NCCL_DEBUG=INFO`): the type-5 arm prints
`GIN/Plugin: Skipping plugin Libfabric index 3 type 2: NCCL_GIN_TYPE=5 requested`,
and `devCommCreate: creating 11 contexts`. `[Proxy Progress] Device N CPU core M`
lines are **not** a discriminator — 16 of them appear in both arms, because NCCL
builds proxy threads for ordinary collectives regardless.

Nothing else needs setting. `FI_EFA_USE_HW_CNTR=1`, `OFI_NCCL_GIN_STRONG_SIGNAL=1`
and `NCCL_RMA_DISABLE=1` are neutral to slightly worse (§ Results).

> **中文完整版 runbook：[`docs/runbook_zh.md`](docs/runbook_zh.md)** — host 安装、镜像构建、
> prefill 带宽 / decode 延迟测试、环境变量速查、故障对照表、CE 探针源码。
> This README is the condensed English version; the Chinese runbook is authoritative
> and carries the full troubleshooting table.

Verified on installer 1.50.0 / `deep_ep 2.1.0+ec623f3` across 2 and 4 nodes (16 and 32
ranks), exit 0 with correctness checks passing.

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

- **`ARG DEEPEP_REF` takes a branch, tag or bare sha, and defaults to a sha**
  (`8e7b42e…`, current `main`, measured perf-neutral against the tables below). Track the
  tip with `--build-arg DEEPEP_REF=main`. Two things make that safe: an
  `ADD https://api.github.com/…/commits/${DEEPEP_REF}` ahead of the clone, without which
  `RUN git fetch origin main` **hits a cached layer** and hands you last week's code while
  looking fresh — worse than an honest pin; and `git rev-parse HEAD` stamped to
  `/opt/DeepEP/BUILD_REF`, which `run_test_ep.sh` prints in every log so no number is
  unattributable. `git clone --depth 1 --branch <sha>` does not accept a bare sha, hence
  the `git init` + `git fetch --depth 1` form. Upstream **rewrites history** — the earlier
  `ec623f3` is no longer reachable from `main` (rewritten as `cc55cce`, with content
  changes, and `main` moved 4 commits past it), which a pin makes visible and a floating
  ref hides. Check for drift without building:
  `git ls-remote https://github.com/amazon-contributing/DeepEP.git refs/heads/main`,
  then re-measure before bumping.
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

**Validation of the hardened image** (rebuilt on every node, then re-run):
`git -C /opt/DeepEP rev-parse HEAD` = `ec623f31…`; `dpkg -l | grep -c libnccl2` = 0;
`ldconfig -p | grep libnccl.so` = empty; `/proc/self/maps` after `import deep_ep` shows
**exactly one** libnccl, pip's `nvidia/nccl/lib/libnccl.so.2`; `ncclGinPlugin_v14` present;
20 `comp_cntr` symbols; and the run logs `Loaded gin plugin Libfabric_GDAKI (v14)`.
Performance is unchanged by the hardening — prefill dispatch **74.0 GB/s SO / 1644.0 µs**
on the default backend, inside that arm's across-rep spread (§ Results).

## Run

```bash
# worker first, then leader; only NODE_RANK differs
GIN='NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0'
ssh <worker> "cd ~/work/deepep-v2-efa-official && TOKENS=8192 NUM_SMS=24 EXTRA_ENV='$GIN' bash run_test_ep.sh 1 <leader-ip>" &
ssh <leader> "cd ~/work/deepep-v2-efa-official && TOKENS=8192 NUM_SMS=24 EXTRA_ENV='$GIN' bash run_test_ep.sh 0 <leader-ip>"
```

`TOKENS=8192` → prefill (report **bandwidth**). `TOKENS=128` → decode (report
**latency**). Nothing else changes between the two.

`EXTRA_ENV="NAME=VALUE …"` is the launcher hook for one-off env A/Bs; it is how
the `NCCL_GIN_TYPE=5` pair is passed. Without that pair you get the type-2 proxy
backend: ~9% less prefill and 2.2–5.4× the decode latency (§ The two environment
variables). It is deliberately not baked in as a default so that the default arm
stays a measurable control.

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

**On this pin `--num-sms` is a pure perf axis — it does not change the QP count.**
`ec623f3` deleted `kMinGinContextSharingFactor`, so the auto path is the constant
`kDefaultGinContextCnt = 11` regardless of SM count (source arithmetic under § Why the GIN
backend is the lever). Every log in `results/p5en_2n4n_20260825/logs/` prints
`#QPs: 11/11` — at 6, 12, 24 and 32 SM, and at both 2 and 4 nodes. The SM-derived count
(`ceil_div(num_sms × 4, 10)`, so 12 SM → 5 and 24 SM → 10, clamped at 17) is the
**`7a6059a3` pin's** formula; the pin's logs print `#QPs: 5/5` at 12 SM. Nor is
`num_qps < num_ranks` fatal here: `--num-allocated-qps 5` at 4 nodes gives 32 ranks
`#QPs: 5/5` and still completes. To reproduce AWS's published
rows, use their operating points: **H200 2-node 12 SM / H200 4-node 6 SM / B200 2-node
12 SM**.

**For the fastest configuration, use 24 SM at both 2 and 4 nodes.** GIN type 5, 8192 tok,
mean over all ranks, no `EP_BUFFER_DEBUG`
([data](results/p5en_2n4n_20260825/summary.txt) TABLE 5/6):

| | 2N dispatch | 2N redComb | 2N sum | 4N dispatch | 4N redComb | 4N sum |
|---|---|---|---|---|---|---|
| 6 SM | 2290.5 µs | 7377.7 µs | 9668.2 µs | 4030.7 µs | 9518.8 µs | 13549.4 µs |
| 12 SM | 1502.9 µs | 4237.9 µs | 5740.8 µs | 3955.3 µs | 7943.2 µs | 11898.5 µs |
| 16 SM | 1510.5 µs | 3584.1 µs | 5094.6 µs | — | — | — |
| **24 SM** | 1535.7 µs | **3362.6 µs** | **4898.3 µs** | 3972.7 µs | **7728.3 µs** | **11701.0 µs** |
| 32 SM | 1576.5 µs | 3486.4 µs | 5062.9 µs | — | — | — |

Reps: 24 SM ×3 at 2N, 12 SM ×3 and 24 SM ×2 at 4N; single runs elsewhere.

**Dispatch is nearly flat from 12 to 24 SM** (+2.2% at 2 nodes, +0.4% at 4) — it is
*reduced combine* that pays for a small SM count. So the trade is 32.8 µs of dispatch for
875.3 µs of reduced combine at 2 nodes (layer total **−14.7%**), and a much smaller but
same-signed trade at 4 nodes (**−1.7%**). Decode agrees: at 2 nodes 24 SM wins outright
(dispatch 147.3 vs 169.4 µs, redComb 160.1 vs 179.0 µs); at 4 nodes decode dispatch is
flat across 6/12/24 SM (181.2–184.7 µs) and 24 SM wins on redComb (239.6 vs 253.3 µs, −5.4%).

**6 SM is the wrong choice at every scale** — against 24 SM, dispatch is +49.2% and
dispatch+redComb +97.4% at 2 nodes; at 4 nodes dispatch is only +1.5% but
dispatch+redComb is +15.8%. State which of the two you mean: they diverge this widely
because reduced combine, not dispatch, is what pays for a small SM count. 6 SM never
hangs, so nothing is protecting you from it.

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

## Results (p5en.48xlarge × 2 and × 4, 2026-08-25)

Ubuntu 24.04, driver 595.91.07, installer 1.50.0 + reboot. Container torch 2.13.0+cu130 /
nccl 2.31.2 / `deep_ep 2.1.0+ec623f3`. `test_ep.py` exits 0 at 16 and 32 ranks with all
correctness checks passing; `fi_pingpong -p efa` passes (64 B 1.73 MB/s, 4 K 282.48 MB/s).
The prefill, decode and layering tables below are on **GIN type 5**, at **24 SM**, with
`EP_BUFFER_DEBUG` **off**, and every figure is a mean over **all** ranks. The comparison
sections after them (source stack, QP layout, PR arms) are at **12 SM**, to match the
hand-built pin and the PR branch; each table states its own SM count. Raw logs and the
full matrix (backend A/B, SM scan, env teardown, PR arms):
[`results/p5en_2n4n_20260825/`](results/p5en_2n4n_20260825/summary.txt).

**Denominators, stated once.** `SU` = the printed per-rank `bytes` ÷ time exactly.
**`SO` is not a wire rate** — without `--ignore-local-traffic` it counts intra-node
destinations too. The per-rank wire ceiling on p5en is **50 GB/s** (16 × 200 Gb/s ÷ 8
GPUs), so a reported 79 GB/s dispatch is by itself proof the figure is not a network
number; the wire fraction is `SO × (N−1)/N ÷ 50`, which at 2 nodes is numerically `SO`.
Pass `IGNORE_LOCAL=1` for a wire-rate run.

### Prefill (`--num-tokens=8192`) — bandwidth

| op | 2 nodes / 16 ranks | | | 4 nodes / 32 ranks | | |
|---|---|---|---|---|---|---|
| | SO GB/s | SU GB/s | time | SO GB/s | SU GB/s | time |
| dispatch | 79–80 | 257–263 | 1535.7 µs | 55–56 | 111–113 | 3972.7 µs |
| expanded dispatch | 79–80 | 258–262 | 1537.3 µs | 55–56 | 111–113 | 3971.3 µs |
| cached dispatch | 75–76 | 244–249 | 1620.7 µs | 51–52 | 103–106 | 4256.6 µs |
| combine | 62–73 | 202–239 | 3534.0 µs | 51–57 | 103–115 | 7710.7 µs |
| reduced combine | 64–79 | 209–259 | 3362.6 µs | 51–57 | 103–115 | 7728.3 µs |

**scale-up** bytes/rank (the `bytes` the log prints): dispatch 395.9–402.4 MB (2N) /
441.4–447.7 MB (4N); combine 759.7–772.1 / 847.0–859.0 MB. **scale-out** bytes/rank are
not printed — recover them as `SO × time`: dispatch 121.3–123.0 MB (2N) / 219.1–223.6 MB
(4N). 3 reps at 2N (48 rank observations), 2 at 4N (64); **time is the mean, `SO`/`SU`
are min–max across ranks** — two different statistics, see below.

**Dispatch runs at 80% of the wire ceiling at 2 nodes and 84% at 4** — 4 nodes is the
higher wire fraction because 3/4 of the traffic leaves the box instead of 1/2, even
though the raw `SO` is lower: true cross-node bytes go 61.0 → 165.9 MB per rank (×2.72)
while time goes ×2.59. **Going 2 → 4 nodes costs 2.59× in dispatch time** for 2× the
ranks, so scale-out here is sublinear in a way dispatch bandwidth alone does not show;
report the µs.

#### Why `SO`/`SU` are ranges — it is not run-to-run variance

The ranges are the **min–max across the 48 (or 64) rank observations**, while the time
column is their mean. Decomposed, at 2N / 24 SM / 8192 tok:

| cross-rank spread | time | printed bytes | `SO` |
|---|---|---|---|
| dispatch | **0.5%** | 1.6% | 1.3% |
| cached dispatch | 0.4% | 1.6% | 1.3% |
| combine | **16.1%** | 1.6% | 16.5% |
| reduced combine | **20.7%** | 1.6% | 21.3% |

1. **For dispatch the range comes from the denominator, not the speed.** Time varies 0.5%
   across ranks (0.9% at 4N), but each rank routes a different token count
   (`get_unbalanced_scores`), so its byte count varies 1.4–1.6%, and `GB/s = that rank's
   bytes ÷ that rank's time` inherits it.
2. **Integer printing adds a floor.** `test_ep.py` prints GB/s with `:.0f`. At `SO ≈ 79.6`
   one integer step is **1.26%**; at `SO ≈ 55.8` it is **1.79%**. So `79–80` and `55–56`
   are two adjacent integers — **the narrowest range this format can express.**
3. **Only combine is genuinely spread, and the driver is time, not bytes** — that is the
   per-node layering below (node1 3300.3 vs node2 3767.7 µs at 2N, 13.2%; the pooled
   4-node figure is 4.2% only because the slow machine differs between its two reps).
   Dispatch does not layer at all (0.1% between nodes).
4. **Reproducibility is excellent even for combine.** All-rank means per rep, 2N / 24 SM:
   dispatch 1535.5 / 1537.0 / 1534.5 µs (**0.16%**), combine 3533.2 / 3535.1 / 3533.6
   (**0.05%**), reduced combine 3366.3 / 3356.1 / 3365.5 (**0.30%**); 4N dispatch 3971.6 /
   3973.8 (**0.06%**). Combine's 16–21% is a **reproducible structural spread across
   ranks**, not instability. Regenerate the whole block with
   `python3 results/p5en_2n4n_20260825/make_tables.py`.

### Decode (`--num-tokens=128`) — latency

At this size only latency is meaningful: 5.6–6.4 MB per rank, 11–19 GB/s SO. It is
**message-rate** bound, not bandwidth bound.

| op | 2 nodes / 16 ranks | 4 nodes / 32 ranks |
|---|---|---|
| dispatch | 147.3 µs | 184.7 µs |
| expanded dispatch | 146.6 µs | 184.0 µs |
| cached dispatch | 135.8 µs | 184.3 µs |
| combine | 151.9 µs | 236.9 µs |
| reduced combine | 160.1 µs | 239.6 µs |

**Crossing from 2 to 4 nodes costs only +37 µs of dispatch (+25%)** while doubling the
rank count — decode dispatch on type 5 barely notices scale. combine is where 4 nodes
actually hurts (+56%). Two pending PRs take 2-node dispatch to 106 µs; see below.

### combine is layered by node — this decides how you aggregate

dispatch is uniform across ranks and across machines; combine and reduced combine split
cleanly by machine. 2 nodes / 24 SM / 8192 tok, 3 reps pooled, mean over each node's
8 ranks:

| op | node 1 | node 2 |
|---|---|---|
| dispatch | 1535.0 µs | 1536.3 µs (0.1% apart) |
| combine | 3300.3 µs | 3767.7 µs (**13.2% apart**) |
| reduced combine | 3074.9 µs | 3650.3 µs (**17.1% apart**) |

Which machine is the slow one is **fixed within a round of consecutive reps and flips
between rounds** — it is a per-launch property, not per-iteration noise. All three
official 24 SM reps put node 2 slow (combine 3764 / 3771 / 3769 µs against
3303 / 3299 / 3299); both `main` 24 SM reps and both pin reps put the other machine
slow. The magnitude is stable across all of them at 14–18%. So pooling reps does *not*
flatten 2-node layering, and a per-node combine mean lands at either ~3100 or ~3770 µs
depending on which log you happen to read. Pooled over all 16 ranks the three reps agree
to 0.30% (reduced combine 3366.3 / 3356.1 / 3365.5 µs) — **always pool every rank.**
At 4 nodes the slow machine differs between the two reps (node 4 in rep 1 at 8087 µs,
node 1 in rep 2 at 8117 µs), which is why the pooled 4-node layering looks small
(4.2%). We draw no mechanism conclusion about *why* one machine is slower.

### Versus building the whole stack from source

There is no performance reason to. Measured on the same nodes on the same day, GDAKI
active on both sides: the `7a6059a3` pin (source NCCL `2.30.7-1` + source aws-ofi-nccl
`--enable-gdaki`, launched with `GDAKI=1`, `--skip-check`) against the packaged
`ec623f3` image with the two env vars. 12 SM, all-rank means:

| | pin, source stack | packaged, type 5 |
|---|---|---|
| 2N / 8192 tok dispatch | 1517.0 µs | **1502.9 µs** |
| 2N / 8192 tok cached dispatch | 1749.1 µs | **1591.0 µs** (−9.0%) |
| 2N / 128 tok dispatch | 301.2 µs | **169.4 µs** (1.78×) |
| 4N / 8192 tok dispatch | 3962.5 µs | **3955.3 µs** |
| 4N / 8192 tok cached dispatch | 4709.2 µs | **4239.7 µs** (−10.0%) |
| 4N / 128 tok dispatch | 320.9 µs | **184.3 µs** (1.74×) |

Prefill dispatch is a tie (0.2–0.9% apart). The packaged path wins cached dispatch by
9.9–11.1% and decode dispatch by 1.74–1.78×. Everything the hand-built stack was assembled
to obtain — GDA ops, the CE counting signal, the GDAKI plugin — ships in installer 1.50.0.

### Why the GIN backend is the lever and the QP layout is not

`--num-allocated-qps` is the obvious suspect, because the two stacks really do run
different QP layouts, and the flag really does work: pass `--num-allocated-qps 5` at
`ec623f3` and the header goes `#QPs: 11/11` → `#QPs: 5/5` with the debug line printing
`5 / 49 / 5`. It buys nothing:

| 2N / 12 SM / 8192 tok | dispatch | cached dispatch | combine | reduced combine |
|---|---|---|---|---|
| type 2, default 11 contexts | 1644.0 µs | 1652.5 µs | 3555.2 µs | 4219.6 µs |
| type 2, `--num-allocated-qps 5` | 1631.1 µs | 1615.0 µs | 3719.3 µs | 4289.8 µs |
| type 5, default 11 contexts | **1502.9 µs** | **1591.0 µs** | 3602.5 µs | 4237.9 µs |
| type 5, `--num-allocated-qps 5` | 1508.5 µs | 1743.9 µs | 3605.1 µs | 4228.9 µs |

Plain dispatch moves −0.8% on type 2 and not at all on type 5. Where the layout *does*
matter is **cached** dispatch, and there 5 contexts is 9.6% **worse** on type 5; at 4
nodes it is 19.3% worse on decode dispatch (1003.2 → 1197.0 µs). **Leave it at the
default 11.** The source arithmetic below explains why the layouts differ and confirms
the kernel is identical on both sides — which is what makes the backend the only
remaining variable.

**It is the same kernel — established at blob level, not from commit messages.** Everything
below is verifiable from source, no hardware needed. At `ec623f3` the hybrid dispatch kernel
exists twice (`ec623f3` = `feat: add EP_HYBRID_KERNEL toggle between unordered and ordered
kernels`, `unordered` the default per `csrc/kernels/elastic/kernel_select.hpp:37-52`), and
comparing blobs settles which is which:

| file at `ec623f3` | vs `af9a040:hybrid_dispatch.cuh` (pre-fork upstream) | vs `7a6059a3:hybrid_dispatch.cuh` (the hand-built pin) |
|---|---|---|
| `hybrid_dispatch.cuh` — the `ordered` variant | **+6 / −1** | +56 / −468 |
| `hybrid_dispatch_unordered.cuh` — **the default** | +485 / −51 | **+45 / −28** (11 of them the license header) |

So `unordered` **is** `7a6059a3`'s kernel — the AWS EFA GDA port — under a new name, and
`ordered` is the upstream kernel this branch forked from, carried along nearly verbatim.
Combine tells the same story (`+12/−6` vs `+24/−7`). The entire functional content of that
+45/−28: a new `num_unaligned_recv_tokens_per_expert` output pointer (one store per expert),
two lambdas hoisted out of the two warp branches, the identity lambda `phys_token_slot`
dropped, and comments.

⇒ **`EP_HYBRID_KERNEL=ordered` is not a valid A/B.** That flag selects the *upstream*
kernel, which publishes a tail signal and lets the receiver assume everything before it
landed — measured **incorrect on EFA GDAKI** (`NCCL_GIN_TYPE=5`, where the signal can
overtake the data); it is only correct on the ordered proxy path. It also asks NCCL for a
different fabric altogether
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
| `7a6059a3` (the pin) | `ceil_div(12×4, kMinGinContextSharingFactor=10)` = **5**, so **SM-dependent** | `(256 − 2·5)/5` = 49 | 4 | `ceil(48/4)` = 12 | `49/12` → **4** | 4 | `5 / 49 / 5` ✅ |
| `ec623f3` (the release) | `kMinGinContextSharingFactor` deleted; auto path is the constant `kDefaultGinContextCnt` = **11**, **SM-independent** | `(256 − 2·11)/11` = 21 | 10 | `ceil(48/10)` = 5 | `21/5` → **4** | 4 | `11 / 21 / 11` ✅ |

**Note what does not change: `kNumParts = 4` and 48 channels on both sides**, so the kernel
is instantiated with identical template arguments. The only live difference is how many QPs
those same 48 channels are spread across — **4 QPs × 12 channels (pin) vs 10 QPs × 5
channels (release)** — and the table above measures that difference to be worth nothing on
plain dispatch.

Two things the arithmetic rules out as candidate mechanisms:

- **`kNumParts` is not it.** Computed from `compute_part_allocation()`
  (`gin_resource_alloc.cuh:122-152`): **4 on both sides**. `ec623f3`'s own comment agrees,
  listing 11's equivalents as `{5, 6, 7, 8, 9, 14}`.
- **The release did not "lose the tuning".** Front-loading (`kMidTotal`) and the forward
  double-buffer (`kNumDispatchFwdBuffers`) are both present in
  `hybrid_dispatch_unordered.cuh` — necessarily, since that file *is* `7a6059a3`'s kernel.

**Flag semantics differ between the two commits**, which matters if you try this on the
pin. Both expose `--num-allocated-qps` / `--num-qps` in `tests/elastic/test_ep.py`, but at
`7a6059a3` the request is capped by the SM-derived context count (`nccl.cu:172-179` warns
and overrides it down), so at 12 SM 5 is a ceiling and 11 is unreachable. At `ec623f3` the
request **replaces** the default, bounded only by
`[kMinGinContextCnt=2, kMaxGinContextCnt=17]` (`nccl.cu:129-137`).

**Putting every 2-node prefill arm on one axis** (16 ranks, 12 SM, 8192 tok, all-rank means):

| arm | dispatch | SO GB/s | wire% |
|---|---|---|---|
| release `ec623f3`, default env (type 2) | 1644.0 µs | 74.0 | 74.0 |
| release, `--num-allocated-qps 5` (type 2) | 1631.1 µs | 75.0 | 75.0 |
| pin `7a6059a3`, source stack, `--skip-check` | 1517.0 µs | 80.6 | 80.6 |
| release, full 5-var route B | 1504.4 µs | 81.0 | 81.0 |
| **release, `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`** | **1502.9 µs** | 81.2 | 81.2 |
| `main` `8e7b42e` + the type-5 pair | 1502.0 µs | 81.0 | 81.0 |

`wire% = SO × (N−1)/N ÷ 50 GB/s`; at N=2 that is numerically SO. The two env vars account
for the entire spread — the other three route-B vars add nothing on top of them, and
`main` is indistinguishable from `ec623f3`. `OFI_NCCL_GIN_STRONG_SIGNAL=1` on its own is
actively bad at 128 tok (750.4 µs mean, 16-rank spread 371.0–1130.0 µs), consistent with the
unordered kernel only requiring weak signals.

### Decode (`--num-tokens=128`) — two independent wins that stack

At this size only latency is meaningful: **5.6–6.4 MB per rank**. It is **message-rate**
bound, not bandwidth bound, and there are two separate levers: the GIN backend (env only)
and the dispatch part geometry (two pending PRs).

Both are measured on one image built from PR #2's head `b097b03`, which has PR #1 as an
ancestor — so that single SHA *is* the "#1 + #2 stacked" arm.

That head's merge-base with `main` is `cc55cce`, so the patched image does **not** share a
base with the unpatched `ec623f3` one. Two things keep the comparison honest: the
clamp-off control (`EP_MIN_TOKENS_PER_PART=1`) runs **inside the patched image**, so the
clamp's effect is measured with the base held constant (171.5 → 112.7 µs at 2 nodes,
−34.3%); and `main` `8e7b42e`, which contains everything the patched base has and more,
measures within 0.7% of `ec623f3` on 8 paired rows (§ Results).

| PR (base `main`) | What |
|---|---|
| [#1](https://github.com/amazon-contributing/DeepEP/pull/1) | Forward `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` / `EP_MIN_TOKENS_PER_PART` to the JIT. Changes no default. |
| [#2](https://github.com/amazon-contributing/DeepEP/pull/2) | Add `kMinTokensPerPart` (default 15, `EP_MIN_TOKENS_PER_PART` overrides): `kNumParts = min(budget, tokens_per_channel / 15)`. |

**2 nodes / 16 ranks / 12 SM, 128 tok** (all-rank means):

| arm | dispatch | vs its own backend | combine | reduced combine |
|---|---|---|---|---|
| unpatched, type 2 | 365.1 µs | 1.00× | 175.1 µs | 192.7 µs |
| #1 + #2, type 2 | 237.1 µs | **−35.1%** | 176.3 | 193.6 |
| unpatched, type 5 | 169.4 µs | 1.00× | 162.7 | 179.0 |
| #1 + #2, type 5 | 112.7 µs | **−33.5%** | 162.2 | 178.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`, type 5 | **106.4 µs** | **−37.2%** | 162.1 | 178.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`, type 5 | 171.5 µs | clamp-off control | 162.6 | 178.8 |

**4 nodes / 32 ranks / 12 SM, 128 tok:**

| arm | dispatch | vs its own backend | combine | reduced combine |
|---|---|---|---|---|
| unpatched, type 2 | 1003.2 µs | 1.00× | 344.2 µs | 357.0 µs |
| #1 + #2, type 2 | 627.4 µs | **−37.5%** | 336.9 | 346.8 |
| unpatched, type 5 | 184.3 µs | 1.00× | 244.2 | 253.3 |
| #1 + #2, type 5 | 169.5 µs | **−8.0%** | 243.5 | 252.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`, type 5 | **155.9 µs** | **−15.4%** | 243.6 | 251.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`, type 5 | 184.1 µs | clamp-off control | 243.5 | 252.9 |

Three things to read off these:

1. **The two levers are independent and they stack.** Stacked, 2-node decode dispatch goes
   365.1 → 106.4 µs — **3.43×** — from one env pair and two commits.
2. **combine and reduced combine are untouched to within 1% in every row.** That is the
   evidence the PR mechanism is part geometry inside dispatch and not the network: EFA is
   doing the same work either way. Expanded and cached dispatch track plain dispatch.
3. **The clamp's win collapses at 4 nodes on type 5** — −8.0% instead of −33.5% — and the
   clamp-off control (184.1 µs) sits inside noise of unpatched (184.3 µs), which confirms
   it is the clamp that stopped paying rather than something else in the PRs. Type-5
   4-node decode dispatch has a floor around **156–185 µs** that part geometry does not
   reach; `EP_NUM_SUB_PARTS=1` gets furthest and no further. This is unexplained — do not
   extrapolate the 2-node ratio to larger clusters.

**Why part geometry costs anything at all.** `kNumParts` (how many `flush_part` puts a
channel's tokens leave in) is set only by `compute_part_allocation()`, which caps *from
above* when the GIN indexed-signal budget is tight — and the budget is loosest precisely
when a channel holds the fewest tokens. So decode always lands on `kMaxParts`, the worst
end of the axis. At 128 tokens / 12 SM a channel holds 3 tokens but is described as 4
parts × 1 token: the last part is always empty, and 3 tokens leave as three single-token
puts instead of one 3-token put. Sub-parts already have both guards parts lack (a clamp to
`kBatchSize`, plus `EP_SM100_MIN_SUB_TOKENS`).

**Which to use.** Only care about prefill → take `main` as-is; the PRs are within noise on
prefill (2N/24 SM/8192: 1535.7 µs unpatched vs 1536.0 µs; 4N/12 SM/8192: 3955.3 vs 3955.3).
**Publishing decode / small-token numbers → cherry-pick both**, and set the type-5 env pair
regardless. #1 alone changes no default, so it only pays stacked on #2. After they merge,
#2's default of 15 applies automatically — no env var needed. To get the unclamped geometry
back as a control use `EP_MIN_TOKENS_PER_PART=1`, which **short-circuits** to the old value
(dividing by 1 is a *third* geometry, not a control).

**With the PRs applied, 12 SM beats 24 SM for 2-node decode dispatch** (112.7 vs 145.3 µs),
so the 24-SM recommendation above holds for unpatched code. At 4 nodes the two tie on
dispatch + reduced combine (422.4 vs 422.2 µs).

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

## Not measured yet

Ranked by what a reader would most likely want and what it costs. None of these
needs an image rebuild.

1. **DeepEP V1 at `--num-tokens 8192` with FP8 dispatch** (1 run, `deepep-v1-efa:dev`).
   The repo's top-level throughput table puts V1 at 4096/BF16 next to DeepEP V2 at
   8192/FP8, so those two rows cannot be compared in either direction: V1 has no
   8192-token run and no GDAKI campaign ever ran 4096. V1 already publishes FP8
   dispatch at 4096 (48.17 p5 / 54.98 p5en), so the missing arm is one run on the
   V1 side, not a re-run of this campaign.
2. **The 4-node `--num-sms` axis is open at the top** (2 runs: 4N/16 SM, 4N/32 SM).
   At 2 nodes the axis is closed and has an *interior* optimum at 24 SM
   (dispatch + reduced combine 4898.3 µs, vs 5062.9 at 32 SM). At 4 nodes only
   6/12/24 exist and the curve is still monotone improving
   (13549.4 → 11898.5 → 11701.0 µs), so the 24-SM recommendation for 4 nodes rests
   on an unclosed axis.
3. **No `--ignore-local-traffic` run at any scale** (2 runs: 2N, 4N). The launcher
   has the hook (`IGNORE_LOCAL=1`, `run_test_ep.sh:124`) and every GB/s here is the
   **SO** denominator, which counts intra-node destinations. `wire% = SO × (N−1)/N ÷ 50`
   was checked exact against a measured run at **2 nodes only**; the ×0.75 factor at
   4 nodes is arithmetic nobody has validated.
4. **No 1-node baseline** (1 run). Everything here is ≥ 2 nodes, so DeepEP's own
   kernel cost and the EFA crossing cost are never separated.
5. **Which machine is the slow one in combine** (1 run with `NODE_RANK` swapped).
   Combine and reduced combine layer by machine (13–17% at 2 nodes) and the slow
   machine is fixed within a batch of reps but flips between batches; swapping the
   leader role tells you whether it follows the machine or the role.
6. **The thin 4-node arms** (a 3rd rep for the 4N rows). Most 4-node arms here are
   1–2 reps against 3 at 2 nodes; rep-to-rep spread is ≤ 0.31% where measured, but
   that is measured mostly at 2 nodes.

## Files

| File | What |
|---|---|
| `Dockerfile` | The image. Pinned DeepEP commit; apt NCCL removed; `-L` for pip NCCL. |
| `run_test_ep.sh` | Launcher. `TOKENS=8192` prefill / `TOKENS=128` decode; preflights a busy GPU and missing devices; auto-detects `EP_NIC_NAME`. |
| `ce_probe.c` | `ibv_create_comp_cntr` probe over every device — the decisive GDAKI check. `gcc -o ce_probe ce_probe.c -libverbs` |
| `docs/runbook_zh.md` | Full Chinese runbook: install → build → test, env-var reference, ~30-row troubleshooting table. |
| `results/p5en_2n4n_20260825/make_tables.py` | **Emits every table published here** from `logs/`. `python3 make_tables.py` and paste — do not hand-edit a table. Includes a completeness audit (0 of 69 run tags short a rank). |
| `results/p5en_2n4n_20260825/parse_ep.py` | Per-tag inspector: `EPRUNS=./logs python3 parse_ep.py <tag>`. Parses with `finditer` — concurrent ranks glue two records onto one physical line. |
| `results/p5en_2n4n_20260825/summary.txt` | The full matrix, generated. `logs/` holds every node's raw log. |

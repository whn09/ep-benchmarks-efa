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
printed per-rank scale-out bytes ÷ time, not a wire rate
([full data](results/p5en_2n4n_20260825/summary.txt)):

| dispatch | type 2 (default) | **type 5** |
|---|---|---|
| 2 nodes / 16 ranks, 8192 tok | 1644.0 µs / 74.0 GB/s | **1502.9 µs / 81.2** |
| 2 nodes / 16 ranks, 128 tok | 365.1 µs | **169.4 µs (2.16×)** |
| 4 nodes / 32 ranks, 8192 tok | 4316.2 µs / 51.5 GB/s | **3955.3 µs / 56.0** (84% of wire) |
| 4 nodes / 32 ranks, 128 tok | 1001.2 µs, reps 833–1083 | **184.5 µs (5.43×)**, all 32 ranks in 183.4–185.9 |

**The gap widens with scale**, which is what makes this worth chasing: from 2 to 4
nodes type-2 decode dispatch goes 365 → 1001 µs (2.7×) while type 5 goes
169.4 → 184.5 µs (1.09×).

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

**And `--num-sms` is not a free parameter.** It also changes the allocated QP count, and
*non-monotonically*: measured 0→17 QPs, 12→5, 24→10. A value landing on
`num_qps < num_ranks` **hangs outright** — the GIN auto-tuner prints its lines and then
nothing for 600 s. 16 ranks / 12 SM is fine (`num_qp=11`). To reproduce AWS's published
rows, use their operating points: **H200 2-node 12 SM / H200 4-node 6 SM / B200 2-node
12 SM**.

**For the fastest configuration, use 24 SM at both 2 and 4 nodes.** GIN type 5, 8192 tok,
mean over all ranks, no `EP_BUFFER_DEBUG`
([data](results/p5en_2n4n_20260825/summary.txt) TABLE 5/6):

| | 2N dispatch | 2N redComb | 2N sum | 4N dispatch | 4N redComb | 4N sum |
|---|---|---|---|---|---|---|
| 6 SM | 2290.5 µs | 7371.6 µs | 9662.1 µs | 4031.7 µs | 9494.3 µs | 13526.0 µs |
| 12 SM | 1502.9 µs | 4226.0 µs | 5728.9 µs | 3955.3 µs | 7963.6 µs | 11918.9 µs |
| 16 SM | 1510.6 µs | 3568.1 µs | 5078.7 µs | — | — | — |
| **24 SM** | 1535.7 µs | **3364.3 µs** | **4900.0 µs** | 3972.6 µs | **7709.8 µs** | **11682.4 µs** |
| 32 SM | 1576.5 µs | 3486.4 µs | 5062.9 µs | — | — | — |

Reps: 24 SM ×3 at 2N, 12 SM ×3 and 24 SM ×2 at 4N; single runs elsewhere.

**Dispatch is nearly flat from 12 to 24 SM** (+2.2% at 2 nodes, +0.4% at 4) — it is
*reduced combine* that pays for a small SM count. So the trade is 33 µs of dispatch for
862 µs of reduced combine at 2 nodes (layer total **−14.5%**), and a much smaller but
same-signed trade at 4 nodes (**−2.0%**). Decode agrees: at 2 nodes 24 SM wins outright
(dispatch 147.3 vs 169.4 µs, redComb 160.4 vs 179.0 µs); at 4 nodes decode dispatch is
flat across 6/12/24 SM (181–185 µs) and 24 SM wins on redComb (239.1 vs 253.5 µs, −5.7%).

**6 SM is the wrong choice at every scale** — 52% worse than 24 SM at 2 nodes, 13.5%
worse at 4 — and it never hangs, so nothing is protecting you from it.

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
Every number below is on **GIN type 5**, at **24 SM**, with `EP_BUFFER_DEBUG` **off**, and
is a mean over **all** ranks. Raw logs and the full matrix (backend A/B, SM scan, env
teardown, PR arms):
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
| dispatch | 79–80 | 257–263 | 1535.6 µs | 55–56 | 111–113 | 3972.6 µs |
| expanded dispatch | 79–80 | 258–262 | 1537.3 µs | 55–56 | 111–113 | 3970.7 µs |
| cached dispatch | 75–76 | 244–249 | 1620.7 µs | 51–52 | 103–106 | 4256.6 µs |
| combine | 62–73 | 202–239 | 3529.1 µs | 51–57 | 103–115 | 7702.1 µs |
| reduced combine | 64–79 | 209–259 | 3364.3 µs | 51–57 | 103–115 | 7710.1 µs |

bytes/rank: dispatch 395.9–402.4 MB (2N) / 441.4–447.7 MB (4N); combine 759.7–772.1 MB /
847.0–859.0 MB. 3 reps at 2N, 2 at 4N; SO/SU are min–max over all ranks and reps, time is
the pooled mean.

**Dispatch runs at 80% of the wire ceiling at 2 nodes and 84% at 4** — 4 nodes is the
higher wire fraction because 3/4 of the traffic leaves the box instead of 1/2, even
though the raw `SO` is lower. **Going 2 → 4 nodes costs 2.59× in dispatch time** for 2×
the ranks and 1.12× the per-rank bytes, so scale-out here is sublinear in a way dispatch
bandwidth alone does not show; report the µs.

### Decode (`--num-tokens=128`) — latency

At this size only latency is meaningful: 5.6–6.4 MB per rank, 11–19 GB/s SO. It is
**message-rate** bound, not bandwidth bound.

| op | 2 nodes / 16 ranks | 4 nodes / 32 ranks |
|---|---|---|
| dispatch | 147.3 µs | 184.7 µs |
| expanded dispatch | 146.6 µs | 184.0 µs |
| cached dispatch | 135.8 µs | 184.2 µs |
| combine | 151.9 µs | 236.5 µs |
| reduced combine | 160.4 µs | 239.1 µs |

**Crossing from 2 to 4 nodes costs only +37 µs of dispatch (+25%)** while doubling the
rank count — decode dispatch on type 5 barely notices scale. combine is where 4 nodes
actually hurts (+56%). Two pending PRs take 2-node dispatch to 106 µs; see below.

### combine is layered by node — this decides how you aggregate

dispatch is uniform across ranks; combine and reduced combine split cleanly by machine.
One 2-node / 24 SM / 8192 tok run, mean over each node's 8 ranks:

| op | node 1 | node 2 |
|---|---|---|
| dispatch | 1536.0 µs | 1535.0 µs (0.1% apart) |
| combine | 3302.9 µs | 3763.9 µs (**14% apart**) |
| reduced combine | 3091.7 µs | 3653.9 µs (**18% apart**) |

Which machine is the slow one flips between runs, so a per-node combine mean lands at
either ~3100 or ~3670 µs depending on which log you happen to read — that looks like
bistable behaviour and is not. Pooled over all 16 ranks the three reps agree to 1.7%
(3391.5 / 3335.7 / 3365.5 µs). **Always pool every rank.** We draw no mechanism
conclusion about *why* one machine is slower.

### Versus building the whole stack from source

There is no performance reason to. Measured on the same nodes on the same day, GDAKI
active on both sides: the `7a6059a3` pin (source NCCL `2.30.7-1` + source aws-ofi-nccl
`--enable-gdaki`, launched with `GDAKI=1`, `--skip-check`) against the packaged
`ec623f3` image with the two env vars. 12 SM, all-rank means:

| | pin, source stack | packaged, type 5 |
|---|---|---|
| 2N / 8192 tok dispatch | 1515.0 µs | **1502.9 µs** |
| 2N / 8192 tok cached dispatch | 1749.0 µs | **1591.0 µs** (−9.0%) |
| 2N / 128 tok dispatch | 303.9 µs | **169.4 µs** (1.79×) |
| 4N / 8192 tok dispatch | 3961.0 µs | **3955.3 µs** |
| 4N / 8192 tok cached dispatch | 4710.3 µs | **4239.4 µs** (−10.0%) |
| 4N / 128 tok dispatch | 320.8 µs | **184.5 µs** (1.74×) |

Prefill dispatch is a tie (0.1–0.8% apart). The packaged path wins cached dispatch by
9–10% and decode dispatch by 1.74–1.79×. Everything the hand-built stack was assembled
to obtain — GDA ops, the CE counting signal, the GDAKI plugin — ships in installer 1.50.0.

### Why the GIN backend is the lever and the QP layout is not

`--num-allocated-qps` is the obvious suspect, because the two stacks really do run
different QP layouts, and the flag really does work: pass `--num-allocated-qps 5` at
`ec623f3` and the header goes `#QPs: 11/11` → `#QPs: 5/5` with the debug line printing
`5 / 49 / 5`. It buys nothing:

| 2N / 12 SM / 8192 tok | dispatch | cached dispatch | combine | reduced combine |
|---|---|---|---|---|
| type 2, default 11 contexts | 1644.0 µs | 1652.2 µs | 3512.5 µs | 4196.6 µs |
| type 2, `--num-allocated-qps 5` | 1631.1 µs | 1614.5 µs | 3719.3 µs | 4281.0 µs |
| type 5, default 11 contexts | **1502.9 µs** | **1591.0 µs** | 3629.9 µs | 4226.0 µs |
| type 5, `--num-allocated-qps 5` | 1508.5 µs | 1743.9 µs | 3605.1 µs | 4236.6 µs |

Plain dispatch moves −0.8% on type 2 and not at all on type 5. Where the layout *does*
matter is **cached** dispatch, and there 5 contexts is 9.6% **worse** on type 5; at 4
nodes it is 19.6% worse on decode dispatch (1001.2 → 1197.5 µs). **Leave it at the
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
| pin `7a6059a3`, source stack, `--skip-check` | 1515.0 µs | 80.6 | 80.6 |
| release, full 5-var route B | 1505.4 µs | 81.0 | 81.0 |
| **release, `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`** | **1502.9 µs** | 81.2 | 81.2 |
| `main` `8e7b42e` + the type-5 pair | 1502.0 µs | 81.0 | 81.0 |

`wire% = SO × (N−1)/N ÷ 50 GB/s`; at N=2 that is numerically SO. The two env vars account
for the entire spread — the other three route-B vars add nothing on top of them, and
`main` is indistinguishable from `ec623f3`. `OFI_NCCL_GIN_STRONG_SIGNAL=1` on its own is
actively bad at 128 tok (750.4 µs mean, 32-rank spread 371–1130 µs), consistent with the
unordered kernel only requiring weak signals.

### Decode (`--num-tokens=128`) — two independent wins that stack

At this size only latency is meaningful: **5.6–6.4 MB per rank**. It is **message-rate**
bound, not bandwidth bound, and there are two separate levers: the GIN backend (env only)
and the dispatch part geometry (two pending PRs).

Both are measured on one image built from PR #2's head `b097b03`, which has PR #1 as an
ancestor — so that single SHA *is* the "#1 + #2 stacked" arm.

| PR (base `main`) | What |
|---|---|
| [#1](https://github.com/amazon-contributing/DeepEP/pull/1) | Forward `EP_NUM_SUB_PARTS` / `EP_MIN_SUB_TOKENS` / `EP_SM100_MIN_SUB_TOKENS` / `EP_MIN_TOKENS_PER_PART` to the JIT. Changes no default. |
| [#2](https://github.com/amazon-contributing/DeepEP/pull/2) | Add `kMinTokensPerPart` (default 15, `EP_MIN_TOKENS_PER_PART` overrides): `kNumParts = min(budget, tokens_per_channel / 15)`. |

**2 nodes / 16 ranks / 12 SM, 128 tok** (all-rank means):

| arm | dispatch | vs its own backend | combine | reduced combine |
|---|---|---|---|---|
| unpatched, type 2 | 365.1 µs | 1.00× | 175.0 µs | 192.4 µs |
| #1 + #2, type 2 | 237.1 µs | **−35.1%** | 176.3 | 193.6 |
| unpatched, type 5 | 169.4 µs | 1.00× | 162.7 | 179.0 |
| #1 + #2, type 5 | 112.7 µs | **−33.5%** | 162.4 | 178.9 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`, type 5 | **106.4 µs** | **−37.2%** | 162.2 | 178.9 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`, type 5 | 171.5 µs | clamp-off control | 162.6 | 179.0 |

**4 nodes / 32 ranks / 12 SM, 128 tok:**

| arm | dispatch | vs its own backend | combine | reduced combine |
|---|---|---|---|---|
| unpatched, type 2 | 1001.2 µs | 1.00× | 343.3 µs | 350.7 µs |
| #1 + #2, type 2 | 627.2 µs | **−37.4%** | 335.1 | 347.8 |
| unpatched, type 5 | 184.5 µs | 1.00× | 243.6 | 253.5 |
| #1 + #2, type 5 | 169.5 µs | **−8.1%** | 243.7 | 253.1 |
| #1 + #2 + `EP_NUM_SUB_PARTS=1`, type 5 | **155.9 µs** | **−15.5%** | 243.9 | 251.7 |
| #1 + #2 + `EP_MIN_TOKENS_PER_PART=1`, type 5 | 184.1 µs | clamp-off control | 243.3 | 253.1 |

Three things to read off these:

1. **The two levers are independent and they stack.** Stacked, 2-node decode dispatch goes
   365.1 → 106.4 µs — **3.43×** — from one env pair and two commits.
2. **combine and reduced combine are untouched to within 1% in every row.** That is the
   evidence the PR mechanism is part geometry inside dispatch and not the network: EFA is
   doing the same work either way. Expanded and cached dispatch track plain dispatch.
3. **The clamp's win collapses at 4 nodes on type 5** — −8.1% instead of −33.5% — and the
   clamp-off control (184.1 µs) sits inside noise of unpatched (184.5 µs), which confirms
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
prefill (2N/24 SM/8192: 1535.7 µs unpatched vs 1536.0 µs; 4N/12 SM/8192: 3955.3 vs 3955.0).
**Publishing decode / small-token numbers → cherry-pick both**, and set the type-5 env pair
regardless. #1 alone changes no default, so it only pays stacked on #2. After they merge,
#2's default of 15 applies automatically — no env var needed. To get the unclamped geometry
back as a control use `EP_MIN_TOKENS_PER_PART=1`, which **short-circuits** to the old value
(dividing by 1 is a *third* geometry, not a control).

**With the PRs applied, 12 SM beats 24 SM for 2-node decode dispatch** (112.7 vs 145.3 µs),
so the 24-SM recommendation above holds for unpatched code. At 4 nodes the two tie on
dispatch + reduced combine (422.6 vs 422.4 µs).

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

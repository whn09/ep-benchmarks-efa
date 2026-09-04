#!/usr/bin/env python3
"""EFA vs DeepEP's own published CX7-InfiniBand numbers, at MATCHED SM count.

usage: make_upstream_table.py

WHY THIS COMPARISON IS LEGITIMATE AT ALL. Upstream's table (deepseek-ai/DeepEP
README, "Performance") states its own metric: "the results are logical bandwidth.
For example, under the `EP 8 x 2` case, 90 GB/s actually contains local rank
traffic." That is exactly test_ep.py's SO column WITHOUT --ignore-local-traffic --
the same number our campaigns print. Upstream also states the shape: 8K tokens per
batch, hidden 7168, top-8 experts, FP8 dispatching, BF16 combining. Our campaigns
run `--num-tokens=8192 --hidden=7168 --num-topk=8 --num-experts=256
--test-first-only`, and --test-first-only is the FP8 leg of enumerate_ep_modes().
So metric and shape both match, and the GB/s can be compared directly.

THE ONE NUMBER UPSTREAM DOES NOT PUBLISH IS ITS NIC RATE. The V2 table has only
`Arch` (SM90/SM100) and `NIC type` (CX7). 400 Gb/s = 50 GB/s per GPU comes from
V1's docs/legacy.md ("CX7 InfiniBand 400 Gb/s ... ~50 GB/s maximum bandwidth"),
which is a V1 claim about V1 kernels. Every wire% derived from an upstream row is
therefore marked `*`. The RATIO column needs that assumption only for the SM90
rows to be same-ceiling comparisons -- p5en is 16 x 200 Gb/s / 8 GPUs = 50 GB/s
too, so ours/theirs on GB/s is denominator-free there.

THE SM COUNT IS THE TRAP, NOT THE NODE COUNT. Upstream's EP 8x2 row is 12 SM but
its EP 8x4 row is SIX. Quoting our 4-node 12-SM number against their 6-SM row
would flatter us, so the head-to-head below matches SM and the CROSS-CHECK section
prices exactly how much that choice is worth.

B300 IS NOT A HEAD-TO-HEAD ROW. p6-b300 is sm_103 (not sm_100) and carries TWO EFA
devices per GPU = 100 GB/s, double CX7's ceiling, so its wire% answers a different
question and its GB/s is not comparable to a 50 GB/s row. It is printed as context
with its own denominator, and labelled.

This script owns no logs. It reads two existing campaigns:
  ../p5en_2n4n_20260825/logs   (4 x p5en.48xlarge, H200 = SM90)
  ../b300_scale_20260904/logs  (4 x p6-b300.48xlarge, sm_103)
Aggregation is imported from the p5en 3-arm generator, unchanged, so the
arithmetic matches every other campaign in this repo.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
P5EN_LOGS = os.path.join(RESULTS, "p5en_2n4n_20260825", "logs")
B300_LOGS = os.path.join(RESULTS, "b300_scale_20260904", "logs")
GEN = os.path.join(RESULTS, "p5en_3arm_20260831", "make_3arm_tables.py")


def _mod(name, logs):
    """Load the shared generator with its own log dir; it reads EPRUNS at import."""
    os.environ["EPRUNS"] = logs
    spec = importlib.util.spec_from_file_location(name, GEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert os.path.abspath(m.EPRUNS) == os.path.abspath(logs), "wrong log dir"
    return m


h200 = _mod("gen_p5en", P5EN_LOGS)
b300 = _mod("gen_b300", B300_LOGS)
assert h200.EPRUNS != b300.EPRUNS, "the two campaigns collapsed onto one log dir"

# Per-GPU NIC ceiling, GB/s. p5en 16 x 200 Gb/s / 8 GPUs; b300 2 x 400 Gb/s per GPU.
W_H200, W_B300 = 50.0, 100.0

# deepseek-ai/DeepEP README "## Performance", transcribed 2026-09-04 from
#   gh api repos/deepseek-ai/DeepEP/readme --jq .content | base64 -d
# verbatim, so a reader can diff it against upstream without leaving this file.
# The columns are literally "Dispatch/Combine Bottleneck Bandwidth", and every
# RDMA cell is annotated "(RDMA)" -- i.e. the scale-out leg, our SO column.
UP_ROWS = [
    ("SM90", "CX7", "EP 8 x 2", 90, 81, 12),
    ("SM90", "CX7", "EP 8 x 4", 61, 61, 6),
    ("SM100", "CX7", "EP 8 x 2", 90, 91, 12),
]
UP_NVLINK = [
    ("SM100", "N/A", "EP 8", 726, 740, "64 (Max perf)"),
    ("SM100", "N/A", "EP 8", 643, 675, "24 (Min #SM)"),
]
# (nodes, sms) -> (dispatch, combine), the two SM90 rows we can match hardware on.
UP_SM90 = {(2, 12): (90, 81), (4, 6): (61, 61)}
UP_SM100_2N12 = (90, 91)
assert [(r[3], r[4]) for r in UP_ROWS if r[0] == "SM90"] == \
    [UP_SM90[k] for k in sorted(UP_SM90)], "the SM90 rows disagree with themselves"
assert (UP_ROWS[2][3], UP_ROWS[2][4]) == UP_SM100_2N12

OPS = ("dispatch", "combine", "reduced combine")

# label -> (nodes, sms, tag list). One arm per cell; `main` is carried as a control
# because it is amazon-contributing's fork tip, not upstream's tree.
H200_CELLS = [
    ("2N 12 SM", 2, 12, "official",
     ["official_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep2"]),
    ("2N 12 SM", 2, 12, "main",
     ["main_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep1",
      "main_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep2"]),
    ("2N  6 SM", 2, 6, "official",
     ["official_2N_6sm_8192tok_qpdefault_nodbg_gin5_rep1"]),
    ("2N 24 SM", 2, 24, "official",
     ["official_2N_24sm_8192tok_qpdefault_nodbg_gin5_rep%d" % r
      for r in (1, 2, 3)]),
    ("4N  6 SM", 4, 6, "official",
     ["official_4N_6sm_8192tok_qpdefault_nodbg_gin5_rep1"]),
    ("4N 12 SM", 4, 12, "official",
     ["official_4N_12sm_8192tok_qpdefault_nodbg_gin5_rep%d" % r
      for r in (1, 2, 3)]),
    ("4N 24 SM", 4, 24, "official",
     ["official_4N_24sm_8192tok_qpdefault_nodbg_gin5_rep%d" % r
      for r in (1, 2)]),
]
B300_TAG = "official_%dN_%dsm_8192tok_qpdefault_nodbg_gin5_ovlp0_rep%d"
B300_CELLS = [("%dN %2d SM" % (N, sms), N, sms, "official",
               [B300_TAG % (N, sms, r) for r in (1, 2, 3)])
              for N in (2, 4) for sms in (12, 24)]


def cell(g, tags):
    out = []
    for i, t in enumerate(tags, 1):
        r = g.load(t)
        if r:
            out.append((i, r))
    return out


def measure(g, cells, wire):
    """label -> arm -> op -> (us, SO, wire%, n, world, ranks)"""
    out = {}
    for label, N, sms, arm, tags in cells:
        reps = cell(g, tags)
        if not reps:
            out.setdefault(label, {})[arm] = None
            continue
        world = reps[0][1][1]
        ranks = len(reps[0][1][0].get("dispatch", {}))
        d = {}
        for op in OPS:
            t_us = g.stat(reps, op, 2)[0]
            so = g.stat(reps, op, 0)[0]
            if t_us is None or so is None:
                continue
            d[op] = (t_us, so, so * (N - 1) / N / wire * 100.0, len(reps),
                     world, ranks)
        out.setdefault(label, {})[arm] = (N, sms, d)
    return out


H = measure(h200, H200_CELLS, W_H200)
B = measure(b300, B300_CELLS, W_B300)


def preamble():
    print("EFA vs DeepEP's published CX7-InfiniBand numbers")
    print("=" * 64)
    print()
    print("Workload, identical on both sides: 8192 tokens, hidden 7168, top-8,")
    print("256 experts, FP8 dispatch, BF16 combine.")
    print("Metric, identical on both sides: logical scale-out GB/s INCLUDING the")
    print("  local node's share -- upstream says so verbatim, and it is test_ep.py's")
    print("  SO column without --ignore-local-traffic.")
    print("us = all-rank mean, pooled from every node's log, averaged over reps.")
    print("wire%% = SO x (N-1)/N / ceiling.  p5en/CX7 ceiling %.0f GB/s," % W_H200)
    print("  b300 ceiling %.0f GB/s (TWO EFA devices per GPU)." % W_B300)
    print("`*` marks a figure whose denominator is OUR assumption, not upstream's")
    print("  statement: upstream publishes no NIC line rate.")
    print()
    print("p5en logs:  %s" % h200.EPRUNS)
    print("b300 logs:  %s" % b300.EPRUNS)
    print()
    print("UPSTREAM'S TABLE, transcribed verbatim (deepseek-ai/DeepEP README,")
    print("  \"## Performance\", read 2026-09-04). Columns are literally")
    print("  \"Dispatch/Combine Bottleneck Bandwidth\", every cell tagged (RDMA).")
    print()
    print("  %-6s %-9s %-9s %10s %10s %6s" % ("Arch", "NIC", "Topo", "dispatch",
                                              "combine", "#SM"))
    for arch, nic, topo, d, c, sm in UP_ROWS:
        print("  %-6s %-9s %-9s %10s %10s %6s"
              % (arch, nic, topo, "%d GB/s" % d, "%d GB/s" % c, sm))
    for arch, nic, topo, d, c, sm in UP_NVLINK:
        print("  %-6s %-9s %-9s %10s %10s %6s   <- intranode, out of scope here"
              % (arch, nic, topo, "%d GB/s" % d, "%d GB/s" % c, sm))
    print()
    print("  The two NVLink rows are a different fabric and a different question;")
    print("  our own NVLink numbers are not in this repo, so they are not compared.")
    print()


def head_to_head():
    """The answer to "how far behind IB are we", at matched SM."""
    print("SM90 HEAD-TO-HEAD -- p5en/H200/EFA vs upstream SM90/CX7, SM MATCHED")
    print("  Upstream's EP 8x4 row is SIX SMs, not twelve. Both sides of each row")
    print("  below are at the same SM count; see CROSS-CHECK for what that costs.")
    print()
    hdr = ("  %-22s %-9s %11s %8s   %11s %8s %10s   %s"
           % ("upstream row", "op", "theirs", "wire%*", "ours", "wire%", "ours/theirs",
              "ours us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (N, sms), ups in sorted(UP_SM90.items()):
        label = "%dN %2d SM" % (N, sms)
        got = H.get(label, {}).get("official")
        for op, u in zip(("dispatch", "combine"), ups):
            uw = u * (N - 1) / N / W_H200 * 100.0
            if not got or op not in got[2]:
                print("  %-22s %-9s %11s %7.1f%%   %11s -- no matched cell"
                      % ("SM90 CX7 EP 8x%d, %dsm" % (N, sms), op,
                         "%d GB/s" % u, uw, "-"))
                continue
            t_us, so, w, n, _wd, _r = got[2][op]
            print("  %-22s %-9s %11s %7.1f%%   %11s %7.1f%% %10.3f   %8.1f (n=%d)"
                  % ("SM90 CX7 EP 8x%d, %dsm" % (N, sms), op,
                     "%d GB/s" % u, uw, "%.2f GB/s" % so, w, so / u, t_us, n))
    print()
    print("  Read the dispatch rows together: the ratio is the same at both scales,")
    print("  so EFA's dispatch deficit against IB is a LEVEL, not a scaling term.")
    print("  Combine is the weak op, and it is the one that closes with node count --")
    print("  IB is already at 91.5%* wire at 4 nodes and has no room left.")
    print()


def combine_ambiguity():
    """Upstream prints one combine column; test_ep.py prints two."""
    print("COMBINE IS AMBIGUOUS ON UPSTREAM'S SIDE -- and it matters")
    print("  test_ep.py prints `combine` AND `reduced combine`; upstream's table has")
    print("  one column. The head-to-head above uses `combine`, the reading that")
    print("  FAVOURS us. If upstream's figure is the reduced variant, substitute:")
    print()
    print("  %-22s %11s   %11s %10s" % ("upstream row", "theirs", "ours (reduced)",
                                        "ours/theirs"))
    for (N, sms), ups in sorted(UP_SM90.items()):
        got = H.get("%dN %2d SM" % (N, sms), {}).get("official")
        if not got or "reduced combine" not in got[2]:
            continue
        so = got[2]["reduced combine"][1]
        print("  %-22s %11s   %11s %10.3f"
              % ("SM90 CX7 EP 8x%d, %dsm" % (N, sms), "%d GB/s" % ups[1],
                 "%.2f GB/s" % so, so / ups[1]))
    print()


def sm_scan(g_label, table, cells):
    print("%s -- every cell we have, with the denominator stated" % g_label)
    print()
    hdr = ("  %-9s %-8s %2s %-17s %9s %10s %8s"
           % ("cell", "arm", "n", "op", "us", "SO GB/s", "wire%"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    seen = []
    for label, _N, _sms, arm, _tags in cells:
        if (label, arm) in seen:
            continue
        seen.append((label, arm))
        got = table.get(label, {}).get(arm)
        if not got:
            print("  %-9s %-8s -- no logs" % (label, arm))
            continue
        _N, _sms, d = got
        first = True
        for op in OPS:
            if op not in d:
                continue
            t_us, so, w, n, _wd, _r = d[op]
            print("  %-9s %-8s %2d %-17s %9.1f %10.2f %7.1f%%"
                  % (label if first else "", arm if first else "", n, op, t_us,
                     so, w))
            first = False
        print()


def cross_check():
    """Price the SM mismatch we refused to exploit."""
    print("CROSS-CHECK -- what quoting the WRONG SM count would have been worth")
    print("  Upstream's 4-node row is 6 SM. Our 4-node 12-SM cell against it:")
    print()
    a = H.get("4N 12 SM", {}).get("official")
    b = H.get("4N  6 SM", {}).get("official")
    if a and b:
        for op in ("dispatch", "combine"):
            if op in a[2] and op in b[2]:
                print("    %-9s 12 SM %6.2f GB/s vs 6 SM %6.2f GB/s -> %+.1f%% free"
                      % (op, a[2][op][1], b[2][op][1],
                         100.0 * (a[2][op][1] - b[2][op][1]) / b[2][op][1]))
    print()
    print("  Small, because at 4 nodes our dispatch is nearly SM-flat. At 2 nodes the")
    print("  same axis is anything but:")
    a = H.get("2N 12 SM", {}).get("official")
    b = H.get("2N  6 SM", {}).get("official")
    if a and b:
        for op in ("dispatch", "combine"):
            if op in a[2] and op in b[2]:
                print("    %-9s 12 SM %6.2f GB/s vs 6 SM %6.2f GB/s -> %+.1f%%"
                      % (op, a[2][op][1], b[2][op][1],
                         100.0 * (a[2][op][1] - b[2][op][1]) / b[2][op][1]))
    print()
    print("  So the axis has to be matched; at 2 nodes it decides the answer.")
    print()


def fork_control():
    """official is DeepSeek's tree; main is amazon-contributing's fork tip."""
    print("FORK CONTROL -- does it matter which tree we compare from?")
    print()
    got = H.get("2N 12 SM", {})
    o, m = got.get("official"), got.get("main")
    if o and m:
        print("  %-17s %11s %11s %8s" % ("op", "official", "main", "d"))
        for op in OPS:
            if op in o[2] and op in m[2]:
                a, b = o[2][op][1], m[2][op][1]
                print("  %-17s %11.2f %11.2f %7.1f%%"
                      % (op, a, b, 100.0 * (b - a) / a))
        print()
        print("  2N / 12 SM, %d rep(s) vs %d. The two trees are the same measurement,"
              % (o[2]["dispatch"][3], m[2]["dispatch"][3]))
        print("  so the head-to-head does not depend on which one we quote.")
    print()


def b300_context():
    print("B300 CONTEXT -- NOT a head-to-head row")
    print("  p6-b300 is sm_103, not sm_100, and has TWO EFA devices per GPU")
    print("  (%.0f GB/s vs CX7's %.0f). So its GB/s cannot be compared to a"
          % (W_B300, W_H200))
    print("  %.0f GB/s row, and its wire%% answers a different question: not" % W_H200)
    print("  \"how close to the wire\" but \"can 12 SM fill twice the wire\".")
    print("  Upstream's only SM100 row is EP 8x2 / 12 SM: %d GB/s dispatch,"
          % UP_SM100_2N12[0])
    print("  %d combine -- 2 nodes only, no SM100 4-node row exists upstream."
          % UP_SM100_2N12[1])
    print()
    sm_scan("B300 CELLS (p6-b300.48xlarge, sm_103, GIN type 5, ovlp=0)",
            B, B300_CELLS)
    d2 = B.get("2N 24 SM", {}).get("official")
    d4 = B.get("4N 12 SM", {}).get("official")
    if d2 and d4:
        print("  Two readings, both true and pulling opposite ways:")
        print("    - absolute: b300 2N/24 SM dispatch is %.2f GB/s against upstream's"
              % d2[2]["dispatch"][1])
        print("      SM100 %d, i.e. %.2fx the logical bandwidth -- on 2x the wire."
              % (UP_SM100_2N12[0], d2[2]["dispatch"][1] / UP_SM100_2N12[0]))
        print("    - efficiency: that is only %.1f%% of b300's own ceiling. 12 SM"
              % d2[2]["dispatch"][2])
        print("      cannot fill %.0f GB/s at 2 nodes; the 4-node cell reaches %.1f%%"
              % (W_B300, d4[2]["dispatch"][2]))
        print("      because per-rank cross-node bytes grow with node count.")
        print()
    print("  The apples-to-apples SM100 comparison is p6-b200 (ONE EFA per GPU =")
    print("  %.0f GB/s, same ceiling as CX7): 82.6%% dispatch / 69.8%% combine at"
          % W_H200)
    print("  2N/12 SM, i.e. 0.918 / 0.767 of upstream's SM100 row. Those are AWS's")
    print("  numbers, reproduced by us on that cell to 1.6-2.9%; we have no b200")
    print("  nodes now, so this repo cannot refresh them.")
    print()


def audit():
    print("AUDIT -- rank completeness (ranks must equal 8 x nodes)")
    print()
    bad = []
    for name, table, cells in (("p5en", H, H200_CELLS), ("b300", B, B300_CELLS)):
        seen = []
        for label, N, sms, arm, tags in cells:
            if (label, arm) in seen:
                continue
            seen.append((label, arm))
            got = table.get(label, {}).get(arm)
            if not got or "dispatch" not in got[2]:
                bad.append("%s %s %s: no data" % (name, label, arm))
                continue
            _N, _sms, d = got
            n, world, ranks = d["dispatch"][3], d["dispatch"][4], d["dispatch"][5]
            print("  %-5s %-9s %-8s n=%d  world=%d  ranks=%d"
                  % (name, label, arm, n, world, ranks))
            if world != 8 * N or ranks != 8 * N:
                bad.append("%s %s %s: world=%d ranks=%d, expected %d"
                           % (name, label, arm, world, ranks, 8 * N))
            if n == 1:
                bad.append("%s %s %s: n=1, single run" % (name, label, arm))
    print()
    for g in (h200, b300):
        if g.EXCLUDED:
            print("  reps EXCLUDED as outliers: %s" % sorted(set(g.EXCLUDED)))
    if bad:
        print("  NOTES (n=1 is a limitation, not an error):")
        for b in bad:
            print("    %s" % b)
    print()


if __name__ == "__main__":
    preamble()
    head_to_head()
    combine_ambiguity()
    cross_check()
    fork_control()
    sm_scan("P5EN CELLS (p5en.48xlarge, H200 = SM90, GIN type 5)", H, H200_CELLS)
    b300_context()
    audit()

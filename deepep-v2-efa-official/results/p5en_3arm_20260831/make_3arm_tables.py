#!/usr/bin/env python3
"""Regenerate every table in comparison.md straight from the per-node logs.

usage: [EPRUNS=./logs] make_3arm_tables.py

Three trees of amazon-contributing/DeepEP, one hardware setup (4x p5en.48xlarge,
H200 sm_90, EFA installer 1.50.0, efa.ko 3.3.0g), one measurement path:

  main   54fffeff810723f574c574b1790dff189f3c6ffb   `main` as of 2026-08-31
  pr12   bfbdd15ff448783f877cb2210cb3246c8452b05e   PRs #1 + #2 (#1 subset of #2)
  pr89   3c737dcf0da5889ba7efd26e05b4808307cc38af   PRs #8 + #9 (#8 subset of #9)

Nothing here is hand-typed. Every number in comparison.md is this script's
output; hand-transcribed benchmark numbers have survived review before.

Aggregation, stated once:
  * a run's value for an op = mean over ALL ranks (16 at 2N, 32 at 4N), pooled
    from every node's log. combine is layered BY NODE -- the per-node means differ
    by 12-23% and which node is slow flips between runs -- so one node's log is
    not a sample of the run, it is a sample of one layer. The NODE LAYERING table
    prints those layers so the pooling is auditable rather than asserted.
  * an arm's value = mean over that arm's reps of the per-run all-rank means. The
    rep values are printed next to it so a reader sees the spread instead of
    trusting a single mean. Reps were ROTATED (every cell once per rep), so a slow
    thermal/network drift cannot masquerade as an arm effect.
  * SO / SU are printed as min-max ACROSS RANKS. That is a different statistic
    from the us column and is dominated by the per-rank byte denominator plus
    1-GB/s integer print quantization, not by time.
  * TIME IS THE PRIMARY METRIC. GB/s is printed beside it, never instead of it,
    because the byte denominator differs per rank and per scale.

Denominator, stated once: --ignore-local-traffic is OFF, matching every number
already published under results/ and in docs/runbook_zh.md. So SO INCLUDES
intra-node traffic and is not a wire rate -- at 2N it reads 81 GB/s against a
50 GB/s per-GPU scale-out ceiling, i.e. 162%, which is why no wire-utilisation
column is printed here. Comparing three arms on one machine does not need one.
What is printed instead is MB/rank, the byte denominator itself, so that a GB/s
that moved can be attributed to bytes or to time rather than guessed at.

Parsing note: use finditer, not search. Ranks writing concurrently to one stdout
regularly glue two per-rank lines onto a single physical line, and search() keeps
only the first -- silently halving the sample. A short n= in the AUDIT means real
data loss, not a slow rank.
"""
import glob
import os
import re
import statistics as st

LINE = re.compile(
    r"EP:\s+(\d+)/(\d+)\s*\|\s*([a-z ]+?):\s*(\d+) GB/s \(SO\), (\d+) GB/s \(SU\), "
    r"([\d.]+) us, (\d+) bytes"
)
CFG = re.compile(r"#SM:\s*(\d+),\s*#QPs:\s*(\d+)/(\d+)")
REF = re.compile(r"DeepEP=([0-9a-f]{40})")

EPRUNS = os.environ.get("EPRUNS", os.path.join(os.path.dirname(__file__), "logs"))
OPS = ["dispatch", "expanded dispatch", "cached dispatch", "combine", "reduced combine"]

# arm order is fixed: main is the baseline every delta is taken against.
ARMS = [("main54fffef", "main", "54fffef"),
        ("pr12bfbdd15", "PR #1+#2", "bfbdd15"),
        ("pr893c737dc", "PR #8+#9", "3c737dc")]
TAG = "%s_%dN_12sm_%dtok_%s_nodbg_gin5_rep%d"

# The knob axis. `qpdefault` is each tree's own default part geometry; `subparts1`
# is EP_NUM_SUB_PARTS=1. That env var only exists as a lever on the #1+#2 tree --
# PR #1 is what forwards it to the JIT, and neither `main` nor #8+#9 carries that
# forwarding (grep EP_NUM_SUB_PARTS csrc/jit/compiler.hpp inside each image: only
# bfbdd15 matches). On the other two arms it is read by nothing, so their
# subparts1 cells are an INERTNESS CONTROL: they must land on their own qpdefault
# number, and if they do not, the difference came from the environment.
DEFAULT_KNOB = "qpdefault"
SUBPARTS1 = "subparts1"


# ---------------------------------------------------------------- loading
def load(tag):
    """-> ({op: {rank: (so,su,us,bytes)}}, world, {node:{op:[us]}}, cfg, ref, tag)"""
    ops, world, bynode, cfg, ref = {}, set(), {}, set(), set()
    files = sorted(glob.glob("%s/%s.node*.log" % (EPRUNS, tag)))
    if not files:
        return None
    for path in files:
        node = re.search(r"\.node(\d+)\.log$", path).group(1)
        for line in open(path, errors="replace"):
            for m in CFG.finditer(line):
                cfg.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            for m in REF.finditer(line):
                ref.add(m.group(1))
            for m in LINE.finditer(line):
                rank, nranks, op, so, su, us, nbytes = m.groups()
                op = op.strip()
                world.add(int(nranks))
                ops.setdefault(op, {})[int(rank)] = (
                    int(so), int(su), float(us), int(nbytes))
                bynode.setdefault(node, {}).setdefault(op, []).append(float(us))
    if not world:
        # Logs exist but carry no EP lines. That is an in-flight run (the file is
        # being written right now) or one that died before the first iteration --
        # either way it is not data. Dropping it silently would let a half-finished
        # campaign print a confident table, so record it for the AUDIT section.
        EMPTY.append("%s (%d log%s, no EP lines)"
                     % (tag, len(files), "" if len(files) == 1 else "s"))
        return None
    return ops, max(world), bynode, sorted(cfg), sorted(ref), tag


EMPTY = []


CACHE = {}


def cell(arm, nodes, tok, knob=DEFAULT_KNOB):
    """every rep that exists on disk for this (arm, nodes, tokens, knob)"""
    key = (arm, nodes, tok, knob)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = load(TAG % (arm, nodes, tok, knob, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    return CACHE[key]


OUTLIER = 0.25   # fraction of the median beyond which a rep is not the same thing
EXCLUDED = []


def stat(reps, op, field=2):
    """(mean over reps of the all-rank mean, [kept per-rep means], [notes])

    One rep of pr893c737dc_2N_12sm_128tok_subparts1 came back ~4x slow on ALL 16
    ranks of BOTH nodes, at a byte-identical docker invocation (same image, same
    env, same #SM/#QPs, same GIN line) -- an environmental transient, not an arm
    effect. Averaging it in would have moved that cell's dispatch from 123 to
    247 us and reversed the arm's sign.

    So a rep further than OUTLIER from the cell's MEDIAN is excluded from the mean
    and recorded in EXCLUDED, which the AUDIT section prints. Median, not mean, so
    one bad rep cannot drag the threshold to cover itself; and excluded loudly
    rather than dropped, because a silent filter is how a real regression gets
    thrown away as noise. Three reps is the floor for this to mean anything -- at
    n=2 there is no median to appeal to, so nothing is ever excluded.
    """
    per_rep, short = [], []
    where = reps[0][1][5].rsplit("_rep", 1)[0] if reps else "?"
    for rep, (ops, W, _bn, _c, _r, _t) in reps:
        if op not in ops:
            continue
        vals = [v[field] for v in ops[op].values()]
        per_rep.append((rep, st.mean(vals)))
        if len(vals) != W:
            short.append("rep%d n=%d/%d" % (rep, len(vals), W))
    if not per_rep:
        return None, [], short
    kept = per_rep
    if len(per_rep) >= 3:
        med = st.median([v for _r, v in per_rep])
        kept = [(r, v) for r, v in per_rep if abs(v - med) <= OUTLIER * med]
        for r, v in per_rep:
            if (r, v) not in kept:
                EXCLUDED.append("%s rep%d %s = %.1f vs median %.1f (%+.0f%%)"
                                % (where, r, op, v, med, 100.0 * (v - med) / med))
                short.append("rep%d EXCLUDED %.1f (%+.0f%% off median)"
                             % (r, v, 100.0 * (v - med) / med))
        if not kept:                      # every rep an outlier: keep them all
            kept = per_rep
    return st.mean([v for _r, v in kept]), [v for _r, v in kept], short


def rng(reps, op, field):
    xs = [v[field] for _rep, (ops, _W, _bn, _c, _r, _t) in reps
          if op in ops for v in ops[op].values()]
    return (min(xs), max(xs)) if xs else (None, None)


def f(x, nd=1):
    return "-" if x is None else "%.*f" % (nd, x)


def pct(new, base):
    if new is None or base is None:
        return "-"
    return "%+.1f%%" % (100.0 * (new - base) / base)


# ---------------------------------------------------------------- tables
def config_table():
    print("CONFIG -- what each arm actually ran (read out of the logs, not assumed)")
    print()
    print("  %-9s %-8s %-42s %-8s %-8s" % ("arm", "tag", "BUILD_REF", "#SM", "#QPs"))
    for arm, label, sha in ARMS:
        refs, cfgs = set(), set()
        for nodes in (2, 4):
            for tok in (8192, 128):
                for _rep, (_o, _W, _bn, c, r, _t) in cell(arm, nodes, tok):
                    refs |= set(r)
                    cfgs |= set(c)
        sms = ",".join(sorted({str(c[0]) for c in cfgs})) or "-"
        qps = ",".join(sorted({"%d/%d" % (c[1], c[2]) for c in cfgs})) or "-"
        print("  %-9s %-8s %-42s %-8s %-8s"
              % (label, sha, ",".join(sorted(refs)) or "-", sms, qps))
    print()
    print("  #QPs is the GIN context count. PR #9 raises kDefaultGinContextCnt 11 -> 13,")
    print("  so a differing #QPs here is the patch landing, not a misconfiguration.")
    print("  #SM 12 on every arm: PR #9 also removes the num_channels_per_sm <= 4 clamp,")
    print("  which changes CHANNELS at a fixed SM count and does not show up in #SM.")
    print()


def perf_table(nodes, tok, what):
    print("%s -- %d nodes / %d ranks, %d tok, 12 SM, GIN type 5"
          % (what, nodes, nodes * 8, tok))
    print("  time is the metric. SO GB/s is min-max across ranks (a different statistic")
    print("  from the us column) and MB/rank is its denominator, so a GB/s that moved can")
    print("  be attributed. --ignore-local-traffic OFF, so SO is not a wire rate.")
    print()
    hdr = ("  %-18s %-9s %10s %9s %-11s %-13s  %s"
           % ("op", "arm", "us", "d vs main", "SO GB/s", "MB/rank", "per-rep us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        base = None
        for arm, label, _sha in ARMS:
            reps = cell(arm, nodes, tok)
            mean, per_rep, short = stat(reps, op)
            if mean is None:
                continue
            if label == "main":
                base = mean
            so_lo, so_hi = rng(reps, op, 0)
            b_lo, b_hi = rng(reps, op, 3)
            print("  %-18s %-9s %10s %9s %-11s %-13s  %s%s"
                  % (op if label == "main" else "",
                     label, f(mean), pct(mean, base) if label != "main" else "",
                     "-" if so_lo is None else "%d-%d" % (so_lo, so_hi),
                     "-" if b_lo is None else "%.1f-%.1f" % (b_lo / 1e6, b_hi / 1e6),
                     " / ".join(f(v) for v in per_rep),
                     ("  !! " + "; ".join(short)) if short else ""))
        print()


def layer_total(nodes, tok):
    """dispatch + reduced combine: the two calls one MoE layer actually makes."""
    print("LAYER TOTAL -- dispatch + reduced combine, %d nodes, %d tok" % (nodes, tok))
    print("  the pair a MoE layer actually issues, so an arm that moves cost from one")
    print("  op to the other shows up here and nowhere else.")
    print()
    print("  %-9s %10s %10s %10s %9s" % ("arm", "dispatch", "redComb", "total", "d vs main"))
    base = None
    for arm, label, _sha in ARMS:
        reps = cell(arm, nodes, tok)
        d, _, _ = stat(reps, "dispatch")
        c, _, _ = stat(reps, "reduced combine")
        if d is None or c is None:
            continue
        tot = d + c
        if label == "main":
            base = tot
        print("  %-9s %10s %10s %10s %9s"
              % (label, f(d), f(c), f(tot), pct(tot, base) if label != "main" else ""))
    print()


def node_layering(nodes, tok):
    print("NODE LAYERING -- per-node mean us, %d nodes, %d tok" % (nodes, tok))
    print("  combine splits by machine and the slow machine flips between runs. If the")
    print("  spread below is large, a table built from the leader's log alone is wrong")
    print("  by that much -- which is why every table above pools all %d ranks." % (nodes * 8))
    print()
    for op in ("dispatch", "reduced combine"):
        print("  %s:" % op)
        print("    %-9s %-6s %-28s %s" % ("arm", "rep", "per-node mean us", "max-min"))
        for arm, label, _sha in ARMS:
            for rep, (_o, _W, bynode, _c, _r, _t) in cell(arm, nodes, tok):
                per = [(n, st.mean(d[op])) for n, d in sorted(bynode.items()) if op in d]
                if not per:
                    continue
                vals = [v for _n, v in per]
                print("    %-9s %-6s %-28s %.1f%%"
                      % (label, "rep%d" % rep,
                         " ".join("n%s %.0f" % (n, v) for n, v in per),
                         100.0 * (max(vals) - min(vals)) / min(vals)))
        print()


def scaling():
    print("SCALING -- 2N -> 4N, same arm, same op. Two axes, printed together:")
    print("  the time ratio (how much slower the bigger job is) and the per-rank byte")
    print("  growth that has to pay for it. A time ratio alone cannot tell a stack that")
    print("  scales badly from one that is simply moving more data.")
    print()
    for tok, what in ((8192, "prefill"), (128, "decode")):
        print("  %s (%d tok):" % (what, tok))
        print("    %-18s %-9s %10s %10s %8s %11s %11s %8s"
              % ("op", "arm", "2N us", "4N us", "4N/2N",
                 "2N MB/rank", "4N MB/rank", "4N/2N B"))
        for op in ("dispatch", "reduced combine"):
            for arm, label, _sha in ARMS:
                r2, r4 = cell(arm, 2, tok), cell(arm, 4, tok)
                a, _, _ = stat(r2, op)
                b, _, _ = stat(r4, op)
                if a is None or b is None:
                    continue
                ma, _, _ = stat(r2, op, 3)
                mb, _, _ = stat(r4, op, 3)
                print("    %-18s %-9s %10s %10s %8s %11s %11s %8s"
                      % (op if label == "main" else "", label, f(a), f(b),
                         "%.2fx" % (b / a), f(ma / 1e6), f(mb / 1e6),
                         "%.2fx" % (mb / ma)))
            print()
        print("    the byte columns are num_scaleup_bytes (the log's `bytes` field), so")
        print("    they are the SU denominator, not cross-node traffic. Cross-node bytes")
        print("    have to be recovered as SO * time and are not printed here.")
        print()


def subparts(nodes, tok):
    """EP_NUM_SUB_PARTS=1 vs each tree's own default geometry.

    PR #1's entire contribution is forwarding this env var to the JIT, so an arm
    measured only at its default geometry undersells #1+#2 by whatever this table
    shows -- and the two arms without the forwarding cannot be tuned at all.
    """
    print("EP_NUM_SUB_PARTS=1 -- %d nodes / %d ranks, %d tok, 12 SM, GIN type 5"
          % (nodes, nodes * 8, tok))
    print("  the knob only exists as a lever on #1+#2: PR #1 is the forwarding, and")
    print("  `grep EP_NUM_SUB_PARTS csrc/jit/compiler.hpp` matches ONLY in bfbdd15.")
    print("  main and #8+#9 rows are therefore an inertness control -- they must land")
    print("  on their own default, and a delta there would mean the environment moved,")
    print("  not the knob.")
    print()
    hdr = ("  %-18s %-9s %10s %10s %9s  %s"
           % ("op", "arm", "default", "subparts1", "d", "per-rep us (subparts1)"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        first = True
        for arm, label, _sha in ARMS:
            base, _, _ = stat(cell(arm, nodes, tok), op)
            reps = cell(arm, nodes, tok, SUBPARTS1)
            mean, per_rep, short = stat(reps, op)
            if mean is None:
                continue
            print("  %-18s %-9s %10s %10s %9s  %s%s"
                  % (op if first else "",
                     label, f(base), f(mean), pct(mean, base),
                     " / ".join(f(v) for v in per_rep),
                     ("  !! " + "; ".join(short)) if short else ""))
            first = False
        if not first:
            print()


def subparts_layer(nodes, tok):
    """Each arm at its OWN operating point, not at whichever default it ships.

    Both deltas are against main-at-default, because that is the thing being
    replaced. dispatch is broken out next to the total because the knob is a
    DISPATCH knob and the two columns do not move together -- at 2N/8192 it makes
    dispatch worse while the total improves, which a total-only table would hide.
    """
    print("BEST CONFIGURATION -- %d nodes, %d tok" % (nodes, tok))
    print("  arm x knob, dispatch + reduced combine (the pair a MoE layer issues).")
    print("  both d columns are vs main-at-default, the configuration being replaced.")
    print()
    print("  %-9s %-11s %10s %9s %10s %10s %9s"
          % ("arm", "knob", "dispatch", "d disp", "redComb", "total", "d total"))
    rows = []
    for arm, label, _sha in ARMS:
        for knob, kname in ((DEFAULT_KNOB, "default"), (SUBPARTS1, "subparts1")):
            reps = cell(arm, nodes, tok, knob)
            d, _, _ = stat(reps, "dispatch")
            c, _, _ = stat(reps, "reduced combine")
            if d is None or c is None:
                continue
            rows.append((label, kname, d, c, d + c))
    base_d = base_t = None
    for label, kname, d, c, tot in rows:
        if label == "main" and kname == "default":
            base_d, base_t = d, tot
    for label, kname, d, c, tot in rows:
        ref = label == "main" and kname == "default"
        print("  %-9s %-11s %10s %9s %10s %10s %9s"
              % (label, kname, f(d), "" if ref else pct(d, base_d),
                 f(c), f(tot), "" if ref else pct(tot, base_t)))
    print()


def audit():
    print("AUDIT -- reps found on disk and rank completeness per cell")
    print("  n must equal the world size. A short n is glued stdout lines, i.e. lost")
    print("  data, and it invalidates that rep's mean rather than widening it.")
    print()
    # Both knobs, because both appear in tables above. A cell that only exists at
    # one knob prints "-- not run", which is the honest state: the EP_NUM_SUB_PARTS
    # matrix is deliberately narrower than the default one (see subparts()).
    for knob in (DEFAULT_KNOB, SUBPARTS1):
        for nodes in (2, 4):
            for tok in (8192, 128):
                for arm, label, _sha in ARMS:
                    reps = cell(arm, nodes, tok, knob)
                    if not reps:
                        print("  %-9s %dN %5dtok %-9s -- not run"
                              % (knob, nodes, tok, label))
                        continue
                    ns = []
                    for rep, (ops, W, _bn, _c, _r, _t) in reps:
                        ns.append("rep%d:%d/%d" % (rep, len(ops.get("dispatch", {})), W))
                    print("  %-9s %dN %5dtok %-9s %s"
                          % (knob, nodes, tok, label, " ".join(ns)))
    print()
    if EMPTY:
        print("  logs present but EMPTY (in-flight, or died before iteration 1) --")
        print("  excluded from every table above:")
        for e in sorted(set(EMPTY)):
            print("    %s" % e)
        print()
    if EXCLUDED:
        print("  reps EXCLUDED as outliers (> %.0f%% off their cell's median) --" % (100 * OUTLIER))
        print("  the raw logs are still under logs/, so this is reversible:")
        for e in sorted(set(EXCLUDED)):
            print("    %s" % e)
        print()
    else:
        print("  no rep was excluded as an outlier.")
        print()


if __name__ == "__main__":
    print("EPRUNS=%s" % EPRUNS)
    print()
    config_table()
    perf_table(2, 8192, "PREFILL")
    perf_table(4, 8192, "PREFILL")
    perf_table(2, 128, "DECODE")
    perf_table(4, 128, "DECODE")
    layer_total(2, 8192)
    layer_total(4, 8192)
    layer_total(2, 128)
    layer_total(4, 128)
    subparts(2, 128)
    subparts(4, 128)
    subparts(2, 8192)
    subparts(4, 8192)
    subparts_layer(2, 128)
    subparts_layer(4, 128)
    subparts_layer(2, 8192)
    subparts_layer(4, 8192)
    scaling()
    node_layering(2, 8192)
    node_layering(4, 128)
    audit()

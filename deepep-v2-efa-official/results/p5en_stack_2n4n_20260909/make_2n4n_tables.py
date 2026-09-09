#!/usr/bin/env python3
"""Does the PR #1+#2+#8+#9 stack still pay at FOUR nodes on p5en?

usage: [EPRUNS_2N4N=./logs] make_2n4n_tables.py

WHY THIS CAMPAIGN EXISTS. Two arms of the question were open and neither could be
closed from what was already on disk:

1. p5en had NO #8+#9 and NO stack cell at four nodes at all. The 4-node p5en logs in
   results/p5en_2n4n_20260825 carry only `official`, `main` and #1+#2, and the #1+#2
   decode cell there is a single rep.
2. results/b300_stack5_4N_20260904 measured, at four nodes, that #8+#9 REGRESSES
   b300 decode combine (269.9 -> 292.7 us, +8.4%; reduced combine 284.3 -> 297.8,
   +4.7%) -- while at two nodes on p5en the same tree WON decode reduced combine
   (180.5 -> 151.0 us, -16.3%). The sign flips between (2 nodes, p5en) and
   (4 nodes, b300), and those two points cannot separate node count from
   architecture. This campaign's 4-node p5en cell is the point that separates them,
   and the SIGN OF #8+#9 ON DECODE COMBINE section is where that is read off.

WHY 2 NODES IS RE-RUN RATHER THAN REUSED. results/p5en_stack_20260831 already has a
2-node 4-arm table, and it is not doubted -- its cross-rep spread is <=0.5% and the
b300 twin campaign reproduced its signs. It is re-run here because a 2N->4N delta is
a difference of differences, the least forgiving thing to take across two node sets
on two different days: the old campaign ran on instances that no longer exist. So the
SCALING section is computed only from THIS campaign's logs, and the old campaign is
used as an independent reproduction check (REPRODUCTION vs 2026-08-31), never pooled.
It also closes two holes in the old table for free: #1+#2 had no 8192-tok cell (so
prefill additivity was not computable) and the stack had one rep.

WHAT IS COMPARABLE TO WHAT. Every cell: 12 SM, GIN type 5
(NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0), --prefer-overlap-with-compute=0,
--test-first-only (so FP8 dispatch at expert_alignment=128), 8 processes per node,
one arm label per tree, EP_BUFFER_DEBUG OFF. The only axes are node count, token
count, arm and the EP_NUM_SUB_PARTS knob, and all four are in every log name.

REPS=2, SO THERE IS NO WORKING OUTLIER FILTER. The imported stat() drops a rep more
than 25% off its cell's median, and a median over two replicates is their mean, so
with n=2 it can only fire on a pair that disagrees by ~2x. Every row prints its
per-rep values for that reason: read the spread, do not trust the mean alone.

TIME IS THE METRIC. --ignore-local-traffic is OFF, so the SO column counts tokens
destined for the sender's own node and is NOT a wire rate; it is printed beside the
microseconds, never instead of them, together with MB/rank so a GB/s that moved can
be attributed to bytes or to time. Aggregation -- all-rank pooling from EVERY node's
log (combine is layered by node, so one node's mean is a sample of one layer, not of
the run), then mean over reps -- is imported unchanged from the p5en 3-arm
generator, so the arithmetic matches every other campaign in this directory.

Nothing in the output is hand-typed.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
LOGS = os.environ.get("EPRUNS_2N4N", os.path.join(HERE, "logs"))
B300_4N = os.path.join(RESULTS, "b300_stack5_4N_20260904", "logs")
OLD_2N = os.path.join(RESULTS, "p5en_stack_20260831", "logs")
SHARED = os.path.join(RESULTS, "p5en_3arm_20260831", "make_3arm_tables.py")


def _mod(name, logs):
    """Load the shared aggregator as its own module bound to one log directory.

    It reads EPRUNS at import time, so three loads give three independent modules
    over three campaigns; a plain `import` would reuse the first and silently read
    the wrong logs -- which is how a cross-campaign table gets one arm from the
    wrong machine.
    """
    os.environ["EPRUNS"] = logs
    spec = importlib.util.spec_from_file_location(name, SHARED)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert os.path.abspath(m.EPRUNS) == os.path.abspath(logs), "wrong log dir"
    return m


g = _mod("gen_2n4n", LOGS)
b3 = _mod("gen_b300_4n", B300_4N)
old = _mod("gen_old_2n", OLD_2N)
assert len({g.EPRUNS, b3.EPRUNS, old.EPRUNS}) == 3, "two campaigns collapsed onto one dir"

f, pct, OPS = g.f, g.pct, g.OPS
DFLT, SUB1 = g.DEFAULT_KNOB, g.SUBPARTS1
B300_KNOB = "qpdefault-nvls0"   # b300 4N ran with NCCL_NVLS_ENABLE=0 in its knob tag

MAIN, PR12, PR89, STACK = "main54fffef", "pr12bfbdd15", "pr893c737dc", "stacka35285f"
ARMS = [(MAIN, "main"), (PR12, "PR #1+#2"), (PR89, "PR #8+#9"), (STACK, "stack")]
TUNABLE = {PR12, STACK}         # only #1+#2 forwards EP_NUM_SUB_PARTS to the JIT
NODESET = (4, 2)                # 4 first: it is the new data and the reason to run
TOKS = ((128, "DECODE"), (8192, "PREFILL"))
TAG = "%s_%dN_12sm_%dtok_%s_nodbg_gin5_ovlp0_rep%d"

CACHE = {}


def cell(mod, arm, nodes, tok, knob=DFLT, only=None):
    key = (id(mod), arm, nodes, tok, knob)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = mod.load(TAG % (arm, nodes, tok, knob, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    reps = CACHE[key]
    return reps if only is None else [x for x in reps if x[0] in only]


def us(arm, nodes, tok, op, knob=DFLT, only=None, mod=None):
    return (mod or g).stat(cell(mod or g, arm, nodes, tok, knob, only), op)[0]


def so(arm, nodes, tok, op, knob=DFLT, only=None, mod=None):
    """all-rank-mean SO GB/s. Integer per rank as printed, so it cannot resolve a
    sub-1% change in time; not a wire rate (--ignore-local-traffic is off)."""
    return (mod or g).stat(cell(mod or g, arm, nodes, tok, knob, only), op, 0)[0]


def mb(arm, nodes, tok, op, knob=DFLT, only=None, mod=None):
    v = (mod or g).stat(cell(mod or g, arm, nodes, tok, knob, only), op, 3)[0]
    return None if v is None else v / 1e6


def layer(arm, nodes, tok, knob=DFLT, only=None, mod=None):
    d = us(arm, nodes, tok, "dispatch", knob, only, mod)
    c = us(arm, nodes, tok, "reduced combine", knob, only, mod)
    return None if (d is None or c is None) else d + c


def shared_reps(nodes, tok, knob_of):
    """Reps present in EVERY compared arm, so a cross-arm delta is same-rep.

    None = some arm has no reps at all (no filter is meaningful); empty set = every
    arm has reps but none in common. Two different gaps, worded differently by the
    caller -- returning None for the second would restore the unbalanced comparison.
    """
    sets = []
    for arm, _ in ARMS:
        reps = {rep for rep, _ in cell(g, arm, nodes, tok, knob_of(arm))}
        if not reps:
            return None
        sets.append(reps)
    return set.intersection(*sets)


# ------------------------------------------------------------------ sections
def config_table():
    print("CONFIG -- read out of the logs, not assumed")
    print()
    print("  %-13s %-4s %-42s %-5s %-8s %s"
          % ("arm", "N", "BUILD_REF", "#SM", "#QPs", "world"))
    for nodes in NODESET:
        for arm, _ in ARMS:
            refs, cfgs, worlds = set(), set(), set()
            for knob in (DFLT, SUB1):
                for _rep, (_o, w, _bn, c, r, _t) in cell(g, arm, nodes, tok=128,
                                                         knob=knob):
                    refs |= set(r)
                    cfgs |= set(c)
                    worlds.add(w)
                for _rep, (_o, w, _bn, c, r, _t) in cell(g, arm, nodes, tok=8192,
                                                         knob=knob):
                    refs |= set(r)
                    cfgs |= set(c)
                    worlds.add(w)
            if not refs:
                continue
            sm = ",".join(str(c[0]) for c in sorted(cfgs))
            qp = ",".join("%d/%d" % (c[1], c[2]) for c in sorted(cfgs))
            print("  %-13s %-4s %-42s %-5s %-8s %s"
                  % (arm, "%dN" % nodes, ",".join(sorted(refs)), sm, qp,
                     ",".join(str(w) for w in sorted(worlds))))
    print()
    print("  one BUILD_REF, one #SM and one world per (arm, node count) is the")
    print("  precondition for every table below; a second value in any cell means")
    print("  two configurations were pooled under one label.")
    print()


def perf_table(nodes, tok, what):
    print("%s -- %d tok, %d nodes / %d ranks, 12 SM, GIN type 5, ovlp=0, default"
          " part geometry" % (what, tok, nodes, nodes * 8))
    print("  `d vs main` is against main54fffef, the merge base of all four PRs.")
    print()
    print("  %-18s %-10s %8s %10s  %-9s %-8s %s"
          % ("op", "arm", "us", "d vs main", "SO GB/s", "MB/rank", "per-rep us"))
    for op in OPS:
        base = us(MAIN, nodes, tok, op)
        for arm, label in ARMS:
            reps = cell(g, arm, nodes, tok)
            if not reps:
                continue
            v = us(arm, nodes, tok, op)
            if v is None:
                continue
            lo, hi = g.rng(reps, op, 0)
            per = " / ".join(f(g.stat([r], op)[0]) for r in reps)
            d = "" if arm == MAIN or base is None else pct(v, base)
            m = mb(arm, nodes, tok, op)
            print("  %-18s %-10s %8s %10s  %-9s %-8s %s"
                  % (op if arm == ARMS[0][0] else "", label, f(v), d,
                     "%d-%d" % (lo, hi) if lo is not None else "-",
                     f(m, 1) if m is not None else "-", per))
        print()
    print("  layer total (dispatch + reduced combine):")
    base = layer(MAIN, nodes, tok)
    for arm, label in ARMS:
        v = layer(arm, nodes, tok)
        if v is None:
            continue
        print("  %-18s %-10s %8s %10s"
              % ("", label, f(v), "" if arm == MAIN or base is None
                 else pct(v, base)))
    print()


def combine_sign():
    """The one table this campaign was run to produce.

    #8+#9 won decode reduced combine at 2 nodes on p5en and LOST it at 4 nodes on
    b300. Those two points confound node count with architecture. Printing all four
    (machine x node count) cells side by side is what separates them: if the p5en
    4-node cell is also positive, the flip is node count and PR #8/#9 carries a
    decode-combine regression at deployment scale on both architectures; if it stays
    negative, the flip is architecture and belongs in the b300 section instead.

    b300's cells come from results/b300_stack5_4N_20260904 (knob tag
    qpdefault-nvls0). It is a DIFFERENT machine, shape and NIC width, so only the
    SIGN and rough magnitude of the main->#8+#9 delta transfer; the absolute
    microseconds are not comparable across the two rows and are printed only so the
    percentage has a visible denominator.
    """
    print("SIGN OF #8+#9 ON DECODE COMBINE -- the question this campaign answers")
    print()
    print("  main -> #8+#9 at 128 tok, 12 SM, ovlp=0, default geometry.")
    print("  A POSITIVE delta is a regression.")
    print()
    print("  %-22s %-18s %9s %9s %9s"
          % ("machine / nodes", "op", "main", "#8+#9", "d"))
    rows = [("p5en 2N (this campaign)", g, 2), ("p5en 4N (this campaign)", g, 4),
            ("b300 4N (2026-09-04)", b3, 4)]
    for label, mod, nodes in rows:
        knob = B300_KNOB if mod is b3 else DFLT
        shown = False
        for op in ("combine", "reduced combine"):
            m = us(MAIN, nodes, 128, op, knob, mod=mod)
            p = us(PR89, nodes, 128, op, knob, mod=mod)
            if m is None or p is None:
                print("  %-22s %-18s -- not measured"
                      % ("" if shown else label, op))
                shown = True
                continue
            print("  %-22s %-18s %9s %9s %9s"
                  % ("" if shown else label, op, f(m), f(p), pct(p, m)))
            shown = True
        print()


def scaling(tok, what):
    """Each arm 2N -> 4N, and whether each PR's win changes with node count.

    The second block is the difference of differences, and it is the reason both node
    counts are in ONE campaign on ONE set of hosts: taking the 2-node leg from an
    older campaign would fold a node-set change and a day change into the residual.
    """
    print("SCALING 2N -> 4N -- %s (%d tok), default geometry" % (what, tok))
    print()
    print("  %-18s %-10s %9s %9s %8s"
          % ("op", "arm", "2N us", "4N us", "4N/2N"))
    for op in OPS + ["layer total"]:
        for arm, label in ARMS:
            def val(nodes, a=arm, o=op):
                return (layer(a, nodes, tok) if o == "layer total"
                        else us(a, nodes, tok, o))
            a2, a4 = val(2), val(4)
            if a2 is None or a4 is None:
                continue
            print("  %-18s %-10s %9s %9s %8s"
                  % (op if arm == ARMS[0][0] else "", label, f(a2), f(a4),
                     "%.2fx" % (a4 / a2) if a2 else "-"))
        print()
    print("  each PR's win against main, at each node count:")
    print()
    print("  %-18s %-10s %10s %10s %9s"
          % ("op", "arm", "d @2N", "d @4N", "erosion"))
    for op in OPS + ["layer total"]:
        for arm, label in ARMS:
            if arm == MAIN:
                continue

            def d(nodes, a=arm, o=op):
                base = (layer(MAIN, nodes, tok) if o == "layer total"
                        else us(MAIN, nodes, tok, o))
                v = (layer(a, nodes, tok) if o == "layer total"
                     else us(a, nodes, tok, o))
                return None if (base is None or v is None or not base) \
                    else 100.0 * (v - base) / base
            d2, d4 = d(2), d(4)
            if d2 is None or d4 is None:
                continue
            print("  %-18s %-10s %+9.1f%% %+9.1f%% %+8.1f pp"
                  % (op if arm == ARMS[1][0] else "", label, d2, d4, d4 - d2))
        print()


def additivity(nodes, tok, what):
    """measured stack vs the sum of the two arms' separate savings.

    expected = main + (pr12 - main) + (pr89 - main): the two savings simply add. A
    stack SLOWER than that means the two PRs partly attack the same cost; FASTER
    means they compound. The residual is printed in us and as a share of the LARGER
    single win, because 10 us means something different next to a 12 us saving than
    next to a 500 us one.
    """
    print("ADDITIVITY -- %s (%d tok), %d nodes" % (what, tok, nodes))
    print()
    print("  %-18s %-10s %9s %9s %9s %9s %9s %8s"
          % ("op", "knob", "main", "#1+#2", "#8+#9", "expected", "stack",
             "residual"))
    for op in OPS + ["layer total"]:
        for knob in (DFLT, SUB1):
            # main and #8+#9 have no tuned variant; at the tuned knob they are
            # represented by their own default, which is what they would ship as.
            def knob_of(arm, k=knob):
                return k if arm in TUNABLE else DFLT
            only = shared_reps(nodes, tok, knob_of)
            if only is not None and not only:
                print("  %-18s %-10s -- not computable: the four arms share no rep"
                      % (op, knob))
                continue

            def val(arm, k=knob):
                kk = knob_of(arm, k)
                return (layer(arm, nodes, tok, kk, only) if op == "layer total"
                        else us(arm, nodes, tok, op, kk, only))
            m, a, b, s = val(MAIN), val(PR12), val(PR89), val(STACK)
            if None in (m, a, b, s):
                missing = [n for n, v in (("main", m), ("#1+#2", a),
                                          ("#8+#9", b), ("stack", s))
                           if v is None]
                print("  %-18s %-10s -- not computable: %s not measured here"
                      % (op, knob, ", ".join(missing)))
                continue
            exp = m + (a - m) + (b - m)
            res = s - exp
            best = max(abs(a - m), abs(b - m))
            print("  %-18s %-10s %9s %9s %9s %9s %9s %8s  (%s of the larger single"
                  " win, reps %s)"
                  % (op, knob, f(m), f(a), f(b), f(exp), f(s), f(res),
                     "-" if not best else "%.0f%%" % (100.0 * res / best),
                     ",".join(str(r) for r in sorted(only))))
        print()


def subparts(nodes, tok):
    """EP_NUM_SUB_PARTS=1 against each arm's own default.

    On main and #8+#9 the variable is read by nothing (only #1+#2's compiler.hpp
    forwards it to the JIT), so those two rows are an INERTNESS CONTROL: they must
    land on their own default. If they move, the difference came from the
    environment and not from the knob.
    """
    print("EP_NUM_SUB_PARTS=1 -- %d tok, %d nodes" % (tok, nodes))
    print()
    print("  %-18s %-10s %9s %9s %9s %s"
          % ("op", "arm", "default", "=1", "d", "reads it?"))
    for op in OPS + ["layer total"]:
        any_row = False
        for arm, label in ARMS:
            base = (layer(arm, nodes, tok) if op == "layer total"
                    else us(arm, nodes, tok, op))
            v = (layer(arm, nodes, tok, SUB1) if op == "layer total"
                 else us(arm, nodes, tok, op, SUB1))
            if base is None or v is None:
                continue
            print("  %-18s %-10s %9s %9s %9s %s"
                  % (op if not any_row else "", label, f(base), f(v),
                     pct(v, base), "yes" if arm in TUNABLE else "NO (control)"))
            any_row = True
        if any_row:
            print()


def node_layering(nodes, tok):
    print("NODE LAYERING -- per-node mean us, %d nodes, %d tok" % (nodes, tok))
    print("  combine is layered BY NODE and which node is slow flips between runs,")
    print("  so one node's log is a sample of one layer. This is why every table")
    print("  above pools all %d ranks." % (nodes * 8))
    print()
    print("  %-10s %-18s %s" % ("arm", "op", "per-node means (node1..nodeN)"))
    for arm, label in ARMS:
        for op in ("dispatch", "reduced combine"):
            rows = []
            for rep, (_o, _w, bynode, _c, _r, _t) in cell(g, arm, nodes, tok):
                import statistics as st
                vals = [st.mean(bynode[n][op]) for n in sorted(bynode)
                        if op in bynode[n]]
                if vals:
                    rows.append("rep%d %s (spread %.1f%%)"
                                % (rep, " ".join(f(v) for v in vals),
                                   100.0 * (max(vals) - min(vals)) / min(vals)))
            if rows:
                print("  %-10s %-18s %s" % (label, op, rows[0]))
                for r in rows[1:]:
                    print("  %-10s %-18s %s" % ("", "", r))
    print()


def reproduction():
    """This campaign's 2-node table against results/p5en_stack_20260831.

    Different instances, nine days apart, same images by BUILD_REF and the same
    merge sha for the stack arm. Nothing is pooled: this prints the two independently
    and their ratio, so an infrastructure change shows up as a level shift here
    rather than hiding inside the SCALING section.
    """
    print("REPRODUCTION vs 2026-08-31 -- 2 nodes, 128 tok, default geometry")
    print("  different instances, 9 days apart, same BUILD_REFs. NOT pooled.")
    print()
    print("  %-18s %-10s %9s %9s %8s"
          % ("op", "arm", "08-31", "this run", "d"))
    for op in OPS:
        for arm, label in ARMS:
            o = old.stat(cell(old, arm, 2, 128, DFLT), op)[0]
            n = us(arm, 2, 128, op)
            if o is None or n is None:
                continue
            print("  %-18s %-10s %9s %9s %8s"
                  % (op if arm == ARMS[0][0] else "", label, f(o), f(n),
                     pct(n, o)))
        print()


def audit():
    print("AUDIT -- what is on disk, and rank completeness")
    print()
    print("  %-13s %-4s %-6s %-10s %6s %s"
          % ("arm", "N", "tok", "knob", "reps", "ranks seen per rep"))
    for nodes in NODESET:
        for tok, _ in TOKS:
            for arm, _ in ARMS:
                for knob in (DFLT, SUB1):
                    reps = cell(g, arm, nodes, tok, knob)
                    if not reps:
                        continue
                    detail = []
                    for rep, (ops, world, _bn, _c, _r, _t) in reps:
                        seen = len(ops.get("dispatch", {}))
                        detail.append("rep%d %d/%d%s"
                                      % (rep, seen, world,
                                         "" if seen == world else "  <-- SHORT"))
                    print("  %-13s %-4s %-6s %-10s %6d %s"
                          % (arm, "%dN" % nodes, tok, knob, len(reps),
                             ", ".join(detail)))
    if g.EXCLUDED:
        print()
        print("  reps excluded as >25%% off their cell median:")
        for e in g.EXCLUDED:
            print("    %s" % e)
    if g.EMPTY:
        print()
        print("  logs present but carrying no EP lines (in-flight or died early):")
        for e in g.EMPTY:
            print("    %s" % e)
    print()


def main():
    print("4 x p5en.48xlarge (H200, sm_90), 16 x EFA @200 Gb/s = 50 GB/s per GPU,")
    print("EFA installer 1.50.0 / efa.ko 3.3.0g, GIN type 5, 2026-09-09")
    print("4N ran on all four nodes; the 2N arm ran on nodes 3+4 of the same four,")
    print("because nodes 1-2 had another tenant by then. Same instances, same images,")
    print("same day -- but a 2N-vs-4N delta below is not a same-hosts delta.")
    print("EPRUNS_2N4N=%s" % LOGS)
    print("b300 4-node reference: %s" % B300_4N)
    print("2-node reproduction reference: %s" % OLD_2N)
    print()
    config_table()
    combine_sign()
    for nodes in NODESET:
        for tok, what in TOKS:
            perf_table(nodes, tok, what)
    for tok, what in TOKS:
        scaling(tok, what)
    for nodes in NODESET:
        for tok, what in TOKS:
            additivity(nodes, tok, what)
    for nodes in NODESET:
        for tok, _ in TOKS:
            subparts(nodes, tok)
    for nodes in NODESET:
        node_layering(nodes, 128)
    reproduction()
    audit()


if __name__ == "__main__":
    main()

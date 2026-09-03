#!/usr/bin/env python3
"""Additivity: is PR #1+#2+#8+#9 worth the sum of #1+#2 and #8+#9 separately?

usage: [EPRUNS=./logs] make_stack_tables.py

Four trees, ONE campaign (2x p5en.48xlarge, H200 sm_90, 12 SM, GIN type 5,
--prefer-overlap-with-compute=0, rotated reps). Every delta here is
within-campaign on one node pair; the 3-arm campaign's numbers are NOT mixed in,
because they were taken on a different node pair on a different day and additivity
is a difference of differences -- the least forgiving thing to cross campaigns with.

The campaign was cut short by the nodes being released: 15 of 30 cells ran, so the
reps on disk are UNEVEN across arms (AUDIT prints exactly which). PERF shows every
rep an arm has, but ADDITIVITY and BEST OPERATING POINT restrict themselves to the
reps ALL the compared arms share and print that set -- a difference of differences
taken over different reps would fold rep-to-rep drift into the residual, which is
the one number this file exists to produce.

  main   54fffeff810723f574c574b1790dff189f3c6ffb
  pr12   bfbdd15ff448783f877cb2210cb3246c8452b05e   PRs #1 + #2   (dispatch side)
  pr89   3c737dcf0da5889ba7efd26e05b4808307cc38af   PRs #8 + #9   (combine side)
  stack  a35285f0af98856625e542df24bd17a985bc05d9   git merge of the two

The stack sha is a LOCAL merge commit -- it exists in no remote. What makes it
reproducible is that the merge is conflict-free (the two branches touch disjoint
files apart from README), so the two parent shas plus `git merge` fully determine
the tree. Every log stamps both parents; build_stack_image.sh pins the git dates so
the merge sha is identical on every host that builds it.

WHY THE KNOB COLUMN.  PR #1's entire contribution is forwarding EP_NUM_SUB_PARTS &
friends to the JIT, so pr12 and the stack have a tuning axis that main and pr89 do
not. Comparing an arm at its DEFAULT geometry against an arm at its TUNED one is the
mistake this table is built to prevent: each arm is shown at both, and the
additivity check is run at each arm's own best point as well as at the default.

Aggregation is imported from make_3arm_tables.py (all-rank pooling from both nodes'
logs, mean over reps, >25%-off-median reps excluded and printed). TIME IS THE
METRIC; --ignore-local-traffic is OFF so SO is not a wire rate.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["EPRUNS"] = os.environ.get("EPRUNS_STACK", os.path.join(HERE, "logs"))
_sib = os.path.join(os.path.dirname(HERE), "p5en_3arm_20260831", "make_3arm_tables.py")
_spec = importlib.util.spec_from_file_location("gen3arm", _sib)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

f, pct, stat, rng, OPS = g.f, g.pct, g.stat, g.rng, g.OPS
DFLT, SUB1 = g.DEFAULT_KNOB, g.SUBPARTS1

MAIN, PR12, PR89, STACK = "main54fffef", "pr12bfbdd15", "pr893c737dc", "stacka35285f"
ARMS = [(MAIN, "main"), (PR12, "PR #1+#2"), (PR89, "PR #8+#9"), (STACK, "stack")]
# Which arms actually have the env forwarding. On the other two the knob is inert and
# was not re-measured here (the 3-arm campaign carries that control).
TUNABLE = {PR12, STACK}
TOKS = ((8192, "PREFILL"), (128, "DECODE"))
TAG = "%s_2N_12sm_%dtok_%s_nodbg_gin5_ovlp0_rep%d"

CACHE = {}


def cell(arm, tok, knob=DFLT, only=None):
    key = (arm, tok, knob)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = g.load(TAG % (arm, tok, knob, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    reps = CACHE[key]
    return reps if only is None else [x for x in reps if x[0] in only]


def us(arm, tok, op, knob=DFLT, only=None):
    return stat(cell(arm, tok, knob, only), op)[0]


def so(arm, tok, op, knob=DFLT, only=None):
    """all-rank-mean SO GB/s -- the statistic the repo-level README's throughput table
    quotes, so that table can be generated instead of transcribed.

    A separate function from us() on purpose: SO and us are different statistics over the
    same cell, and each rank's SO is printed by test_ep.py as an INTEGER, so this cannot
    resolve a sub-1% change in time (p5en prefill dispatch moves +0.2% here and the GB/s
    does not budge). Time stays the metric; this is for the one table that is in GB/s.
    Not a wire rate: --ignore-local-traffic is off, so SO counts intra-node destinations.
    """
    return stat(cell(arm, tok, knob, only), op, 0)[0]


def mb(arm, tok, op, knob=DFLT, only=None):
    """MB/rank: so()'s byte denominator, so a published GB/s can be checked against it"""
    v = stat(cell(arm, tok, knob, only), op, 3)[0]
    return None if v is None else v / 1e6


def layer(arm, tok, knob=DFLT, only=None):
    d = us(arm, tok, "dispatch", knob, only)
    c = us(arm, tok, "reduced combine", knob, only)
    return None if (d is None or c is None) else d + c


def shared_reps(tok, knob_of):
    """Reps present in EVERY arm to be compared, so a cross-arm delta is same-rep.

    knob_of(arm) says which knob each arm is represented by. Returns None when an arm
    has no reps at all at its knob, and an EMPTY set when every arm has reps but none
    in common -- two different gaps, and the caller words them differently. Never
    returns None for the second case, because None means "no filter" downstream and
    would silently restore the unbalanced comparison.
    """
    sets = []
    for arm, _label in ARMS:
        reps = {rep for rep, _ in cell(arm, tok, knob_of(arm))}
        if not reps:
            return None
        sets.append(reps)
    return set.intersection(*sets)


def config_table():
    print("CONFIG -- read out of the logs, not assumed")
    print()
    print("  %-9s %-8s %-42s %-6s %-8s %s"
          % ("arm", "knob", "BUILD_REF", "#SM", "#QPs", "merge parents"))
    for arm, label in ARMS:
        for knob in (DFLT, SUB1):
            reps = cell(arm, 128, knob) + cell(arm, 8192, knob)
            if not reps:
                continue
            refs, cfgs = set(), set()
            for _rep, (_o, _W, _bn, c, r, _t) in reps:
                refs |= set(r)
                cfgs |= set(c)
            par = set()
            for _rep, (_o, _W, _bn, _c, _r, tag) in reps:
                for path in __import__("glob").glob("%s/%s.node*.log" % (g.EPRUNS, tag)):
                    for line in open(path, errors="replace"):
                        if "DeepEP is a MERGE of:" in line:
                            par.add(line.split("MERGE of:")[1].split("===")[0].strip())
            print("  %-9s %-8s %-42s %-6s %-8s %s"
                  % (label, knob, ",".join(sorted(refs)) or "-",
                     ",".join(sorted({str(c[0]) for c in cfgs})) or "-",
                     ",".join(sorted({"%d/%d" % (c[1], c[2]) for c in cfgs})) or "-",
                     " | ".join("%s+%s" % tuple(p.split()[:2]) for p in sorted(par))
                     if par else ""))
    print()
    print("  #QPs 13/13 on both #8+#9 and the stack, 11/11 on main and #1+#2: PR #9's")
    print("  kDefaultGinContextCnt bump survived the merge, which is the cheapest")
    print("  available proof that the stack is not silently one arm.")
    print()


def perf_table(tok, what):
    print("%s -- %d tok, 2 nodes / 16 ranks, 12 SM, GIN type 5, ovlp=0" % (what, tok))
    print("  d is vs main at its default. `subparts1` rows exist only on the arms that")
    print("  carry PR #1's env forwarding.")
    print()
    hdr = ("  %-18s %-9s %-10s %10s %10s %-11s  %s"
           % ("op", "arm", "knob", "us", "d vs main", "SO GB/s", "per-rep us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        base = us(MAIN, tok, op)
        printed = False
        for arm, label in ARMS:
            for knob in (DFLT, SUB1):
                reps = cell(arm, tok, knob)
                mean, per_rep, short = stat(reps, op)
                if mean is None:
                    continue
                lo, hi = rng(reps, op, 0)
                print("  %-18s %-9s %-10s %10s %10s %-11s  %s%s"
                      % (op if not printed else "", label, knob, f(mean),
                         "" if (arm == MAIN and knob == DFLT) else pct(mean, base),
                         "-" if lo is None else "%d-%d" % (lo, hi),
                         " / ".join(f(v) for v in per_rep),
                         ("  !! " + "; ".join(short)) if short else ""))
                printed = True
        if printed:
            print()


def additivity(tok, what):
    """measured stack vs the sum of the two arms' separate savings.

    expected = main + (pr12 - main) + (pr89 - main), i.e. the two savings simply add.
    A measured stack SLOWER than that means the two PRs are partly attacking the same
    cost; FASTER means they compound. The residual is printed in us AND as a share of
    the larger of the two individual savings, because a 10 us residual means something
    different next to a 12 us saving than next to a 500 us one.
    """
    print("ADDITIVITY -- %s (%d tok)" % (what, tok))
    print()
    print("  expected = main + (#1+#2 - main) + (#8+#9 - main)  [the two savings add]")
    print("  residual = measured stack - expected. > 0 means they overlap (the stack")
    print("  gets less than the sum); < 0 means they compound.")
    print()
    print("  %-18s %-10s %9s %9s %9s %9s %9s %8s"
          % ("op", "knob", "main", "#1+#2", "#8+#9", "expected", "stack", "residual"))
    for op in OPS + ["layer total"]:
        for knob in (DFLT, SUB1):
            # main and #8+#9 have no tuned variant; at the tuned knob they are
            # represented by their own default, which is what they would ship as.
            def knob_of(arm, k=knob):
                return k if arm in TUNABLE else DFLT
            only = shared_reps(tok, knob_of)
            if only is not None and not only:
                print("  %-18s %-10s -- not computable: the four arms share no rep"
                      % (op, knob))
                continue

            def val(arm, k=knob):
                kk = knob_of(arm, k)
                return (layer(arm, tok, kk, only) if op == "layer total"
                        else us(arm, tok, op, kk, only))
            m, a, b, s = val(MAIN), val(PR12), val(PR89), val(STACK)
            if None in (m, a, b, s):
                # Silence here would look like "no residual" rather than "one of the
                # four arms was never run at this shape", which is the difference
                # between a result and a gap.
                missing = [n for n, v in (("main", m), ("#1+#2", a), ("#8+#9", b),
                                          ("stack", s)) if v is None]
                print("  %-18s %-10s -- not computable: %s not measured in this campaign"
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


def best_of(tok, what):
    """Each arm at its own operating point, which is what a deployment would pick."""
    print("BEST OPERATING POINT -- %s (%d tok), dispatch + reduced combine" % (what, tok))
    print("  each arm at whichever knob is faster FOR IT, since that is what would be")
    print("  deployed; the knob column says which one won. `reps` is per row: an arm")
    print("  with more reps on disk is NOT averaged over them here unless main has")
    print("  them too, so every `d vs main` is a same-rep difference.")
    print()
    print("  %-9s %-10s %10s %10s %10s %9s  %s"
          % ("arm", "knob", "dispatch", "redComb", "layer", "d vs main", "reps"))
    main_knob, base_reps = None, None
    for arm, label in ARMS:
        opts = [(k, layer(arm, tok, k)) for k in ((DFLT, SUB1) if arm in TUNABLE
                                                  else (DFLT,))]
        opts = [(k, v) for k, v in opts if v is not None]
        if not opts:
            continue
        knob, _ = min(opts, key=lambda kv: kv[1])
        if arm == MAIN:
            # main is the denominator, so it sets the rep basis every other row is
            # held to -- and main's own value is RE-averaged over each row's basis,
            # not carried over from this row. Otherwise an arm with fewer reps than
            # main is divided by a mean main never had at those reps.
            main_knob, base_reps = knob, {rep for rep, _ in cell(arm, tok, knob)}
        if base_reps is None:   # main not run at this shape: nothing to hold rows to
            print("  main was not run at %d tok -- no rep basis, table skipped" % tok)
            return
        only = base_reps & {rep for rep, _ in cell(arm, tok, knob)}
        tot = layer(arm, tok, knob, only)
        if tot is None:
            print("  %-9s %-10s -- shares no rep with main" % (label, knob))
            continue
        base = layer(MAIN, tok, main_knob, only)
        print("  %-9s %-10s %10s %10s %10s %9s  %s"
              % (label, knob, f(us(arm, tok, "dispatch", knob, only)),
                 f(us(arm, tok, "reduced combine", knob, only)), f(tot),
                 "" if arm == MAIN else pct(tot, base),
                 ",".join(str(r) for r in sorted(only))))
    print()


def audit():
    print("AUDIT -- reps on disk and rank completeness (n must equal 16)")
    print()
    for tok, _ in TOKS:
        for arm, label in ARMS:
            for knob in (DFLT, SUB1):
                reps = cell(arm, tok, knob)
                if not reps:
                    if knob == DFLT or arm in TUNABLE:
                        print("  %5dtok %-9s %-10s -- not run" % (tok, label, knob))
                    continue
                print("  %5dtok %-9s %-10s %s"
                      % (tok, label, knob,
                         " ".join("rep%d:%d/%d" % (rep, len(o.get("dispatch", {})), W)
                                  for rep, (o, W, _b, _c, _r, _t) in reps)))
    print()
    if g.EMPTY:
        print("  logs present but EMPTY (in-flight, or died before iteration 1):")
        for e in sorted(set(g.EMPTY)):
            print("    %s" % e)
        print()
    if g.EXCLUDED:
        print("  reps EXCLUDED as outliers (> %.0f%% off their cell's median):"
              % (100 * g.OUTLIER))
        for e in sorted(set(g.EXCLUDED)):
            print("    %s" % e)
        print()
    else:
        print("  no rep was excluded as an outlier.")
        print()


if __name__ == "__main__":
    print("EPRUNS=%s" % g.EPRUNS)
    print()
    config_table()
    for tok, what in TOKS:
        perf_table(tok, what)
    for tok, what in TOKS:
        additivity(tok, what)
    for tok, what in TOKS:
        best_of(tok, what)
    audit()

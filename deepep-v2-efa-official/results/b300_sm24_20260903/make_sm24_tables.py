#!/usr/bin/env python3
"""b300 at 24 SM: does PR #8+#9's combine win survive once combine has the SMs?

usage: make_sm24_tables.py            (logs are read from ./logs, or $EPRUNS_SM24)

THE QUESTION. results/b300_stack_20260903 measured all four arms at 12 SM. There
the stack's 8192-token combine is 2816.0 us / 83.21 GB/s, against a 2026-08 GDAKI
row at 24 SM (1788.1 us / 131 GB/s) -- and route B's own SM sweep
(deepep-v2-efa-gdaki-b200/results/b300_20260813/b300_pfsm_p1_*) shows 12 -> 24 SM
is worth -36.1% on combine by itself, with the two arms at parity at matched
12 SM. So the 12 SM campaign cannot say whether #8+#9's combine win is a real
saving or was only recovering SM starvation. This campaign runs the same four arms
at 24 SM to find out, and re-runs two 12 SM cells in the same campaign so the
24-vs-12 delta is measured on one node pair on one day.

WHAT IS COMPARABLE TO WHAT. Everything here is the DEFAULT part geometry, 2 nodes
/ 16 ranks, GIN type 5, --prefer-overlap-with-compute=0, --test-first-only (so FP8
dispatch at expert_alignment=128), one arm label per tree. Two SM counts is the
only axis this campaign adds, and it is in every log name, so a 12 SM cell here
can never pool with a 24 SM one.

The ANCHOR section is the load-bearing one for the README: it prints this
campaign's 12 SM stack cell next to results/b300_stack_20260903's. If those two
disagree by more than the campaign's own cross-rep spread, then the cross-campaign
comparisons in the README are the thing that is wrong, and the 24-vs-12 deltas
below should be read only within this campaign.

TIME IS THE METRIC. --ignore-local-traffic is off, so SO counts intra-node
destinations and is not a wire rate -- halve it. Aggregation (all-rank pooling
from every node's log, mean over rotated reps, >25%-off-median reps excluded and
named in AUDIT) is imported from the p5en 3-arm generator, unchanged, so this
table's arithmetic is the same as every other campaign's.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
LOGS = os.environ.get("EPRUNS_SM24", os.path.join(HERE, "logs"))


def _mod(path, name, env):
    """Load a generator as its own module object with its own log dir.

    Each generator reads EPRUNS at import time, so two importlib loads give two
    independent modules pointing at two campaigns. A plain `import` would reuse
    the first one and silently read the wrong logs.
    """
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The 12 SM campaign, loaded FIRST and with its own EPRUNS, so that the anchor
# comparison below reads its logs and not ours.
D12 = os.path.join(RESULTS, "b300_stack_20260903")
old = _mod(os.path.join(D12, "make_stack_tables.py"), "gen_b300_12sm",
           {"EPRUNS_B300": os.path.join(D12, "logs")}).m
g = _mod(os.path.join(RESULTS, "p5en_3arm_20260831", "make_3arm_tables.py"),
         "gen_sm24", {"EPRUNS": LOGS})
assert g.EPRUNS != old.g.EPRUNS, "this campaign collapsed onto the 12 SM log dir"

f, pct, stat, rng, OPS = g.f, g.pct, g.stat, g.rng, g.OPS
DFLT = g.DEFAULT_KNOB

MAIN, PR12, PR89, STACK = "main54fffef", "pr12bfbdd15", "pr893c737dc", "stacka35285f"
ARMS = [(MAIN, "main"), (PR12, "PR #1+#2"), (PR89, "PR #8+#9"), (STACK, "stack")]
TOKS = ((8192, "PREFILL"), (128, "DECODE"))
# The SM counts this campaign adds, and the one every delta is taken against. Mirrors
# run_b300_sm24_campaign.sh's SMS env, so `SMS="24 48" ... && SM_LIST="24 48" this.py`
# needs no edit here. 12 is the anchor because that is what results/b300_stack_20260903
# and the p5en campaigns ran at.
SM_NEW = sorted({int(x) for x in os.environ.get("SM_LIST", "24").split()})
SM_ANCHOR = 12
SM_ALL = sorted(set(SM_NEW) | {SM_ANCHOR})
TAG = "%s_2N_%dsm_%dtok_%s_nodbg_gin5_ovlp0_rep%d"

CACHE = {}


def cell(arm, sm, tok, only=None):
    key = (arm, sm, tok)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = g.load(TAG % (arm, sm, tok, DFLT, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    reps = CACHE[key]
    return reps if only is None else [x for x in reps if x[0] in only]


def us(arm, sm, tok, op, only=None):
    return stat(cell(arm, sm, tok, only), op)[0]


def so(arm, sm, tok, op, only=None):
    """all-rank-mean SO GB/s -- the statistic the repo README's throughput table
    quotes. Integer per rank in the log, so it cannot resolve a sub-1% change in
    time; the us column stays the primary number."""
    return stat(cell(arm, sm, tok, only), op, 0)[0]


def layer(arm, sm, tok, only=None):
    d = us(arm, sm, tok, "dispatch", only)
    c = us(arm, sm, tok, "reduced combine", only)
    return None if (d is None or c is None) else d + c


def shared(sm, tok, arms=None):
    """reps present in EVERY arm compared, so a cross-arm delta is same-rep"""
    sets = []
    for arm, _l in (arms or ARMS):
        reps = {rep for rep, _ in cell(arm, sm, tok)}
        if not reps:
            return None
        sets.append(reps)
    return set.intersection(*sets) if sets else None


def config_table():
    """Read the config out of the logs. The SM count especially: test_ep.py CLAMPS
    --num-sms to what the QP budget allows (64 requested came back as 55 on the
    2026-08 campaign), so a tag saying 24sm is a request, not a measurement."""
    print("CONFIG -- read out of the logs, not assumed")
    print()
    print("  %-9s %-6s %-42s %-6s %-8s %s"
          % ("arm", "tag SM", "BUILD_REF", "#SM", "#QPs", "world"))
    bad = []
    for sm in SM_ALL:
        for arm, label in ARMS:
            reps = cell(arm, sm, 128) + cell(arm, sm, 8192)
            if not reps:
                continue
            refs, cfgs, world = set(), set(), set()
            for _rep, (_o, W, _bn, c, r, _t) in reps:
                refs |= set(r)
                cfgs |= set(c)
                world.add(W)
            sms = sorted({c[0] for c in cfgs})
            print("  %-9s %-6d %-42s %-6s %-8s %s"
                  % (label, sm, ",".join(sorted(refs)) or "-",
                     ",".join(str(s) for s in sms),
                     ",".join(sorted({"%d/%d" % (c[1], c[2]) for c in cfgs})),
                     ",".join(str(w) for w in sorted(world))))
            if sms != [sm]:
                bad.append("%s @%dsm: log says #SM %s" % (label, sm, sms))
    print()
    if bad:
        print("  !! REQUESTED SM != MEASURED SM -- the tag is a lie for these cells,")
        print("     and a 24-vs-12 delta computed over them is not an SM delta:")
        for b in bad:
            print("       %s" % b)
    else:
        print("  every cell's measured #SM equals the SM in its tag.")
    print()


def perf_table(tok, what):
    print("%s -- %d tok, 2 nodes / 16 ranks, default geometry, GIN type 5, ovlp=0"
          % (what, tok))
    print("  `d vs main` is against main AT THE SAME SM COUNT; `d vs 12sm` is the")
    print("  same arm across the two SM counts of this campaign.")
    print()
    hdr = ("  %-18s %-9s %-6s %10s %10s %10s %-11s  %s"
           % ("op", "arm", "SM", "us", "d vs main", "d vs 12sm", "SO GB/s",
              "per-rep us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        printed = False
        for arm, label in ARMS:
            for sm in SM_ALL:
                reps = cell(arm, sm, tok)
                mean, per_rep, short = stat(reps, op)
                if mean is None:
                    continue
                base = us(MAIN, sm, tok, op)
                a12 = us(arm, SM_ANCHOR, tok, op) if sm != SM_ANCHOR else None
                lo, hi = rng(reps, op, 0)
                print("  %-18s %-9s %-6d %10s %10s %10s %-11s  %s%s"
                      % (op if not printed else "", label, sm, f(mean),
                         "" if arm == MAIN else pct(mean, base),
                         "" if a12 is None else pct(mean, a12),
                         "-" if lo is None else "%d-%d" % (lo, hi),
                         " / ".join(f(v) for v in per_rep),
                         ("  !! " + "; ".join(short)) if short else ""))
                printed = True
        if printed:
            print()


def sm_effect(tok, what, sm):
    """The headline: what 12 -> 24 SM is worth PER ARM.

    If the percentages are the same on every arm, the SM count and the PRs are
    independent and the README's cross-SM comparison is sound. If main gains much
    more than the stack, then #8+#9 was recovering SM starvation and its win at
    12 SM does not add to a 24 SM deployment -- which is the whole question.
    """
    print("SM EFFECT -- %s (%d tok): what %d -> %d SM buys, per arm"
          % (what, tok, SM_ANCHOR, sm))
    print()
    print("  %-18s %-9s %10s %10s %10s %10s"
          % ("op", "arm", "%dsm" % SM_ANCHOR, "%dsm" % sm, "d",
             "SO %d->%d" % (SM_ANCHOR, sm)))
    for op in ("dispatch", "combine", "reduced combine"):
        for arm, label in ARMS:
            a, b = us(arm, SM_ANCHOR, tok, op), us(arm, sm, tok, op)
            if a is None or b is None:
                continue
            print("  %-18s %-9s %10s %10s %10s %10s"
                  % (op, label, f(a), f(b), pct(b, a),
                     "%s -> %s" % (f(so(arm, SM_ANCHOR, tok, op)),
                                   f(so(arm, sm, tok, op)))))
        print()


def additivity(sm, tok, what):
    """measured stack vs the sum of the two arms' separate savings, at one SM count"""
    print("ADDITIVITY -- %s (%d tok) @ %d SM" % (what, tok, sm))
    print("  expected = main + (#1+#2 - main) + (#8+#9 - main). residual > 0 means")
    print("  the two PRs overlap; < 0 means they compound.")
    print()
    print("  %-18s %9s %9s %9s %9s %9s %8s"
          % ("op", "main", "#1+#2", "#8+#9", "expected", "stack", "residual"))
    only = shared(sm, tok)
    if only is not None and not only:
        print("  -- not computable: the four arms share no rep at this cell")
        print()
        return
    for op in OPS + ["layer total"]:
        def val(arm):
            return (layer(arm, sm, tok, only) if op == "layer total"
                    else us(arm, sm, tok, op, only))
        m, a, b, s = val(MAIN), val(PR12), val(PR89), val(STACK)
        if None in (m, a, b, s):
            missing = [n for n, v in (("main", m), ("#1+#2", a), ("#8+#9", b),
                                      ("stack", s)) if v is None]
            print("  %-18s -- not computable: %s not measured here"
                  % (op, ", ".join(missing)))
            continue
        exp = m + (a - m) + (b - m)
        res = s - exp
        best = max(abs(a - m), abs(b - m))
        print("  %-18s %9s %9s %9s %9s %9s %8s  (%s of the larger single win)"
              % (op, f(m), f(a), f(b), f(exp), f(s), f(res),
                 "-" if not best else "%.0f%%" % (100.0 * res / best)))
    if only:
        print("  reps: %s (the set all four arms share)"
              % ",".join(str(r) for r in sorted(only)))
    else:
        print("  reps: unrestricted -- an arm is missing at this cell, so there is no")
        print("        shared-rep set to restrict to.")
    print()


def anchor():
    """This campaign's 12 SM stack cells vs the ones results/b300_stack_20260903
    already published. Same tree, same hosts, same tag -- so any difference is
    drift, and it bounds how much of a cross-campaign delta is real."""
    print("ANCHOR -- this campaign's 12 SM stack cell vs results/b300_stack_20260903")
    print("  Same arm label, same tag scheme, different day. A difference here is")
    print("  drift; it is the error bar on every cross-campaign number in the README.")
    print()
    print("  %-18s %6s %12s %12s %8s"
          % ("op", "tok", "here", "20260903", "d"))
    any_row = False
    for tok, _what in TOKS:
        for op in ("dispatch", "combine", "reduced combine"):
            here = us(STACK, SM_ANCHOR, tok, op)
            there = old.us(STACK, tok, op)
            if here is None or there is None:
                continue
            any_row = True
            print("  %-18s %6d %12s %12s %8s"
                  % (op, tok, f(here), f(there), pct(here, there)))
        h, t = layer(STACK, SM_ANCHOR, tok), old.layer(STACK, tok)
        if h is not None and t is not None:
            print("  %-18s %6d %12s %12s %8s" % ("layer total", tok, f(h), f(t),
                                                 pct(h, t)))
        print()
    if not any_row:
        print("  no 12 SM stack cell in this campaign (ANCHOR=0?) -- then every")
        print("  24-vs-12 number here is cross-campaign and carries that risk.")
        print()


def audit():
    print("AUDIT -- reps on disk and rank completeness (n must equal 16)")
    print()
    for tok, _ in TOKS:
        for sm in SM_ALL:
            for arm, label in ARMS:
                reps = cell(arm, sm, tok)
                if not reps:
                    print("  %5dtok %-9s %2dsm -- not run" % (tok, label, sm))
                    continue
                print("  %5dtok %-9s %2dsm %s"
                      % (tok, label, sm,
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
    print("2 x p6-b300.48xlarge (sm_103), EFA 1.50.0 + efa.ko 3.3.0, GIN type 5")
    print("EPRUNS=%s" % g.EPRUNS)
    print("12 SM reference campaign: %s" % old.g.EPRUNS)
    print()
    config_table()
    for tok, what in TOKS:
        perf_table(tok, what)
    for sm in SM_NEW:
        for tok, what in TOKS:
            sm_effect(tok, what, sm)
    for sm in SM_NEW:
        for tok, what in TOKS:
            additivity(sm, tok, what)
    anchor()
    audit()

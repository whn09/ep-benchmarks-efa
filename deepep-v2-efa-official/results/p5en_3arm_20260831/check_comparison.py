#!/usr/bin/env python3
"""Re-derive comparison.md's claims straight from logs/. Exits nonzero on drift.

usage: [EPRUNS=./logs] check_comparison.py

comparison.md's tables are Markdown, i.e. hand-placed. Two things can go wrong
that reading the file cannot catch:

  1. a number present in tables.txt but pasted into the WRONG row -- grepping for
     it succeeds and the table still lies;
  2. logs/ changing (a rep added, a bad log removed) without the doc being
     regenerated, so the prose silently stops describing the data.

So this asserts each claim as (arm, nodes, tokens, op) -> value, recomputed from
the logs through make_3arm_tables.py's own aggregation. Tolerance is 0.06, i.e.
one unit of the last printed digit.

Add a row here whenever you add a number to comparison.md.
"""
import importlib.util
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "gen", os.path.join(HERE, "make_3arm_tables.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

MAIN = "main54fffef"
PR12 = "pr12bfbdd15"
PR89 = "pr893c737dc"
DFLT = g.DEFAULT_KNOB
SUB1 = g.SUBPARTS1
TOL = 0.06


def us(arm, n, tok, op, knob=DFLT):
    m, _, _ = g.stat(g.cell(arm, n, tok, knob), op)
    if m is None:
        raise SystemExit("no data for %s %dN %dtok %s %s" % (arm, n, tok, knob, op))
    return m


def delta(arm, n, tok, op, knob=DFLT):
    """% vs main at DEFAULT geometry -- the configuration being replaced."""
    base = us(MAIN, n, tok, op)
    return 100.0 * (us(arm, n, tok, op, knob) - base) / base


def knob_delta(arm, n, tok, op):
    """% of the knob against that arm's OWN default -- a different denominator."""
    base = us(arm, n, tok, op)
    return 100.0 * (us(arm, n, tok, op, SUB1) - base) / base


def layer(arm, n, tok, knob=DFLT):
    return (us(arm, n, tok, "dispatch", knob)
            + us(arm, n, tok, "reduced combine", knob))


def layer_delta(arm, n, tok, knob=DFLT):
    return 100.0 * (layer(arm, n, tok, knob) / layer(MAIN, n, tok) - 1)


def reps(arm, n, tok, op, knob=DFLT):
    _m, kept, _s = g.stat(g.cell(arm, n, tok, knob), op)
    return sorted(kept)


def all_reps(arm, n, tok, op, knob=DFLT):
    """every rep's all-rank mean, INCLUDING any the outlier policy dropped."""
    out = {}
    for rep, (ops, _W, _bn, _c, _r, _t) in g.cell(arm, n, tok, knob):
        if op in ops:
            out[rep] = st.mean([v[2] for v in ops[op].values()])
    return out


def one_rep(arm, n, tok, op, knob, rep):
    return all_reps(arm, n, tok, op, knob)[rep]


def median_of(arm, n, tok, op, knob=DFLT):
    return st.median(all_reps(arm, n, tok, op, knob).values())


CLAIMS = [
    # section: The headline
    ("headline: PR12 decode dispatch 2N",   delta(PR12, 2, 128, "dispatch"),          -33.3),
    ("headline: PR89 decode dispatch 2N",   delta(PR89, 2, 128, "dispatch"),          -27.2),
    ("headline: PR89 prefill redComb 2N",   delta(PR89, 2, 8192, "reduced combine"),  -13.4),
    ("headline: PR89 decode combine 2N",    delta(PR89, 2, 128, "combine"),           -11.6),
    ("headline: PR12 decode combine 2N",    delta(PR12, 2, 128, "combine"),            +0.0),
    # section: The headline -- layer total table
    ("layer 2N/128 main",                   layer(MAIN, 2, 128),                      348.8),
    ("layer 2N/128 PR12",                   layer(PR12, 2, 128),                      292.0),
    ("layer 2N/128 PR89",                   layer(PR89, 2, 128),                      273.0),
    # section: The headline -- the delta-shrinks table
    ("shrink: PR12 decode disp 4N",         delta(PR12, 4, 128, "dispatch"),           -7.5),
    ("shrink: PR89 decode disp 4N",         delta(PR89, 4, 128, "dispatch"),           -7.1),
    ("shrink: PR89 prefill redComb 4N",     delta(PR89, 4, 8192, "reduced combine"),   -2.2),
    ("shrink: PR89 decode redComb 4N",      delta(PR89, 4, 128, "reduced combine"),    -6.5),
    # section: Prefill
    ("prefill 2N main dispatch",            us(MAIN, 2, 8192, "dispatch"),           1499.8),
    ("prefill 2N PR12 dispatch",            us(PR12, 2, 8192, "dispatch"),           1498.7),
    ("prefill 2N PR89 dispatch",            us(PR89, 2, 8192, "dispatch"),           1498.4),
    ("prefill 2N main cachedDisp",          us(MAIN, 2, 8192, "cached dispatch"),    1588.8),
    ("prefill 2N PR89 cachedDisp",          us(PR89, 2, 8192, "cached dispatch"),    1496.2),
    ("prefill 2N main combine",             us(MAIN, 2, 8192, "combine"),            3587.8),
    ("prefill 2N PR89 combine",             us(PR89, 2, 8192, "combine"),            3172.5),
    ("prefill 2N main redComb",             us(MAIN, 2, 8192, "reduced combine"),    4238.0),
    ("prefill 2N PR89 redComb",             us(PR89, 2, 8192, "reduced combine"),    3670.5),
    ("prefill layer 2N main",               layer(MAIN, 2, 8192),                    5737.8),
    ("prefill layer 2N PR89",               layer(PR89, 2, 8192),                    5168.8),
    ("prefill layer 2N PR89 delta",         100.0 * (layer(PR89, 2, 8192) / layer(MAIN, 2, 8192) - 1), -9.9),
    ("prefill 4N main dispatch",            us(MAIN, 4, 8192, "dispatch"),           3965.2),
    ("prefill 4N PR12 dispatch",            us(PR12, 4, 8192, "dispatch"),           3969.2),
    ("prefill 4N PR89 dispatch",            us(PR89, 4, 8192, "dispatch"),           3961.2),
    ("prefill 4N main cachedDisp",          us(MAIN, 4, 8192, "cached dispatch"),    4254.0),
    ("prefill 4N PR89 cachedDisp",          us(PR89, 4, 8192, "cached dispatch"),    3946.2),
    ("prefill 4N PR89 cachedDisp delta",    delta(PR89, 4, 8192, "cached dispatch"),   -7.2),
    ("prefill 2N PR89 cachedDisp delta",    delta(PR89, 2, 8192, "cached dispatch"),   -5.8),
    ("prefill layer 4N PR89 delta",         100.0 * (layer(PR89, 4, 8192) / layer(MAIN, 4, 8192) - 1), -1.5),
    # section: Decode
    ("decode 2N main dispatch",             us(MAIN, 2, 128, "dispatch"),             169.6),
    ("decode 2N PR12 dispatch",             us(PR12, 2, 128, "dispatch"),             113.1),
    ("decode 2N PR89 dispatch",             us(PR89, 2, 128, "dispatch"),             123.5),
    ("decode 2N main cachedDisp",           us(MAIN, 2, 128, "cached dispatch"),      166.2),
    ("decode 2N PR12 cachedDisp",           us(PR12, 2, 128, "cached dispatch"),      107.1),
    ("decode 2N PR89 cachedDisp",           us(PR89, 2, 128, "cached dispatch"),      120.2),
    ("decode 2N main combine",              us(MAIN, 2, 128, "combine"),              162.5),
    ("decode 2N PR89 combine",              us(PR89, 2, 128, "combine"),              143.6),
    ("decode 2N main redComb",              us(MAIN, 2, 128, "reduced combine"),      179.2),
    ("decode 2N PR12 redComb",              us(PR12, 2, 128, "reduced combine"),      178.9),
    ("decode 2N PR89 redComb",              us(PR89, 2, 128, "reduced combine"),      149.5),
    ("decode 4N main dispatch",             us(MAIN, 4, 128, "dispatch"),             184.0),
    ("decode 4N PR12 dispatch",             us(PR12, 4, 128, "dispatch"),             170.3),
    ("decode 4N PR89 dispatch",             us(PR89, 4, 128, "dispatch"),             171.0),
    ("decode 4N main cachedDisp",           us(MAIN, 4, 128, "cached dispatch"),      178.9),
    ("decode 4N PR12 cachedDisp",           us(PR12, 4, 128, "cached dispatch"),      172.2),
    ("decode 4N PR89 cachedDisp",           us(PR89, 4, 128, "cached dispatch"),      167.6),
    ("decode 4N main combine",              us(MAIN, 4, 128, "combine"),              244.8),
    ("decode 4N PR89 combine",              us(PR89, 4, 128, "combine"),              234.6),
    ("decode 4N main redComb",              us(MAIN, 4, 128, "reduced combine"),      253.6),
    ("decode 4N PR12 redComb",              us(PR12, 4, 128, "reduced combine"),      253.3),
    ("decode 4N PR89 redComb",              us(PR89, 4, 128, "reduced combine"),      237.2),
    ("decode layer 4N main",                layer(MAIN, 4, 128),                      437.6),
    ("decode layer 4N PR12",                layer(PR12, 4, 128),                      423.6),
    ("decode layer 4N PR89",                layer(PR89, 4, 128),                      408.2),
    ("decode layer 4N PR12 delta",          100.0 * (layer(PR12, 4, 128) / layer(MAIN, 4, 128) - 1), -3.2),
    ("decode layer 4N PR89 delta",          100.0 * (layer(PR89, 4, 128) / layer(MAIN, 4, 128) - 1), -6.7),
    ("decode layer 2N PR12 delta",          100.0 * (layer(PR12, 2, 128) / layer(MAIN, 2, 128) - 1), -16.3),
    ("decode layer 2N PR89 delta",          100.0 * (layer(PR89, 2, 128) / layer(MAIN, 2, 128) - 1), -21.7),
    # section: Scaling, both axes -- ratios
    ("scale prefill disp main 4N/2N",       us(MAIN, 4, 8192, "dispatch") / us(MAIN, 2, 8192, "dispatch"), 2.64),
    ("scale prefill redComb PR89 4N/2N",    us(PR89, 4, 8192, "reduced combine") / us(PR89, 2, 8192, "reduced combine"), 2.12),
    ("scale decode disp main 4N/2N",        us(MAIN, 4, 128, "dispatch") / us(MAIN, 2, 128, "dispatch"), 1.08),
    ("scale decode disp PR12 4N/2N",        us(PR12, 4, 128, "dispatch") / us(PR12, 2, 128, "dispatch"), 1.51),
    ("scale decode disp PR89 4N/2N",        us(PR89, 4, 128, "dispatch") / us(PR89, 2, 128, "dispatch"), 1.38),
    ("scale decode redComb PR89 4N/2N",     us(PR89, 4, 128, "reduced combine") / us(PR89, 2, 128, "reduced combine"), 1.59),
    # prose: "57 us of which the patches show is removable"
    # prose rounds 56.6 to "57 us"; the claim is checked at full precision
    ("prose: removable us, decode disp 2N", us(MAIN, 2, 128, "dispatch") - us(PR12, 2, 128, "dispatch"), 56.6),

    # ---------------- section: EP_NUM_SUB_PARTS=1 ----------------
    # inertness control: main and PR89 must land on their OWN default
    ("inert: main 2N/128 disp knob",        us(MAIN, 2, 128, "dispatch", SUB1),        169.8),
    ("inert: main 2N/128 disp d",           knob_delta(MAIN, 2, 128, "dispatch"),        +0.1),
    ("inert: main 2N/128 redComb d",        knob_delta(MAIN, 2, 128, "reduced combine"), -0.1),
    ("inert: PR89 2N/128 disp knob",        us(PR89, 2, 128, "dispatch", SUB1),        122.9),
    ("inert: PR89 2N/128 disp d",           knob_delta(PR89, 2, 128, "dispatch"),        -0.5),
    ("inert: PR89 2N/128 redComb d",        knob_delta(PR89, 2, 128, "reduced combine"), +0.1),
    # PR12 decode, the arm where the knob is a lever
    ("knob PR12 2N/128 disp",               us(PR12, 2, 128, "dispatch", SUB1),        106.1),
    ("knob PR12 2N/128 disp d",             knob_delta(PR12, 2, 128, "dispatch"),        -6.2),
    ("knob PR12 2N/128 expDisp",            us(PR12, 2, 128, "expanded dispatch", SUB1), 105.7),
    ("knob PR12 2N/128 expDisp d",          knob_delta(PR12, 2, 128, "expanded dispatch"), -6.7),
    ("knob PR12 2N/128 expDisp base",       us(PR12, 2, 128, "expanded dispatch"),     113.2),
    ("knob PR12 2N/128 cachedDisp",         us(PR12, 2, 128, "cached dispatch", SUB1), 103.2),
    ("knob PR12 2N/128 cachedDisp d",       knob_delta(PR12, 2, 128, "cached dispatch"), -3.6),
    ("knob PR12 2N/128 combine",            us(PR12, 2, 128, "combine", SUB1),         162.3),
    ("knob PR12 2N/128 combine d",          knob_delta(PR12, 2, 128, "combine"),         -0.2),
    ("knob PR12 2N/128 redComb",            us(PR12, 2, 128, "reduced combine", SUB1), 178.7),
    ("knob PR12 2N/128 redComb d",          knob_delta(PR12, 2, 128, "reduced combine"), -0.1),
    ("knob PR12 4N/128 disp",               us(PR12, 4, 128, "dispatch", SUB1),        156.4),
    ("knob PR12 4N/128 disp d",             knob_delta(PR12, 4, 128, "dispatch"),        -8.1),
    ("knob PR12 4N/128 expDisp",            us(PR12, 4, 128, "expanded dispatch", SUB1), 155.4),
    ("knob PR12 4N/128 expDisp d",          knob_delta(PR12, 4, 128, "expanded dispatch"), -8.7),
    ("knob PR12 4N/128 expDisp base",       us(PR12, 4, 128, "expanded dispatch"),     170.2),
    ("knob PR12 4N/128 cachedDisp",         us(PR12, 4, 128, "cached dispatch", SUB1), 155.1),
    ("knob PR12 4N/128 cachedDisp d",       knob_delta(PR12, 4, 128, "cached dispatch"), -9.9),
    ("knob PR12 4N/128 combine",            us(PR12, 4, 128, "combine", SUB1),         244.6),
    ("knob PR12 4N/128 redComb",            us(PR12, 4, 128, "reduced combine", SUB1), 253.3),
    # PR12 prefill: the knob COSTS dispatch at 2N and pays it back in combine
    ("knob PR12 2N/8192 disp",              us(PR12, 2, 8192, "dispatch", SUB1),      1565.0),
    ("knob PR12 2N/8192 disp d",            knob_delta(PR12, 2, 8192, "dispatch"),       +4.4),
    ("knob PR12 2N/8192 cachedDisp",        us(PR12, 2, 8192, "cached dispatch", SUB1), 1636.6),
    ("knob PR12 2N/8192 cachedDisp d",      knob_delta(PR12, 2, 8192, "cached dispatch"), +3.1),
    ("knob PR12 2N/8192 combine",           us(PR12, 2, 8192, "combine", SUB1),       3515.7),
    ("knob PR12 2N/8192 combine d",         knob_delta(PR12, 2, 8192, "combine"),        -1.7),
    ("knob PR12 2N/8192 redComb",           us(PR12, 2, 8192, "reduced combine", SUB1), 4175.9),
    ("knob PR12 2N/8192 redComb d",         knob_delta(PR12, 2, 8192, "reduced combine"), -1.5),
    ("knob PR12 2N/8192 layer",             layer(PR12, 2, 8192, SUB1),               5740.9),
    ("knob PR12 2N/8192 layer base",        layer(PR12, 2, 8192),                     5738.3),
    ("knob PR12 4N/8192 disp",              us(PR12, 4, 8192, "dispatch", SUB1),      3974.8),
    ("knob PR12 4N/8192 disp d",            knob_delta(PR12, 4, 8192, "dispatch"),       +0.1),
    ("knob PR12 4N/8192 cachedDisp",        us(PR12, 4, 8192, "cached dispatch", SUB1), 4259.1),
    ("knob PR12 4N/8192 cachedDisp base",   us(PR12, 4, 8192, "cached dispatch"),     4253.5),
    ("knob PR12 4N/8192 combine",           us(PR12, 4, 8192, "combine", SUB1),       7554.4),
    ("knob PR12 4N/8192 combine base",      us(PR12, 4, 8192, "combine"),             7868.3),
    ("knob PR12 4N/8192 combine d",         knob_delta(PR12, 4, 8192, "combine"),        -4.0),
    ("knob PR12 4N/8192 redComb",           us(PR12, 4, 8192, "reduced combine", SUB1), 7695.1),
    ("knob PR12 4N/8192 redComb d",         knob_delta(PR12, 4, 8192, "reduced combine"), -3.1),
    ("knob PR12 4N/8192 layer",             layer(PR12, 4, 8192, SUB1),              11669.8),
    ("knob PR12 4N/8192 layer base",        layer(PR12, 4, 8192),                    11911.5),
    # prose: the two 4N prefill redComb distributions do not overlap. Both extremes
    # are asserted, because "do not overlap" is a claim about the ENDPOINTS -- a
    # mean-only check would still pass if one rep crossed over.
    ("prose: 4N/8192 redComb dflt min",     reps(PR12, 4, 8192, "reduced combine")[0],  7930.4),
    ("prose: 4N/8192 redComb dflt max",     reps(PR12, 4, 8192, "reduced combine")[-1], 7960.7),
    ("prose: 4N/8192 redComb knob min",     reps(PR12, 4, 8192, "reduced combine", SUB1)[0],  7674.8),
    ("prose: 4N/8192 redComb knob max",     reps(PR12, 4, 8192, "reduced combine", SUB1)[-1], 7724.6),
    # prose: the 2N prefill cancellation, in us -- "66 us against 64 us", "2.6 us"
    ("prose: 2N/8192 disp cost us",         us(PR12, 2, 8192, "dispatch", SUB1) - us(PR12, 2, 8192, "dispatch"), 66.3),
    ("prose: 2N/8192 redComb gain us",      us(PR12, 2, 8192, "reduced combine") - us(PR12, 2, 8192, "reduced combine", SUB1), 63.7),
    ("prose: 2N/8192 layer net us",         layer(PR12, 2, 8192, SUB1) - layer(PR12, 2, 8192), 2.6),
    ("prose: 4N/8192 layer gain us",        layer(PR12, 4, 8192) - layer(PR12, 4, 8192, SUB1), 241.7),
    # section: best configuration -- deltas vs MAIN at default (not vs own default)
    ("best: PR12 2N/128 disp vs main",      delta(PR12, 2, 128, "dispatch", SUB1),      -37.5),
    ("best: PR12 4N/128 disp vs main",      delta(PR12, 4, 128, "dispatch", SUB1),      -15.0),
    ("best: PR12 2N/128 layer",             layer(PR12, 2, 128, SUB1),                  284.8),
    ("best: PR12 2N/128 layer d",           layer_delta(PR12, 2, 128, SUB1),            -18.4),
    ("best: PR12 4N/128 layer",             layer(PR12, 4, 128, SUB1),                  409.7),
    ("best: PR12 4N/128 layer d",           layer_delta(PR12, 4, 128, SUB1),             -6.4),
    ("best: PR12 2N/8192 disp vs main",     delta(PR12, 2, 8192, "dispatch", SUB1),      +4.3),
    ("best: PR12 4N/8192 disp vs main",     delta(PR12, 4, 8192, "dispatch", SUB1),      +0.2),
    ("best: PR12 2N/8192 layer d",          layer_delta(PR12, 2, 8192, SUB1),            +0.1),
    ("best: PR12 4N/8192 layer d",          layer_delta(PR12, 4, 8192, SUB1),            -2.0),
    ("best: main 4N/8192 layer",            layer(MAIN, 4, 8192),                     11912.5),
    ("best: PR89 4N/8192 layer",            layer(PR89, 4, 8192),                     11734.5),
    # prose: "2.5x rather than 4.4x". Pure arithmetic on two checked deltas, but
    # asserted because it is the sentence carrying the section's conclusion.
    ("prose: shrinkage at knob, x",         delta(PR12, 2, 128, "dispatch", SUB1) / delta(PR12, 4, 128, "dispatch", SUB1), 2.50),
    ("prose: shrinkage at default, x",      delta(PR12, 2, 128, "dispatch") / delta(PR12, 4, 128, "dispatch"), 4.44),
    # prose: the excluded outlier and the median it was judged against. Asserted so
    # that quietly deleting the log, or changing OUTLIER, fails the check instead of
    # silently rewriting the doc's account of what was dropped and why.
    ("prose: excluded rep3 dispatch us",    one_rep(PR89, 2, 128, "dispatch", SUB1, 3),  495.5),
    ("prose: that cell's median us",        median_of(PR89, 2, 128, "dispatch", SUB1),   123.1),
]

if __name__ == "__main__":
    print("EPRUNS=%s" % g.EPRUNS)
    print()
    bad = 0
    for name, got, claimed in CLAIMS:
        ok = abs(got - claimed) <= TOL
        bad += 0 if ok else 1
        print("  %-40s doc %9.2f   logs %9.2f   %s"
              % (name, claimed, got, "ok" if ok else "MISMATCH"))
    if g.EMPTY:
        print()
        print("  WARNING: empty logs were skipped -- %s" % "; ".join(sorted(set(g.EMPTY))))
    print()
    print("%d claim%s checked, %d MISMATCH%s"
          % (len(CLAIMS), "" if len(CLAIMS) == 1 else "s", bad, "" if bad == 1 else "ES"))
    sys.exit(1 if bad else 0)

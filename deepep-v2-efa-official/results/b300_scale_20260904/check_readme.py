#!/usr/bin/env python3
"""Re-derive every number the README's node-scaling section prints, from logs/.

usage: check_readme.py            (exits nonzero on drift)

The README's tables are Markdown, i.e. hand-placed, and two failures survive
reading the file: a number that is real but pasted into the WRONG row (grep finds
it and the table still lies), and logs/ changing without the prose being
regenerated. So each claim is asserted as (arm, nodes, tokens, SM, knob, op) ->
value, recomputed through make_scale_tables.py's own aggregation.

Tolerance is one unit of the last printed digit: 0.06 for a us printed to 0.1,
0.6 for one printed as an integer, 0.06 percentage points for a percentage.

Add a row here whenever a number is added to that README section.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "gen_check", os.path.join(HERE, "make_scale_tables.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

OFF, PRS = g.OFFICIAL, g.PRS
D, P, M = "qpdefault", "prsdflt", "prsmtpp1"
FAILS = []


def chk(label, got, want, tol):
    if got is None:
        FAILS.append("%s: no data" % label)
    elif abs(got - want) > tol:
        FAILS.append("%s: README says %s, logs say %.3f" % (label, want, got))
    else:
        print("  ok  %-52s %8.2f" % (label, got))


def us(arm, n, tok, op, sm=12, knob=None):
    return g.us(arm, n, tok, op, sm, knob)


def pc(new, base):
    return None if (new is None or base is None) else 100.0 * (new - base) / base


# ---- table 1: time per doubling, official arm, 12 SM -----------------------
# op, tok, [1N, 2N, 4N], 1N->2N, 2N->4N
DOUBLING = [
    ("dispatch",        8192, [464.2, 1052.6, 2024.2], 2.27, 1.92),
    ("reduced combine", 8192, [1343.2, 3765.2, 4864.4], 2.80, 1.29),
    ("dispatch",         128, [26.0, 291.8, 318.8], 11.24, 1.09),
    ("reduced combine",  128, [54.4, 185.1, 285.3], 3.40, 1.54),
]
print("README table 1 -- time per doubling (official, 12 SM)")
for op, tok, vals, r12, r24 in DOUBLING:
    got = [us(OFF, n, tok, op) for n in (1, 2, 4)]
    for n, gv, wv in zip((1, 2, 4), got, vals):
        chk("%s %dtok %dN us" % (op, tok, n), gv, wv, 0.06)
    chk("%s %dtok 1N->2N x" % (op, tok), got[1] / got[0], r12, 0.006)
    chk("%s %dtok 2N->4N x" % (op, tok), got[2] / got[1], r24, 0.006)

# the 320 us decode floor claim: SM-invariant at 4 nodes
print("README prose -- the decode floor is SM-invariant at 4N")
chk("dispatch 128tok 4N 12sm us", us(OFF, 4, 128, "dispatch", 12), 318.8, 0.06)
chk("dispatch 128tok 4N 24sm us", us(OFF, 4, 128, "dispatch", 24), 317.7, 0.06)

# ---- table 2: PRs #1+#2 across the axis, 12 SM -----------------------------
# op, tok, [(nodes, official_int, prs_int, pct)]
PR_TABLE = [
    ("dispatch", 128, [(1, 26, 26, +0.1), (2, 292, 131, -55.0), (4, 319, 183, -42.5)]),
]
LAYER = [
    (128,  [(1, 80, 81, +1.0), (2, 477, 316, -33.7), (4, 604, 468, -22.5)]),
    (8192, [(1, 1807, 1806, -0.1), (2, 4818, 4823, +0.1), (4, 6889, 6902, +0.2)]),
]
print("README table 2 -- PRs #1+#2 vs official across the node axis")
for op, tok, rows in PR_TABLE:
    for n, o_i, p_i, d in rows:
        o, p = us(OFF, n, tok, op), us(PRS, n, tok, op, knob=P)
        chk("%s %dtok %dN official us" % (op, tok, n), o, o_i, 0.6)
        chk("%s %dtok %dN prs us" % (op, tok, n), p, p_i, 0.6)
        chk("%s %dtok %dN delta pct" % (op, tok, n), pc(p, o), d, 0.06)
for tok, rows in LAYER:
    for n, o_i, p_i, d in rows:
        o = g.layer(OFF, n, tok)
        p = g.layer(PRS, n, tok, knob=P)
        chk("layer %dtok %dN official us" % (tok, n), o, o_i, 0.6)
        chk("layer %dtok %dN prs us" % (tok, n), p, p_i, 0.6)
        chk("layer %dtok %dN delta pct" % (tok, n), pc(p, o), d, 0.06)

# ---- prose: the 1N arm-equality control (|delta| <= 1.4% on every op) ------
print("README prose -- at 1N the two arms are identical to within 1.4%")
worst = 0.0
for op in g.OPS:
    d = pc(us(PRS, 1, 128, op, knob=P), us(OFF, 1, 128, op))
    d8 = pc(us(PRS, 1, 8192, op, knob=P), us(OFF, 1, 8192, op))
    worst = max(worst, abs(d), abs(d8))
print("  ok  %-52s %8.2f" % ("worst |delta| over both tok and every op", worst))
if worst > 1.4 + 0.06:
    FAILS.append("1N arms differ by %.2f%%, README claims <= 1.4%%" % worst)

# ---- prose: the clamp control lands on official ----------------------------
print("README prose -- EP_MIN_TOKENS_PER_PART=1 lands on official")
for n, mtpp_us, off_us, d in ((2, 294.6, 291.8, +0.9), (4, 318.4, 318.8, -0.1)):
    c = us(PRS, n, 128, "dispatch", knob=M)
    o = us(OFF, n, 128, "dispatch")
    chk("clamp-off %dN us" % n, c, mtpp_us, 0.06)
    chk("clamp-off %dN official us" % n, o, off_us, 0.06)
    chk("clamp-off %dN vs official pct" % n, pc(c, o), d, 0.06)

# ---- prose: prs decode dispatch grows 1.40x vs official's 1.09x -----------
print("README prose -- what is left after the clamp still grows")
chk("prs dispatch 128tok 2N->4N x",
    us(PRS, 4, 128, "dispatch", knob=P) / us(PRS, 2, 128, "dispatch", knob=P),
    1.40, 0.006)
chk("official dispatch 128tok 2N->4N x",
    us(OFF, 4, 128, "dispatch") / us(OFF, 2, 128, "dispatch"), 1.09, 0.006)

# ---- prose: 12 -> 24 SM is two different trades ---------------------------
print("README prose -- 24 SM keeps prefill at every scale, loses decode at 4N")
for n, tok, d in ((1, 8192, -13.9), (2, 8192, -34.7), (4, 8192, -13.8),
                  (1, 128, -14.1), (2, 128, -12.2), (4, 128, +0.3)):
    a, b = g.layer(OFF, n, tok, 12), g.layer(OFF, n, tok, 24)
    chk("layer %dtok %dN 12->24sm pct" % (tok, n), pc(b, a), d, 0.06)

# ---- prose: 4N node layering spread ---------------------------------------
print("README prose -- 4N per-node spread and which node is slow")


def spreads(tok, op, arm=OFF, sm=12, knob=None):
    out = []
    for _rep, (_o, _W, bn, _c, _r, _t) in g.cell(arm, 4, tok, sm, knob):
        vals = [sum(bn[nd][op]) / len(bn[nd][op]) for nd in sorted(bn, key=int)
                if op in bn[nd]]
        if len(vals) > 1:
            out.append((100.0 * (max(vals) - min(vals)) / min(vals),
                        1 + vals.index(max(vals)), vals))
    return out


pf = spreads(8192, "reduced combine")
dc = spreads(128, "combine")
lo, hi = min(s for s, _i, _v in pf), max(s for s, _i, _v in pf)
chk("prefill reduced combine 4N spread lo", lo, 19.1, 0.06)
chk("prefill reduced combine 4N spread hi", hi, 19.4, 0.06)
pc_ = spreads(8192, "combine")
chk("prefill combine 4N spread lo", min(s for s, _i, _v in pc_), 21.4, 0.06)
chk("prefill combine 4N spread hi", max(s for s, _i, _v in pc_), 22.0, 0.06)
lo, hi = min(s for s, _i, _v in dc), max(s for s, _i, _v in dc)
chk("decode combine 4N spread lo", lo, 10.6, 0.06)
chk("decode combine 4N spread hi", hi, 14.3, 0.06)
slow = {i for _s, i, _v in pf}
if slow != {4}:
    FAILS.append("prefill reduced combine's slow node is %s, README says node4 in "
                 "every rep" % sorted(slow))
else:
    print("  ok  %-52s %8s" % ("prefill reduced combine slow node = 4, all reps", "yes"))
plain = spreads(8192, "combine")
if {i for _s, i, _v in plain} == {4}:
    FAILS.append("plain combine is ALSO slowest on node4; the README claims it is a "
                 "different node")
else:
    print("  ok  %-52s %8s"
          % ("plain combine slow node differs from node4", "yes"))
for _s, _i, v in pf:
    if max(v) < 5500 or max(v) > 5540:
        FAILS.append("prefill reduced combine slow-node value %.0f outside the "
                     "5530/5514/5520 the README quotes" % max(v))

print()
if FAILS:
    print("DRIFT -- the README does not describe these logs:")
    for x in FAILS:
        print("  !! %s" % x)
    sys.exit(1)
print("every README number in the node-scaling section re-derives from logs/.")

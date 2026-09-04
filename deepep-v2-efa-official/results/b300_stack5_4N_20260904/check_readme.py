#!/usr/bin/env python3
"""Re-derive every number the README's five-arm four-node section prints, from logs/.

usage: check_readme.py            (exits nonzero on drift)

The README's tables are Markdown, i.e. hand-placed, and two failures survive reading
the file: a number that is real but pasted into the WRONG row (grep finds it and the
table still lies), and logs/ changing without the prose being regenerated. So each
claim is asserted as (arm, tokens, op) -> value, recomputed through
make_stack5_tables.py's own aggregation.

Tolerance is one unit of the last printed digit: 0.06 for a us printed to 0.1, 0.06
percentage points for a percentage printed to 0.1.

Add a row here whenever a number is added to that README section.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "gen_check", os.path.join(HERE, "make_stack5_tables.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

OFF, MAIN, PR12, PR89, STACK = g.OFFICIAL, g.MAIN, g.PR12, g.PR89, g.STACK
FAILS = []


def chk(label, got, want, tol=0.06):
    if got is None:
        FAILS.append("%s: no data" % label)
    elif abs(got - want) > tol:
        FAILS.append("%s: README says %s, logs say %.3f" % (label, want, got))
    else:
        print("  ok  %-56s %9.2f" % (label, got))


def pc(new, base):
    return None if (new is None or base is None) else 100.0 * (new - base) / base


def val(arm, tok, op):
    return g.layer(arm, tok) if op == "layer total" else g.us(arm, tok, op)


# ---- the two arm tables ----------------------------------------------------
# tok -> [(op, main_us, [(arm, us, pct), ...]), ...]
TABLES = {
    128: [
        ("dispatch", 315.3, [(PR12, 182.7, -42.1), (PR89, 244.4, -22.5),
                             (STACK, 152.1, -51.8)]),
        ("cached dispatch", 318.6, [(PR12, 182.9, -42.6), (PR89, 242.3, -24.0),
                                    (STACK, 144.4, -54.7)]),
        ("combine", 269.9, [(PR12, 270.2, +0.1), (PR89, 292.7, +8.4),
                            (STACK, 292.6, +8.4)]),
        ("reduced combine", 284.3, [(PR12, 284.0, -0.1), (PR89, 297.8, +4.7),
                                    (STACK, 297.7, +4.7)]),
        ("layer total", 599.6, [(PR12, 466.7, -22.2), (PR89, 542.2, -9.6),
                                (STACK, 449.8, -25.0)]),
    ],
    8192: [
        ("dispatch", 2032.4, [(PR12, 2031.1, -0.1), (PR89, 2018.5, -0.7),
                              (STACK, 2017.4, -0.7)]),
        ("cached dispatch", 2133.8, [(PR12, 2132.6, -0.1), (PR89, 1993.3, -6.6),
                                     (STACK, 1992.7, -6.6)]),
        ("combine", 4505.1, [(PR12, 4511.5, +0.1), (PR89, 4204.4, -6.7),
                             (STACK, 4176.1, -7.3)]),
        ("reduced combine", 4886.8, [(PR12, 4865.2, -0.4), (PR89, 4303.8, -11.9),
                                     (STACK, 4300.2, -12.0)]),
        ("layer total", 6919.2, [(PR12, 6896.3, -0.3), (PR89, 6322.3, -8.6),
                                 (STACK, 6317.6, -8.7)]),
    ],
}
for tok, rows in sorted(TABLES.items()):
    print("README %d-tok arm table" % tok)
    for op, main_us, arms in rows:
        m = val(MAIN, tok, op)
        chk("%dtok %s main us" % (tok, op), m, main_us)
        for arm, arm_us, d in arms:
            v = val(arm, tok, op)
            chk("%dtok %s %s us" % (tok, op, arm), v, arm_us)
            chk("%dtok %s %s pct" % (tok, op, arm), pc(v, m), d)

# ---- prose: the two mains are equivalent ----------------------------------
print("README prose -- 54fffef -> 97d8f9b moves nothing by more than 0.9%")
worst = (0.0, "")
for tok in (8192, 128):
    for op in g.OPS:
        d = pc(g.us(OFF, tok, op), g.us(MAIN, tok, op))
        if d is not None and abs(d) > abs(worst[0]):
            worst = (d, "%s %dtok" % (op, tok))
    dl = pc(g.layer(OFF, tok), g.layer(MAIN, tok))
    chk("main advance layer %dtok |pct| <= 0.1" % tok, abs(dl), 0.1, 0.06)
print("  ok  %-56s %9.2f (%s)" % ("largest |delta| between the two mains", worst[0],
                                  worst[1]))
if abs(worst[0]) > 0.9 + 0.06:
    FAILS.append("the two mains differ by %.2f%% on %s, README claims <= 0.9%%"
                 % (worst[0], worst[1]))
if worst[1] != "combine 8192tok":
    FAILS.append("the largest main-advance delta is on %s, README names prefill "
                 "combine" % worst[1])
chk("main advance prefill combine 54fffef us", g.us(MAIN, 8192, "combine"), 4505.1)
chk("main advance prefill combine 97d8f9b us", g.us(OFF, 8192, "combine"), 4462.6)

# ---- prose: the decode floor and the 30% overlap --------------------------
print("README prose -- the decode floor and the additivity residual")
only = g.shared(128, [(MAIN, ""), (PR12, ""), (PR89, ""), (STACK, "")])
m = g.us(MAIN, 128, "dispatch", only)
a = g.us(PR12, 128, "dispatch", only)
b = g.us(PR89, 128, "dispatch", only)
s = g.us(STACK, 128, "dispatch", only)
exp = m + (a - m) + (b - m)
chk("decode dispatch expected-if-independent us", exp, 111.7)
chk("decode dispatch stack us", s, 152.1)
chk("decode dispatch residual us", s - exp, 40.4)
chk("decode dispatch residual as pct of larger single win",
    100.0 * (s - exp) / max(abs(a - m), abs(b - m)), 30.0, 0.6)
onp = g.shared(8192, [(MAIN, ""), (PR12, ""), (PR89, ""), (STACK, "")])
m8 = g.us(MAIN, 8192, "combine", onp)
a8 = g.us(PR12, 8192, "combine", onp)
b8 = g.us(PR89, 8192, "combine", onp)
s8 = g.us(STACK, 8192, "combine", onp)
exp8 = m8 + (a8 - m8) + (b8 - m8)
chk("prefill combine residual us", s8 - exp8, -34.8)
chk("prefill combine residual as pct of larger single win",
    100.0 * (s8 - exp8) / max(abs(a8 - m8), abs(b8 - m8)), -12.0, 0.6)

# ---- prose: node layering -------------------------------------------------
print("README prose -- node layering, decode gradient and prefill single slow node")


def per_node(arm, tok, op):
    """(arm, rep) -> per-node mean us, in node order"""
    out = []
    for rep, (_o, _W, bn, _c, _r, _t) in g.cell(arm, tok):
        nodes = sorted(bn, key=int)
        out.append((rep, nodes,
                    [sum(bn[nd][op]) / len(bn[nd][op]) for nd in nodes]))
    return out


slow, spreads = set(), []
for arm, _l in g.ARMS:
    for op in ("combine", "reduced combine"):
        for _rep, nodes, v in per_node(arm, 128, op):
            slow.add(nodes[v.index(max(v))])
            if op == "reduced combine":
                spreads.append(100.0 * (max(v) - min(v)) / min(v))
n_obs = 2 * 2 * len(g.ARMS)
if slow != {"4"}:
    FAILS.append("decode's slowest node is %s, README says node4 in all %d "
                 "observations" % (sorted(slow), n_obs))
else:
    print("  ok  %-56s %9s" % ("decode slowest node = node4, all %d obs" % n_obs,
                               "yes"))
chk("decode reduced combine spread lo", min(spreads), 12.0)
chk("decode reduced combine spread hi", max(spreads), 13.9)

acc = {}
for _rep, nodes, v in per_node(OFF, 128, "reduced combine"):
    for nd, x in zip(nodes, v):
        acc.setdefault(nd, []).append(x)
means = [sum(acc[nd]) / len(acc[nd]) for nd in sorted(acc, key=int)]
for i, want in enumerate((270.4, 278.3, 284.4, 304.3)):
    chk("official decode reduced combine node%d us" % (i + 1), means[i], want)
if means != sorted(means):
    FAILS.append("official's decode per-node means are not monotone in node index, "
                 "README calls it a gradient")
else:
    print("  ok  %-56s %9s" % ("monotone in node index", "yes"))

pf_sp, pf_hi, pr89_sp = [], [], []
for arm, _l in g.ARMS:
    for _rep, _nodes, v in per_node(arm, 8192, "reduced combine"):
        sp = 100.0 * (max(v) - min(v)) / min(v)
        if arm in (OFF, MAIN, PR12):
            pf_sp.append(sp)
            pf_hi.append(max(v))
        else:
            pr89_sp.append(sp)
chk("prefill slow-node us lo", min(pf_hi), 5517.0, 0.6)
chk("prefill slow-node us hi", max(pf_hi), 5549.0, 0.6)
chk("prefill spread lo (main/official/#1+#2)", min(pf_sp), 19.0)
chk("prefill spread hi (main/official/#1+#2)", max(pf_sp), 19.6)
chk("prefill spread lo (#8+#9/stack)", min(pr89_sp), 8.1)
chk("prefill spread hi (#8+#9/stack)", max(pr89_sp), 11.2)

# ---- prose: the NVLS control ---------------------------------------------
print("README prose -- the NVLS control")
ontag = "official_4N_12sm_%dtok_qpdefault_nodbg_gin5_ovlp0_rep%d"
worst_nv = (0.0, "")
for tok in (8192, 128):
    on_reps = []
    for rep in range(1, 9):
        r = g.on.load(ontag % (tok, rep))
        if r:
            on_reps.append((rep, r))
    for op in g.OPS:
        a = g.on.stat(on_reps, op)[0]
        b = g.us(OFF, tok, op)
        if a is None or b is None:
            continue
        d = 100.0 * (b - a) / a
        if abs(d) > abs(worst_nv[0]):
            worst_nv = (d, "%s %dtok" % (op, tok))
        if tok == 128 and op == "dispatch":
            chk("NVLS control decode dispatch pct", d, -0.9)
            pr = g.on.stat(on_reps, op)[1]
            chk("NVLS-on decode dispatch cross-rep spread pct",
                100.0 * (max(pr) - min(pr)) / min(pr), 1.8)
print("  ok  %-56s %9.2f (%s)" % ("largest NVLS on-vs-off |delta|", worst_nv[0],
                                  worst_nv[1]))
if abs(worst_nv[0]) > 1.2 + 0.06:
    FAILS.append("NVLS on-vs-off reaches %.2f%% on %s, README claims 1.2%%"
                 % (worst_nv[0], worst_nv[1]))
if worst_nv[1] != "combine 8192tok":
    FAILS.append("the largest NVLS delta is on %s, README names prefill combine"
                 % worst_nv[1])
chk("#1+#2 decode dispatch here us", g.us(PR12, 128, "dispatch"), 182.7)

# ---- the campaign's own acceptance claims -------------------------------
print("README prose -- 20/20 cells at 32/32 ranks, one BUILD_REF per arm")
REFS = {OFF: "97d8f9bcc1be31e9036db2ab591ef9b9f4e38619",
        MAIN: "54fffeff810723f574c574b1790dff189f3c6ffb",
        PR12: "bfbdd15ff448783f877cb2210cb3246c8452b05e",
        PR89: "3c737dcf0da5889ba7efd26e05b4808307cc38af",
        STACK: "a35285f0af98856625e542df24bd17a985bc05d9"}
QPS = {OFF: 11, MAIN: 11, PR12: 11, PR89: 13, STACK: 13}
cells = 0
for arm, label in g.ARMS:
    for tok in (8192, 128):
        reps = g.cell(arm, tok)
        for _rep, (o, W, _bn, c, r, _t) in reps:
            cells += 1
            if W != 32 or len(o.get("dispatch", {})) != 32:
                FAILS.append("%s %dtok: %d/%d ranks, README claims 32/32"
                             % (label, tok, len(o.get("dispatch", {})), W))
            if sorted(r) != [REFS[arm]]:
                FAILS.append("%s %dtok: BUILD_REF %s, expected %s"
                             % (label, tok, sorted(r), REFS[arm]))
            if [x[0] for x in c] != [12]:
                FAILS.append("%s %dtok: measured #SM %s, tag says 12"
                             % (label, tok, [x[0] for x in c]))
            qp = sorted({(x[1], x[2]) for x in c})
            if qp != [(QPS[arm], QPS[arm])]:
                FAILS.append("%s %dtok: #QPs %s, README says %d/%d"
                             % (label, tok, qp, QPS[arm], QPS[arm]))
if cells != 20:
    FAILS.append("%d cells on disk, README claims 20" % cells)
else:
    print("  ok  %-56s %9s"
          % ("20 cells, 32/32 ranks, 12 SM, refs + QPs match", "yes"))
# g is make_stack5_tables; the aggregation state lives on the p5en module it loaded.
if g.g.EXCLUDED:
    FAILS.append("reps were excluded as outliers: %s" % sorted(set(g.g.EXCLUDED)))
if g.g.EMPTY:
    FAILS.append("empty logs on disk: %s" % sorted(set(g.g.EMPTY)))

print()
if FAILS:
    print("DRIFT -- the README does not describe these logs:")
    for x in FAILS:
        print("  !! %s" % x)
    sys.exit(1)
print("every README number in the five-arm four-node section re-derives from logs/.")

#!/usr/bin/env python3
"""Re-derive every number the README's "Versus DeepSeek's own CX7-InfiniBand
numbers" section prints, from the two campaigns' logs/.

usage: check_readme.py            (exits nonzero on drift)

Markdown tables are hand-placed, so two failures survive proof-reading: a number
that is real but pasted into the WRONG cell (grep finds it, the table still lies),
and logs/ changing without the prose being regenerated. Each published figure is
therefore asserted as (cell, arm, op, field) -> value, recomputed through
make_upstream_table.py's own aggregation.

Ratios against upstream are checked too, because those are the section's actual
claims and each one is a division we typed by hand.

Tolerance is one unit of the last printed digit: 0.006 for a ratio printed to 3
places, 0.06 for a us printed to 0.1 or a percentage printed to 0.1, 0.006 for a
GB/s printed to 0.01.

The p6-b200 paragraph is the one thing here that is NOT log-backed -- those are
AWS's numbers and we have no b200 nodes. Its arithmetic is still checked, so a
typo'd ratio fails; its provenance is stated in the README instead.

Add a row here whenever a number is added to that section.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "gen_upstream", os.path.join(HERE, "make_upstream_table.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

FAILS = []
US, SO, WIRE, N = 0, 1, 2, 3


def chk(label, got, want, tol=0.06):
    if got is None:
        FAILS.append("%s: no data" % label)
    elif abs(got - want) > tol:
        FAILS.append("%s: README says %s, logs say %.4f" % (label, want, got))
    else:
        print("  ok  %-58s %10.3f" % (label, got))


def val(table, cell, arm, op, field):
    got = table.get(cell, {}).get(arm)
    if not got or op not in got[2]:
        return None
    return got[2][op][field]


def h(cell, op, field, arm="official"):
    return val(g.H, cell, arm, op, field)


def b(cell, op, field):
    return val(g.B, cell, "official", op, field)


# ---- upstream's rows, as the README transcribes them ----------------------
print("README -- upstream's table, transcribed verbatim")
WANT_UP = [("SM90", "CX7", "EP 8 x 2", 90, 81, 12),
           ("SM90", "CX7", "EP 8 x 4", 61, 61, 6),
           ("SM100", "CX7", "EP 8 x 2", 90, 91, 12)]
WANT_NVL = [("SM100", "N/A", "EP 8", 726, 740, "64 (Max perf)"),
            ("SM100", "N/A", "EP 8", 643, 675, "24 (Min #SM)")]
if g.UP_ROWS != WANT_UP:
    FAILS.append("README's upstream rows %s != generator's %s" % (WANT_UP, g.UP_ROWS))
elif g.UP_NVLINK != WANT_NVL:
    FAILS.append("README's NVLink rows %s != generator's %s"
                 % (WANT_NVL, g.UP_NVLINK))
else:
    print("  ok  %-58s %10s" % ("3 RDMA rows + 2 NVLink rows match", "yes"))

# ---- the head-to-head table ----------------------------------------------
# (cell, op) -> (theirs, theirs_wire, ours_SO, ours_wire, ratio, ours_us)
print("README -- SM90 head-to-head, SM matched")
HH = {
    ("2N 12 SM", "dispatch"): (90, 90.0, 81.25, 81.2, 0.903, 1502.9),
    ("2N 12 SM", "combine"): (81, 81.0, 65.50, 65.5, 0.809, 3602.5),
    ("4N  6 SM", "dispatch"): (61, 91.5, 55.00, 82.5, 0.902, 4030.7),
    ("4N  6 SM", "combine"): (61, 91.5, 52.00, 78.0, 0.852, 8268.2),
}
for (cell, op), (u, uw, so, w, ratio, t_us) in sorted(HH.items()):
    nodes = int(cell[0])
    chk("%s %s theirs wire%%" % (cell, op),
        u * (nodes - 1) / nodes / g.W_H200 * 100.0, uw)
    chk("%s %s ours SO GB/s" % (cell, op), h(cell, op, SO), so, 0.006)
    chk("%s %s ours wire%%" % (cell, op), h(cell, op, WIRE), w)
    chk("%s %s ours us" % (cell, op), h(cell, op, US), t_us)
    got = h(cell, op, SO)
    chk("%s %s ours/theirs" % (cell, op), None if got is None else got / u,
        ratio, 0.006)
# the README says the two dispatch ratios agree to within a tenth of a percent
d2 = h("2N 12 SM", "dispatch", SO) / 90.0
d4 = h("4N  6 SM", "dispatch", SO) / 61.0
chk("dispatch ratio 2N vs 4N, |delta| in pct points", abs(d2 - d4) * 100.0, 0.11, 0.006)
if not (d2 > 0.85 and d4 > 0.85):
    FAILS.append("a dispatch ratio fell below 0.85; the README calls the gap ~9%")
# the README says combine CLOSES with scale, dispatch does not
c2 = h("2N 12 SM", "combine", SO) / 81.0
c4 = h("4N  6 SM", "combine", SO) / 61.0
if not c4 > c2:
    FAILS.append("combine ratio does not improve with node count (%.3f -> %.3f), "
                 "README says 0.809 -> 0.852" % (c2, c4))
else:
    print("  ok  %-58s %10s" % ("combine ratio improves with scale", "yes"))
# the ~1.2 ms/layer claim: our 4N combine time vs the same bytes at 61 GB/s
t4 = h("4N  6 SM", "combine", US)
chk("4N combine time lost vs a 61 GB/s reference, us", t4 - t4 * c4, 1220.0, 6.0)
# the head-to-head cells are n=1 and the README says so
for cell in ("2N 12 SM", "4N  6 SM"):
    if h(cell, "dispatch", N) != 1:
        FAILS.append("%s is now n=%d; the README calls both head-to-head cells n=1"
                     % (cell, h(cell, "dispatch", N)))
# ... and that the n>=3 cells bound the spread well under the claimed gaps. Use us,
# not SO: SO is printed with a coarse floor and its cross-rep range collapses to 0.
TAGS = dict(((cell, arm), tags) for cell, _N, _s, arm, tags in g.H200_CELLS)
for cell in ("4N 12 SM", "2N 24 SM"):
    reps = g.h200.rng(g.cell(g.h200, TAGS[(cell, "official")]), "dispatch", 2)
    if not reps:
        FAILS.append("%s: no spread available" % cell)
    else:
        lo, hi = min(reps), max(reps)
        d = 100.0 * (hi - lo) / lo
        if cell == "4N 12 SM":
            # the README quotes this one as the bound on the 0.11 pp claim above
            chk("4N 12 SM dispatch cross-rep spread pct", d, 1.20)
        else:
            chk("%s dispatch cross-rep spread pct" % cell, d, min(d, 3.0), 3.0)
        if d > 3.0:
            FAILS.append("%s dispatch spreads %.2f%% across reps; README claims the "
                         "n>=3 cells sit well under the 9-15%% gaps" % (cell, d))

# ---- the reduced-combine alternative reading ------------------------------
print("README -- if upstream's combine is the reduced variant")
for cell, u, so, ratio in (("2N 12 SM", 81, 55.44, 0.684),
                           ("4N  6 SM", 61, 44.94, 0.737)):
    chk("%s reduced combine SO" % cell, h(cell, "reduced combine", SO), so, 0.006)
    chk("%s reduced combine ours/theirs" % cell,
        h(cell, "reduced combine", SO) / u, ratio, 0.006)
# the README claims the ordering survives either reading
for cell, ud, uc in (("2N 12 SM", 90, 81), ("4N  6 SM", 61, 61)):
    for op in ("combine", "reduced combine"):
        if not h(cell, op, SO) / uc < h(cell, "dispatch", SO) / ud:
            FAILS.append("%s: %s ratio is not below dispatch's; README says combine "
                         "is behind dispatch under either reading" % (cell, op))
r2, r4 = h("2N 12 SM", "reduced combine", SO) / 81.0, \
    h("4N  6 SM", "reduced combine", SO) / 61.0
if not r4 > r2:
    FAILS.append("reduced combine ratio does not narrow with node count (%.3f -> "
                 "%.3f); README says the gap narrows under either reading" % (r2, r4))
else:
    print("  ok  %-58s %10s" % ("reduced combine also narrows with scale", "yes"))

# ---- the SM axis, priced -------------------------------------------------
print("README -- the SM axis, priced")


def sm_gain(a_cell, b_cell, op):
    a, bb = h(a_cell, op, SO), h(b_cell, op, SO)
    return None if (a is None or bb is None) else 100.0 * (a - bb) / bb


for op, want in (("dispatch", 1.8), ("combine", 4.6)):
    chk("4N 12 SM vs 6 SM %s pct" % op, sm_gain("4N 12 SM", "4N  6 SM", op), want)
for op, want in (("dispatch", 51.9), ("combine", 69.0)):
    chk("2N 12 SM vs 6 SM %s pct" % op, sm_gain("2N 12 SM", "2N  6 SM", op), want)
chk("what 0.903 would have been at 6 SM", h("2N  6 SM", "dispatch", SO) / 90.0,
    0.594, 0.006)

# ---- the full p5en grid --------------------------------------------------
print("README -- every p5en cell")
GRID = {
    # cell: (n, {op: (SO, wire%, us)})
    "2N  6 SM": (1, {"dispatch": (53.50, 53.5, 2290.5),
                     "combine": (38.75, 38.8, 6052.8),
                     "reduced combine": (31.88, 31.9, 7377.7)}),
    "2N 12 SM": (1, {"dispatch": (81.25, 81.2, 1502.9),
                     "combine": (65.50, 65.5, 3602.5),
                     "reduced combine": (55.44, 55.4, 4237.9)}),
    "2N 24 SM": (3, {"dispatch": (79.65, 79.6, 1535.7),
                     "combine": (66.65, 66.6, 3534.0),
                     "reduced combine": (70.27, 70.3, 3362.6)}),
    "4N  6 SM": (1, {"dispatch": (55.00, 82.5, 4030.7),
                     "combine": (52.00, 78.0, 8268.2),
                     "reduced combine": (44.94, 67.4, 9518.8)}),
    "4N 12 SM": (3, {"dispatch": (56.00, 84.0, 3955.3),
                     "combine": (54.42, 81.6, 7842.6),
                     "reduced combine": (53.79, 80.7, 7943.2)}),
    "4N 24 SM": (2, {"dispatch": (55.83, 83.7, 3972.7),
                     "combine": (55.14, 82.7, 7710.7),
                     "reduced combine": (55.03, 82.5, 7728.3)}),
}
for cell, (n, ops) in sorted(GRID.items()):
    if h(cell, "dispatch", N) != n:
        FAILS.append("%s: n=%s on disk, README says %d"
                     % (cell, h(cell, "dispatch", N), n))
    for op, (so, w, t_us) in sorted(ops.items()):
        chk("%s %s SO" % (cell, op), h(cell, op, SO), so, 0.006)
        chk("%s %s wire%%" % (cell, op), h(cell, op, WIRE), w)
        chk("%s %s us" % (cell, op), h(cell, op, US), t_us)

print("README -- the two readings that fall out of the grid")
band = [h(c, "dispatch", WIRE) for c in ("4N  6 SM", "4N 12 SM", "4N 24 SM")]
chk("4N dispatch wire%% band lo", min(band), 82.5)
chk("4N dispatch wire%% band hi", max(band), 84.0)
if not all(81.0 <= x <= 84.5 for x in band):
    FAILS.append("4N dispatch wire%% left the 81-84%% band: %s" % band)
if not (h("2N 24 SM", "dispatch", SO) < h("2N 12 SM", "dispatch", SO)
        and h("2N 24 SM", "dispatch", US) > h("2N 12 SM", "dispatch", US)):
    FAILS.append("2N 24 SM is no longer slower than 2N 12 SM on dispatch; README "
                 "says 79.65 vs 81.25 GB/s and 1535.7 vs 1502.9 us")
else:
    print("  ok  %-58s %10s" % ("2N 24 SM slower than 2N 12 SM on dispatch", "yes"))

print("README -- the AWS cross-check and the fork control")
for op, aws, want in (("dispatch", 54.72, 0.5), ("combine", 51.84, 0.3)):
    chk("4N 6 SM %s vs AWS 2026-08-08, pct apart" % op,
        100.0 * (h("4N  6 SM", op, SO) - aws) / aws, want, 0.06)
for op, off, mn, d in (("dispatch", 81.25, 81.00, -0.3),
                       ("combine", 65.50, 65.62, 0.2),
                       ("reduced combine", 55.44, 55.31, -0.2)):
    chk("2N 12 SM %s official SO" % op, h("2N 12 SM", op, SO), off, 0.006)
    chk("2N 12 SM %s main SO" % op, h("2N 12 SM", op, SO, "main"), mn, 0.006)
    # the README quotes main RELATIVE TO official, matching the generator's column.
    chk("2N 12 SM %s main vs official pct" % op,
        100.0 * (mn - off) / off, d, 0.06)
if h("2N 12 SM", "dispatch", N, "main") != 2:
    FAILS.append("main 2N 12 SM is n=%s, README says n=2"
                 % h("2N 12 SM", "dispatch", N, "main"))

# ---- the b300 grid ------------------------------------------------------
print("README -- b300 is not a head-to-head row")
BGRID = {
    "2N 12 SM": {"dispatch": (116.19, 58.1, 1052.6),
                 "combine": (80.73, 40.4, 2904.9),
                 "reduced combine": (62.25, 31.1, 3765.2)},
    "2N 24 SM": {"dispatch": (136.56, 68.3, 894.1),
                 "combine": (127.65, 63.8, 1849.3),
                 "reduced combine": (104.04, 52.0, 2254.2)},
    "4N 12 SM": {"dispatch": (109.27, 82.0, 2024.2),
                 "combine": (94.75, 71.1, 4515.8),
                 "reduced combine": (87.90, 65.9, 4864.4)},
    "4N 24 SM": {"dispatch": (109.43, 82.1, 2022.9),
                 "combine": (107.62, 80.7, 3971.7),
                 "reduced combine": (109.22, 81.9, 3916.4)},
}
for cell, ops in sorted(BGRID.items()):
    if b(cell, "dispatch", N) != 3:
        FAILS.append("b300 %s: n=%s, README says n=3" % (cell, b(cell, "dispatch", N)))
    for op, (so, w, t_us) in sorted(ops.items()):
        chk("b300 %s %s SO" % (cell, op), b(cell, op, SO), so, 0.006)
        chk("b300 %s %s wire%%" % (cell, op), b(cell, op, WIRE), w)
        chk("b300 %s %s us" % (cell, op), b(cell, op, US), t_us)
chk("b300 ceiling GB/s", g.W_B300, 100.0, 0.0)
chk("b300 2N 24 SM dispatch vs upstream SM100 90, x",
    b("2N 24 SM", "dispatch", SO) / g.UP_SM100_2N12[0], 1.52, 0.006)
if not b("4N 12 SM", "dispatch", WIRE) > b("2N 24 SM", "dispatch", WIRE):
    FAILS.append("b300 4N wire%% no longer exceeds 2N; README says 82.0 vs 68.3")
else:
    print("  ok  %-58s %10s" % ("b300 4N dispatch wire% above 2N's", "yes"))

print("README -- the p6-b200 paragraph (AWS numbers, arithmetic only)")
for op, w, u, ratio in (("dispatch", 82.56, 90, 0.917), ("combine", 69.75, 91, 0.766)):
    chk("b200 2N 12 SM %s wire%%" % op, w, round(w, 1), 0.06)
    chk("b200 2N 12 SM %s ours/theirs" % op, w / u, ratio, 0.006)
if g.UP_SM100_2N12 != (90, 91):
    FAILS.append("the SM100 row is %s, the b200 ratios divide by (90, 91)"
                 % (g.UP_SM100_2N12,))

# ---- provenance ---------------------------------------------------------
print("README -- provenance and completeness")
for name, gm, want_dir in (("p5en", g.h200, "p5en_2n4n_20260825"),
                           ("b300", g.b300, "b300_scale_20260904")):
    if os.path.basename(os.path.dirname(os.path.abspath(gm.EPRUNS))) != want_dir:
        FAILS.append("%s logs came from %s, README names %s"
                     % (name, gm.EPRUNS, want_dir))
    if gm.EXCLUDED:
        FAILS.append("%s: reps excluded as outliers: %s"
                     % (name, sorted(set(gm.EXCLUDED))))
    if gm.EMPTY:
        FAILS.append("%s: empty logs on disk: %s" % (name, sorted(set(gm.EMPTY))))
for name, table, cells in (("p5en", g.H, g.H200_CELLS), ("b300", g.B, g.B300_CELLS)):
    for cell, nodes, _sm, arm, _tags in cells:
        got = table.get(cell, {}).get(arm)
        if not got:
            FAILS.append("%s %s %s: no data" % (name, cell, arm))
            continue
        ranks = got[2]["dispatch"][5]
        if got[2]["dispatch"][4] != 8 * nodes or ranks != 8 * nodes:
            FAILS.append("%s %s %s: %d/%d ranks, expected %d"
                         % (name, cell, arm, ranks, got[2]["dispatch"][4], 8 * nodes))
print("  ok  %-58s %10s" % ("every cell at 8 x nodes ranks, no exclusions", "yes"))

print()
if FAILS:
    print("DRIFT -- the README does not describe these logs:")
    for x in FAILS:
        print("  !! %s" % x)
    sys.exit(1)
print("every README number in the EFA-vs-CX7 section re-derives from the two "
      "campaigns' logs/.")

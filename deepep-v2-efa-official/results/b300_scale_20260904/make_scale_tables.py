#!/usr/bin/env python3
"""b300 across 1 / 2 / 4 nodes: what does adding nodes cost, and does PR #1+#2 hold?

usage: make_scale_tables.py            (logs are read from ./logs, or $EPRUNS_SCALE)

THE QUESTION. Every earlier b300 campaign under results/ is 2 nodes. Two nodes
measures the crossing cost once and cannot separate a fixed per-crossing tax from a
term that grows with rank count -- and those two have opposite implications for a
deployment. This campaign runs the SAME 7 cells at 1, 2 and 4 nodes on the same
four hosts on the same day, so the growth term is measured rather than extrapolated
from one doubling.

The 1-node cells matter as the zero of the crossing axis: at 1 node nothing leaves
the box, so 2N-vs-1N is the whole cost of going off-node (EFA + the wider all-to-all
fan-out together), and 4N-vs-2N is what a doubling costs once the code is already
paying to cross. A per-doubling factor near 1.0 is a fixed tax; near 2.0 is a term
that scales with the world.

WHAT IS COMPARABLE TO WHAT. One axis is added versus results/b300_stack_20260903 --
node count -- and it is in every log name, so a 2N cell here can never pool with a
1N or 4N one. Held fixed across every cell: GIN type 5 (NCCL_GIN_TYPE=5,
NCCL_SYM_GIN_KERNELS_ENABLE=0), --prefer-overlap-with-compute=0, --test-first-only
(so FP8 dispatch at expert_alignment=128), 8 processes per node, EP_BUFFER_DEBUG
unset, default part geometry unless the knob column says otherwise.

TWO ARMS, and neither is upstream `main`:
  official  the `deepep-v2-efa-official:sm103` image, tree 97d8f9bcc1be31e9036db2ab
            591ef9b9f4e38619 -- NOT upstream main 54fffef. The CONFIG table prints
            the BUILD_REF each arm actually ran so this cannot be taken on trust.
  prs       bfbdd15ff448783f877cb2210cb3246c8452b05e = amazon-contributing/DeepEP
            PRs #1 + #2 (#1 is a subset of #2).
`prs` therefore measures #1+#2 against 97d8f9b, not against 54fffef; the p5en
3-arm campaign is the one that measures against upstream main.

GB/s CANNOT BE COMPARED ACROSS THIS AXIS, for two separate reasons, both read out
of tests/elastic/test_ep.py rather than assumed:
  * SO GB/s is 0 at 1 node BY CONSTRUCTION, not because nothing moved. test_ep.py
    :229 iterates `range(num_scaleout_ranks if num_scaleout_ranks > 1 else 0)`, so
    with one node the scale-out send count is exactly zero. The 1N rows' SO column
    is therefore a structural zero and SU is the only meaningful rate there.
  * --ignore-local-traffic is OFF (matching every number already published under
    results/), so SO counts tokens whose destination is the sender's OWN node as
    well -- at 2N it reads ~81 GB/s against a 50 GB/s per-GPU scale-out ceiling,
    i.e. 162%. It is not a wire rate at any node count.
The printed `bytes` field is num_scaleup_bytes, the SU denominator (bytes RECEIVED
via scale-up); SO's denominator is a send-token count test_ep.py never prints. So
the MB/rank column below is labelled SU MB, and it is what SU GB/s divides by.

TIME IS THE METRIC. Both rates are printed beside it, never instead of it.

Aggregation (all-rank pooling from EVERY node's log -- combine is layered by node,
so one node's log is a sample of one layer and not of the run; mean over rotated
reps; a rep >25% off its cell's median excluded and named in AUDIT; finditer so
glued per-rank lines are not silently halved) is imported unchanged from the p5en
3-arm generator, so this table's arithmetic is every other campaign's.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
LOGS = os.environ.get("EPRUNS_SCALE", os.path.join(HERE, "logs"))


def _mod(path, name, env):
    """Load a generator as its own module object with its own log dir."""
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = _mod(os.path.join(RESULTS, "p5en_3arm_20260831", "make_3arm_tables.py"),
         "gen_scale", {"EPRUNS": LOGS})
assert os.path.abspath(g.EPRUNS) == os.path.abspath(LOGS), "wrong log dir"

f, pct, stat, OPS = g.f, g.pct, g.stat, g.OPS

OFFICIAL, PRS = "official", "prs"
ARMS = [(OFFICIAL, "official"), (PRS, "PR #1+#2")]
TOKS = ((8192, "PREFILL"), (128, "DECODE"))
# The node counts run_campaign.sh was pointed at, in order. Every table walks this
# list, so a campaign that only got 1N and 2N in prints a two-column scaling table
# rather than a wrong one.
NODES = sorted({int(x) for x in os.environ.get("NODE_LIST", "1 2 4").split()})
SM_OP = 12          # the operating point: run_test_ep.sh's default, both arms
SM_AXIS = 24        # carried on the `official` arm only
DFLT = "qpdefault"          # official's default geometry
PRSD = "prsdflt"            # prs's default geometry (its own tag, same meaning)
MTPP1 = "prsmtpp1"          # EP_MIN_TOKENS_PER_PART=1: PR #2's clamp turned OFF
KNOB = {OFFICIAL: DFLT, PRS: PRSD}
TAG = "%s_%dN_%dsm_%dtok_%s_nodbg_gin5_ovlp0_rep%d"

CACHE = {}


def cell(arm, nodes, tok, sm=SM_OP, knob=None):
    knob = knob or KNOB[arm]
    key = (arm, nodes, tok, sm, knob)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = g.load(TAG % (arm, nodes, sm, tok, knob, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    return CACHE[key]


def us(arm, nodes, tok, op, sm=SM_OP, knob=None):
    return stat(cell(arm, nodes, tok, sm, knob), op)[0]


def so(arm, nodes, tok, op, sm=SM_OP, knob=None):
    """all-rank-mean SO GB/s -- 0 at 1 node by construction (see header). Integer
    per rank in the log, so it cannot resolve a sub-1% change in time."""
    return stat(cell(arm, nodes, tok, sm, knob), op, 0)[0]


def su(arm, nodes, tok, op, sm=SM_OP, knob=None):
    """all-rank-mean SU GB/s -- the rate that is non-zero at every node count."""
    return stat(cell(arm, nodes, tok, sm, knob), op, 1)[0]


def mb(arm, nodes, tok, op, sm=SM_OP, knob=None):
    """all-rank-mean SU MB/rank -- the byte denominator SU divides by, printed so
    that a moved GB/s can be attributed to bytes or to time rather than guessed."""
    v = stat(cell(arm, nodes, tok, sm, knob), op, 3)[0]
    return None if v is None else v / 1e6


def layer(arm, nodes, tok, sm=SM_OP, knob=None):
    """dispatch + reduced combine: the two calls one MoE layer actually makes."""
    d = us(arm, nodes, tok, "dispatch", sm, knob)
    c = us(arm, nodes, tok, "reduced combine", sm, knob)
    return None if (d is None or c is None) else d + c


# ---------------------------------------------------------------- tables
def config_table():
    """Read the config out of the logs. The SM count especially: test_ep.py CLAMPS
    --num-sms to what the QP budget allows (64 requested came back as 55 on the
    2026-08 campaign), so a tag saying 24sm is a request, not a measurement. And
    `world` is the check that a tag saying 4N really ran 32 ranks."""
    print("CONFIG -- read out of the logs, not assumed")
    print()
    print("  %-9s %-4s %-6s %-42s %-6s %-9s %s"
          % ("arm", "N", "tag SM", "BUILD_REF", "#SM", "#QPs", "world"))
    bad = []
    for arm, label in ARMS:
        for n in NODES:
            for sm in (SM_OP, SM_AXIS):
                reps = []
                for tok, _ in TOKS:
                    reps += cell(arm, n, tok, sm)
                if not reps:
                    continue
                refs, cfgs, world = set(), set(), set()
                for _rep, (_o, W, _bn, c, r, _t) in reps:
                    refs |= set(r)
                    cfgs |= set(c)
                    world.add(W)
                sms = sorted({c[0] for c in cfgs})
                print("  %-9s %-4d %-6d %-42s %-6s %-9s %s"
                      % (label, n, sm, ",".join(sorted(refs)) or "-",
                         ",".join(str(s) for s in sms),
                         ",".join(sorted({"%d/%d" % (c[1], c[2]) for c in cfgs})),
                         ",".join(str(w) for w in sorted(world))))
                if sms != [sm]:
                    bad.append("%s %dN @%dsm: log says #SM %s"
                               % (label, n, sm, sms))
                if world != {8 * n}:
                    bad.append("%s %dN @%dsm: world is %s, expected %d"
                               % (label, n, sm,
                                  ",".join(str(w) for w in sorted(world)), 8 * n))
    print()
    if bad:
        print("  !! THE TAG DOES NOT MATCH THE LOG for these cells; a delta computed")
        print("     over them is not the delta its heading claims:")
        for b in bad:
            print("       %s" % b)
    else:
        print("  every cell's measured #SM and world size equal what its tag claims.")
    print()


def perf_table(nodes, tok, what):
    print("%s -- %d tok, %d node%s / %d ranks, %d SM, default geometry, GIN type 5, ovlp=0"
          % (what, tok, nodes, "" if nodes == 1 else "s", 8 * nodes, SM_OP))
    print("  `d vs official` is within this node count. Neither rate is a wire rate and")
    print("  neither is comparable across node counts (see BYTES); the us column is the")
    print("  metric. SO is a structural 0 at 1 node -- see the header.")
    print()
    hdr = ("  %-18s %-10s %10s %12s %8s %8s %9s  %s"
           % ("op", "arm", "us", "d vs offic.", "SO GB/s", "SU GB/s", "SU MB/rank",
              "per-rep us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        printed = False
        for arm, label in ARMS:
            reps = cell(arm, nodes, tok)
            mean, per_rep, short = stat(reps, op)
            if mean is None:
                continue
            base = us(OFFICIAL, nodes, tok, op)
            print("  %-18s %-10s %10s %12s %8s %8s %9s  %s%s"
                  % (op if not printed else "", label, f(mean),
                     "" if arm == OFFICIAL else pct(mean, base),
                     f(so(arm, nodes, tok, op), 0),
                     f(su(arm, nodes, tok, op), 0),
                     f(mb(arm, nodes, tok, op)),
                     " / ".join(f(v) for v in per_rep),
                     ("  !! " + "; ".join(short)) if short else ""))
            printed = True
        if printed:
            print()
    for arm, label in ARMS:
        t = layer(arm, nodes, tok)
        if t is not None:
            print("  %-18s %-10s %10s %12s"
                  % ("layer total" if arm == ARMS[0][0] else "", label, f(t),
                     "" if arm == OFFICIAL else pct(t, layer(OFFICIAL, nodes, tok))))
    print()


def scaling(tok, what):
    """The headline. Per arm and op: absolute us at each node count, plus what each
    DOUBLING costs. `x per 2x` near 1.0 = a fixed per-crossing tax; near 2.0 = a
    term that grows with the world size."""
    print("SCALING -- %s (%d tok), %d SM: what adding nodes costs" % (what, tok, SM_OP))
    print("  x1N is the ratio to the single-node cell (all traffic local, the zero of")
    print("  the crossing axis). `x per 2x` is each doubling separately: 2N/1N is the")
    print("  cost of leaving the box at all, 4N/2N the cost of a doubling once")
    print("  crossing is already paid for. Those two being different is the point.")
    print()
    cols = "".join("%10s" % ("%dN us" % n) for n in NODES)
    print("  %-18s %-10s%s  %-20s %s" % ("op", "arm", cols, "x vs 1N", "x per 2x"))
    for op in OPS:
        printed = False
        for arm, label in ARMS:
            vals = [us(arm, n, tok, op) for n in NODES]
            if all(v is None for v in vals):
                continue
            base = vals[0] if NODES[0] == 1 else None
            steps = []
            for i in range(1, len(NODES)):
                a, b = vals[i - 1], vals[i]
                if a and b and NODES[i] == 2 * NODES[i - 1]:
                    steps.append("%dN->%dN %.2fx" % (NODES[i - 1], NODES[i], b / a))
            ratio = ("  ".join("%dN %.2fx" % (n, v / base)
                               for n, v in zip(NODES, vals) if v and base and n != 1)
                     if base else "-")
            print("  %-18s %-10s%s  %-20s %s"
                  % (op if not printed else "", label,
                     "".join("%10s" % f(v) for v in vals), ratio,
                     "  ".join(steps) or "-"))
            printed = True
        if printed:
            print()
    for arm, label in ARMS:
        vals = [layer(arm, n, tok) for n in NODES]
        if all(v is None for v in vals):
            continue
        steps = ["%dN->%dN %.2fx" % (NODES[i - 1], NODES[i],
                                     vals[i] / vals[i - 1])
                 for i in range(1, len(NODES))
                 if vals[i] and vals[i - 1] and NODES[i] == 2 * NODES[i - 1]]
        print("  %-18s %-10s%s  %-20s %s"
              % ("layer total", label, "".join("%10s" % f(v) for v in vals), "",
                 "  ".join(steps) or "-"))
    print()


def pr_effect(tok, what):
    """Does #1+#2's win survive scale? The 2026-08 p5en campaign found the clamp's
    win collapsing at 4 nodes, which is the one thing 2 nodes cannot see."""
    print("PR #1+#2 vs official ACROSS SCALE -- %s (%d tok), %d SM"
          % (what, tok, SM_OP))
    print("  One row per op; one column per node count. A percentage that shrinks")
    print("  left-to-right is a win that does not survive scale-out.")
    print()
    cols = "".join("%22s" % ("%dN (offic. -> prs)" % n) for n in NODES)
    print("  %-18s%s" % ("op", cols))
    for op in OPS + ["layer total"]:
        cells = []
        any_row = False
        for n in NODES:
            if op == "layer total":
                a, b = layer(OFFICIAL, n, tok), layer(PRS, n, tok)
            else:
                a, b = us(OFFICIAL, n, tok, op), us(PRS, n, tok, op)
            if a is None or b is None:
                cells.append("%22s" % "-")
                continue
            any_row = True
            cells.append("%22s" % ("%s->%s %s" % (f(a, 0), f(b, 0), pct(b, a))))
        if any_row:
            print("  %-18s%s" % (op, "".join(cells)))
    print()


def clamp_control(tok=128):
    """EP_MIN_TOKENS_PER_PART=1 short-circuits PR #2 to the pre-patch geometry, so
    it is the OLD behaviour inside the NEW binary. It must land on `official`; if
    it does not, the arm difference came from the build or the environment rather
    than from the clamp."""
    print("CLAMP CONTROL -- %d tok, %d SM: prs with EP_MIN_TOKENS_PER_PART=1" % (tok, SM_OP))
    print("  `d vs official` should be ~0: same geometry, different binary. A large")
    print("  value there invalidates the PR #1+#2 column above.")
    print()
    print("  %-18s %-4s %10s %10s %10s %10s %10s"
          % ("op", "N", "official", "prs", "prs mtpp1", "d clamp", "d vs offic."))
    for n in NODES:
        rows = 0
        for op in OPS:
            o = us(OFFICIAL, n, tok, op)
            p = us(PRS, n, tok, op)
            c = us(PRS, n, tok, op, knob=MTPP1)
            if c is None:
                continue
            rows += 1
            print("  %-18s %-4d %10s %10s %10s %10s %10s"
                  % (op, n, f(o), f(p), f(c),
                     pct(p, c) if p is not None else "-",
                     pct(c, o) if o is not None else "-"))
        if rows:
            print()
    if not any(cell(PRS, n, tok, knob=MTPP1) for n in NODES):
        print("  -- the control cell was not run at any node count.")
        print()


def sm_axis(tok, what):
    """24 vs 12 SM on the `official` arm, at every node count. The 2N-only campaign
    could not say whether the SM trade moves with scale."""
    print("SM AXIS -- %s (%d tok), official arm: %d vs %d SM at each node count"
          % (what, tok, SM_AXIS, SM_OP))
    print()
    print("  %-18s %-4s %10s %10s %10s   %s"
          % ("op", "N", "%dsm us" % SM_OP, "%dsm us" % SM_AXIS, "d",
             "SU %d->%d" % (SM_OP, SM_AXIS)))
    for n in NODES:
        rows = 0
        for op in OPS:
            a = us(OFFICIAL, n, tok, op, SM_OP)
            b = us(OFFICIAL, n, tok, op, SM_AXIS)
            if a is None or b is None:
                continue
            rows += 1
            print("  %-18s %-4d %10s %10s %10s   %s -> %s"
                  % (op, n, f(a), f(b), pct(b, a),
                     f(su(OFFICIAL, n, tok, op, SM_OP), 0),
                     f(su(OFFICIAL, n, tok, op, SM_AXIS), 0)))
        a, b = layer(OFFICIAL, n, tok, SM_OP), layer(OFFICIAL, n, tok, SM_AXIS)
        if a is not None and b is not None:
            print("  %-18s %-4d %10s %10s %10s" % ("layer total", n, f(a), f(b),
                                                   pct(b, a)))
            rows += 1
        if rows:
            print()


def node_layering(nodes, tok):
    """combine is layered BY NODE: the per-node means differ by 12-23% and which
    node is slow flips between runs. So a table built from one node's log is a
    sample of one layer, not of the run. This prints the layers, which is why the
    pooling above is auditable rather than asserted."""
    if nodes < 2:
        return
    print("NODE LAYERING -- %d nodes, %d tok, %d SM: per-node mean us, official arm"
          % (nodes, tok, SM_OP))
    print("  spread = (max-min)/min across nodes WITHIN one rep. If it is large, a")
    print("  single-node log could not have produced the numbers above.")
    print()
    print("  %-18s %-6s %s" % ("op", "rep", "per-node mean us (node1, node2, ...)"))
    for op in ("dispatch", "combine", "reduced combine"):
        for rep, (_o, _W, bn, _c, _r, _t) in cell(OFFICIAL, nodes, tok):
            vals = []
            for node in sorted(bn, key=int):
                xs = bn[node].get(op)
                if xs:
                    vals.append(sum(xs) / len(xs))
            if len(vals) < 2:
                continue
            spread = 100.0 * (max(vals) - min(vals)) / min(vals)
            print("  %-18s %-6d %s   spread %.1f%%"
                  % (op, rep, "  ".join(f(v) for v in vals), spread))
        print()


def bytes_table():
    """The byte denominator itself, per node count. This is the table that makes
    the 'GB/s is not comparable across the node axis' statement checkable instead
    of a caveat: if MB/rank moves between node counts, so does GB/s for reasons
    that have nothing to do with time."""
    print("BYTES -- SU MB/rank per node count (official arm, %d SM), the denominator"
          % SM_OP)
    print("  SU GB/s divides by: num_scaleup_bytes, i.e. bytes RECEIVED via scale-up.")
    print("  --ignore-local-traffic is OFF, so nothing is subtracted from it. A GB/s")
    print("  ratio across two columns here is a bytes ratio as much as a time ratio.")
    print()
    for tok, what in TOKS:
        print("  %s (%d tok)" % (what, tok))
        print("    %-18s%s" % ("op", "".join("%12s" % ("%dN SU MB" % n) for n in NODES)))
        for op in OPS:
            vals = [mb(OFFICIAL, n, tok, op) for n in NODES]
            if all(v is None for v in vals):
                continue
            print("    %-18s%s" % (op, "".join("%12s" % f(v) for v in vals)))
        print()


def audit():
    print("AUDIT -- reps on disk and rank completeness (n must equal 8 x nodes)")
    print()
    for tok, _ in TOKS:
        for n in NODES:
            for arm, label in ARMS:
                for sm in (SM_OP, SM_AXIS):
                    for knob in (KNOB[arm], MTPP1):
                        if knob == MTPP1 and arm != PRS:
                            continue
                        reps = cell(arm, n, tok, sm, knob)
                        if not reps:
                            continue
                        print("  %5dtok %-10s %dN %2dsm %-9s %s"
                              % (tok, label, n, sm, knob,
                                 " ".join("rep%d:%d/%d"
                                          % (rep, len(o.get("dispatch", {})), W)
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
    print("p6-b300.48xlarge x %s (sm_103), EFA 1.50.0 + efa.ko 3.3.0, GIN type 5"
          % "/".join(str(n) for n in NODES))
    print("EPRUNS=%s" % g.EPRUNS)
    print()
    config_table()
    for n in NODES:
        for tok, what in TOKS:
            perf_table(n, tok, what)
    for tok, what in TOKS:
        scaling(tok, what)
    for tok, what in TOKS:
        pr_effect(tok, what)
    clamp_control()
    for tok, what in TOKS:
        sm_axis(tok, what)
    for n in NODES:
        for tok, _what in TOKS:
            node_layering(n, tok)
    bytes_table()
    audit()

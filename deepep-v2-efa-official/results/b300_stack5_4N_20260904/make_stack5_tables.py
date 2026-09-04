#!/usr/bin/env python3
"""Five arms at FOUR nodes: do the PRs still pay at 32 ranks, and what did upstream
main gain between 54fffef and 97d8f9b?

usage: make_stack5_tables.py            (logs are read from ./logs, or $EPRUNS_ST5)

THE TWO QUESTIONS.
1. results/b300_stack_20260903 measured main / #1+#2 / #8+#9 / stack at TWO nodes.
   results/b300_scale_20260904 then showed the rank-scaling term is what erodes the
   PRs (#1+#2's decode-dispatch win shrank -55.0% -> -42.5% from 2 to 4 nodes,
   because the patched arm grows 1.40x per node-doubling against official's 1.09x).
   So the four-arm comparison has to be re-run at four nodes; a 2-node ranking
   cannot be extrapolated.
2. `official` is 97d8f9bcc1be31e9036db2ab591ef9b9f4e38619, the LATEST upstream
   main; `main54fffef` is 54fffeff810723f574c574b1790dff189f3c6ffb, the previous
   main and the merge base of all four PRs. Both are in this campaign, so the
   MAIN ADVANCE section measures what upstream gained on its own -- a number no
   previous campaign could produce, and the thing that decides whether the PRs
   are still worth what they were measured to be worth against 54fffef.

WHAT IS COMPARABLE TO WHAT. Every cell is 4 nodes / 32 ranks, 12 SM, default part
geometry, GIN type 5, --prefer-overlap-with-compute=0, --test-first-only (so FP8
dispatch at expert_alignment=128), one arm label per tree, 8 processes per node.
The only axes are arm and token count, and both are in every log name.

NCCL_NVLS_ENABLE=0 IS ON EVERY CELL, and it is in the knob tag (`qpdefault-nvls0`)
so these logs can never pool with an NVLS-on campaign. It is not a tuning choice:
the NVLink-multicast team on these hosts wedges (fabricmanager: "All GPUs in the
partition need to be reset to recover") and NCCL then dies in init_process_group
with CUDA error 401. NVLS is used by torch's NCCL process group, not by DeepEP's
dispatch/combine, which go through GIN/EFA -- the NVLS CONTROL section tests that
claim for free against results/b300_scale_20260904's `official` arm, same hosts,
same image, same 口径, NVLS on.

REPS=2, SO THERE IS NO OUTLIER FILTER. stat() drops a rep more than 25% off its
cell's median, and a median over two replicates is their mean -- so with n=2 the
filter can only fire on a pair that disagrees by ~2x. Per-rep values are printed
in every row for exactly this reason: read them, do not trust the mean alone.

TIME IS THE METRIC. --ignore-local-traffic is off, so the SO column counts tokens
destined for the sender's own node and is not a wire rate. Aggregation (all-rank
pooling from every node's log, mean over reps) is imported from the p5en 3-arm
generator, unchanged, so the arithmetic matches every other campaign's.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
LOGS = os.environ.get("EPRUNS_ST5", os.path.join(HERE, "logs"))
NVLS_ON = os.path.join(RESULTS, "b300_scale_20260904", "logs")
P5EN = os.path.join(RESULTS, "p5en_3arm_20260831", "make_3arm_tables.py")


def _mod(name, logs):
    """Load the shared generator as its own module with its own log dir.

    It reads EPRUNS at import time, so two loads give two independent modules over
    two campaigns. A plain `import` would reuse the first and read the wrong logs.
    """
    os.environ["EPRUNS"] = logs
    spec = importlib.util.spec_from_file_location(name, P5EN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert os.path.abspath(m.EPRUNS) == os.path.abspath(logs), "wrong log dir"
    return m


g = _mod("gen_st5", LOGS)
on = _mod("gen_nvls_on", NVLS_ON)
assert g.EPRUNS != on.EPRUNS, "this campaign collapsed onto the NVLS-on log dir"

f, pct, stat, rng, OPS = g.f, g.pct, g.stat, g.rng, g.OPS

OFFICIAL = "official"          # 97d8f9b -- latest upstream main
MAIN = "main54fffef"           # 54fffef -- previous main, merge base of the PRs
PR12 = "pr12bfbdd15"           # amazon-contributing/DeepEP #1 + #2
PR89 = "pr893c737dc"           # #8 + #9
STACK = "stacka35285f"         # #1+#2 merged with #8+#9
ARMS = [(OFFICIAL, "official"), (MAIN, "main"), (PR12, "PR #1+#2"),
        (PR89, "PR #8+#9"), (STACK, "stack")]
PR_ARMS = [(PR12, "PR #1+#2"), (PR89, "PR #8+#9"), (STACK, "stack")]
TOKS = ((8192, "PREFILL"), (128, "DECODE"))
NODES, SM, KNOB = 4, 12, "qpdefault-nvls0"
TAG = "%s_%dN_%dsm_%dtok_%s_nodbg_gin5_ovlp0_rep%d"

CACHE = {}


def cell(arm, tok, only=None):
    key = (arm, tok)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = g.load(TAG % (arm, NODES, SM, tok, KNOB, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    reps = CACHE[key]
    return reps if only is None else [x for x in reps if x[0] in only]


def us(arm, tok, op, only=None):
    return stat(cell(arm, tok, only), op)[0]


def so(arm, tok, op, only=None):
    """all-rank-mean SO GB/s. Integer per rank in the log, so it cannot resolve a
    sub-1% change in time; the us column stays the primary number."""
    return stat(cell(arm, tok, only), op, 0)[0]


def layer(arm, tok, only=None):
    d, c = us(arm, tok, "dispatch", only), us(arm, tok, "reduced combine", only)
    return None if (d is None or c is None) else d + c


def shared(tok, arms):
    """reps present in EVERY arm compared, so a cross-arm delta is same-rep"""
    sets = []
    for arm, _l in arms:
        reps = {rep for rep, _ in cell(arm, tok)}
        if not reps:
            return None
        sets.append(reps)
    return set.intersection(*sets) if sets else None


def config_table():
    """Read the config out of the logs. The SM count especially: test_ep.py CLAMPS
    --num-sms to what the QP budget allows, so a tag saying 12sm is a request."""
    print("CONFIG -- read out of the logs, not assumed")
    print()
    print("  %-9s %-42s %-6s %-8s %s"
          % ("arm", "BUILD_REF", "#SM", "#QPs", "world"))
    bad = []
    for arm, label in ARMS:
        reps = cell(arm, 128) + cell(arm, 8192)
        if not reps:
            print("  %-9s -- not run" % label)
            bad.append("%s: no logs" % label)
            continue
        refs, cfgs, world = set(), set(), set()
        for _rep, (_o, W, _bn, c, r, _t) in reps:
            refs |= set(r)
            cfgs |= set(c)
            world.add(W)
        sms = sorted({c[0] for c in cfgs})
        print("  %-9s %-42s %-6s %-8s %s"
              % (label, ",".join(sorted(refs)) or "-",
                 ",".join(str(s) for s in sms),
                 ",".join(sorted({"%d/%d" % (c[1], c[2]) for c in cfgs})),
                 ",".join(str(w) for w in sorted(world))))
        if sms != [SM]:
            bad.append("%s: log says #SM %s, tag says %d" % (label, sms, SM))
        if sorted(world) != [8 * NODES]:
            bad.append("%s: world %s, expected %d" % (label, sorted(world),
                                                      8 * NODES))
        if len(refs) != 1:
            bad.append("%s: %d BUILD_REFs in one arm -- %s"
                       % (label, len(refs), sorted(refs)))
    print()
    if bad:
        print("  !! CONFIG DOES NOT MATCH THE TAGS -- the table below is not what it")
        print("     says it is:")
        for b in bad:
            print("       %s" % b)
    else:
        print("  every arm: one BUILD_REF, measured #SM == %d, world == %d."
              % (SM, 8 * NODES))
    print()


def perf_table(tok, what):
    print("%s -- %d tok, %d nodes / %d ranks, %d SM, GIN type 5, ovlp=0, NVLS off"
          % (what, tok, NODES, 8 * NODES, SM))
    print("  `d vs main` is against main54fffef, the merge base of all four PRs.")
    print("  `d vs off` is against official (97d8f9b, latest upstream main).")
    print()
    hdr = ("  %-18s %-9s %10s %10s %10s %-11s  %s"
           % ("op", "arm", "us", "d vs main", "d vs off", "SO GB/s", "per-rep us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        printed = False
        for arm, label in ARMS:
            reps = cell(arm, tok)
            mean, per_rep, short = stat(reps, op)
            if mean is None:
                continue
            base, offb = us(MAIN, tok, op), us(OFFICIAL, tok, op)
            lo, hi = rng(reps, op, 0)
            print("  %-18s %-9s %10s %10s %10s %-11s  %s%s"
                  % (op if not printed else "", label, f(mean),
                     "" if arm == MAIN else pct(mean, base),
                     "" if arm == OFFICIAL else pct(mean, offb),
                     "-" if lo is None else "%d-%d" % (lo, hi),
                     " / ".join(f(v) for v in per_rep),
                     ("  !! " + "; ".join(short)) if short else ""))
            printed = True
        if printed:
            print()
    print("  layer total (dispatch + reduced combine):")
    for arm, label in ARMS:
        v = layer(arm, tok)
        if v is None:
            continue
        print("  %-18s %-9s %10s %10s %10s"
              % ("", label, f(v),
                 "" if arm == MAIN else pct(v, layer(MAIN, tok)),
                 "" if arm == OFFICIAL else pct(v, layer(OFFICIAL, tok))))
    print()


def main_advance():
    """54fffef -> 97d8f9b: what upstream main gained on its own.

    This is the number that decides how to read the PR deltas. The PRs branch from
    54fffef, so their measured wins are against 54fffef. If main has since moved in
    the same direction, part of each PR's win is already upstream.
    """
    print("MAIN ADVANCE -- 54fffef -> 97d8f9b, measured for the first time")
    print("  negative d = the newer main is faster.")
    print()
    print("  %-18s %6s %12s %12s %8s   %s"
          % ("op", "tok", "54fffef", "97d8f9b", "d", "per-rep us (old | new)"))
    print("  " + "-" * 84)
    for tok, _what in TOKS:
        for op in OPS:
            a, a_pr, _ = stat(cell(MAIN, tok), op)
            b, b_pr, _ = stat(cell(OFFICIAL, tok), op)
            if a is None or b is None:
                continue
            print("  %-18s %6d %12s %12s %8s   %s | %s"
                  % (op, tok, f(a), f(b), pct(b, a),
                     " ".join(f(v) for v in a_pr), " ".join(f(v) for v in b_pr)))
        a, b = layer(MAIN, tok), layer(OFFICIAL, tok)
        if a is not None and b is not None:
            print("  %-18s %6d %12s %12s %8s" % ("layer total", tok, f(a), f(b),
                                                 pct(b, a)))
        print()


def pr_effect(tok, what):
    """each PR arm against the merge base, at 4 nodes"""
    print("PR EFFECT -- %s (%d tok), each arm vs main54fffef at %d nodes"
          % (what, tok, NODES))
    print()
    print("  %-18s %9s %17s %17s %17s"
          % ("op", "main", "#1+#2", "#8+#9", "stack"))
    for op in OPS + ["layer total"]:
        def val(arm):
            return layer(arm, tok) if op == "layer total" else us(arm, tok, op)
        m = val(MAIN)
        if m is None:
            continue
        row = []
        for arm, _l in PR_ARMS:
            v = val(arm)
            row.append("-" if v is None else "%s (%s)" % (f(v), pct(v, m)))
        print("  %-18s %9s %17s %17s %17s" % (op, f(m), row[0], row[1], row[2]))
    print()


def additivity(tok, what):
    """measured stack vs the sum of the two arms' separate savings"""
    print("ADDITIVITY -- %s (%d tok) at %d nodes" % (what, tok, NODES))
    print("  expected = main + (#1+#2 - main) + (#8+#9 - main). residual > 0 means")
    print("  the two PRs overlap; < 0 means they compound.")
    print()
    print("  %-18s %9s %9s %9s %9s %9s %8s"
          % ("op", "main", "#1+#2", "#8+#9", "expected", "stack", "residual"))
    only = shared(tok, [(MAIN, ""), (PR12, ""), (PR89, ""), (STACK, "")])
    if only is not None and not only:
        print("  -- not computable: the four arms share no rep at this cell")
        print()
        return
    for op in OPS + ["layer total"]:
        def val(arm):
            return (layer(arm, tok, only) if op == "layer total"
                    else us(arm, tok, op, only))
        m, a, b, s = val(MAIN), val(PR12), val(PR89), val(STACK)
        if None in (m, a, b, s):
            continue
        exp = m + (a - m) + (b - m)
        res = s - exp
        best = max(abs(a - m), abs(b - m))
        print("  %-18s %9s %9s %9s %9s %9s %8s  (%s of the larger single win)"
              % (op, f(m), f(a), f(b), f(exp), f(s), f(res),
                 "-" if not best else "%.0f%%" % (100.0 * res / best)))
    print("  reps: %s (the set all four arms share)"
          % ",".join(str(r) for r in sorted(only)) if only else "  reps: all")
    print()


def nvls_control():
    """Is NCCL_NVLS_ENABLE=0 measurement-neutral?

    The `official` arm at 4 nodes / 12 SM / {8192,128} tok was measured with NVLS ON
    in results/b300_scale_20260904 -- same hosts, same image, same 口径. So the flag
    is testable at zero GPU cost. Any delta here is the flag plus day-to-day drift;
    the on-side per-rep spread (n=3) is the scale to read it against.
    """
    print("NVLS CONTROL -- official arm, NVLS on (b300_scale_20260904) vs off (here)")
    print("  NVLS is used by torch's NCCL process group, not by DeepEP dispatch/")
    print("  combine, so this should be ~0. A delta larger than the on-side spread")
    print("  would mean every number in this campaign carries it.")
    print()
    ontag = "official_%dN_%dsm_%dtok_qpdefault_nodbg_gin5_ovlp0_rep%d"
    print("  %-18s %6s %10s %10s %8s   %s"
          % ("op", "tok", "NVLS on", "NVLS off", "d", "on per-rep (spread) | off"))
    print("  " + "-" * 92)
    worst = (0.0, "")
    for tok, _what in TOKS:
        on_reps = []
        for rep in range(1, 9):
            r = on.load(ontag % (NODES, SM, tok, rep))
            if r:
                on_reps.append((rep, r))
        for op in OPS:
            a, a_pr, _ = on.stat(on_reps, op)
            b, b_pr, _ = stat(cell(OFFICIAL, tok), op)
            if a is None or b is None:
                continue
            d = 100.0 * (b - a) / a
            if abs(d) > abs(worst[0]):
                worst = (d, "%s %dtok" % (op, tok))
            spread = (100.0 * (max(a_pr) - min(a_pr)) / min(a_pr)) if a_pr else 0.0
            print("  %-18s %6d %10s %10s %7.1f%%   %s (%.1f%%) | %s"
                  % (op, tok, f(a), f(b), d,
                     " ".join(f(v) for v in a_pr), spread,
                     " ".join(f(v) for v in b_pr)))
        print("    on n=%d, off n=%d" % (len(on_reps), len(cell(OFFICIAL, tok))))
        print()
    print("  largest |delta| over all ops and both token counts: %.1f%% on %s"
          % (worst[0], worst[1]))
    print()


def node_layering(tok, what):
    """combine splits by machine, and the slow node is what sets the op's time.

    Printed per rep, not averaged: the point is whether the SAME node is slow every
    time (a topology/placement fact) or whether it moves (a scheduling artifact).
    """
    print("NODE LAYERING -- %s (%d tok), per-node mean us across %d nodes"
          % (what, tok, NODES))
    print()
    for op in ("combine", "reduced combine"):
        print("  %s:" % op)
        print("    %-9s %4s %s" % ("arm", "rep", "node1..node%d (spread, slowest)"
                                   % NODES))
        for arm, label in ARMS:
            for rep, (_o, _W, bn, _c, _r, _t) in cell(arm, tok):
                vals, nodes = [], sorted(bn, key=int)
                for nd in nodes:
                    if op in bn[nd]:
                        vals.append(sum(bn[nd][op]) / len(bn[nd][op]))
                if len(vals) < 2:
                    continue
                spread = 100.0 * (max(vals) - min(vals)) / min(vals)
                print("    %-9s %4d %s  (%.1f%%, node%s)"
                      % (label, rep, " ".join(f(v) for v in vals), spread,
                         nodes[vals.index(max(vals))]))
        print()


def audit():
    print("AUDIT -- reps on disk and rank completeness (n must equal %d)"
          % (8 * NODES))
    print()
    for tok, _ in TOKS:
        for arm, label in ARMS:
            reps = cell(arm, tok)
            if not reps:
                print("  %5dtok %-9s -- not run" % (tok, label))
                continue
            print("  %5dtok %-9s %s"
                  % (tok, label,
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
        print("  no rep was excluded as an outlier -- but with n=2 the filter can")
        print("  only fire on a pair that disagrees by ~2x. Read the per-rep columns.")
        print()


if __name__ == "__main__":
    print("4 x p6-b300.48xlarge (sm_103), EFA 1.50.0 + efa.ko 3.3.0, GIN type 5")
    print("EPRUNS=%s" % g.EPRUNS)
    print("NVLS-on reference campaign: %s" % on.EPRUNS)
    print()
    config_table()
    for tok, what in TOKS:
        perf_table(tok, what)
    main_advance()
    for tok, what in TOKS:
        pr_effect(tok, what)
    for tok, what in TOKS:
        additivity(tok, what)
    nvls_control()
    for tok, what in TOKS:
        node_layering(tok, what)
    audit()

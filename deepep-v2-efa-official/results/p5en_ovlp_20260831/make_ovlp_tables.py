#!/usr/bin/env python3
"""The --prefer-overlap-with-compute bracket: which of PR #8+#9's four changes pays.

usage: [EPRUNS=./logs] make_ovlp_tables.py

Two trees, 2x p5en.48xlarge (H200 sm_90, EFA installer 1.50.0, efa.ko 3.3.0g),
12 SM, GIN type 5, both token sizes, 3 rotated reps:

  main   54fffeff810723f574c574b1790dff189f3c6ffb   `main` as of 2026-08-31
  pr89   3c737dcf0da5889ba7efd26e05b4808307cc38af   PRs #8 + #9 (#8 subset of #9)

WHY THE FLAG IS THE INSTRUMENT.  PR #8+#9 changes four things at once. Two of them
-- the num_channels_per_sm <= 4 clamp removal and the forward-warp pairing -- are
guarded by `not prefer_overlap_with_compute`, so running with the flag ON disables
exactly those two while leaving the other two live (kDefaultGinContextCnt 11 -> 13,
which the log confirms as `#QPs: 13/13`, and remote-first scheduling). That makes a
main-vs-#9 pair at =1 a measurement of the OTHER TWO alone, with no rebuild.

WHAT THIS IS AND IS NOT.  =1 also changes double-buffering and warp counts, so it
is a different kernel configuration, not the same one with two patches removed.
Therefore:
  * arms are only ever compared WITHIN one overlap value; the two values are never
    pooled and no =0-vs-=1 delta is presented as a patch effect;
  * the bracket line (D0 - D1) assumes the QP bump and remote-first scheduling are
    worth about the same in both configurations. That assumption is not tested
    here, which is why the output says "bracket" and not "attribution". A clean
    attribution needs a build with the two guarded hunks reverted.

Aggregation is not reimplemented: this script imports make_3arm_tables.py from the
sibling campaign and uses its load / stat / rng, so both campaigns pool ranks,
average reps and exclude outliers by the same code. A run's value for an op = mean
over all 16 ranks pooled from BOTH nodes' logs (combine is layered by node and the
slow node flips between runs); an arm's value = mean over reps of those means, with
a rep further than 25% from its cell's median excluded and printed in the AUDIT.

TIME IS THE METRIC. --ignore-local-traffic is OFF, matching every other number
under results/, so SO is not a wire rate and is printed only as a min-max across
ranks beside the us column.
"""
import importlib.util
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
# Assigned, not setdefault: an EPRUNS inherited from the shell must not silently
# redirect the sibling module's loader at the logs of a different campaign.
os.environ["EPRUNS"] = os.environ.get("EPRUNS_OVLP", os.path.join(HERE, "logs"))
_sib = os.path.join(os.path.dirname(HERE), "p5en_3arm_20260831", "make_3arm_tables.py")
_spec = importlib.util.spec_from_file_location("gen3arm", _sib)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)          # import-safe: it has an __main__ guard

f, pct, stat, rng, OPS = g.f, g.pct, g.stat, g.rng, g.OPS

ARMS = [("main54fffef", "main", "54fffef"),
        ("pr893c737dc", "PR #8+#9", "3c737dc")]
OVLPS = (0, 1)
TOKS = ((8192, "PREFILL"), (128, "DECODE"))
# 2 nodes only. The bracket is a difference of differences, so it needs the reps to
# be tight rather than the scale to be large; 4N would double the machine time to
# widen a confidence interval that is already the limiting factor at 2N.
TAG = "%s_2N_12sm_%dtok_qpdefault_nodbg_gin5_ovlp%d_rep%d"

CACHE = {}


def cell(arm, tok, ovlp):
    key = (arm, tok, ovlp)
    if key not in CACHE:
        out = []
        for rep in range(1, 9):
            r = g.load(TAG % (arm, tok, ovlp, rep))
            if r:
                out.append((rep, r))
        CACHE[key] = out
    return CACHE[key]


def config_table():
    print("CONFIG -- read out of the logs, not assumed")
    print()
    print("  %-9s %-8s %-6s %-42s %-6s %-8s"
          % ("arm", "tag", "ovlp", "BUILD_REF", "#SM", "#QPs"))
    for arm, label, sha in ARMS:
        for ovlp in OVLPS:
            refs, cfgs = set(), set()
            for tok, _ in TOKS:
                for _rep, (_o, _W, _bn, c, r, _t) in cell(arm, tok, ovlp):
                    refs |= set(r)
                    cfgs |= set(c)
            print("  %-9s %-8s %-6s %-42s %-6s %-8s"
                  % (label, sha, ovlp, ",".join(sorted(refs)) or "-",
                     ",".join(sorted({str(c[0]) for c in cfgs})) or "-",
                     ",".join(sorted({"%d/%d" % (c[1], c[2]) for c in cfgs})) or "-"))
    print()
    print("  #QPs must read 13/13 on the #8+#9 rows at BOTH overlap values -- the QP")
    print("  bump is not guarded by the flag, and that is the premise of the bracket.")
    print("  The clamp removal changes CHANNELS at a fixed SM count, so it is invisible")
    print("  in #SM: the flag is the only handle on it.")
    print()


def perf_table(tok, what):
    print("%s -- %d tok, 2 nodes / 16 ranks, 12 SM, GIN type 5" % (what, tok))
    print("  each arm is compared only against main AT THE SAME overlap value.")
    print()
    hdr = ("  %-18s %-9s %-6s %10s %10s %-11s  %s"
           % ("op", "arm", "ovlp", "us", "d vs main", "SO GB/s", "per-rep us"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for op in OPS:
        printed = False
        for ovlp in OVLPS:
            base = None
            for arm, label, _sha in ARMS:
                reps = cell(arm, tok, ovlp)
                mean, per_rep, short = stat(reps, op)
                if mean is None:
                    continue
                if label == "main":
                    base = mean
                lo, hi = rng(reps, op, 0)
                print("  %-18s %-9s %-6s %10s %10s %-11s  %s%s"
                      % (op if not printed else "", label, ovlp, f(mean),
                         pct(mean, base) if label != "main" else "",
                         "-" if lo is None else "%d-%d" % (lo, hi),
                         " / ".join(f(v) for v in per_rep),
                         ("  !! " + "; ".join(short)) if short else ""))
                printed = True
        if printed:
            print()


def bracket(tok, what):
    """D0 - D1: what the two flag-guarded changes are worth, bracketed.

    D1 (at =1) is the QP bump + remote-first scheduling alone, because the clamp
    removal and the warp pairing are compiled out by the flag. D0 (at =0) is all
    four. The difference is the pair that only exists at =0 -- to the extent that
    the unguarded two are worth the same in both configurations.
    """
    print("BRACKET -- %s (%d tok): splitting PR #8+#9's four changes" % (what, tok))
    print()
    print("  D1 = #9 - main at ovlp=1  -> QP bump + remote-first only (the other two")
    print("       hunks are compiled out by the flag)")
    print("  D0 = #9 - main at ovlp=0  -> all four changes (the shipped configuration)")
    print("  D0-D1                     -> clamp removal + forward-warp pairing")
    print()
    print("  %-18s %12s %12s %12s %12s" % ("op", "D1 us", "D1 %", "D0 us", "D0 %"))
    print("  %-18s %12s %12s %12s %12s" % ("", "", "", "", ""))
    rows = []
    for op in OPS + ["layer total"]:
        vals = {}
        for ovlp in OVLPS:
            for arm, label, _sha in ARMS:
                reps = cell(arm, tok, ovlp)
                if op == "layer total":
                    d, _, _ = stat(reps, "dispatch")
                    c, _, _ = stat(reps, "reduced combine")
                    m = None if (d is None or c is None) else d + c
                else:
                    m, _, _ = stat(reps, op)
                vals[(label, ovlp)] = m
        m0, p0 = vals.get(("main", 0)), vals.get(("PR #8+#9", 0))
        m1, p1 = vals.get(("main", 1)), vals.get(("PR #8+#9", 1))
        if None in (m0, p0, m1, p1):
            continue
        d0, d1 = p0 - m0, p1 - m1
        rows.append((op, d1, 100.0 * d1 / m1, d0, 100.0 * d0 / m0, d0 - d1))
        print("  %-18s %12.1f %11.1f%% %12.1f %11.1f%%"
              % (op, d1, 100.0 * d1 / m1, d0, 100.0 * d0 / m0))
    print()
    print("  %-18s %14s   %s" % ("op", "D0-D1 us", "reading"))
    for op, d1, _r1, d0, _r0, diff in rows:
        if diff < 0:
            reading = "clamp+pairing help (%.1f us of the %.1f us win)" % (-diff, -d0)
        elif d0 < 0:
            reading = "the win comes from the unguarded two; clamp+pairing cost %.1f us" % diff
        else:
            reading = "no win at either value"
        print("  %-18s %14.1f   %s" % (op, diff, reading))
    print()
    print("  This is a bracket, not an attribution: ovlp=1 also changes")
    print("  double-buffering and warp counts, so D1 is measured in a different kernel")
    print("  configuration than D0. Reverting the two guarded hunks in a build is the")
    print("  only way to close that gap.")
    print()


def flag_as_config(tok, what):
    """The flag is also a knob someone can just set, so measure it as one.

    This is the ONE place an ovlp=0-vs-1 delta is printed, and it is a
    configuration comparison within a single arm -- never a patch effect. The
    caveat is real and belongs next to the numbers: test_ep.py runs the collective
    with NO concurrent compute, so this says what the flag costs or saves in pure
    communication, not what it is for.
    """
    print("FLAG AS A CONFIGURATION -- %s (%d tok), ovlp=0 vs ovlp=1 within one arm"
          % (what, tok))
    print("  dispatch + reduced combine, the pair a MoE layer issues.")
    print()
    print("  %-9s %10s %10s %10s %10s %9s"
          % ("arm", "op", "ovlp=0", "ovlp=1", "d", "d %"))
    for arm, label, _sha in ARMS:
        tot = {}
        for ovlp in OVLPS:
            reps = cell(arm, tok, ovlp)
            d, _, _ = stat(reps, "dispatch")
            c, _, _ = stat(reps, "reduced combine")
            if d is None or c is None:
                continue
            tot[ovlp] = (d, c, d + c)
        if len(tot) != 2:
            continue
        for i, name in ((0, "dispatch"), (1, "redComb"), (2, "layer")):
            a, b = tot[0][i], tot[1][i]
            print("  %-9s %10s %10s %10s %10s %8.1f%%"
                  % (label if i == 0 else "", name, f(a), f(b), f(b - a),
                     100.0 * (b - a) / a))
    print()
    print("  test_ep.py issues no concurrent compute, so a flag whose purpose is to")
    print("  overlap with compute is being measured with nothing to overlap. Read these")
    print("  as pure-communication cost, and do not conclude anything about end-to-end")
    print("  throughput under load from them.")
    print()


def headline(tok, what):
    """Every sentence's numbers computed here, so the prose cannot drift from logs."""
    def v(arm, ovlp, op):
        return stat(cell(arm, tok, ovlp), op)[0]

    print("HEADLINE -- %s (%d tok). Generated; do not edit these sentences by hand."
          % (what, tok))
    print()
    for op in ("dispatch", "reduced combine"):
        m0, p0 = v("main54fffef", 0, op), v("pr893c737dc", 0, op)
        m1, p1 = v("main54fffef", 1, op), v("pr893c737dc", 1, op)
        if None in (m0, p0, m1, p1):
            continue
        d0, d1 = p0 - m0, p1 - m1
        # A split of a delta that is itself inside the run spread is arithmetic, not
        # a finding: at prefill dispatch the reps agree to ~0.2% and the arms differ
        # by 0.2%, so "50/50" there would be dividing noise by noise. FLAT is the
        # honest output, and the threshold is stated rather than hidden.
        if abs(d0) < 0.01 * m0:
            print("  %s: FLAT -- %.1f -> %.1f us at ovlp=0 (%+.2f%%) and %.1f -> %.1f"
                  % (op, m0, p0, 100.0 * d0 / m0, m1, p1))
            print("    at ovlp=1 (%+.2f%%). Both inside 1%% of the baseline, which is the"
                  % (100.0 * d1 / m1))
            print("    scale of the rep spread here, so there is no win to attribute.")
            print()
            continue
        verb = "saves" if d0 < 0 else "costs"
        print("  %s: PR #8+#9 %s %.1f us (%.1f%%) at the shipped ovlp=0"
              % (op, verb, abs(d0), abs(100.0 * d0 / m0)))
        print("    (%.1f -> %.1f us). With the flag ON, which compiles out the clamp"
              % (m0, p0))
        if d1 * d0 <= 0:
            print("    removal and the warp pairing, the effect DISAPPEARS or reverses:")
            print("    %.1f -> %.1f us (%+.1f%%). On this op the whole effect needs the"
                  % (m1, p1, 100.0 * d1 / m1))
            print("    two guarded changes.")
            print()
            continue
        share = 100.0 * (d1 / d0)
        print("    removal and the warp pairing, it still %s %.1f us (%.1f%%)"
              % (verb, abs(d1), abs(100.0 * d1 / m1)))
        print("    (%.1f -> %.1f us) -- so the two UNGUARDED changes (QP bump 11 -> 13,"
              % (m1, p1))
        print("    remote-first scheduling) account for about %.0f%% of it and the two"
              % share)
        print("    guarded ones for the remaining %.0f%% (%.1f us)."
              % (100.0 - share, abs(d0 - d1)))
        print()


def node_layering(tok):
    print("NODE LAYERING -- per-node mean us, %d tok" % tok)
    print("  combine splits by machine and the slow machine flips between runs, so a")
    print("  table built from one node is wrong by this much. Every number above pools")
    print("  all 16 ranks from both logs.")
    print()
    for op in ("dispatch", "reduced combine"):
        print("  %s:" % op)
        print("    %-9s %-6s %-6s %-24s %s"
              % ("arm", "ovlp", "rep", "per-node mean us", "max-min"))
        for arm, label, _sha in ARMS:
            for ovlp in OVLPS:
                for rep, (_o, _W, bynode, _c, _r, _t) in cell(arm, tok, ovlp):
                    per = [(n, st.mean(d[op])) for n, d in sorted(bynode.items())
                           if op in d]
                    if not per:
                        continue
                    vals = [v for _n, v in per]
                    print("    %-9s %-6s %-6s %-24s %.1f%%"
                          % (label, ovlp, "rep%d" % rep,
                             " ".join("n%s %.0f" % (n, v) for n, v in per),
                             100.0 * (max(vals) - min(vals)) / min(vals)))
        print()


def audit():
    print("AUDIT -- reps on disk and rank completeness. n must equal 16; a short n is")
    print("  glued stdout lines, i.e. lost data, and invalidates that rep's mean.")
    print()
    for tok, _ in TOKS:
        for arm, label, _sha in ARMS:
            for ovlp in OVLPS:
                reps = cell(arm, tok, ovlp)
                if not reps:
                    print("  %5dtok ovlp%d %-9s -- not run" % (tok, ovlp, label))
                    continue
                print("  %5dtok ovlp%d %-9s %s"
                      % (tok, ovlp, label,
                         " ".join("rep%d:%d/%d"
                                  % (rep, len(ops.get("dispatch", {})), W)
                                  for rep, (ops, W, _b, _c, _r, _t) in reps)))
    print()
    if g.EMPTY:
        print("  logs present but EMPTY (in-flight, or died before iteration 1) --")
        for e in sorted(set(g.EMPTY)):
            print("    %s" % e)
        print()
    if g.EXCLUDED:
        print("  reps EXCLUDED as outliers (> %.0f%% off their cell's median);"
              % (100 * g.OUTLIER))
        print("  the raw logs are still under logs/, so this is reversible:")
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
        bracket(tok, what)
    for tok, what in TOKS:
        headline(tok, what)
    for tok, what in TOKS:
        flag_as_config(tok, what)
    node_layering(128)
    node_layering(8192)
    audit()

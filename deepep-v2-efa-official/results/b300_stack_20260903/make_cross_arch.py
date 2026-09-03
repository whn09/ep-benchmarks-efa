#!/usr/bin/env python3
"""Same arm, same cell, two machines: b300 (sm_103) vs p5en (sm_90).

usage: make_cross_arch.py

Loads the p5en stack generator twice -- once pointed at ./logs (b300), once at
../p5en_stack_20260831/logs -- so both columns come out of the same parser and the
same pooling rule. The arm labels are identical across the two campaigns by
construction (the stack merge sha is pinned in Dockerfile.stack), so a row really
is the same tree on two different machines.

Per-GPU wire: p5en 400 Gb/s = 50 GB/s, b300 100 GB/s. So on any op that is
wire-limited, b300 should be ~2x faster in TIME; where it is not, the op is not
wire-limited. TIME IS THE METRIC -- the SO column is printed by the b300/p5en
tables and is not a wire rate (no --ignore-local-traffic).

Cells with fewer reps on one side are marked (n=N); a single rep is n=1 and has no
spread, so read it as a point estimate, not a measurement with an error bar.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P5EN = os.path.join(os.path.dirname(HERE), "p5en_stack_20260831")
GEN = os.path.join(P5EN, "make_stack_tables.py")


def load(logdir):
    os.environ["EPRUNS_STACK"] = logdir
    spec = importlib.util.spec_from_file_location("gen_%s" % abs(hash(logdir)), GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load(os.path.join(HERE, "logs"))
P = load(os.path.join(P5EN, "logs"))

ARMS = [("main", "main"), ("PR #1+#2", "pr12"), ("PR #8+#9", "pr89"), ("stack", "stack")]


def arm_of(mod, short):
    return {"main": mod.MAIN, "pr12": mod.PR12, "pr89": mod.PR89, "stack": mod.STACK}[short]


def cell_us(mod, short, tok, op):
    """(mean us over reps, n reps) or (None, 0) -- same stat() the tables use."""
    arm = arm_of(mod, short)
    mean = mod.us(arm, tok, op)
    return (None, 0) if mean is None else (mean, len(mod.cell(arm, tok)))


def table(tok, what):
    print()
    print("%s -- %d tok/rank, 2 nodes / 16 ranks, 12 SM, GIN type 5, ovlp=0, default knob" % (what, tok))
    print("  b300 = 2x p6-b300 (sm_103, 100 GB/s per GPU, EFA 1.50.0 / efa.ko 3.3.0)")
    print("  p5en = 2x p5en    (sm_90,   50 GB/s per GPU)")
    print()
    print("  %-18s %-9s %12s %12s %10s" % ("op", "arm", "b300 us", "p5en us", "b300/p5en"))
    print("  " + "-" * 66)
    for op in B.OPS:
        first = True
        for label, short in ARMS:
            b, bn = cell_us(B, short, tok, op)
            p, pn = cell_us(P, short, tok, op)
            if b is None and p is None:
                continue
            bs = "--" if b is None else "%.1f%s" % (b, " (n=1)" if bn == 1 else "")
            ps = "--" if p is None else "%.1f%s" % (p, " (n=1)" if pn == 1 else "")
            rs = "--" if (b is None or p is None) else "%.2fx" % (b / p)
            print("  %-18s %-9s %12s %12s %10s" % (op if first else "", label, bs, ps, rs))
            first = False
        if not first:
            print()


if __name__ == "__main__":
    print("b300 logs: %s" % B.g.EPRUNS)
    print("p5en logs: %s" % P.g.EPRUNS)
    for tok, what in B.TOKS:
        table(tok, what)

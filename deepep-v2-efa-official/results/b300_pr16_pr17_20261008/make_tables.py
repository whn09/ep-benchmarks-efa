#!/usr/bin/env python3
"""amazon-contributing/DeepEP #16 (combine) and #17 (forward chunk) on b300, vs main and the old best.

usage: make_tables.py            (logs from ./logs, or $EPRUNS_PR1617)

Both PRs are cut from main @ b90a617 (ahead 5 / ahead 1, behind 0), so each arm
differs from `main` by exactly its own commits:

  main     b90a61714ac48713408d8c62b6912d66daabb9a0
  pr16     8d45880adc39c1479fa63ef3d764a9f9ce14f95e   #16 "Combine kernel optimization" (5 commits)
  pr17     bb787506214fdc3c290798003b74dcb0b494f286   #17 kNumSlotsPerForwardChunk 48 -> 16
  stack    33734bb238cd11aba506cd492c9fa447d3d1fe3d   git merge of the two (Dockerfile.stack,
                                                      pinned dates; identical on both hosts)

All four arms rotate inside every rep of ONE campaign on one host pair (2 x
p6-b300, ap-south-2, GIN type 5, 12 SM, ovlp=0, --test-first-only), so the
in-campaign deltas are same-host. The `old best` row is the #1+#2+#8+#9 stack
from results/b300_stack_20260903 on a DIFFERENT pair (ap-northeast-2): use it as
a level, not as a sub-2% delta. TIME IS THE METRIC.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
GEN = os.path.join(RES, "p5en_3arm_20260831", "make_3arm_tables.py")


def _mod(name, logdir):
    """one independent copy of the generator per campaign (it reads EPRUNS at import)"""
    os.environ["EPRUNS"] = logdir
    spec = importlib.util.spec_from_file_location(name, GEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NEW = _mod("g_new", os.environ.get("EPRUNS_PR1617", os.path.join(HERE, "logs")))
OLD = _mod("g_old", os.path.join(RES, "b300_stack_20260903", "logs"))
f, pct, OPS = NEW.f, NEW.pct, NEW.OPS

TAG = "%s_2N_12sm_%dtok_qpdefault_nodbg_gin5_ovlp0_rep%d"
BASE = ("main b90a617", NEW, "mainb90a617")
ARMS = [BASE,
        ("#16 8d45880", NEW, "pr168d45880"),
        ("#17 bb78750", NEW, "pr17bb78750"),
        ("#16+#17 33734bb", NEW, "stack33734bb"),
        ("old best #1+#2+#8+#9", OLD, "stacka35285f")]


def reps(g, arm, tok):
    out = []
    for rep in range(1, 9):
        r = g.load(TAG % (arm, tok, rep))
        if r:
            out.append((rep, r))
    return out


def us(g, arm, tok, op):
    rr = reps(g, arm, tok)
    return g.stat(rr, op)[0] if rr else None


def config():
    print("CONFIG -- read out of the logs")
    print("  %-22s %5s  %-42s %-9s %s" % ("arm", "tok", "BUILD_REF", "#SM/#QPs", "reps"))
    for label, g, arm in ARMS:
        for tok in (8192, 128):
            rr = reps(g, arm, tok)
            refs = sorted({x for _n, r in rr for x in r[4]})
            cfg = sorted({"%d/%d/%d" % c for _n, r in rr for c in r[3]})
            print("  %-22s %5d  %-42s %-9s %d" % (label, tok, ",".join(refs) or "-",
                                                  ",".join(cfg) or "-", len(rr)))
    print()


def table(tok, what):
    print("%s -- %d tok, 2 nodes / 16 ranks, 12 SM, GIN type 5, ovlp=0" % (what, tok))
    print("  d = vs main b90a617 in this campaign (negative = faster than main)")
    print("  %-17s %-22s %9s %8s %8s   %s" % ("op", "arm", "us", "d main", "SO GB/s", "per-rep us"))
    print("  " + "-" * 96)
    for op in OPS:
        b = us(*BASE[1:], tok, op)
        first = True
        for label, g, arm in ARMS:
            rr = reps(g, arm, tok)
            mu, kept, notes = g.stat(rr, op) if rr else (None, [], [])
            so = g.stat(rr, op, 0)[0] if rr else None
            print("  %-17s %-22s %9s %8s %8s   %s%s" % (
                op if first else "", label, f(mu), "" if arm == BASE[2] else pct(mu, b),
                f(so, 0), " / ".join(f(v) for v in kept),
                ("  [" + "; ".join(notes) + "]") if notes else ""))
            first = False
        print()


def layer():
    print("LAYER -- dispatch + reduced combine, 12 SM (us); d vs main")
    for tok in (8192, 128):
        b = None
        for label, g, arm in ARMS:
            d, c = us(g, arm, tok, "dispatch"), us(g, arm, tok, "reduced combine")
            tot = None if d is None or c is None else d + c
            if arm == BASE[2]:
                b = tot
            print("  %5d tok  %-22s %9s %8s" % (tok, label, f(tot), "" if arm == BASE[2] else pct(tot, b)))
    print()


def audit():
    print("AUDIT")
    for name, g in (("new", NEW), ("old", OLD)):
        for e in g.EXCLUDED:
            print("  excluded (%s): %s" % (name, e))
        for e in sorted(set(g.EMPTY)):
            print("  empty (%s): %s" % (name, e))
    print("  done.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print()
    config()
    table(8192, "PREFILL")
    table(128, "DECODE")
    layer()
    audit()

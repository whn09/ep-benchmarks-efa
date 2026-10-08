#!/usr/bin/env python3
"""amazon-contributing/DeepEP main @ b90a617 on b300, against the best earlier b300 arms.

usage: make_tables.py            (new logs from ./logs; the old campaigns from their own dirs)

b90a617 is main after #15 (PR #2's kernel clamp, `EP_MIN_TOKENS_PER_PART` = 15
compiled in; the JIT env forwarding of #1/#2 was NOT merged, so no knob is
tunable at runtime on this tree). It also carries #11 and #14 over 54fffef.

New:  2 x p6-b300.48xlarge, ap-south-2 (B300-1 ip-172-31-9-47, B300-2 ip-172-31-3-47),
      driver 595.91.07, EFA installer 1.50.0 + efa.ko 3.3.0g, GIN type 5, ovlp=0,
      --test-first-only (FP8 dispatch, expert_alignment=128), 3 rotated reps.
Old:  results/b300_stack_20260903 (12 SM) and results/b300_sm24_20260903 (24 SM),
      the ap-northeast-2 pair -- DIFFERENT HOSTS, so a cross-campaign delta carries
      host-to-host variance on top of the code change (combine is node-layered by
      14-18% even between two identical boxes).

Every statistic is the 3-arm generator's: per rep the mean over all 16 ranks of
both nodes, then the mean over reps with a >25%-off-median rep excluded loudly.
TIME IS THE METRIC; SO GB/s is the all-rank mean with --ignore-local-traffic OFF
(byte denominator includes intra-node destinations; it is not a wire rate).
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


NEW = _mod("g_new", os.environ.get("EPRUNS_NEW", os.path.join(HERE, "logs")))
S12 = _mod("g_s12", os.path.join(RES, "b300_stack_20260903", "logs"))
S24 = _mod("g_s24", os.path.join(RES, "b300_sm24_20260903", "logs"))
f, pct, OPS = NEW.f, NEW.pct, NEW.OPS

TAG = "%s_2N_%dsm_%dtok_qpdefault_nodbg_gin5_ovlp0_rep%d"
# (label, generator module, arm prefix) -- the new main first; deltas are vs it.
ARMS = {
    12: [("main b90a617 (new)", NEW, "mainb90a617"),
         ("main 54fffef (old)", S12, "main54fffef"),
         ("PR #1+#2 (old)", S12, "pr12bfbdd15"),
         ("stack #1+#2+#8+#9 (old best)", S12, "stacka35285f")],
    24: [("main b90a617 (new)", NEW, "mainb90a617"),
         ("main 54fffef (old)", S24, "main54fffef"),
         ("PR #1+#2 (old)", S24, "pr12bfbdd15"),
         ("stack #1+#2+#8+#9 (old best)", S24, "stacka35285f")],
}


def reps(g, arm, sms, tok):
    out = []
    for rep in range(1, 9):
        r = g.load(TAG % (arm, sms, tok, rep))
        if r:
            out.append((rep, r))
    return out


def config():
    print("CONFIG -- read out of the logs")
    print("  %-30s %3s %5s  %-42s %s" % ("arm", "SM", "tok", "BUILD_REF", "#SM/#QPs  reps"))
    for sms, arms in ARMS.items():
        for label, g, arm in arms:
            for tok in (8192, 128):
                rr = reps(g, arm, sms, tok)
                if not rr:
                    continue
                refs = sorted({x for _n, r in rr for x in r[4]})
                cfg = sorted({"%d/%d/%d" % c for _n, r in rr for c in r[3]})
                print("  %-30s %3d %5d  %-42s %s  %d" % (label, sms, tok, ",".join(refs) or "?",
                                                        ",".join(cfg) or "?", len(rr)))
    print()


def table(sms, tok, what):
    arms = ARMS[sms]
    base = arms[0]
    print("%s -- %d tok, 2 nodes / 16 ranks, %d SM, GIN type 5, ovlp=0" % (what, tok, sms))
    print("  d = new main relative to that row (negative = new main is faster)")
    print("  %-17s %-30s %9s %9s %8s   %s" % ("op", "arm", "us", "new d", "SO GB/s", "per-rep us"))
    print("  " + "-" * 100)
    for op in OPS:
        b = NEW.stat(reps(base[1], base[2], sms, tok), op)[0] if True else None
        first = True
        for label, g, arm in arms:
            rr = reps(g, arm, sms, tok)
            mu, kept, notes = g.stat(rr, op) if rr else (None, [], [])
            so = g.stat(rr, op, 0)[0] if rr else None
            d = "" if arm == base[2] else pct(b, mu)
            print("  %-17s %-30s %9s %9s %8s   %s%s" % (
                op if first else "", label, f(mu), d, f(so, 0),
                " / ".join(f(v) for v in kept), ("  [" + "; ".join(notes) + "]") if notes else ""))
            first = False
        print()


def layer(sms):
    print("LAYER -- dispatch + reduced combine, %d SM (us)" % sms)
    for tok in (8192, 128):
        for label, g, arm in ARMS[sms]:
            rr = reps(g, arm, sms, tok)
            d = g.stat(rr, "dispatch")[0] if rr else None
            c = g.stat(rr, "reduced combine")[0] if rr else None
            tot = None if d is None or c is None else d + c
            print("  %5d tok  %-30s %9s" % (tok, label, f(tot)))
    print()


def audit():
    print("AUDIT")
    for name, g in (("new", NEW), ("stack12", S12), ("sm24", S24)):
        for e in g.EXCLUDED:
            print("  excluded (%s): %s" % (name, e))
        for e in g.EMPTY:
            print("  empty (%s): %s" % (name, e))
    print("  done.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print()
    config()
    for sms in (12, 24):
        for tok, what in ((8192, "PREFILL"), (128, "DECODE")):
            table(sms, tok, what)
        layer(sms)
    audit()

#!/usr/bin/env python3
"""Regenerate every table in summary.txt straight from the per-node logs.

usage: [EPRUNS=./logs] make_tables.py

Every number published in summary.txt / README.md / docs/runbook_zh.md comes out
of this script. Hand-transcribing them has introduced errors before, so treat the
script as the source of truth and paste its output rather than editing a table.

Aggregation, stated once:
  * a run's value for an op = mean over ALL ranks (16 at 2N, 32 at 4N), pooled
    from every node's log. combine is layered by node, so one node is not enough.
  * an arm's value = mean over that arm's reps of the per-run means. Rep values
    are printed alongside so a reader can see the spread instead of trusting it.
  * SO / SU are printed as min-max ACROSS RANKS -- a different statistic from the
    us column. For dispatch that range is dominated by the per-rank byte
    denominator plus 1-GB/s integer print quantization, not by time. See the
    SPREAD table.
  * the log's `bytes` field is num_scaleup_bytes, so SU = bytes / time exactly and
    scale-out bytes must be recovered as SO * time.

Parsing note: use finditer, not search. Ranks writing concurrently to the same
stdout regularly glue two per-rank lines onto one physical line, and search()
silently keeps only the first. A short n= in the audit means real data loss.
"""
import glob
import os
import re
import statistics as st
import sys

LINE = re.compile(
    r"EP:\s+(\d+)/(\d+)\s*\|\s*([a-z ]+?):\s*(\d+) GB/s \(SO\), (\d+) GB/s \(SU\), "
    r"([\d.]+) us, (\d+) bytes"
)
EPRUNS = os.environ.get("EPRUNS", os.path.join(os.path.dirname(__file__), "logs"))
OPS = ["dispatch", "expanded dispatch", "cached dispatch", "combine", "reduced combine"]
WIRE_PER_GPU = 50.0  # GB/s: 16 x 200 Gb/s / 8 GPUs


def load(tag):
    """-> {op: {rank: (so, su, us, bytes)}}, world_size"""
    ops, world = {}, set()
    files = sorted(glob.glob("%s/%s.node*.log" % (EPRUNS, tag)))
    if not files:
        raise SystemExit("no logs for tag %s in %s" % (tag, EPRUNS))
    for f in files:
        for line in open(f, errors="replace"):
            for m in LINE.finditer(line):
                rank, nranks, op, so, su, us, nbytes = m.groups()
                world.add(int(nranks))
                ops.setdefault(op.strip(), {})[int(rank)] = (
                    int(so), int(su), float(us), int(nbytes))
    return ops, max(world)


CACHE = {}


def runs(tags):
    for t in tags:
        if t not in CACHE:
            CACHE[t] = load(t)
        yield t, CACHE[t]


def arm(tags, op, field=2):
    """mean over reps of the per-run all-rank mean; also returns the rep values"""
    per_rep, incomplete = [], []
    for t, (ops, W) in runs(tags):
        if op not in ops:
            continue
        vals = [v[field] for v in ops[op].values()]
        per_rep.append(st.mean(vals))
        if len(vals) != W:
            incomplete.append("%s n=%d/%d" % (t, len(vals), W))
    if not per_rep:
        return None, [], incomplete
    return st.mean(per_rep), per_rep, incomplete


def rng(tags, op, field):
    xs = [v[field] for t, (ops, W) in runs(tags) if op in ops for v in ops[op].values()]
    return (min(xs), max(xs)) if xs else (None, None)


def f(x, nd=1):
    return "-" if x is None else "%.*f" % (nd, x)


# ---------------------------------------------------------------- arm definitions
def T(*tags):
    return list(tags)


O2_12_8192_T2 = T("official_2N_12sm_8192tok_qpdefault_nodbg_rep1")
O2_12_8192_T5 = T("official_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep2")
O2_12_128_T2 = T("official_2N_12sm_128tok_qpdefault_nodbg_rep1",
                 "official_2N_12sm_128tok_qpdefault_nodbg_rep2")
O2_12_128_T5 = T("official_2N_12sm_128tok_qpdefault_nodbg_gin5symgin0_rep1",
                 "official_2N_12sm_128tok_qpdefault_nodbg_gin5_rep2")
O4_12_8192_T2 = T("official_4N_12sm_8192tok_qpdefault_rep1",
                  "official_4N_12sm_8192tok_qpdefault_rep2")
O4_12_8192_T5 = T("official_4N_12sm_8192tok_qpdefault_nodbg_gin5_rep1",
                  "official_4N_12sm_8192tok_qpdefault_nodbg_gin5_rep2",
                  "official_4N_12sm_8192tok_qpdefault_nodbg_gin5_rep3")
O4_12_128_T2 = T("official_4N_12sm_128tok_qpdefault_rep1",
                 "official_4N_12sm_128tok_qpdefault_rep2",
                 "official_4N_12sm_128tok_qpdefault_rep3",
                 "official_4N_12sm_128tok_qpdefault_rep4",
                 "official_4N_12sm_128tok_qpdefault_nodbg_rep1")
O4_12_128_T5 = T("official_4N_12sm_128tok_qpdefault_nodbg_gin5_rep1")

SMS_2N = [("6", T("official_2N_6sm_8192tok_qpdefault_nodbg_gin5_rep1")),
          ("12", O2_12_8192_T5),
          ("16", T("official_2N_16sm_8192tok_qpdefault_nodbg_gin5_rep1")),
          ("24", T("official_2N_24sm_8192tok_qpdefault_nodbg_gin5_rep1",
                   "official_2N_24sm_8192tok_qpdefault_nodbg_gin5_rep2",
                   "official_2N_24sm_8192tok_qpdefault_nodbg_gin5_rep3")),
          ("32", T("official_2N_32sm_8192tok_qpdefault_nodbg_gin5_rep1"))]
SMS_4N = [("6", T("official_4N_6sm_8192tok_qpdefault_nodbg_gin5_rep1")),
          ("12", O4_12_8192_T5),
          ("16", None), ("24", T("official_4N_24sm_8192tok_qpdefault_nodbg_gin5_rep1",
                                 "official_4N_24sm_8192tok_qpdefault_nodbg_gin5_rep2")),
          ("32", None)]
SMD_2N = [("6", None), ("12", O2_12_128_T5),
          ("24", T("official_2N_24sm_128tok_qpdefault_nodbg_gin5_rep1"))]
SMD_4N = [("6", T("official_4N_6sm_128tok_qpdefault_nodbg_gin5_rep1")),
          ("12", O4_12_128_T5),
          ("24", T("official_4N_24sm_128tok_qpdefault_nodbg_gin5_rep1"))]

PRS_2N_128 = {
    "unpatched ec623f3, type 2": O2_12_128_T2,
    "#1+#2, type 2": T("prs_2N_12sm_128tok_prsdflt_nodbg_type2_rep1",
                       "prs_2N_12sm_128tok_prsdflt_nodbg_type2_rep2"),
    "unpatched ec623f3, type 5": O2_12_128_T5,
    "#1+#2, type 5": T("prs_2N_12sm_128tok_prsdflt_nodbg_gin5_rep1",
                       "prs_2N_12sm_128tok_prsdflt_nodbg_gin5_rep2"),
    "#1+#2 + EP_NUM_SUB_PARTS=1, t5": T("prs_2N_12sm_128tok_prssub1_nodbg_gin5_rep1"),
    "#1+#2 + EP_MIN_TOKENS_PER_PART=1": T("prs_2N_12sm_128tok_prsmtpp1_nodbg_gin5_rep1"),
}
PRS_4N_128 = {
    "unpatched, type 2": O4_12_128_T2,
    "#1+#2, type 2": T("prs_4N_12sm_128tok_prsdflt_nodbg_type2_rep1"),
    "unpatched, type 5": O4_12_128_T5,
    "#1+#2, type 5": T("prs_4N_12sm_128tok_prsdflt_nodbg_gin5_rep1",
                       "prs_4N_12sm_128tok_prsdflt_nodbg_gin5_rep2"),
    "#1+#2 + EP_NUM_SUB_PARTS=1, t5": T("prs_4N_12sm_128tok_prssub1_nodbg_gin5_rep1"),
    "#1+#2 + EP_MIN_TOKENS_PER_PART=1": T("prs_4N_12sm_128tok_prsmtpp1_nodbg_gin5_rep1"),
}
T8 = [("2N/12SM/8192  official ec623f3", O2_12_8192_T5),
      ("2N/12SM/8192  main 8e7b42e", T("main_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep1",
                                       "main_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep2")),
      ("2N/24SM/8192  official ec623f3", SMS_2N[3][1]),
      ("2N/24SM/8192  main 8e7b42e", T("main_2N_24sm_8192tok_qpdefault_nodbg_gin5_rep1",
                                       "main_2N_24sm_8192tok_qpdefault_nodbg_gin5_rep2")),
      ("4N/12SM/8192  official ec623f3", O4_12_8192_T5),
      ("4N/12SM/8192  main 8e7b42e", T("main_4N_12sm_8192tok_qpdefault_nodbg_gin5_rep1")),
      ("2N/12SM/128   official ec623f3", O2_12_128_T5),
      ("2N/12SM/128   main 8e7b42e", T("main_2N_12sm_128tok_qpdefault_nodbg_gin5_rep1",
                                       "main_2N_12sm_128tok_qpdefault_nodbg_gin5_rep2"))]
PIN = [("2N/12SM/8192  pin 7a6059a3", T("pin_2N_12sm_8192tok_qpauto_skipchk_rep1",
                                        "pin_2N_12sm_8192tok_qpauto_skipchk_rep2")),
       ("2N/12SM/8192  release ec623f3", O2_12_8192_T5),
       ("2N/12SM/128   pin 7a6059a3", T("pin_2N_12sm_128tok_qpauto_skipchk_rep1",
                                        "pin_2N_12sm_128tok_qpauto_skipchk_rep2")),
       ("2N/12SM/128   release ec623f3", O2_12_128_T5),
       ("4N/12SM/8192  pin 7a6059a3", T("pin_4N_12sm_8192tok_qpauto_skipchk_rep1",
                                        "pin_4N_12sm_8192tok_qpauto_skipchk_rep2")),
       ("4N/12SM/8192  release ec623f3", O4_12_8192_T5),
       ("4N/12SM/128   pin 7a6059a3", T("pin_4N_12sm_128tok_qpauto_skipchk_rep1",
                                        "pin_4N_12sm_128tok_qpauto_skipchk_rep2")),
       ("4N/12SM/128   release ec623f3", O4_12_128_T5)]
T3 = [("type-2 default", T("official_2N_12sm_128tok_qpdefault_nodbg_rep1")),
      ("type-2 default, rep 2", T("official_2N_12sm_128tok_qpdefault_nodbg_rep2")),
      ("+ FI_EFA_USE_HW_CNTR=1", T("official_2N_12sm_128tok_qpdefault_nodbg_only_hwcntr1_rep1")),
      ("+ NCCL_RMA_DISABLE=1", T("official_2N_12sm_128tok_qpdefault_nodbg_only_rmadis1_rep1")),
      ("+ OFI_NCCL_GIN_STRONG_SIGNAL=1", T("official_2N_12sm_128tok_qpdefault_nodbg_strongsig_rep1")),
      ("+ NCCL_SYM_GIN_KERNELS_ENABLE=0 alone", T("official_2N_12sm_128tok_qpdefault_nodbg_only_symgin0_rep1")),
      ("+ the pair (GIN_TYPE=5 + SYM_GIN=0)", T("official_2N_12sm_128tok_qpdefault_nodbg_gin5symgin0_rep1")),
      ("+ all five route-B vars", T("official_2N_12sm_128tok_qpdefault_nodbg_routeB_rep1")),
      ("+ all five, rep 2", T("official_2N_12sm_128tok_qpdefault_nodbg_routeB_rep2"))]
T4 = [("2N/12SM/8192, type 2, default", T("official_2N_12sm_8192tok_qpdefault_nodbg_rep1")),
      ("2N/12SM/8192, type 2, qps 5", T("official_2N_12sm_8192tok_qps5_rep1")),
      ("2N/12SM/8192, type 5, default", O2_12_8192_T5),
      ("2N/12SM/8192, type 5, qps 5", T("official_2N_12sm_8192tok_qps5_nodbg_gin5_rep1")),
      ("4N/12SM/128,  type 2, default", O4_12_128_T2),
      ("4N/12SM/128,  type 2, qps 5", T("official_4N_12sm_128tok_qps5_rep1"))]

SPREAD_2N = SMS_2N[3][1]
SPREAD_4N = SMS_4N[3][1]


def line(label, tags, ops=("dispatch", "cached dispatch", "combine", "reduced combine"),
         width=34, show_reps=False):
    cells = []
    for op in ops:
        m, reps, bad = arm(tags, op)
        cells.append(f(m).rjust(9))
    s = label.ljust(width) + "".join(cells)
    if show_reps:
        _, reps, _ = arm(tags, ops[0])
        s += "   reps " + "/".join("%.1f" % r for r in reps)
    return s


def main():
    out = sys.stdout.write

    out("TABLE 1 -- which GIN backend, 2 nodes / 16 ranks / 12 SM\n")
    out("--------------------------------------------------------\n")
    out("                          8192 tok             128 tok\n")
    out("arm                       dispatch  SO GB/s    dispatch  SO GB/s\n")
    for name, a8, a1 in [("type 2 (default env)", O2_12_8192_T2, O2_12_128_T2),
                         ("type 5 (the two vars)", O2_12_8192_T5, O2_12_128_T5)]:
        d8 = arm(a8, "dispatch")[0]
        s8 = arm(a8, "dispatch", 0)[0]
        d1 = arm(a1, "dispatch")[0]
        s1 = arm(a1, "dispatch", 0)[0]
        out("%-25s %-9s %-10s %-9s %s\n" % (name, f(d8), f(s8), f(d1), f(s1)))
    b8 = arm(O2_12_8192_T2, "dispatch")[0], arm(O2_12_8192_T5, "dispatch")[0]
    b1 = arm(O2_12_128_T2, "dispatch")[0], arm(O2_12_128_T5, "dispatch")[0]
    out("%-25s %-9s %-10s %-9s\n" % ("", "%.1f%%" % (100 * (b8[1] - b8[0]) / b8[0]), "",
                                     "%.1f%% (%.2fx)" % (100 * (b1[1] - b1[0]) / b1[0],
                                                         b1[0] / b1[1])))

    out("\n\nTABLE 2 -- which GIN backend, 4 nodes / 32 ranks / 12 SM\n")
    out("--------------------------------------------------------\n")
    out("                          8192 tok                       128 tok\n")
    out("arm                       dispatch  SO GB/s  wire%        dispatch  SO GB/s\n")
    for name, a8, a1 in [("type 2 (default env)", O4_12_8192_T2, O4_12_128_T2),
                         ("type 5 (the two vars)", O4_12_8192_T5, O4_12_128_T5)]:
        d8 = arm(a8, "dispatch")[0]
        s8 = arm(a8, "dispatch", 0)[0]
        d1 = arm(a1, "dispatch")[0]
        s1 = arm(a1, "dispatch", 0)[0]
        out("%-25s %-9s %-8s %-11s %-9s %s\n"
            % (name, f(d8), f(s8), "%.1f%%" % (s8 * 0.75 / WIRE_PER_GPU * 100), f(d1), f(s1)))
    c8 = arm(O4_12_8192_T2, "dispatch")[0], arm(O4_12_8192_T5, "dispatch")[0]
    c1 = arm(O4_12_128_T2, "dispatch")[0], arm(O4_12_128_T5, "dispatch")[0]
    out("%-25s %-9s %-8s %-11s %-9s\n"
        % ("", "%.1f%%" % (100 * (c8[1] - c8[0]) / c8[0]), "", "",
           "%.1f%% (%.2fx)" % (100 * (c1[1] - c1[0]) / c1[0], c1[0] / c1[1])))
    out("type-2 4N/128 reps: %s us\n"
        % " / ".join("%.1f" % r for r in sorted(arm(O4_12_128_T2, "dispatch")[1])))
    out("type-5 4N/128 all-rank range: %s us\n"
        % ("%.1f-%.1f" % rng(O4_12_128_T5, "dispatch", 2)))

    out("\n\nTABLE 3 -- what else we tried, and what it was worth\n")
    out("    2 nodes / 16 ranks / 12 SM / 128 tok, one arm per row, dispatch us\n")
    out("----------------------------------------------------------------------\n")
    for name, tags in T3:
        out("%-40s %s\n" % (name, f(arm(tags, "dispatch")[0])))

    out("\n\nTABLE 4 -- --num-allocated-qps 5 : not worth setting\n")
    out("----------------------------------------------------\n")
    out("arm                                dispatch  cached disp  combine  redComb\n")
    for name, tags in T4:
        out("%-33s%s\n" % (name, "".join(
            f(arm(tags, op)[0]).rjust(10) for op in
            ("dispatch", "cached dispatch", "combine", "reduced combine"))))

    out("\n\nTABLE 5 -- --num-sms scan on type 5, 8192 tok (prefill)\n")
    out("-------------------------------------------------------\n")
    out("2 nodes / 16 ranks                        4 nodes / 32 ranks\n")
    out("SM   dispatch  combine  redComb  d+rC     dispatch  combine  redComb  d+rC\n")
    for (sm, t2), (_, t4) in zip(SMS_2N, SMS_4N):
        cells = []
        for tags in (t2, t4):
            if tags is None:
                cells += ["-"] * 4
                continue
            d = arm(tags, "dispatch")[0]
            c = arm(tags, "combine")[0]
            r = arm(tags, "reduced combine")[0]
            cells += [f(d), f(c), f(r), f(d + r)]
        out("%-4s %-9s %-8s %-8s %-8s %-9s %-8s %-8s %s\n" % tuple([sm] + cells))

    out("\n\nTABLE 6 -- --num-sms on type 5, 128 tok (decode)\n")
    out("------------------------------------------------\n")
    out("2 nodes / 16 ranks                4 nodes / 32 ranks\n")
    out("SM   dispatch  combine  redComb   dispatch  combine  redComb\n")
    for (sm, t2), (_, t4) in zip(SMD_2N, SMD_4N):
        cells = []
        for tags in (t2, t4):
            if tags is None:
                cells += ["-"] * 3
                continue
            cells += [f(arm(tags, op)[0]) for op in ("dispatch", "combine", "reduced combine")]
        out("%-4s %-9s %-8s %-9s %-9s %-8s %s\n" % tuple([sm] + cells))

    out("\n\nTABLE 7 -- the two pending PRs stack with the backend switch\n")
    out("------------------------------------------------------------\n")
    for title, rows, base_keys in [
            ("2 nodes / 16 ranks / 12 SM", PRS_2N_128,
             {"type 2": "unpatched ec623f3, type 2", "type 5": "unpatched ec623f3, type 5"}),
            ("4 nodes / 32 ranks / 12 SM", PRS_4N_128,
             {"type 2": "unpatched, type 2", "type 5": "unpatched, type 5"})]:
        out("\n%-33s dispatch  vs backend  combine  redComb\n" % title)
        base = {k: arm(rows[v], "dispatch")[0] for k, v in base_keys.items()}
        for name, tags in rows.items():
            d = arm(tags, "dispatch")[0]
            ref = base["type 2"] if "type 2" in name else base["type 5"]
            rel = "1.00x" if abs(d - ref) < 1e-9 else "%.1f%%" % (100 * (d - ref) / ref)
            if "MIN_TOKENS_PER_PART=1" in name:
                rel = "clamp-off"
            out("%-33s %-9s %-11s %-8s %s\n"
                % (name, f(d), rel, f(arm(tags, "combine")[0]),
                   f(arm(tags, "reduced combine")[0])))
    st2 = arm(PRS_2N_128["unpatched ec623f3, type 2"], "dispatch")[0]
    st5 = arm(PRS_2N_128["#1+#2 + EP_NUM_SUB_PARTS=1, t5"], "dispatch")[0]
    out("\nstacked 2N decode dispatch: %.1f -> %.1f us (%.2fx)\n" % (st2, st5, st2 / st5))
    out("prefill unaffected: 2N/24SM/8192 dispatch %s (official) vs %s (PRs)\n"
        % (f(arm(SMS_2N[3][1], "dispatch")[0]),
           f(arm(T("prs_2N_24sm_8192tok_prsdflt_nodbg_gin5_rep1"), "dispatch")[0])))
    out("                    4N/12SM/8192 dispatch %s (official) vs %s (PRs)\n"
        % (f(arm(O4_12_8192_T5, "dispatch")[0]),
           f(arm(T("prs_4N_12sm_8192tok_prsdflt_nodbg_gin5_rep1"), "dispatch")[0])))
    p12 = arm(PRS_2N_128["#1+#2, type 5"], "dispatch")[0]
    p24 = arm(T("prs_2N_24sm_128tok_prsdflt_nodbg_gin5_rep1"), "dispatch")[0]
    out("with the PRs, 2N decode prefers 12 SM: %.1f vs %.1f us at 24 SM\n" % (p12, p24))
    q12 = (arm(PRS_4N_128["#1+#2, type 5"], "dispatch")[0]
           + arm(PRS_4N_128["#1+#2, type 5"], "reduced combine")[0])
    q24t = T("prs_4N_24sm_128tok_prsdflt_nodbg_gin5_rep1")
    q24 = arm(q24t, "dispatch")[0] + arm(q24t, "reduced combine")[0]
    out("at 4N the two tie on dispatch+redComb: %.1f vs %.1f us\n" % (q12, q24))

    out("\n\nTABLE 8 -- main is perf-neutral, and the release beats the 08-13 pin\n")
    out("--------------------------------------------------------------------\n")
    out("arm                                 dispatch  cached d  combine  redComb\n")
    for name, tags in T8:
        out(line(name, tags, width=36) + "\n")
    out("\n")
    for name, tags in PIN:
        out(line(name, tags, width=36) + "\n")

    out("\n\nONE AXIS -- every 2-node prefill arm, 16 ranks / 12 SM / 8192 tok\n")
    out("-----------------------------------------------------------------\n")
    out("arm                                        dispatch   SO mean  wire%\n")
    for name, tags in [
            ("release ec623f3, default env (type 2)", O2_12_8192_T2),
            ("release, --num-allocated-qps 5 (type 2)",
             T("official_2N_12sm_8192tok_qps5_rep1")),
            ("pin 7a6059a3, source stack, --skip-check",
             T("pin_2N_12sm_8192tok_qpauto_skipchk_rep1",
               "pin_2N_12sm_8192tok_qpauto_skipchk_rep2")),
            ("release, full 5-var route B",
             T("official_2N_12sm_8192tok_qpdefault_nodbg_routeB_rep1")),
            ("release, the type-5 pair", O2_12_8192_T5),
            ("main 8e7b42e + the type-5 pair",
             T("main_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep1",
               "main_2N_12sm_8192tok_qpdefault_nodbg_gin5_rep2"))]:
        so = arm(tags, "dispatch", 0)[0]
        out("%-42s %-10s %-8s %.1f%%\n"
            % (name, f(arm(tags, "dispatch")[0]), f(so),
               so * 0.5 / WIRE_PER_GPU * 100))  # N=2 -> (N-1)/N = 0.5

    out("\n\nDECODE LATENCY -- 24 SM, type 5, 128 tok\n")
    out("---------------------------------------\n")
    out("op                  2N time    4N time\n")
    for op in OPS:
        out("%-19s %-10s %s\n"
            % (op, f(arm(SMD_2N[2][1], op)[0]), f(arm(SMD_4N[2][1], op)[0])))
    d2 = arm(SMD_2N[2][1], "dispatch")[0]
    d4 = arm(SMD_4N[2][1], "dispatch")[0]
    c2 = arm(SMD_2N[2][1], "combine")[0]
    c4 = arm(SMD_4N[2][1], "combine")[0]
    out("2N -> 4N: dispatch +%.1f us (+%.0f%%), combine +%.0f%%\n"
        % (d4 - d2, 100 * (d4 - d2) / d2, 100 * (c4 - c2) / c2))

    out("\n\nPREFILL BANDWIDTH -- 24 SM, type 5, 8192 tok\n")
    out("--------------------------------------------\n")
    out("time is a mean; SO/SU are min-max ACROSS RANKS\n")
    out("op                  2N time    SO      SU        4N time    SO      SU\n")
    for op in OPS:
        c = []
        for tags in (SPREAD_2N, SPREAD_4N):
            c += [f(arm(tags, op)[0]),
                  "%d-%d" % rng(tags, op, 0), "%d-%d" % rng(tags, op, 1)]
        out("%-19s %-10s %-7s %-9s %-10s %-7s %s\n" % tuple([op] + c))

    out("\n\nSPREAD -- where the SO/SU range comes from\n")
    out("------------------------------------------\n")
    for label, tags, N in [("2 nodes / 24 SM / 8192 tok", SPREAD_2N, 2),
                           ("4 nodes / 24 SM / 8192 tok", SPREAD_4N, 4)]:
        out("\n%s\n" % label)
        out("op                  cross-rank spread%: time  bytes   SO      "
            "| rep-to-rep spread% of the all-rank mean\n")
        for op in OPS:
            us = [v[2] for t, (o, W) in runs(tags) if op in o for v in o[op].values()]
            by = [v[3] for t, (o, W) in runs(tags) if op in o for v in o[op].values()]
            so = [v[0] for t, (o, W) in runs(tags) if op in o for v in o[op].values()]
            # round the reps first so the printed spread% is derivable from the
            # printed rep values -- a reader must be able to check it by hand
            _, reps, _ = arm(tags, op)
            reps = [round(r, 1) for r in reps]
            def s(xs):
                return 100.0 * (max(xs) - min(xs)) / st.mean(xs)
            out("%-19s %22.1f %6.1f %6.1f     | %.2f%%  (reps %s)\n"
                % (op, s(us), s(by), s(so), s(reps) if len(reps) > 1 else 0.0,
                   "/".join("%.1f" % r for r in reps)))
        d = arm(tags, "dispatch", 0)[0]
        out("one printed GB/s unit is %.2f%% of SO at this magnitude (SO mean %.1f), "
            "so a 2-integer range is the floor\n" % (100.0 / d, d))
        so_mb = [v[0] * v[2] * 1e-6 * 1e9 / 1e6 for t, (o, W) in runs(tags)
                 for v in o["dispatch"].values()]
        su_mb = [v[3] / 1e6 for t, (o, W) in runs(tags) for v in o["dispatch"].values()]
        out("dispatch bytes/rank: scale-up (printed) %.1f-%.1f MB, "
            "scale-out (SO*t, incl intra-node) %.1f-%.1f MB\n"
            % (min(su_mb), max(su_mb), min(so_mb), max(so_mb)))
        out("wire%% = SO*(N-1)/N/50: %.1f%%\n" % (d * (N - 1) / N / WIRE_PER_GPU * 100))

    out("\n\nNODE LAYERING -- per-node all-rank means, 24 SM / 8192 tok, type 5\n")
    out("------------------------------------------------------------------\n")
    for label, tags in [("2 nodes", SPREAD_2N), ("4 nodes", SPREAD_4N)]:
        out("\n%s\n" % label)
        for op in ("dispatch", "combine", "reduced combine"):
            per = {}
            for t, (o, W) in runs(tags):
                nranks_per_node = W // len(glob.glob("%s/%s.node*.log" % (EPRUNS, t)))
                for r, v in o[op].items():
                    per.setdefault(r // nranks_per_node, []).append(v[2])
            means = {k: st.mean(v) for k, v in sorted(per.items())}
            vals = list(means.values())
            out("  %-18s %s   layer spread %.1f%%\n"
                % (op, "  ".join("node%d %.1f" % (k + 1, v) for k, v in means.items()),
                   100.0 * (max(vals) - min(vals)) / st.mean(vals)))

    out("\n\nCOMPLETENESS AUDIT -- every tag used above\n")
    out("------------------------------------------\n")
    bad = 0
    for t in sorted(CACHE):
        ops, W = CACHE[t]
        miss = {op: W - len(ops[op]) for op in ops if len(ops[op]) != W}
        if miss:
            bad += 1
            out("%s  world=%d  MISSING %s\n" % (t, W, miss))
    out("%d of %d tags incomplete\n" % (bad, len(CACHE)))


main()

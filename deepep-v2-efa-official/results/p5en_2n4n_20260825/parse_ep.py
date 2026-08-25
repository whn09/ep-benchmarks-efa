#!/usr/bin/env python3
"""Aggregate DeepEP test_ep.py per-rank lines across every node log of one run tag.

usage: [EPRUNS=<dir>] parse_ep.py <tag-without-.nodeN.log> [more tags...]

This is the per-tag inspector. The published tables come from make_tables.py in
this directory, which pools the reps of each arm -- use that one to regenerate
summary.txt / README.md / docs/runbook_zh.md rather than transcribing by hand.

Per op prints: n ranks seen, min/max SO GB/s, min/max SU GB/s, mean/min/max us,
MB/rank, and the cross-rank spread of time vs bytes -- the SO/SU range is a
cross-rank min/max, and for dispatch it is driven by the byte denominator (each
rank routes a different number of tokens), not by time. See runbook_zh.md 5.1.

`bytes` in the log is num_scaleup_bytes (test_ep.py:271), so SU = bytes / time.
Scale-out bytes are not printed; SO * time recovers them (SO includes intra-node
traffic unless the run passed --ignore-local-traffic).

Use finditer, not search: concurrent ranks writing to the same stdout regularly
glue two per-rank lines onto one physical line, and search() would silently drop
the second one. A short `n=` is the tell that lines were lost some other way.
"""
import glob
import os
import re
import sys

LINE = re.compile(
    r"EP:\s+(\d+)/(\d+)\s*\|\s*([a-z ]+?):\s*(\d+) GB/s \(SO\), (\d+) GB/s \(SU\), "
    r"([\d.]+) us, (\d+) bytes"
)
ORDER = ["dispatch", "expanded dispatch", "cached dispatch", "combine", "reduced combine"]
EPRUNS = os.environ.get("EPRUNS", "/tmp/epruns")


def spread(xs):
    mean = sum(xs) / len(xs)
    return 100.0 * (max(xs) - min(xs)) / mean if mean else 0.0


def main():
    for tag in sys.argv[1:]:
        files = sorted(glob.glob("%s/%s.node*.log" % (EPRUNS, tag)))
        if not files:
            print("%s: NO LOGS in %s" % (tag, EPRUNS))
            continue
        ops = {}
        world = set()
        for f in files:
            for line in open(f, errors="replace"):
                for m in LINE.finditer(line):
                    rank, nranks, op, so, su, us, nbytes = m.groups()
                    world.add(int(nranks))
                    ops.setdefault(op.strip(), {})[int(rank)] = (
                        int(so), int(su), float(us), int(nbytes))
        expect = max(world) if world else 0
        print("=== %s   (%d node logs, world=%d)" % (tag, len(files), expect))
        for op in ORDER + [o for o in ops if o not in ORDER]:
            if op not in ops:
                continue
            v = list(ops[op].values())
            so = [x[0] for x in v]
            su = [x[1] for x in v]
            us = [x[2] for x in v]
            by = [x[3] for x in v]
            short = "" if len(v) == expect else "  <-- MISSING %d rank(s)" % (expect - len(v))
            print("  %-18s n=%2d  SO %d-%d  SU %d-%d  us mean %.1f (%.1f-%.1f)  "
                  "MB/rank %.1f-%.1f  spread%% t/bytes %.1f/%.1f%s"
                  % (op, len(v), min(so), max(so), min(su), max(su),
                     sum(us) / len(us), min(us), max(us),
                     min(by) / 1e6, max(by) / 1e6,
                     spread(us), spread(by), short))


main()

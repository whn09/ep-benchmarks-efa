#!/usr/bin/env python3
"""Aggregate DeepEP test_ep.py per-rank lines across every node log of one run tag.

usage: parse_ep.py <tag-without-.nodeN.log> [more tags...]
Prints, per op: n ranks, min/max SO GB/s, min/max SU GB/s, mean/min/max us, bytes/rank.
"""
import glob
import re
import sys

LINE = re.compile(
    r"EP:\s+(\d+)/(\d+)\s*\|\s*([a-z ]+?):\s*(\d+) GB/s \(SO\), (\d+) GB/s \(SU\), "
    r"([\d.]+) us, (\d+) bytes"
)
ORDER = ["dispatch", "expanded dispatch", "cached dispatch", "combine", "reduced combine"]


def main():
    for tag in sys.argv[1:]:
        files = sorted(glob.glob("/tmp/epruns/%s.node*.log" % tag))
        if not files:
            print("%s: NO LOGS" % tag)
            continue
        ops = {}
        for f in files:
            for line in open(f, errors="replace"):
                m = LINE.search(line)
                if not m:
                    continue
                rank, _, op, so, su, us, nbytes = m.groups()
                ops.setdefault(op.strip(), {})[int(rank)] = (
                    int(so), int(su), float(us), int(nbytes))
        print("=== %s   (%d node logs)" % (tag, len(files)))
        for op in ORDER + [o for o in ops if o not in ORDER]:
            if op not in ops:
                continue
            v = list(ops[op].values())
            so = [x[0] for x in v]
            su = [x[1] for x in v]
            us = [x[2] for x in v]
            by = [x[3] for x in v]
            print("  %-18s n=%2d  SO %d-%d  SU %d-%d  us mean %.1f (%.1f-%.1f)  "
                  "MB/rank %.1f-%.1f"
                  % (op, len(v), min(so), max(so), min(su), max(su),
                     sum(us) / len(us), min(us), max(us),
                     min(by) / 1e6, max(by) / 1e6))


main()

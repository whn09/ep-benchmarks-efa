"""Summarise EPTL timeline records: per kernel variant, the median (over every
complete record = one block of one rank in one sampled call) of each event's
first/last stamp, in us relative to that block's kernel entry."""
import re, sys, glob, statistics as st, collections
REC = re.compile(r'EPTL (disp|comb) v(\d+) r(\d+) b(\d+) c(\d+) t0 (\d+)((?: -?\d+){22})(?=\s*(?:\n|$))', re.M)
NAMES = {
 'disp': {1: 'start barrier exit', 2: 'scale-out RDMA put (first..last)', 3: 'scale-out role done',
          4: 'forward: source arrived (first..last)', 5: 'forward loop done', 6: 'forward drain done',
          7: 'notify role done', 8: 'end NVLink barrier exit'},
 'comb': {1: 'start barrier exit', 2: 'scale-up (NVLink push) done', 3: 'forward staging done',
          4: 'remote arrivals complete', 5: 'proxy RDMA put (first..last)', 6: 'proxy done',
          7: 'batch handed to proxy (first..last)'},
}
VAR = {('disp', 4): 'dispatch', ('disp', 5): 'expanded dispatch', ('disp', 6): 'cached dispatch',
       ('disp', 7): 'cached+expanded', ('comb', 0): 'combine', ('comb', 1): 'reduced combine'}
def main(paths):
    recs = collections.defaultdict(list); raw = 0
    for p in paths:
        txt = open(p, errors='ignore').read()
        raw += txt.count('EPTL ')
        for m in REC.finditer(txt):
            k, v, r, b, c, t0 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
            vals = [int(x) for x in m.group(7).split()]
            recs[(k, v)].append((r, b, c, t0, vals))
    good = sum(len(x) for x in recs.values())
    print(f'records: {good} complete of {raw} raw ({good / max(raw, 1):.0%})')
    for key in sorted(recs):
        rs = recs[key]; k = key[0]
        ranks = len({r for r, *_ in rs})
        print(f'\n== {VAR.get(key, key)}  ({len(rs)} block-records, {ranks} ranks)')
        print(f'   {"event":42s} {"first":>8s} {"last":>8s}   (median us since block entry; p90 of last)')
        for e, name in NAMES[k].items():
            lo = [vals[2 * (e - 1)] for *_, vals in rs if vals[2 * (e - 1)] >= 0]
            hi = [vals[2 * (e - 1) + 1] for *_, vals in rs if vals[2 * (e - 1) + 1] >= 0]
            if not lo: continue
            p90 = sorted(hi)[int(0.9 * (len(hi) - 1))]
            print(f'   {e}: {name:39s} {st.median(lo) / 1e3:8.1f} {st.median(hi) / 1e3:8.1f}   p90 {p90 / 1e3:6.1f}  n={len(lo)}')
        # block-entry skew inside one rank/call: spread of t0 across that rank's blocks
        by = collections.defaultdict(list)
        for r, b, c, t0, _ in rs: by[(r, c)].append(t0)
        sk = [ (max(v) - min(v)) / 1e3 for v in by.values() if len(v) > 1]
        if sk: print(f'   block-entry spread within a rank: median {st.median(sk):.1f} us')
        # rank-entry skew across ranks on the same node is meaningful only intra-node (same clock)
if __name__ == '__main__':
    main(sys.argv[1:])

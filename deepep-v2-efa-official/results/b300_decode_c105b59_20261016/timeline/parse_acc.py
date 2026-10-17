"""Per-token scale-up phase cost from accumulator events 9..11 of `comb` records."""
import re, sys, collections, statistics as st
REC = re.compile(r'EPTL comb v(\d+) r(\d+) b(\d+) c(\d+) t0 (\d+)((?: -?\d+){22})(?=\s*(?:\n|$))', re.M)
ghz = float(sys.argv[1]); paths = sys.argv[2:]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for p in paths:
    for m in REC.finditer(open(p, errors='ignore').read()):
        v = int(m.group(1)); vals = [int(x) for x in m.group(6).split()]
        for e in (9, 10, 11):
            tot, cnt = vals[2 * (e - 1)], vals[2 * (e - 1) + 1]
            if cnt > 0: agg[v][e].append((tot, cnt))
names = {9: 'metadata / routing', 10: 'load issue (direct) | register reduce', 11: 'wait load + store issue'}
for v in sorted(agg):
    print(f"== {'reduced combine' if v else 'combine'}")
    tokens = [c for t, c in agg[v][9]]
    print(f"   tokens per block per call: median {st.median(tokens):.0f}  (5 scale-up warps per block at 8 SM)")
    tot_all = 0
    for e in (9, 10, 11):
        per_tok = [t / c / ghz / 1e3 for t, c in agg[v][e]]
        tot_all += st.median(per_tok)
        print(f"   {e}: {names[e]:40s} {st.median(per_tok):6.2f} us/token")
    print(f"   sum: {tot_all:.2f} us/token  x tokens/warp {st.median(tokens)/5:.1f} = {tot_all*st.median(tokens)/5:.1f} us per warp")

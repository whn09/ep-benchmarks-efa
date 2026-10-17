"""Align the dispatch kernel's end with its copy epilogue, per (rank, sampled call), on
the same GPU clock. Uses dispatch v4 (plain dispatch) and copy v0 (its epilogue)."""
import re, sys, collections, statistics as st
REC = re.compile(r'EPTL (disp|copy) v(\d+) r(\d+) b(\d+) c(\d+) t0 (\d+)((?: -?\d+){22})(?=\s*(?:\n|$))', re.M)
d = collections.defaultdict(list); c = collections.defaultdict(list)
for p in sys.argv[1:]:
    for m in REC.finditer(open(p, errors='ignore').read()):
        k, v, r, call, t0 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(5)), int(m.group(6))
        vals = [int(x) for x in m.group(7).split()]
        if k == 'disp' and v == 4: d[(r, call)].append((t0, vals))
        if k == 'copy' and v == 0: c[(r, call)].append((t0, vals))
rows = collections.defaultdict(list)
# Pair each copy call with the dispatch call that ended closest BEFORE it on the same
# rank (call counters of the two JIT modules are not guaranteed to line up).
dend = collections.defaultdict(list)
for (r, call), ds in d.items():
    d_start = min(t0 for t0, _ in ds)
    d_end = max(t0 + v[15] for t0, v in ds if v[15] >= 0)
    dend[r].append((d_end, d_start))
npairs = 0
for (r, call), cs in c.items():
    c_entry_first = min(t0 for t0, _ in cs); c_entry_last = max(t0 for t0, _ in cs)
    cand = [(c_entry_first - e, e, s0) for e, s0 in dend[r] if -50000 <= c_entry_first - e <= 200000]
    if not cand: continue
    _, d_end, d_start = min(cand, key=lambda x: abs(x[0]))
    c_sync = max(t0 + v[1] for t0, v in cs if v[1] >= 0)
    c_end = max(t0 + v[3] for t0, v in cs if v[3] >= 0)
    npairs += 1
    rows['dispatch kernel (first entry -> last end)'].append(d_end - d_start)
    rows['copy first block entry - dispatch end'].append(c_entry_first - d_end)
    rows['copy last block entry  - dispatch end'].append(c_entry_last - d_end)
    rows['copy GridDepSync done  - dispatch end'].append(c_sync - d_end)
    rows['copy work (sync done -> last block end)'].append(c_end - c_sync)
    rows['copy kernel span (first entry -> last end)'].append(c_end - c_entry_first)
print(f'{npairs} (rank, call) pairs; medians [p10, p90] in us')
for k, v in rows.items():
    v = sorted(v); q = lambda f: v[int(f * (len(v) - 1))] / 1e3
    print(f'  {k:45s} {st.median(v) / 1e3:7.1f}  [{q(0.1):6.1f}, {q(0.9):6.1f}]')

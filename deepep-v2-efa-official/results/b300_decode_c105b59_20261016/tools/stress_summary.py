"""Summarise ep_determinism.py runs: per run, iterations and per-tensor mismatches over ALL
ranks of both nodes (finditer: rank lines can be glued). recv_raw always mismatches on the
unordered kernels (arrival order) and is not a correctness signal."""
import re, sys, glob, collections
PAT = re.compile(r'EP_DET rank=(\d+) mode=deepep det=0 iters=(\d+)((?: [a-z_]+:mis=\d+,nan=\d+)+)')
for tag in sys.argv[2:]:
    ranks = {}
    for f in glob.glob(f'{sys.argv[1]}/det_{tag}.node*.log'):
        for m in PAT.finditer(open(f, errors='ignore').read()):
            ranks[int(m.group(1))] = (int(m.group(2)), dict(
                (k, (int(a), int(b))) for k, a, b in re.findall(r'([a-z_]+):mis=(\d+),nan=(\d+)', m.group(3))))
    if not ranks: print(f'{tag:6s} no results'); continue
    iters = sorted({v[0] for v in ranks.values()})
    bad = collections.defaultdict(list)
    for r, (_, d) in sorted(ranks.items()):
        for k, (mis, nan) in d.items():
            if k != 'recv_raw' and (mis or nan): bad[k].append(f'r{r}:mis={mis},nan={nan}')
    print(f'{tag:6s} ranks={len(ranks):2d} iters={iters} ' + ('CLEAN' if not bad else ' '.join(f'{k}[{", ".join(v)}]' for k, v in bad.items())))

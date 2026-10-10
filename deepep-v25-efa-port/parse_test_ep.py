import re,glob,collections,sys
pat=re.compile(r'EP:\s+(\d+)/(\d+) \| (dispatch|expanded dispatch|cached dispatch|combine|reduced combine): .*?, ([\d.]+) us, \d+ bytes \| (copy|reduce): \d+ GB/s, ([\d.]+) us')
runs=collections.defaultdict(lambda: collections.defaultdict(dict))
for f in glob.glob(sys.argv[1]+'/*.log'):
    tag=f.split('/')[-1].split('.node')[0]
    for m in pat.finditer(open(f).read()):
        r,n,op,t,_,t2=m.groups(); runs[tag][op][int(r)]=(float(t),float(t2))
ops=['dispatch','cached dispatch','combine','reduced combine']
print('tag'.ljust(14),' '.join(f'{o:>16} {"(epi)":>7}' for o in ops),' ranks')
agg=collections.defaultdict(list)
for tag in sorted(runs):
    row=[]; nr=min(len(runs[tag][o]) for o in ops)
    for o in ops:
        v=list(runs[tag][o].values()); a=sum(x[0] for x in v)/len(v); b=sum(x[1] for x in v)/len(v); row.append(f'{a:16.1f} {b:7.1f}'); agg[(tag.rsplit('_r',1)[0],o)].append((a,b))
    print(tag.ljust(14),' '.join(row),nr)
print('-- mean of runs')
for arm in sorted({k[0] for k in agg}):
    print(arm.ljust(14),' '.join(f'{sum(x[0] for x in agg[(arm,o)])/len(agg[(arm,o)]):16.1f} {sum(x[1] for x in agg[(arm,o)])/len(agg[(arm,o)]):7.1f}' for o in ops))

import re,glob,sys,collections,statistics as st
pat=re.compile(r'EP:\s*(\d+)/(\d+) \| ([a-z ]+?): .*?([\d.]+) us(?:, \d+ bytes \| (copy|reduce): [\d.]+ GB/s, ([\d.]+) us)?')
D=sys.argv[1]; tok=sys.argv[2]; ext=sys.argv[3] if len(sys.argv)>3 else 'default'
ops=['dispatch','cached dispatch','combine','reduced combine']
def cell(tag):
    vals=collections.defaultdict(dict); epi=collections.defaultdict(dict); nodes=0
    for f in sorted(glob.glob(f'{D}/{tag}.node*.log')):
        nodes+=1; txt=open(f,errors='ignore').read().split('Testing with')
        txt=txt[1] if len(txt)>1 else ''
        for m in pat.finditer(txt):
            r,op,us=int(m.group(1)),m.group(3).strip(),float(m.group(4))
            if r in vals[op]: continue
            vals[op][r]=us
            if m.group(6): epi[op][r]=float(m.group(6))
    return vals,epi
res={}
for arm in ['main','pr89','pr16','pr17']:
    reps=[]
    for rep in (1,2,3):
        v,e=cell(f'{arm}_1N_12sm_{tok}tok_{ext}_nodbg_gin5_ovlp0_rep{rep}')
        if all(len(v.get(o,{}))==8 for o in ops): reps.append((v,e))
    if not reps: continue
    row={}
    for o in ops:
        ms=[st.mean(v[o].values()) for v,e in reps]; es=[st.mean(e[o].values()) for v,e in reps if e.get(o)]
        row[o]=(st.mean(ms),min(ms),max(ms),st.mean(es) if es else 0)
    res[arm]=(row,len(reps))
base=res['main'][0]
print(f'{tok} tok, 1N/8 ranks, 12 SM, GIN5, ovlp0 -- mean of reps (each = 16-rank mean), us')
print(f'{"arm":6s} reps ' + ' '.join(f'{o:>22s}' for o in ops) + '   disp+redComb  +epilogues')
for arm,(row,n) in res.items():
    cells=[]
    for o in ops:
        m,lo,hi,ep=row[o]; d=(m/base[o][0]-1)*100
        cells.append(f'{m:7.1f} ({d:+5.1f}%) [{hi-lo:4.1f}]')
    layer=row['dispatch'][0]+row['reduced combine'][0]
    full=layer+row['dispatch'][3]+row['reduced combine'][3]
    bl=base['dispatch'][0]+base['reduced combine'][0]
    print(f'{arm:6s} {n:4d} '+' '.join(f'{c:>22s}' for c in cells)+f'   {layer:6.1f} ({(layer/bl-1)*100:+5.1f}%)  {full:6.1f}')
print('epilogue us (mean): ' + ', '.join(f'{a}: copy {r["dispatch"][3]:.1f} / reduce {r["reduced combine"][3]:.1f}' for a,(r,n) in res.items()))

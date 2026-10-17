import re,glob,sys,collections,statistics as st
pat=re.compile(r'EP:\s*(\d+)/(\d+) \| ([a-z ]+?): .*?([\d.]+) us(?:, \d+ bytes \| (copy|reduce): [\d.]+ GB/s, ([\d.]+) us)?')
D,arm,sm,tok=sys.argv[1:5]; knobs=sys.argv[5].split(','); reps=int(sys.argv[6]) if len(sys.argv)>6 else 2
ops=['dispatch','cached dispatch','combine','reduced combine']
def cell(tag):
    v=collections.defaultdict(dict); e=collections.defaultdict(dict)
    for f in glob.glob(f'{D}/{tag}.node*.log'):
        t=open(f,errors='ignore').read().split('Testing with'); t=t[1] if len(t)>1 else ''
        for m in pat.finditer(t):
            r,op,us=int(m.group(1)),m.group(3).strip(),float(m.group(4))
            if r in v[op]: continue
            v[op][r]=us
            if m.group(6): e[op][r]=float(m.group(6))
    return v,e
print(f'{arm} {sm} SM {tok} tok, 2N/16 ranks, mean of reps; [rep range]')
print(f'{"knob":7s}'+''.join(f'{o:>20s}' for o in ops)+'   disp+redC   +epi  n')
base=None
for k in knobs:
    rs=[]
    for rep in range(1,reps+1):
        v,e=cell(f'{arm}_2N_{sm}sm_{tok}tok_{k}_nodbg_gin5_ovlp0_rep{rep}')
        if all(len(v.get(o,{}))==16 for o in ops): rs.append((v,e))
    if not rs: print(k,'incomplete'); continue
    row={o:(st.mean([st.mean(v[o].values()) for v,e in rs]), max(st.mean(v[o].values()) for v,e in rs)-min(st.mean(v[o].values()) for v,e in rs), st.mean([st.mean(e[o].values()) for v,e in rs if e.get(o)] or [0])) for o in ops}
    layer=row['dispatch'][0]+row['reduced combine'][0]; full=layer+row['dispatch'][2]+row['reduced combine'][2]
    if base is None: base=layer
    print(f'{k:7s}'+''.join(f'{row[o][0]:12.1f} [{row[o][1]:4.1f}]' for o in ops)+f'   {layer:7.1f} ({(layer/base-1)*100:+5.1f}%) {full:6.1f}  {len(rs)}')

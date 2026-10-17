import re,glob,sys,collections,statistics as st
pat=re.compile(r'EP:\s*(\d+)/(\d+) \| ([a-z ]+?): .*?([\d.]+) us(?:, \d+ bytes \| (copy|reduce): [\d.]+ GB/s, ([\d.]+) us)?')
D=sys.argv[1]; ops=['dispatch','cached dispatch','combine','reduced combine']
def cell(tag):
    v=collections.defaultdict(dict); e=collections.defaultdict(dict); qps=set()
    for f in glob.glob(f'{D}/{tag}.node*.log'):
        t=open(f,errors='ignore').read(); qps|=set(re.findall(r'#QPs: (\d+/\d+)',t)); t=t.split('Testing with'); t=t[1] if len(t)>1 else ''
        for m in pat.finditer(t):
            r,op,us=int(m.group(1)),m.group(3).strip(),float(m.group(4))
            if r in v[op]: continue
            v[op][r]=us
            if m.group(6): e[op][r]=float(m.group(6))
    return v,e,qps
print('128 tok, 2N/16 ranks, GIN5, ovlp0, 12 SM, channel cap, mean of 2 reps; [rep range]')
print(f'{"arm":5s} {"SM":>3s} {"QPs":>6s} '+''.join(f'{o:>20s}' for o in ops)+'   disp+redC  +epi')
for arm in ['main','pr89']:
    for sm in ['ch1','ch2','ch3','ch4','choff']:
        reps=[];q=set()
        for rep in (1,2):
            v,e,qq=cell(f'{arm}_2N_12sm_128tok_{sm}_nodbg_gin5_ovlp0_rep{rep}'); q|=qq
            if all(len(v.get(o,{}))==16 for o in ops): reps.append((v,e))
        if not reps: print(arm,sm,'incomplete'); continue
        row={}
        for o in ops:
            ms=[st.mean(v[o].values()) for v,e in reps]; es=[st.mean(e[o].values()) for v,e in reps if e.get(o)]
            row[o]=(st.mean(ms),max(ms)-min(ms),st.mean(es) if es else 0)
        layer=row['dispatch'][0]+row['reduced combine'][0]; full=layer+row['dispatch'][2]+row['reduced combine'][2]
        print(f'{arm:5s} {sm:>5s} {",".join(sorted(q)):>6s} '+''.join(f'{row[o][0]:12.1f} [{row[o][1]:4.1f}]' for o in ops)+f'   {layer:7.1f}  {full:6.1f}  (n={len(reps)})')

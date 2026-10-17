import re,glob,sys,collections,statistics as st
pat=re.compile(r'EP:\s*(\d+)/(\d+) \| ([a-z ]+?): .*?([\d.]+) us, \d+ bytes \| (copy|reduce): [\d.]+ GB/s, ([\d.]+) us')
D=sys.argv[1]
rows=[]
for tag in sys.argv[2:]:
    per=collections.defaultdict(list)
    for f in glob.glob(f'{D}/{tag}_rep*.node*.log'):
        t=open(f,errors='ignore').read().split('Testing with'); t=t[1] if len(t)>1 else ''
        seen=set()
        for m in pat.finditer(t):
            key=(m.group(1),m.group(3).strip())
            if key in seen: continue
            seen.add(key); per[m.group(3).strip()].append((float(m.group(4)),float(m.group(6))))
    if not per: print(tag,'none'); continue
    d=per.get('dispatch',[]); rc=per.get('reduced combine',[])
    print(f'{tag:58s} disp {st.mean(x for x,_ in d):6.1f} copy {st.mean(y for _,y in d):5.1f} [{min(y for _,y in d):4.1f}-{max(y for _,y in d):4.1f}] | redComb {st.mean(x for x,_ in rc):6.1f} reduce {st.mean(y for _,y in rc):5.1f}')

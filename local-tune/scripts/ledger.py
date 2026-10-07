#!/usr/bin/env python3
# Build the results page (ledger.html) from results/ and ledger.template.html.
# Usage: ledger.py [out.html]. Verdicts are in DEC below; numbers come from the csv files.
import csv, json, os, re, statistics as st, sys
from paths import rp, LOCAL, SCRIPTS
HERE=SCRIPTS
ITERS=[
 dict(id='b1',n='1',name='Starting build, earlier session',desc='Card treated as if it had tensor cores. Long-context numbers are single runs.'),
 dict(id='b2',n='1',name='Starting build, tonight',desc='Same code, measured again tonight with 3 repetitions. The fair reference for build 6.'),
 dict(id='f16',n='2',name='Forced f16 attention',desc='Experiment: compressed conversation memory converted to f16 before attention. Rejected.'),
 dict(id='fix1',n='3',name='Detection fix, first version',desc='Card correctly flagged as having no tensor cores. f16 memory fell to the tile kernel.'),
 dict(id='fix1f16',n='4',name='Fix + forced f16 attention',desc='Builds 2 and 3 combined. Rejected.'),
 dict(id='off',n='5',name='Fix switched off by env var',desc='Build 3 with GGML_CUDA_FORCE_TURING_MMA=1. Shows the switch restores the old behavior.'),
 dict(id='final',n='6',name='Fix + vector kernel for f16 (final)',desc='Build 3 plus the two small edits below. Committed.'),
]
def rd(f, pre=0):
    out=[]
    for r in csv.reader(open(rp(f))):
        try: ts=float(r[-2]); sd=float(r[-1]); int(r[-8])
        except Exception: continue
        out.append(dict(pre=r[:pre], model=os.path.basename(r[pre+5]), ngl=r[pre+17], np=int(r[-8]), ng=int(r[-7]), depth=int(r[-6]), t=r[-5], ts=ts, sd=sd))
    return out
BENCH={'b1':'base1','b2':'final-base','fix1':'exp4-nomma','off':'exp4-forcedmma','final':'final-fix'}
KV={'b1':('base1',1),'b2':('final-base',3),'f16':('exp1-forced',1),'fix1':('exp4-nomma',1),'fix1f16':('exp1-forced-nomma',1),'final':('final-fix',3)}
order=[i['id'] for i in ITERS]
bench={k:rd(v+'.csv') for k,v in BENCH.items()}
kv={k:rd('kvmatrix-'+v[0]+'.csv',2) for k,v in KV.items()}
MODELS=[('deepseek-coder','99','DeepSeek Coder 1.3B, all on GPU'),('qwen2.5-3b','99','Qwen2.5 3B, all on GPU'),('Qwythos','22','Qwythos 9B, 22 layers on GPU'),('Qwythos','0','Qwythos 9B, model on CPU')]
def bench_panels(pp):
    P=[]
    for key,ngl,title in MODELS:
        bars=[]
        for it in order:
            if it not in bench: continue
            x=[r for r in bench[it] if r['model'].startswith(key) and r['ngl']==ngl and bool(r['np'])==pp]
            if x: bars.append(dict(it=it,v=x[0]['ts'],sd=x[0]['sd'],reps=3,showDelta=it=='final'))
        P.append(dict(title=title,unit='tokens/sec',ref='b2',groups=[dict(label='',bars=bars)]))
    return P
def cellv(it,k,nk,d,pp):
    x=[r for r in kv[it] if r['pre'][0]==k and r['pre'][1]==f'nkvo{nk}' and r['depth']==d and bool(r['np'])==pp]
    return x[0] if x else None
def kv_panels(pp):
    P=[]
    for k in ('f16','q8_0','q4_0'):
        for nk in (0,1):
            groups=[]
            for d in (0,8192,16384):
                bars=[]
                for it in order:
                    if it not in kv: continue
                    c=cellv(it,k,nk,d,pp)
                    bars.append(dict(it=it,v=c['ts'] if c else None,sd=c['sd'] if c else None,reps=KV[it][1],showDelta=it=='final'))
                groups.append(dict(label=f'{d//1024} thousand tokens in context' if d else 'empty context',bars=bars))
            P.append(dict(title=f'{k} memory, kept {"in system RAM" if nk else "on the GPU"}',unit='tokens/sec',ref='b2',groups=groups,foot=('The empty-context reading for build 6 was the first test after another job and is low. Back to back it is 55.0 against 54.1 for build 1 (see the kernel section).' if (k,nk,pp)==('f16',0,False) else None)))
    return P
# kernel comparisons
def cmp(name):
    return [dict(label=r[0], np=int(r[-8]), depth=int(r[-6]), ts=float(r[-2]), sd=float(r[-1])) for r in csv.reader(open(rp(f'compare-{name}.csv'))) if len(r)>20]
dec=cmp('f16-decode'); prm=cmp('f16-prompt'); cf=cmp('f16-confirm')
KD=[('tile','fix1','T','Tile kernel (what build 3 used)'),('vector','final','V','Vector kernel (what build 6 uses)'),('tensor','b2','M','Tensor-core kernel (what build 1 uses)')]
def dl(d): return f'{d//1024} thousand tokens in context' if d else 'empty context'
kern=[dict(title='Generation with f16 memory on GPU, by attention kernel',unit='tokens/sec',ref=None,
  groups=[dict(label=dl(d),bars=[dict(cls=c,n=n,name=nm,v=x['ts'],sd=x['sd'],reps=3) for key,c,n,nm in KD for x in dec if x['label'].startswith(key) and x['depth']==d]) for d in (0,8192,14336)],
  foot='T = tile, V = vector, M = tensor-core kernel. 3 repetitions each.')]
PD=[('tile, 2 parallel blocks (fix)','fix1','T2','Tile kernel with temporary buffer, run 1'),('tile, 2 parallel blocks again','fix1','T2','Tile kernel with temporary buffer, run 2'),('tile, 1 block, no temp buffer','final','T1','Tile kernel without temporary buffer, run 1'),('tile, 1 block again','final','T1','Tile kernel without temporary buffer, run 2')]
kern.append(dict(title='Prompt reading with f16 memory on GPU: tile kernel with and without its temporary buffer',unit='tokens/sec',ref=None,
  groups=[dict(label=dl(d),bars=[dict(cls=c,n=n,name=nm,v=x['ts'],sd=x['sd'],reps=5) for key,c,n,nm in PD for x in prm if x['label']==key and x['depth']==d]) for d in (0,4096,8192)],
  foot='T2 = two parallel blocks plus a 14 MiB temporary buffer. T1 = one block, no buffer. Each measured twice, 5 repetitions.'))
CD=[('new build (final)','final'),('new build again','final'),('old build','b2'),('old build again','b2')]
for pp,title in ((False,'Back-to-back check, generation, f16 memory on GPU'),(True,'Back-to-back check, prompt reading, f16 memory on GPU')):
    kern.append(dict(title=title,unit='tokens/sec',ref='b2',
      groups=[dict(label=dl(d),bars=[dict(it=c,v=x['ts'],sd=x['sd'],reps=3,showDelta=(key=='new build (final)')) for key,c in sorted(CD,key=lambda z:z[1]!='b2') for x in cf if x['label']==key and x['depth']==d and bool(x['np'])==pp]) for d in (0,8192)],
      foot='Builds 1 and 6 alternated twice in one sitting, 3 repetitions each, to remove session drift.'))
# timing from logs + csv test start times
def hms(s): h,m,x=map(int,s.split(':')); return h*3600+m*60+x
def kv_timing(it):
    name,reps=KV[it]; rows={}; cur=None
    for l in open(rp(f'kvmatrix-{name}.log'),errors='replace'):
        m=re.match(r'(\d\d:\d\d:\d\d) START ctk=ctv=(\S+) nkvo=(\d)',l)
        if m: cur=(m[2],int(m[3])); rows[cur]=[hms(m[1]),None]; continue
        m=re.match(r'(\d\d:\d\d:\d\d) DONE',l)
        if m and cur: rows[cur][1]=hms(m[1])
    out={}
    for (k,nk),(s,e) in rows.items():
        if e is None: continue
        tests=[r for r in kv[it] if r['pre'][0]==k and r['pre'][1]==f'nkvo{nk}']
        starts=[(hms(r['t'][11:19])-7*3600)%86400 for r in tests]+[e]
        per={}
        for i,r in enumerate(tests):
            per[r['depth']]=per.get(r['depth'],0)+(starts[i+1]-starts[i])%86400
        out[(k,nk)]=dict(total=(e-s)%86400,per=per)
    return out
T={it:kv_timing(it) for it in KV}
tim=[dict(title='Whole long-context test, start to finish',unit='minutes, shorter is better',ref='b2',lower=True,
  groups=[dict(label='all six rows, as run',bars=[dict(it=it,v=round(sum(r['total'] for r in T[it].values())/60,1),reps=1,showDelta=it=='final') for it in order if it in T])],
  foot='Tonight\'s two builds ran 3 repetitions, the earlier ones ran 1. Compare tonight\'s grey 1 with blue 6 for a like-for-like reading. The first row of tonight\'s grey build is missing from its log, so its real total is about 1.5 minutes longer.')]
for d in (0,8192,16384):
    groups=[]
    for k in ('f16','q8_0','q4_0'):
        for nk in (0,1):
            bars=[dict(it=it,v=round(T[it][(k,nk)]['per'][d],1),reps=1,showDelta=it=='final') for it in order if it in T and (k,nk) in T[it] and d in T[it][(k,nk)]['per']]
            if bars: groups.append(dict(label=f'{k} memory {"in RAM" if nk else "on GPU"}',bars=bars))
    tim.append(dict(title=f'Time spent at {dl(d)}',unit='seconds, shorter is better',ref='b2',lower=True,groups=groups,foot=('Fill the context, read 512 tokens, write 32 tokens.' if d else 'Read 512 tokens, write 32 tokens.')+' Tonight\'s builds did this 3 times, earlier builds once.'))
# verdict numbers
def ratio(a,b): return a/b
pps=[]; 
for p in bench_panels(True)+kv_panels(True):
    for g in p['groups']:
        f=[b for b in g['bars'] if b['it']=='final' and b['v']]; b=[b for b in g['bars'] if b['it']=='b2' and b['v']]
        if f and b: pps.append(f[0]['v']/b[0]['v'])
tgq=[]
for k in ('q8_0','q4_0'):
    for d in (0,8192,16384):
        f=cellv('final',k,0,d,False); b=cellv('b2',k,0,d,False) or cellv('b1',k,0,d,False)
        tgq.append((f['ts']/b['ts']-1)*100)
n8=cellv('final','f16',0,8192,False)['ts']
cn=st.mean(x['ts'] for x in cf if x['label'].startswith('new') and x['depth']==8192 and not x['np'])
co=st.mean(x['ts'] for x in cf if x['label'].startswith('old') and x['depth']==8192 and not x['np'])
tiles=[
 dict(k='Prompt reading',v=f'{min(pps):.1f}x to {max(pps):.1f}x',c='good',d='Build 6 against the starting build, same night, every model and context size measured.'),
 dict(k='Generation, q8_0 and q4_0 on GPU',v=f'{min(tgq):+.0f}% to {max(tgq):+.0f}%',c='',d=f'Median {st.median(tgq):+.0f}%. Session-to-session drift on this machine is 1 to 7%, so this is no real change.'),
 dict(k='Generation, f16 on GPU at 8K',v=f'{cn:.1f} vs {co:.1f}',c='warn',d=f'{(cn/co-1)*100:+.0f}% against the starting build, back to back. Build 3 was at 39.2, so the new kernel choice recovers about half the loss.'),
 dict(k='f16 at 8K, out-of-memory stops',v='0 of 9',c='good',d='Build 6 across three separate sittings. Build 3 stopped in 3 of 6. Peak video memory is about 130 to 200 MiB lower than the starting build.'),
]
# stability table
logre=re.compile(r'vram=(\d+) MiB')
def peak(name):
    try: return max(int(m[1]) for m in logre.finditer(open(rp(f'kvmatrix-{name}.log'),errors='replace').read()))
    except Exception: return None
def missing(it):
    out=[]
    for k in ('f16','q8_0','q4_0'):
        for nk in (0,1):
            for d in (0,8192,16384):
                if (k,nk,d)==('f16',0,16384): continue
                if not cellv(it,k,nk,d,False): out.append(f'{k} {"RAM" if nk else "GPU"} {d//1024}K')
    return out
NOTE={'b1':'Earlier repeat at higher desktop memory use: context creation failed 2 of 2.','b2':'','f16':'CUDA out-of-memory abort in the q8_0 GPU row.','fix1':'f16 GPU 8K stopped with out of memory in 3 of 6 repeat runs.','fix1f16':'CUDA out-of-memory abort in the f16 GPU row.','final':'f16 GPU 8K completed in all 9 runs (matrix plus two back-to-back checks).'}
rows=''
for i in ITERS:
    it=i['id']
    if it not in KV: continue
    ms=missing(it); pk=peak(KV[it][0])
    rows+=f'<tr><td><span class="chip c-{it}">{i["n"]}</span>{i["name"]}</td><td>{"<span class=no>"+", ".join(ms)+"</span>" if ms else "<span class=ok>none</span>"}</td><td class="num">{pk} MiB</td><td>{NOTE[it]}</td></tr>'
stab=f'<table><thead><tr><th>Build</th><th>Long-context cells that did not run</th><th>Peak video memory</th><th>Notes</th></tr></thead><tbody>{rows}</tbody></table>'
correct='''<div class="tablewrap"><table><thead><tr><th>Perplexity, lower is better</th><th>Build 1</th><th>Build 6</th></tr></thead><tbody>
<tr><td>f16 memory</td><td class="num">9.2361 &plusmn;0.92</td><td class="num">9.2751 &plusmn;0.92</td></tr>
<tr><td>q8_0 memory</td><td class="num">9.2387 &plusmn;0.92</td><td class="num">9.2413 &plusmn;0.92</td></tr></tbody></table></div>
<div class="stack"><p class="note"><b>Perplexity</b> measures how well the model predicts a reference text (R1-distill 7B, 8 chunks of 512 tokens). The differences are 20 times smaller than the measurement uncertainty, so output quality is unchanged.</p>
<p class="note"><b>Operator tests:</b> the backend's own flash-attention test set passed on the GPU with build 6 <span class="ok">(OK)</span>.</p></div>'''
code='''<div class="stack"><h3>fattn.cu, kernel choice for one-token batches</h3><pre>if (Q-&gt;ne[1] == 1) {
-    if (!gqa_opt_applies) {
+    if (!gqa_opt_applies || cc == GGML_CUDA_CC_TURING_NO_MMA) {
         return BEST_FATTN_KERNEL_VEC;
     }
 }</pre><p class="note">f16 memory during generation now uses the vector kernel instead of falling through to the tile kernel.</p></div>
<div class="stack"><h3>fattn-common.cuh, tile kernel launch</h3><pre>+if (cc == GGML_CUDA_CC_TURING_NO_MMA &amp;&amp; ntiles_dst &gt;= blocks_per_wave) {
+    parallel_blocks = 1;
+}</pre><p class="note">During prompt reading the tile kernel no longer allocates its temporary buffer from the nearly full card. Measured cost: none.</p></div>'''
# at-a-glance grid: one row per build, one column per question
def med(xs): return st.median(xs) if xs else None
def bv(it,key,ngl,pp):
    x=[r for r in bench.get(it,[]) if r['model'].startswith(key) and r['ngl']==ngl and bool(r['np'])==pp]
    return x[0]['ts'] if x else None
def kvv(it,k,nk,d,pp):
    c=cellv(it,k,nk,d,pp) if it in kv else None
    return c['ts'] if c else None
def glance_row(it,ref):
    pp=[]; tg0=[]; lng=[]
    for key,ngl,_ in MODELS:
        for pflag,dst in ((True,pp),(False,tg0)):
            a,r=bv(it,key,ngl,pflag),bv(ref,key,ngl,pflag)
            if a and r: dst.append(a/r)
    for k in ('f16','q8_0','q4_0'):
        for nk in (0,1):
            for d in (0,8192,16384):
                a,r=kvv(it,k,nk,d,True),kvv(ref,k,nk,d,True)
                if a and r: pp.append(a/r)
            a,r=kvv(it,k,nk,0,False),kvv(ref,k,nk,0,False)
            if a and r and not (it=='final' and (k,nk)==('f16',0)): tg0.append(a/r)
    for k in ('q8_0','q4_0'):
        for d in (8192,16384):
            a=kvv(it,k,0,d,False); r=kvv(ref,k,0,d,False) or kvv('b1',k,0,d,False)
            if a and r: lng.append(a/r)
    if it=='final': f8=cn/co
    else:
        a,r=kvv(it,'f16',0,8192,False),kvv(ref,'f16',0,8192,False)
        f8=(a/r if a else 'crash') if it in kv else None
    return med(pp),med(tg0),med(lng),f8
def gcell(x,kind):
    if x is None: return '<td class="g na"><span class="big">&ndash;</span><span class="sm">not measured</span></td>'
    if x=='crash': return '<td class="g bad"><span class="big">stopped</span><span class="sm">out of video memory</span></td>'
    if kind=='x':
        tone='good' if x>=1.5 else ('bad' if x<0.95 else 'same')
        return f'<td class="g {tone}"><span class="big">{x:.1f}x</span><span class="sm">{"much faster" if x>=1.5 else ("slower" if x<0.95 else "same speed")}</span></td>'
    pc=(x-1)*100; tone='good' if pc>3 else ('bad' if pc<-3 else 'same')
    return f'<td class="g {tone}"><span class="big">{pc:+.0f}%</span><span class="sm">{"faster" if pc>3 else ("slower" if pc<-3 else "same speed")}</span></td>'
GST={'f16':('bad','1 crash','in the q8_0 GPU row'),'fix1':('bad','3 of 6','f16 at 8K ran out of memory'),'fix1f16':('bad','2 failures','f16 and q8_0 GPU rows'),'off':('na','&ndash;','not measured'),'final':('good','0 of 9','f16 at 8K, three sittings')}
GV={'f16':('no','Rejected'),'fix1':('no','Superseded'),'fix1f16':('no','Rejected'),'off':('mid','Safety switch'),'final':('yes','Committed')}
grow='<tr><td class="gb"><span class="chip c-b2">1</span><b>Starting build</b></td>'+'<td class="g ref"><span class="big">reference</span><span class="sm">everything below is measured against this</span></td>'*1+'<td class="g ref"></td>'*3+'<td class="g same"><span class="big">1 failure</span><span class="sm">q8_0 on GPU at 16K tonight</span></td><td class="gv"><span class="pill mid">Baseline</span></td></tr>'
for i in ITERS:
    it=i['id']
    if it in ('b1','b2'): continue
    a,b_,c,d=glance_row(it,'b2' if it=='final' else 'b1')
    s=GST[it]; v=GV[it]
    grow+=f'<tr class="{"keep" if it=="final" else ""}"><td class="gb"><span class="chip c-{it}">{i["n"]}</span><b>{i["name"]}</b></td>{gcell(a,"x")}{gcell(b_,"p")}{gcell(c,"p")}{gcell(d,"p")}<td class="g {s[0]}"><span class="big">{s[1]}</span><span class="sm">{s[2]}</span></td><td class="gv"><span class="pill {v[0]}">{v[1]}</span></td></tr>'
glance=f'<table class="glance"><thead><tr><th>Build</th><th>Reading a prompt</th><th>Writing, short chat</th><th>Writing, long chat<br>q8_0 / q4_0 memory on GPU</th><th>Writing, long chat<br>f16 memory on GPU, 8K</th><th>Out-of-memory stops</th><th>Outcome</th></tr></thead><tbody>{grow}</tbody></table>'
D=dict(glance=glance,iters=ITERS,tiles=tiles,benchPP=bench_panels(True),benchTG=bench_panels(False),kvTG=kv_panels(False),kvPP=kv_panels(True),kern=kern,tim=tim,
 benchNote='Builds 2 and 4 only changed long-context behavior and were not run on this suite. <b>Model on CPU:</b> generation runs on the CPU and dropped from 9.9 to 8.6 between sessions for both builds alike, so that is the machine, not the code. Its prompt reading still speeds up because large prompt batches are sent to the GPU.',
 kernNote=f'<p class="note"><b>Generation:</b> the vector kernel beats the tile kernel by 9% at 8 thousand tokens and 18% at 14 thousand. The tensor-core kernel is still the fastest for this one case even though the card only emulates it, which is the remaining 7% gap.</p><p class="note"><b>Prompt reading:</b> dropping the temporary buffer changes nothing measurable, and removes the allocation that pushed build 3 over the memory limit.</p>',
 stab=stab,correct=correct,code=code)
# --- trials after the kernel fix: runtime settings, no code change ---
ALIAS = {'graphs on': 'graphs on (current)', 'fusion on': 'fusion on (default)'}
def base(label):
    l = re.sub(r' again$', '', label).strip()
    return ALIAS.get(l, l)
def cmp_avg(name):
    # (base label, pp?, depth) -> mean t/s and mean spread over the passes
    acc = {}
    for x in cmp(name):
        acc.setdefault((base(x['label']), bool(x['np']), x['depth']), []).append(x)
    return {k: dict(v=st.mean(x['ts'] for x in v), sd=st.mean(x['sd'] for x in v), n=len(v)) for k, v in acc.items()}
def cmp_labels(name):
    return list(dict.fromkeys(base(x['label']) for x in cmp(name)))
def cmp_peak(name):
    out = {}; cur = None
    for l in open(rp(f'compare-{name}.log'), errors='replace'):
        m = re.match(r'\d\d:\d\d:\d\d START (.+)', l)
        if m: cur = base(m[1]); continue
        m = logre.search(l)
        if m and cur: out[cur] = max(out.get(cur, 0), int(m[1]))
    return out
def spec(label):
    acc = {}
    for r in csv.DictReader(open(rp(f'spec-{label}.csv'))):
        acc.setdefault((r['variant'], r['prompt']), []).append(float(r['tg_tps']))
    return {k: dict(v=st.mean(v), sd=st.pstdev(v), n=len(v)) for k, v in acc.items()}
def bar(lab, c, cls='b1', ref=False, fail='does not fit'):
    return dict(cls=cls, n='', name=lab, lab=lab, v=c['v'] if c else None, sd=c['sd'] if c else None, reps=3, ref=ref, showDelta=not ref, fail=fail)
def pick(lab, old, new): return 'b2' if lab == old else 'final' if lab == new else 'b1'

def ngl_panels(files, short, old, new, depths, newlab=', now'):
    data = {}; peak = {}
    for f in files:
        for (lab, pp, d), c in cmp_avg(f).items():
            n = re.match(r'ngl (\d+)', lab)[1]
            data.setdefault((n, pp, d), []).append(c)
        for lab, v in cmp_peak(f).items():
            n = re.match(r'ngl (\d+)', lab)[1]
            peak[n] = max(peak.get(n, 0), v)
    cell = lambda n, pp, d: (lambda cs: dict(v=st.mean(c['v'] for c in cs), sd=st.mean(c['sd'] for c in cs)) if cs else None)(data.get((n, pp, d)))
    ns = sorted({k[0] for k in data}, key=int)
    name = lambda n: ('all layers' if n == '99' else f'{n} layers') + (', before' if n == old else newlab if n == new else '')
    P = [dict(title=f'{short}, writing speed by layers on the GPU', unit='tokens/sec', labels=True,
        groups=[dict(label=dl(d) if d != 15360 else '15 thousand tokens in context', bars=[bar(name(n), cell(n, False, d), pick(n, old, new), n == old) for n in ns]) for d in depths])]
    P.append(dict(title=f'{short}, peak video memory in the run', unit='MiB, desktop included', labels=True, mark=5750, markLabel='contexts fail above about 5750',
        groups=[dict(label='', bars=[dict(cls=pick(n, old, new), n='', name=name(n), lab=name(n), v=peak[n] if cell(n, False, 15360) else None, sd=None, reps=1, fail='does not fit at 16K') for n in ns])]))
    return P, cell
ngl9, c9 = ngl_panels(['ngl-9b-16k', 'ngl-9b-edge'], 'Qwythos 9B', '22', '25', (0, 15360))
ngl7, c7 = ngl_panels(['ngl-7b-16k', 'ngl-7b-edge'], 'R1-distill 7B', '24', '28', (0, 15360), ', picked for q8/q8')

PROMPT = {'rewrite': 'copy a file with one rename', 'edit': 'change one function, print the file', 'refactor': 'add type hints to every function', 'free': 'open question, nothing to copy', 'complete': 'continue a code file'}
def spec_panel(title, label, variants, prompts=None, foot=None, groups=None):
    s = spec(label)
    prompts = prompts or list(dict.fromkeys(k[1] for k in s))
    return dict(title=title, unit='tokens/sec', labels=True, foot=foot,
        groups=[dict(label=PROMPT[p], bars=[bar(lab, s.get((var, p)), cls, i == 0, 'not run') for i, (var, lab, cls) in enumerate(variants)]) for p in prompts])
ON = [('none', 'off', 'b2'), ('ngram-simple', 'ngram-simple', 'final')]
s1, s4 = spec('single-1p3b'), spec('load-1p3b')
ngram = [
    spec_panel('Qwen2.5 3B, all on GPU', 'tune-qwen3b', [('none', 'off', 'b2'), ('simple n12 m48 (default)', 'ngram-simple', 'final')]),
    spec_panel('Qwythos 9B, 25 layers on GPU', 'edit-qwythos9b', ON),
    dict(title='DeepSeek Coder 1.3B, raw completion', unit='tokens/sec', labels=True,
        groups=[dict(label=lab, bars=[bar('off', s[('graphs on (default)', 'complete')], 'b2', True), bar('ngram-simple', s[('ngram-simple', 'complete')], 'final')]) for lab, s in (('one request', s1), ('4 requests at once, total', s4))]),
    spec_panel('R1-distill 7B, all on GPU: not applied', 'r1-7b', [('none', 'off', 'b2'), ('ngram-simple', 'ngram-simple', 'b1')], foot='This model thinks first, so the 400 measured tokens are reasoning text with nothing to copy.'),
    spec_panel('R1-distill 7B, 2000-token answers: not applied', 'r1-7b-long', [('none', 'off', 'b2'), ('ngram-simple', 'ngram-simple', 'b1')], foot='Faster, but the text changes. On the edit prompt the answer was still in the thinking part at 2000 tokens and gave no code; without ngram-simple the code came after 1760 tokens.'),
]
tune = [spec_panel('Lookup sizes and other ngram types, Qwen2.5 3B', 'tune-qwen3b', [('none', 'off', 'b2'), ('simple n12 m48 (default)', 'n=12 m=48, default', 'final'), ('simple n6 m48', 'n=6 m=48', 'b1'), ('simple n8 m48', 'n=8 m=48', 'b1'), ('simple n16 m48', 'n=16 m=48', 'b1'), ('simple n12 m24', 'n=12 m=24', 'b1'), ('simple n12 m96', 'n=12 m=96', 'b1'), ('simple n8 m96', 'n=8 m=96', 'b1'), ('map-k', 'ngram-map-k', 'b1'), ('map-k4v', 'ngram-map-k4v', 'b1')],
    foot='n=6 and n=8 are faster on code but slow the open question by 2 to 7% and change its text. The map types only help on a straight copy.')]
draft = [spec_panel('Qwen2.5 0.5B as draft model for Qwen2.5 3B', 'draft-qwen3b', [('none', 'off', 'b2'), ('ngram-simple', 'ngram-simple, kept', 'final'), ('draft n-max 3 (default)', 'draft, 3 tokens', 'b1'), ('draft n-max 8', 'draft, 8 tokens', 'b1'), ('draft n-max 16', 'draft, 16 tokens', 'b1'), ('draft n-max 16 p-min 0.75', 'draft, 16, p-min 0.75', 'b1'), ('ngram-simple + draft 16', 'ngram + draft', 'b1')],
    foot='Every draft variant slows the open question. The 7B model refuses this draft (its start token differs) and the 9B has no small model with the same vocabulary.')]

def sw_panel(title, name, tests, short=None, foot=None, lower=False):
    a = cmp_avg(name); labs = cmp_labels(name); groups = []
    for pp, d, gl in tests:
        groups.append(dict(label=gl, bars=[bar((short or {}).get(l, l), a.get((l, pp, d)), 'b2' if i == 0 else 'b1', i == 0) for i, l in enumerate(labs)]))
    return dict(title=title, unit='tokens/sec', labels=True, groups=groups, foot=foot)
W, RD = (False, 0, 'writing'), (True, 0, 'reading a prompt')
G = {'graphs on (current)': 'graphs on, default', 'graphs on + GRAPH_OPT': 'with GRAPH_OPT', 'graphs off': 'graphs off'}
FU = {'fusion on (default)': 'fusion on, default', 'fusion off': 'fusion off'}
UB = {'ub 512 (default)': '-ub 512, default', 'ub 1024': '-ub 1024', 'ub 2048': '-ub 2048'}
switches = [
    sw_panel('CUDA Graphs, DeepSeek Coder 1.3B', 'graphs-1p3b', [W, RD], G),
    sw_panel('CUDA Graphs, R1-distill 7B', 'graphs-7b-q8', [W, (False, 8192, 'writing, 8 thousand tokens in context'), RD], G),
    sw_panel('CUDA Graphs, Qwythos 9B hybrid', 'graphs-9b-hybrid', [W, RD], G),
    sw_panel('Operator fusion, DeepSeek Coder 1.3B', 'fusion-1p3b', [W, RD], FU),
    sw_panel('Operator fusion, R1-distill 7B', 'fusion-7b-q8', [W, (False, 8192, 'writing, 8 thousand tokens in context'), RD], FU),
    sw_panel('Operator fusion, Qwythos 9B hybrid', 'fusion-9b-hybrid', [W, RD], FU),
    dict(title='Larger prompt batches (-ub), reading a 2048-token prompt', unit='tokens/sec', labels=True, foot='-ub 2048 does not fit next to the 7B and 9B models. The gain costs 70 to 900 MiB of video memory.',
        groups=[dict(label=gl, bars=[bar(UB[l], cmp_avg(f).get((l, True, 0)), 'b2' if i == 0 else 'b1', i == 0) for i, l in enumerate(UB)]) for f, gl in (('ubatch-1p3b', 'DeepSeek Coder 1.3B'), ('ubatch-3b', 'Qwen2.5 3B'), ('ubatch-7b-q8', 'R1-distill 7B'), ('ubatch-9b-hybrid', 'Qwythos 9B hybrid'))]),
    sw_panel('CPU-side switches, Qwythos 9B hybrid', 'host-9b-hybrid', [W], {'pinned host register': 'REGISTER_HOST=1', 'poll 100': '--poll 100', 'poll 0': '--poll 0', 'prio 2': '--prio 2', 'one thread per core, strict': 'pinned to 6 cores'}, foot='The two baseline passes were 21.3 and 19.1, so every variant is inside the noise.'),
]

# decision ledger: one row per decision, verdict first
def pct(a, b): return f'{(a/b-1)*100:+.0f}%'
def xf(a, b): return f'{a/b:.1f}x'
r7 = spec('r1-7b-long'); t3 = spec('tune-qwen3b'); e9 = spec('edit-qwythos9b'); dr = spec('draft-qwen3b'); DEF = 'simple n12 m48 (default)'
g9 = [e9[('ngram-simple', p)]['v']/e9[('none', p)]['v'] for p in ('edit', 'refactor')]
g3 = [t3[(DEF, p)]['v']/t3[('none', p)]['v'] for p in ('rewrite', 'edit', 'refactor')]
gall = g9+g3
dfree = [dr[(v, 'free')]['v']/dr[('none', 'free')]['v']-1 for v in ('draft n-max 3 (default)', 'draft n-max 8', 'draft n-max 16', 'draft n-max 16 p-min 0.75')]
fus = []
for f in ('fusion-1p3b', 'fusion-7b-q8', 'fusion-9b-hybrid'):
    a = cmp_avg(f)
    fus += [a[('fusion off', False, d)]['v']/a[('fusion on (default)', False, d)]['v']-1 for d in (0, 8192) if ('fusion off', False, d) in a]
ub = [cmp_avg(f)[('ub 1024', True, 0)]['v']/cmp_avg(f)[('ub 512 (default)', True, 0)]['v']-1 for f in ('ubatch-1p3b', 'ubatch-3b', 'ubatch-7b-q8')]
n9o, n9n, n7o, n7n, n7a = c9('22', False, 0)['v'], c9('25', False, 0)['v'], c7('24', False, 0)['v'], c7('28', False, 0)['v'], c7('99', False, 0)['v']
forced = [kvv('f16', k, 0, d, False)/kvv('b1', k, 0, d, False)-1 for k in ('q8_0', 'q4_0') for d in (8192, 16384) if kvv('f16', k, 0, d, False)]
def mix(name, label, pp, depth):
    x = [float(r[-2]) for r in csv.reader(open(rp(f'compare-{name}.csv'))) if r[0].startswith(label) and bool(int(r[-8])) == pp and int(r[-6]) == depth]
    return sum(x)/len(x)
def pplv(name): return {r[0]: float(r[1]) for r in csv.reader(open(rp(f'ppl-{name}.csv'))) if r[1]}
MC = 'kvmix-7b-confirm'; m_old = [mix(MC, 'ngl 28, q8/q8 (shipped)', False, d) for d in (0, 8192, 15360)]; m_new = [mix(MC, 'ngl 99, q8/q4', False, d) for d in (0, 8192, 15360)]
pk = pplv('kv-7b'); p9 = pplv('kv-9b'); fit = spec('fit-7b')
M9 = 'kvmix-9b-ngl'
DEC = [
 ('yes', 'Shipped', 'Plain kernels on GTX 16xx, the card has no tensor cores', tiles[0]['v'], 'faster prompt reading; writing unchanged', 'commit a260d811e', 'fix'),
 ('yes', 'Shipped', 'Qwythos 9B: skip the fused gate path on CPU layers', '+20 to 35%', 'writing with layers on the CPU, 15.6 to 20.6 t/s hybrid', 'commit 52b730eb4', None),
 ('yes', 'Applied', 'R1-distill 7B: 28 layers on the GPU, was 24', pct(n7n, n7o), f'writing, {n7o:.1f} to {n7n:.1f} t/s; {pct(c7("28", False, 15360)["v"], c7("24", False, 15360)["v"])} with 15K in context. Since raised to all 29 layers (next row but one)', 'compare-ngl-7b-*', 'layers'),
 ('yes', 'Applied', 'Qwythos 9B: 25 layers on the GPU, was 22', pct(n9n, n9o), f'writing, {n9o:.1f} to {n9n:.1f} t/s; {pct(c9("25", False, 15360)["v"], c9("22", False, 15360)["v"])} with 15K in context', 'compare-ngl-9b-*', 'layers'),
 ('yes', 'Applied', 'R1-distill 7B: all 29 layers on the GPU, with K q8 / V q4 memory', f'{pct(m_new[0], m_old[0])} to {pct(m_new[2], m_old[2])}', f'writing, {m_old[0]:.1f} to {m_new[0]:.1f} t/s empty and {m_old[2]:.1f} to {m_new[2]:.1f} with 15K in context; perplexity {pk["K q8 / V q8 (shipped)"]:.2f} to {pk["K q8 / V q4"]:.2f}, inside the error. build-live is rebuilt and the gateway line is now -ngl 99 --parallel 1 -c 16384 -ctk q8_0 -ctv q4_0 -fa on', 'compare-kvmix-7b-confirm, ppl-kv-7b', None),
 ('yes', 'Applied', 'ngram-simple on the 9B, 3B and 1.3B models', f'{min(gall):.1f}x to {max(gall):.1f}x', 'writing when the answer repeats the prompt (code edits); no cost on open questions, same text', 'spec-tune-qwen3b, spec-edit-qwythos9b', 'ngram'),
 ('same', 'Default kept', 'CUDA Graphs on or off', '0%', 'no difference larger than the gap between two passes of one variant', 'compare-graphs-*', 'switches'),
 ('same', 'Default kept', 'GGML_CUDA_GRAPH_OPT=1', '0%', 'on llama-server; llama-bench showed +2 to 5% on the 1.3B model only', 'spec-single-1p3b, spec-load-1p3b', 'switches'),
 ('same', 'Default kept', 'ngram-simple lookup sizes, ngram-map-k, ngram-map-k4v', 'n=12 m=48', 'shorter lookups are 9 to 25% faster on code but cost 2 to 7% on open questions and change the text', 'spec-tune-qwen3b', 'ngram'),
 ('same', 'Default kept', 'Larger prompt batches, -ub 1024 and 2048', f'{min(ub)*100:+.1f} to {max(ub)*100:+.1f}%', 'prompt reading, for 70 to 900 MiB of video memory', 'compare-ubatch-*', 'switches'),
 ('same', 'Default kept', 'Pinned host memory, --poll, --prio, core pinning', 'noise', 'all 20.5 to 21.3 t/s against 21.3 and 19.1 for the two baseline passes', 'compare-host-9b-hybrid', 'switches'),
 ('same', 'Default kept', 'Mixed K / V precision as a speed setting', '0%', 'q8/q8, q8/q4, q4/q8 and q4/q4 write at the same speed with the same layer count; the gain is memory only (about 130 MiB at 16K on the 7B)', 'compare-kvmix-7b', None),
 ('same', 'Default kept', 'Qwythos 9B: K q8 / V q4', 'no gain', f'this model keeps its perplexity with every type ({p9["K f16 / V f16"]:.2f} f16, {p9["K q4 / V q4"]:.2f} q4/q4) and its memory is small, so the pair frees almost nothing. 26 layers run with q8/q8 too: {mix(M9, "ngl 26, q8/q8", False, 0):.1f} against {mix(M9, "ngl 25, q8/q8 (shipped)", False, 0):.1f} t/s, a candidate', 'ppl-kv-9b, compare-kvmix-9b-ngl', None),
 ('same', 'Already there', 'Automatic GPU layer selection (--fit)', '25 or 29', f'the default 1024 MiB margin picks 25 of 29 layers ({fit[("fit, default margin 1024 MiB", "free")]["v"]:.1f} t/s); with -fitt 512 it picks all 29, equal to the fixed count ({fit[("fit, margin 512 MiB", "free")]["v"]:.1f} against {fit[("all 29 layers, fixed", "free")]["v"]:.1f} t/s). No new code needed', 'spec-fit-7b', None),
 ('same', 'Already there', 'Dequantize inside the attention kernel; q8 K with DP4A', 'in place', 'the vector kernel reads q4 / q8 memory directly and its K dot product uses dp4a. There is no f16 copy to remove', 'fattn-common.cuh', None),
 ('no', 'Rejected', 'f16-converting attention kernel for q8_0 / q4_0 memory', f'{min(forced)*100:+.0f}%', 'slower or equal in every cell, and the q8_0 16K case stopped running', 'kvmatrix-exp1-forced*', 'fix'),
 ('no', 'Dropped', 'Shared-GQA kernel for quantized memory', 'not built', 'the trial above was its gate and failed', 'TRIALS.md, item 5', None),
 ('no', 'Rejected', 'Draft model (0.5B drafting for the 3B)', f'{min(dfree)*100:+.0f}%', f'on open questions in the worst variant, {max(dfree)*100:+.0f}% in the best; at most 1.7x on code, where ngram-simple gives 3x', 'spec-draft-qwen3b, spec-r1-7b', 'draft'),
 ('no', 'Rejected', 'ngram-mod', 'changes answers', 'faster than ngram-simple, but it remembers earlier requests and the text differs from the baseline', 'spec-qwen3b, spec-qwythos9b', None),
 ('no', 'Rejected', 'GGML_CUDA_DISABLE_FUSION=1', f'{min(fus)*100:+.0f} to {max(fus)*100:+.0f}%', 'writing is slower on every model; prompt reading equal', 'compare-fusion-*', 'switches'),
 ('no', 'Rejected', 'q4_0 for K (keys) on R1-distill 7B', f'{pk["K q4 / V q4"]:.0f}', f'perplexity, against {pk["K f16 / V f16"]:.2f} with f16. Same on the old build and on the CPU, so it is the format and not a kernel fault. The q4_0 rows of the KV matrix are speeds of a broken memory', 'ppl-kv-7b, ppl-kq4-check', None),
 ('no', 'Not built', 'Recent tokens at high precision, older ones lower', 'no case', 'V at q4 is already inside the error for all tokens, and K below q8 fails on the 7B at any age. Nothing is left for a window to win', 'ppl-kv-7b', None),
 ('no', 'Not built', 'Tiled long-context kernel, shared GQA loads', 'earlier result', 'the tile kernel was slower than the vector kernel at long context (38.6 against 42.2 t/s at 8K), and the shared-GQA kernel was dropped with its gate', 'README, TRIALS.md item 5', None),
 ('no', 'Not applied', 'ngram-simple on R1-distill 7B', f'{r7[("ngram-simple", "refactor")]["v"]/r7[("none", "refactor")]["v"]:.1f}x to {r7[("ngram-simple", "edit")]["v"]/r7[("none", "edit")]["v"]:.1f}x', f'on 2000-token answers ({r7[("none", "edit")]["v"]:.0f} to {r7[("ngram-simple", "edit")]["v"]:.0f} t/s on the edit), but the text changes: the edit answer was still thinking at the token limit and gave no code. 0% on 400-token answers', 'spec-r1-7b-long, spec-r1-7b', 'ngram'),
 ('no', 'Not applied', 'R1-distill 7B with all 29 layers on the GPU', pct(n7a, n7n), 'with q8/q8 memory it leaves 180 MiB and fails at 15K. Superseded: it fits with K q8 / V q4 (first row group)', 'compare-ngl-7b-*', 'layers'),
 ('no', 'Not available', 'Lower-bit conversation memory (Turbo4 and similar)', 'no code', 'no such memory type in this tree or in the ik_llama.cpp clone. The 7B result also rules out fewer bits for K; only V could go lower', 'TRIALS.md, section 4', None),
 ('no', 'Not available', 'Draft types with their own draft model, on the 3B, 7B and 1.3B', 'no draft', 'no published draft head for these three models; draft-mtp needs a head inside the model file and the 9B file has none', 'TRIALS.md, section 4', None),
 ('open', 'Blocked', 'draft-dflash on Qwythos 9B', '-', 'one draft exists (Qwen3.5-9B-DFlash, 914 MB, downloaded); the run that loads this third-party file was not permitted in the session', 'TRIALS.md, section 4', None),
 ('open', 'Blocked', 'ik_llama.cpp as an upper-bound probe', '-', 'cloned to ~/src/ik_llama.cpp; the build of external code was not permitted in the session', 'TRIALS.md, section 4', None),
 ('open', 'Blocked', 'EXPO / DDR5 speed, kernel and system tuning', '-', 'needs the BIOS, root and a reboot; affects CPU layers and memory kept in RAM only', 'TRIALS.md, section 4', None),
]
GROUPS = [('yes', 'Kept', 'In the code or in the gateway config today.'), ('same', 'Measured, default kept', 'No gain large enough to change a setting.'), ('no', 'Rejected', 'Slower, unstable, or it changes the output.'), ('open', 'Not measured yet', 'Blocked: each one needs a step by the owner of the machine.')]
ledger = ''
for kind, gname, gdesc in GROUPS:
    rows = [d for d in DEC if d[0] == kind]
    ledger += f'<tr class="lg"><td colspan="4"><b>{gname}</b> <span class="cnt">{len(rows)}</span> <span class="gd">{gdesc}</span></td></tr>'
    for k, pill, what, num, why, ev, sec in rows:
        link = f' <a href="#{sec}">charts</a>' if sec else ''
        ledger += f'<tr><td class="lp"><span class="pill {k}">{pill}</span></td><td class="lw">{what}</td><td class="ln {k}">{num}</td><td class="ly">{why}<span class="ev">{ev}{link}</span></td></tr>'
ledger = f'<table class="ledger"><thead><tr><th>Verdict</th><th>Decision</th><th>Deciding number</th><th>What it means</th></tr></thead><tbody>{ledger}</tbody></table>'
counts = [dict(k=g[1], v=sum(d[0] == g[0] for d in DEC), c=g[0]) for g in GROUPS]
now = [
 dict(k='R1-distill 7B', v=f'{m_new[0]:.1f}', was=f'{n7o:.1f}', d='-ngl 99 with -ctk q8_0 -ctv q4_0, was -ngl 24 with q8_0 / q8_0', x=f'{m_new[2]:.1f} t/s with 15K in context, was {c7("24", False, 15360)["v"]:.1f}'),
 dict(k='Qwythos 9B', v=f'{n9n:.1f}', was=f'{n9o:.1f}', d='-ngl 25, was 22, and ngram-simple', x=f'{e9[("ngram-simple", "edit")]["v"]:.0f} t/s on a code edit, was {e9[("none", "edit")]["v"]:.0f}'),
 dict(k='Qwen2.5 3B', v=f'{t3[(DEF, "free")]["v"]:.1f}', was=None, d='ngram-simple', x=f'{t3[(DEF, "edit")]["v"]:.0f} t/s on a code edit, was {t3[("none", "edit")]["v"]:.0f}'),
 dict(k='DeepSeek Coder 1.3B', v=f'{s1[("graphs on (default)", "complete")]["v"]:.0f}', was=None, d='ngram-simple', x=f'{s1[("ngram-simple", "complete")]["v"]:.0f} t/s when the completion repeats code, was {s1[("graphs on (default)", "complete")]["v"]:.0f}'),
]
# plain-language page top: headline figures and one line per experiment, grouped by outcome
TPS = 'tokens per second'
heads = [
 dict(k='Reading your question', v=tiles[0]['v'], u='faster', was=None, d='How fast the PC takes in what you typed. Faster on every model tested.'),
 dict(k='Writing the answer, main chat model', v=f'{m_new[0]:.1f}', u=TPS, was=f'was {n7o:.1f}', d='The 7B model. More of it now runs on the graphics card.'),
 dict(k='Writing the answer, largest model', v=f'{n9n:.1f}', u=TPS, was=f'was {n9o:.1f}', d='The 9B model. It does not fit on the card in full, so the gain is smaller.'),
 dict(k='Editing code you pasted in', v=f'{min(gall):.1f}x to {max(gall):.1f}x', u='faster', was=None, d='When the answer repeats text from the question. No cost on other questions.'),
]
ub1 = [cmp_avg(f)[('ub 1024', True, 0)]['v']/cmp_avg(f)[('ub 512 (default)', True, 0)]['v']-1 for f in ('ubatch-1p3b', 'ubatch-3b', 'ubatch-7b-q8')]
PLAIN = [
 ('yes', 'Kept', 'In use on this PC today.', [
  ('Stopped the card from using a feature it does not have.', f'The software took this card for a newer one and ran routines made for hardware it lacks (tensor cores). With the plain routines it reads questions {tiles[0]["v"]} faster. Writing speed is the same. Kept.'),
  ('A second pass on that fix, for long chats.', f'The first version wrote slower in one setup: a long chat with the chat memory stored at full precision ({kvv("fix1", "f16", 0, 8192, False):.1f} {TPS} against {co:.1f} before). A different routine brought it back to {cn:.1f}. That is the one cost left: about {(1-cn/co)*100:.0f}% slower in that one setup. Kept.'),
  ('Removed a temporary memory grab.', 'The first version of the fix stopped with "out of memory" in 3 of 6 runs. After this change it stopped in 0 of 9. No speed cost. Kept.'),
  ('An off switch for the fix.', 'One setting brings the old behaviour back, so before and after can be compared. It works. Kept.'),
  ('Skipped a slow shortcut on the 9B model.', 'Part of this model runs on the main processor, not on the graphics card. A combined step there was slower than doing the steps one by one. Without it, writing is 20 to 35% faster (15.6 to 20.6 tokens per second). Kept.'),
  ('Moved one of the two screens off the graphics card.', 'The desktop used about 1 GB of the card\'s 6 GB of memory. Now it uses about 660 to 760 megabytes, so the models have more room. Kept.'),
  ('More of the 7B model on the graphics card: 28 of its 29 layers, was 24.', f'A model is a stack of layers. A layer on the card runs faster than one on the main processor. Writing went from {n7o:.1f} to {n7n:.1f} {TPS} ({pct(n7n, n7o)}). Kept, then improved by the next item.'),
  ('All 29 layers of the 7B model on the card.', f'The PC keeps the chat so far in memory, in two halves. Storing one half more compactly frees enough room for the last layer. Writing went from {m_old[0]:.1f} to {m_new[0]:.1f} {TPS} ({pct(m_new[0], m_old[0])}), and {pct(m_new[2], m_old[2])} in a long chat. The quality score stayed inside its error margin. Kept, and live today.'),
  ('More of the 9B model on the graphics card: 25 layers, was 22.', f'Writing went from {n9o:.1f} to {n9n:.1f} {TPS} ({pct(n9n, n9o)}). Kept.'),
  ('Let the PC copy text it has already seen (9B, 3B and 1.3B models).', f'When an answer repeats parts of the question, as in a code edit, the PC guesses the repeated text and checks it in one go. Writing is {min(gall):.1f}x to {max(gall):.1f}x faster on code edits, and {s1[("ngram-simple", "complete")]["v"]/s1[("graphs on (default)", "complete")]["v"]:.1f}x on the small code model. No cost on ordinary questions. The answers were word for word the same in almost every test. Kept.'),
 ]),
 ('same', 'Tried, no real gain', 'Measured, then left as it was.', [
  ('A graphics-card feature that bundles work into batches (CUDA Graphs), on and off.', 'No difference larger than the normal wobble between two runs. Left on. The longest-chat part of this test could not run that day for lack of memory.'),
  ('An extra tuning switch for that feature.', 'In a benchmark the smallest model wrote 2 to 5% faster. The larger models gained nothing, and in real use the gain did not show at all. Left off.'),
  ('Different sizes and types of the copy shortcut.', 'Shorter matches were 9 to 25% faster on code, but 2 to 7% slower on ordinary questions, and they changed the answers. Two other types only helped on a straight copy. Standard setting kept.'),
  ('Reading the question in larger chunks.', f'Reading got {min(ub1)*100:.1f} to {max(ub1)*100:.1f}% faster for 70 to 900 megabytes more card memory. The largest chunk size did not fit next to the two bigger models. Standard size kept.'),
  ('Reading the question in smaller chunks.', 'Slower. Standard size kept.'),
  ('Four processor-side settings on the 9B model (pinned memory, busy waiting, higher priority, threads tied to cores).', 'All gave 20.5 to 21.3 tokens per second. Two runs with no change at all gave 21.3 and 19.1, so nothing stood out. Left as they were.'),
  ('Number of processor threads.', '6, one per physical core, stays. 4 and 8 were equal and 12 was slightly worse.'),
  ('A build option that forces one kind of math routine.', 'No change, within 1%. Left at the default.'),
  ('Mixes of precision for the two halves of the chat memory, as a speed setting.', 'All four mixes wrote at the same speed. The only gain is memory: about 130 megabytes on the 7B model in a long chat. That memory is what made room for the last layer in the "Kept" list.'),
  ('The same compact chat memory on the 9B model.', 'Quality was the same, but this model\'s chat memory is small, so it frees only about 40 megabytes. Not applied.'),
  ('One more layer of the 9B model on the card (26).', f'It works: {mix(M9, "ngl 26, q8/q8", False, 0):.1f} against {mix(M9, "ngl 25, q8/q8 (shipped)", False, 0):.1f} {TPS}. 25 was kept so the desktop has spare memory. A candidate for later.'),
  ('Letting the software choose how many layers go on the card.', f'With its standard safety margin it picks 25 of 29 layers and is slower ({fit[("fit, default margin 1024 MiB", "free")]["v"]:.1f} {TPS}). With a smaller margin it picks all 29 and matches the hand-set value ({fit[("fit, margin 512 MiB", "free")]["v"]:.1f} against {fit[("all 29 layers, fixed", "free")]["v"]:.1f}). Hand-set value kept. No new code needed.'),
  ('The copy shortcut on the 7B model, short answers.', 'This model thinks out loud before it answers, so there was nothing to copy. 51.5 against 51.4 tokens per second. No gain.'),
  ('Two ideas for reading compact chat memory faster.', 'A look at the code showed that both are already in place. Nothing to add.'),
  ('Checked an early finding that one compact memory format was 19% faster in long chats.', 'It did not repeat. The slow figure (28.2 tokens per second) was a single run. Later runs gave 32.5 to 33.6, level with the other format. The gap was measurement noise.'),
 ]),
 ('no', 'Tried, made things worse or did not work', 'Slower, unstable, or the answers changed. Not used.', [
  ('A different routine for compact chat memory, meant to speed up long chats.', 'Slower or equal in every case (for example 41.7 down to 34.0 tokens per second), and one long-chat case stopped running. Rejected.'),
  ('That routine together with the card fix.', 'Slower again, and it ran out of memory. Rejected.'),
  ('A small helper model that writes ahead while the large one checks.', f'On ordinary questions it was {-max(dfree)*100:.0f} to {-min(dfree)*100:.0f}% slower. At best 1.7x faster on code, where the copy shortcut gives about 3x. Both models share one card, so wrong guesses cost more than right ones save. Rejected. The 7B model refuses this helper, and the 9B model has no matching one.'),
  ('A copy shortcut that also remembers earlier requests.', 'Faster again (up to 5.9x), but the answers were not the same as without it. Rejected.'),
  ('Switching off a built-in feature that merges small steps.', f'Writing got {-max(fus)*100:.0f} to {-min(fus)*100:.0f}% slower. Reading was equal. Left on.'),
  ('The most compact format for the other half of the chat memory, on the 7B model.', f'The quality score went from {pk["K f16 / V f16"]:.2f} to {pk["K q4 / V q4"]:.0f}, where lower is better. The answers break. It was the same on the old build and on the main processor, so the format is the cause. Rejected.'),
  ('The copy shortcut on the 7B model, long answers.', f'{r7[("ngram-simple", "refactor")]["v"]/r7[("none", "refactor")]["v"]:.1f}x to {r7[("ngram-simple", "edit")]["v"]/r7[("none", "edit")]["v"]:.1f}x faster. But in one of two tests the model got stuck repeating its own reasoning and never gave the code. Not applied.'),
  ('All 29 layers of the 7B model with the original chat memory format.', '11 to 18% faster, but only 180 megabytes of card memory were left and long chats failed to start. Not applied in that form. Solved later with the compact memory in the "Kept" list.'),
  ('28 or 29 layers of the 9B model on the card.', 'Does not fit together with a long chat.'),
  ('Chat memory at full precision with a very long chat (16 thousand tokens).', 'Has never fit in the card\'s 6 GB, with any version.'),
  ('Keeping the chat memory in ordinary RAM instead of on the card.', 'It runs, but long chats are much slower: 13.9 against 32.6 tokens per second. Not used.'),
 ]),
 ('nb', 'Looked at, not built', 'Ruled out before any code was written.', [
  ('A routine that shares memory reads for compact chat memory.', 'It depended on the first item in the list above. That failed, so this was dropped.'),
  ('High precision for recent words, lower for older ones.', 'One half of the chat memory is already fine at low precision for all words. The other half breaks at low precision at any age. Nothing left to win.'),
  ('A different long-chat routine (tiled).', 'An earlier test had it slower than the routine in use: 38.6 against 42.2 tokens per second.'),
  ('Even more compact chat memory formats.', 'No such code exists in this software or in the related project that was checked.'),
  ('Other kinds of helper models for the 3B, 7B and 1.3B models.', 'Each kind needs a helper trained for the exact model. None is published for these three.'),
  ('Settings for "mixture of experts" models.', 'Does not apply. None of the models on this PC is of that kind.'),
  ('Spilling chat memory to RAM or disk in long chats.', 'Left out on purpose. Chat memory off the card is much slower (see above).'),
  ('A build option for much newer cards.', 'Does not apply to this card. Left off.'),
 ]),
 ('open', 'Could not test', 'Each one needs a step by the owner of the PC.', [
  ('A ready-made helper model for the 9B model.', 'One exists and is downloaded (914 MB). Running this outside file was not permitted in the session.'),
  ('A rival version of the same software, as a yardstick.', 'Downloaded. Building outside code was not permitted in the session.'),
  ('Faster RAM settings in the PC\'s setup menu (BIOS).', 'Needs the setup menu and a restart. It would only help the parts that run on the main processor.'),
  ('Operating system tuning.', 'A list of system settings. They need administrator rights or a restart, and some trade security for speed. None measured.'),
  ('Moving the second screen off the graphics card.', 'Not done. It would free more card memory.'),
  ('Running the card\'s memory faster than its rated speed.', 'Not tried. It risks silent errors, so it needs a quality check and not only a speed number.'),
  ('Two figures from the first review of the code (memory speed, and "15 to 20% headroom").', 'Both are calculations. No measurement supports or contradicts them.'),
  ('A review of how the chat memory and work space are set aside on the card.', 'Not done.'),
  ('Research ideas for dropping or shrinking old chat memory, and faster model loading.', 'Not tried. The first have no code here. The second changes start-up time only, not answer speed.'),
 ]),
]
plain = [dict(c=c, name=n, desc=d, items=[dict(w=w, r=r) for w, r in items]) for c, n, d, items in PLAIN]
try: LAB = json.load(open(rp('lab-state.json')))
except Exception: LAB = {}
import time
LW = {'same': ('same', 'No change'), 'faster': ('yes', 'Faster'), 'slower': ('no', 'Slower'), 'fail': ('no', 'Failed'), 'build failed': ('no', 'Build failed')}
lab = ''
for name, sv in LAB.items():
    if not isinstance(sv, dict) or 'steps' not in sv: continue
    k, word = LW.get(sv.get('state'), ('open', sv.get('state', 'not run').capitalize()))
    when = time.strftime('%d %b %H:%M', time.localtime(sv['at'])) if sv.get('at') else 'never'
    lab += f'<tr class="lg"><td colspan="4"><b>{name}</b> <span class="pill {k}">{word}</span> <span class="gd">last run: {when}</span></td></tr>'
    for step, (stt, text, _) in sv['steps'].items():
        k2, w2 = LW.get(stt, ('open', stt.capitalize()))
        lab += f'<tr><td class="lp"><span class="pill {k2}">{w2}</span></td><td class="lw">{step}</td><td class="ly" colspan="2">{text or "-"}</td></tr>'
lab = f'<table class="ledger"><thead><tr><th>Result</th><th>Step</th><th colspan="2">Candidate build against shipped build</th></tr></thead><tbody>{lab}</tbody></table>'
D.update(heads=heads, plain=plain, lab=lab, ledger=ledger, counts=counts, now=now, ngl=ngl7+ngl9, ngram=ngram, tune=tune, draft=draft, switches=switches)

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LOCAL, 'ledger.html')
open(out, 'w').write(open(os.path.join(HERE, 'ledger.template.html')).read().replace('__DATA__', json.dumps(D)))
print(f'{out}: {len(DEC)} decisions, ' + ', '.join(f'{c["v"]} {c["k"].lower()}' for c in counts))

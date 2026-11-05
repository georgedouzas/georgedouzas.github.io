"""Build the label index from rendered pages. Usage: harvest.py <rendered site dir> <site root>"""
import html,json,re,sys
from pathlib import Path
OUT=Path(sys.argv[1]);SITE=Path(sys.argv[2]);posts=json.load(open(Path(__file__).parent/'_tutorial_chapters.json'))
strip=lambda s:html.unescape(re.sub(r'<[^>]+>','',s)).strip()
THM=re.compile(r'id="((?:def|thm|lem|cor|exm|exr|prp|cnj|rem|alg)-[\w-]+)"[^>]*>(?:(?!id=").){0,600}?<span class="[^"]*theorem-title[^"]*"><strong>(.*?)</strong>',re.S)
FIG=re.compile(r'<div id="(fig-[\w-]+)"(?:(?!<div id="fig-).){0,4000}?<figcaption[^>]*>\s*(Figure(?:&nbsp;|\s)*[\d.]+)',re.S)
SEC=re.compile(r'<section id="(sec-[\w-]+)"[^>]*>\s*<h\d[^>]*>(.*?)</h\d>',re.S)
EQ=re.compile(r'<span id="(eq-[\w-]+)">.{0,20000}?\\tag\{([\d.]+)\}',re.S)
idx={};dup=[]
num=lambda s:re.match(r'[^\d]*\d+(?:\.\d+)*',s).group(0)
for post,files in posts.items():
    for f in files+['registry.qmd','glossary.qmd']:
        page=f.replace('.qmd','.html');p=OUT/'posts'/post/page
        if not p.exists(): continue
        h=p.read_text(encoding='utf-8');t=re.search(r'<title>(.*?)</title>',h,re.S);ch=strip(t.group(1)).split(' – ')[0] if t else ''
        h1=re.search(r'<h1[^>]*id="(sec-[\w-]+)"[^>]*>(.*?)</h1>|<section id="(sec-[\w-]+)"[^>]*class="level1[^"]*"[^>]*>\s*<h1[^>]*>(.*?)</h1>',h,re.S)
        def add(i,text,local=False):
            s=[{'t':'Str','text':w} if False else w for w in text.split()]
            if i in idx and idx[i]['page']!=page: dup.append(i)
            idx.setdefault(i,{'post':post,'page':page,'text':text})
        for i,tx in THM.findall(h): add(i,f'{num(strip(tx))} of {ch}')
        for i,tx in FIG.findall(h): add(i,f'{strip(tx)} of {ch}'.replace(chr(160),' '))
        for i,tx in EQ.findall(h): add(i,f'Equation {tx} of {ch}')
        for i,tx in SEC.findall(h): add(i,strip(tx))
        if h1:
            i=h1.group(1) or h1.group(3);add(i,strip(h1.group(2) or h1.group(4)))
print('labels',len(idx))
for post in posts:
    (SITE/'posts'/post/'cross-references.json').write_text(json.dumps(idx,ensure_ascii=False,indent=0),encoding='utf-8')
src=''.join((SITE/'posts'/p/f).read_text(encoding='utf-8') for p,fs in posts.items() for f in fs+['registry.qmd','glossary.qmd'])
P='sec|def|thm|lem|cor|exm|exr|prp|cnj|rem|alg|eq|fig|tbl'
defined=set(re.findall(r'\{#((?:'+P+r')-[\w-]+)',src))
used=set(re.findall(r'(?<![\w.])@((?:'+P+r')-[\w-]*\w)',src))
m=sorted(defined-set(idx));u=sorted(used-defined)
print('defined',len(defined),'used',len(used),'missing from index',len(m),m[:15]);print('used but undefined',len(u),u[:15]);print('defined on two pages',sorted(set(dup))[:15])
sys.exit(1 if m or u else 0)

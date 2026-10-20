"""Self-contained local user-audit UI; keeps drafts separate from exported audit."""
import base64
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import FEATURES,read
root=Path(__file__).resolve().parents[1]/'runs/feature_bank_01'
data=read(root/'draft_annotations.json')
for row in data['images']:
    row['pixels']='data:image/png;base64,'+base64.b64encode((root/row['image']).read_bytes()).decode()
template='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Feature annotation audit</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1080px;color:#18212c;background:#f6f7f9}button,select,input{font:inherit;padding:8px}button{cursor:pointer}header,main{background:white;padding:24px;border:1px solid #dce1e6;border-radius:10px;margin-bottom:16px}#features{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:flex;align-items:center;justify-content:space-between;gap:12px}small{display:block;color:#475569}img{display:block;max-width:100%;margin:24px 0;min-width:400px;image-rendering:auto}textarea{width:100%;height:70px;font:inherit}nav{display:flex;gap:10px;margin:18px 0;align-items:center}#progress{font-weight:600}#notice{color:#92400e}h1{margin-top:0}</style>
<header><h1>Review visible CAPTCHA features</h1><p>200 images · 12 features per image · no transcription needed.</p><p>These are assistant drafts, not audited ground truth. Check each label, keep <b>uncertain</b> when evidence is ambiguous, then mark that image reviewed. No classifier predictions are shown.</p><label>Your name <input id="reviewer" placeholder="Required for export"></label><p id="progress"></p><button id="export">Export completed audit</button><span id="notice"></span></header>
<main><nav><button id="prev">Previous</button><select id="jump"></select><button id="next">Next</button></nav><h2 id="title"></h2><img id="picture"><small>Enlarged for review; use the saved native image for fine-detail checks. No resizing enters the annotation data.</small><p id="draftnote"></p><div id="features"></div><p>Reviewer notes</p><textarea id="notes"></textarea><p><label><span>I reviewed every feature for this image</span><input type="checkbox" id="reviewed"></label></p></main>
<script>
const draft=__DATA__, definitions=__DEFS__, key='feature-bank-audit-'+draft.manifest_sha256;
let state={reviewer:'',images:draft.images.map(r=>({id:r.id,labels:{...r.labels},notes:'',reviewed:false}))},index=0;
try{const saved=JSON.parse(localStorage.getItem(key));if(saved && saved.images.length===200 && saved.images.every((r,i)=>r.id===draft.images[i].id))state=saved}catch(e){}
const $=id=>document.getElementById(id), save=()=>{try{localStorage.setItem(key,JSON.stringify(state))}catch(e){$('notice').textContent='Local saving unavailable; keep this page open.'}$('progress').textContent=state.images.filter(r=>r.reviewed).length+' / 200 reviewed'};
draft.images.forEach((r,i)=>{let o=new Option(r.id,i);$('jump').add(o)});
function show(){const d=draft.images[index],r=state.images[index];$('jump').value=index;$('title').textContent=d.id;$('picture').src=d.pixels;$('draftnote').textContent='Draft note: '+d.notes;$('notes').value=r.notes;$('reviewed').checked=r.reviewed;$('features').replaceChildren();
for(const [f,definition] of Object.entries(definitions)){let label=document.createElement('label'),span=document.createElement('span'),small=document.createElement('small'),select=document.createElement('select');span.textContent=f.replaceAll('_',' ');small.textContent=definition;span.append(small);for(const v of ['present','absent','uncertain'])select.add(new Option(v,v));select.value=r.labels[f];select.onchange=()=>{r.labels[f]=select.value;r.reviewed=false;$('reviewed').checked=false;save()};label.append(span,select);$('features').append(label)}save()}
$('reviewer').value=state.reviewer;$('reviewer').oninput=()=>{state.reviewer=$('reviewer').value;save()};$('notes').oninput=()=>{state.images[index].notes=$('notes').value;save()};$('reviewed').onchange=()=>{state.images[index].reviewed=$('reviewed').checked;save()};$('prev').onclick=()=>{index=Math.max(0,index-1);show()};$('next').onclick=()=>{index=Math.min(199,index+1);show()};$('jump').onchange=()=>{index=+$('jump').value;show()};
$('export').onclick=()=>{if(!state.reviewer.trim()||state.images.some(r=>!r.reviewed)){$('notice').textContent=' Enter your name and review all 200 images before export.';return}let result={version:'user-audit-v1',role:'user',reviewer:state.reviewer,manifest_sha256:draft.manifest_sha256,reviewed_at:new Date().toISOString(),images:state.images};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='feature_bank_user_audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};show();
</script></html>'''
html=template.replace('__DATA__',json.dumps(data).replace('</','<\\/')).replace('__DEFS__',json.dumps(FEATURES))
(root/'annotation_review.html').write_text(html,encoding='utf-8')
print(root/'annotation_review.html')

"""Audit recipe overlap and publish the completed DarkBay run."""
from pipeline import *
import html,collections,zipfile,io

def style(entry):
    text=copy.deepcopy(entry['recipe']['text_spec']);text.pop('text',None)
    return key(dict(size=entry['recipe']['size'],text=text,effects=entry['recipe']['effects_spec'],controls=entry['controls']))

def main():
    entries=U.read(ROOT/'entries.json');assert len(entries)==300
    signatures=[style(e) for e in entries];pairs={}
    for a,b in [('custom','dino'),('custom','mixed'),('dino','mixed')]:
        aa={e['input_index']:style(e) for e in entries if e['controller']==a};bb={e['input_index']:style(e) for e in entries if e['controller']==b}
        pairs[a+'_'+b]=sum(aa[i]==bb[i] for i in aa)
    diversity=dict(entries=300,unique_parameter_signatures=len(set(signatures)),identical_parameters_per_100_inputs=pairs,
        per_controller={a:len({style(e) for e in entries if e['controller']==a}) for a in ARMS},
        note='Signatures exclude sampled text and seeds. These are parameter differences, not a count of perceptually distinct families; inactive or overridden fields can still differ.')
    U.dump(ROOT/'diversity.json',diversity)
    summary=U.read(ROOT/'pilot_summary.json');summary.pop('unique_settings',None);summary['diversity']=diversity;U.dump(ROOT/'pilot_summary.json',summary)
    length=U.read(ROOT/'length_decision.json');raw=U.read(ROOT/'raw_hybrid_predictions.json');R.install()
    rawrecipes=[recipe(r['prediction'],1100003+i*7919,i) for i,r in enumerate(raw)];U.dump(ROOT/'raw_hybrid_recipes.json',rawrecipes)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    for page in range(10):
        sheet=Image.new('RGB',(1500,1080),'#f4f6f8');d=ImageDraw.Draw(sheet)
        for j,s in enumerate(['Original pixels','Raw hybrid','Custom','DINO','Mixed']):d.text((j*300+8,12),s,fill='black',font=font)
        for ii in range(10):
            i=page*10+ii;r=raw[i];rr=R.safe_render((rawrecipes[i],R.expand({})))
            tiles=[Image.open(r['source']['source']).convert('RGB'),Image.fromarray(rr.get('image',np.full((64,256,3),255,np.uint8)))]
            tiles.extend(Image.open(ROOT/a/f'{i:04d}.png').convert('RGB') for a in ARMS)
            for j,t in enumerate(tiles):
                tile=ImageOps.contain(t,(286,82),Image.Resampling.NEAREST);sheet.paste(tile,(j*300+(300-tile.width)//2,60+ii*100))
        sheet.save(ROOT/f'comparison_{page+1:02d}.png')
    corpus=U.read(ROOT/'corpus_audit.json') if (ROOT/'corpus_audit.json').exists() else None
    sourcefiles=[ROOT/'pipeline.py',ROOT/'corpus.py',ROOT/'report.py',WORK/'controller_mvp_v10_mix/mix10.py',WORK/'controller_mvp_v7_dino/render7.py',WORK/'controller_mvp_v7_dino/perception7.py']
    U.dump(ROOT/'code_manifest.json',{str(p):U.sha(p) for p in sourcefiles})
    from setup7 import WEIGHTS_PATH
    checkpoints={'hybrid':U.HYBRID,'custom_encoder':U.ENCODER,'text_mask':WORK/'controller_mvp_v3/mask_training/best.pt','dino':WEIGHTS_PATH}
    U.dump(ROOT/'model_manifest.json',{k:dict(path=str(p),sha256=U.sha(p)) for k,p in checkpoints.items()})
    if corpus:
        shards=U.read(ROOT/'corpus_summary.json')
        U.dump(ROOT/'dataset_manifest.json',dict(format='One ZIP per controller recipe; PNG images + labels.jsonl + rejections.jsonl',
            count=corpus['images'],entries_sha256=U.sha(ROOT/'entries.json'),length_decision=length,controllers=corpus['controllers'],
            shards=[dict(path='corpus/'+s['entry_id']+'.zip',**s) for s in sorted(shards,key=lambda s:s['entry_id'])]))
        U.dump(ROOT/'status.json',dict(stage='complete',inputs=100,controller_entries=300,images=corpus['images'],audit='corpus_audit.json'))
    content='<!doctype html><meta charset="utf-8"><title>CAPRI DarkBay</title><style>body{font:17px system-ui;max-width:1500px;margin:30px auto;padding:20px;background:#f4f6f8;color:#182033}img{width:100%}pre{white-space:pre-wrap;background:white;padding:16px}a{color:#145abd}</style>'
    content+='<h1>CAPRI — DarkBay</h1><p>100 target images → 100 individually refined hybrid recipes → 3 controllers → 300 recipe entries → balanced synthetic corpus.</p>'
    content+=f'<p>Default length <b>{length["default_length"]}</b>: {length["votes"]}/100 raw hybrid votes, clearing the 60% rule. No target strings, human length labels or VLMs. Agreement is not calibrated accuracy.</p>'
    content+='<p>Frozen custom and DINO comparators guide renderer search; the mixed controller uses their normalized 50/50 score. Each score combines the corresponding target image with the full 100-image distribution. The generated strings differ from the targets.</p>'
    content+='<h2>Audit</h2><pre>'+html.escape(json.dumps(corpus or {'status':'pilot only; corpus not yet audited'},indent=2))+'</pre>'
    content+='<h2>Diversity</h2><pre>'+html.escape(json.dumps(diversity,indent=2))+'</pre>'
    content+='<p>All three arms retain equal representation, including duplicate choices. Some thick, faint and outlined lettering remains. Renderer validity does not establish visual fidelity, human readability, or downstream solver benefit. Target images used for control are not an unseen evaluation set.</p>'
    if corpus:
        content+=f'<p><b>Strict readability pass: {corpus["strict"]:,}/{corpus["images"]:,} ({100*corpus["strict"]/corpus["images"]:.1f}%).</b> Export uses the broader completeness/usability guard; it does not require this stricter flag. The strict check requires every character to meet contrast ≥70, retained visibility ≥80%, height ≥12 px, clipping ≤5%, and stroke width ≥1.2 px. Each sample keeps its strict flag so it can be filtered explicitly. Filtering changes the recipe mixture.</p>'
    if (ROOT/'pilot_readability.json').exists():content+='<pre>'+html.escape(json.dumps(U.read(ROOT/'pilot_readability.json'),indent=2))+'</pre>'
    content+='<p>Corpus: <a href="corpus/">300 ZIP shards</a>, one per recipe, containing PNGs, labels.jsonl and rejection logs. <a href="entries.json">Recipe provenance</a> · <a href="allocation.json">Allocation</a> · <a href="raw_hybrid_predictions.json">Raw model outputs</a> · <a href="refined_predictions.json">Cleanup log</a></p>'
    content+='<h2>All 100 inputs, original order</h2><p>Each row compares the original, raw hybrid render, custom controller, DINO controller, and mixed controller. No visual selection.</p>'
    for page in range(10):content+=f'<img loading="lazy" src="comparison_{page+1:02d}.png" alt="Inputs {page*10+1}–{page*10+10}">'
    (ROOT/'report.html').write_text(content,encoding='utf8');print(json.dumps(diversity),flush=True)

if __name__=='__main__':main()

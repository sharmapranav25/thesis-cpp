"""Keep failed examples visible and recover already-selected partial-run outputs."""
import hashlib
import html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from . import runtime as U
from .smaller_hybrid_preview import OUT,FAMILIES,ARMS
from .decoder.common import now


def main():
    state=U.read(OUT/'status.json');assert state['state'] in ['complete','completed_with_errors']
    dest=OUT/'review'
    if dest.exists():raise FileExistsError('Preserving review export')
    dest.mkdir();results=U.read(OUT/'results.json');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    overall=Image.new('RGB',(1200,5*240+65),'#f4f6f8');ov=ImageDraw.Draw(overall)
    for j,title in enumerate(['Real input','Custom','DINO','Mixed']):ov.text((10+300*j,12),title,fill='#182033',font=font)
    audits={};total=0
    for family_index,family in enumerate(FAMILIES):
        root=OUT/family;refs=U.read(root/'inputs.json');entries=U.read(root/'entries.json')
        output_root=root
        partial='error' in results[family]
        if partial:
            from . import pipeline as P
            from .artifacts import save_rows
            output_root=dest/family;output_root.mkdir();P.R.install()
            # Replay only previously selected entries, with their saved pilot
            # attempt seed. Never run a new search or invent the failed entry.
            for arm in ARMS:
                rows=[]
                chosen=[e for e in entries if e['controller']==arm]
                assert [e['input_index'] for e in chosen]==list(range(len(chosen)))
                for entry in chosen:
                    index=entry['input_index'];seed=1400003+index*1009+(entry['pilot_attempts']-1)*10000019
                    row=P.R.safe_render((P.draw_recipe(entry['recipe'],seed),entry['controls']))
                    assert P.usable(row,entry['controls']['length'])
                    # Recheck the three original fresh-seed validity tests.
                    for j in range(3):
                        test=P.R.safe_render((P.draw_recipe(entry['recipe'],1300003+index*1009+j*104729),entry['controls']))
                        assert P.usable(test,entry['controls']['length'])
                    rows.append(row)
                save_rows(output_root,arm,rows)
        labels={a:U.read(output_root/a/'labels.json') for a in ARMS}
        sheet=Image.new('RGB',(1200,735),'#f4f6f8');d=ImageDraw.Draw(sheet)
        for j,title in enumerate(['Target original','Custom','DINO','Mixed']):d.text((10+300*j,12),title,fill='#182033',font=font)
        hashes=[];target_hashes={r['pixel_sha256'] for r in refs}
        for i in range(6):
            images=[Image.open(refs[i]['source']).convert('RGB')]
            for arm in ARMS:
                if i<len(labels[arm]):
                    row=labels[arm][i];im=Image.open(output_root/row['image']).convert('RGB')
                    digest=hashlib.sha256(np.asarray(im).tobytes()).hexdigest();assert digest not in target_hashes;hashes.append(digest)
                    assert row['complete'] and row['usable']
                else:
                    im=Image.new('RGB',(286,76),'#ffe7e7');q=ImageDraw.Draw(im)
                    q.text((8,8),'No usable controller recipe',fill='#812b2b',font=font)
                    q.text((8,38),'Failure preserved',fill='#812b2b',font=font)
                images.append(im)
            for j,im in enumerate(images):
                tile=ImageOps.contain(im,(286,84),Image.Resampling.NEAREST)
                sheet.paste(tile,(300*j+(300-tile.width)//2,55+i*108))
                if i<2:
                    y=60+family_index*240+i*110
                    if j==0:ov.text((10,y),family+' — input '+str(i+1),fill='#182033',font=font)
                    small=ImageOps.contain(im,(286,76),Image.Resampling.NEAREST)
                    overall.paste(small,(300*j+(300-small.width)//2,y+27))
        if partial:d.text((10,705),'Input6 failed the existing candidate/recovery checks. Inputs1–5 replay the saved controller choices.',fill='#812b2b',font=font)
        sheet.save(dest/(family+'.png'));total+=len(hashes)
        audits[family]=dict(examples=len(hashes),failed_input_indices=list(range(len(labels['custom']),6)),
            strict_per_arm={a:sum(bool(r['strict']) for r in labels[a]) for a in ARMS},
            recovered_from_saved_entries=partial,entry_sha256=U.sha(root/'entries.json'),output_pixel_sha256=hashes)
    overall.save(dest/'overview.png')
    audit=dict(completed_at=now(),examples=total,expected_examples=90,families=audits,
        code_sha256=U.sha(__file__),training_started=False,new_controller_search=False,
        limitation='Synthetic images from fitted references; no target fidelity/reader accuracy claim. Failed example retained explicitly.')
    U.dump(dest/'audit.json',audit)
    page='<html><meta charset="utf-8"><title>Smaller hybrid controller results</title><style>body{font:18px system-ui;max-width:1200px;margin:24px auto}img{width:100%}</style><h1>Smaller hybrid through custom, DINO and mixed controllers</h1><p>First6 inputs per scheme, each fitted against the full100-image reference distribution. Fresh synthetic strings; no target answers, VLMs, model training or corpus generation. All three controller outputs retained. DarkBay reuses its exact-input/checkpoint-verified historical run.</p><p>87 images produced. Simple CAPTCHA input6 failed all existing candidate/recovery checks; its failed row is shown rather than replaced. Saved choices for its first5 inputs were deterministically replayed and revalidated on the original3 fresh seeds.</p><p>City and DarkBay use the model majority length. CryptBB, Simple CAPTCHA and Outlined lack60/100 agreement and use the documented equal-observed-length preview fallback. The controllers hold that assigned length fixed; these previews do not establish improved count accuracy.</p>'
    for name in FAMILIES:page+='<h2>'+html.escape(name)+'</h2><img src="'+name+'.png">'
    (dest/'report.html').write_text(page+'</html>',encoding='utf8')
    print({k:{a:b for a,b in v.items() if a not in ['output_pixel_sha256']} for k,v in audits.items()},flush=True)

from setup7 import ROOT,U
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np
import html

def main():
    summary=U.read(ROOT/'summary.json');gallery=ROOT/'gallery';gallery.mkdir(exist_ok=True)
    review=U.read(ROOT/'review.json') if (ROOT/'review.json').exists() else {'conclusion':'Visual review pending.'}
    cross=U.read(ROOT/'crosscheck.json')['scores'] if (ROOT/'crosscheck.json').exists() else {}
    font='C:/Windows/Fonts/arial.ttf';title=ImageFont.truetype(font,27);header=ImageFont.truetype(font,20);small=ImageFont.truetype(font,15)
    sections=[];table=[]
    for family,result in summary.items():
        dest=ROOT/family;refs=U.read(dest/'target_splits.json')['test'];labels={a:U.read(dest/a/'labels.json') for a in ['hybrid','dino_v1','dino_v2']}
        masks=np.load(dest/'target_regions.npz')
        for mode in ['comparison','regions']:
            sheet=Image.new('RGB',(900,704),'#f4f6f8');draw=ImageDraw.Draw(sheet)
            draw.text((12,10),family+' | '+('DINO controller update' if mode=='comparison' else 'inferred regions'),font=title,fill='#172033')
            names=['Original target','DINO v1','DINO v2'] if mode=='comparison' else ['Original target','DINO attention','Pixel + DINO region']
            for j,s in enumerate(names):draw.text((12+j*300,50),s,font=header,fill='#172033')
            for i in range(6):
                original=Image.open(refs[i]['source']).convert('RGB')
                if mode=='comparison':images=[original,*[Image.open(dest/labels[a][i]['image']) for a in ['dino_v1','dino_v2']]]
                else:
                    pixels=np.asarray(original.resize((256,64),Image.Resampling.BILINEAR),float);images=[original]
                    for k in ['attention','region']:
                        alpha=.65*masks[k][i][:,:,None]
                        images.append(Image.fromarray((pixels*(1-alpha)+np.array([255,65,20])*alpha).clip(0,255).astype(np.uint8)))
                for j,im in enumerate(images):
                    tile=ImageOps.contain(im.convert('RGB'),(286,80),Image.Resampling.NEAREST)
                    sheet.paste(tile,(300*j+(300-tile.width)//2,82+98*i+(80-tile.height)//2))
            caption='First 6 samples. Synthetic strings differ; samples were not picked for appearance.' if mode=='comparison' else 'Inferred from target pixels; these are uncertain estimates, not known text masks.'
            draw.text((12,678),caption,font=small,fill='#465364');sheet.save(gallery/f'{family}_{mode}.png')
        selected=U.read(dest/'selected.json');p=selected['controls']
        rows=[]
        for a,s in result['scores'].items():
            old=f'{cross[family][a]["appearance"]:.3f}' if family in cross else 'pending'
            rows.append(f'<tr><td>{a}</td><td>{s["objective"]:.3f}</td><td>{s["appearance"]:.3f}</td><td>{s["dino_v1_metric"]["appearance"]:.3f}</td><td>{old}</td><td>{s["usable"]:.1%}</td><td>{s["region_mask_iou"]:.3f}</td></tr>')
        notes=html.escape(review.get('families',{}).get(family,'Review pending.'))
        control={k:p[k] for k in ['charset','layout','length','font','size_absolute','spacing_absolute','slant_absolute','size_jitter_absolute','glyph_outline','frame_width']}
        sections.append(f'<section><h2>{family}</h2><p>{notes}</p><p>Selected bank: {result["selected_bank"]}; validation gate passed: {result["search_pass"]}. {result["archive_count"]} archived configurations.</p><table><tr><th>Controller</th><th>New objective ↓</th><th>New appearance ↓</th><th>Old DINO appearance ↓</th><th>Legacy comparator ↓</th><th>Usable</th><th>Synthetic region IoU</th></tr>{"".join(rows)}</table><img src="gallery/{family}_comparison.png"><details><summary>Inferred target regions</summary><img src="gallery/{family}_regions.png"></details><details><summary>Selected text controls</summary><pre>{html.escape(str(control))}</pre></details></section>')
        before=result['scores']['dino_v1'];after=result['scores']['dino_v2'];table.append(f'<tr><td>{family}</td><td>{before["objective"]:.3f}</td><td>{after["objective"]:.3f}</td><td>{after["usable"]:.1%}</td><td>{result["selected_bank"]}</td></tr>')
    checks=U.read(ROOT/'checks.json')
    doc='''<!doctype html><meta charset="utf-8"><title>DINO controller v2</title><style>body{font:17px system-ui;max-width:1120px;margin:30px auto;padding:0 16px;background:#f4f6f8;color:#172033}section{padding:20px;background:white;margin:22px 0}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.5}pre{white-space:pre-wrap}summary{padding:12px}</style><h1>DINO controller v2 — MVP7</h1>'''
    doc+=f'<p><strong>{html.escape(review["conclusion"])}</strong></p>'
    doc+='''<p>Same frozen DINO ViT-S/8; no VLM or OCR. Target inputs are RGB pixels only. Added multiscale colour-contrast regions, glyph/stroke/layout/background descriptors and a contrast-collapse penalty. Absolute controls now expose charset, layout, size, width, spacing, weight, bold, slant, size variation and curve. Generic glyph outlines and rotated square frames extend the renderer.</p><p>Search accumulates coordinate changes and tries a hybrid bank plus a fresh random bank. Eight fit renders per candidate. Six shortlisted candidates per bank plus the previous DINO profile face two validation batches: 32 target images and 24 fresh synthetic examples each. Rank mean objective plus 0.25 standard deviation, with renderer validity gates.</p><p>Freeze controls before diagnostics: 48 fresh synthetic samples per arm, and previously exposed target diagnostic pixels. Larger budget, new controls and new score change together; this is not an equal-budget backbone ablation. The region estimator remains heuristic. Renderer usability is not human legibility. No solver was trained. The legacy cross-check shares handcrafted descriptor code with this update, so it is not fully independent.</p>'''
    doc+=f'<p>Generic synthetic region probe: attention IoU {checks["raw_attention_iou"]:.3f} → pixel-region IoU {checks["new_region_iou"]:.3f} over 24 samples. This does not measure target segmentation accuracy. Disabled new controls reproduce old-renderer pixels exactly in the regression probe. Blank/blur degradation checks pass. Frames are included in synthetic glyph masks.</p>'
    doc+='<table><tr><th>Family</th><th>v1, new score ↓</th><th>v2, new score ↓</th><th>v2 usable</th><th>Selected start</th></tr>'+''.join(table)+'</table>'+''.join(sections)
    doc+='<p><a href="protocol.json">Protocol</a> · <a href="checks.json">Checks</a> · <a href="audit.json">Audit</a> · <a href="summary.json">Measurements</a></p>'
    (ROOT/'report.html').write_text(doc,encoding='utf-8')

if __name__=='__main__':main()

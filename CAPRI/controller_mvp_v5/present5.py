"""Unselected samples and true DINO attention maps, suitable for phone viewing."""
from setup5 import ROOT,U,FAMILIES
import argparse,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=FAMILIES);args=ap.parse_args()
    gallery=ROOT/'gallery';gallery.mkdir(exist_ok=True)
    font='C:/Windows/Fonts/arial.ttf';title=ImageFont.truetype(font,27);header=ImageFont.truetype(font,20);small=ImageFont.truetype(font,15)
    sections=[]
    audit=U.read(ROOT/'audit.json') if (ROOT/'audit.json').exists() else None
    for family in ([args.family] if args.family else FAMILIES):
        dest=ROOT/family;result=U.read(dest/'results.json');refs=U.read(dest/'target_splits.json')['test'];rows={a:U.read(dest/a/'labels.json') for a in ['hybrid','controller1','dino','cls_only']}
        attention=np.load(dest/'target_attention.npz')['focus'];n=min(8,len(refs),len(rows['dino']))
        for mode in ['comparison','attention','controllers']:
            names={'comparison':['Original target','Hybrid start','Controller 2: DINO'], 'attention':['Original target','DINO focus','Complementary region'], 'controllers':['Original target','Controller 1','Controller 2: DINO']}[mode]
            sheet=Image.new('RGB',(900,112+n*98),'#f4f6f8');draw=ImageDraw.Draw(sheet);draw.text((12,10),family+' — '+('attention' if mode=='attention' else 'generated examples'),font=title,fill='#172033')
            for j,name in enumerate(names):draw.text((300*j+12,49),name,font=header,fill='#172033')
            for i in range(n):
                original=Image.open(refs[i]['source']).convert('RGB')
                if mode=='attention':
                    a=np.asarray(original.resize((256,64),Image.Resampling.BILINEAR),dtype=float);m=attention[i]
                    red=np.array([255,65,20]);blue=np.array([20,100,255])
                    focus=Image.fromarray((a*(1-.65*m[:,:,None])+red*.65*m[:,:,None]).clip(0,255).astype(np.uint8))
                    context=Image.fromarray((a*(1-.65*(1-m[:,:,None]))+blue*.65*(1-m[:,:,None])).clip(0,255).astype(np.uint8));images=[original,focus,context]
                else:
                    other='hybrid' if mode=='comparison' else 'controller1';images=[original,Image.open(dest/rows[other][i]['image']),Image.open(dest/rows['dino'][i]['image'])]
                for j,im in enumerate(images):
                    tile=ImageOps.contain(im,(286,80),Image.Resampling.NEAREST);sheet.paste(tile,(300*j+(300-tile.width)//2,82+98*i+(80-tile.height)//2))
            text='Attention is saliency; noise and background can receive attention.' if mode=='attention' else 'First 8 samples. Generated strings are synthetic and may differ in length from targets.'
            draw.text((12,90+n*98),text,font=small,fill='#465364');sheet.save(gallery/(family+'_'+mode+'.png'))
        records=[]
        for arm,score in result['scores'].items():
            old=f'{audit["legacy_crosscheck"][family][arm]["appearance"]:.3f}' if audit else 'pending'
            records.append(f'<tr><td>{arm}</td><td>{score["appearance"]:.3f}</td><td>{old}</td><td>{score["complete"]:.1%}</td><td>{score["usable"]:.1%}</td><td>{score["attention_mask_iou"]:.3f}</td></tr>')
        review=U.read(dest/'visual_review.json') if (dest/'visual_review.json').exists() else {'text':'Visual review pending; model score alone is not a quality pass.'}
        sections.append(f'<section><h2>{family}</h2><p>{html.escape(review["text"])}</p><p>{result["retained"]} retained profiles; fonts: {html.escape(", ".join(result["font_choices"]))}.</p><table><tr><th>Controller</th><th>DINO distance ↓</th><th>Legacy distance ↓</th><th>Complete</th><th>Usable</th><th>Attention / synthetic text IoU</th></tr>{"".join(records)}</table><img src="gallery/{family}_comparison.png"><details><summary>Controller 1 comparison</summary><img src="gallery/{family}_controllers.png"></details><details><summary>What DINO attends to</summary><img src="gallery/{family}_attention.png"></details><p><a href="{family}/selected.json">Selected controls</a> · <a href="{family}/archive.json">Profile collection</a> · <a href="{family}/results.json">Measurements</a></p></section>')
    if args.family:return
    text='''<!doctype html><meta charset="utf-8"><title>Controller 2 — DINO attention</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:0 16px;background:#f4f6f8;color:#172033}section{background:white;padding:24px;margin:22px 0;border-radius:12px}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}p{line-height:1.5}summary{padding:12px;cursor:pointer}</style><h1>Controller 2: DINO attention</h1><p>Same goal as Controller 1: improve text and background resemblance from unlabeled target pixels, starting with the frozen hybrid. Different underlying comparison model: official frozen DINO ViT-S/8. No VLMs, OCR, target strings, boxes, masks or texture labels enter this controller.</p><p>DINO supplies six attention heads and patch embeddings. The search compares global appearance, attended and complementary regions, layout and color distributions. Attention is saliency, not calibrated text localization. All font and effect choices come from the existing generic renderer.</p><p>Selection uses fit/selection pixels only. The displayed test pixels were exposed in earlier MVPs: these are diagnostic comparisons, not a fresh generalization claim. DINO's own lower score does not establish recognition improvement. The CLS-only selector shares the DINO-guided proposal pool. Controller 1 used a different earlier search budget.</p><p>No new reader was trained in this iteration. Synthetic completeness/usability and attention overlap are diagnostics, not target solving accuracy. Original and generated strings differ.</p>'''+''.join(sections)+'''<p><a href="protocol.json">Protocol</a> · <a href="checks.json">Attention implementation checks</a> · <a href="audit.json">Final audit</a> · <a href="https://github.com/facebookresearch/dino">Official DINO source</a></p>'''
    conclusion='<section><strong>Built: 79 retained profiles across five families. Visual outcome: partial for City, CryptBB and DarkBay; failed for Outlined and SimpleCaptcha. No family-match pass.</strong><p>DINO distance improves22–39% over the hybrid starts on reused diagnostic pixels, but this does not establish better typography or recognition. Synthetic attention/text IoU is only0.10–0.24. The separate legacy comparator favors Controller1 on every family and flags DINO regressions versus hybrid for DarkBay and Outlined. Attention selects different controls from CLS-only on four families, yet does not consistently improve either independent measurements or visual quality. No new solver was trained.</p><p>The79 archives have79 distinct render signatures on three fixed recipes each. This finite probe is not proof of global diversity. Both controllers remain available independently.</p></section>'
    text=text.replace('<h1>Controller 2: DINO attention</h1>','<h1>Controller 2: DINO attention</h1>'+conclusion)
    (ROOT/'report.html').write_text(text,encoding='utf-8')

if __name__=='__main__':main()

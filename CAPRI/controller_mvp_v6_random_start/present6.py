"""Display prespecified samples and all random seeds; never select a winning seed."""
from ablate6 import ROOT,U,FAMILIES,SEEDS
from PIL import Image,ImageDraw,ImageFont,ImageOps
import html,json

def main():
    summary=U.read(ROOT/'summary.json');gallery=ROOT/'gallery';gallery.mkdir(exist_ok=True)
    cross=U.read(ROOT/'crosscheck.json')['scores'] if (ROOT/'crosscheck.json').exists() else None
    review=U.read(ROOT/'review.json') if (ROOT/'review.json').exists() else {'conclusion':'Visual review pending. Scores alone do not establish better typography or recognition.'}
    font='C:/Windows/Fonts/arial.ttf'
    title=ImageFont.truetype(font,28);header=ImageFont.truetype(font,19);small=ImageFont.truetype(font,15)
    sections=[];table=[]
    for family,result in summary.items():
        dest=ROOT/family;refs=U.read(dest/'target_splits.json')['test'];runs=result['runs']
        for mode in ['final','random_progress','all_seeds']:
            if mode=='final':
                names=['Original target','Hybrid -> controller','Random -> controller']
                specs=[None,('hybrid','final'),(f'random_{SEEDS[0]}','final')]
            elif mode=='random_progress':
                names=['Original target','Random starting point','Random -> controller']
                specs=[None,(f'random_{SEEDS[0]}','start'),(f'random_{SEEDS[0]}','final')]
            else:
                names=['Random seed '+str(s) for s in SEEDS]
                specs=[(f'random_{s}','final') for s in SEEDS]
            labels=[U.read(dest/a/b/'labels.json') if spec else None for spec in specs for a,b in ([spec] if spec else [(None,None)])]
            n=6;sheet=Image.new('RGB',(900,116+98*n),'#f4f6f8');draw=ImageDraw.Draw(sheet)
            draw.text((12,10),family+' | '+('all random runs' if mode=='all_seeds' else 'initialization experiment'),font=title,fill='#172033')
            for j,name in enumerate(names):draw.text((j*300+10,50),name,font=header,fill='#172033')
            for i in range(n):
                for j,spec in enumerate(specs):
                    path=refs[i]['source'] if spec is None else dest/spec[0]/labels[j][i]['image']
                    im=Image.open(path).convert('RGB');tile=ImageOps.contain(im,(286,80),Image.Resampling.NEAREST)
                    sheet.paste(tile,(300*j+(300-tile.width)//2,82+98*i+(80-tile.height)//2))
            caption='First 6 samples from every predeclared random seed. Synthetic strings differ.' if mode=='all_seeds' else 'First 6 samples; random seed 60101 fixed before testing. Synthetic strings differ.'
            draw.text((12,90+98*n),caption,font=small,fill='#465364')
            sheet.save(gallery/f'{family}_{mode}.png')
        rows=[]
        for arm,r in runs.items():
            old=f'{cross[family][arm]["appearance"]:.3f}' if cross else 'pending'
            score=r['final'];rows.append(f'<tr><td>{arm}</td><td>{r["start"]["objective"]:.3f}</td><td>{score["objective"]:.3f}</td><td>{score["appearance"]:.3f}</td><td>{old}</td><td>{score["complete"]:.0%}</td><td>{score["usable"]:.0%}</td><td>{r["search_pass"]}</td></tr>')
        rv=[runs[f'random_{s}']['final']['objective'] for s in SEEDS]
        table.append(f'<tr><td>{family}</td><td>{runs["hybrid"]["final"]["objective"]:.3f}</td><td>{sum(rv)/3:.3f} ({min(rv):.3f}–{max(rv):.3f})</td><td>{result["random_wins"]}/3</td></tr>')
        note=html.escape(review.get('families',{}).get(family,'Visual review pending.'))
        sections.append(f'<section><h2>{family}</h2><p>{note}</p><table><tr><th>Start</th><th>Before ↓</th><th>After ↓</th><th>Appearance ↓</th><th>Legacy comparator ↓</th><th>Complete</th><th>Usable</th><th>Selection gate passed</th></tr>{"".join(rows)}</table><img src="gallery/{family}_final.png"><details><summary>Random before/after</summary><img src="gallery/{family}_random_progress.png"></details><details><summary>All three random seeds</summary><img src="gallery/{family}_all_seeds.png"></details></section>')
    doc='''<!doctype html><meta charset="utf-8"><title>Hybrid versus random initialization</title><style>body{font:17px system-ui;max-width:1050px;margin:30px auto;padding:0 16px;background:#f4f6f8;color:#172033}section{padding:20px;background:white;margin:20px 0}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}p{line-height:1.5}summary{padding:12px}</style><h1>Does the hybrid starting point help?</h1>'''
    doc+=f'<p>{html.escape(review["conclusion"])}</p>'
    if review.get('selection_warning'):doc+=f'<p><strong>Choosing a start:</strong> {html.escape(review["selection_warning"])}</p>'
    doc+='''<p>Same frozen DINO attention controller, renderer and target pixels. Per run: 192 fit candidate slots × 4 renderings, 16 selection slots × 8 renderings, then 24 diagnostic samples. Three generic random banks per family versus one fixed hybrid bank. Same renderer seeds, font proposal subset and search RNG. Score calibration uses a separate random bank shared by all arms.</p><p>Random banks contain no hybrid predictions; only canvas dimensions come from target fit pixels. Generic text specs draw independent fonts, character groups, geometry and effects. This tests replacing the entire starting recipe bank. It also exposes fixed inherited settings: the current controller cannot directly change charset, layout, slant or size jitter. It is not a pure initialization comparison in a fully controllable parameter space.</p><p>Lower score means closer under this DINO comparator, with penalties for incomplete/unusable renders. Previously exposed diagnostic targets; no target answers, OCR or VLM. No reader trained. Three seeds do not establish statistical certainty or an “always” claim. Blank render failures, if any, count against validity and stay visible.</p><table><tr><th>Family</th><th>Hybrid final ↓</th><th>Random final mean (range) ↓</th><th>Random wins</th></tr>'''+''.join(table)+'</table>'+''.join(sections)
    doc+='<p>Legacy comparator: evaluated only after controls freeze; hybrid final is normalized to 1. This second metric is also learned, not ground truth.</p><p><a href="protocol.json">Protocol</a> · <a href="summary.json">All measurements</a> · <a href="audit.json">Audit</a> · <a href="crosscheck.json">Second comparator</a></p>'
    (ROOT/'report.html').write_text(doc,encoding='utf-8')

if __name__=='__main__':main()

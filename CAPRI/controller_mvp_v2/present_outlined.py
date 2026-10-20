"""Show the unchanged outlined-text experiment, including its original hybrid start."""
from bootstrap import ROOT,V1,U
from concurrent.futures import ProcessPoolExecutor
import html
from PIL import Image,ImageDraw,ImageOps
import controls as C
from run import save_rows

def main():
    out=ROOT/'Outlined_run_01';r=U.read(out/'results.json');split=U.read(out/'target_splits.json')
    recipes=U.read(out/'test_recipes.json')
    with ProcessPoolExecutor(max_workers=8,initializer=C.install) as pool:
        rows=list(pool.map(C.render,[(recipe,C.expand(C.R.NEUTRAL)) for recipe in recipes]))
    save_rows(out,'hybrid_start',rows)
    groups={k:U.read(out/k/'labels.json') for k in ['hybrid_start','mvp1','expanded']}
    sheet=Image.new('RGB',(1200,850),'#f4f5f7');d=ImageDraw.Draw(sheet)
    for j,title in enumerate(['Target','Hybrid start','MVP 1','MVP 2']):d.text((j*300+8,14),title,fill='black')
    for i in range(8):
        paths=[split['test'][i]['source'],*[out/groups[k][i]['image'] for k in groups]]
        for j,path in enumerate(paths):
            im=ImageOps.contain(Image.open(path).convert('RGB'),(288,88),Image.Resampling.NEAREST)
            sheet.paste(im,(j*300+(300-im.width)//2,45+i*100+(88-im.height)//2))
    sheet.save(out/'outlined_comparison.png')
    baseline=r['test']['mvp1']['new'];scores=r['test']['expanded']['new']
    reductions={k:100*(1-scores[k]/max(1e-9,baseline[k])) for k in ['text','background','appearance','audit']}
    review=U.read(out/'visual_review.json') if (out/'visual_review.json').exists() else {'conclusion':'Visual review pending.'}
    table=''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{v["new"][k]:.3f}</td>' for k in ['text','background','appearance','audit','usable'])+'</tr>' for name,v in r['test'].items())
    doc=f'''<!doctype html><meta charset="utf-8"><title>Outlined text: controller test</title><style>body{{font:17px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#172033;background:#f4f6f9}}img{{width:100%}}td,th{{padding:9px 18px;border-bottom:1px solid #ccc;text-align:left}}table{{border-collapse:collapse}}p{{line-height:1.5}}</style>
    <h1>Outlined-text CAPTCHA test</h1><p>{html.escape(review['conclusion'])}</p>
    <p>Unchanged hybrid, perception model, scoring and search. No target strings, labels, fonts or renderer parameters were supplied. 128 fit / 64 selection images; 64 separate test images for MVP 1 and 128 for MVP 2. Known images from the earlier localization pilot were excluded; other historical project exposure is not ruled out.</p>
    <h2>First eight unselected final-test examples</h2><p>Target → original hybrid start → MVP 1 → MVP 2. Synthetic strings differ from targets; aspect ratios are preserved.</p><img src="outlined_comparison.png">
    <h2>Four-arm test</h2><p>All distances below use the new comparator; lower is better. The separate whole-image audit did not select candidates. Usable is a synthetic renderer check, not target OCR accuracy.</p><table><tr><th>Arm</th><th>Text</th><th>Background</th><th>Combined</th><th>Audit</th><th>Usable</th></tr>{table}</table>
    <p>The paired old/new comparators ranked exactly the same 96 candidates. Expanded search adds 96 candidates and controls. Policies were frozen before final test inference.</p><p><a href="results.json">All results</a> · <a href="verification.json">Checks and mask diagnostics</a> · <a href="frozen_policies.json">Frozen settings</a></p>'''
    (out/'report.html').write_text(doc,encoding='utf-8')
    U.dump(out/'summary.json',dict(reduction_vs_mvp1_percent=reductions,visual_review=review))
    print(__import__('json').dumps(dict(reductions=reductions,usable=scores['usable'])))
if __name__=='__main__':main()

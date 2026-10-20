from boot import ROOT,U
import html

def main():
    mask=U.read(ROOT/'mask_training/comparison.json');sections=[];summary={}
    for family in ['City','CryptBB','DarkBay','Outlined']:
        folder=ROOT/(family+'_run_01')
        if not (folder/'results.json').exists():continue
        r=U.read(folder/'results.json');review=U.read(folder/'visual_review.json') if (folder/'visual_review.json').exists() else {'conclusion':'Visual review pending; metric improvements are not a quality pass.'}
        blocks=[];summary[family]={}
        for split,scores in r['scores'].items():
            baseline=scores['v2']['typography'];rows=[]
            for arm,label in [('v2','MVP v2'),('old','Old mask + old score'),('mask','New mask + old score'),('typography','New mask + typography'),('full','Full v3 + broader search')]:
                s=scores[arm]['typography'];change={k:100*(baseline[k]-s[k])/max(1e-10,baseline[k]) for k in ['text','background','appearance','audit']}
                summary[family].setdefault(split,{})[arm]=change
                rows.append('<tr><td>'+label+'</td>'+''.join(f'<td>{s[k]:.3f} ({change[k]:+.1f}%)</td>' for k in ['text','background','audit'])+f'<td>{100*s["usable"]:.1f}%</td></tr>')
            name='Reused diagnostic images' if split=='test' else f'Fresh controller holdout: {r["fresh_count"]} images'
            blocks.append(f'<h3>{name}</h3><table><tr><th>Arm</th><th>Text distance</th><th>Background distance</th><th>Independent audit</th><th>Synthetic usable</th></tr>{"".join(rows)}</table><p>Positive percentages mean lower distance versus v2. These are appearance proxies, not font/OCR accuracy.</p><img loading="lazy" src="{family}_run_01/{split}_preview.png"><details><summary>Predicted text masks (target / old / new)</summary><img loading="lazy" src="{family}_run_01/{split}_masks.png"></details>')
        if not r['fresh_count']:blocks.append('<p><strong>No fresh CryptBB holdout remains. Its numbers are development diagnostics only.</strong></p>')
        sections.append(f'<section><h2>{family}</h2><p><strong>{html.escape(review["conclusion"])}</strong></p>{"".join(blocks)}<p><a href="{family}_run_01/results.json">All scores</a> · <a href="{family}_run_01/frozen_policies.json">Frozen controls</a> · <a href="{family}_run_01/audit.json">Verification</a> · <a href="{family}_run_01/history.json">Candidate history</a></p></section>')
    text=f'''<!doctype html><html><meta charset="utf-8"><title>Controller MVP v3 — no VLMs</title><style>body{{font:16px system-ui;background:#f7f8fa;color:#202633;max-width:1450px;margin:35px auto;padding:0 22px}}h1{{font-size:34px}}section{{background:white;padding:24px;margin:25px 0;border-radius:12px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}p{{line-height:1.5}}summary{{cursor:pointer;margin:20px 0}}</style>
    <h1>Controller MVP v3 — no VLMs</h1><p>Pixels → frozen hybrid starting labels → predicted text masks → typography/background comparison → renderer search.</p>
    <p>Added a synthetic-trained U-Net, hole/stroke/contrast descriptors, and search over all 38 sampled effects. No target answers, boxes, masks, font identities or source recipes enter the controller. Existing font bank retained; no target-library fonts imported.</p>
    <h2>Synthetic mask validation</h2><p>512 validation images using font families excluded from the new model's 4,096-image training split. Mean mask IoU: <strong>{mask['old']['mean_iou']:.1%} → {mask['new']['mean_iou']:.1%}</strong>. Hollow text plus interference: <strong>{mask['old']['groups']['1']:.1%} → {mask['new']['groups']['1']:.1%}</strong>. These are synthetic results; real target mask accuracy is unknown.</p>
    <h2>Evaluation boundaries</h2><p>The three paired search arms rank the same 96 candidate renders. Full v3 adds search candidates and renderer freedom, so its benefit cannot be attributed to one change alone. Controls are selected on fit/selection images and frozen before evaluating these test images. Fresh means excluded from known earlier controller/ablation/localization splits, not never encountered anywhere in the project. First eight images are shown without quality-based selection; generated strings differ from target strings.</p>
    {''.join(sections)}</html>'''
    (ROOT/'report.html').write_text(text,encoding='utf-8');U.dump(ROOT/'summary.json',summary)

if __name__=='__main__':main()


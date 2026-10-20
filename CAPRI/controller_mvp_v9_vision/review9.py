"""Read-only frozen-profile audit and optional independent DINO-v1 diagnostic."""
from features9 import *
from run import load_images
from dino5 import DinoComparator,Objective as DinoObjective
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['sam','dinov3'],required=True);args=ap.parse_args()
    root=ROOT/args.backend;summary=U.read(root/'validated_summary.json');model=DinoComparator();review={}
    for family,result in summary.items():
        dest=root/family;split=U.read(dest/'target_splits.json');target=model.extract(load_images(split['test']))
        groups=[{r['pixel_sha256'] for r in split[k]} for k in ['fit','selection','test']]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
        labels={arm:U.read(dest/('validated_controller' if arm=='controller' else arm)/'labels.json') for arm in ['baseline','controller']};features={}
        for arm,rows in labels.items():
            ims=[np.asarray(Image.open(dest/r['image']).convert('RGB')) for r in rows]
            for r,im in zip(rows,ims):
                if 'pixel_sha256' in r:assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
                else:assert 'error' in r
            features[arm]=model.extract(ims)
        assert [r['seed'] for r in labels['baseline']]==[r['seed'] for r in labels['controller']]
        selection=U.read(dest/'selection.json');eligible=[r for r in selection if r['score']['complete']==1 and r['score']['usable']>=.9]
        best=min(eligible,key=lambda x:x['score']['objective']) if eligible else next(r for r in selection if r['index']==0)
        chosen=U.read(dest/'validated_selected.json')
        assert best['index']==chosen['index'] and best['controls']==chosen['controls'] and U.sha(dest/'validated_selected.json')==result['selected_sha256']
        obj=DinoObjective(target,features['baseline']);scores={a:obj.score(features[a],target,labels[a]) for a in labels}
        review[family]=dict(dino_diagnostic=scores,audit=dict(split_disjoint=True,selection_verified=True,image_hashes_verified=True,matched_seeds=True,
            renderer_errors={a:sum('error' in r for r in rows) for a,rows in labels.items()}))
        print(family,{a:round(s['appearance'],3) for a,s in scores.items()},flush=True)
    U.dump(root/'review.json',review)
    html=['<!doctype html><meta charset="utf-8"><title>DINOv3 and SAM controller MVPs</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:16px;background:#f4f6f8;color:#182033}img{width:100%}td,th{padding:10px;border:1px solid #ccd}table{border-collapse:collapse}</style>',
        '<h1>DINOv3 and SAM controller MVPs</h1><p>Target RGB pixels only. No VLM, OCR, supplied target strings or target masks. Synthetic render labels are saved. Same hybrid recipe initialization and coordinate-search code for both backends.</p>',
        '<p>DINOv3 is prepared and architecture-tested, but cannot be evaluated until approved official ViT-S/16 weights are supplied. No DINOv3 result below is fabricated or substituted with another backbone.</p>' if not (ROOT/'dinov3/summary.json').exists() else '',
        '<p>SAM regions are object-mask proposals filtered with a geometry heuristic. They are not verified character masks. DINOv3 uses dense features and a border-background clustering heuristic. This is a small MVP, with only 3 fit renders per candidate; test targets were previously exposed in earlier experiments. No solver-transfer claim. A validation usability gate was added after the City export exposed failures: require 100% complete and at least 90% usable validation renders, otherwise retain the baseline. This is a development revision, not a pristine blind test.</p>',
        f'<h2>{args.backend.upper()} results</h2><p>Appearance scores exclude renderer-validity penalties. Lower is better. DINO-v1 is an independent diagnostic, not used in this controller search or selection.</p>',
        '<table><tr><th>Family</th><th>Appearance: baseline → controller</th><th>Usable</th><th>DINO diagnostic</th></tr>']
    for family,result in summary.items():
        a=result['scores']['baseline'];b=result['scores']['controller'];d=review[family]['dino_diagnostic']
        html.append(f'<tr><td>{family}</td><td>{a["appearance"]:.3f} → {b["appearance"]:.3f}</td><td>{a["usable"]:.0%} → {b["usable"]:.0%}</td><td>{d["baseline"]["appearance"]:.3f} → {d["controller"]["appearance"]:.3f}</td></tr>')
    html.append('</table>')
    for family in summary:html.extend([f'<h2>{family}</h2>',f'<p>Retained baseline: {summary[family]["retained_baseline"]}. Validation gate passed: {summary[family]["passed_usability_gate"]}.</p>',f'<img src="{args.backend}/{family}/regions.png">',f'<img src="{args.backend}/{family}/validated_renders.png">'])
    html.append('<p>Official sources: <a href="https://github.com/facebookresearch/dinov3">DINOv3</a> | <a href="https://github.com/facebookresearch/segment-anything">SAM</a>.</p>')
    (ROOT/'report.html').write_text('\n'.join(html),encoding='utf8')

if __name__=='__main__':main()

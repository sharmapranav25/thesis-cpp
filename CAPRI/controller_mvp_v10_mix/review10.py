"""Audit selection, scoring and actual saved pixels; render a concise report."""
from mix10 import *
import hashlib

def main():
    summary=U.read(ROOT/'summary.json');audit={};verdict={}
    for family,result in summary.items():
        dest=ROOT/family;selection=U.read(dest/'selection.json');chosen=U.read(dest/'selected.json');eligible=[r for r in selection if r['eligible']]
        assert U.sha(dest/'selected.json')==result['selected_sha256']
        for arm,metric in [('mixed','objective'),('custom_selector','custom_objective'),('dino_selector','dino_objective')]:
            expected=min(eligible,key=lambda r:r['robust'][metric]) if eligible else next(r for r in selection if r['phase']=='neutral')
            assert chosen['winners'][arm]==expected
        seeds=None;counts={}
        for arm in result['scores']:
            labels=U.read(dest/arm/'labels.json');current=[r['seed'] for r in labels]
            if seeds is None:seeds=current
            assert seeds==current
            for row in labels:
                image=np.asarray(Image.open(dest/row['image']).convert('RGB'))
                if 'pixel_sha256' in row:assert hashlib.sha256(image.tobytes()).hexdigest()==row['pixel_sha256']
                else:assert 'error' in row
            counts[arm]=dict(images=len(labels),errors=sum('error' in r for r in labels),unique_pixels=len({r.get('pixel_sha256') for r in labels}))
        scores=[r['score'] for r in U.read(dest/'fit.json')]+[s for r in selection for s in r['scores']]+list(result['scores'].values())
        for s in scores:
            assert abs(s['appearance']-.5*(s['custom_appearance']+s['dino_appearance']))<1e-8
            assert abs(s['objective']-(s['appearance']+s['contrast_penalty']+4*(1-s['complete'])+2*(1-s['usable'])))<1e-8
            assert all(np.isfinite(s[k]) for k in ['appearance','objective','custom_objective','dino_objective'])
        ss=U.read(dest/'target_splits.json');groups=[{r['pixel_sha256'] for r in ss[k]} for k in ['fit','selection','test']]
        assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
        s=result['scores'];m=s['mixed']
        verdict[family]=dict(combined_better_than_both_existing=m['appearance']<min(s[a]['appearance'] for a in ['custom','dino']),
            custom_metric_better_than_custom=m['custom_appearance']<s['custom']['custom_appearance'],dino_metric_better_than_dino=m['dino_appearance']<s['dino']['dino_appearance'],
            pixel_audit_better_than_both=m['audit_pixel_distance']<min(s[a]['audit_pixel_distance'] for a in ['custom','dino']),
            beats_same_pool_selectors=m['appearance']<min(s[a]['appearance'] for a in ['custom_selector','dino_selector']),
            complete=m['complete'],usable=m['usable'],phase=result['selected_phase'])
        audit[family]=dict(selection_verified=True,scoring_equation_verified=True,matched_seeds=True,split_disjoint=True,pixels_verified=True,arms=counts)
    U.dump(ROOT/'audit.json',audit);U.dump(ROOT/'verdict.json',verdict)
    html=['<!doctype html><meta charset="utf-8"><title>Mixed custom and DINO controller</title><style>body{font:17px system-ui;max-width:1250px;margin:30px auto;padding:16px;background:#f4f6f8;color:#182033}img{width:100%}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccd}</style>',
        '<h1>Mixed custom and DINO controller</h1><p>Two frozen comparators score every candidate: custom mask/typography features and DINO v2 features, calibrated equally. Search mixes text/background settings and mutates the resulting recipes. Target input is unlabeled pixels only; no VLM or target strings.</p>',
        '<p>Profiles are selected using two separate validation batches and a renderer-usability gate. Diagnostic targets were previously exposed. Existing controllers had different original budgets; the mixture receives extra search. Component scores are optimization criteria, not independent proof of family resemblance or solver transfer.</p>',
        '<p>The same-pool selector arms isolate how each comparator chooses from the shared shortlist. The shortlist includes mixture-guided mutations, so this is a selector comparison rather than an equal-budget end-to-end retraining benchmark.</p>',
        '<table><tr><th>Family</th><th>Custom</th><th>DINO</th><th>Mixture</th><th>Mixture usable</th><th>Selected recipe</th></tr>']
    for family,r in summary.items():
        s=r['scores'];html.append(f'<tr><td>{family}</td><td>{s["custom"]["appearance"]:.3f}</td><td>{s["dino"]["appearance"]:.3f}</td><td>{s["mixed"]["appearance"]:.3f}</td><td>{s["mixed"]["usable"]:.0%}</td><td>{r["selected_phase"]}</td></tr>')
    html.append('</table><p>Table: combined appearance score, lower is better. Validity penalties excluded.</p>')
    for family,r in summary.items():
        html.extend([f'<h2>{family}</h2>',f'<img src="{family}/renders.png">','<table><tr><th>Arm</th><th>Custom appearance</th><th>DINO appearance</th><th>Pixel audit</th><th>Complete</th><th>Usable</th></tr>'])
        for arm,s in r['scores'].items():html.append(f'<tr><td>{arm}</td><td>{s["custom_appearance"]:.3f}</td><td>{s["dino_appearance"]:.3f}</td><td>{s["audit_pixel_distance"]:.3f}</td><td>{s["complete"]:.0%}</td><td>{s["usable"]:.0%}</td></tr>')
        html.append('</table>')
    html.append('<p>First six samples shown, without visual selection. Synthetic strings are independent of target text. This experiment does not modify the saved two-controller CAPRI workflow.</p>')
    (ROOT/'report.html').write_text('\n'.join(html),encoding='utf8');print(json.dumps(verdict,indent=2))

if __name__=='__main__':main()

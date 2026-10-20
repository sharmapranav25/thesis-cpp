"""Present frozen results; this script never chooses or changes renderer policies."""
from bootstrap import ROOT,U
import html

def main():
    sections=[];summary={}
    for family in ['City','CryptBB','DarkBay']:
        out=ROOT/f'{family}_run_01';r=U.read(out/'results.json');audit=U.read(out/'verification.json')
        review=U.read(out/'visual_review.json') if (out/'visual_review.json').exists() else dict(conclusion='Visual review pending.')
        r['visual_review']=review;U.dump(out/'results.json',r)
        base=r['test']['mvp1']['new'];table=[];deltas={}
        for arm,label in [('mvp1','MVP 1'),('old_comparator','Old score, continued search'),('new_comparator','New score, same candidates'),('expanded','New score + expanded controls')]:
            score=r['test'][arm]['new'];deltas[arm]={k:100*(1-score[k]/max(base[k],1e-9)) for k in ['text','background','appearance','audit']}
            table.append('<tr><td>'+label+'</td>'+''.join(f'<td>{score[k]:.3f}</td>' for k in ['text','background','appearance','audit','usable'])+'</tr>')
        summary[family]=dict(reductions_vs_mvp1_percent=deltas,review=review,verification=audit['passed'])
        section=f'''<section><h2>{family}</h2><p><strong>{html.escape(review['conclusion'])}</strong></p>
        <table><tr><th>Arm</th><th>Text</th><th>Background</th><th>Combined</th><th>Independent audit</th><th>Usable</th></tr>{''.join(table)}</table>
        <p>All appearance distances in this table use the new comparator; lower is better. Audit uses the separate whole-image descriptor. Usable is a synthetic renderer check, not OCR accuracy.</p>
        <img src="{family}_run_01/preview.png"><p>First eight test examples, unselected. Target / MVP 1 / new comparator / expanded controls. Different synthetic strings are intentional.</p>
        <p><a href="{family}_run_01/results.json">All scores</a> · <a href="{family}_run_01/frozen_policies.json">Frozen policies</a> · <a href="{family}_run_01/verification.json">Verification and mask diagnostics</a></p></section>'''
        sections.append(section)
    doc='''<!doctype html><meta charset="utf-8"><title>Controller MVP — iteration 2</title><style>body{font:17px system-ui;max-width:1320px;margin:35px auto;padding:0 24px;background:#f5f7fa;color:#182030}img{width:100%}section{background:white;margin:30px 0;padding:22px;border-radius:12px}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.55}a{color:#2353a0}</style>
    <h1>Controller MVP — iteration 2</h1><p>Stronger glyph and background structure comparisons, editable synthetic length and layout, font mixtures and palette distributions. Target input remains unlabeled pixels.</p>
    <p>The old and new comparators first ranked exactly the same 96 rendered candidates. Expanded search then added 96 candidates and new controls. Thus the paired comparison isolates ranking changes; the expanded result also benefits from additional search.</p>
    <p>Fresh final target images: City 128, CryptBB 24, DarkBay 128. Each arm generates 72 examples with paired seeds. CryptBB's remaining fresh test is small. Policies were frozen before test inference. No test-driven reruns or family-specific rendering rules.</p>'''+''.join(sections)
    (ROOT/'report.html').write_text(doc,encoding='utf-8');U.dump(ROOT/'summary.json',summary)
    print(__import__('json').dumps({k:v['reductions_vs_mvp1_percent'] for k,v in summary.items()},indent=2))
if __name__=='__main__':main()

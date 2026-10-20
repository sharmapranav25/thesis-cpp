import json,csv,html,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]/'run_03'
def read(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def label(off,tuned):
    if tuned and tuned['guard_pass'] and tuned['after_minus_before_95pct'][1]<0 and tuned['audit_gain_pct']>0:
        return 'Tuning helps (exploratory)'
    if off['pixel_changed_fraction']==0:return 'Inactive/no visible intervention'
    if off['after_minus_before_95pct'][0]>0 and off['audit_gain_pct']<0:return 'Removal hurts (exploratory)'
    if off['guard_pass'] and off['after_minus_before_95pct'][1]<0 and off['audit_gain_pct']>0:return 'Removal helps (exploratory)'
    return 'Unresolved or judges disagree'
def pct(v):return f'{v:+.1f}%'
def main():
    records=[];sections=[];audit=[]
    for family in ('City','CryptBB','DarkBay'):
        folder=ROOT/family
        if not (folder/'results.json').exists():continue
        r=read(folder/'results.json');base=r['baseline'];sections.append(f'<h2>{family}</h2>')
        font_extra=read(folder/'font_followup/results.json') if (folder/'font_followup/results.json').exists() else None
        if font_extra:
            for key,v in font_extra['individual'].items():r['individual'][key]['tuned']=v
        sections.append(f'<p>Simple baseline: appearance {base["appearance"]:.4f}; usability {base["usable"]:.1%}; strict readability {base["strict"]:.1%}. Individual tuning covered: {", ".join(r["drill_groups"])}.</p>')
        table=[]
        for g,v in r['groups'].items():
            table.append([g,pct(v['off']['appearance_gain_pct']),pct(v['tuned']['appearance_gain_pct']),pct(v['tuned']['audit_gain_pct']),f'{v["tuned"]["metrics"]["strict"]:.1%}',str(v['tuned']['guard_pass'])])
        sections.append(table_html(['Group','Remove: appearance gain','Tune: appearance gain','Tune: independent audit gain','Strict readability','Guard passes'],table))
        rows=[]
        for key,v in r['individual'].items():
            off,tuned=v['off'],v['tuned'];decision=label(off,tuned)
            record=dict(family=family,feature=key,finding=decision,removal_gain_pct=off['appearance_gain_pct'],removal_audit_gain_pct=off['audit_gain_pct'],
                        removal_ci_low=off['after_minus_before_95pct'][0],removal_ci_high=off['after_minus_before_95pct'][1],
                        changed_fraction=off['pixel_changed_fraction'],tuned=tuned is not None,
                        tuning_gain_pct=tuned['appearance_gain_pct'] if tuned else None,tuning_guard=tuned['guard_pass'] if tuned else None,
                        tuning_audit_gain_pct=tuned['audit_gain_pct'] if tuned else None,
                        tuning_ink_geometry=tuned['metrics']['ink_geometry'] if tuned else None,
                        tuning_ci_low=tuned['after_minus_before_95pct'][0] if tuned else None,tuning_ci_high=tuned['after_minus_before_95pct'][1] if tuned else None)
            records.append(record)
            rows.append([key,pct(off['appearance_gain_pct']),pct(tuned['appearance_gain_pct']) if tuned else 'Not tuned',decision])
        sections.append('<details><summary>All 56 individual style attributes</summary>'+table_html(['Attribute','Removal gain','Tuning gain','Evidence'],rows)+'</details>')
        if font_extra:
            c=font_extra['combined']
            sections.append(f'<p><b>All seven font attributes tuned:</b> selected font-only combination gain {pct(c["appearance_gain_pct"])}, audit gain {pct(c["audit_gain_pct"])}, strict readability {c["metrics"]["strict"]:.1%}, test guard passes: {c["guard_pass"]}. Edits: <code>{html.escape(json.dumps(c["ops"]))}</code>.</p>')
            sections.append('<p>This coverage extension reused existing selections where available. Its fixed catalogue was declared after City ablations were inspected; it reuses the same test sets and is exploratory.</p>')
            if (folder/'font_followup/visual_review.json').exists():
                review=read(folder/'font_followup/visual_review.json')
                sections.append('<p>Font-only coded visual review: '+html.escape(review['assessment'])+'</p>')
                sections.append(f'<a href="{family}/font_followup/coded_comparison.jpg"><img src="{family}/font_followup/coded_comparison.jpg" style="width:100%"></a>')
                sections.append(table_html(['Panel','Font-only variant'],list(read(folder/'font_followup/panel_key.json').items())))
        c=r['combination'];p=r['missing_palette_control']
        sections.append(f'<p>Selection-chosen combination: appearance gain {pct(c["appearance_gain_pct"])}, independent audit gain {pct(c["audit_gain_pct"])}, strict readability {c["metrics"]["strict"]:.1%}, test guard passes: {c["guard_pass"]}. Edits: <code>{html.escape(json.dumps(c["ops"]))}</code>.</p>')
        sections.append(f'<p>Separate missing-palette diagnostic (white canvas + gray effects): appearance gain {pct(p["appearance_gain_pct"])}, audit gain {pct(p["audit_gain_pct"])}, guard passes: {p["guard_pass"]}. This is not a learned feature head.</p>')
        sections.append(f'<a href="{family}/coded_comparison.jpg"><img src="{family}/coded_comparison.jpg" style="width:100%"></a>')
        if (folder/'visual_review.json').exists():
            review=read(folder/'visual_review.json')
            sections.append('<p><b>Visual inspection:</b> '+html.escape(review['assessment'])+'</p>')
            sections.append('<p>Eight consecutive examples inspected with coded panels before consulting the key. This is an assistant visual assessment, not an independent human blind study.</p>')
            sections.append(table_html(['Panel','Variant'],list(read(folder/'panel_key.json').items())))
        recipes=read(folder/'test_recipes.json')
        saved=0;failures=0
        for labels in list(folder.glob('*/labels.jsonl'))+list((folder/'font_followup').glob('*/labels.jsonl')):
            rr=[json.loads(line) for line in labels.read_text().splitlines()]
            assert len(rr)==72
            for i,row in enumerate(rr):
                with Image.open(labels.parent.parent/row['image']) as im:h=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
                assert h==row['pixel_sha256'],str(labels)
                assert row['seed']==recipes[i]['seed']
                if 'text_spec' in row:assert row['text_spec']['text']==recipes[i]['text_spec']['text']
                if row['text_complete']:assert row['text']==recipes[i]['text_spec']['text'] and len(row['glyphs'])==len(row['text'])
                failures+=not row['text_complete'];saved+=1
        audit.append(dict(family=family,saved_pngs_checked=saved,incomplete_saved_renders=failures,strings_seeds_hashes_verified=True))
    if not records:return
    with (ROOT/'feature_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    dump(ROOT/'artifact_verification.json',audit)
    intro='''<h1>Simple-hybrid feature utility experiment</h1>
<p>Frozen simple neural model; no specialist or XGBoost inference, no training. Positive gains mean lower distance than the simple baseline. Each evaluated variant uses 72 paired fresh-text recipes across three seed blocks. Target families are scored separately.</p>
<p>All 56 style attributes were neutralized individually. Five groups were screened; members of the top two groups were tuned. Fonts are reset to explicit defaults, not literally removed. Font substitutions and overlapping width/slant controls cannot isolate intrinsic attributes perfectly.</p>
<p>Selection uses the existing 60% DINO / 40% image-statistics discrepancy with readability penalties. Independent grayscale/HOG and dark-ink geometry are diagnostic only. Dark-ink geometry includes clutter and is not a validated font metric. Strict readability is a synthetic guard, not OCR accuracy.</p>
<p>City reuses recorded fitting/selection/test images, preserving its last 591 unused images. CryptBB and DarkBay use pixel-disjoint 128/64/256 splits. Policies were frozen before this run's test scoring. Bootstrap intervals are descriptive, uncorrected for many comparisons; seed blocks measure rendering variation, not independent model/search retraining.</p>
<p>These are renderer interventions. They identify candidate controls to prioritize; they do not prove that retraining with fewer heads will improve the feature model. Length and content heads are excluded to preserve paired strings.</p>'''
    document='<html><head><meta charset="utf-8"><title>Feature utility results</title><style>body{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#18202c}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}td,th{border:1px solid #d1d8df;padding:8px;text-align:left}th{background:#edf3fa}p{line-height:1.5}code{overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}</style></head><body>'+intro+''.join(sections)+'</body></html>'
    (ROOT/'report.html').write_text(document,encoding='utf-8')
    for family in ('City','CryptBB','DarkBay'):
        rr=[r for r in records if r['family']==family]
        if not rr:continue
        strongest=sorted(rr,key=lambda r:max(abs(r['removal_gain_pct']),r['tuning_gain_pct'] or 0),reverse=True)[:8]
        print(json.dumps(dict(family=family,top=strongest),indent=2))
def table_html(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
if __name__=='__main__':main()

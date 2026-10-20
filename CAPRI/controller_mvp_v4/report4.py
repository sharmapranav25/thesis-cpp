from setup4 import ROOT,U,FAMILIES,ARMS
import html

def main():
    result=U.read(ROOT/'evaluation/results.json');summary=U.read(ROOT/'collections/summary.json');sections=[]
    for family in FAMILIES:
        r=result['families'][family];rows=[]
        if 'scores' in r:
            for arm in ARMS:
                s=r['scores'][arm];rows.append(f'<tr><td>{arm}</td><td>{s["mean_exact"]:.2%}</td><td>{s["mean_cer"]:.2%}</td><td>'+', '.join(f'{x["exact"]:.2%}' for x in s['seeds'])+'</td></tr>')
            table=f'<p>{r["images"]} labeled evaluation images. Split: {r["split_status"]}. Missing labels: {r["unlabeled"]}.</p><table><tr><th>Training arm</th><th>Exact CAPTCHA accuracy ↑</th><th>Character error rate ↓</th><th>Exact accuracy, each seed</th></tr>{"".join(rows)}</table>'
            for baseline in ['hybrid','single']:
                p=r['diverse_vs'][baseline];lo,hi=p['exact_gain_95ci'];table+=f'<p>Diverse minus {baseline}: {100*p["exact_gain"]:+.2f} percentage points exact accuracy; paired image-bootstrap 95% interval [{100*lo:+.2f}, {100*hi:+.2f}].</p>'
        else:table=f'<p>{html.escape(r["status"])}</p>'
        if family=='CryptBB' and (ROOT/'cryptbb_supplement/results.json').exists():
            extra=U.read(ROOT/'cryptbb_supplement/results.json');extra_rows=[]
            for arm in ARMS:
                s=extra['scores'][arm];extra_rows.append(f'<tr><td>{arm}</td><td>{s["mean_exact"]:.2%}</td><td>{s["mean_cer"]:.2%}</td></tr>')
            table+=f'<h3>Supplementary labeled check — {extra["images"]} images</h3><p>Added after the main annotation check found no answers for the 24 diagnostic images. Uses the same frozen readers and all eligible native-size CSV-labeled images, with exact-pixel duplicates and known controller images excluded. This is exploratory and does not replace the primary test.</p><table><tr><th>Arm</th><th>Exact accuracy</th><th>CER</th></tr>{"".join(extra_rows)}</table><p><a href="cryptbb_supplement/results.json">Supplementary results and intervals</a> · <a href="cryptbb_supplement/protocol.json">Selection and provenance</a></p>'
        sections.append(f'<section><h2>{family}</h2>{table}<details><summary>16 retained schemes</summary><img src="collections/{family}/preview.png"></details><p><a href="collections/{family}/archive.json">Renderer configurations</a> · <a href="collections/{family}/selection.json">Selection scores and mask checks</a></p></section>')
    learning=[]
    for name,entry in U.read(ROOT/'solvers/frozen_models.json').items():
        s=entry['selection'];last=U.read(ROOT/'solvers'/name/'history.json')[-1]
        learning.append(f'<tr><td>{name}</td><td>{s["epoch"]}</td><td>{s["validation"]["exact"]:.2%}</td><td>{s["validation"]["cer"]:.2%}</td><td>{last["validation"]["cer"]:.2%}</td><td>{entry["steps"]}</td></tr>')
    learning_html='<section><h2>Common synthetic validation</h2><p>These scores selected checkpoints before real annotations were opened. Poor synthetic recognition limits what the target results can establish.</p><table><tr><th>Reader</th><th>Selected epoch</th><th>Exact accuracy</th><th>CER</th><th>Final epoch CER</th><th>Updates</th></tr>'+''.join(learning)+'</table></section>'
    review=U.read(ROOT/'conclusion.json') if (ROOT/'conclusion.json').exists() else {'text':'Evaluation complete; see accuracy results below. No visual score is treated as solver success.'}
    source=f'''<!doctype html><meta charset="utf-8"><title>Controller MVP v4 — diversity transfer test</title><style>body{{font:17px system-ui;color:#172033;background:#f5f6f8;max-width:1200px;margin:32px auto;padding:0 20px}}section{{padding:24px;background:white;margin:24px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}img{{max-width:100%}}p{{line-height:1.5}}summary{{cursor:pointer;padding:15px 0}}</style>
    <h1>Controller MVP v4</h1><p><strong>{html.escape(review['text'])}</strong></p>
    <p>80 saved renderer configurations: 16 per target family. Shared training adjustments leave 77 distinct effective profiles: 13 for SimpleCaptcha and 16 for each other family. Sampling is balanced over the original slots, so collapsed SimpleCaptcha slots upweight their matching profiles. Selection uses unlabeled target pixels, appearance similarity and diversity. Synthetic mask truth screens out generated samples where predicted localization breaks. No target masks or answers enter selection. No VLMs.</p>
    <p>Three pooled-family training arms: hybrid-only, single eligible controller configuration per family, and a balanced collection of 16 per family. Each gets 12,800 images, the same synthetic transcripts, 20% identical clean anchors, the same common 1,600-image synthetic validation set, two matched initialization seeds and 20 epochs per seed. Transcript lengths 2–8 and the generic 62-character alphabet are shared, independent of target answer statistics.</p>
    <p>Shared recognition curriculum: line layout, compatible full-alphabet fonts, native height 64 before 192×48 reader resizing. All arms retry together when a generated transcript loses characters; after 10/20 failed seeds, all receive 1/2 extra pixels of stroke. The hybrid arm therefore includes these common repairs. It is not an untouched replay. <a href="corpus_review/audit.json">Repair counts and effective configuration counts</a>.</p>
    <p>All six readers and preregistered target predictions were frozen before answer annotations were opened. Primary evaluation: 1,000 Kaggle images reserved, 999 scored; one irregular filename annotation was excluded before scoring. Public dataset answers use original archive filename stems matched by decoded-pixel hash; DarkBay uses the original labeled directory. City and the 24 CryptBB diagnostic images have no verified answers. The separate CryptBB labeled check was added afterward with the same frozen readers. Exact accuracy is case-sensitive; CER includes insertions/deletions/substitutions. Bootstrap intervals resample images while averaging two fixed seeds; they do not measure broad seed variability. When neither arm has successes, a degenerate zero interval does not establish equivalence.</p>
    <p>Dataset provenance: <a href="https://www.kaggle.com/datasets/bjosttveit/simple-captcha">SimpleCaptcha publisher</a> · <a href="https://huggingface.co/datasets/muddokon/outlined-text-captcha">Outlined publisher</a>. Filename annotations are accepted as supplied, without human or OCR re-annotation.</p>
    <img src="target_scores.png" alt="Held-out recognition scores for all training arms">
    {''.join(sections)}{learning_html}<img src="learning_curves.png" alt="Synthetic validation learning curves for all six readers"><p><a href="evaluation/results.json">Raw accuracy results</a> · <a href="solvers/frozen_models.json">Frozen checkpoints</a> · <a href="corpus/audit.json">Training-data audit</a> · <a href="evaluation/answer_provenance.json">Evaluation answer provenance</a> · <a href="final_audit.json">Final integrity checks</a></p>'''
    (ROOT/'report.html').write_text(source,encoding='utf-8')

if __name__=='__main__':main()

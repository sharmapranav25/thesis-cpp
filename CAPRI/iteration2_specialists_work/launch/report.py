"""Build a concise human report and a per-head CSV from completed run artifacts."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'run_20261003_01'
BASELINE_REPORT=Path('E:/pranav/Final sem/sourcefree_synth/runs/iter2_baseline_15_20261003_04/baseline_training/final_report.json')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def task_metric(score, task):
    key,name=task['output'],task['name']
    if key.startswith(('eff_','txt_')):
        value=score['effects' if key.startswith('eff_') else 'text_heads'][name]
        return value.get('lvl_pm1',0.) if key.endswith('ordinal') else value['f1']
    if key=='length':return score['reading']['length_head']
    return score['font'][name+'_pm1' if key=='font_ordinal' else
                         {'font_cat':'category','font_stroke':'stroke','font_italic':'italic_f1'}[key]]

def font_table(lines, before, after):
    lines+=['| Font attribute / metric | Simple | Specialist | Change (points) |',
            '|---|---:|---:|---:|']
    for name,key in [('Category accuracy','category'),('Stroke accuracy','stroke'),
                     ('Weight within ±1','weight_pm1'),('Width within ±1','width_pm1'),
                     ('X-height within ±1','x_height_pm1'),('Slant within ±1','slant_pm1'),('Italic F1','italic_f1')]:
        a,b=before[key],after[key]
        lines.append(f'| {name} | {100*a:.2f}% | {100*b:.2f}% | {100*(b-a):+.2f} |')

def main():
    final=RUN/'final';selection=read(final/'selection.json');comparison=read(final/'per_head_comparison.json')
    audit=read(RUN/'label_audit.json');tasks=read(RUN/'task_manifest.json');chosen=selection['selected'];baseline=selection['baseline']['selection']
    teacher={};merge_rows=[]
    for phase in ['presence_teachers','intensity_teachers']:
        path=RUN/phase/'manifest.json'
        if path.exists():
            entries=read(path)['comparison'];teacher.update(entries)
            student=read(RUN/('presence_merge' if phase=='presence_teachers' else 'intensity_merge')/'selected.json')['scores']['selection']
            for task,value in entries.items():
                end=task_metric(student,tasks[task]);best=max(value['baseline'],value['specialist'])
                merge_rows.append(dict(phase=phase,task=task,name=tasks[task]['name'],
                    start_score=value['baseline'],specialist_score=value['specialist'],
                    assigned_teacher_score=best,merged_student_score=end,delta_student_minus_teacher=end-best,
                    teacher_checkpoint=value['teacher_checkpoint']))
    rows=[]
    for task,value in comparison['selection'].items():
        applicable=not tasks[task]['output'].endswith('ordinal') or bool(audit[task]['counts'])
        row=dict(task=task,name=value['name'],score_type='within_one' if 'ordinal' in task else 'f1' if 'presence' in task or task=='font_italic' else 'accuracy',
          baseline=value['baseline'] if applicable else None,final=value['final'] if applicable else None,
          delta=value['delta'] if applicable else None,applicable=applicable,counts=json.dumps(audit[task]['counts']),
          teacher_checkpoint=teacher.get(task,{}).get('teacher_checkpoint',''),
          assigned_teacher_score=max(teacher[task]['baseline'],teacher[task]['specialist']) if task in teacher and applicable else None)
        rows.append(row)
    with open(final/'per_head_comparison.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with open(final/'merged_teacher_comparison.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(merge_rows[0]));writer.writeheader();writer.writerows(merge_rows)
    dataset=read(RUN/'dataset'/'report.json');config=read(RUN/'configuration.json');groups=read(RUN/'presence_groups'/'groups.json')
    assert dataset['stage']=='complete' and not dataset['failures']
    assert all(r['generated']==100 and r['failed']==0 for r in dataset['audit'].values())
    baseline_tests=read(BASELINE_REPORT)
    assert Path(baseline_tests['chosen']).name=='epoch_15'
    selected=chosen['scores']['selection'];valid=[r for r in rows if r['applicable']]
    improved=sum(r['delta']>0 for r in valid);regressed=sum(r['delta']<0 for r in valid)
    lines=['# Iteration-2 specialist results','',f"Selected checkpoint: `{final/'hybrid.pt'}`.",
      f"SHA256: `{chosen['checkpoint_sha256']}`.",'',
      f"Completed: five presence specialists × four 20k-draw epochs, eight full-100k presence merge epochs, three intensity specialists × four 20k-draw epochs, and four full-100k intensity merge epochs. Selected intensity merge epoch 4. Training settings stayed fixed throughout.",'',
      f"Presence grouping: {len(groups)} groups, {sum(map(len,groups.values()))} assigned tasks. Every presence specialist starts from the original frozen baseline; intensity specialists start from presence merge epoch 8. Student merges use ground truth plus assigned-output distillation.",'',
      f"On selection, {improved}/{len(valid)} scored outputs improved and {regressed}/{len(valid)} regressed. Seven binary text features have no intensity target and are explicitly excluded from scored ordinal comparisons.",'',
      '| Selection metric | Baseline | Final |','|---|---:|---:|']
    metrics=[('Effect presence F1','eff_f1'),('Text presence F1','txt_f1'),('Effect positive-level within-one','eff_lvl_pm1'),
             ('CTC exact','read_exact'),('Character error rate','cer'),('Length accuracy','length'),('Font category accuracy','font_cat'),('Font ordinal within-one','font_ord_pm1')]
    for name,key in metrics:lines.append(f"| {name} | {baseline['summary'][key]:.4f} | {selected['summary'][key]:.4f} |")
    lines+=['','These selection presence values use the same 0.5 threshold. Calibration thresholds are fitted separately to maximize F1 subject to FPR <= .10. Checkpoint choice uses selection only with the preservation gates recorded before launch. The comparison is against the saved 15-epoch baseline; the specialist pipeline received additional training. No equal-compute continued-simple control was run, so this does not isolate specialization from the benefit of more updates.','',
            '## All font attributes on selection','']
    font_table(lines,baseline['font'],selected['font'])
    lines+=['','The ordinal scores permit an error of one bin; they are not exact-value accuracy. These are per-attribute scores, not the proportion of images with every font attribute correct. Slant is heavily imbalanced and its text/font targets duplicate each other; its high score should not be read as strong evidence of useful slant inference.','',
      '## Reused benchmark comparison','',
      '| Benchmark / metric | Simple | Specialist | Change (points) |',
      '|---|---:|---:|---:|']
    benchmark_comparison={}
    for split in ['seen_test','generalisation']:
        s=read(final/f'{split}.json');before=baseline_tests['tests'][split]
        benchmark_comparison[split]=dict(baseline=before,final=s)
        for label,key in [('Effect F1 at .5','eff_f1'),('Text F1 at .5','txt_f1'),('Effect intensity within ±1','eff_lvl_pm1'),
                          ('CTC exact','read_exact'),('Length accuracy','length')]:
            a,b=before['summary'][key],s['summary'][key]
            lines.append(f'| {split} / {label} | {100*a:.2f}% | {100*b:.2f}% | {100*(b-a):+.2f} |')
    lines+=['','These are reused benchmark sets already observed in previous experiments, not untouched validation evidence. Tests are not used for selection. Historical baseline results are read from its saved final report; the baseline was not scored again. Presence summary F1 in the table above uses .5 for both models. Final per-head F1/recall/FPR in the JSON uses calibration thresholds. Intensity scores use measured positive levels; font ordinal zero is valid.']
    for split in ['seen_test','generalisation']:
        lines+=['',f'### Font attributes: {split}','']
        font_table(lines,benchmark_comparison[split]['baseline']['font'],benchmark_comparison[split]['final']['font'])
    lines+=['','Font category improves 4.80 points on seen_test but only 0.68 points on held-out-font generalisation (56.56% → 57.24%). Synthetic selection gains are clearer than the evidence for category transfer to new fonts. The real families do not have measured font ground truth, so no real-family font-accuracy claim is made.','',
            '| Family | Predictions | Previews | Strong readability passed | Mapping distance |','|---|---:|---:|---:|---:|']
    manifest=read(RUN/'real_inference'/'manifest.json')
    for family,r in dataset['audit'].items():lines.append(f"| {family} | {manifest['counts'][family]} | {r['generated']} | {r['strict_training_readability_passed']}/100 | {r['mean_mapping_distance']:.4f} |")
    lines+=['','All 300 PNGs pass renderer usability, exact requested character counts, unique pixel hashes, saved image integrity, and unchanged sampled-prototype checks. The original six-attempt run failed only CryptBB row 25/prototype 415. Retrying the same prototype succeeded on zero-based attempt 8; the original failed report/labels and exact retry seed are preserved. No predictions, training settings, sampled prototypes or checkpoint weights were altered. Completion now rejects partial preview reports rather than treating file existence as success.','',
      '## Visual assessment','',
      'The three sheets were inspected against saved real reference examples. This is qualitative visual review, not a blinded similarity score. City has predominantly five digits and bands, but excessive font/palette/curves variation. DarkBay often has six characters and blocks/inverted regions, but font families, case mix, block geometry and colours remain far from the reference. CryptBB is the weakest: many short strings and thin/narrow glyphs with noisy grids/blobs, lacking the heavy multicolour mixed-font target appearance.','',
      'Whole-set predictions show the upstream problem: CryptBB length 2–4 in 487/600 images and predicted font weight 1–2 in 456/600. City is digits in all 2,000 predictions, but font categories remain spread across five classes. DarkBay blocks are requested in 75 previews and measured in 74, an additional renderer/mapping discrepancy; decoy characters are requested in five and measured in four. Correct mapping alone cannot fix the wrong predicted count/font/content or infer unspecified colours and geometry.','',
      'Renderer usability is not proof of target-family resemblance. Strict training readability passes only City 77/100, CryptBB 68/100, DarkBay 61/100, so these previews are not an approved solver-training corpus. No solver training was launched.','',
      '![CryptBB previews](../dataset/CryptBB_generated.jpg)','',
      '![DarkBay previews](../dataset/DarkBay_generated.jpg)','',
      '![City previews](../dataset/City_generated.jpg)','',
      'The static corpus lacks high measured thin-stroke and occlusion levels and usually varies one major text appearance property at a time. These coverage limitations remain explicit. All 53 presence pools contain diverse font categories and other features (`pool_diversity.json`).','',
      'Full per-head numbers, regressions, teacher checkpoint routes, label counts and reusable commands are retained in the run artifacts. `merged_teacher_comparison.csv` compares each phase student against the actual assigned teacher score, including baseline/start fallback. Raw/EMA and complete resume checkpoints are saved for every phase. The original baseline hash was verified unchanged.']
    (final/'benchmark_comparison.json').write_text(json.dumps(dict(baseline_source=str(BASELINE_REPORT),
          policy='Historical saved baseline; no benchmark rerun or selection.',results=benchmark_comparison),indent=2)+'\n',encoding='utf-8')
    (final/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(final/'REPORT.md'),scored_outputs=len(valid),improved=improved,regressed=regressed)))

if __name__=='__main__':main()

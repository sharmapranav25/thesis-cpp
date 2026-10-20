from pathlib import Path
import sys,shutil,json
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');HERE=Path(__file__).parent
run=HERE/'v2_tier1_03';dest=ROOT/'runs/style_v2_tier1_03'
complete=(run/'status.json').exists()
dest.mkdir(exist_ok=True)
for name in ('script_snapshot.py','protocol.json','fonts.json','results.json','status.json','whole_image_training.json','single_grid_training.json','multiview_training.json'):
    if (run/name).exists():shutil.copy2(run/name,dest/name)
if complete:
    for filename in ('validate_v2_tier1_03.py','audit_v2_tier1_03.py','build_v2_tier1_03.py'):
        shutil.copy2(HERE/filename,run/filename)
    shutil.copytree(run,dest,dirs_exist_ok=True)
    old=json.loads((HERE/'v2_tier1_02/results.json').read_text());new=json.loads((run/'results.json').read_text())
    rows=[]
    for model in new:
        for metric in ('f1','false_positive_rate','visible_mask_iou'):
            rows.append(f"| {model} | {metric} | "+' | '.join(', '.join(f'{x:.3f}' for x in r[model]['test'][metric]) for r in (old,new))+' |')
    report='# V2 Tier1 revision03: native-resolution feature recovery\n\nArchitectural diagnosis on the same synthetic images as02, now development benchmarks. No fresh acceptance claim. Five feature order: line, curve, dot, rectangle, grain. Added full-resolution detail branch, stride2 skip branch and full-resolution decoder; presence sees mask mean/top32 evidence as well as pooled views. 1800steps vs02 1500, so improvements cannot be attributed solely to architecture. All three view alternatives retrained, synthetic only. Real inference after checkpoint selection only. Density/geometry heads remain deferred.\n\n| Model | Metric | Revision02 | Revision03 |\n|---|---|---|---|\n'+'\n'.join(rows)+'\n\nNo automatic promotion. Same procedural generator and contrast/semantic limitations remain. Reused test data are development data. Fresh independent recipe testing and false-positive controls remain necessary. Native-resolution IoU compared with02 resized-mask IoU is not strictly like-for-like; presence metrics use identical labels. No new paper reproduction claimed.\n'
    fresh=json.loads((run/'fresh_validation.json').read_text())
    report+='\n## Fresh-seed validation and common-resolution localization\n\n420 calibration images and 420 each fresh same-recipe, font-only shift, and curve-recipe-only shift. Seeds disjoint from development corpus. Fonts remain within the same nine-family scope. Original procedural generator remains shared: this is not independent-source validation.\n\n| Multi-view feature | 02 fresh F1 | 03 fresh F1 | 02 FPR | 03 FPR | 02 IoU | 03 IoU |\n|---|---:|---:|---:|---:|---:|---:|\n'
    a=fresh['02_multiview']['fresh_same'];b=fresh['03_multiview']['fresh_same']
    for k,name in enumerate(('line','curve','dot','rectangle','grain')):
        values=[a['fixed']['f1'][k],b['fixed']['f1'][k],a['fixed']['fpr'][k],b['fixed']['fpr'][k],a['common32x64_iou'][k],b['common32x64_iou'][k]]
        report+='| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |\n'
    report+='\nClear synthetic improvement in dot/rectangle presence and sparse localization. Full-view aggregation still does not consistently outperform simpler alternatives. Fixed-threshold line/curve FPR remains high. The calibration trial chose thresholds on separate synthetic Anton-font samples, subject to calibration FPR<=0.10. It generalizes poorly: fresh multiview line recall falls to0.236 and rectangle recall0.404. REJECT this calibration as a deployment policy; it illustrates font/selection distribution sensitivity, not a fix. Report fixed and calibrated results separately; never hide lost recall.\n\nAssistant visual review of20 previously exposed real fit examples finds uncertain rectangle responses, including missed backgrounds. No manual feature labels, quantitative real accuracy, rendering promotion or real gate pass. Next substantive work: varied synthetic background strips/low-contrast blocks; semantic separation of nearly-straight curves; feature-isolated examples and glyph-only hard negatives; perceptual visibility audit; multi-font calibration with reserved seeds. Do not declare native resolution alone sufficient.\n'
    (ROOT/'docs/V2_TIER1_REVISION03.md').write_text(report,encoding='utf8')
    message='D080 outcome: revision03 and fresh validation complete; V2_TIER1_REVISION03.md. Fresh multiview dot F1 .587->.860, rectangle .585->.828; false positives .575->.028 and .252->.016. Common-resolution dot IoU .018->.211. Lines/curves still high FPR. Separate calibration rejected (poor recall/shift robustness). No promotion, no active training. Preserved02/03, five cohorts/nine fonts. Next generator visibility/hard negatives and multi-font calibration; no PARSeq or Phase3. Automation remains paused.'
else:
    message='D080: User requested fixing known weaknesses rather than stopping at diagnosis. Revision03 tests a native-resolution detail decoder with stride2 and stride8 context plus spatial evidence for presence. Reason:02 reduced sparse features to8x16 then upsampled. First isolate architecture on same synthetic corpus; previously examined test/heldout now development, not unseen acceptance. 1800steps/model; whole/single/multi comparisons. Five real cohorts/nine fonts retained; no real training, no PARSeq, no Phase3. Further generator hard-negative/contrast work follows this diagnostic.'
for name in ('STYLE_DECISIONS.md','STYLE_HANDOFF.md'):
    with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
sys.path.insert(0,str(ROOT));from style_pipeline.monitor import update
update('v2_tier1_03_complete' if complete else 'v2_tier1_03_training',message,running=not complete,run=str(dest))
print(message)

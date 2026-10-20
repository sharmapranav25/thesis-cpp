import sys
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,load
root=Path(__file__).resolve().parents[1]
run=root/'runs/feature_bank_pilot_01'
status=read(run/'status.json'); inventory=read(run/'resource_inventory.json')
notes={
 'gregwar':'OBSERVED: straight strokes and deformation are visible. Cases 048–051 reach the text edge; reject for supervised use pending clipping repair. Whole-image warp also bends interference lines.',
 'securimage':'OBSERVED: curves and fine scattered marks are visible. Some curve segments are nearly straight. Enabling noise changes text position despite the same seed; these four-column rows are not counterfactual pairs.',
 'mobicms':'OBSERVED: varied fonts, colors and rotations. All initial seed-only repeats differ. Native decorative fonts were not used; case-tuning disabled. Capture/replay repair is a separate run.',
 'simple':'OBSERVED: shadow and rotation are visible. Some text/shadows approach or cross an edge; e.g. 022–023. Corner checks flag four cases but are not a complete clipping detector. No bundled texture backgrounds were used because per-asset licenses remain unresolved.',
 'python':'OBSERVED: thin lines, sparse dots, grid and block combinations are visible. Several marks are faint; nominal requested operations must not automatically become positive visible labels. Glyph shapes are stretched by the existing renderer. Internal geometric checks are not readability proof.'}
review=dict(reviewer='assistant visual review',scope='All 320 initial pilot images, ten contact sheets opened',
             notes=notes,stage_gate='PARTIAL_NOT_PASSED',
             reasons=['Final per-effect visibility labels/masks incomplete','Clipping guards incomplete',
                      'Full dependency resolution and withheld renderer pending','Several vocabulary features absent from pilot',
                      'Real evaluation labels not user-audited'])
write(run/'visual_review.json',review)
lines=['# Feature-bank pilot 01 — resource and adapter gate','',
 'Date: 2026-09-28 UTC. **PARTIAL — supervised-corpus gate not passed.**','',
 '**OBSERVED:** 64 initial cases for each of four PHP backends and the existing Python renderer: 320 images. Every image was decoded and all ten contact sheets opened for assistant review. Each initial successful render was independently repeated. No synthetic training corpus or classifiers were trained.','',
 '| Backend | Rendered | Exact seed replays | Edge / bbox flags |','|---|---:|---:|---:|']
for backend,s in status['backends'].items(): lines.append(f"| {backend} | {s['rendered']}/64 | {s['exact_replays']}/64 | {s['edge_or_bbox_flags']} |")
lines += ['', '**OBSERVED:** All five invalid-parameter tests passed and all six annotation-gate unit tests passed. No numerical classifier metrics exist. Initial run elapsed time: %.2f seconds, including repeat renders and source verification, excluding visual review and later repairs.'%status['elapsed_seconds'],'',
 f"**OBSERVED:** {inventory['downloaded_source_files_unchanged']} downloaded source files and {inventory['historical_source_paths_unchanged']} original/vendor paths retained their recorded hashes. The original repository and historical experiments were not edited.",'',
 '## Visual findings','']
for backend,note in notes.items():
    lines += [f'### {backend}','',note,'',f'[Cases 000–031](../../runs/feature_bank_pilot_01/{backend}_gallery_1.png) · [Cases 032–063](../../runs/feature_bank_pilot_01/{backend}_gallery_2.png) · [Per-case records](../../runs/feature_bank_pilot_01/{backend}/results.json)','']
repair=root/'runs/feature_bank_mobicms_replay_02/status.json'
if repair.exists():
    s=read(repair)['summary']
    lines += ['## MobiCMS repair 02','',f"**OBSERVED:** The separately instrumented adapter records native random_int draws and validates their bounds/order when replayed. {s['exact_replays']}/64 outputs reproduce exactly in the repair run. The one-import change and limitation are preserved in [instrumentation.json](../../runs/feature_bank_mobicms_replay_02/instrumentation.json) and [diff](../../runs/feature_bank_mobicms_replay_02/instrumentation.diff). This resolves the replay issue for the instrumented variant; it does not pass clipping or final visible-label checks.",'']
lines += ['## Licenses and dependency scope','',
 '**READ:** Gregwar and MobiCMS code notices are MIT; simple-php-captcha is MIT. Securimage composer metadata says BSD-3-Clause while its LICENSE.txt contains two redistribution conditions; preserve both and resolve the discrepancy before redistribution. Gregwar supplies per-font OFL notices. The three pilot fonts passed ASCII alphanumeric cmap checks and their notices were copied alongside the MobiCMS font subset. This does not prove visual distinguishability of every glyph.','',
 '**READ:** Securimage supplies a separate freeware font notice. MobiCMS baby_blocks.txt explicitly says its license is unknown; other decorative-font rights need per-asset review. simple-php-captcha attributes its textures and font but does not establish their individual licenses in the readme. Those unresolved assets were not used in pilots.','',
 '**OBSERVED:** [pilot_runtime.lock.json](../../runs/feature_bank_pilot_01/pilot_runtime.lock.json) freezes the direct-load execution closure and runtime; it is not a Composer dependency lock. Gregwar declares symfony/finder but the exercised code did not load it. Full dependency resolution, including Laminas, remains pending. Laminas has not been piloted or used in training.','',
 '## Gate and next work','',
 '**PROPOSED:** Finish per-glyph clipping rejection, captured-randomness effect toggles and final visible-effect masks; isolate additional background/text operations; resolve dependency and asset gaps; pilot the withheld renderer. Rerun into new directories. Only then build the capped corpus. Do not silently discard these failures or mark the entire stage passed.','',
 '**PENDING:** [User annotation audit](ANNOTATION_PROTOCOL.md). The draft labels remain evaluation-only and cannot authorize scored real-image claims. Classifier ordering remains CV rules → logistic regression → Extra Trees → selection/matching, with five-dataset review gates.','',
 '## Related records','',
 '- [Acquisition](RESOURCE_ACQUISITION_01.md)','- [Adapter design and limitations](ADAPTERS.md)','- [Annotation protocol](ANNOTATION_PROTOCOL.md)','- [Mapping coverage and gaps](MAPPING_COVERAGE.md)','- [Machine-readable results](../../runs/feature_bank_pilot_01/status.json)','']
(root/'docs/feature_bank/PILOT_01.md').write_text('\n'.join(lines),encoding='utf-8')
# Compact preview: hand-picked to illustrate tested operations, not performance.
canvas=Image.new('RGB',(1050,640),'#eeeeee'); draw=ImageDraw.Draw(canvas)
for j,(backend,indices) in enumerate([('gregwar',[0,1,2]),('securimage',[0,1,2]),('simple',[0,1,3]),('python',[0,1,3])]):
    for k,i in enumerate(indices):
        x=12+k*350; y=12+j*158
        rows=read(run/backend/'results.json'); row=rows[i]
        draw.text((x,y),row['id']+' '+row['mode'],fill='black')
        im=load(run/row['image']); im.thumbnail((328,120)); canvas.paste(im,(x,y+25))
canvas.save(run/'selected_examples.png')
print(root/'docs/feature_bank/PILOT_01.md')

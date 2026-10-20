from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');local=Path(__file__).resolve().parent;target=root/'runs/style_phase3_clutter_ablation_01'
if target.exists():raise RuntimeError('Preserve history')
shutil.copytree(local/'phase3_clutter_ablation_01',target)
for name in ('ablate_phase3_clutter.py','render_color_locked.py'):shutil.copy2(local/name,target/name)
text='''# Color-locked Phase3 clutter ablation01

Four exposed Sloth development samples, five diagnostic variants,20renders. Original full recipe reproduced pixel-exactly for4/4. All glyphRGB and operationcolor membership assertions pass. No fitting, retraining, threshold changes or new-family claim. Cached frozen feature inference plus fresh CV retained; font/effect unchanged.

Across4samples: fullrecipe75lines, totalwidth225; line-count multiplier disabled26lines,totalwidth78; line-width multiplier disabled75lines,totalwidth75. Dots stay2988 in these variants. Disabling dot multiplier instead increases dots to5879; curve-count multiplier removal leaves8curves. Thus line count/width are the measured amplification, not increased dotcount.

Assistant reviewed all20renders: reducing linecount or width reduces clutter but neither solves wronghollowfont or generictextlength. SharedRNG changes geometry/jitter when operationcountschange; colorlocks hold. This is exploratory ablation, not independentqualityvalidation. User requested othercontrols unchanged, so no diagnosticvariant replaces currentsettings.

Next research should examine synthetic-only recipe transfer controls or background-specific validation without silently changing userchosen settings. No newpaperused. Historical/defaultpipeline preserved.
'''
(root/'docs/STYLE_PHASE3_CLUTTER_ABLATION_01.md').write_text(text,encoding='utf-8')
for name,entry in {
'STYLE_DECISIONS.md':'D067: Diagnostic-only colorlocked Sloth ablation,20renders,4pixel-exact baseline replays. Recipe raises26lines to75 and width1 to3; dots reduce5879 to2988. Colorassertionspass. No runtimecontrolchange because user said retainothercontrols. No gatepass/newfamily claim; see STYLE_PHASE3_CLUTTER_ABLATION_01.md.',
'STYLE_HANDOFF.md':'LATEST COMPLETE: colorlocked Phase3 clutter ablation01,20diagnosticrenders,4baseline exactreplays. Linecount/width amplify clutter, dots actually reduced. Usercontrols remain unchanged. No activejob; saved report. Continue respecting colorlocks and userothercontrols constraint; do not silentlypromote an ablation. No newpaper.',
'THESIS_PAPERS.md':'Phase3 clutter ablation is original local one-control-at-a-time diagnostic, no newpaper.20renders confirm linecount/width amplification; not generalization evidence. See STYLE_PHASE3_CLUTTER_ABLATION_01.md.'
}.items():
    with (root/'docs'/name).open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')

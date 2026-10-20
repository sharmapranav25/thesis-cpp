from pathlib import Path
import sys,shutil,json
root=Path('E:/pranav/Final sem/sourcefree_synth');here=Path(__file__).parent
old=here/'direct_phase3b_01';dst=root/'runs/style_direct_phase3b_01'
if not dst.exists():shutil.copytree(old,dst)
for filename,text in {
'STYLE_DECISIONS.md':'D073 — Prior shared Phase3B trial completed90renders (CNN/transformer/GAN,300steps each,86fit30report). Mostly identity: CNN/GAN change2families, transformer4. Protected pixels/colors passed but visual benefit weak. User correctly requests family-isolated training; prior run preserved, not promoted. D074 — Eight separate GANs now initialized independently; no familyconditioning or shared learned weights. Larger96reference cohorts split72fit12selection12report inside existingfit pools, small sources proportional/Pymlex2/1/1. Two synthetic draws per fitting reference. Max1500steps, minimum500, validation every100, patience4; identity baseline eligible. DirectRGB residual projected to input colors, character mask exact. This replaces restrictive spatial-only edits without introducing RGB colors. Candidate color coverage can change. No countmodel, fixed5. Preserve exposure caveats; no pristine evaluation claim.',
'STYLE_HANDOFF.md':'LATEST ACTIVE: separate own-family GAN trial family_phase3b_01 at writable workspace. User rejected shared-family approach. Own real and synthetic fit inputs only, validation checkpoints and early stopping. Local script run_family3b.py. Automation staysPAUSED. Prior sharedrun saved runs/style_direct_phase3b_01. Do not launch duplicates.',
}.items():
    with (root/'docs'/filename).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
sys.path.insert(0,str(root));from style_pipeline.monitor import update
update('family_gan_training','Eight independently trained family GANs running; validation-selected checkpoints, colors/glyphs protected, automation paused.',running=True,run=str(here/'family_phase3b_01'))

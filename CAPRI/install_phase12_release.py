from pathlib import Path
import shutil,sys,subprocess
root=Path('E:/pranav/Final sem/sourcefree_synth');source=Path(__file__).parent/'phase12_release_01';dest=root/'releases/phase12_v1'
if not dest.exists():shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
else:assert (dest/'INTEGRITY.json').read_bytes()==(source/'INTEGRITY.json').read_bytes()
start='''# Phase1+Phase2: current supported entrypoint

The next session should start with [release README](../releases/phase12_v1/README.md) and [PARSeq training handoff](../releases/phase12_v1/TRAINING_HANDOFF.md).

Self-contained generation release: `releases/phase12_v1/pipeline.py`. Frozen models, font files/licenses, code, thresholds and8family profiles are included using relative runtime paths. Historical absolute paths in provenance are informational, not dependencies. Read `VALIDATION.md` and run `verify` before use.

Supported sequence:Phase1 CV10x3blocks + frozen learned feature inference ->Phase2 font-aware rendering. Generic text:3buckets(numeric,alphabetic,alphanumeric) x7lengths(2–8). Learned length module and all failed Phase3refinement variants excluded. No PARSeqtraining performed by this session. Next session handles corpus production, real-label/split audit and training one family at a time.

Defaultbaseline preserves previous directPhase1+Phase2behavior; explicit diverse mode is an unproven augmentation arm. Historical visual/generation limitations are not solved by packaging. Old experiment scripts remain intact for reproducibility and should not be used as the new entrypoint.

Training/refinement jobs are stopped; continuous automation remains paused. User declared Phase3B failed. Prior familyGANhandoff saying ACTIVE is superseded. Do not resume those experiments.
'''
(root/'docs/PHASE12_START_HERE.md').write_text(start,encoding='utf8')
readme=root/'README.md';old=readme.read_text(encoding='utf-8-sig');readme.write_text('> **Current Phase1+Phase2 generation entrypoint:** [Start here](docs/PHASE12_START_HERE.md). This self-contained release supersedes historical run scripts for new corpus generation.\n\n'+old,encoding='utf8')
for filename,text in {
'STYLE_HANDOFF.md':'LATEST USER SCOPE: this session prepares generation pipeline and training guide only, NO PARSeq training. Completed portable releases/phase12_v1; start docs/PHASE12_START_HERE.md.336allfamily samples+freshinference+byteparity+relocation+leaknegative+determinism checks pass. Models/assets/fonts bundled.21generic cells, trainvaltest manifests, no countinference/Phase3. Prior GANtask stopped and failed by user, noactivejob. AutomationPAUSED. Next session trains PARSeq onefamilyatime after corpus/split audits.',
'STYLE_PIPELINE_PLAN.md':'LATEST SUPPORTED IMPLEMENTATION: releases/phase12_v1, docs/PHASE12_START_HERE.md. Phase1+Phase2 only; 2–8lengths ×numeric/alphabetic/alphanumeric generation priors. No real type/count inference. This session did packaging/validation only. Next session owns PARSeq implementation/training/evaluation. Historical variants retained outside release.',
'STYLE_DECISIONS.md':'D075 — User declared Phase3B failed and stopped further experiments. D076 — User requests self-contained Phase1+Phase2 plus training guide, no training in this session. Bundled exact frozen assets and independent fontlicenses with relativepaths; exported96fitprofiles for6families,4Pymlex,16Sloth; provenance retained. Default renderer byteparity8/8. Added explicit newreference CV+frozeninference entrypoint, balanced21cell generation, stabletextpartition and crosspixel/evaluation audit. Tested336images, relocated21, deterministic42, rejectionofreference/evaloverlap. Unprovendiversitymode explicit, no silent baselinechange. No qualitygatepass; no100kcorpus/training claim.',
'THESIS_PAPERS.md':'Phase1+Phase2 release01 is packaging and data-integrity engineering, not a new research method. Existing feature/font methods copied unchanged with source hashes; no new paper adopted. Generic21cell balancing and stable text-hash split are original experiment controls. No PARSeqpaper reproduction/training claimed; next-session guide explicitly separates proposal from executed work. See releases/phase12_v1/VALIDATION.md.'
}.items():
    with (root/'docs'/filename).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
subprocess.run([sys.executable,'-B',str(dest/'pipeline.py'),'verify'],check=True)
sys.path.insert(0,str(root));from style_pipeline.monitor import update
update('phase12_release_ready','Self-contained Phase1+Phase2 generation release verified. Training guide ready; no training running; Phase3 failed/stopped; automation paused.',running=False,release=str(dest))
print('INSTALLED',dest)

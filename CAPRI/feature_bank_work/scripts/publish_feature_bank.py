"""Publish additive experiment files and archive two documentation preimages."""
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--destination',required=True); args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]; dest=Path(args.destination).resolve()
    if dest.name!='sourcefree_synth' or not (dest/'docs/FEATURE_BANK_CLASSIFICATION_PLAN.md').is_file():
        raise ValueError('Unexpected publication root')
    modifications={'docs/FEATURE_BANK_CLASSIFICATION_PLAN.md','docs/STATUS.md'}
    old={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file()}
    files=[p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    # Resolve all collisions before the first write. Preserve every old experiment.
    for p in files:
        rel=p.relative_to(source); target=dest/rel
        if target.exists() and digest(target)!=digest(p): raise FileExistsError('Publication collision: '+str(target))
    print(f'Preflight: {len(old)} existing files protected; {len(files)} experiment files to publish',flush=True)
    for p in files:
        target=dest/p.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(p,target)
    for rel in modifications:
        p=dest/rel
        archive=dest/'docs/archive'/(p.stem+'_before_feature_bank_stage2_20260928.md')
        if archive.exists(): raise FileExistsError('Documentation archive already exists')
        archive.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,archive)
        original=p.read_text(encoding='utf-8')
        if rel.endswith('FEATURE_BANK_CLASSIFICATION_PLAN.md'):
            update='''\n## Execution update — 2026-09-28 UTC\n\n**OBSERVED:** Stage 1 input/selection checks completed across five datasets: 6,708 historical unique images verified, 40 validation images per dataset selected, and 200 assistant feature-label drafts saved. User audit remains pending. Stage 2 initial pilots generated and reviewed 320 images; a separate 64-image MobiCMS random-tape repair achieved exact replay. **Stage 2 is PARTIAL; corpus/classifier/matching gates have not passed.** No model was trained.\n\n- [Pilot results, failures and next gate](feature_bank/PILOT_01.md)\n- [Resources and acquisition](feature_bank/RESOURCE_ACQUISITION_01.md)\n- [Feature definitions and user annotation audit](feature_bank/ANNOTATION_PROTOCOL.md)\n- [Adapter architecture and capability-check limits](feature_bank/ADAPTERS.md)\n- [Feature-to-function coverage and unsupported cases](feature_bank/MAPPING_COVERAGE.md)\n- [User review page](../runs/feature_bank_01/annotation_review.html)\n\nThe dated planning statements below describe the pre-implementation baseline. Original planning text is retained in [the archive](archive/FEATURE_BANK_CLASSIFICATION_PLAN_before_feature_bank_stage2_20260928.md).\n\n'''
            first,rest=original.split('\n',1); p.write_text(first+'\n'+update+rest,encoding='utf-8')
        else:
            p.write_text(original+'''\n\n## Parallel feature-bank experiment — 2026-09-28 UTC\n\n**OBSERVED:** Input references and the 200-image validation selection are frozen; assistant feature-label drafts await user auditing. Four PHP backends plus the existing Python renderer produced 320 reviewed pilot images. MobiCMS needed a separate random-tape adapter; its additional 64 pilot cases now replay exactly. Clipping, reliable visible-effect labels, dependency resolution, missing capabilities and withheld-renderer validation remain open.\n\n**Stage 2 PARTIAL; corpus and classifiers not started.** This does not advance original Methods 2–6. See [pilot results](feature_bank/PILOT_01.md), [annotation audit](feature_bank/ANNOTATION_PROTOCOL.md), and the [master plan](FEATURE_BANK_CLASSIFICATION_PLAN.md).\n''',encoding='utf-8')
    for rel,expected in old.items():
        if rel not in modifications and digest(dest/rel)!=expected:
            raise ValueError('Protected file changed: '+rel)
    for p in files:
        if digest(dest/p.relative_to(source))!=digest(p): raise ValueError('Published checksum mismatch')
    receipt=dict(existing_files_verified_unchanged=len(old)-len(modifications),
                 archived_documentation_preimages=sorted(modifications),published_files=len(files),
                 source=str(source),destination=str(dest),
                 note='Run configs retain original execution paths; published relative report links resolve here. Reruns require new output directories.')
    write(dest/'runs/feature_bank_publication_01/receipt.json',receipt)
    print(receipt)

if __name__=='__main__': main()

"""Separate instrumented MobiCMS variant; preserve the failed original pilot."""
import difflib
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,digest,new_run,pixel_hash
from feature_bank.pilot import execute_php,php_command,gallery,summary

root=Path(__file__).resolve().parents[1]
out=new_run(root/'runs/feature_bank_mobicms_replay_02')
old=read(root/'runs/feature_bank_pilot_01/mobicms/results.json')
source=Path(old[0]['requested']['source'])
derived=out/'instrumented_source'
shutil.copytree(source/'src',derived/'src')
file=derived/'src/Image.php'
before=file.read_text(encoding='utf-8')
needle='use function random_int;'
if before.count(needle)!=1: raise ValueError('Upstream source changed; refusing instrumentation')
after=before.replace(needle,'use function FeatureBank\\Pilot\\random_int;')
file.write_text(after,encoding='utf-8')
(out/'instrumentation.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='upstream/src/Image.php',tofile='instrumented/src/Image.php')),encoding='utf-8')
write(out/'instrumentation.json',dict(upstream_sha256=digest(source/'src/Image.php'),derived_sha256=digest(file),
  change='Only random_int import redirected to record/native-call or validated replay wrapper. Drawing code unchanged.',
  noninterference_limit='Recording delegates every random draw to native random_int. Exact replay is verified; no claim of seeded equivalence to an independent uninstrumented crypto-random run.'))
rows=[]; cmd=php_command(root/'resources/feature_bank_01')
for original in old:
    i=int(original['id'].rsplit('-',1)[1]); folder=out/'mobicms'/f'{i:03d}'; folder.mkdir(parents=True); (folder/'trace').mkdir()
    cfg=dict(original['requested'],source=str(derived),output=str(folder/'image.png'),trace_dir=str(folder/'trace'),capture_random_tape=True)
    row=dict(id=original['id'],mode=original['mode'],image=f'mobicms/{i:03d}/image.png',training_eligible=False)
    try:
        im,meta=execute_php(cmd,cfg,folder)
        repeat=folder/'repeat'; repeat.mkdir(); (repeat/'trace').mkdir()
        cfg2=dict(cfg,output=str(repeat/'image.png'),trace_dir=str(repeat/'trace'),random_tape_replay=cfg['output']+'.random.json')
        im2,_=execute_php(cmd,cfg2,repeat)
        row.update(status='RENDERED',repeat_equal=pixel_hash(im)==pixel_hash(im2),random_draws=len(meta['actual']['random_choices']))
    except Exception as exc: row.update(status='FAILED',error=str(exc))
    rows.append(row)
write(out/'mobicms/results.json',rows); gallery(rows,out,'mobicms')
write(out/'status.json',dict(summary=summary(rows),training_gate='NOT_PASSED: final visibility and clipping checks pending'))
print(summary(rows))

"""One-time, non-destructive extraction of the active CAPRI stack (2026-10-05).

Original experiment code and weights are retained. Every source hash is recorded.
Run once; refuses to replace an already extracted package.
"""
from pathlib import Path
import ast,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'src/capri_controller'
MANIFEST=ROOT/'maintenance/extraction_20261005.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if DEST.exists() and {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}!={'__init__.py'}:
        raise SystemExit('Package already exists; do not overwrite maintained code.')
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'vendor/dino').mkdir(parents=True,exist_ok=True)
    records=[]
    def write(name,source,text):
        path=DEST/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf8')
        records.append(dict(source=source,destination=str(path.relative_to(ROOT)),source_sha256=sha(ROOT/source) if source else None,extracted_sha256=sha(path)))
    def copy(source,name,replacements=()):
        text=(ROOT/source).read_text(encoding='utf-8-sig')
        for before,after in replacements:
            if before not in text:raise ValueError((source,before))
            text=text.replace(before,after)
        write(name,source,text)
    write('__init__.py',None,'"""Maintained pixel-only CAPRI controllers. No VLMs or XGBoost."""\n__version__ = "0.1.0"\n')
    copy('controller_mvp_20261004/comparison.py','features/base.py',[
        ('import common as U','from .. import runtime as U'),('from network import Encoder','from ..networks.encoder import Encoder')])
    copy('controller_mvp_v2/perception.py','features/regions.py',[
        ('from bootstrap import U','from .. import runtime as U'),('from comparison import Comparator,energy','from .base import Comparator,energy')])
    copy('controller_mvp_v3/perception3.py','features/custom.py',[
        ('from boot import ROOT,U','from .. import runtime as U'),('from comparison import descriptors,energy','from .base import descriptors,energy'),
        ('from perception import Perception as V2Perception,extra,Objective as V2Objective','from .regions import Perception as V2Perception,extra,Objective as V2Objective'),
        ('from mask_model import MaskNet','from ..networks.mask import MaskNet'),("ROOT/'mask_training/best.pt'","U.model_path('text_mask')")])
    copy('controller_mvp_v5/dino5.py','features/dino_base.py',[
        ('from setup5 import WEIGHTS_PATH',"from ..runtime import model_path\nWEIGHTS_PATH=model_path('dino')"),
        ('from vendor.vision_transformer import vit_small','from ..vendor.dino.vision_transformer import vit_small')])
    copy('controller_mvp_v7_dino/perception7.py','features/dino.py',[
        ('from setup7 import U','from .. import runtime as U'),('from dino5 import','from .dino_base import'),
        ('from perception import extra','from .regions import extra'),('from perception3 import ink_shape','from .custom import ink_shape')])
    copy('learned_controller_20261004/network.py','networks/encoder.py',[('import scheme as S','from . import schema as S')])
    schema=(ROOT/'learned_controller_20261004/scheme.py').read_text();tree=ast.parse(schema)
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['CONT','CAT','CN','KN','CHARSETS'] for t in n.targets)]
    write('networks/schema.py','learned_controller_20261004/scheme.py','"""Frozen output dimensions used by the custom encoder checkpoint."""\n'+'\n'.join(ast.get_source_segment(schema,n) for n in nodes)+'\n')
    copy('controller_mvp_v3/mask_model.py','networks/mask.py')
    copy('controller_mvp_20261004/render_control.py','rendering/base.py',[('import common as U','from .. import runtime as U')])
    copy('controller_mvp_v2/controls.py','rendering/controls.py',[('from bootstrap import U','from .. import runtime as U'),('import render_control as R','from . import base as R')])
    copy('controller_mvp_v7_dino/render7.py','rendering/renderer.py',[('from setup7 import U','from .. import runtime as U'),('import controls as C','from . import controls as C')])
    text=(ROOT/'controller_mvp_v10_mix/mix10.py').read_text();tree=ast.parse(text)
    classes=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Mixed','Objective']]
    write('controllers.py','controller_mvp_v10_mix/mix10.py','"""Three scoring arms: custom, DINO, and their calibrated mixture."""\nimport json\nfrom .features.custom import Perception,Objective as CustomObjective\nfrom .features.dino import Comparator,Objective7\n\ndef key(p):return json.dumps(p,sort_keys=True)\n\n'+'\n\n'.join(classes)+'\n')
    text=(ROOT/'controller_mvp_20261004/run.py').read_text();tree=ast.parse(text)
    funcs=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['load_images','save_rows']]
    write('artifacts.py','controller_mvp_20261004/run.py','"""Verified pixel input and labeled render output."""\nimport hashlib\nimport numpy as np\nfrom PIL import Image\nfrom . import runtime as U\n\n'+'\n\n'.join(funcs)+'\n')
    original=(ROOT/'capri_darkbay_workflow/pipeline.py').read_text()
    start=original.index('ROOT=Path(__file__).resolve().parent');end=original.index('from feature_bank.iter2.controller')
    text=original[:start]+'from . import runtime as U\nfrom .runtime import WORK,RUN_ROOT as ROOT\nfrom .controllers import Mixed,Objective,key\nfrom .rendering import renderer as R\nfrom .artifacts import load_images,save_rows\n'+original[end:]
    text=text.replace("U.read(WORK/'controller_mvp_v7_dino/DarkBay/target_splits.json')['fit'][:100]","U.read(U.target_manifest())['fit'][:100]")
    text=text.replace("previous=U.read(WORK/'capri_length_100/DarkBay/summary.json');assert counts==previous['counts']","previous=counts.copy()")
    text=text.replace("note='72% agreement is not calibrated accuracy.'","note='Vote agreement is not calibrated accuracy.'")
    text=text.replace("source_hashes=dict(hybrid=U.sha(U.HYBRID))","source_hashes={k:v['sha256'] for k,v in U.registry()['models'].items() if v['status']=='active'}")
    write('pipeline.py','capri_darkbay_workflow/pipeline.py',text)
    original=(ROOT/'capri_darkbay_workflow/corpus.py').read_text()
    original=original.replace('sys.path.insert(0,str(Path(__file__).resolve().parent))','')
    original=original.replace('from pipeline import ROOT,U,R,ARMS,draw_recipe,usable,key','from .pipeline import ROOT,U,R,ARMS,draw_recipe,usable,key')
    write('corpus.py','capri_darkbay_workflow/corpus.py',original)
    for name in ['vision_transformer.py','utils.py','LICENSE','__init__.py']:copy('controller_mvp_v5/vendor/'+name,'vendor/dino/'+name)
    for folder in ['features','rendering','networks','vendor']:write(folder+'/__init__.py',None,'')
    # Copy only selected active checkpoints; keep originals for historical scripts.
    old=json.loads((ROOT/'capri_darkbay_workflow/model_manifest.json').read_text());models={}
    for name,info in old.items():
        source=Path(info['path']);dest=ROOT/'models/active'/name/(name+source.suffix)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(source,dest)
        if sha(source)!=info['sha256'] or sha(dest)!=info['sha256']:raise ValueError('Checkpoint integrity failure '+name)
        models[name]=dict(path=dest.relative_to(ROOT).as_posix(),source=str(source),sha256=info['sha256'],bytes=dest.stat().st_size,status='active')
    models['hybrid']['role']='Simple hybrid: initial labels and length votes; CTC is not decoded.'
    models['custom_encoder']['role']='Frozen custom appearance encoder; not an adjustment policy.'
    models['text_mask']['role']='Synthetic-trained text-region mask estimator.'
    models['dino']['role']='Original DINO ViT-S/8, used by controller iteration v2. Not DINOv2 or DINOv3 pretrained weights.'
    (ROOT/'models/registry.json').write_text(json.dumps(dict(schema_version=1,models=models),indent=2)+'\n')
    (ROOT/'configs').mkdir(exist_ok=True);(ROOT/'configs/targets').mkdir(exist_ok=True)
    refs=json.loads((ROOT/'controller_mvp_v7_dino/DarkBay/target_splits.json').read_text())
    (ROOT/'configs/targets/darkbay.json').write_text(json.dumps(dict(fit=refs['fit'][:100]),indent=2)+'\n')
    runtime=dict(project='E:/pranav/Final sem/sourcefree_synth',renderer='E:/pranav/Final sem/captcha-effects',python='E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe',default_run='capri_darkbay_workflow',target_manifest='configs/targets/darkbay.json')
    (ROOT/'configs/local.json').write_text(json.dumps(runtime,indent=2)+'\n')
    MANIFEST.parent.mkdir(exist_ok=True);MANIFEST.write_text(json.dumps(dict(date='2026-10-05',method='Copy and explicit import rewrites; original sources retained.',files=records),indent=2)+'\n')
    print('Extracted',len(records),'files; centralized',len(models),'verified models.')

if __name__=='__main__':main()

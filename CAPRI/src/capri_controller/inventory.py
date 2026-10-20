"""Metadata-only inventory of experiments and checkpoints; no deletion or model loading."""
import collections,json,os
from datetime import datetime,timezone
from pathlib import Path
from . import runtime as U

def category(name):
    if name in ['src','configs','models','tools','docs','maintenance']:return 'maintained_infrastructure'
    if name=='capri_darkbay_workflow':return 'current_completed_run'
    if name in ['CAPRI','CAPRI_v2','CAPRI_typography_v3','CAPRI_Thesis_Chapter','phase12_release_01','phase12_release_02']:return 'preserved_release'
    if name.startswith(('controller_mvp','learned_controller')):return 'historical_controller_source_and_results'
    if 'xgb' in name:return 'historical_xgboost_not_active'
    if 'specialist' in name:return 'specialist_training_or_comparison'
    if name in ['sourcefree_synth','thesis_regeneration_work','feature_bank_work']:return 'historical_source_or_resources'
    if name=='archive':return 'archived_root_artifacts'
    return 'historical_experiment_or_unclassified'

def refresh():
    groups={};models=[];errors=[];extensions={'.pt','.pth','.ckpt','.safetensors','.onnx'}
    for base,dirs,files in os.walk(U.WORK,followlinks=False,onerror=lambda e:errors.append(str(e))):
        dirs[:]=[d for d in dirs if d not in ['.git','.venv','node_modules','__pycache__'] and not Path(base,d).is_symlink()]
        for name in files:
            path=Path(base,name);rel=path.relative_to(U.WORK);group=rel.parts[0] if len(rel.parts)>1 else '(root files)'
            try:size=path.stat().st_size
            except OSError as e:errors.append(str(e));continue
            g=groups.setdefault(group,dict(category=category(group),files=0,bytes=0,python_files=0,checkpoints=0));g['files']+=1;g['bytes']+=size;g['python_files']+=path.suffix=='.py'
            if path.suffix.lower() in extensions:
                g['checkpoints']+=1;models.append(dict(path=rel.as_posix(),bytes=size,status='active_copy' if rel.parts[:2]==('models','active') else 'historical_not_selected',sha256=None))
    selected=U.registry()['models']
    for m in models:
        match=next((v for v in selected.values() if v['path']==m['path']),None)
        if match:m['sha256']=match['sha256']
    storage=U.CONFIG.get('decoder_storage')
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),scope='Logical workspace catalog, not physical drive usage; active external source paths and storage mappings are in models/registry.json and configs/local.json.',
        configured_decoder_storage=storage,
        totals=dict(files=sum(x['files'] for x in groups.values()),bytes=sum(x['bytes'] for x in groups.values()),checkpoint_files=len(models)),
        exclusions=['.git','.venv','node_modules','__pycache__','directory symlinks'],groups=groups,errors=errors)
    U.dump(U.WORK/'maintenance/workspace_inventory.json',result)
    U.dump(U.WORK/'models/historical_inventory.json',dict(note='Metadata catalog, not a promotion or duplicate-content audit. Historical hashes are not computed.',models=models))
    lines=['# Workspace catalog','','Generated '+result['generated_at']+'. Use the root README for the maintained entry point. Historical folders remain in place because old scripts and evidence use fixed paths.','','| Folder | Classification | Files | GiB | Checkpoints |','|---|---|---:|---:|---:|']
    if storage:
        lines[4:4]=['CAPRI storage mapping: `'+storage['logical_path']+'` points to `'+storage['physical_path']+'`. This catalog uses logical workspace paths; totals are not C-drive usage.','']
    for name,g in sorted(groups.items()):
        label=name if name=='(root files)' else f'[{name}](../{name}/)'
        lines.append(f'| {label} | {g["category"]} | {g["files"]:,} | {g["bytes"]/2**30:.2f} | {g["checkpoints"]} |')
    lines+=['','Historical checkpoint metadata: [inventory](../models/historical_inventory.json). Labels marked unclassified need review before any retirement. No files are deleted by this command.']
    (U.WORK/'docs').mkdir(exist_ok=True);(U.WORK/'docs/WORKSPACE_CATALOG.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result['totals']));print('Inventory errors:',len(errors))

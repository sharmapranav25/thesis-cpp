"""Disposable integration check: dataset guards, epoch loop, checkpoint reload."""
import csv
import json
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'shared'))
from runner import ManifestDataset, Recognizer, dump, environment, evaluate, loader, seed_everything, train

def manifest(folder,name,rows):
    path=folder/name
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['path','label'])
        writer.writerows(rows)
    return path

report={}
with tempfile.TemporaryDirectory(prefix='runner_verification_',dir=root) as tmp:
    folder=Path(tmp)
    texts=['A12','b34','C56','d78','E90']
    for i,text in enumerate(texts):
        im=Image.new('RGB',(128,32),(255-i*10,255,250))
        ImageDraw.Draw(im).text((8,8),text,fill=(0,0,0))
        im.save(folder/f'{i}.png')
    cfg=json.loads((root/'CRNN/config.json').read_text())
    cfg.update(epochs=1,batch_size=2,effective_batch_size=4,num_workers=0)
    cfg['train_manifest']=str(manifest(folder,'train.csv',[(f'{i}.png',texts[i]) for i in range(3)]))
    cfg['val_manifest']=str(manifest(folder,'val.csv',[(f'{i}.png',texts[i]) for i in range(3,5)]))
    seed_everything(42)
    torch.set_num_threads(4)
    device=torch.device('cuda')
    train(cfg,folder,device,'fixture')
    state=torch.load(folder/'checkpoints/fixture/best.pt',map_location='cpu',weights_only=True)
    model=Recognizer(state['config']).to(device)
    model.load_state_dict(state['model'])
    ds=ManifestDataset(cfg['val_manifest'],cfg)
    metrics=evaluate(model,loader(ds,cfg),device)
    for key,value in metrics.items():
        assert state['metrics'][key]==value,(key,value,state['metrics'][key])
    assert state['metrics']['optimizer_steps']==1
    report['epoch_loop_partial_batch_accumulation_and_checkpoint_reload']='passed'
    bad=manifest(folder,'bad.csv',[('0.png','INVALID!')])
    try:
        ManifestDataset(bad,cfg)
        raise AssertionError('Invalid character accepted')
    except ValueError:
        report['invalid_label_rejected']='passed'
    bad=manifest(folder,'duplicates.csv',[('0.png','A12'),('0.png','A12')])
    try:
        ManifestDataset(bad,cfg)
        raise AssertionError('Duplicate image accepted')
    except ValueError:
        report['duplicate_pixels_rejected']='passed'
    cfg['val_manifest']=str(manifest(folder,'overlap.csv',[('0.png','A12')]))
    try:
        train(cfg,folder,device,'must_not_start')
        raise AssertionError('Train/validation overlap accepted')
    except ValueError:
        assert not (folder/'checkpoints/must_not_start').exists()
        report['cross_split_overlap_rejected_before_training']='passed'
report['fixture_weights_retained']=False
report['real_dataset_training_started']=False
dump(root/'runner_verification.json',report)
dump(root/'environment.json',environment())
print(json.dumps(report,indent=2))

"""Lossless conversion of the complete CAPRI corpus to OpenOCR LMDB."""
from pathlib import Path
import collections,hashlib,io,json,sys,zipfile
from .. import runtime as U

RUN=U.WORK/'runs/parseq_darkbay_100k_20261005'
THESIS=Path('E:/pranav/Final sem/CaptchaThesis')
OPENOCR=Path('E:/pranav/Final sem/Models/level 1/upstream/OpenOCR')

def prepare():
    sys.path.insert(0,str(U.WORK/'.runtime/parseq_packages'))
    import lmdb,yaml
    from PIL import Image
    import numpy as np
    RUN.mkdir(parents=True,exist_ok=True)
    source=U.WORK/'capri_darkbay_workflow';manifest=U.read(source/'dataset_manifest.json')
    receipt=RUN/'data_receipt.json';db=RUN/'train_lmdb'
    if receipt.exists():
        saved=U.read(receipt)
        if saved['source_manifest_sha256']!=U.sha(source/'dataset_manifest.json') or saved['lmdb_sha256']!=U.sha(db/'data.mdb'):
            raise RuntimeError('Prepared training data changed')
        print('Prepared dataset already verified:',saved['images']);return
    if db.exists():raise RuntimeError('Incomplete LMDB exists; inspect it before retrying')
    db.mkdir();env=lmdb.open(str(db),map_size=4*1024**3)
    count=0;controllers=collections.Counter();lengths=collections.Counter();characters=set();seen=set();strict=0
    indexfile=(RUN/'sample_provenance.jsonl').open('w',encoding='utf8')
    try:
        for shard in manifest['shards']:
            path=source/shard['path']
            if U.sha(path)!=shard['zip_sha256']:raise RuntimeError('ZIP changed: '+str(path))
            with zipfile.ZipFile(path) as archive,env.begin(write=True) as txn:
                labels=[json.loads(x) for x in archive.read('labels.jsonl').decode().splitlines()]
                assert len(labels)==shard['count']
                for row in labels:
                    data=archive.read(row['image']);image=Image.open(io.BytesIO(data)).convert('RGB');pixels=np.asarray(image)
                    digest=hashlib.sha256(pixels.tobytes()).hexdigest()
                    assert digest==row['pixel_sha256'] and digest not in seen
                    assert row['text']==row['text_spec']['text'] and row['complete'] and row['usable']
                    seen.add(digest);count+=1;text=row['text'];characters.update(text);lengths[len(text)]+=1;controllers[row['controller']]+=1;strict+=row['strict']
                    txn.put(f'image-{count:09d}'.encode(),data);txn.put(f'label-{count:09d}'.encode(),text.encode())
                    txn.put(f'wh-{count:09d}'.encode(),f'{image.width}_{image.height}'.encode())
                    indexfile.write(json.dumps(dict(index=count,shard=shard['path'],image=row['image'],text=text,pixel_sha256=digest,controller=row['controller'],entry_id=row['entry_id']))+'\n')
            if count%10000<334:print('Packed',count,'/100000',flush=True)
        assert count==100000 and dict(controllers)==manifest['controllers']
        with env.begin(write=True) as txn:txn.put(b'num-samples',str(count).encode())
        env.sync()
    finally:env.close();indexfile.close()
    charset_source=THESIS/'inputs/charsets/charset_universal.txt';charset=''.join(charset_source.read_text().splitlines())
    assert len(charset)==62 and characters<=set(charset)
    (RUN/'charset.txt').write_text('\n'.join(charset)+'\n',encoding='utf8')
    config=yaml.safe_load((THESIS/'inputs/configs/parseq.yml').read_text())
    def relocate(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k=='character_dict_path':obj[k]=str(RUN/'charset.txt')
                elif k=='use_space_char':obj[k]=False
                else:relocate(v)
        elif isinstance(obj,list):
            for v in obj:relocate(v)
    relocate(config)
    config['Global'].update(epoch_num=20,checkpoints=None,pretrained_model=None,cal_metric_during_train=False,
        eval_batch_step=2**63-1,eval_epoch_step=2**63-1,save_epoch_step=[1,1],print_batch_step=50,use_tensorboard=False)
    config['Train']['dataset']['data_dir_list']=[str(db)]
    config['Train']['loader']['num_workers']=0  # Windows cannot pickle the live LMDB environment.
    config['Eval']=None
    for seed in [48,123,456]:
        config['Global'].update(seed=seed,output_dir=str(RUN/f'seed_{seed}'),save_res_path=str(RUN/f'seed_{seed}/predictions_unused.txt'))
        (RUN/f'seed_{seed}.yml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf8')
    U.dump(receipt,dict(images=count,controllers=dict(controllers),lengths=dict(lengths),strict_readability=strict,filtering='None; all100k retained',
        source_manifest_sha256=U.sha(source/'dataset_manifest.json'),lmdb_sha256=U.sha(db/'data.mdb'),provenance_sha256=U.sha(RUN/'sample_provenance.jsonl'),
        observed_synthetic_characters=''.join(sorted(characters)),charset_source=str(charset_source),charset_sha256=U.sha(RUN/'charset.txt')))
    U.dump(RUN/'protocol.json',dict(epochs=20,seeds=[48,123,456],final_checkpoint='latest.pth',images=100000,
        architecture='SVTRv2LNConvTwo33 encoder + PARSeq decoder (thesis configuration)',batch_size=128,optimizer='AdamW',lr=.00035,weight_decay=.05,
        scheduler='OneCycleLR',warmup_epochs=2,mixed_precision=True,target_labels=False,target_evaluation_during_training=False,
        checkpoint_selection='Fixed final epoch; no accuracy-based selection',initialization='Pending required scene-text-base.pth; never silently fall back to random initialization',
        adaptation='Apply the newer20-epoch/3-seed/final-checkpoint protocol to PARSeq; primary benchmark model was NRTR.',
        portability_changes=['Windows data-loader workers0','All charset paths agree; spaces disabled','No eval dataset constructed'],
        sources={str(p):U.sha(p) for p in [THESIS/'inputs/configs/parseq.yml',THESIS/'experiments/six_system_benchmark/protocol.py',THESIS/'experiments/six_system_benchmark/train_str.py']}))
    U.dump(RUN/'status.json',dict(stage='prepared_waiting_for_pretrained_checkpoint',training_started=False,optimizer_steps=0))
    print('Prepared and hash-verified all100000 synthetic images',flush=True)

if __name__=='__main__':prepare()

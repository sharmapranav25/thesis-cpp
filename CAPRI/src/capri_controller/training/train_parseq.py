"""Fixed20-epoch, three-seed PARSeq training without target-label access."""
import argparse,gc,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .. import runtime as U
from .prepare_parseq import RUN,OPENOCR

def one(seed,checkpoint,scratch,resume=False):
    import yaml,numpy as np,torch
    sys.path[:0]=[str(U.WORK/'.runtime/parseq_packages'),str(OPENOCR)]
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing silent CPU training')
    receipt=U.read(RUN/'data_receipt.json')
    if U.sha(RUN/'train_lmdb/data.mdb')!=receipt['lmdb_sha256']:raise RuntimeError('Training LMDB changed')
    output=RUN/f'seed_{seed}';latest=output/'latest.pth'
    if output.exists() and not resume:raise RuntimeError('Existing seed directory requires explicit --resume')
    if resume and not latest.exists():raise RuntimeError('Resume requested but no latest.pth exists')
    if not scratch and (checkpoint is None or not checkpoint.is_file()):raise FileNotFoundError('Required pretrained checkpoint absent; no silent scratch fallback')
    config=Config(str(RUN/f'seed_{seed}.yml'))
    config.cfg['Global']['pretrained_model']=str(checkpoint) if checkpoint else None
    config.cfg['Global']['checkpoints']=str(latest) if resume else None
    if config.cfg['Global']['epoch_num']!=20 or config.cfg.get('Eval') is not None:raise RuntimeError('Training protocol mismatch')
    torch.set_num_threads(8)
    trainer=Trainer(config,mode='train',task='rec')
    assert trainer.valid_dataloader is None
    assert len(trainer.train_dataloader.dataset)==100000
    # The upstream decoder has an independent, otherwise unseeded numpy Generator.
    for index,module in enumerate(trainer.model.modules()):
        if isinstance(getattr(module,'rng',None),np.random.Generator):module.rng=np.random.default_rng(seed+index)
    initialization=dict(mode='from_scratch' if scratch else 'pretrained',checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=U.sha(checkpoint) if checkpoint else None,model_parameters=sum(p.numel() for p in trainer.model.parameters()),seed=seed)
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);state=saved.get('state_dict',saved)
        parameters=dict(trainer.model.named_parameters());missing_encoder=[n for n in parameters if n.startswith('encoder.') and n not in state]
        mismatched=[n for n in parameters if n in state and parameters[n].shape!=state[n].shape]
        if missing_encoder or mismatched:raise RuntimeError('Pretrained encoder coverage/shape mismatch: '+str((missing_encoder,mismatched)))
        initialization.update(loaded_parameter_names=[n for n in parameters if n in state],new_parameter_names=[n for n in parameters if n not in state])
        del saved,state
    U.dump(output/'initialization.json',initialization)
    U.dump(RUN/'status.json',dict(stage='training',current_seed=seed,epochs=20,training_started=True,selection='final latest.pth'))
    started=time.time();trainer.train()
    result=torch.load(latest,map_location='cpu',weights_only=False)
    if result['epoch']!=20:raise RuntimeError('Training ended before fixed epoch20')
    if not all(torch.isfinite(v).all() for v in result['state_dict'].values() if torch.is_floating_point(v)):
        raise RuntimeError('Nonfinite final model weights')
    record=dict(seed=seed,epoch=result['epoch'],global_step=result['global_step'],checkpoint=str(latest),sha256=U.sha(latest),seconds=time.time()-started,
        selection='Fixed final epoch; no target labels or accuracy selection',data_sha256=receipt['lmdb_sha256'])
    U.dump(output/'completion.json',record);print(json.dumps(record),flush=True)

def main():
    parser=argparse.ArgumentParser();initial=parser.add_mutually_exclusive_group(required=True)
    initial.add_argument('--checkpoint',type=Path);initial.add_argument('--from-scratch',action='store_true')
    parser.add_argument('--seed',type=int,choices=[48,123,456]);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not (RUN/'data_receipt.json').is_file():raise RuntimeError('Prepare and verify training LMDB first')
    checkpoint=args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():raise FileNotFoundError('Required pretrained checkpoint absent: '+str(checkpoint))
    if args.seed is not None:one(args.seed,checkpoint,args.from_scratch,args.resume);return
    protocol=U.read(RUN/'protocol.json')
    protocol['initialization']='Random initialization, explicitly selected by the launcher; disclosed when proceeding after the user requested a start with the pretrained checkpoint unavailable.' if args.from_scratch else dict(checkpoint=str(checkpoint),sha256=U.sha(checkpoint))
    protocol['independent_decoder_rng']='Seeded from model seed; no entropy-seeded permutation generator'
    U.dump(RUN/'protocol.json',protocol)
    env=os.environ.copy();env['PYTHONPATH']=str(U.WORK/'src')+os.pathsep+env.get('PYTHONPATH','')
    receipts=[]
    for seed in [48,123,456]:
        done=RUN/f'seed_{seed}/completion.json'
        if args.resume and done.exists():
            item=U.read(done)
            if U.sha(item['checkpoint'])!=item['sha256']:raise RuntimeError('Completed checkpoint changed')
            receipts.append(item);continue
        command=[sys.executable,'-m','capri_controller.training.train_parseq','--seed',str(seed)]
        command+=['--from-scratch'] if args.from_scratch else ['--checkpoint',str(checkpoint)]
        if args.resume and (RUN/f'seed_{seed}/latest.pth').exists():command+=['--resume']
        with (RUN/f'seed_{seed}.console.log').open('a',encoding='utf8') as log:
            completed=subprocess.run(command,cwd=U.WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            U.dump(RUN/'status.json',dict(stage='failed',seed=seed,exit_code=completed.returncode,log=str(RUN/f'seed_{seed}.console.log')))
            raise RuntimeError('Training failed; inspect log, no accuracy-based retries: '+str(seed))
        receipts.append(U.read(done))
    U.dump(RUN/'completion.json',dict(seeds=receipts,epochs=20,all_complete=True))
    U.dump(RUN/'status.json',dict(stage='complete',seeds=[48,123,456],epochs=20,training_started=True))

if __name__=='__main__':main()

"""Frozen iteration-2 backbone -> XGBoost heads -> real family profiles -> preview datasets."""
import os,sys,json,time,hashlib,argparse,gc
from pathlib import Path
import numpy as np
import torch,yaml,xgboost as xgb
from sklearn.metrics import average_precision_score
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth');sys.path.insert(0,str(PROJECT))
from feature_bank.iter2 import data as D,metrics as M
from feature_bank.iter2.generate import effect_heads,TEXT_HEADS,CATEGORIES,STROKES
from feature_bank.iter2.model import Iter2Model
from feature_bank.iter2.train import leak_check

PARAMS=dict(tree_method='hist',device='cuda',n_estimators=800,learning_rate=.05,max_depth=6,subsample=.8,colsample_bytree=.8,min_child_weight=1.,reg_lambda=1.,early_stopping_rounds=30,n_jobs=8,random_state=42)
ROOT=PROJECT/'runs/iter2_baseline_15_20261003_04'
CHECKPOINT=ROOT/'baseline_training/epoch_15/ema.pt'
NEURAL_KEYS=('eff_presence','eff_ordinal','txt_presence','txt_ordinal','length','font_cat','font_stroke','font_ordinal','font_italic')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def status(out,stage,**kw):
    value=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw);dump(out/'status.json',value);print(json.dumps(value),flush=True)

class Extractor:
    def __init__(self):
        self.model=Iter2Model(**yaml.safe_load((ROOT/'training_config.yaml').read_text())['model']).cuda().eval()
        self.model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=True));self.model.requires_grad_(False)
        self.saved={};self.hooks=[]
        for name,module in [('cnn',self.model.cnn),('text',self.model.txt_pool)]:
            self.hooks.append(module.register_forward_hook(lambda mod,args,output,name=name:self.saved.__setitem__(name,output.detach())))
    @torch.inference_mode()
    def batch(self,images,cols):
        images=torch.from_numpy(images.copy()).cuda().float()/255;cols=torch.from_numpy(cols).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model(images,cols)
        f=self.saved['cnn'].float();vectors=[]
        for i in range(len(cols)):
            real=f[i,:,:,:int(cols[i])*2];parts=[real.mean((1,2)),real.amax((1,2))]
            edges=np.linspace(0,real.shape[-1],5).round().astype(int)
            for start,end in zip(edges[:-1],edges[1:]):
                segment=real[:,:,start:max(start+1,end)];parts.extend([segment.mean((1,2)),segment.amax((1,2))])
            vectors.append(torch.cat(parts))
        features=dict(C=torch.stack(vectors).cpu().numpy().astype(np.float16),T=self.saved['text'].float().cpu().numpy().astype(np.float16))
        neural={key:pred[key].float().cpu().numpy() for key in NEURAL_KEYS}
        neural['text']=M.decode(pred['ctc'].float().cpu().numpy(),cols.cpu().numpy()*self.model.ctc_frames)
        self.saved.clear();return features,neural
    def close(self):
        for h in self.hooks:h.remove()
        del self.model;gc.collect();torch.cuda.empty_cache()

def extract(out):
    training=D.open_split(ROOT,'train');fixed={s:D.open_split(ROOT,s) for s in ('selection','calibration','seen_test','generalisation')}
    leaks=leak_check(training,[dataset for sets in fixed.values() for dataset in sets]);extractor=Extractor();counts={};start=time.perf_counter()
    for split,sets in [('train',training)]+list(fixed.items()):
        directory=out/'features'/split;directory.mkdir(parents=True)
        items,widths=D.items_of(sets);n=len(items);counts[split]=n
        arrays={key:np.lib.format.open_memmap(directory/f'{key}.npy',mode='w+',dtype=np.float16,shape=shape) for key,shape in dict(C=(n,960),T=(n,24,160)).items()}
        labels={key:np.stack([dataset.ix[key][i] for dataset,i in items]) for key in ('eff_present','eff_level','txt_present','txt_level','chars','length','font')}
        np.savez(directory/'labels.npz',**labels)
        neural={};readings=['']*n
        # Keep output aligned to labels despite width-sorted inference.
        order=np.argsort(widths,kind='stable');chunks=[];current=[]
        for index in order:
            if current and (len(current)>=128 or (len(current)+1)*(int(widths[index])+D.MARGIN)>D.PIXEL_BUDGET):chunks.append(current);current=[]
            current.append(int(index))
        if current:chunks.append(current)
        completed=0
        for batch in chunks:
            b=D.collate([items[i] for i in batch]);f,p=extractor.batch(b['image'],b['cols'])
            for key in arrays:arrays[key][batch]=f[key]
            for key in NEURAL_KEYS:
                if key not in neural:neural[key]=np.zeros((n,)+p[key].shape[1:],np.float32)
                neural[key][batch]=p[key]
            for i,text in zip(batch,p['text']):readings[i]=text
            completed+=len(batch)
            if completed%10000<128 or completed==n:status(out,'extracting',split=split,completed=completed,total=n,seconds=round(time.perf_counter()-start,1))
        for value in arrays.values():value.flush()
        np.savez(directory/'neural.npz',**neural,text=np.asarray(readings));del arrays,labels,neural;gc.collect()
    extractor.close()
    dump(out/'feature_manifest.json',dict(counts=counts,leak_check=leaks,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),features=dict(C='CNN stem spatial mean/max: full image and 4 vertical bands, 960 dimensions',T='Transformer query vectors: 16 text, length, 7 font queries; 160 dimensions each'),backbone_frozen=True))

def threshold(p,y):
    candidates=np.unique(np.r_[np.quantile(p,np.linspace(0,1,401)),1.000001]);best=(-1.,0.,1.000001)
    for t in candidates:
        pred=p>=t;fp=np.sum(pred&(y==0));neg=np.sum(y==0)
        if fp/max(1,neg)>.1:continue
        f1=M.f1(pred,y==1);recall=np.sum(pred&(y==1))/max(1,np.sum(y==1))
        candidate=(f1,recall,float(t))
        if candidate[:2]>best[:2]:best=candidate
    return best[2]

def tasks():
    result=[]
    for group,names in [('eff',effect_heads()),('txt',TEXT_HEADS)]:
        for j,name in enumerate(names):
            feature='C' if group=='eff' else 'T';slot=None if group=='eff' else j
            result.append(dict(name=f'{group}_presence_{name}',group=group,head=name,index=j,kind='binary',feature=feature,slot=slot))
            if group=='eff' or name in ('per_char_rotation','vertical_jitter','size_variation','text_on_curve','slant','overlap_tight_spacing','wide_spacing','condensed','expanded'):
                result.append(dict(name=f'{group}_level_{name}',group=group,head=name,index=j,kind='regression',feature=feature,slot=slot))
    for j,name in enumerate(('category','stroke','weight','width','x_height','slant','italic')):
        result.append(dict(name='font_'+name,group='font',head=name,index=j,kind='multiclass' if j<2 else 'binary' if j==6 else 'regression',feature='T',slot=17+j))
    result.append(dict(name='length',group='length',head='length',index=0,kind='multiclass',feature='T',slot=16))
    return result

def target(labels,task):
    if task['group'] in ('eff','txt'):
        y=labels[task['group']+('_present' if task['kind']=='binary' else '_level')][:,task['index']]
        mask=y>=0 if task['kind']=='binary' else y>0
    elif task['group']=='font':y=labels['font'][:,task['index']];mask=y>=0
    else:y=labels['length'];mask=y>=0
    return y,mask
def matrix(features,task):
    x=features[task['feature']]
    return x if task['slot'] is None else x[:,task['slot']]

def fit(out):
    F={split:{key:np.load(out/'features'/split/f'{key}.npy',mmap_mode='r') for key in ('C','T')} for split in ('train','selection','calibration','seen_test','generalisation')}
    L={split:dict(np.load(out/'features'/split/'labels.npz')) for split in F};spec=[];start=time.perf_counter();all_tasks=tasks()
    for number,task in enumerate(all_tasks):
        task=task.copy();train,mask=target(L['train'],task);selection,smask=target(L['selection'],task);classes=np.unique(train[mask])
        if len(classes)==1:
            task['constant']=float(classes[0]);task['classes']=classes.tolist();task['threshold']=.5;spec.append(task);continue
        xtrain=np.asarray(matrix(F['train'],task)[mask],dtype=np.float32,order='C');xselection=np.asarray(matrix(F['selection'],task)[smask],dtype=np.float32,order='C')
        parameters=PARAMS.copy();task['classes']=classes.tolist()
        if task['kind']=='regression':
            model=xgb.XGBRegressor(objective='reg:squarederror',eval_metric='mae',**parameters)
            ytrain=train[mask];yselection=selection[smask]
            bins=np.bincount(ytrain.astype(int),minlength=11);weight=np.asarray([len(ytrain)/(max(1,np.sum(bins>0))*bins[int(v)]) for v in ytrain],np.float32)
        else:
            ytrain=np.searchsorted(classes,train[mask]);yselection=np.searchsorted(classes,selection[smask])
            if not np.isin(selection[smask],classes).all():raise ValueError('Selection class absent from training')
            objective='binary:logistic' if task['kind']=='binary' else 'multi:softprob'
            if task['kind']=='multiclass':parameters['num_class']=len(classes)
            model=xgb.XGBClassifier(objective=objective,eval_metric='logloss' if task['kind']=='binary' else 'mlogloss',**parameters);weight=None
        model.fit(xtrain,ytrain,sample_weight=weight,eval_set=[(xselection,yselection)],verbose=False)
        task['best_iteration']=int(model.best_iteration);task['model_file']=task['name']+'.json';model.save_model(out/'boosters'/task['model_file'])
        if task['kind']=='binary':
            yc,cmask=target(L['calibration'],task);prob=model.predict_proba(np.asarray(matrix(F['calibration'],task)[cmask],dtype=np.float32,order='C'))[:,1]
            task['threshold']=threshold(prob,yc[cmask]);task['selection_ap']=float(average_precision_score(yselection,model.predict_proba(xselection)[:,1]))
        spec.append(task);dump(out/'heads.json',dict(tasks=spec,params=PARAMS,threshold_rule='Calibration max F1 subject to FPR <= 0.10; includes all-negative cutoff. No test or real-image tuning.',feature_routing='CNN for effects; transformer query vectors for text/font/length. CTC remains unchanged.'))
        status(out,'fitting_xgboost',completed=number+1,total=len(all_tasks),head=task['name'],best_iteration=task['best_iteration'],seconds=round(time.perf_counter()-start,1))
        del model,xtrain,xselection;gc.collect()
    return spec,F,L

def load_models(out,spec):
    models={}
    for task in spec:
        if 'constant' in task:continue
        model=xgb.XGBRegressor() if task['kind']=='regression' else xgb.XGBClassifier();model.load_model(out/'boosters'/task['model_file']);model.set_params(device='cpu');models[task['name']]=model
    return models

def boosted(features,spec,models):
    n=len(features['C']);values={}
    for task in spec:
        if 'constant' in task:value=np.full(n,task['constant']);prob=None
        else:
            x=np.asarray(matrix(features,task),dtype=np.float32,order='C');model=models[task['name']]
            if task['kind']=='binary':prob=model.predict_proba(x)[:,1];value=prob>=task['threshold']
            elif task['kind']=='regression':value=np.rint(np.clip(model.predict(x),0,10)).astype(int);prob=None
            else:value=np.asarray(task['classes'])[model.predict_proba(x).argmax(axis=1)];prob=None
        if task['kind']=='binary' and prob is None:prob=value.astype(float)
        values[task['name']]=dict(value=value,probability=prob)
    return values

def score(out,spec,F,L):
    models=load_models(out,spec);scores={}
    for split in ('selection','seen_test','generalisation'):
        values=boosted(F[split],spec,models);neural=dict(np.load(out/'features'/split/'neural.npz'));P={key:value.copy() for key,value in neural.items()};P['text']=P['text'].tolist()
        for task in spec:
            value=values[task['name']]['value'];group=task['group'];j=task['index']
            if group in ('eff','txt'):
                if task['kind']=='binary':P[group+'_presence'][:,j]=np.where(value,1.,-1.)
                else:P[group+'_ordinal'][:,j]=np.where(np.arange(1,11)[None,:]<=value[:,None],1.,-1.)
            elif group=='length':P['length']=np.eye(7)[value.astype(int)-2]
            elif j<2:P['font_cat' if j==0 else 'font_stroke']=np.eye(5 if j==0 else 3)[value.astype(int)]
            elif j==6:P['font_italic']=np.where(value,1.,-1.)
            else:P['font_ordinal'][:,j-2]=np.where(np.arange(1,11)[None,:]<=value[:,None],1.,-1.)
        labels=L[split].copy();labels['text']=[''.join(M.CHARS[int(c)] for c in row[:int(length)]) for row,length in zip(labels['chars'],labels['length'])]
        labels['eff_present']=labels['eff_present'];labels['txt_present']=labels['txt_present']
        scores[split]=dict(neural=M.scores(neural|{'text':neural['text'].tolist()},labels,effect_heads(),TEXT_HEADS),xgboost=M.scores(P,labels,effect_heads(),TEXT_HEADS))
        # Same calibrated decision rule for a fair neural presence comparison.
        for task in spec:
            if task['kind']!='binary':continue
            group=task['group'];j=task['index']
            if group=='font':continue
            cal=dict(np.load(out/'features/calibration/neural.npz'));y,mask=target(dict(np.load(out/'features/calibration/labels.npz')),task)
            t=threshold(1/(1+np.exp(-cal[group+'_presence'][mask,j])),y[mask])
            pred=1/(1+np.exp(-neural[group+'_presence'][:,j]))>=t
            scores[split]['neural'][group=='eff' and 'effects' or 'text_heads'][task['head']]['calibrated_f1']=M.f1(pred,labels[group+'_present'][:,j]==1)
        status(out,'synthetic_evaluation',split=split,xgb=scores[split]['xgboost']['summary'],neural=scores[split]['neural']['summary'])
    dump(out/'synthetic_results.json',scores);return models

def real_inference(out,spec,models):
    previous=Path('C:/Users/pranav/Documents/ChatGPT/thesis/iter2_full_target_recreation_01');manifest=json.loads((previous/'manifest.json').read_text())
    realdir=out/'real_inference';realdir.mkdir();dump(realdir/'manifest.json',manifest)
    source_rows=[json.loads(line) for line in (previous/'predictions.jsonl').read_text().splitlines()];extractor=Extractor();results=[]
    lookup={task['name']:task for task in spec}
    from PIL import Image
    for begin in range(0,len(source_rows),64):
        rows=source_rows[begin:begin+64];ims=[];cols=[]
        for row in rows:
            with Image.open(row['source']) as image:
                image=image.convert('RGB');w=min(480,round(image.width*96/image.height));ims.append(np.asarray(image.resize((w,96),Image.Resampling.BILINEAR),np.uint8));cols.append((w+7)//8)
        cols=np.asarray(cols,dtype=np.int64);wp=((int(cols.max())*8+D.MARGIN+7)//8)*8
        pixels=np.stack([np.pad(im,((0,0),(0,wp-im.shape[1]),(0,0)),mode='edge') for im in ims]).transpose(0,3,1,2).copy()
        features,neural=extractor.batch(pixels,cols);values=boosted(features,spec,models)
        for j,row in enumerate(rows):
            record={key:row[key] for key in ('family','index','source','source_sha256')};record['effects']={};record['text_features']={}
            for group,names,key in [('eff',effect_heads(),'effects'),('txt',TEXT_HEADS,'text_features')]:
                for name in names:
                    binary=values[f'{group}_presence_{name}'];ordinal=values.get(f'{group}_level_{name}')
                    record[key][name]=dict(probability=float(binary['probability'][j]),present=bool(binary['value'][j]),level=int(ordinal['value'][j]) if ordinal else 0,threshold=lookup[f'{group}_presence_{name}']['threshold'])
            font={name:int(values['font_'+name]['value'][j]) for name in ('category','stroke','weight','width','x_height','slant','italic')}
            font['category']=CATEGORIES[font['category']];font['stroke']=STROKES[font['stroke']];font['italic']=bool(font['italic'])
            record.update(font=font,length=int(values['length']['value'][j]),ctc_diagnostic=neural['text'][j]);results.append(record)
        if len(results)%512<64 or len(results)==len(source_rows):status(out,'real_inference',completed=len(results),total=len(source_rows))
    extractor.close()
    with (realdir/'predictions.jsonl').open('w') as stream:
        for row in results:stream.write(json.dumps(row)+'\n')
    dump(realdir/'report.json',dict(state='complete',epoch=15,checkpoint_sha256=sha(CHECKPOINT),total=len(results),counts=manifest['counts'],method='Frozen CNN/transformer with XGBoost heads; synthetic-calibrated thresholds; CTC unchanged; no real labels used.'))
    return realdir

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--preview-count',type=int,default=100);parser.add_argument('--resume-evaluation',action='store_true');a=parser.parse_args();out=Path(a.out);out.mkdir(exist_ok=a.resume_evaluation);(out/'boosters').mkdir(exist_ok=a.resume_evaluation);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8)
    checkpoint_hash=sha(CHECKPOINT);dump(out/'protocol.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_hash,train_count=100000,params=PARAMS,selection='Early stopping only',calibration='Per binary head max F1 with FPR <= .1',backbone_frozen=True,real_training=False,ctc_changed=False));status(out,'starting')
    if a.resume_evaluation:
        spec=json.loads((out/'heads.json').read_text())['tasks'];assert len(spec)==len(tasks())
        F={split:{key:np.load(out/'features'/split/f'{key}.npy',mmap_mode='r') for key in ('C','T')} for split in ('selection','calibration','seen_test','generalisation')}
        L={split:dict(np.load(out/'features'/split/'labels.npz')) for split in F}
    else:
        extract(out);spec,F,L=fit(out)
    models=score(out,spec,F,L);del F,L;gc.collect();torch.cuda.empty_cache()
    realdir=real_inference(out,spec,models)
    from feature_bank.iter2.controller import build_profiles,generate_profiles
    build_profiles(realdir/'predictions.jsonl',realdir/'manifest.json',out/'profiles');status(out,'generating_family_datasets',count_per_family=a.preview_count)
    generate_profiles(out/'profiles',out/'dataset',a.preview_count,12,42,6)
    assert sha(CHECKPOINT)==checkpoint_hash
    status(out,'complete',checkpoint_unchanged=True,preview_count_per_family=a.preview_count)

if __name__=='__main__':main()

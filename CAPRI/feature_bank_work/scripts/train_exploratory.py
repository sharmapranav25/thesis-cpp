"""User-authorized synthetic-only exploratory models, not full-plan acceptance."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import collections,json,sys,time,warnings
from pathlib import Path
import cv2,joblib,numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,accuracy_score
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_bank.core import read,write,new_run,digest
from feature_bank.descriptors import extract
from feature_bank.labeled_bricks import FEATURES
from feature_bank.regions import check_group_partitions
cv2.setNumThreads(1)
root=Path(__file__).resolve().parents[1];corpus=root/'runs/feature_bank_corpus_candidate_01'
out=new_run(root/'runs/feature_bank_exploratory_01');start=time.perf_counter()
write(out/'protocol.json',dict(scope='User-authorized exploratory synthetic-only training; corpus and real evaluation gates remain incomplete',
    features=list(FEATURES),real_data_access=False,training_limit_per_feature=12000,
    sampling='seeded stratified crop sample; all available held-out known labels used for evaluation',
    uncertainty='unknown/uncertain omitted per feature; base-group dependence retained and reported',
    representation='native statistics plus pooled HOG on proportionally padded 256x96; no whole-image/multiscale aggregation yet',
    ablation='LR selects ordinary or ordinary+feature-pop on synthetic selection AP; ET compares same chosen view',
    calibration='sigmoid fit and thresholds on calibration only; estimates are exploratory in-sample calibration precision',
    selection='LR preferred within .01 AP of best ET on selection',
    no_real_scoring=True,corpus_manifest_sha256=digest(corpus/'bricks.jsonl')))
for rel in ['scripts/train_exploratory.py','feature_bank/descriptors.py','feature_bank/views.py']:
    p=out/'code_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/rel).read_bytes())
audit=read(corpus/'integrity_audit.json')
assert audit['grouped_partitions_pass'] and not audit['near_duplicate_pixel_candidates']
rows=[]
with (corpus/'bricks.jsonl').open() as stream:
    for line in stream:
        row=json.loads(line)
        if row['retained']:rows.append({k:row[k] for k in ('id','base_group','partition','parent_pixel_sha256','crop_pixel_sha256','parent_file','box','labels')})
check_group_partitions(rows)
assert len({r['crop_pixel_sha256'] for r in rows})==len(rows)
parts=np.array([r['partition'] for r in rows]);groups=np.array([r['base_group'] for r in rows])
ys=np.array([[{'present':1,'absent':0}.get(r['labels'][f],-1) for f in FEATURES] for r in rows])
# Preselect training indices before descriptor extraction to keep the run CPU-first.
fit_indices={};needed=set(np.flatnonzero(parts!='train').tolist())
for j,f in enumerate(FEATURES):
    rng=np.random.default_rng(601+j);chosen=[]
    for label in (0,1):
        eligible=np.flatnonzero((parts=='train')&(ys[:,j]==label))
        assert len(eligible)>=1000
        chosen.extend(rng.choice(eligible,min(6000,len(eligible)),replace=False).tolist())
    fit_indices[f]=np.array(sorted(chosen));needed.update(chosen)
needed=sorted(needed);mapping={old:new for new,old in enumerate(needed)}
rows=[rows[i] for i in needed];parts=parts[needed];groups=groups[needed];ys=ys[needed]
fit_indices={f:np.array([mapping[i] for i in indices]) for f,indices in fit_indices.items()}
write(out/'training_membership.json',{f:[rows[i]['id'] for i in ids] for f,ids in fit_indices.items()})
X=None;current=None;im=None
for i,row in enumerate(rows):
    if row['parent_file']!=current:
        if im is not None:im.close()
        im=Image.open(corpus/row['parent_file']).convert('RGB');current=row['parent_file']
    values,base_length=extract(im.crop(row['box']))
    if X is None:X=np.lib.format.open_memmap(out/'descriptors.npy',mode='w+',dtype='float32',shape=(len(rows),len(values)))
    X[i]=values
    if (i+1)%2000==0:print(f'Descriptors {i+1}/{len(rows)}',flush=True)
if im is not None:im.close()
X.flush();write(out/'descriptor_rows.json',[dict(id=r['id'],partition=r['partition'],base_group=r['base_group']) for r in rows])
write(out/'descriptor_schema.json',dict(base_length=base_length,total_length=X.shape[1],rows=len(rows),source='pixels only'))
results={}
def score(model,xx):
    if hasattr(model,'decision_function'):return model.decision_function(xx)
    p=np.clip(model.predict_proba(xx)[:,1],1e-6,1-1e-6);return np.log(p/(1-p))
def thresholds(y,p):
    positive=[];negative=[]
    for t in np.unique(np.quantile(p,np.linspace(0,1,201))):
        hi=p>=t;lo=p<=t
        if hi.sum()>=30 and y[hi].mean()>=.9:positive.append(float(t))
        if lo.sum()>=30 and (1-y[lo]).mean()>=.9:negative.append(float(t))
    if not positive or not negative:return None
    upper=min(positive);lower=max(negative)
    return [lower,upper] if lower<upper else None
def evaluate(y,p,threshold):
    pred=p>=.5;precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    certain=np.zeros(len(y),bool) if threshold is None else (p<=threshold[0])|(p>=threshold[1])
    decisions=np.zeros(len(y),bool) if threshold is None else p>=threshold[1]
    return dict(n=len(y),positive=int(y.sum()),negative=int((1-y).sum()),ap=float(average_precision_score(y,p)),
        precision=float(precision),recall=float(recall),f1=float(f1),coverage=float(certain.mean()),
        selective_accuracy=float(accuracy_score(y[certain],decisions[certain])) if certain.any() else None,
        false_positive_rate=float(pred[y==0].mean()) if (y==0).any() else None)
for j,f in enumerate(FEATURES):
    train=fit_indices[f];sel=np.flatnonzero((parts=='selection')&(ys[:,j]>=0));cal=np.flatnonzero((parts=='calibration')&(ys[:,j]>=0))
    candidates=[];fitted=[]
    for view,end in [('ordinary',base_length),('feature_pop',X.shape[1])]:
        for C in (.1,1,10):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=17))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always');model.fit(X[train,:end],ys[train,j])
            if any(issubclass(w.category,ConvergenceWarning) for w in captured):raise RuntimeError('LR did not converge')
            ap=float(average_precision_score(ys[sel,j],score(model,X[sel,:end])))
            candidates.append(dict(algorithm='LR',view=view,C=C,selection_ap=ap));fitted.append((model,end))
            print(f'{f} LR {view} C={C}: AP={ap:.4f}',flush=True)
    lr_index=max(range(6),key=lambda k:candidates[k]['selection_ap']);end=fitted[lr_index][1]
    for depth in (16,32):
        for leaf in (2,5):
            model=ExtraTreesClassifier(n_estimators=256,max_depth=depth,min_samples_leaf=leaf,max_features='sqrt',class_weight='balanced',n_jobs=4,random_state=17)
            model.fit(X[train,:end],ys[train,j]);ap=float(average_precision_score(ys[sel,j],score(model,X[sel,:end])))
            candidates.append(dict(algorithm='ExtraTrees',view=candidates[lr_index]['view'],depth=depth,leaf=leaf,selection_ap=ap));fitted.append((model,end))
            print(f'{f} ET depth={depth} leaf={leaf}: AP={ap:.4f}',flush=True)
    tree_index=max(range(6,10),key=lambda k:candidates[k]['selection_ap'])
    winner=lr_index if candidates[lr_index]['selection_ap']>=candidates[tree_index]['selection_ap']-.01 else tree_index
    algorithms={}
    for index in (lr_index,tree_index):
        model,end=fitted[index];raw=score(model,X[cal,:end]);sigmoid=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000)
        sigmoid.fit(raw[:,None],ys[cal,j]);prob=sigmoid.predict_proba(raw[:,None])[:,1];threshold=thresholds(ys[cal,j],prob)
        algorithm=candidates[index]['algorithm'];metrics={}
        for part in ('test','withheld_font','withheld_combination'):
            ids=np.flatnonzero((parts==part)&(ys[:,j]>=0));p=sigmoid.predict_proba(score(model,X[ids,:end])[:,None])[:,1]
            metrics[part]=evaluate(ys[ids,j],p,threshold)
            np.savez_compressed(out/f'{f}_{algorithm}_{part}_predictions.npz',indices=ids,y=ys[ids,j],probability=p,groups=groups[ids])
        joblib.dump(dict(model=model,calibrator=sigmoid,thresholds=threshold,descriptor_length=end,feature=f,exploratory=True),out/f'{f}_{algorithm}.joblib',compress=3)
        algorithms[algorithm]=dict(configuration=candidates[index],thresholds=threshold,metrics=metrics)
    results[f]=dict(candidates=candidates,selected=candidates[winner],algorithms=algorithms,
                   training_crops=len(train),training_groups=len(set(groups[train])))
    write(out/'results.json',results)
write(out/'status.json',dict(state='EXPLORATORY_TRAINING_COMPLETE',features=list(FEATURES),real_evaluation_pending=True,
    full_plan_gate_passed=False,elapsed_seconds=time.perf_counter()-start,descriptor_rows=len(rows)))
print('EXPLORATORY TRAINING COMPLETE',flush=True)

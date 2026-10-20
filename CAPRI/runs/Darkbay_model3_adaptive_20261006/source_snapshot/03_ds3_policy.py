"""Learn prediction-risk on synthetic holdout, freeze, then test on real pixels."""
from pathlib import Path
import sys,collections,hashlib,json,math,random
WORK=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(WORK/'src'))
from capri_controller import runtime as U
from .ds3 import ROOT, DATA
sys.path.insert(0,str(WORK/'.runtime/parseq_packages'))
from rapidfuzz.distance import Levenshtein
DS1=WORK/'runs/parseq_darkbay_100k_20261005'
DS2=WORK/'runs/Darkbay_ds2_noFont-intel'
OUT=DS1/'synthetic_validation_shoot_policy'
SEEDS=[48,123,456]


def combine(candidates):
    text,n=collections.Counter(r['prediction'] for r in candidates).most_common(1)[0]
    if n==1:text=max(candidates,key=lambda r:r['confidence'] if r['confidence'] is not None else -1)['prediction']
    support=sum((r['confidence'] or 0) for r in candidates if r['prediction']==text)/3
    return text,support


def wilson(correct,n):
    if not n:return 0
    z=1.959963984540054;p=correct/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def measures(rows):
    n=len(rows);good=sum(r['correct'] for r in rows)
    return dict(attempts=n,correct=good,accuracy_percent=100*good/n if n else None,wilson95_lower_percent=100*wilson(good,n) if n else None)


def main(model2=True):
    assert model2
    OUT=ROOT/'synthetic_validation_shoot_policy'
    if OUT.exists():raise FileExistsError(str(OUT))
    OUT.mkdir()
    samples=U.read(DS2/'curriculum_samples.json')
    assert U.sha(DS2/'curriculum_samples.json')==U.read(DS2/'bucket_summary.json')['curriculum_sha256']
    validation=[r for r in samples if r['split']=='validation'];assert len(validation)==5000
    teachers={}
    for s in SEEDS:
        path=ROOT/f'synthetic_policy_inference/seed_{s}.json' if model2 else DS2/f'teacher_{s}.json'
        receipt=path.with_suffix('.receipt.json') if model2 else DS2/f'teacher_{s}.receipt.json'
        assert U.sha(path)==U.read(receipt)['predictions_sha256']
        teachers[s]={r['index']:r for r in U.read(path)}
    rows=[]
    for r in validation:
        text,confidence=combine([teachers[s][r['index']] for s in SEEDS])
        if not model2:assert text==r['prediction']
        rows.append(dict(index=r['index'],prediction=text,support=confidence,label=r['text'],correct=text==r['text']))
    random.Random(6202601).shuffle(rows)
    fit,cal=rows[:2500],rows[2500:]
    assert not {r['index'] for r in fit}&{r['index'] for r in cal}
    U.dump(OUT/'protocol.json',dict(dataset=('Model3 on identical DS2 synthetic validation; excluded from weight updates but used for checkpoint selection' if model2 else 'DS2 synthetic holdout; unseen by frozen DS1 epoch20 teachers'),
        discovery=2500,calibration=2500,split_seed=6202601,
        score='Product of Laplace-smoothed correctness rates of emitted characters multiplied by mean confidence support for winning string across all three models; ranking score, not calibrated probability.',
        targets=[.90,.95,.98,.99],selection='Maximum coverage with >=100 calibration attempts and Wilson95 lower accuracy bound >= target; no real labels consulted',
        warning='Calibration bound is descriptive after threshold search, not a formal post-selection guarantee. Real test set was examined previously; this is a prospective rule freeze on historical data.',
        code_sha256=U.sha(Path(__file__)),synthetic_source_sha256=U.sha(DS2/'curriculum_samples.json'),
        no_true_label_confidence='NLL and p_correct from dataset scoring are never features; deployment only uses predictions/confidences.'))
    stats=collections.defaultdict(lambda:[0,0]);confusions=collections.Counter();deletions=collections.Counter()
    for r in fit:
        truth,pred=r['label'],r['prediction']
        for tag,a,b,c,d in Levenshtein.opcodes(truth,pred):
            if tag=='equal':
                for char in pred[c:d]:stats[char][0]+=1;stats[char][1]+=1
            elif tag=='replace':
                for true,char in zip(truth[a:b],pred[c:d]):stats[char][1]+=1;confusions[(true,char)]+=1
            elif tag=='insert':
                for char in pred[c:d]:stats[char][1]+=1;confusions[('<insertion>',char)]+=1
            elif tag=='delete':
                for char in truth[a:b]:deletions[char]+=1
    reliability={char:(good+1)/(n+2) for char,(good,n) in stats.items()}
    def score(r):return r['support']*math.prod(reliability.get(c,.5) for c in r['prediction']) if r['prediction'] else 0.
    for r in rows:r['score']=score(r)
    U.dump(OUT/'character_analysis.json',dict(discovery_images=2500,characters={c:dict(correct=g,total=n,reliability=reliability[c]) for c,(g,n) in sorted(stats.items())},
        confusions=[dict(truth=a,predicted=b,count=n) for (a,b),n in confusions.most_common()],deletions=dict(deletions)))
    ordered=sorted(cal,key=lambda r:-r['score']);candidates=[];good=0
    for i,r in enumerate(ordered):
        good+=r['correct'];n=i+1
        if i+1<len(ordered) and ordered[i+1]['score']==r['score']:continue
        if n>=100:candidates.append(dict(threshold=r['score'],n=n,correct=good,lower=wilson(good,n)))
    policies=[]
    for target in [.90,.95,.98,.99]:
        eligible=[r for r in candidates if r['lower']>=target]
        choice=max(eligible,key=lambda r:r['n']) if eligible else None
        policies.append(dict(target_accuracy=target,threshold=choice['threshold'] if choice else None,available=choice is not None))
    frozen=dict(character_reliability=reliability,unseen_character_reliability=.5,policies=policies,
        score='support times product character reliability',discovery_indices=[r['index'] for r in fit],calibration_indices=[r['index'] for r in cal])
    old=U.read(DATA/'synthetic_validation_shoot_policy/frozen_policy.json')
    assert frozen['discovery_indices']==old['discovery_indices'] and frozen['calibration_indices']==old['calibration_indices']
    U.dump(OUT/'frozen_policy.json',frozen);policy_sha=U.sha(OUT/'frozen_policy.json')
    U.dump(OUT/'policy_freeze.json',dict(sha256=policy_sha))
    U.dump(OUT/'synthetic_predictions.json',rows)
    synthetic=[]
    for p in policies:
        selected=[r for r in cal if p['available'] and r['score']>=p['threshold']]
        synthetic.append(dict(**p,**measures(selected),shoot_rate_percent=100*len(selected)/2500))
    # Policy is now frozen. Read real predictions, apply it without truth.
    real_root=ROOT
    real_preds={s:U.read(real_root/f'evaluation_darkbay/seed_{s}_predictions.json') for s in SEEDS}
    original_freeze=U.read(real_root/'evaluation_darkbay/prediction_freeze.json')
    for s in SEEDS:assert U.sha(real_root/f'evaluation_darkbay/seed_{s}_predictions.json')==(original_freeze[f'seed_{s}'] if model2 else original_freeze[str(s)]['sha256'])
    real=[]
    for i in range(900):
        candidates=[real_preds[s][i] for s in SEEDS]
        assert len({r['opaque_id'] for r in candidates})==1
        text,support=combine(candidates)
        r=dict(opaque_id=candidates[0]['opaque_id'],prediction=text,support=support)
        r['score']=score(r)
        r['attempt']=[p['available'] and r['score']>=p['threshold'] for p in policies]
        real.append(r)
    U.dump(OUT/'real_decisions.json',real);decisions_sha=U.sha(OUT/'real_decisions.json')
    U.dump(OUT/'real_decision_freeze.json',dict(sha256=decisions_sha,policy_sha256=policy_sha))
    labels=U.read(DS1/'evaluation_darkbay/scoring_labels.json')
    for r in real:r['correct']=r['prediction']==labels[r['opaque_id']]
    assert sum(r['correct'] for r in real)==U.read(ROOT/'evaluation_darkbay/summary.json')['results']['ensemble']['correct']
    results=[]
    for i,p in enumerate(policies):
        selected=[r for r in real if r['attempt'][i]]
        results.append(dict(**p,**measures(selected),shoot_rate_percent=100*len(selected)/900))
    summary=dict(synthetic_discovery=measures(fit),synthetic_calibration_baseline=measures(cal),synthetic_policies=synthetic,
        real_baseline=measures(real),real_policies=results,policy_sha256=policy_sha,
        interpretation='Rules learned only on synthetic holdout; real labels used only after attempt decisions frozen. Synthetic precision need not transfer to real images.')
    assert U.sha(OUT/'frozen_policy.json')==policy_sha and U.sha(OUT/'real_decisions.json')==decisions_sha
    U.dump(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

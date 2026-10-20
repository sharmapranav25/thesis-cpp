"""Evaluation-only answers must be provided AFTER models and predictions are frozen."""
from setup4 import ROOT,U,FAMILIES,ARMS
import numpy as np
from solver4 import metrics,edit

def main():
    folder=ROOT/'evaluation';frozen_predictions=U.read(folder/'inference_complete.json');assert frozen_predictions['answers_read'] is False
    for name,digest in frozen_predictions['prediction_sha256'].items():assert U.sha(folder/name)==digest,'Frozen predictions changed'
    boundary=U.read(folder/'boundary.json');assert U.sha(ROOT/'solvers/frozen_models.json')==boundary['models_frozen_sha256']
    answers=U.read(folder/'answers.json');results={}
    for family in FAMILIES:
        pred=U.read(folder/(family+'_predictions.json'));lookup=answers.get(family,{});ids=[i for i,h in enumerate(pred['pixel_sha256']) if h in lookup]
        if not ids:
            results[family]=dict(status='No verified ground-truth answers available; accuracy not computed',images=len(pred['pixel_sha256']));continue
        labels=[lookup[pred['pixel_sha256'][i]] for i in ids];scores={};correct={};errors={}
        for arm in ARMS:
            each=[];corr=[];err=[]
            for seed in [43103,43109]:
                predictions=[pred['predictions'][f'{arm}_{seed}'][i] for i in ids];each.append(dict(seed=seed,**metrics(labels,predictions)))
                corr.append([a==b for a,b in zip(labels,predictions)]);err.append([edit(a,b) for a,b in zip(labels,predictions)])
            correct[arm]=np.mean(corr,axis=0);errors[arm]=np.mean(err,axis=0)
            scores[arm]=dict(seeds=each,mean_exact=float(np.mean([r['exact'] for r in each])),mean_cer=float(np.mean([r['cer'] for r in each])))
        rng=np.random.default_rng(45007);paired={};lengths=np.asarray(list(map(len,labels)))
        for baseline in ['hybrid','single']:
            exact=[];cer=[]
            for _ in range(2000):
                ix=rng.integers(0,len(labels),len(labels));exact.append(float((correct['diverse'][ix]-correct[baseline][ix]).mean()))
                cer.append(float((errors[baseline][ix]-errors['diverse'][ix]).sum()/lengths[ix].sum()))
            paired[baseline]=dict(exact_gain=float((correct['diverse']-correct[baseline]).mean()),exact_gain_95ci=np.quantile(exact,[.025,.975]).tolist(),
                cer_reduction=float((errors[baseline]-errors['diverse']).sum()/lengths.sum()),cer_reduction_95ci=np.quantile(cer,[.025,.975]).tolist())
        results[family]=dict(images=len(labels),requested_images=len(pred['pixel_sha256']),unlabeled=len(pred['pixel_sha256'])-len(labels),split_status=pred['status'],scores=scores,diverse_vs=paired)
    U.dump(folder/'results.json',dict(families=results,primary='SimpleCaptcha',uncertainty='Paired bootstrap over images, averaging the two prespecified training seeds. Intervals do not establish robustness across arbitrary training seeds. A degenerate zero interval when neither arm has successes does not establish equivalence or a zero population effect.',answers_sha256=U.sha(folder/'answers.json'),models_frozen_sha256=boundary['models_frozen_sha256']))
    print(results)

if __name__=='__main__':main()

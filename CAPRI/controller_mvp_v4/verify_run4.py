"""Final integrity checks; consumes already-scored evaluation artifacts only."""
from setup4 import ROOT,U,FAMILIES,ARMS
from pathlib import Path
from datetime import datetime,timezone

def main():
    corpus=U.read(ROOT/'corpus/audit.json');assert corpus['passed']
    for folder in ['corpus','solvers']:
        for name,digest in U.read(ROOT/folder/'protocol.json')['source_sha256'].items():assert U.sha(ROOT/name)==digest,'Training source changed after its recorded protocol'
    for family,row in U.read(ROOT/'collections/summary.json').items():assert U.sha(ROOT/'collections'/family/'archive.json')==row['archive_sha256']
    for name,digest in corpus['files'].items():assert U.sha(ROOT/'corpus'/name)==digest
    for split in ['train','validation']:assert U.sha(ROOT/'corpus'/f'{split}_labels.json')==corpus[f'{split}_labels_sha256']
    frozen=U.read(ROOT/'solvers/frozen_models.json')
    assert set(frozen)=={f'{arm}_{seed}' for arm in ARMS for seed in [43103,43109]}
    for seed in [43103,43109]:assert len({frozen[f'{arm}_{seed}']['initial_sha256'] for arm in ARMS})==1
    for name,entry in frozen.items():
        assert entry['steps']==2000 and U.sha(entry['checkpoint'])==entry['sha256']
        history=U.read(ROOT/'solvers'/name/'history.json');assert [r['epoch'] for r in history]==list(range(1,21))
        best=min(history,key=lambda r:(r['validation']['cer'],-r['validation']['exact']))
        assert best==entry['selection']
    predictions=U.read(ROOT/'evaluation/inference_complete.json');assert predictions['answers_read'] is False
    assert predictions['models_frozen_sha256']==U.sha(ROOT/'solvers/frozen_models.json')
    for name,digest in predictions['prediction_sha256'].items():assert U.sha(ROOT/'evaluation'/name)==digest
    results=U.read(ROOT/'evaluation/results.json');assert results['answers_sha256']==U.sha(ROOT/'evaluation/answers.json')
    counts={}
    for family in FAMILIES:
        refs=U.read(ROOT/'evaluation_pixels'/f'{family}.json');pred=U.read(ROOT/'evaluation'/f'{family}_predictions.json')
        assert [r['pixel_sha256'] for r in refs['rows']]==pred['pixel_sha256']
        assert len(set(pred['pixel_sha256']))==len(pred['pixel_sha256'])
        assert all(len(p)==len(pred['pixel_sha256']) for p in pred['predictions'].values())
        counts[family]=dict(predicted=len(pred['pixel_sha256']),scored=results['families'][family].get('images') if 'scores' in results['families'][family] else 0)
    supplement=U.read(ROOT/'cryptbb_supplement/results.json')
    assert supplement['models_frozen_sha256']==U.sha(ROOT/'solvers/frozen_models.json')
    assert supplement['predictions_sha256']==U.sha(ROOT/'cryptbb_supplement/predictions.json')
    U.dump(ROOT/'final_audit.json',dict(passed=True,equal_updates=2000,models=6,matched_initializations=True,synthetic_corpus_hashes_verified=True,
        checkpoint_selection_verified=True,frozen_predictions_verified=True,evaluation_counts=counts,
        supplementary_cryptbb_images=supplement['images'],supplementary_predictions_verified=True,
        sources={p.name:U.sha(p) for p in ROOT.glob('*.py')}))
    U.dump(ROOT/'completion.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),primary_transfer_success=False,
        report=str(ROOT/'report.html'),final_audit_sha256=U.sha(ROOT/'final_audit.json'),results_sha256=U.sha(ROOT/'evaluation/results.json')))
    print('Final experiment integrity checks passed')

if __name__=='__main__':main()

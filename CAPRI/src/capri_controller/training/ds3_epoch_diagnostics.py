"""Explicitly requested real-test sweep; never imported by the training runner."""
import csv
import importlib.util
import sys
import time
from pathlib import Path
from .ds3 import ROOT, DATA, DS1_RUN, SEEDS, U

OUT=ROOT/'all_epoch_diagnostics'


def available():
    return [(seed,epoch) for seed in SEEDS for epoch in range(1,21)
            if (ROOT/f'seed_{seed}/epoch_{epoch:02d}.receipt.json').exists()]


def evaluate(module,root,seed,epoch):
    folder=root/f'interim_evaluation/seed_{seed}_epoch_{epoch}'
    checkpoint=root/f'seed_{seed}/epoch_{epoch}.pth'
    if (folder/'summary.json').exists():
        assert U.sha(checkpoint)==U.read(folder/'protocol.json')['sha256']
        assert U.sha(folder/'predictions.json')==U.read(folder/'prediction_freeze.json')['sha256']
        return U.read(folder/'summary.json')
    module.ROOT=root
    sys.argv=['evaluate_ds2_checkpoint.py','--seed',str(seed),'--epoch',str(epoch)]
    module.main()
    return U.read(folder/'summary.json')


def write_report(rows):
    U.dump(OUT/'summary.json',dict(rows=rows,total=900,
        note='Requested historical real-test sweep; no real-based checkpoint selection or curriculum changes. Model1 epoch1 was not saved. Models2/3 use identical data and non-curriculum settings.'))
    fields=list(rows[0])
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# Model 3 all saved epoch diagnostics','',
        'Correct counts are out of the same900 historical DarkBay images. These requested real tests do not change the synthetic checkpoint-selection rule or curriculum. Each prediction file was frozen before scoring. Model1 did not retain epoch1.', '']
    for seed in SEEDS:
        subset=[r for r in rows if r['seed']==seed]
        if not subset:continue
        lines += [f'## Seed {seed}','','| Epoch | Model1 | Model2 | Model3 | Model3 accuracy | Delta vs Model2 | Synthetic validation |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in subset:
            lines.append(f"| {r['epoch']} | {r['model1_correct'] if r['model1_correct'] is not None else 'unavailable'} | {r['model2_correct']} | {r['model3_correct']} | {r['model3_accuracy']:.2f}% | {r['delta_vs_model2']:+d} | {r['synthetic_validation_accuracy']:.2f}% |")
        lines.append('')
    lines += ['Only completed saved checkpoints are included. Unfinished seeds/epochs and three-seed ensemble results remain pending.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    OUT.mkdir(exist_ok=True)
    lock=OUT/'running.lock'
    with lock.open('x') as f:f.write(str(__import__('os').getpid()))
    try:
        spec=importlib.util.spec_from_file_location('capri_interim_evaluator',U.WORK/'tools/evaluate_ds2_checkpoint.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old={r['epoch']:r for r in U.read(DS1_RUN/'epoch_evaluation_darkbay/summary.json')['epochs']}
        protocol=dict(request='Test all epochs',started=time.strftime('%Y-%m-%d %H:%M:%S'),
            initial_checkpoints=available(),images=900,model3_root=str(ROOT),model2_root=str(DATA),
            evaluator_sha256=U.sha(U.WORK/'tools/evaluate_ds2_checkpoint.py'),
            code_sha256=U.sha(Path(__file__)),training_unchanged=True,
            rule='Test all committed checkpoints available during this sweep; reuse hash-verified prior tests; do not wait for untrained epochs. No threshold sweeps or checkpoint selection on these scores.')
        if not (OUT/'protocol.json').exists():U.dump(OUT/'protocol.json',protocol)
        done=set();rows=[]
        while True:
            pending=[pair for pair in available() if pair not in done]
            if not pending:break
            for seed,epoch in pending:
                U.dump(OUT/'status.json',dict(stage='testing',seed=seed,epoch=epoch,completed=len(done)))
                receipt=U.read(ROOT/f'seed_{seed}/epoch_{epoch:02d}.receipt.json')
                assert U.sha(ROOT/f'seed_{seed}/epoch_{epoch}.pth')==receipt['checkpoint_sha256']
                result=evaluate(module,ROOT,seed,epoch)
                baseline=evaluate(module,DATA,seed,epoch)
                ds1=old.get(epoch,{}).get(str(seed))
                rows.append(dict(seed=seed,epoch=epoch,model1_correct=ds1,model2_correct=baseline['correct'],
                    model3_correct=result['correct'],model3_accuracy=result['accuracy'],model3_CER=result['CER'],
                    delta_vs_model2=result['correct']-baseline['correct'],
                    delta_vs_model1=result['correct']-ds1 if ds1 is not None else None,
                    synthetic_validation_accuracy=100*result['synthetic_validation']['acc']))
                done.add((seed,epoch));rows.sort(key=lambda r:(SEEDS.index(r['seed']),r['epoch']))
                write_report(rows)
                print('COMPARED',seed,epoch,result['correct'],baseline['correct'],flush=True)
        U.dump(OUT/'status.json',dict(stage='complete_available',completed=len(done),available=available(),
            pending_untrained=60-len(done),finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    finally:
        lock.unlink()


if __name__=='__main__':main()

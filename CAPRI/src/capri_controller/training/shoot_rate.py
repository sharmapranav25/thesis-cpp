"""Reproduce historical confidence-threshold selective prediction metrics."""
import copy
import gc
import sys
from pathlib import Path
from .. import runtime as U
from .prepare_parseq import RUN, OPENOCR


def main():
    out = RUN / 'shoot_rate_analysis'
    if out.exists():
        raise FileExistsError(str(out))
    out.mkdir()
    sys.path[:0] = [str(U.WORK / '.runtime/parseq_packages'), str(OPENOCR)]
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    torch.set_num_threads(8)
    source = RUN / 'evaluation_darkbay'
    manifest = U.read(source / 'image_manifest.json')
    frozen = U.read(source / 'prediction_freeze.json')
    thresholds = [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]
    U.dump(out / 'protocol.json', dict(thresholds=thresholds,
        reference='E:/pranav/model 1/src/eval/selective_with_ecps.py',
        confidence='Product of emitted character maximum probabilities, excluding EOS; no forced length',
        ensemble_confidence='Sum of joint confidences from seeds supporting the frozen winning string, divided by three. Disagreeing seeds contribute zero. This is a ranking score, not a calibrated probability.',
        metrics='shoot_rate=attempts/900; accuracy=correct/attempts; ECPS=900/correct; undefined values saved as null',
        limitation='Descriptive sweep on historical test data, not independent threshold calibration; no threshold chosen for deployment',
        code_sha256=U.sha(Path(__file__))))
    transform = T.Compose([T.Resize((32,128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5,.5)])
    scores = {}
    for run in U.read(RUN / 'completion.json')['seeds']:
        seed = str(run['seed'])
        assert U.sha(run['checkpoint']) == run['sha256']
        predpath = source / frozen[seed]['file']
        assert U.sha(predpath) == frozen[seed]['sha256']
        old = {r['opaque_id']: r for r in U.read(predpath)}
        checkpoint = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
        cfg = copy.deepcopy(checkpoint['config'])
        decoder = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = decoder.get_character_num()
        model = build_model(cfg['Architecture'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.cuda().eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(manifest), 64):
                chunk = manifest[start:start+64]
                x = torch.stack([transform(Image.open(r['image_path']).convert('RGB')) for r in chunk]).cuda()
                probs = model(x)
                if isinstance(probs, list):
                    probs = probs[-1]
                decoded = decoder(probs)
                probs = probs.cpu().numpy()
                assert np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-5)
                for record, p, (text, mean) in zip(chunk, probs, decoded):
                    oid = record['opaque_id']
                    assert text == old[oid]['prediction'], 'Prediction replay changed'
                    cp = []
                    for index, confidence in zip(p.argmax(-1), p.max(-1)):
                        char = decoder.character[int(index)]
                        if char == decoder.EOS:
                            break
                        if char not in (decoder.BOS, decoder.PAD):
                            cp.append(float(confidence))
                    assert len(cp) == len(text)
                    rows.append(dict(opaque_id=oid, prediction=text, confidence=float(np.prod(cp)) if cp else 0.0))
        scores[seed] = rows
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print('Recovered joint confidence, seed', seed, flush=True)
    ensemble_dir = RUN / 'prediction_ensemble'
    assert U.sha(ensemble_dir / 'predictions.json') == U.read(ensemble_dir / 'prediction_freeze.json')['sha256']
    lookup = {s: {r['opaque_id']: r for r in rows} for s, rows in scores.items()}
    ensemble = []
    for row in U.read(ensemble_dir / 'predictions.json'):
        oid, text = row['opaque_id'], row['prediction']
        confidence = sum(p[oid]['confidence'] for p in lookup.values() if p[oid]['prediction'] == text) / 3
        ensemble.append(dict(opaque_id=oid, prediction=text, confidence=confidence))
    scores['ensemble'] = ensemble
    U.dump(out / 'confidence_predictions.json', scores)
    U.dump(out / 'prediction_freeze.json', dict(sha256=U.sha(out / 'confidence_predictions.json')))
    labels = U.read(source / 'scoring_labels.json')
    results = {}
    for name, rows in scores.items():
        assert len(rows) == 900 and {r['opaque_id'] for r in rows} == set(labels)
        results[name] = []
        for threshold in thresholds:
            attempted = [r for r in rows if r['confidence'] >= threshold]
            correct = sum(r['prediction'] == labels[r['opaque_id']] for r in attempted)
            results[name].append(dict(threshold=threshold, attempts=len(attempted), correct=correct,
                shoot_rate_percent=100*len(attempted)/900,
                accuracy_on_attempts_percent=100*correct/len(attempted) if attempted else None,
                ecps=900/correct if correct else None))
    assert results['ensemble'][-1]['correct'] == 496
    U.dump(out / 'summary.json', results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    for name, rows in results.items():
        ax.plot([r['shoot_rate_percent'] for r in rows if r['attempts']],
                [r['accuracy_on_attempts_percent'] for r in rows if r['attempts']], 'o-', label=name)
    ax.set(xlabel='Shoot rate (% of 900 images attempted)', ylabel='Accuracy on attempted images (%)', title='DarkBay: confidence-threshold analysis', xlim=(0,105), ylim=(0,105))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'shoot_rate.png', dpi=160)
    print(results['ensemble'], flush=True)


if __name__ == '__main__':
    main()

"""Equal-weight checkpoint average and fixed DarkBay evaluation; no refitting."""
import collections
import copy
import sys
import time
from pathlib import Path
from .. import runtime as U
from .prepare_parseq import RUN, OPENOCR


def main():
    import torch
    from PIL import Image
    from torchvision import transforms as T
    out = RUN / 'equal_weight_average'
    if out.exists():
        raise FileExistsError('Preserve existing experiment: ' + str(out))
    out.mkdir()
    sys.path[:0] = [str(U.WORK / '.runtime/parseq_packages'), str(OPENOCR)]
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    torch.set_num_threads(8)
    sources = U.read(RUN / 'completion.json')['seeds']
    assert [r['seed'] for r in sources] == [48, 123, 456]
    checkpoints = []
    for row in sources:
        assert U.sha(row['checkpoint']) == row['sha256']
        checkpoints.append(torch.load(row['checkpoint'], map_location='cpu', weights_only=False))
    states = [c['state_dict'] for c in checkpoints]
    cfg = copy.deepcopy(checkpoints[0]['config'])
    for c, state in zip(checkpoints, states):
        assert c['config']['Architecture'] == cfg['Architecture']
        assert c['config']['PostProcess'] == cfg['PostProcess']
        assert set(state) == set(states[0])
    averaged = collections.OrderedDict()
    integer_keys = []
    for key, first in states[0].items():
        values = [s[key] for s in states]
        assert all(v.shape == first.shape and v.dtype == first.dtype for v in values)
        if first.is_floating_point():
            assert all(torch.isfinite(v).all() for v in values)
            averaged[key] = torch.stack([v.double() for v in values]).mean(0).to(first.dtype)
            assert torch.isfinite(averaged[key]).all()
        else:
            assert all(torch.equal(first, v) for v in values), 'Unequal nonfloating buffer: ' + key
            averaged[key] = first.clone()
            integer_keys.append(key)
    path = out / 'averaged.pth'
    torch.save({'state_dict': averaged, 'config': cfg, 'averaging': 'equal 1/3 each; float64 accumulation', 'sources': sources}, path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert all(torch.equal(saved['state_dict'][k], v) for k, v in averaged.items())
    prior = RUN / 'evaluation_darkbay'
    manifest_path = prior / 'image_manifest.json'
    records = U.read(manifest_path)
    assert len(records) == 900
    U.dump(out / 'protocol.json', dict(sources=sources, checkpoint_sha256=U.sha(path),
        method='Arithmetic mean of all floating weights and buffers; equal nonfloating buffers preserved',
        nonfloating_keys=integer_keys, state_tensors=len(averaged),
        image_manifest=str(manifest_path), image_manifest_sha256=U.sha(manifest_path),
        evaluation='Same 900 images, same preprocessing and decoder as individual seeds; no recalibration or fine-tuning',
        code_sha256=U.sha(Path(__file__)), limitation='Historical DarkBay dataset; follow-up experiment after individual seed scores were known'))
    decoder = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = decoder.get_character_num()
    model = build_model(cfg['Architecture'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to('cuda').eval()
    transform = T.Compose([T.Resize((32,128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5,.5)])
    rows = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(records), 64):
            chunk = records[start:start+64]
            x = torch.stack([transform(Image.open(r['image_path']).convert('RGB')) for r in chunk]).cuda()
            outputs = decoder(model(x))
            assert len(outputs) == len(chunk)
            for record, (text, confidence) in zip(chunk, outputs):
                rows.append(dict(opaque_id=record['opaque_id'], prediction=text, confidence=float(confidence)))
    predictions_path = out / 'predictions.json'
    U.dump(predictions_path, rows)
    frozen = dict(sha256=U.sha(predictions_path), count=len(rows), seconds=time.time()-started)
    U.dump(out / 'prediction_freeze.json', frozen)
    # Only now load scoring labels; no label-based model modifications.
    from rapidfuzz.distance import Levenshtein
    labels = U.read(prior / 'scoring_labels.json')
    assert set(labels) == {r['opaque_id'] for r in rows}
    correct = sum(r['prediction'] == labels[r['opaque_id']] for r in rows)
    distance = sum(Levenshtein.distance(labels[r['opaque_id']], r['prediction']) for r in rows)
    summary = dict(correct=correct, total=len(rows), exact_match_percent=100*correct/len(rows),
        character_error_rate_percent=100*distance/sum(map(len, labels.values())),
        predicted_lengths=dict(collections.Counter(len(r['prediction']) for r in rows)),
        individual_seed_results=U.read(prior / 'summary.json')['results'],
        checkpoint=str(path), checkpoint_sha256=U.sha(path))
    assert U.sha(predictions_path) == frozen['sha256']
    U.dump(out / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()

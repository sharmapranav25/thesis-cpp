"""Diagnose exhausted deterministic retries without changing rendering or quotas."""
import collections
import json
import time
from pathlib import Path
from .common import read, write, now, sha
from .sampling import Sampler
from .renderer import render


def retry_pair(task):
    """Continue the frozen sampler's attempt stream; never relax acceptance."""
    from .pilot import pair
    start = task[2] if len(task) > 2 else 0
    prior = []
    for offset in range(0, 400, 80):
        results = pair((*task[:2], start+offset))
        if not results[0].get('failed'):
            for result in results:
                result['rejections'] = prior + result['rejections']
            return results
        prior.extend(results[0]['rejections'])
    for result in results:
        result['rejections'] = prior.copy()
    return results


def verify_writer(run, protocol):
    current = sha(Path(__file__).with_name('production.py'))
    if current == protocol['writer_sha256']:
        return
    amendment = read(run/'recovery/retry_extension.json')
    if (not amendment['passed'] or amendment['original_writer_sha256'] != protocol['writer_sha256']
            or amendment['writer_sha256'] != current or amendment['recovery_sha256'] != sha(__file__)
            or amendment['protocol_sha256'] != sha(run/'corpus_protocol.json')):
        raise ValueError('Unverified writer recovery revision')


def check_recovery(run):
    """Verify old/new identical ordinary payloads and the exact exhausted quota."""
    import importlib.util
    from . import production
    from .pilot import initialize, data_source_hashes
    original = run/'recovery/production_before_retry_extension.py'
    protocol = read(run/'corpus_protocol.json')
    assert sha(original) == protocol['writer_sha256']
    assert data_source_hashes() == protocol['source_hashes']
    assert sha(run/'fonts/font_manifest.json') == protocol['font_manifest_sha256']
    spec = importlib.util.spec_from_file_location('capri_controller.decoder.original_writer', original)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    initialize(read(run/'fonts/font_manifest.json'))
    for task in [('train', 0), ('train', 1), ('train', 2), ('train', 3)]:
        before = old.encoded_pair(task); after = production.encoded_pair(task)
        assert before == after, 'Ordinary accepted payload changed'
    try:
        old.encoded_pair(('train', 106013))
    except RuntimeError as exc:
        assert 'Quota could not be rendered' in str(exc)
    else:
        raise AssertionError('Known exhausted quota did not reproduce')
    fixed = production.encoded_pair(('train', 106013))
    assert len(fixed) == 2 and all(r['attempt'] == 110 for r in fixed)
    assert len({r['pixel_sha256'] for r in fixed}) == 2
    for row in fixed:
        label = json.loads(next(v for k,v in row['payloads'].items() if k.endswith('.json')))
        assert label['valid'] and label['count'] == 7 and label['recipe']['quota_family'] == 'jsmathcmex10'
        assert len(row['rejections']) == 110
    result = dict(created_at=now(), passed=True, original_writer_sha256=sha(original),
                  writer_sha256=sha(Path(__file__).with_name('production.py')), recovery_sha256=sha(__file__),
                  protocol_sha256=sha(run/'corpus_protocol.json'), original_attempt_budget=80,
                  continued_attempt_budget=400, normal_pair_payloads_identical=4,
                  recovered_pair='train/106013', first_accepted_attempt=110,
                  unchanged_source_hashes=data_source_hashes(),
                  scope='Operational retry continuation only. Frozen renderer, sampler, labels, gates, fonts, count anchors and existing shards unchanged.')
    path = run/'recovery/retry_extension.json'
    if path.exists():
        path = run/'recovery'/('retry_extension_recheck_'+str(time.time_ns())+'.json')
    write(path, result)
    return result


def diagnose(run):
    failures = [json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
    failures = [e for e in failures if e.get('stage') == 'generate-corpus' and e.get('state') == 'failed'
                and 'Quota could not be rendered' in e.get('error', '')]
    if not failures:
        raise ValueError('No exhausted quota event to diagnose')
    error = failures[-1]['error']
    failed, _ = json.JSONDecoder().raw_decode(error[error.index('{"failed"'):])
    split, first = failed['split'], failed['index']
    if split != 'train':
        raise ValueError('This exploratory diagnostic is TRAIN-only')
    sampler = Sampler(read(run/'fonts/font_manifest.json'))
    rows = []; reasons = collections.Counter(); accepted = None
    for attempt in range(400):
        siblings = []
        for index in (first, first+1):
            recipe = sampler.recipe(split, index, attempt)
            out = render(recipe, sampler.fonts)
            bad = []
            for g in out['glyphs']:
                issues = []
                if not g['box_core'] or not g['box_visible']: issues.append('empty')
                if g['clipped_fraction'] > .05: issues.append('clipping')
                if g['visible_fraction'] < .25: issues.append('visibility')
                if g['touches_canvas_edge']: issues.append('edge')
                if g['visible_pixels'] < 10: issues.append('pixels')
                if not g['box_core'] or g['box_core'][3]-g['box_core'][1] < 7: issues.append('height')
                if g['local_contrast'] < 12: issues.append('contrast')
                if issues:
                    bad.append(dict(font=g['font_key'], char=g['char'], issues=issues))
                    reasons.update(issues)
            siblings.append(dict(index=index, valid=out['valid'], bad=bad, quality=out['quality'],
                                 fonts=recipe['text']['font_keys'], quota_family=recipe['quota_family'],
                                 count=recipe['requested_count']))
            if not out['valid']: break
        rows.append(dict(attempt=attempt, siblings=siblings))
        if len(siblings) == 2 and all(s['valid'] for s in siblings):
            accepted = attempt
            break
    report = dict(created_at=now(), failed_event=failures[-1]['updated_at'], split=split, first=first,
                  first_accepted_attempt=accepted, attempts=rows, rejection_reasons=dict(reasons),
                  unchanged_renderer_and_quality_gates=True)
    path = run/'recovery'/('diagnosis_'+str(time.time_ns())+'.json')
    write(path, report)
    return dict(path=str(path), first=first, first_accepted_attempt=accepted, rejection_reasons=dict(reasons))

"""Whole-string majority vote over frozen predictions, confidence tie-break."""
import collections
import math
import sys
from pathlib import Path
from .. import runtime as U
from .prepare_parseq import RUN


def main():
    source = RUN / 'evaluation_darkbay'
    out = RUN / 'prediction_ensemble'
    if out.exists():
        raise FileExistsError('Preserve existing experiment: ' + str(out))
    out.mkdir()
    seeds = [48, 123, 456]
    frozen = U.read(source / 'prediction_freeze.json')
    predictions = {}
    for seed in seeds:
        record = frozen[str(seed)]
        path = source / record['file']
        assert U.sha(path) == record['sha256']
        rows = U.read(path)
        predictions[seed] = {r['opaque_id']: r for r in rows}
        assert len(rows) == len(predictions[seed]) == 900
    ids = sorted(predictions[48])
    assert all(set(p) == set(ids) for p in predictions.values())
    U.dump(out / 'protocol.json', dict(
        rule='Whole-string majority vote; if all three differ, choose highest saved sequence confidence. Exact confidence ties use seed order 48,123,456.',
        seeds=seeds, sources=frozen, code_sha256=U.sha(Path(__file__)),
        evaluation='Same 900 historical DarkBay images; no label-based choice of voting rule, retraining, or new inference',
        limitation='Sequence confidence is not calibrated across models; this is a follow-up after previous test results were known.'))
    combined = []
    for opaque_id in ids:
        candidates = [predictions[s][opaque_id] for s in seeds]
        votes = collections.Counter(r['prediction'] for r in candidates)
        text, count = votes.most_common(1)[0]
        if count == 1:
            def confidence(r):
                value = r['confidence']
                return value if value is not None and math.isfinite(value) else float('-inf')
            text = max(candidates, key=confidence)['prediction']
        combined.append(dict(opaque_id=opaque_id, prediction=text, agreement=count,
                             method='confidence_tiebreak' if count == 1 else 'majority'))
    path = out / 'predictions.json'
    U.dump(path, combined)
    digest = U.sha(path)
    U.dump(out / 'prediction_freeze.json', dict(sha256=digest, count=len(combined)))
    # Ground truth is used only after the ensemble outputs are frozen.
    sys.path.insert(0, str(U.WORK / '.runtime/parseq_packages'))
    from rapidfuzz.distance import Levenshtein
    labels = U.read(source / 'scoring_labels.json')
    assert set(labels) == set(ids)
    correct = sum(r['prediction'] == labels[r['opaque_id']] for r in combined)
    distance = sum(Levenshtein.distance(labels[r['opaque_id']], r['prediction']) for r in combined)
    summary = dict(correct=correct, total=len(combined), exact_match_percent=100*correct/len(combined),
        character_error_rate_percent=100*distance/sum(map(len, labels.values())),
        agreement_counts=dict(collections.Counter(r['agreement'] for r in combined)),
        individual_seed_results=U.read(source / 'summary.json')['results'])
    assert U.sha(path) == digest
    U.dump(out / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()

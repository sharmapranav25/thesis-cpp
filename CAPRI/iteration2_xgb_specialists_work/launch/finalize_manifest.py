"""Record the completed exported model and user-requested four-run comparison."""
from datetime import datetime
from pathlib import Path
from fit_xgb import PROJECT, dump, read, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'run_20261004_02'
BUNDLE = RUN/'bundle'

checks = read(RUN/'verification.json')
assert checks['passed'] and read(RUN/'replacement_checks.json')['passed']
assert read(RUN/'status.json')['stage'] == 'complete'
for name in ('__init__.py', 'feature_encoder.py', 'fit_xgb.py', 'predict.py'):
    expected = sha(ROOT/'launch'/name)
    assert sha(BUNDLE/'runtime'/name) == expected
    assert sha(PROJECT/'feature_bank/iter2/xgb_replacements'/name) == expected
assert sha(ROOT/'README.md') == sha(PROJECT/'docs/ITERATION2_XGB_REPLACEMENTS.md')
assert sha(RUN/'FOUR_MODEL_COMPARISON.md') == sha(PROJECT/'docs/ITERATION2_FOUR_MODEL_COMPARISON.md')
checksums = read(BUNDLE/'checksums.json')
checksums['files'] = {str(path.relative_to(BUNDLE)).replace('\\', '/'):dict(bytes=path.stat().st_size,sha256=sha(path))
                      for path in BUNDLE.rglob('*') if path.is_file() and path.name != 'checksums.json'}
dump(BUNDLE/'checksums.json', checksums)
artifacts = ('REPORT.md','FOUR_MODEL_COMPARISON.md','four_model_comparison.json','four_model_per_head.csv',
             'per_head_comparison.csv','verification.json','replacement_checks.json','export_checks.json',
             'synthetic_results.json','bundle/checksums.json')
dump(RUN/'completion_manifest.json',dict(completed=datetime.now().isoformat(), status='complete',
    request='107 XGBoost feature replacements with neural classifier removal, followed by four-model comparison',
    heads=107, encoder_parameters=2179108, selected_rounds_median=323, selected_rounds_max=1578,
    source_checkpoint_sha256=checksums['source_checkpoint_sha256'], real_predictions=3599, previews=300,
    runtime_versions=dict(torch=checksums['torch'],xgboost=checksums['xgboost']),
    model_bundle=str(BUNDLE), artifacts={name:sha(RUN/name) for name in artifacts},
    inference_module='feature_bank.iter2.xgb_replacements.predict',
    four_way_label_alignment=read(RUN/'four_model_comparison.json')['provenance']['label_alignment'],
    encoder_training_changed=False, solver_training_started=False))
print('Completed bundle and four-model comparison; deployment files and documentation hashes match.')

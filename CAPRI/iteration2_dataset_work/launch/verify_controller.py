import sys,json,tempfile
from pathlib import Path
root=Path('C:/Users/pranav/Documents/ChatGPT/thesis')
sys.path.insert(0,'E:/pranav/Final sem/sourcefree_synth')
from feature_bank.iter2.controller import build_profiles
full=json.loads((root/'iter2_full_target_recreation_01/report.json').read_text())
for family,expected in full['summary'].items():
    p=json.loads((root/'iter2_controller_profiles_01'/f'{family}.json').read_text())
    assert sum(v['count'] for v in p['joint_distribution'])==expected['images']
    assert p['summaries']['length_distribution']==expected['predicted_lengths']
    for head,count in expected['effect_positive_counts'].items():
        assert abs(p['summaries']['effects'][head]['prevalence']-count/expected['images'])<1e-9
    for entry in p['joint_distribution']:
        assert set(entry['description'])=={'size','length','font','content','effects','text_features'}
with tempfile.TemporaryDirectory() as tmp:
    bad=Path(tmp)/'incomplete.json';bad.write_text(json.dumps(dict(total=1,counts={'City':1})))
    try:build_profiles(root/'iter2_full_target_recreation_01/predictions.jsonl',bad,Path(tmp)/'should_not_exist')
    except ValueError:pass
    else:raise AssertionError('Incomplete inference was accepted')
    assert not (Path(tmp)/'should_not_exist').exists()
print('Verified all family marginals against full-set inference; joint counts; profile excludes source text/recipes; incomplete inference rejected before output creation.')

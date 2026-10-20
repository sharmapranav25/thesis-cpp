"""Regression checks for pruned features, label masks and two-class categories."""
from pathlib import Path
import argparse
import json
import numpy as np
import xgboost as xgb
from fit_xgb import decode_font, target, predict_task, tasks, load_encoder, REMOVED, FONT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parents[1]/'run_20261004_02')
    out = parser.parse_args().out
    encoder, _ = load_encoder(out/'bundle/encoder.pt')
    assert not any(hasattr(encoder, name) for name in REMOVED)
    assert not any(name.split('.')[0] in REMOVED for name in encoder.state_dict())
    definitions = tasks()
    assert len(definitions) == 107
    assert sum(task['group'] == 'eff' for task in definitions) == 74
    assert sum(task['group'] == 'txt' for task in definitions) == 25
    labels = dict(font=np.array([[0, -1, 0, 0, 0, 0, 0], [1, 0, -1, -1, -1, -1, 1]]),
                  eff_level=np.array([[0], [1]]))
    _, mask = target(labels, dict(group='font', index=2))
    assert mask.tolist() == [True, False]
    _, mask = target(labels, dict(group='eff', index=0, kind='regression'))
    assert mask.tolist() == [False, True]
    font_values = {'font_'+name: np.zeros(4) for name in FONT_NAMES}
    font_values['font_italic'] = np.array([.1, .4999, .5, .9])
    assert [decode_font(font_values, i)['italic'] for i in range(4)] == [False, False, True, True]
    rng = np.random.default_rng(42)
    values = rng.normal(size=(64, 4)).astype(np.float32)
    y = (values[:, 0] > 0).astype(int)
    model = xgb.train(dict(objective='binary:logistic', tree_method='hist', device='cpu',
                           max_depth=2, nthread=2), xgb.DMatrix(values, label=y), num_boost_round=12)
    task = dict(kind='multiclass', classes=[4, 9])
    expected = np.asarray(task['classes'])[(model.inplace_predict(values) >= .5).astype(int)]
    actual = predict_task(model, task, values)
    assert np.array_equal(expected, actual) and actual.shape == (64,)
    path = out/'two_class_check.json'
    model.save_model(path)
    loaded = xgb.Booster()
    loaded.load_model(path)
    assert np.array_equal(predict_task(loaded, task, values), actual)
    report = dict(passed=True, task_counts=dict(effects=74, text=25, font=7, length=1),
                  classifier_modules_absent=True, font_zero_valid=True, unknown_masked=True,
                  effect_ordinal_absence_masked=True, two_class_category_mapping=True,
                  saved_booster_replay_exact=True, italic_probability_threshold_correct=True)
    (out/'replacement_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

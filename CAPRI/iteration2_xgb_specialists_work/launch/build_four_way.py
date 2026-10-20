"""Compare all four saved runs on identical synthetic benchmark labels."""
import copy
import csv
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from fit_xgb import BASE, M, SPECIALISTS, SPLITS, read, sha, dump
from build_report import metric

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'run_20261004_02'
LEGACY = ROOT.parent/'iter2_xgb_01'
MODELS = ('simple', 'simple_xgboost', 'specialist', 'specialist_xgboost')
HEADERS = ('Simple', 'Simple + XGB', 'Specialist', 'Specialist + XGB')
FONT = [('Category accuracy', 'category'), ('Stroke accuracy', 'stroke'),
        ('Weight within ±1', 'weight_pm1'), ('Width within ±1', 'width_pm1'),
        ('X-height within ±1', 'x_height_pm1'), ('Slant within ±1', 'slant_pm1'), ('Italic F1', 'italic_f1')]


def main():
    original = read(BASE/'baseline_training/final_report.json')
    legacy = read(LEGACY/'synthetic_results.json')
    assert read(LEGACY/'protocol.json')['checkpoint_sha256'] == read(SPECIALISTS/'configuration.json')['baseline_sha256']
    assert read(LEGACY/'feature_manifest.json')['counts'] == {s:read(RUN/'features'/s/'manifest.json')['count'] for s in SPLITS}
    label_alignment = {}
    for split in SPLITS:
        a, b = dict(np.load(LEGACY/'features'/split/'labels.npz')), dict(np.load(RUN/'features'/split/'labels.npz'))
        assert a.keys() == b.keys() and all(np.array_equal(a[key], b[key]) for key in a)
        label_alignment[split] = len(a['length'])
    legacy_boosted = {s:copy.deepcopy(block['xgboost']) for s,block in legacy.items()}
    legacy_tasks = read(LEGACY/'heads.json')['tasks']
    new_tasks = read(RUN/'bundle/heads.json')['tasks']
    assert {task['name'] for task in legacy_tasks} == {task['name'] for task in new_tasks}
    # Old fitting calibrated italic too; all four font tables now use .5.
    task = next(task for task in legacy_tasks if task['name'] == 'font_italic')
    booster = xgb.Booster()
    booster.load_model(LEGACY/'boosters'/task['model_file'])
    booster.set_param(dict(device='cpu', nthread=4))
    italic = {}
    for split in legacy_boosted:
        features = np.load(LEGACY/'features'/split/'T.npy', mmap_mode='r')
        values = np.asarray(features[:,task['slot']], dtype=np.float32, order='C')
        probability = booster.inplace_predict(values, iteration_range=(0, task['best_iteration']+1))
        labels = np.load(LEGACY/'features'/split/'labels.npz')['font'][:,6]
        valid = labels >= 0
        calibrated = M.f1(probability[valid] >= task['threshold'], labels[valid] == 1)
        assert abs(calibrated-legacy_boosted[split]['font']['italic_f1']) <= .000051
        matched = M.f1(probability[valid] >= .5, labels[valid] == 1)
        italic[split] = dict(historical_calibrated_f1=calibrated, threshold=task['threshold'], matched_point5_f1=matched)
        legacy_boosted[split]['font']['italic_f1'] = matched
    neural_specialist = {s:read(SPECIALISTS/'final'/f'{s}.json') for s in ('seen_test','generalisation')}
    boosted_specialist = read(RUN/'synthetic_results.json')
    records, tables = [], {}
    for split in ('seen_test', 'generalisation'):
        data = dict(simple=original['tests'][split], simple_xgboost=legacy_boosted[split],
                    specialist=neural_specialist[split], specialist_xgboost=boosted_specialist[split])
        table = {label:{name:data[name]['font'][key] for name in MODELS} for label,key in FONT}
        table['Font ordinal mean within ±1'] = {name:data[name]['summary']['font_ord_pm1'] for name in MODELS}
        for block, label in [('effects','Effect presence F1 (calibrated)'),('text_heads','Text presence F1 (calibrated)')]:
            table[label] = dict(
                simple=float(np.mean([v['calibrated_f1'] for v in legacy[split]['neural'][block].values()])),
                simple_xgboost=float(np.mean([v['f1'] for v in data['simple_xgboost'][block].values()])),
                specialist=float(np.mean([v['f1'] for v in data['specialist'][block].values()])),
                specialist_xgboost=float(np.mean([v['calibrated_f1'] for v in data['specialist_xgboost'][block].values()])))
        for block,label in [('effects','Effect intensity within ±1'),('text_heads','Text intensity within ±1')]:
            table[label] = {name:float(np.mean([v['lvl_pm1'] for v in data[name][block].values() if 'lvl_pm1' in v])) for name in MODELS}
        table['Length accuracy'] = {name:data[name]['reading']['length_head'] for name in MODELS}
        tables[split] = table
        for task in new_tasks:
            presence = task['group'] in ('eff','txt') and task['kind'] == 'binary'
            results = {name:metric(data[name],task,calibrated=(presence and name=='specialist_xgboost')) for name in MODELS}
            if presence:
                results['simple'] = metric(legacy[split]['neural'],task,calibrated=True)
            records.append(dict(split=split,head=task['name'],**results))
    sources = [BASE/'baseline_training/final_report.json', LEGACY/'synthetic_results.json',
               SPECIALISTS/'final/seen_test.json', SPECIALISTS/'final/generalisation.json', RUN/'synthetic_results.json']
    provenance = dict(label_alignment=label_alignment,source_files={str(path):sha(path) for path in sources},
        legacy_italic_matched_threshold=italic,
        presence='All four calibrated effect/text presence: max F1 with FPR <= .10.',
        fonts='Italic .5 throughout; categories exact, four ordinals within ±1.',
        comparison='Saved runs; different CNN routes and tree fitting protocols, not a backbone-only ablation.')
    dump(RUN/'four_model_comparison.json',dict(tables=tables,per_head=records,provenance=provenance))
    with (RUN/'four_model_per_head.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines = ['# All four model runs', '',
        'Specialist neural has the strongest mean font ordinal result on both benchmarks. Specialist + XGBoost leads font category on both and improves effect intensity, but loses weight/slant/italic quality relative to specialist neural. Simple + XGBoost is severely weaker on slant and effect detection.', '',
        'All models use the original 100,000-image synthetic corpus. Seen-test has 2,000 images; generalisation has 2,500 with held-out fonts. Cached label arrays match exactly across the two XGBoost experiments on every split. No new fitting or threshold tuning was done for this comparison.', '']
    for split,title in [('seen_test','Seen-test'),('generalisation','Generalisation: held-out fonts')]:
        lines += [f'## {title}', '', '| Metric | '+' | '.join(HEADERS)+' |', '|---|---:|---:|---:|---:|']
        for label,values in tables[split].items():
            maximum=max(values.values())
            cells=[f'{100*values[name]:.2f}%' for name in MODELS]
            cells=[f'**{cell}**' if abs(values[name]-maximum)<.000001 else cell for name,cell in zip(MODELS,cells)]
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines += ['']
    lines += ['Within ±1 permits one ordinal-bin error. Font ordinal mean averages weight, width, x-height and slant; it excludes category, stroke and italic. No all-font-attributes-exact metric was saved. F1 percentages are averages across feature heads, not ordinary classification accuracy.', '',
        '## Comparison protocol', '',
        'Effect/text presence uses the same synthetic calibration rule for all four: maximum F1 subject to FPR <= .10. The older Simple + XGBoost summary already reports calibrated decisions; comparing it with raw .5 neural F1 would overstate its gain. This table instead reads calibrated neural reports.', '',
        'Font italic uses .5 for all four. The older Simple + XGBoost fit also calibrated italic; this comparison replays only its saved italic booster on cached vectors at .5, using selected rounds. Its original reports/model are preserved; historical and matched scores are in four_model_comparison.json.', '',
        'Simple font/length/ordinal scores come from its official saved final report. The prior XGBoost experiment contains slightly different simple neural arithmetic from another batching path; tiny differences are not evidence of meaningful improvement. No confidence intervals or new independent benchmark are available.', '',
        '## Differences between XGBoost runs', '',
        'Simple + XGBoost used frozen simple features: CNN stem mean/max over the image and four vertical bands (960 dimensions) for effects, and transformer task queries for text/font/length. Specialist + XGBoost uses each trained CNN effect query (128 dimensions) and specialist transformer queries. The newer export physically removes neural feature classifiers and executes no CTC reader.', '',
        'Older fitting allowed 800 rounds, 30-round early stopping and balanced ordinal regressions. New fitting allows 2,000 rounds, 50-round early stopping and balanced/unweighted selection for ordinal regression. Depth, learning rate, sampling and regularisation match. This compares four actual runs; it cannot attribute every difference solely to specialist training.', '',
        'Old Simple + XGBoost CTC scores belong to the unchanged simple neural reader. New specialist XGBoost inference executes no reader; its optional sidecar preserves specialist reader weights. Boosting did not improve sequence reading.', '',
        'All 107 output scores on both benchmarks: four_model_per_head.csv. Sources and label alignment: four_model_comparison.json. Replacement bundle QA and preview sheets: REPORT.md.', '']
    (RUN/'FOUR_MODEL_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(RUN/'FOUR_MODEL_COMPARISON.md'),tables=tables,legacy_italic=italic,label_alignment=label_alignment)))


if __name__ == '__main__':
    main()

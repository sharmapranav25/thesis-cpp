"""Compare XGBoost replacements with the saved specialist and simple models."""
import csv
import json
from pathlib import Path
import numpy as np
from fit_xgb import BASE, SPECIALISTS, FONT_NAMES, read, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'run_20261004_02'


def metric(scores, task, calibrated=False):
    group, head, kind = task['group'], task['head'], task['kind']
    if group in ('eff', 'txt'):
        entry = scores['effects' if group == 'eff' else 'text_heads'][head]
        if kind == 'regression':
            return entry['lvl_pm1']
        return entry.get('calibrated_f1', entry['f1']) if calibrated else entry['f1']
    if group == 'length':
        return scores['reading']['length_head']
    return scores['font'][head + '_pm1' if kind == 'regression' else 'italic_f1' if kind == 'binary' else head]


def main():
    scores = read(RUN/'synthetic_results.json')
    heads = read(RUN/'bundle/heads.json')['tasks']
    assert len(heads) == 107
    selection = read(SPECIALISTS/'final/selection.json')
    neural = dict(selection=selection['selected']['scores']['selection'],
                  seen_test=read(SPECIALISTS/'final/seen_test.json'),
                  generalisation=read(SPECIALISTS/'final/generalisation.json'))
    baseline_report = read(BASE/'baseline_training/final_report.json')
    baseline = dict(selection=selection['baseline']['selection'], **baseline_report['tests'])
    baseline_calibrated_root = ROOT.parent/'iter2_xgb_01'
    baseline_calibrated = read(baseline_calibrated_root/'synthetic_results.json')
    assert read(baseline_calibrated_root/'protocol.json')['checkpoint_sha256'] == read(SPECIALISTS/'configuration.json')['baseline_sha256']
    rows = []
    for split in ('selection', 'seen_test', 'generalisation'):
        for task in heads:
            # Saved specialist benchmark per-head presence values use their
            # synthetic calibration. Selection is at the same .5 threshold.
            calibrated = split != 'selection' and task['group'] in ('eff', 'txt') and task['kind'] == 'binary'
            simple_score = metric(baseline_calibrated[split]['neural'], task, True) if calibrated else metric(baseline[split], task)
            neural_score = metric(neural[split], task)
            xgb_score = metric(scores[split], task, calibrated)
            rows.append(dict(split=split, head=task['name'], kind=task['kind'],
                presence_threshold_policy='equally calibrated' if calibrated else '.5' if task['kind'] == 'binary' else 'not applicable',
                simple=simple_score, specialist_neural=neural_score, xgboost=xgb_score,
                delta_vs_simple=xgb_score-simple_score, delta_vs_specialist=xgb_score-neural_score,
                labelled_train=task['labelled_train'], best_iteration=task.get('best_iteration'),
                model=task.get('model_file', 'constant')))
    with (RUN/'per_head_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    regression = [row for row in rows if row['delta_vs_specialist'] < 0]
    (RUN/'per_head_comparison.json').write_text(json.dumps(dict(rows=rows, regressions=regression), indent=2)+'\n')
    manifest = read(RUN/'bundle/encoder_manifest.json')
    bundle_bytes = sum(path.stat().st_size for path in (RUN/'bundle').rglob('*') if path.is_file())
    lines = ['# Specialist encoder with XGBoost replacements', '',
        'All 107 image-feature outputs now use XGBoost. The exported frozen encoder executes no neural effect/text/font/length classifier and no CTC reader. CTC is preserved only in an optional separate diagnostic sidecar.', '',
        f"Encoder: {manifest['original_neural_parameters']:,} → **{manifest['feature_encoder_parameters']:,}** neural parameters. Exact FP32 feature equality passed (CNN 0, transformer 0 max absolute difference).", '',
        f'Current bundle size including tree files and the optional reader: **{bundle_bytes/1024**2:.2f} MiB**. Removing neural parameters does not remove the storage/compute cost of the tree ensemble.', '',
        'Source checkpoint: specialist intensity merge epoch 4. The CNN/transformer stayed frozen; effects use their learned effect-branch queries, while text/font/length use their corresponding transformer queries. Synthetic train fits each booster; selection early-stops; calibration tunes effect/text presence thresholds. Font italic uses .5 across comparators. Seven binary text attributes have no ordinal targets and receive no invented intensity predictor.', '',
        'The first pass stopped before any benchmark evaluation after selection showed severe ordinal regressions (font slant .579 versus neural .917). The second pass compares two fixed regression weighting choices, balanced and unweighted, per ordinal task. Choice uses higher selection within-one, then lower selection MAE, with the first candidate winning exact ties. Completed unchanged classification fits and balanced candidates are reused with hash checks. All first-pass evidence is retained; no benchmark or real result chooses the weighting.', '',
        '## Overall comparison at the same .5 presence threshold', '',
        '| Split / metric | Simple | Specialist neural | XGBoost replacement |', '|---|---:|---:|---:|']
    for split in ('selection', 'seen_test', 'generalisation'):
        for name, key in [('Effect F1', 'eff_f1'), ('Text F1', 'txt_f1'), ('Effect intensity within ±1', 'eff_lvl_pm1'),
                          ('Length accuracy', 'length'), ('Font category', 'font_cat'), ('Font ordinal mean within ±1', 'font_ord_pm1')]:
            lines.append(f"| {split} / {name} | {baseline[split]['summary'][key]:.4f} | {neural[split]['summary'][key]:.4f} | {scores[split]['summary'][key]:.4f} |")
    lines += ['', '## All font attributes', '']
    for split in ('selection', 'seen_test', 'generalisation'):
        lines += [f'### {split}', '', '| Attribute / metric | Simple | Specialist neural | XGBoost | Change vs specialist (points) |',
                  '|---|---:|---:|---:|---:|']
        for label, key in [('Category accuracy', 'category'), ('Stroke accuracy', 'stroke'), ('Weight within ±1', 'weight_pm1'),
                           ('Width within ±1', 'width_pm1'), ('X-height within ±1', 'x_height_pm1'), ('Slant within ±1', 'slant_pm1'), ('Italic F1', 'italic_f1')]:
            a, b, c = baseline[split]['font'][key], neural[split]['font'][key], scores[split]['font'][key]
            lines.append(f'| {label} | {100*a:.2f}% | {100*b:.2f}% | {100*c:.2f}% | {100*(c-b):+.2f} |')
        lines += ['']
    lines += ['The within-one scores allow one ordinal-bin error; these are per-attribute metrics, not every-font-attribute-exact accuracy. Slant is heavily imbalanced and duplicates a text target. Stroke class 2 has no measured examples in the training corpus. These limitations remain.', '',
        '## Calibrated presence comparison on reused benchmarks', '',
        '| Benchmark / mean F1 | Simple | Specialist neural | XGBoost |', '|---|---:|---:|---:|']
    for split in ('seen_test', 'generalisation'):
        for block, label, key in [('effects', 'Effects', 'eff_calibrated_f1'), ('text_heads', 'Text', 'txt_calibrated_f1')]:
            a = np.mean([entry['calibrated_f1'] for entry in baseline_calibrated[split]['neural'][block].values()])
            b = np.mean([entry['f1'] for entry in neural[split][block].values()])
            c = scores[split]['summary'][key]
            lines.append(f'| {split} / {label} | {a:.4f} | {b:.4f} | {c:.4f} |')
    lines += ['', 'All three presence methods use the same calibration rule: maximum F1 subject to FPR <= .10. Neural comparison uses saved reports, without rerunning or retuning those checkpoints. The simple calibrated presence report was previously recorded by iter2_xgb_01 for the identical baseline checkpoint. Its arithmetic differs slightly from the baseline original report; it is labelled separately from the .5-threshold table.', '',
        'The seen_test/generalisation sets were already observed in prior experiments and are reused benchmarks. They were evaluated for this replacement after all heads were fitted and were never used for early stopping, configuration changes or per-head fallback. Every requested image-feature output is boosted even if weaker; all regressions remain in per_head_comparison.csv/json.', '']
    for split in ('selection', 'seen_test', 'generalisation'):
        subset = [row for row in rows if row['split'] == split]
        improved = sum(row['delta_vs_specialist'] > 0 for row in subset)
        regressed = sum(row['delta_vs_specialist'] < 0 for row in subset)
        simple_improved = sum(row['delta_vs_simple'] > 0 for row in subset)
        simple_regressed = sum(row['delta_vs_simple'] < 0 for row in subset)
        lines.append(f'- {split}: versus simple, {simple_improved}/107 improve and {simple_regressed}/107 regress; versus specialist neural, {improved}/107 improve and {regressed}/107 regress. Presence uses .5 on selection and equal calibration on benchmark per-head comparisons.')
    lines += ['', '## Fit completion and inference checks', '',
        'Two passes were fitting configurations, not two epochs. Each boosted model had a 2,000-round limit with 50-round early stopping. This was a limited comparison of fixed settings, not an exhaustive parameter search.', '',
        'Final inference verification caught an italic output-decoding error: converting a probability to an integer before bool conversion produced false. Both deployed and batch inference now use probability >= .5 through the same decoder. Fitted boosters, cached vectors and synthetic benchmark scores were unchanged. Initial inference/profiles/previews are retained in inference_initial_italic_decode; all 3,599 corrected predictions and all 300 previews were regenerated.', '']
    if (RUN/'verification.json').exists():
        checks = read(RUN/'verification.json')
        lines += [f"Verification passed: {checks['heads']} outputs, {checks['saved_boosters']} saved boosters, exact benchmark-score replay, source-checkpoint hashes, all feature-cache hashes, correct task routing and no neural prediction modules. All 300 preview pixel hashes, character counts and sampled prototypes passed. Selected boosting rounds: median {checks['selected_rounds_median']:g}, maximum {checks['selected_rounds_max']}; {checks['selected_models_at_round_limit']} models selected the 2,000-round limit.", '',
            'Both workspace and published project CLI ran successfully; a real italic-positive example decoded true. Checks and provenance: verification.json, replacement_checks.json and bundle/checksums.json. The bundle includes the exact runtime source snapshot.', '']
    if (RUN/'dataset/report.json').exists():
        dataset = read(RUN/'dataset/report.json')
        lines += ['', '## Target-family previews', '', '| Family | Generated | Strict readability | Mapping distance |', '|---|---:|---:|---:|']
        for family, value in dataset['audit'].items():
            lines.append(f"| {family} | {value['generated']} | {value['strict_training_readability_passed']}/100 | {value['mean_mapping_distance']:.4f} |")
        lines += ['', 'All 3,599 real images were inferred separately (CryptBB 600, City 2,000, DarkBay 999), using new boosted predictions. Only identity/source fields were reused; real answers and old predictions never fit a model. Renderer usability is separate from strict training readability and target resemblance. No CAPTCHA-solver training was launched.', '',
            'The three corrected sheets were inspected against the existing target examples. City captures five digits and horizontal bands but has broad colour/font variation and extra curves. DarkBay often has six characters and block backgrounds but varies fonts, case, colour and layout beyond the target. CryptBB now produces more five-character strings and heavier glyphs than the specialist neural previews, yet still spreads across grids, monochrome noise, hollow glyphs and short strings. These are qualitative observations, not measured real-family accuracy. Renderer palette/position defaults and prediction errors both contribute; greater readability alone does not establish closer family resemblance.', '',
            '![CryptBB previews](dataset/CryptBB_generated.jpg)', '', '![DarkBay previews](dataset/DarkBay_generated.jpg)', '', '![City previews](dataset/City_generated.jpg)', '']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(RUN/'REPORT.md'), bundle_mebibytes=round(bundle_bytes/1024**2, 2),
                         font=scores['seen_test']['font'], regressions=len(regression))))


if __name__ == '__main__':
    main()

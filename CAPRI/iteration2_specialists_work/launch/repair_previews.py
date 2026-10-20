"""Retry failed preview rows without changing their sampled family prototype."""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0, str(PROJECT))
from feature_bank.iter2.controller import generate_item


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--additional-attempts', type=int, default=24)
    args = parser.parse_args()
    if args.additional_attempts < 1:
        raise ValueError('additional-attempts must be positive')
    output = Path(args.dataset).resolve()
    report = read(output / 'report.json')
    plan = read(output / 'sampling_plan.json')
    original_labels = (output / 'labels.jsonl').read_text(encoding='utf-8')
    rows = [json.loads(line) for line in original_labels.splitlines()]
    profiles = {family: read(output / 'profiles' / f'{family}.json')
                for family in plan['prototype_ids']}
    start = time.perf_counter()
    repairs = []
    for name, value in [('report_before_retry.json', report)]:
        if not (output / name).exists():
            dump(output / name, value)
    if not (output / 'labels_before_retry.jsonl').exists():
        (output / 'labels_before_retry.jsonl').write_text(original_labels, encoding='utf-8')
    for position, row in enumerate(rows):
        if 'error' not in row:
            continue
        family, index, prototype_id = row['family'], row['index'], row['prototype_id']
        assert plan['prototype_ids'][family][index] == prototype_id
        description = profiles[family]['joint_distribution'][prototype_id]['description']
        family_seed = plan['seed'] + int.from_bytes(hashlib.sha256(family.encode()).digest()[:4], 'little')
        failures = list(row['failures'])
        initial_attempt = len(failures)
        for attempt in range(initial_attempt, initial_attempt + args.additional_attempts):
            print(json.dumps(dict(stage='retrying_preview', family=family, index=index,
                                  prototype_id=prototype_id, attempt=attempt)), flush=True)
            result = generate_item((family, index, prototype_id, description,
                                    family_seed + attempt * 9176, str(output), 1))
            if 'error' in result:
                failures.extend(result['failures'])
                continue
            result['attempt'] = attempt
            result['previous_failures'] = failures
            rows[position] = result
            repairs.append(dict(family=family, index=index, prototype_id=prototype_id,
                                attempt=attempt, seed=result['seed'], prototype_unchanged=True))
            break
        else:
            dump(output / 'retry_failure.json', dict(family=family, index=index,
                                                     prototype_id=prototype_id, failures=failures))
            raise RuntimeError(f'{family}/{index} still failed; original report remains incomplete')

    count = plan['count_per_family']
    assert len(rows) == count * len(profiles)
    assert Counter(row['family'] for row in rows) == {family: count for family in profiles}
    assert not any('error' in row for row in rows)
    audit = {}
    for family, profile in profiles.items():
        valid = sorted((row for row in rows if row['family'] == family), key=lambda row: row['index'])
        assert [row['index'] for row in valid] == list(range(count))
        assert len({row['pixel_sha256'] for row in valid}) == count
        for row in valid:
            assert row['renderer_usable'] and row['length'] == row['requested']['length']
            assert row['prototype_id'] == plan['prototype_ids'][family][row['index']]
            assert row['requested'] == profile['joint_distribution'][row['prototype_id']]['description']
            with Image.open(output / row['image']) as image:
                assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() == row['pixel_sha256']
        names = profile['summaries']['effects']
        audit[family] = dict(requested=count, generated=len(valid), failed=0, unique_pixels=count,
            sampled_lengths=dict(Counter(row['requested']['length'] for row in valid)),
            actual_lengths=dict(Counter(row['length'] for row in valid)),
            requested_effect_prevalence={name: sum(row['requested']['effects'][name]['present'] for row in valid)/count for name in names},
            actual_effect_prevalence={name: sum(row['effects'][name]['present'] for row in valid)/count for name in names},
            strict_training_readability_passed=sum(row['strict_training_readability'] for row in valid),
            mean_mapping_distance=float(np.mean([row['mapping_distance_after'] for row in valid])))
        with (output / family / 'labels.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['filename', 'label'])
            writer.writeheader()
            writer.writerows(dict(filename=str(Path(row['image']).relative_to(family)).replace('\\', '/'),
                                  label=row['text']) for row in valid)
        sheet = Image.new('RGB', (1000, 35 + math.ceil(count/5)*85), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'{family}: family-profile generated dataset ({count} images)', fill='black')
        for j, row in enumerate(valid):
            with Image.open(output / row['image']) as image:
                image.thumbnail((190, 60))
                x, y = (j % 5)*200, 35 + (j//5)*85
                sheet.paste(image, (x, y))
                draw.text((x, y+62), f'{row["index"]}: {row["text"]}', fill='black')
        sheet.save(output / f'{family}_generated.jpg', quality=93)
    (output / 'labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    completed = copy.deepcopy(report)
    completed.update(stage='complete', audit=audit, failures=[],
                     retry=dict(additional_attempt_budget=args.additional_attempts,
                                seconds=round(time.perf_counter()-start, 2), rows=repairs,
                                policy='Same sampled prototypes; only fresh render seeds, text and fonts are retried.'))
    dump(output / 'report.json', completed)
    dump(output / 'status.json', dict(stage='complete', completed=len(rows), total=len(rows)))
    dump(output / 'retry_report.json', completed['retry'])
    print(json.dumps(dict(stage='complete', generated=len(rows), repairs=repairs, audit=audit)), flush=True)


if __name__ == '__main__':
    main()

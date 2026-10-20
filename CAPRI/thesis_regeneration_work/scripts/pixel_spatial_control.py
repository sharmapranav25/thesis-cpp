"""Negative control: identical global color statistics despite destroyed layout."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from sourcefree_synth.data import load_image, write_json
from sourcefree_synth.pixel_statistics import measure, intersection

def main():
    run = ROOT / 'runs/external_pixels_outlined_01'
    split = json.loads((run / 'outlined_text_captcha/split.json').read_text(encoding='utf-8'))
    selected = sorted(split['validation'], key=lambda r: r['pixel_sha256'])[:4]
    out = run / 'spatial_negative_control'
    out.mkdir(exist_ok=False)
    board = Image.new('RGB', (800, 80+120*len(selected)), '#eeeeee')
    draw = ImageDraw.Draw(board)
    draw.text((10, 10), 'Negative control: matching global statistics does not establish CAPTCHA structure', fill='black')
    draw.text((10, 42), 'Original validation image', fill='black')
    draw.text((405, 42), 'Same pixels randomly rearranged', fill='black')
    records = []
    for i, record in enumerate(selected):
        original = load_image(record['path'])
        arr = np.asarray(original).copy()
        flat = arr.reshape(-1, 3)
        shuffled = Image.fromarray(np.random.default_rng(20260927+i).permutation(flat).reshape(arr.shape))
        a, ha = measure(original)
        b, hb = measure(shuffled)
        fields = ['rgb_mean', 'rgb_std', 'brightness_quantiles']
        error = max(float(np.max(np.abs(np.array(a[k])-b[k]))) for k in fields)
        overlap = intersection(ha, hb)
        records.append({'original_pixel_sha256': record['pixel_sha256'], 'seed': 20260927+i,
                        'histogram_intersection': overlap, 'histograms_exactly_equal': bool(np.array_equal(ha, hb)),
                        'max_absolute_moment_error': error, 'same_dimensions': original.size == shuffled.size,
                        'control_pass': bool(np.array_equal(ha, hb) and error < 1e-9)})
        original.save(out / f'{i:02d}_original.png')
        shuffled.save(out / f'{i:02d}_shuffled.png')
        board.paste(original.resize((360, 90)), (10, 70+i*120))
        board.paste(shuffled.resize((360, 90)), (405, 70+i*120))
        draw.text((10, 163+i*120), record['pixel_sha256'][:12], fill='black')
        draw.text((405, 163+i*120), f'Histogram intersection = {overlap:.6f}', fill='black')
    board.save(out / 'comparison.png')
    result = {'purpose': 'Test spatial blindness of Method 1, not CAPTCHA generation', 'records': records,
              'status': 'PASS' if all(r['control_pass'] for r in records) else 'FAIL',
              'conclusion': 'Global pixel summaries cannot establish preserved text or spatial arrangement.'}
    write_json(out / 'result.json', result)
    print(json.dumps(result))
    if result['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()

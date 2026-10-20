"""Method 1: native-pixel descriptions only; no text or spatial inference."""
from collections import Counter
import numpy as np

QUANTILES = [0, 5, 25, 50, 75, 95, 100]

def measure(image):
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    values = flat.astype(np.float64)
    bins = flat.astype(np.int32) // 32
    keys = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    counts = np.bincount(keys, minlength=512)
    histogram = counts.astype(np.float64) / len(flat)
    occupied = np.flatnonzero(counts)
    order = sorted(occupied, key=lambda i: (-int(counts[i]), int(i)))[:6]
    palette = [{'bin': int(i), 'rgb': values[keys == i].mean(0).tolist(),
                'fraction': float(histogram[i])} for i in order]
    brightness = values @ np.array([0.299, 0.587, 0.114])
    positive = histogram[histogram > 0]
    row = {'width': image.width, 'height': image.height, 'pixel_count': len(flat),
           'rgb_mean': values.mean(0).tolist(), 'rgb_std': values.std(0).tolist(),
           'brightness_mean': float(brightness.mean()), 'brightness_std': float(brightness.std()),
           'brightness_quantiles': np.percentile(brightness, QUANTILES).tolist(),
           'color_entropy_bits': float(-np.sum(positive * np.log2(positive))),
           'occupied_color_bins': len(occupied), 'palette': palette,
           'palette_covered_fraction': float(sum(p['fraction'] for p in palette)),
           'histogram_sum': float(histogram.sum())}
    return row, histogram

def aggregate(rows, histograms):
    if not rows:
        raise ValueError('Cannot summarize an empty set')
    dimensions = Counter((r['width'], r['height']) for r in rows)
    return {'image_count': len(rows), 'weighting': 'equal weight per unique image',
            'dimensions': [{'width': w, 'height': h, 'count': n, 'probability': n/len(rows)}
                           for (w, h), n in sorted(dimensions.items())],
            'mean_rgb_histogram': np.mean(histograms, axis=0).tolist(),
            'mean_image_rgb_mean': np.mean([r['rgb_mean'] for r in rows], axis=0).tolist(),
            'mean_image_rgb_std': np.mean([r['rgb_std'] for r in rows], axis=0).tolist(),
            'mean_image_brightness': float(np.mean([r['brightness_mean'] for r in rows])),
            'mean_image_brightness_std': float(np.mean([r['brightness_std'] for r in rows])),
            'mean_image_color_entropy_bits': float(np.mean([r['color_entropy_bits'] for r in rows])),
            'roles_inferred': False, 'alphabet': None, 'text_length': None,
            'classification': 'MEASURED whole-image statistics; no text/background assignment'}

def intersection(a, b):
    return float(np.minimum(a, b).sum())

def sample_canvases(profile, count, seed):
    """Sample observed JOINT width-height modes; no family size constants."""
    rng = np.random.default_rng(seed)
    modes = profile['dimensions']
    indices = rng.choice(len(modes), size=count, p=[m['probability'] for m in modes])
    return [(modes[i]['width'], modes[i]['height']) for i in indices]

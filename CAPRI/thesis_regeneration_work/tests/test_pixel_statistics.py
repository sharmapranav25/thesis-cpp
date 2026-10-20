"""Independent, known-value checks for thesis measurements."""
import unittest
import numpy as np
from PIL import Image
from sourcefree_synth.pixel_statistics import measure, aggregate, intersection, sample_canvases

class PixelStatisticsTests(unittest.TestCase):
    def test_solid(self):
        row, hist = measure(Image.new('RGB', (37, 19), (0, 0, 0)))
        self.assertEqual(row['pixel_count'], 703)
        self.assertEqual(row['rgb_mean'], [0, 0, 0])
        self.assertEqual(row['rgb_std'], [0, 0, 0])
        self.assertEqual(row['brightness_quantiles'], [0]*7)
        self.assertEqual(hist[0], 1)
        self.assertEqual(row['color_entropy_bits'], 0)

    def test_two_colors(self):
        arr = np.zeros((20, 40, 3), dtype=np.uint8)
        arr[:, 20:] = 255
        row, hist = measure(Image.fromarray(arr))
        np.testing.assert_allclose(row['rgb_mean'], [127.5]*3, atol=1e-12)
        np.testing.assert_allclose(row['rgb_std'], [127.5]*3, atol=1e-12)
        self.assertEqual(hist[0], .5)
        self.assertEqual(hist[-1], .5)
        self.assertEqual(row['color_entropy_bits'], 1)
        self.assertAlmostEqual(row['brightness_mean'], 127.5)

    def test_equal_image_weight_not_pixel_weight(self):
        a, ha = measure(Image.new('RGB', (20, 20), 'black'))
        b, hb = measure(Image.new('RGB', (200, 200), 'white'))
        profile = aggregate([a, b], [ha, hb])
        self.assertEqual(profile['mean_image_rgb_mean'], [127.5]*3)
        self.assertEqual(profile['mean_rgb_histogram'][0], .5)
        self.assertEqual(intersection(ha, hb), 0)
        self.assertEqual(intersection(ha, ha), 1)

    def test_canvas_sampling_preserves_joint_modes(self):
        rows, histograms = zip(*(measure(Image.new('RGB', size)) for size in [(100, 40), (300, 80)]))
        profile = aggregate(rows, histograms)
        sampled = sample_canvases(profile, 100, 42)
        self.assertEqual(set(sampled), {(100, 40), (300, 80)})
        self.assertEqual(sampled, sample_canvases(profile, 100, 42))

if __name__ == '__main__':
    unittest.main()

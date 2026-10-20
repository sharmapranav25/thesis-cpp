"""Small artificial fixtures only. These are not target-data regeneration runs."""
import argparse
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from sourcefree_synth.__main__ import audit, parser
from sourcefree_synth.data import new_output, prepare_split, read_json, write_json
from sourcefree_synth.features import DistributionMetric, FeatureExtractor, describe, palette_profile
from sourcefree_synth.fonts import discover_fonts
from sourcefree_synth.pipeline import fit, generate, validate_text_prior
from sourcefree_synth.renderer import QualityError, default_parameters, render_checked
from sourcefree_synth.vlm import aggregate, parse_style, representative_indices


def fixture_images(root, count=24):
    root.mkdir()
    for i in range(count):
        image = Image.new('RGB', (180, 64), (240 + i % 15, 238, 250))
        draw = ImageDraw.Draw(image)
        for j in range(5):
            x = 15 + j * 29
            draw.rectangle((x, 15 + i % 5, x + 12 + i % 4, 48), fill=(10 + i * 3, 20, 30))
        draw.point((i, 0), fill=(i, 100, 200))
        image.save(root / f'opaque_{i:03d}.png')


class DataTests(unittest.TestCase):
    def test_dedup_renaming_and_reencoded_eval_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_images(root / 'input')
            before = prepare_split(root / 'input')
            with Image.open(root / 'input' / 'opaque_000.png') as image:
                image.save(root / 'input' / 'duplicate.bmp')
            after = prepare_split(root / 'input')
            self.assertEqual(after['duplicates_removed'], 1)
            self.assertEqual([r['pixel_sha256'] for r in before['fit']], [r['pixel_sha256'] for r in after['fit']])
            for i, p in enumerate(sorted((root / 'input').glob('*.png'))):
                p.rename(root / 'input' / f'NEW_{100-i}.png')
            renamed = prepare_split(root / 'input')
            self.assertEqual([r['pixel_sha256'] for r in after['fit']], [r['pixel_sha256'] for r in renamed['fit']])
            (root / 'eval').mkdir()
            shutil.copyfile(root / 'input' / 'duplicate.bmp', root / 'eval' / 'different_name.bmp')
            with self.assertRaisesRegex(ValueError, 'Evaluation overlap'):
                prepare_split(root / 'input', [root / 'eval'])

    def test_output_refuses_existing_and_nested_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                new_output(tmp)
            with self.assertRaises(ValueError):
                new_output(Path(tmp) / 'inside', [tmp])


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fonts, _ = discover_fonts('ABCDE01234')
        cls.params = default_parameters()
        cls.params.update(background='flat', lines=0, dots=0, wave_x=0, wave_y=0, rotation=0, jitter=0)
        cls.profile = {'id': 'fixture', 'width': 200, 'height': 64,
                       'background': [[250, 250, 250]], 'foreground': [[20, 20, 20]]}

    def test_determinism_label_alignment_and_per_character_masks(self):
        a = render_checked('AB01E', self.params, self.profile, self.fonts, 24)
        b = render_checked('AB01E', self.params, self.profile, self.fonts, 24)
        self.assertEqual(a[0].tobytes(), b[0].tobytes())
        self.assertEqual(a[3], b[3])
        self.assertEqual(len(a[2]), 5)
        self.assertEqual(a[3]['label'], 'AB01E')
        c = render_checked('EDCBA', self.params, self.profile, self.fonts, 24)
        self.assertNotEqual(a[0].tobytes(), c[0].tobytes())
        self.assertEqual(c[3]['label'], 'EDCBA')

    def test_exhausted_retries_never_return_previous_image_or_change_config(self):
        render_checked('AB01E', self.params, self.profile, self.fonts, 24)
        with patch('sourcefree_synth.renderer.render_once', side_effect=QualityError('forced')) as mock:
            with self.assertRaises(QualityError):
                render_checked('EDCBA', self.params, self.profile, self.fonts, 25, retries=3)
            self.assertEqual(mock.call_count, 3)
            for call in mock.call_args_list:
                self.assertEqual(call.args[0], 'EDCBA')
                self.assertEqual(call.args[1], self.params)

    def test_low_contrast_is_rejected(self):
        profile = dict(self.profile, foreground=[[250, 250, 250]])
        with self.assertRaises(QualityError):
            render_checked('AB01E', self.params, profile, self.fonts, 5, retries=2)

    def test_observing_stages_does_not_change_render_or_metadata(self):
        trace = {}
        observed = render_checked('AB01E', self.params, self.profile, self.fonts, 24, trace=trace)
        plain = render_checked('AB01E', self.params, self.profile, self.fonts, 24)
        self.assertEqual(observed[0].tobytes(), plain[0].tobytes())
        self.assertEqual(observed[3], plain[3])
        self.assertEqual(len(trace), 6)
        self.assertEqual(trace['04_interference'].tobytes(), plain[0].tobytes())


class FeatureAndVLMTests(unittest.TestCase):
    def test_palette_ignores_thin_black_border_and_retains_actual_colors(self):
        image = Image.new('RGB', (200, 60), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 199, 59), outline='black', width=2)
        draw.rectangle((20, 15, 50, 45), fill=(200, 10, 10))
        draw.rectangle((80, 15, 110, 45), fill=(10, 10, 200))
        draw.rectangle((140, 15, 170, 45), fill=(10, 180, 10))
        profile = palette_profile(image, 'fixture')
        self.assertEqual(profile['background'][0], [255, 255, 255])
        self.assertEqual({tuple(c) for c in profile['foreground']},
                         {(200, 10, 10), (10, 10, 200), (10, 180, 10)})

    def test_feature_shape_and_identical_distribution(self):
        images = [Image.new('RGB', (200, 60), color) for color in ('white', 'gray', 'black')]
        extractor = FeatureExtractor()
        x = extractor.batch(images)
        self.assertEqual(x.shape, (3, 110))
        scores = DistributionMetric(x, extractor.groups).compare(x, x)
        self.assertLess(scores['total'], 1e-5)

    def test_style_schema_rejects_transcriptions_and_wrong_types(self):
        valid = {'background': 'grid', 'distortion': 'low', 'outline': True, 'shadow': False, 'multicolor': None}
        parsed = parse_style(json.dumps(valid))
        self.assertNotIn('multicolor', parsed)
        with self.assertRaises(ValueError):
            parse_style(json.dumps(dict(valid, transcript='ABC')))
        with self.assertRaises(ValueError):
            parse_style(json.dumps(dict(valid, outline='yes')))
        hints = aggregate([{'cluster': 0, 'style': parsed}])
        self.assertEqual(hints['0']['background'], 'grid')

    def test_representatives_cover_clusters(self):
        x = np.arange(80).reshape(20, 4).astype(float)
        clusters = np.array([0] * 10 + [1] * 10)
        indices = representative_indices(x, clusters, 4)
        self.assertEqual(len(set(indices)), 4)
        self.assertEqual(set(clusters[indices]), {0, 1})

    def test_invalid_text_priors(self):
        for charset, lengths in [('AA', [5]), ('A{', [5]), ('ABC', [0]), ('ABC', [5, 5])]:
            with self.assertRaises(ValueError):
                validate_text_prior(charset, lengths)


class IntegrationTests(unittest.TestCase):
    def test_fit_generate_audit_without_models_or_real_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_images(root / 'input')
            # A trap CSV must remain unread by ingestion.
            (root / 'input' / 'labels.csv').write_text('THIS IS NOT A VALID LABEL FILE', encoding='utf-8')
            args = parser().parse_args(['fit', '--real-dir', str(root / 'input'), '--out', str(root / 'fit'),
                                       '--clusters', '1', '--trials', '5', '--samples', '4',
                                       '--charset', 'ABCDE01234', '--lengths', '5'])
            with contextlib.redirect_stdout(io.StringIO()):
                fit(args)
            config = read_json(root / 'fit' / 'generator.json')
            report = read_json(root / 'fit' / 'report.json')
            self.assertFalse(report['model_selection_used_validation'])
            self.assertFalse(set(config['fit_image_ids']) & set(config['validation_image_ids']))
            gen_args = parser().parse_args(['generate', '--config', str(root / 'fit' / 'generator.json'),
                                           '--out', str(root / 'generated'), '--count', '6'])
            with contextlib.redirect_stdout(io.StringIO()):
                generate(gen_args)
                audit(argparse.Namespace(run=str(root / 'generated')))
            status = read_json(root / 'generated' / 'status.json')
            self.assertEqual(status['written'], 6)
            self.assertFalse(status['generation_uses_vlm'])
            with (root / 'generated' / 'labels.csv').open(encoding='utf-8') as label_file:
                rows = list(csv.DictReader(label_file))
            self.assertTrue(all(len(r['label']) == 5 for r in rows))
            # Audit must catch pixel tampering, not merely trust manifest counts.
            Image.new('RGB', (200, 60), 'red').save(root / 'generated' / rows[0]['filename'])
            with self.assertRaisesRegex(ValueError, 'Image changed'):
                audit(argparse.Namespace(run=str(root / 'generated')))


if __name__ == '__main__':
    unittest.main()

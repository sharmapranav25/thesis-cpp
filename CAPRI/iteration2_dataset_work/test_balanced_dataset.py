"""Meaningful static-quota and renderer-label regression checks; no GPU or training."""
import unittest
from collections import Counter

import numpy as np
from feature_bank.iter2 import balanced as B
from captcha_effects import text as T


class StaticDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tickets, cls.heads = B.schedule(100000, 20261003)

    def test_full_size_quota_feasibility(self):
        self.assertEqual(len(self.heads), 37)
        self.assertEqual(sum(self.heads.values()), 399999)
        self.assertLessEqual(max(self.heads.values()) - min(self.heads.values()), 1)
        self.assertEqual(sum(t["italic"] for t in self.tickets), 25000)
        for key in ["length", "category", "content", "x_height"]:
            counts = Counter(t[key] for t in self.tickets)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1, key)

    def test_no_impossible_effect_or_layout_tickets(self):
        for ticket in self.tickets:
            if ticket["appearance"] == "two_layer":
                self.assertGreaterEqual(ticket["length"], 4)
            for group in B.R.EXCLUSIVE_K:
                self.assertLessEqual(sum(n in group for n in ticket["effects"]), 1)
            self.assertEqual(len({B.REGISTRY[n]["head"] for n in ticket["effects"]}), ticket["k"])
        # Check every occupied coupled cell, without repeatedly scanning fonts per image.
        cells = {(t["category"], t["x_height"], t["italic"], t["appearance"] == "font_mixture") for t in self.tickets}
        for cat, xh, italic, mixture in cells:
            fonts = [f for f in B.bank().pool("train", category=cat)
                     if T.x_height_level(f) == xh and bool(f.get("italic")) == italic]
            self.assertGreaterEqual(len(fonts), 2 if mixture else 1)

    def test_invisible_text_fails_guard(self):
        rgb = np.full((40, 80, 3), 255, np.uint8)
        mask = np.zeros((40, 80), np.uint32)
        mask[10:30, 10:16] = 1
        out = dict(image=rgb.copy(), opaque=mask, records=[dict(head="text", id=1, params={})],
                   text=dict(legibility=dict(min_visible=1.)))
        self.assertFalse(B.readability(out)[0])
        out["image"][mask == 1] = 0
        self.assertTrue(B.readability(out)[0])

    def test_two_row_transition_is_not_letter_overlap(self):
        glyphs = [dict(mask=np.ones((10, 5)), x0=x, y0=y, size_px=10, curve_off=0,
                       jitter_off=0, angle=0, slant=0, x_scale=1, font="fixture", colour=[0, 0, 0],
                       stroke_rel=.1, clipped_fraction=0)
                  for x, y in [(0, 0), (7, 0), (0, 12), (7, 12)]]
        measured = T.measure_text(glyphs, None, dict(layout="two_layer"))
        self.assertEqual(measured["mean_gap"], .2)
        self.assertEqual(measured["overlap"], 0.)
        line = [dict(g, x0=i * 7, y0=0) for i, g in enumerate(glyphs)]
        self.assertEqual(T.measure_text(line, None, dict(layout="line"))["mean_gap"], .2)


if __name__ == "__main__":
    unittest.main()

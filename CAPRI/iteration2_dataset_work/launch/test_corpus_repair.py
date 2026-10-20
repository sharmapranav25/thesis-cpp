"""Quota preservation and atomic rejection, plus the actual thin-font regression."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from feature_bank.iter2 import balanced as B
from feature_bank.iter2 import corpus as C


class InlinePool:
    def map(self, fn, items):
        return map(fn, items)


class RepairTests(unittest.TestCase):
    def test_rare_slanted_italic_font_profile_can_be_repaired(self):
        failed = dict(length=3, category='Serif', content='mixed', appearance='slant',
                      weight=1, width=0, italic=True, x_height=8, index=82598,
                      effects=['thin','gradient','occluders','pixelate','fill_texture'],
                      k=5, seed=2026100307, font_split='train', split='train')
        donor = dict(length=3, category='Sans Serif', content='lower', appearance='neutral',
                     weight=8, width=6, italic=False, x_height=4, index=80486,
                     effects=['gradient','grain','blur','perspective','occluders'],
                     k=5, seed=2026100307, font_split='train', split='train')
        before = [failed, donor]
        after = C.repair_pair(failed, donor, 7, 0)
        for key in ['length','category','content','appearance','weight','width','italic','x_height','k']:
            self.assertEqual(Counter(t[key] for t in before), Counter(t[key] for t in after))
        self.assertEqual(Counter((t['category'],t['x_height'],t['italic']) for t in before),
                         Counter((t['category'],t['x_height'],t['italic']) for t in after))
        for ticket in after:
            row=B.render_ticket(ticket)
            self.assertFalse(row['failed'],row['rejections'])
            self.assertGreaterEqual(row['label']['quality']['min_glyph_contrast'],70)
            self.assertGreaterEqual(row['label']['quality']['min_final_stroke'],1.2)

    def test_independent_repairs_are_batched_without_shared_donors(self):
        scheduled, _ = B.schedule(140, 2026100302)
        tickets = copy.deepcopy([t for t in scheduled if t['k'] == 3][:9])
        original = copy.deepcopy(tickets)
        rows = [dict(failed=i < 3) for i in range(len(tickets))]
        calls = []
        class RecordingPool:
            def map(self, fn, items):
                items = list(items)
                calls.append(items)
                return map(fn, items)
        with tempfile.TemporaryDirectory() as directory, patch.object(B, 'render_ticket', side_effect=lambda t: dict(failed=False)):
            C.repair_rows(RecordingPool(), Path(directory), tickets, rows, 'trial')
        self.assertFalse(any(r['failed'] for r in rows))
        self.assertEqual(len(calls[0]), 6)
        self.assertEqual(len({t['index'] for t in calls[0]}), 6)
        for key in ['length', 'category', 'content', 'appearance', 'weight', 'width', 'italic', 'x_height', 'k']:
            self.assertEqual(Counter(t[key] for t in original), Counter(t[key] for t in tickets))
        self.assertEqual(Counter(e for t in original for e in t['effects']), Counter(e for t in tickets for e in t['effects']))

    def test_pair_preserves_margins_and_coupled_fonts(self):
        tickets, _ = B.schedule(140, 2026100302)
        left = next(t for t in tickets if t['weight'] < 3)
        right = next(t for t in tickets if t['weight'] >= 3 and t['k'] == left['k'])
        before = [left, right]
        for round_no in [1, 2, 4, 6]:
            after = C.repair_pair(left, right, round_no, 9)
            for key in ['length', 'category', 'content', 'appearance', 'weight', 'width', 'italic', 'x_height', 'k']:
                self.assertEqual(Counter(t[key] for t in before), Counter(t[key] for t in after))
            self.assertEqual(Counter(e for t in before for e in t['effects']), Counter(e for t in after for e in t['effects']))
            for old, new in zip(before, after):
                for key in ['category', 'x_height', 'italic', 'appearance', 'length']:
                    self.assertEqual(old[key], new[key])
        self.assertEqual([left, right], before)

    def test_failed_trial_keeps_accepted_donor(self):
        tickets, _ = B.schedule(140, 2026100302)
        tickets = [tickets[0], next(t for t in tickets[1:] if t['k'] == tickets[0]['k'])]
        original = copy.deepcopy(tickets)
        rows = [dict(failed=True), dict(failed=False, marker='accepted')]
        with tempfile.TemporaryDirectory() as directory, patch.object(B, 'render_ticket', return_value=dict(failed=True)):
            C.repair_rows(InlinePool(), Path(directory), tickets, rows, 'trial')
        self.assertEqual(tickets, original)
        self.assertEqual(rows[1], dict(failed=False, marker='accepted'))

    def test_failed_ticket_722_can_be_repaired_readably(self):
        evidence = Path(r'E:\pranav\Final sem\sourcefree_synth\runs\iter2_baseline_15_20261003_01\steering_failed_tickets.json')
        failed = json.loads(evidence.read_text())[0]['ticket']
        scheduled, _ = B.schedule(2000, 2026100302, split='steering')
        donors = [t for t in scheduled if t['k'] == failed['k'] and t['weight'] == 5
                  and t['appearance'] not in B.GEOMETRIC and t['category'] != 'Monospace'][:12]
        tickets = [copy.deepcopy(failed)]
        rows = [dict(failed=True)]
        for ticket in donors:
            row = B.render_ticket(ticket)
            if not row['failed']:
                tickets.append(copy.deepcopy(ticket))
                rows.append(row)
        original = copy.deepcopy(tickets)
        self.assertGreater(len(tickets), 2)
        # The real repair searches alternatives; an individual trial pair may
        # fail and must be discarded, rather than asserted to be universally valid.
        with tempfile.TemporaryDirectory() as directory:
            C.repair_rows(InlinePool(), Path(directory), tickets, rows, 'ticket_722')
        self.assertFalse(any(row['failed'] for row in rows))
        for key in ['length', 'category', 'content', 'appearance', 'weight', 'width', 'italic', 'x_height', 'k']:
            self.assertEqual(Counter(t[key] for t in original), Counter(t[key] for t in tickets))
        for ticket, row in zip(tickets, rows):
            quality = row['label']['quality']
            self.assertGreaterEqual(quality['min_glyph_contrast'], 70)
            self.assertGreaterEqual(quality['min_final_stroke'], 1.2)
            self.assertEqual(row['label']['ticket']['weight'], ticket['weight'])


if __name__ == '__main__':
    unittest.main()

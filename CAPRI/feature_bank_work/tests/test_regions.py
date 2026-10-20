import unittest
import numpy as np
from PIL import Image
from feature_bank.regions import regions,verify_crops,overlap_edges,check_group_partitions

class RegionTests(unittest.TestCase):
    def test_ten_across_three_down(self):
        image=Image.fromarray(np.arange(200*50*3,dtype=np.uint8).reshape(50,200,3))
        rs=regions(200,50,scales=((3,10),))
        self.assertEqual(len(rs),30)
        self.assertEqual(len({r['column'] for r in rs}),10)
        self.assertEqual(len({r['row'] for r in rs}),3)
        self.assertTrue(verify_crops(image,rs)['full_coverage_per_scale'])
    def test_odd_dimensions_reassemble_exactly(self):
        rng=np.random.default_rng(3)
        for w,h in ((200,50),(319,79),(17,13)):
            image=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
            rs=regions(w,h); self.assertEqual(len(rs),26)
            self.assertTrue(verify_crops(image,rs)['overlap_reassembly_exact'])
    def test_overlap_has_correct_shared_pixels(self):
        edges=overlap_edges(regions(200,60))
        self.assertGreater(len(edges),0)
        for e in edges:
            x,y,xx,yy=e['shared_box'];self.assertEqual(e['shared_pixels'],(xx-x)*(yy-y))
    def test_variants_cannot_cross_splits(self):
        rows=[dict(base_group='a',parent_pixel_sha256='p',partition='train'),dict(base_group='a',parent_pixel_sha256='q',partition='test')]
        with self.assertRaises(ValueError):check_group_partitions(rows)
    def test_duplicate_parents_cannot_cross_splits(self):
        rows=[dict(base_group='a',parent_pixel_sha256='p',partition='train'),dict(base_group='b',parent_pixel_sha256='p',partition='test')]
        with self.assertRaises(ValueError):check_group_partitions(rows)
    def test_invalid_grid_rejected(self):
        with self.assertRaises(ValueError):regions(2,2)
        with self.assertRaises(ValueError):regions(200,60,halo=-1)

if __name__=='__main__':unittest.main()

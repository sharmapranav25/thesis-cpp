import unittest
import numpy as np
from PIL import Image,ImageDraw
from feature_bank.views import image_views,padded_canvas

class FeaturePopTests(unittest.TestCase):
    def test_flat_image_has_no_pop(self):
        self.assertFalse(image_views(Image.new('RGB',(40,20),(60,60,60)))['feature_pop'].any())
    def test_both_polarities_show_small_marks(self):
        im=Image.new('RGB',(40,40),'white');ImageDraw.Draw(im).ellipse((18,18,22,22),fill='black')
        a=image_views(im);b=image_views(Image.fromarray(255-np.asarray(im)))
        np.testing.assert_array_equal(a['feature_pop'],b['feature_pop'])
        self.assertGreater(int(a['feature_pop'][20,20]),0)
    def test_deterministic_and_original_unchanged(self):
        im=Image.new('RGB',(41,19),'gray');before=im.tobytes()
        a=image_views(im);b=image_views(im)
        for name in a:np.testing.assert_array_equal(a[name],b[name])
        self.assertEqual(before,im.tobytes())
    def test_padding_mask_preserves_valid_area(self):
        canvas,valid,meta=padded_canvas(Image.new('RGB',(100,50)))
        self.assertEqual(canvas.size,(256,96));self.assertEqual(valid.sum(),192*96)
        self.assertEqual(meta['offset'],[32,0])
    def test_tiny_image_is_supported(self):
        self.assertEqual(image_views(Image.new('RGB',(1,1)))['feature_pop'].shape,(1,1))

if __name__=='__main__':unittest.main()

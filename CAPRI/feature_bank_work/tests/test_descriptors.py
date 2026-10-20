import unittest
import numpy as np
from PIL import Image,ImageDraw
from feature_bank.descriptors import extract,pooled_hog

class DescriptorTests(unittest.TestCase):
    def test_pixel_only_reproducible_finite(self):
        im=Image.new('RGB',(30,35),'white');ImageDraw.Draw(im).line([(2,8),(27,20)],fill='black')
        a,n=extract(im);b,m=extract(im)
        np.testing.assert_array_equal(a,b);self.assertEqual(n,m)
        self.assertTrue(np.isfinite(a).all());self.assertGreater(len(a),n)
    def test_no_valid_pixels_produce_no_hog(self):
        gray=np.random.default_rng(2).integers(0,256,(96,256),dtype='uint8')
        self.assertEqual(pooled_hog(gray,np.zeros(gray.shape,bool)),[0.0]*27)

if __name__=='__main__':unittest.main()

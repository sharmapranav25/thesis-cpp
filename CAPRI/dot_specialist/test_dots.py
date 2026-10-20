import unittest
import numpy as np,torch
from corpus import fonts,pair,pop
from train_dots import Classifier

class Dots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bank=fonts()[:6];torch.set_num_threads(2)
    def test_pairs_are_deterministic_and_visible(self):
        for seed in range(91000000,91000036):
            a,r=pair(seed,'train',self.bank);b,s=pair(seed,'train',self.bank)
            self.assertTrue(np.array_equal(a,b));self.assertEqual(r,s);self.assertFalse(np.array_equal(a[0],a[1]));self.assertGreaterEqual(r['visible_fraction'],.4);self.assertIn(r['radius'],[1,2,3]);self.assertIn(r['negative_kind'],['dash','no_added_marks'])
    def test_pop_constant_and_shape(self):
        self.assertEqual(pop(np.full((1,64,128,3),127,np.uint8)).max(),0)
        a,_=pair(91000000,'train',self.bank);self.assertEqual(pop(a).shape,(2,64,128));self.assertTrue(np.array_equal(pop(a),pop(a)))
    def test_heldout_paths(self):
        _,r=pair(96000000,'locked_shape',self.bank);self.assertEqual(r['recipe'],'clustered_disks')
        _,r=pair(97000000,'locked_raster',self.bank);self.assertEqual(r['rasterizer'],'opencv')
    def test_independent_trainable_model(self):
        a=Classifier([(2,2)],True);b=Classifier([(2,2)],True)
        self.assertNotEqual(a.encoder[0].weight.data_ptr(),b.encoder[0].weight.data_ptr());z=a(torch.rand(2,4,64,128));z.sum().backward();self.assertGreater(float(a.encoder[0].weight.grad.abs().sum()),0)
if __name__=='__main__':unittest.main()

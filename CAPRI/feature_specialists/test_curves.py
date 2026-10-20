import unittest
import numpy as np,torch
from corpus import pair,pop,fonts
from train_curves import Classifier,stats

class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bank=fonts()[:6];torch.set_num_threads(2)
    def test_repeatable_pair_and_visible_labels(self):
        for seed in range(81000000,81000024):
            a,r=pair(seed,'train',self.bank);b,s=pair(seed,'train',self.bank)
            self.assertTrue(np.array_equal(a,b));self.assertEqual(r,s);self.assertFalse(np.array_equal(a[0],a[1]));self.assertGreaterEqual(r['visible_fraction'],.4);self.assertGreaterEqual(r['curve_bend'],2)
    def test_pop_deterministic_finite_and_constant(self):
        x=np.full((2,64,128,3),127,np.uint8);self.assertEqual(int(pop(x).max()),0)
        a,_=pair(81000100,'train',self.bank);self.assertTrue(np.array_equal(pop(a),pop(a)));self.assertEqual(pop(a).shape,(2,64,128))
    def test_alternate_recipes_and_rasterizer(self):
        a,r=pair(86000000,'locked_shape',self.bank);self.assertEqual(r['recipe'],'cubic')
        a,r=pair(87000000,'locked_raster',self.bank);self.assertEqual(r['rasterizer'],'opencv')
    def test_model_view_shapes_and_spatial_gradient(self):
        for grids in [[(1,1)],[(10,3)],[(1,1),(3,2)]]:
            for use_pop in (False,True):
                model=Classifier(grids,use_pop);x=torch.rand(2,4,64,128);a=model(x);self.assertEqual(tuple(a.shape),(2,));a.sum().backward();self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0)
    def test_metrics_report_errors(self):
        s=stats(np.array([.1,.8,.7,.2]),np.array([0,1,0,1]),.5);self.assertEqual(s['fp'],1);self.assertEqual(s['fn'],1);self.assertEqual(s['f1'],.5)

if __name__=='__main__':unittest.main()

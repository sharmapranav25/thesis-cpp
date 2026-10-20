"""Preflight invariants and actual forward/backward smoke tests."""
import json, unittest
import numpy as np
import torch
from common import groups, group_key, metrics, text_partition
from models import Reader

class Invariants(unittest.TestCase):
    def test_multilabel(self):
        self.assertEqual(groups('aB3'),[1,1,1]);self.assertEqual(groups('aa'),[0,1,0]);self.assertEqual(groups('AA'),[0,0,1])
    def test_metrics(self):
        m=metrics(['aB3','11'],['aB3','1']);self.assertAlmostEqual(m['cer'],.2);self.assertEqual(m['decoded_count_accuracy'],.5)
    def test_text_identity(self):self.assertEqual(text_partition('Ab13'),text_partition('Ab13'))
    def test_architectures(self):
        torch.set_num_threads(4)
        for name in ['crnn','svtr','parseq']:
            torch.manual_seed(731);model=Reader(name,True).to('cuda').train();x=torch.rand(4,3,32,192,device='cuda')*2-1
            with torch.autocast('cuda',dtype=torch.float16):loss,_=model.loss(x,['aa','AA','aB3','11112222'])
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertGreater(float(model.count.weight.grad.abs().sum()),0);self.assertGreater(float(model.group.weight.grad.abs().sum()),0)
            model.eval();p,c,a=model.predict(x);self.assertEqual(len(p),4);self.assertEqual(a['group_logits'].shape,(4,3));print(name,'parameters',sum(p.numel() for p in model.parameters()),'loss',float(loss),flush=True)
            del model;torch.cuda.empty_cache()

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from PIL import Image,ImageDraw
from feature_bank.labeled_bricks import FEATURES,sample_operations,compose,evidence,crop_labels,local_label

class LabelEvidenceTests(unittest.TestCase):
    def test_replay_does_not_sample_again(self):
        base=Image.new('RGB',(240,96),'white');ops=sample_operations(240,96,8)
        a=compose(base,ops,FEATURES);b=compose(base,ops,FEATURES)
        self.assertEqual(a.tobytes(),b.tobytes())
    def test_no_effect_elsewhere_inherits_positive(self):
        op=dict(feature='straight_interference',kind='polyline',points=[(0,2),(30,2)],width=1,color=[0,0,0])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['straight_interference'])
        state,_=local_label(op,ev['straight_interference'],[0,20,40,40])
        self.assertEqual(state,'absent')
    def test_hidden_effect_is_uncertain(self):
        op=dict(feature='straight_interference',kind='polyline',points=[(1,20),(38,20)],width=1,color=[255,255,255])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['straight_interference'])
        state,_=local_label(op,ev['straight_interference'],[0,0,40,40])
        self.assertEqual(state,'uncertain')
    def test_occlusion_erases_only_covered_contribution(self):
        first=dict(feature='straight_interference',kind='polyline',points=[(1,20),(38,20)],width=1,color=[0,0,0])
        last=dict(first,feature='curved_interference',color=[200,0,0])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[first,last],FEATURES)
        self.assertEqual(int(ev['straight_interference']['visible'].sum()),0)
        self.assertGreater(int(ev['curved_interference']['visible'].sum()),0)
    def test_clean_glyph_like_marks_are_not_overlay_labels(self):
        base=Image.new('RGB',(240,96),'white');d=ImageDraw.Draw(base)
        d.text((25,25),'Il1 ij OO0',fill='black');d.ellipse((70,20,80,30),outline='black')
        ops=sample_operations(240,96,1);_,ev=evidence(base,ops,[])
        labels,_=crop_labels(ops,ev,[0,0,240,96]);self.assertEqual(set(labels.values()),{'absent'})
    def test_partial_single_dot_abstains(self):
        op=dict(feature='dot_scatter',kind='dots',dots=[[10,10,2]],color=[0,0,0])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['dot_scatter'],background=Image.new('RGB',(40,40),'white'))
        state,_=local_label(op,ev['dot_scatter'],[0,0,40,40]);self.assertEqual(state,'uncertain')
    def test_duplicate_dot_draws_do_not_count_twice(self):
        op=dict(feature='dot_scatter',kind='dots',dots=[[10,10,2],[10,10,2]],color=[0,0,0])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['dot_scatter'],background=Image.new('RGB',(40,40),'white'))
        state,info=local_label(op,ev['dot_scatter'],[0,0,40,40])
        self.assertEqual(state,'uncertain');self.assertEqual(info['complete_visible_dots'],1)

    def test_dots_touching_glyph_strokes_abstain(self):
        base=Image.new('RGB',(40,40),'white');ImageDraw.Draw(base).line([(10,0),(10,39)],fill='black',width=2)
        op=dict(feature='dot_scatter',kind='dots',dots=[[13,10,2],[13,28,2]],color=[80,0,0])
        _,ev=evidence(base,[op],['dot_scatter'],background=Image.new('RGB',base.size,'white'))
        self.assertEqual(local_label(op,ev['dot_scatter'],[0,0,40,40])[0],'uncertain')

    def test_isolated_dots_positive(self):
        base=Image.new('RGB',(40,40),'white')
        op=dict(feature='dot_scatter',kind='dots',dots=[[10,10,2],[28,28,2]],color=[80,0,0])
        _,ev=evidence(base,[op],['dot_scatter'],background=base)
        self.assertEqual(local_label(op,ev['dot_scatter'],[0,0,40,40])[0],'present')

    def test_no_background_context_abstains(self):
        op=dict(feature='dot_scatter',kind='dots',dots=[[10,10,2],[28,28,2]],color=[80,0,0])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['dot_scatter'])
        self.assertEqual(local_label(op,ev['dot_scatter'],[0,0,40,40])[0],'uncertain')

    def test_fragmented_line_cannot_bridge_gaps(self):
        base=Image.new('RGB',(40,40),'white')
        op=dict(feature='straight_interference',kind='polyline',points=[(1,20),(38,20)],width=1,color=[0,0,0])
        cover=dict(feature='dot_scatter',kind='dots',dots=[[x,20,1] for x in (8,18,28)],color=[255,255,255])
        _,ev=evidence(base,[op,cover],['straight_interference','dot_scatter'])
        self.assertEqual(local_label(op,ev['straight_interference'],[0,0,40,40])[0],'uncertain')

    def test_faint_contribution_abstains(self):
        op=dict(feature='straight_interference',kind='polyline',points=[(1,20),(38,20)],width=1,color=[240,240,240])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['straight_interference'])
        self.assertEqual(local_label(op,ev['straight_interference'],[0,0,40,40])[0],'uncertain')

    def test_hidden_curve_geometry_cannot_supply_curvature(self):
        op=dict(feature='curved_interference',kind='polyline',
                points=[(x,round(10+((x-20)/4)**2)) for x in range(5,36)],width=1,color=[0,0,0])
        _,ev=evidence(Image.new('RGB',(40,40),'white'),[op],['curved_interference'])
        # A controlled missing-contribution fixture: recorded geometry remains
        # curved, but only disconnected portions survive in final evidence.
        ev['curved_interference']['visible'][:,12:15]=False
        ev['curved_interference']['visible'][:,25:28]=False
        self.assertEqual(local_label(op,ev['curved_interference'],[0,0,40,40])[0],'uncertain')

if __name__=='__main__':unittest.main()

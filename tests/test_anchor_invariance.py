"""只测候选统计的数学性质；服务器执行，无模型、优化器或真实数据。"""
import unittest

import torch

from anchor_invariance_candidates import extract_scale_stable, affine_perturbation, update_invariance
from physiological_anchors import extract_anchors


class AnchorInvarianceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_dimensions_and_names(self):
        for dataset, channels, dim in [('DEAP',3,4),('HCI',4,5)]:
            o = extract_scale_stable(torch.randn(2,32,768), torch.randn(2,channels,768), dataset)
            self.assertEqual(o['c'].shape,(2,160))
            self.assertEqual(o['a'].shape,(2,dim))
            self.assertEqual(len(o['names_c']),160)
            self.assertEqual(len(o['names_a']),dim)

    def test_three_affine_transforms_preserve_candidates(self):
        c, a = torch.randn(3,2,768), torch.randn(3,3,768)
        before = extract_scale_stable(c,a,'DEAP')
        for kind in ('gain','offset','channel_affine'):
            after = extract_scale_stable(affine_perturbation(c,kind), affine_perturbation(a,kind),'DEAP')
            for side in ('c','a'):
                self.assertTrue(torch.equal(before['mask_'+side],after['mask_'+side]))
                self.assertTrue(torch.allclose(before[side],after[side],atol=1e-5,rtol=1e-5))

    def test_constants_are_finite_but_invalid(self):
        o = extract_scale_stable(torch.ones(2,1,768),torch.ones(2,3,768),'DEAP')
        for side in ('c','a'):
            self.assertTrue(torch.isfinite(o[side]).all())
            self.assertFalse(o['mask_'+side].any())

    def test_normalized_linear_trend_known_value(self):
        t = torch.arange(768,dtype=torch.float64)/128
        a = torch.stack((t,2*t,3+.2*t))[None]
        c = torch.cos(2*torch.pi*10*t)[None,None]
        o = extract_scale_stable(c,a,'DEAP')
        expected = 6./float(t.std(unbiased=False))
        self.assertAlmostEqual(float(o['a'][0,-2]),expected,places=5)
        self.assertAlmostEqual(float(o['a'][0,-1]),float((t[-1]-t[0])/t.std(unbiased=False)),places=5)

    def test_original_amplitude_positive_control_changes(self):
        c,a = torch.randn(2,1,768),torch.randn(2,3,768)
        before = extract_anchors(c,a,'DEAP'); after = extract_anchors(2*c,2*a,'DEAP')
        self.assertTrue(torch.allclose(after['c'][:,0]-before['c'][:,0],torch.full((2,),2*torch.log(torch.tensor(2.)).item()),atol=1e-5))
        self.assertTrue(torch.allclose(after['a'][:,0]-before['a'][:,0],torch.full((2,),torch.log(torch.tensor(2.)).item()),atol=1e-5))

    def test_no_input_mutation(self):
        c,a = torch.randn(2,1,768),torch.randn(2,3,768)
        before_c,before_a = c.clone(),a.clone()
        extract_scale_stable(c,a,'DEAP'); affine_perturbation(a,'channel_affine')
        self.assertTrue(torch.equal(c,before_c) and torch.equal(a,before_a))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            extract_scale_stable(torch.zeros(2,1,20),torch.zeros(2,3,768),'DEAP')
        with self.assertRaises(ValueError): affine_perturbation(torch.zeros(2,3,768),'unknown')

    def test_accumulator_detects_violation_and_mask_change(self):
        x = dict(c=torch.ones(2,1),a=torch.ones(2,1),mask_c=torch.ones(2,1,dtype=torch.bool),mask_a=torch.ones(2,1,dtype=torch.bool))
        y = {k:v.clone() for k,v in x.items()}; y['c'][0,0] += .1; y['mask_a'][1,0] = False
        record = {}; update_invariance(record,x,y)
        self.assertEqual(record['c']['violations'],1)
        self.assertEqual(record['a']['mask_changes'],1)


if __name__ == '__main__':
    unittest.main()

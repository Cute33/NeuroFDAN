"""新候选接入配对合成的服务器CPU测试，不读取真实数据。"""
import unittest
import numpy as np
import torch

from anchor_synthetic_scm import AnchorSyntheticSCM, coefficient_anchors
from physiology_anchor_experiment import anchor_view, validate_view, anchor_learning_terms
from physiological_anchors import PhysiologicalAnchorReadouts
from frontdoor_scm import FrontDoorEstimator, FrontDoorConfig
from phase6_learning import ModalityReadouts


class ScaleAnchorLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(1)

    def test_default_preserves_original_mapping(self):
        raw = AnchorSyntheticSCM().sample(8,1.,42)
        a = coefficient_anchors(raw['x_c'],raw['x_a'])
        b = coefficient_anchors(raw['x_c'],raw['x_a'],'original')
        self.assertTrue(torch.equal(a['c'],b['c']) and torch.equal(a['a'],b['a']))

    def test_scale_targets_and_original_control_share_xy(self):
        raw = AnchorSyntheticSCM(anchor_variant='scale_stable').sample(8,1.,42)
        a,b = anchor_view(raw,'scale_stable'),anchor_view(raw,'original')
        self.assertEqual(a['c'].shape,(8,5)); self.assertEqual(a['a'].shape,(8,4))
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in ('x_c','x_a','y')))
        validate_view(a,'observation_physiological_anchors');validate_view(b,'observation_original_anchors')
        with self.assertRaises(ValueError):validate_view(a|{'u':torch.zeros(8,8)},'observation_original_anchors')

    def test_aligned_positive_gain_invariance_and_failure_case(self):
        x = np.random.default_rng(42).normal(size=(4,16)); a = np.random.default_rng(43).normal(size=(4,8))
        scm = AnchorSyntheticSCM('aligned',anchor_variant='scale_stable')
        before=scm.mediator_inputs(x,a); after=scm.mediator_inputs(2*x,2*a)
        for b,v in zip(before,after):np.testing.assert_allclose(b,v,atol=1e-6)
        wrong=AnchorSyntheticSCM('misaligned',anchor_variant='scale_stable')
        self.assertFalse(np.allclose(wrong.mediator_inputs(x,a)[0],wrong.mediator_inputs(2*x,2*a)[0]))

    def test_both_training_targets_have_finite_backward_and_step(self):
        raw=AnchorSyntheticSCM(anchor_variant='scale_stable').sample(8,1.,42)
        for variant,condition in [('scale_stable','observation_physiological_anchors'),('original','observation_original_anchors')]:
            data=anchor_view(raw,variant)
            model=FrontDoorEstimator(16,8,FrontDoorConfig(S_train=2))
            read=PhysiologicalAnchorReadouts(64,data);legacy=ModalityReadouts(64,data['x_c'],data['x_a'])
            parameters=list(model.parameters())+list(read.parameters())
            opt=torch.optim.Adam(parameters,lr=.001)
            before=read.cns.weight.detach().clone()
            terms,_=anchor_learning_terms(model,read,legacy,data,condition,1,.1,.1)
            terms['total'].backward();opt.step()
            self.assertFalse(torch.equal(before,read.cns.weight))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))


if __name__=='__main__':unittest.main()

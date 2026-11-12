"""服务器执行的CPU小测试；不使用DEAP/HCI真实数据，不作为方法效果。"""
import unittest

import numpy as np
import torch

from anchor_synthetic_scm import AnchorSyntheticSCM, coefficient_anchors, render_coefficients
from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator
from physiological_anchors import extract_anchors, PhysiologicalAnchorReadouts, training_statistics
from physiology_anchor_experiment import anchor_view, validate_view, anchor_learning_terms
from phase6_learning import ModalityReadouts, LearningWeights, learning_terms
from scripts.audit_physiological_anchors import matrix_windows


class AnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_pure_alpha_spectrum_and_unit_power(self):
        t = torch.arange(768)/128
        c = torch.sin(2*torch.pi*10*t)[None,None]
        a = torch.zeros(1,3,768)
        target = extract_anchors(c,a,'DEAP')
        self.assertAlmostEqual(float(target['c'][0,0]), np.log(.5), places=5)
        self.assertGreater(float(target['c'][0,3]), .9999)
        self.assertTrue(target['mask_c'].all())

    def test_real_interface_dimensions_and_constant_validity(self):
        for dataset, a, d in [('DEAP',3,8),('HCI',4,10)]:
            out = extract_anchors(torch.zeros(2,32,768), torch.zeros(2,a,768), dataset)
            self.assertEqual(out['c'].shape, (2,192))
            self.assertEqual(out['a'].shape, (2,d))
            self.assertFalse(out['mask_c'].any())
            self.assertTrue(torch.isfinite(out['a']).all())
            stats = training_statistics(out['a'],out['mask_a'])
            self.assertFalse(stats['active'].any())

    def test_nan_and_wrong_window_rejected(self):
        c = torch.zeros(2,1,768); a = torch.zeros(2,3,768)
        c[0,0,0] = float('nan')
        with self.assertRaises(ValueError): extract_anchors(c,a,'DEAP')
        with self.assertRaises(ValueError): extract_anchors(c[..., :100],a,'DEAP')

    def test_gsr_slope_and_window_source_order(self):
        t = torch.arange(768)/128
        a = torch.zeros(2,3,768); a[:,-1] = 3+.2*t
        o = extract_anchors(torch.ones(2,1,768),a,'DEAP')
        self.assertAlmostEqual(float(o['a'][0,-3]), .2, places=5)
        raw = np.arange(2*3*2*768,dtype=np.float32).reshape(2,3,1536)
        win = matrix_windows(raw,2)
        np.testing.assert_array_equal(win[2].numpy(),raw[1,:,:768])
        np.testing.assert_array_equal(win[1].numpy(),raw[0,:,768:])

    def test_training_statistics_never_refit_on_test(self):
        targets = dict(c=torch.randn(8,6),a=torch.randn(8,8),
                       mask_c=torch.ones(8,6,dtype=torch.bool),mask_a=torch.ones(8,8,dtype=torch.bool))
        read = PhysiologicalAnchorReadouts(64,targets)
        before = {k:v.clone() for k,v in read.state_dict().items()}
        read.standardized('c',targets['c']+100,targets['mask_c'])
        self.assertTrue(all(torch.equal(v,read.state_dict()[k]) for k,v in before.items()))

    def test_sampling_axis_is_averaged_and_gradients_exist(self):
        targets = dict(c=torch.randn(4,6),a=torch.randn(4,8),
                       mask_c=torch.ones(4,6,dtype=torch.bool),mask_a=torch.ones(4,8,dtype=torch.bool))
        read = PhysiologicalAnchorReadouts(64,targets)
        mc,ma = torch.randn(4,1,64,requires_grad=True),torch.randn(4,1,64,requires_grad=True)
        one = read.losses(dict(m_c=mc,m_a=ma),targets)
        repeated = read.losses(dict(m_c=mc.expand(-1,5,-1),m_a=ma.expand(-1,5,-1)),targets)
        self.assertTrue(all(torch.allclose(x,y) for x,y in zip(one,repeated)))
        sum(one).backward()
        self.assertGreater(float(mc.grad.norm()),0)
        self.assertGreater(float(ma.grad.norm()),0)

    def test_permission_and_anchor_computation_ignore_truth(self):
        scm = AnchorSyntheticSCM(); raw = scm.sample(8,1.,42)
        a = anchor_view(raw)
        raw.update(m_c=raw['m_c']+100,u=raw['u']+100,y=1-raw['y'])
        b = anchor_view(raw)
        self.assertTrue(torch.equal(a['c'],b['c']) and torch.equal(a['a'],b['a']))
        validate_view(b,'observation_physiological_anchors')
        with self.assertRaises(ValueError): validate_view(b|{'u':torch.zeros(8,8)},'observation_physiological_anchors')

    def test_zero_alpha_recovers_existing_total(self):
        raw = AnchorSyntheticSCM().sample(4,1.,42); data = anchor_view(raw)
        model = FrontDoorEstimator(16,8,FrontDoorConfig(dropout=0.,S_train=2))
        read = PhysiologicalAnchorReadouts(64,data); legacy = ModalityReadouts(64,data['x_c'],data['x_a'])
        base = {k:data[k] for k in ('x_c','x_a','y')}
        torch.manual_seed(123)
        t,_ = learning_terms(model,legacy,base,'observation_only',1,LearningWeights(fidelity=0.,fire=0.))
        torch.manual_seed(123)
        tt,_ = anchor_learning_terms(model,read,legacy,data,'observation_physiological_anchors',1,0.,0.)
        self.assertTrue(torch.equal(t['total'],tt['total']))

    def test_ans_anchor_gradient_reaches_cns_only_through_allowed_path(self):
        data = anchor_view(AnchorSyntheticSCM().sample(4,1.,42))
        for off in (False,True):
            model = FrontDoorEstimator(16,8,FrontDoorConfig(no_cns2ans=off,S_train=2))
            read = PhysiologicalAnchorReadouts(64,data)
            out = model(data['x_c'],data['x_a'],fd_mode='none')
            loss = read.losses(out,data)[1]
            g = torch.autograd.grad(loss,out['m_c'],allow_unused=True)[0]
            self.assertIsNone(g) if off else self.assertGreater(float(g.norm()),0)

    def test_aligned_inputs_invariant_to_global_cns_sign(self):
        x = np.random.default_rng(42).normal(size=(3,16)); a = np.random.default_rng(43).normal(size=(3,8))
        aligned, wrong = AnchorSyntheticSCM('aligned'),AnchorSyntheticSCM('misaligned')
        z,_ = aligned.mediator_inputs(x,a); zz,_ = aligned.mediator_inputs(-x,a)
        np.testing.assert_allclose(z,zz,atol=1e-7)
        self.assertFalse(np.allclose(wrong.mediator_inputs(x,a)[0],wrong.mediator_inputs(-x,a)[0]))

    def test_rendered_signal_map_has_full_coefficient_rank(self):
        c,a = render_coefficients(np.eye(16),np.pad(np.eye(8),((0,8),(0,0))))
        self.assertEqual(int(torch.linalg.matrix_rank(c.flatten(1))),16)
        self.assertEqual(int(torch.linalg.matrix_rank(a[:8].flatten(1))),8)

    def test_direct_and_independent_frontdoor_truth(self):
        for mechanism in ('aligned','misaligned'):
            scm = AnchorSyntheticSCM(mechanism); p = scm.sample(2,0.,45)
            for gamma in (0.,1.):
                direct,se = scm.direct_intervention(p['x_c'],p['x_a'],gamma,8192,101)
                fd,fe = scm.frontdoor_integral(p['x_c'],p['x_a'],gamma,512,256,102)
                self.assertTrue(np.all(abs(direct-fd) < 4*np.sqrt(se**2+fe**2)+.02))

    def test_candidate_cpu_optimizer_step_and_probability_normalization(self):
        data = anchor_view(AnchorSyntheticSCM().sample(4,1.,42))
        model = FrontDoorEstimator(16,8,FrontDoorConfig(S_train=2))
        read = PhysiologicalAnchorReadouts(64,data); legacy = ModalityReadouts(64,data['x_c'],data['x_a'])
        parameters = list(model.parameters())+list(read.parameters())
        optimizer = torch.optim.Adam(parameters,lr=.001)
        before = read.cns.weight.detach().clone()
        terms,out = anchor_learning_terms(model,read,legacy,data,'observation_physiological_anchors',1,.1,.1)
        terms['total'].backward(); optimizer.step()
        self.assertFalse(torch.equal(before,read.cns.weight))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
        self.assertTrue(torch.allclose(out['probs_obs'].sum(-1),torch.ones(4)))


if __name__ == '__main__':
    unittest.main()

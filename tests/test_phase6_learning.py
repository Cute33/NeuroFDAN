"""Phase6监督隔离与数学接口测试；仅供服务器执行，不代表真实数据实验。"""
import unittest

import numpy as np
import torch

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator
from phase6_learning import LearningWeights, ModalityReadouts, check_training_view, gradient_components, learning_terms
from phase6_nonidentifiability import counterexample
from phase6_synthetic_experiment import training_view


class Phase6LearningTests(unittest.TestCase):
    def test_observation_view_never_reads_hidden_mediators_or_u(self):
        class Protected(dict):
            def __getitem__(self,key):
                if key in ('u','m_c','m_a','oracle_probability'):
                    raise AssertionError('无监督组读取了禁用字段')
                return super().__getitem__(key)
        raw=Protected(x_c=np.zeros((4,2)),x_a=np.ones((4,1)),y=np.array([0,1,0,1]))
        for condition in ('observation_only','observation_input_fidelity'):
            self.assertEqual(set(training_view(raw,condition)),{'x_c','x_a','y'})

    def test_permission_guard_rejects_hidden_fields(self):
        data={'x_c':torch.zeros(4,2),'x_a':torch.zeros(4,1),'y':torch.zeros(4,dtype=torch.long),'m_a':torch.zeros(4,3)}
        with self.assertRaises(ValueError):check_training_view(data,'observation_only')

    def test_reconstruction_uses_samples_and_conditional_ans_path(self):
        torch.manual_seed(42)
        model=FrontDoorEstimator(2,1,FrontDoorConfig(mediator_dim=4,hidden_dim=8,time_steps=2))
        batch={'x_c':torch.randn(4,2),'x_a':torch.randn(4,1),'y':torch.tensor([0,1,0,1])}
        readouts=ModalityReadouts(4,batch['x_c'],batch['x_a'])
        terms,out=learning_terms(model,readouts,batch,'observation_input_fidelity',1,LearningWeights())
        self.assertEqual(out['mu_a'].shape,(4,4,4))
        gradient=torch.autograd.grad(terms['fidelity_a'],model.ans.cns_current.weight,retain_graph=True)[0]
        self.assertGreater(float(gradient.norm()),0.)
        terms['total'].backward()
        self.assertTrue(torch.isfinite(readouts.cns.weight.grad).all())

    def test_gradient_components_separate_branches_and_conflicting_terms(self):
        a=torch.nn.Parameter(torch.tensor([1.,2.]))
        b=torch.nn.Parameter(torch.tensor([3.]))
        c=torch.nn.Parameter(torch.tensor([4.]))
        terms={'factual_ce':a.sum()+b.sum()+c.sum(),'kl_c':-a.sum()}
        result=gradient_components(terms,[('frontdoor.cns.encoder.weight',a),('frontdoor.ans.cns_current.weight',b),('frontdoor.cns.encoder.bias',c)])
        self.assertIn('cns',result['factual_ce'])
        self.assertIn('cns2ans',result['factual_ce'])
        self.assertAlmostEqual(result['kl_c']['cns']['cosine_to_ce'],-float(np.sqrt(2/3)))
        self.assertEqual(result['kl_c']['cns']['missing_tensors'],1)
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def test_zero_fidelity_weight_recovers_baseline_total_for_paired_noise(self):
        torch.manual_seed(42)
        model=FrontDoorEstimator(2,1,FrontDoorConfig(mediator_dim=4,hidden_dim=8,time_steps=2))
        batch={'x_c':torch.randn(4,2),'x_a':torch.randn(4,1),'y':torch.tensor([0,1,0,1])}
        readouts=ModalityReadouts(4,batch['x_c'],batch['x_a'])
        torch.manual_seed(11)
        baseline,_=learning_terms(model,readouts,batch,'observation_only',1,LearningWeights())
        torch.manual_seed(11)
        candidate,_=learning_terms(model,readouts,batch,'observation_input_fidelity',1,LearningWeights(fidelity=0.))
        self.assertTrue(torch.equal(baseline['total'],candidate['total']))

    def test_nonidentification_has_equal_observations_and_distinct_effects(self):
        r=counterexample()
        self.assertEqual(r['models']['A']['observed_covariance'],r['models']['B']['observed_covariance'])
        self.assertGreater(r['unavoidable_max_absolute_error_lower_bound'],.05)


if __name__=='__main__':
    unittest.main()
